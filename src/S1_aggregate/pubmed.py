"""PubMed fetcher via NCBI E-utilities (E-Search + E-Fetch)."""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import httpx

from ..models import NewsItem

RAW_DIR = Path("data/raw")


@dataclass(frozen=True)
class _EutilsConfig:
    email: str | None
    api_key: str | None
    tool: str


def default_eutils_config() -> _EutilsConfig:
    return _EutilsConfig(
        email=_get_env("NCBI_EMAIL"),
        api_key=_get_env("NCBI_API_KEY"),
        tool="scholarpipe",
    )


def fetch(source: dict, save_raw: bool = True) -> list[NewsItem]:
    """
    Fetch and parse PubMed results.

    Source example:
      {
        "type": "pubmed",
        "name": "PubMed Top Journals Speed",
        "term": "(...query...)",
        "retmax": 50,
        "email": "...",        # optional (or env NCBI_EMAIL)
        "api_key": "...",      # optional (or env NCBI_API_KEY)
      }
    """
    name = source["name"]
    term = source["term"]
    retmax = int(source.get("retmax", 50))

    eutils = _EutilsConfig(
        email=source.get("email") or _get_env("NCBI_EMAIL"),
        api_key=source.get("api_key") or _get_env("NCBI_API_KEY"),
        tool=source.get("tool") or "scholarpipe",
    )

    try:
        pmids = _esearch(term=term, retmax=retmax, eutils=eutils)
        if not pmids:
            return []
        items = _efetch(pmids, source_name=name, term=term, eutils=eutils)
    except Exception as e:
        print(f"  [{name}] Fetch error: {e}")
        return []

    if save_raw and items:
        _save_raw(name, items)

    return items


def _esearch(*, term: str, retmax: int, eutils: _EutilsConfig) -> list[str]:
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": str(retmax),
        "sort": "date",
        "retmode": "json",
    }
    params |= _eutils_params(eutils)

    data = _get_json("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params)
    return list(data.get("esearchresult", {}).get("idlist", []))


def enrich_items_by_doi(
    items: list[NewsItem],
    *,
    min_content_len: int = 80,
    eutils: _EutilsConfig | None = None,
) -> list[NewsItem]:
    """
    Enrich RSS-like items by looking up PubMed records via DOI and filling missing abstracts.

    Notes:
      - Keeps original item link/source fields (does not replace with PubMed link).
      - Only overwrites content when it's empty/too short (< min_content_len).
      - Keeps existing published_at unless missing.
    """
    if not items:
        return items

    eutils = eutils or default_eutils_config()

    need: dict[str, list[int]] = {}
    for idx, item in enumerate(items):
        doi = _normalize_doi(item.doi)
        if not doi:
            continue
        content = (item.content or "").strip()
        if len(content) >= min_content_len:
            continue
        need.setdefault(doi, []).append(idx)

    if not need:
        return items

    sleep_s = 0.12 if eutils.api_key else 0.34

    # Batch DOI -> PMID lookup:
    # Some feeds (e.g. Science AOP) include only "Ahead of Print" in RSS summary,
    # so enriching every DOI individually can be very slow. We instead search for
    # multiple DOIs per ESearch request, then map results back by DOI after EFetch.
    pmids: list[str] = []
    for batch in _chunk_dois_for_esearch(list(need.keys())):
        term = _build_doi_or_term(batch)
        pmids.extend(_esearch(term=term, retmax=max(1, len(batch) * 2), eutils=eutils))
        time.sleep(sleep_s)

    pmids = list(dict.fromkeys(pmids))
    if not pmids:
        return items

    pubmed_items = _efetch(pmids, source_name="PubMed DOI Enrich", term="doi-enrich", eutils=eutils)
    pubmed_by_doi = {_normalize_doi(it.doi): it for it in pubmed_items if _normalize_doi(it.doi)}

    updated = list(items)
    for doi, indices in need.items():
        pm = pubmed_by_doi.get(doi)
        if not pm:
            continue
        abstract = (pm.content or "").strip()
        if not abstract:
            continue

        for idx in indices:
            orig = updated[idx]
            new_fields: dict = {"content": abstract}
            if not orig.authors and pm.authors:
                new_fields["authors"] = pm.authors
            if orig.published_at is None and pm.published_at is not None:
                new_fields["published_at"] = pm.published_at
            updated[idx] = orig.model_copy(update=new_fields)

    return updated


def _build_doi_or_term(dois: list[str]) -> str:
    """Build an ESearch term that ORs multiple DOI queries."""
    parts = [f"\"{doi}\"[DOI]" for doi in dois if doi]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return "(" + " OR ".join(parts) + ")"


def _chunk_dois_for_esearch(dois: list[str], *, max_terms: int = 20, max_len: int = 3500) -> list[list[str]]:
    """
    Chunk DOIs so the ESearch query stays within a safe length.

    Notes:
      - max_len is conservative to avoid very long URLs/query strings.
      - max_terms caps per-request workload to keep latency predictable.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    for doi in [d for d in dois if d]:
        token = f"\"{doi}\"[DOI]"
        extra = len(token) + (4 if current else 0)  # " OR "
        if current and (len(current) >= max_terms or current_len + extra > max_len):
            batches.append(current)
            current = [doi]
            current_len = len(token)
            continue

        current.append(doi)
        current_len += extra

    if current:
        batches.append(current)
    return batches


def _efetch(pmids: Iterable[str], *, source_name: str, term: str, eutils: _EutilsConfig) -> list[NewsItem]:
    pmid_list = list(pmids)
    if not pmid_list:
        return []

    params = {"db": "pubmed", "id": ",".join(pmid_list), "retmode": "xml"}
    params |= _eutils_params(eutils)

    xml_text = _get_text("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params)
    root = ET.fromstring(xml_text)

    items: list[NewsItem] = []
    search_url = _pubmed_search_url(term)

    for article in root.findall("./PubmedArticle"):
        citation = article.find("./MedlineCitation")
        pmid = (citation.findtext("./PMID") if citation is not None else "") or ""
        if not pmid:
            continue

        art = citation.find("./Article") if citation is not None else None
        title = _collect_text(art.find("./ArticleTitle")) if art is not None else ""
        abstract = _extract_abstract(art)
        authors = _extract_authors(art)
        doi = _extract_doi(article)
        published_at = _extract_published_at(article)
        journal_name = _extract_journal_name(art)

        items.append(
            NewsItem(
                title=title or f"PubMed {pmid}",
                content=abstract,
                link=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                authors=authors,
                doi=doi,
                source_name=source_name,
                source_url=search_url,
                published_at=published_at,
                journal_name=journal_name,
            )
        )

    return items


def _extract_abstract(article_elem: ET.Element | None) -> str:
    if article_elem is None:
        return ""
    abstract = article_elem.find("./Abstract")
    if abstract is None:
        return ""

    parts: list[str] = []
    for node in abstract.findall("./AbstractText"):
        text = _collect_text(node).strip()
        if not text:
            continue
        label = (node.attrib.get("Label") or "").strip()
        parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts).strip()


def _extract_authors(article_elem: ET.Element | None) -> list[str]:
    if article_elem is None:
        return []
    authors: list[str] = []
    for author in article_elem.findall("./AuthorList/Author"):
        collective = author.findtext("./CollectiveName")
        if collective and collective.strip():
            authors.append(collective.strip())
            continue

        last = (author.findtext("./LastName") or "").strip()
        fore = (author.findtext("./ForeName") or "").strip()
        initials = (author.findtext("./Initials") or "").strip()

        if fore and last:
            authors.append(f"{fore} {last}".strip())
        elif initials and last:
            authors.append(f"{initials} {last}".strip())
        elif last:
            authors.append(last)
    return authors


def _extract_journal_name(article_elem: ET.Element | None) -> str:
    """Extract journal name from Article element."""
    if article_elem is None:
        return ""
    # Prefer ISO abbreviation (e.g., "Nature", "Cell") over full title
    iso_abbrev = article_elem.findtext("./Journal/ISOAbbreviation")
    if iso_abbrev and iso_abbrev.strip():
        return iso_abbrev.strip()
    # Fallback to full title
    full_title = article_elem.findtext("./Journal/Title")
    if full_title and full_title.strip():
        return full_title.strip()
    return ""


def _extract_doi(article: ET.Element) -> str:
    for article_id in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
        if (article_id.attrib.get("IdType") or "").lower() == "doi":
            doi = _collect_text(article_id).strip()
            if doi:
                return doi

    for eloc in article.findall(".//MedlineCitation/Article/ELocationID"):
        if (eloc.attrib.get("EIdType") or "").lower() == "doi":
            doi = _collect_text(eloc).strip()
            if doi:
                return doi

    return ""


def _extract_published_at(article: ET.Element) -> datetime | None:
    # Prefer online-first / record-first dates over print issue year-only PubDate.
    dt = _extract_history_date(article, ["epublish"])
    if dt:
        return dt

    # Most reliable: ArticleDate (YYYY/MM/DD)
    year = article.findtext(".//MedlineCitation/Article/ArticleDate/Year")
    month = article.findtext(".//MedlineCitation/Article/ArticleDate/Month")
    day = article.findtext(".//MedlineCitation/Article/ArticleDate/Day")
    dt = _parse_ymd(year, month, day, require_month=True)
    if dt:
        return dt

    dt = _extract_history_date(article, ["ppublish"])
    if dt:
        return dt

    dt = _extract_history_date(article, ["medline", "pubmed", "entrez"])
    if dt:
        return dt

    # Journal issue PubDate (can be incomplete, sometimes year-only/future)
    year = article.findtext(".//MedlineCitation/Article/Journal/JournalIssue/PubDate/Year")
    month = article.findtext(".//MedlineCitation/Article/Journal/JournalIssue/PubDate/Month")
    day = article.findtext(".//MedlineCitation/Article/Journal/JournalIssue/PubDate/Day")
    dt = _parse_ymd(year, month, day, require_month=True)
    if dt:
        return dt

    medline_date = article.findtext(".//MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate") or ""
    dt = _parse_medline_date(medline_date)
    if dt:
        return dt

    dt = _extract_history_date(article, ["pmc-release"])
    if dt:
        return dt

    return None


_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _extract_history_date(article: ET.Element, statuses: list[str]) -> datetime | None:
    for status in statuses:
        year = article.findtext(f".//PubmedData/History/PubMedPubDate[@PubStatus='{status}']/Year")
        month = article.findtext(f".//PubmedData/History/PubMedPubDate[@PubStatus='{status}']/Month")
        day = article.findtext(f".//PubmedData/History/PubMedPubDate[@PubStatus='{status}']/Day")
        dt = _parse_ymd(year, month, day, require_month=True)
        if dt:
            return dt
    return None


def _parse_ymd(
    year: str | None,
    month: str | None,
    day: str | None,
    *,
    require_month: bool = False,
) -> datetime | None:
    try:
        y = int((year or "").strip())
    except Exception:
        return None

    m_raw = (month or "").strip()
    d_raw = (day or "").strip()

    if require_month and not m_raw:
        return None

    m = 1
    if m_raw:
        if m_raw.isdigit():
            m = int(m_raw)
        else:
            m = _MONTHS.get(m_raw.lower(), 1)

    d = 1
    if d_raw and d_raw.isdigit():
        d = int(d_raw)

    try:
        return datetime(y, m, d)
    except Exception:
        return None


def _parse_medline_date(text: str) -> datetime | None:
    text = (text or "").strip()
    if not text:
        return None

    # Examples:
    # - "2025 Dec 12"
    # - "2025 Dec"
    # - "2025"
    m = re.search(r"(?P<y>\\d{4})(?:\\s+(?P<mon>[A-Za-z]{3,9}|\\d{1,2}))?(?:\\s+(?P<d>\\d{1,2}))?", text)
    if not m:
        return None
    return _parse_ymd(m.group("y"), m.group("mon"), m.group("d"))


def _pubmed_search_url(term: str) -> str:
    return "https://pubmed.ncbi.nlm.nih.gov/?term=" + urllib.parse.quote(term, safe="")


def _eutils_params(eutils: _EutilsConfig) -> dict[str, str]:
    params: dict[str, str] = {"tool": eutils.tool}
    if eutils.email:
        params["email"] = eutils.email
    if eutils.api_key:
        params["api_key"] = eutils.api_key
    return params


def _get_env(key: str) -> str | None:
    import os

    value = os.getenv(key, "").strip()
    return value or None


def _get_json(url: str, params: dict[str, str]) -> dict:
    text = _get_text(url, params)
    return json.loads(text)


def _get_text(url: str, params: dict[str, str]) -> str:
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = httpx.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(0.8 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {resp.status_code}")
        except Exception as e:
            last_exc = e
            time.sleep(0.8 * (attempt + 1))
            continue
    raise RuntimeError(f"Request failed: {last_exc}")


def _collect_text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()


def _save_raw(source_name: str, items: list[NewsItem]) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    dir_path = RAW_DIR / today
    dir_path.mkdir(parents=True, exist_ok=True)

    filename = _sanitize_filename(source_name) + ".json"
    filepath = dir_path / filename

    data = [item.model_dump() for item in items]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _sanitize_filename(name: str) -> str:
    cleaned = []
    for ch in name.lower():
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in (" ", "-", "_", ":"):
            cleaned.append("_")
    out = "".join(cleaned).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "pubmed"


def _normalize_doi(doi: str | None) -> str:
    if not doi:
        return ""
    doi = doi.strip()
    for prefix in ("doi:", "DOI:", "https://doi.org/", "http://doi.org/"):
        if doi.lower().startswith(prefix.lower()):
            doi = doi[len(prefix):].strip()
            break
    return doi.lower().strip()
