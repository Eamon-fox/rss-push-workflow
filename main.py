"""ScholarPipe - Main entry point."""

from src.database import Database
from src.aggregator import Aggregator
from src.dedup import Deduplicator
from src.screener import Screener
from src.fetcher import Fetcher
from src.digester import Digester
from src.delivery import Delivery


def run_pipeline():
    """Run the complete ScholarPipe pipeline."""

    # Initialize components
    db = Database()
    aggregator = Aggregator()
    dedup = Deduplicator(db)
    screener = Screener()
    fetcher = Fetcher()
    digester = Digester()
    delivery = Delivery()

    # Step 1: Aggregate
    print("[1/6] Aggregating papers...")
    papers = aggregator.fetch("nature_rss")

    # Step 2: Deduplicate
    print("[2/6] Deduplicating...")
    new_papers = []
    for paper in papers:
        if not dedup.exists(paper.doi):
            db.insert(paper)
            new_papers.append(paper)

    print(f"    Found {len(new_papers)} new papers")

    # Step 3: Screen
    print("[3/6] AI Screening...")
    for paper in new_papers:
        score, reason = screener.evaluate(paper.title, paper.abstract)
        db.update_score(paper.doi, score, reason)
        paper.score = score

    # Step 4: Fetch PDFs
    print("[4/6] Fetching PDFs...")
    high_score = [p for p in new_papers if p.score and p.score >= 6]
    for paper in high_score:
        pdf_path = fetcher.download(paper)
        if pdf_path:
            db.update_pdf_path(paper.doi, pdf_path)
            paper.pdf_path = pdf_path

    # Step 5: Deep Analysis
    print("[5/6] Deep analysis...")
    for paper in high_score:
        if paper.pdf_path:
            summary = digester.analyze(paper)
            paper.summary = summary

    # Step 6: Deliver
    print("[6/6] Outputting results...")
    delivery.output(high_score)


if __name__ == "__main__":
    run_pipeline()
