"""PDF download functionality."""

from pathlib import Path

from .models import Paper


class Fetcher:
    """Download PDFs from various sources."""

    def __init__(self, download_dir: str = "downloads/", timeout: int = 30):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def download(self, paper: Paper) -> str | None:
        """
        Download PDF for a paper.

        Returns:
            str: Path to downloaded file, or None if failed
        """
        # TODO: Implement download logic
        pass

    def _generate_filename(self, paper: Paper) -> str:
        """Generate standardized filename."""
        # Format: {Year}_{Source}_{FirstAuthor}_{ShortTitle}.pdf
        year = paper.published_at.year if paper.published_at else "XXXX"
        source = paper.source.replace(" ", "")[:20]
        author = paper.authors[0].split()[-1] if paper.authors else "Unknown"
        title = "".join(c for c in paper.title[:30] if c.isalnum() or c.isspace())
        title = title.replace(" ", "_")
        return f"{year}_{source}_{author}_{title}.pdf"

    def _try_direct_download(self, url: str, filepath: Path) -> bool:
        """Try direct HTTP download."""
        # TODO: Implement with httpx
        pass

    def _try_scihub(self, doi: str, filepath: Path) -> bool:
        """Try downloading via Sci-Hub."""
        # TODO: Implement Sci-Hub fallback
        pass
