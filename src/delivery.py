"""Output and distribution - WordPress, Zotero, etc."""

from .models import Paper


class Delivery:
    """Distribute processed papers to various outputs."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def output(self, papers: list[Paper]) -> None:
        """Output papers (default: console)."""
        self.print_summary(papers)

    def print_summary(self, papers: list[Paper]) -> None:
        """Print formatted summary to console."""
        print(f"\n{'='*60}")
        print(f"ScholarPipe Results: {len(papers)} papers processed")
        print(f"{'='*60}\n")

        for i, paper in enumerate(papers, 1):
            print(f"[{i}] {paper.title[:70]}...")
            print(f"    DOI: {paper.doi or 'N/A'}")
            print(f"    Score: {paper.score or 'N/A'}")
            print(f"    Status: {paper.status}")
            if paper.pdf_path:
                print(f"    PDF: {paper.pdf_path}")
            print()

    def to_json(self, papers: list[Paper], filepath: str) -> None:
        """Export papers to JSON file."""
        # TODO: Implement
        pass

    def to_wordpress(self, paper: Paper) -> str:
        """Publish paper to WordPress. Returns post URL."""
        # TODO: Implement WordPress API
        pass

    def to_zotero(self, paper: Paper) -> None:
        """Add paper to Zotero library."""
        # TODO: Implement Zotero API
        pass
