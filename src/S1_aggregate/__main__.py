"""Run aggregate module standalone."""

from . import fetch_all

if __name__ == "__main__":
    items = fetch_all()
    print(f"\nTotal: {len(items)} items")
