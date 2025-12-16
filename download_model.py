"""Pre-download BGE-M3 model for hybrid filtering."""

import os
from pathlib import Path

def download_bge_m3():
    """Download BGE-M3 model to HuggingFace cache."""
    print("Downloading BGE-M3 model...")
    print("This may take 5-10 minutes (2.24 GB)")

    try:
        from huggingface_hub import snapshot_download

        model_id = "BAAI/bge-m3"
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

        print(f"Model ID: {model_id}")
        print(f"Cache dir: {cache_dir}")
        print()

        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            resume_download=True,
        )

        print("\n✓ Model downloaded successfully!")
        print(f"   Location: {cache_dir}")

    except ImportError:
        print("Error: huggingface_hub not installed")
        print("Install with: pip install huggingface_hub")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("You can retry later or let it download on first use")

if __name__ == "__main__":
    download_bge_m3()
