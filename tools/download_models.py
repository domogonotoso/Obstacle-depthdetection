"""Download the two official checkpoints once, without initializing models."""
import argparse
import logging
from pathlib import Path

from app.model_assets import ensure_asset, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        for name in ("yolo", "depth"):
            path = ensure_asset(name, args.model_dir)
            print(f"{path.name}: {path.stat().st_size:,} bytes; SHA256={sha256_file(path)}")
    except Exception as exc:
        print(f"Download failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
