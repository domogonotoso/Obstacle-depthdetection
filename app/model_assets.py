"""Pinned official model downloads, validated before loading any weights."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.request

LOGGER = logging.getLogger(__name__)
YOLO_FILENAME = "yolo11n.pt"
DEPTH_FILENAME = "depth_anything_v2_metric_hypersim_vits.pth"
DEPTH_SHA256 = "b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545"
DEPTH_REVISION = "3bc65d4e14a6786a61acec16453c50e12bf5f338"
YOLO_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
ASSETS = {
    "yolo": (
        YOLO_FILENAME,
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        YOLO_SHA256,
        20_000_000,
    ),
    "depth": (
        DEPTH_FILENAME,
        f"https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/{DEPTH_REVISION}/{DEPTH_FILENAME}",
        DEPTH_SHA256,
        110_000_000,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_asset(kind: str, model_dir: Path) -> Path:
    """Cache downloads atomically; incomplete/incorrect files never become weights."""
    filename, url, expected_hash, max_bytes = ASSETS[kind]
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / filename
    if target.is_file():
        size = target.stat().st_size
        if size == 0 or size > max_bytes:
            raise RuntimeError(f"Cached {filename} has invalid size; remove it and retry")
        if expected_hash and sha256_file(target) != expected_hash:
            raise RuntimeError(f"Cached {filename} SHA256 mismatch; remove it and retry")
        return target

    LOGGER.info("Downloading official %s weights (cached after first run)", kind)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=model_dir, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            request = urllib.request.Request(url, headers={"User-Agent": "obstacle-vision/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                total = 0
                digest = hashlib.sha256()
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > max_bytes:
                        raise RuntimeError(f"Download exceeds expected maximum size for {filename}")
                    output.write(block)
                    digest.update(block)
                if not total:
                    raise RuntimeError(f"Empty download for {filename}")
                if expected_hash and digest.hexdigest() != expected_hash:
                    raise RuntimeError(f"Official {filename} download SHA256 mismatch")
        os.replace(temporary, target)
    except (OSError, urllib.error.URLError, RuntimeError) as exc:
        raise RuntimeError(
            f"Cannot prepare {filename}: {exc}. Check network access to GitHub/Hugging Face, "
            "or place the official checkpoint in MODEL_DIR. See THIRD_PARTY.md."
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target
