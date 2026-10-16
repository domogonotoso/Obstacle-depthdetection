"""Append explicitly measured ground-truth comparisons to a portable CSV file."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

from app.types import InferenceResult


CSV_FIELDS = (
    "timestamp_utc", "status", "truth_m", "estimated_distance_m",
    "signed_error_m", "absolute_error_m", "relative_error_pct",
    "object_index", "label", "class_id", "confidence", "direction",
    "bbox_left", "bbox_top", "bbox_right", "bbox_bottom",
    "valid_fraction", "depth_samples", "frame_width", "frame_height", "device",
)


class MeasurementRecorder:
    """Record one selected object against a user-supplied tape measurement.

    Selection is an index in the current result, not a persistent object ID.
    The user must visually ensure it is the intended physical object each time.
    Signed error is estimate minus truth; relative error is absolute error/truth.
    """

    def __init__(self, path: str | Path, truth_m: float):
        truth_m = float(truth_m)
        if not math.isfinite(truth_m) or truth_m <= 0:
            raise ValueError("Ground-truth distance must be finite and positive")
        self.path = Path(path)
        self.truth_m = truth_m

    def record(self, result: InferenceResult, object_index: int = 0) -> dict[str, Any] | None:
        """Append and return one row; return None for a frame without objects.

        Missing/invalid depth is recorded with ``status=invalid_depth`` and empty
        error fields. An out-of-range index in a nonempty result raises IndexError.
        """
        if object_index < 0:
            raise IndexError("Measurement object index cannot be negative")
        if not result.objects:
            return None
        if object_index >= len(result.objects):
            raise IndexError(
                f"Object index {object_index} is unavailable; frame has {len(result.objects)} objects"
            )
        obj = result.objects[object_index]
        estimate = obj.distance_m
        valid = estimate is not None and math.isfinite(estimate) and estimate > 0
        signed_error = float(estimate) - self.truth_m if valid else None
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "status": "ok" if valid else "invalid_depth",
            "truth_m": self.truth_m,
            "estimated_distance_m": float(estimate) if valid else "",
            "signed_error_m": signed_error if valid else "",
            "absolute_error_m": abs(signed_error) if valid else "",
            "relative_error_pct": abs(signed_error) / self.truth_m * 100 if valid else "",
            "object_index": object_index,
            "label": obj.label,
            "class_id": obj.class_id,
            "confidence": obj.confidence,
            "direction": obj.direction,
            "bbox_left": obj.bbox[0],
            "bbox_top": obj.bbox[1],
            "bbox_right": obj.bbox[2],
            "bbox_bottom": obj.bbox[3],
            "valid_fraction": obj.valid_fraction,
            "depth_samples": obj.depth_samples,
            "frame_width": result.width,
            "frame_height": result.height,
            "device": result.device,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Refuse appending a different schema to an existing file.
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("r", encoding="utf-8", newline="") as handle:
                if next(csv.reader(handle), []) != list(CSV_FIELDS):
                    raise ValueError(f"Existing CSV has an incompatible header: {self.path}")
        needs_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
        return row
