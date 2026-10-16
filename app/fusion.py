"""Align detection boxes with an already aligned metric depth map.

The central crop reduces background contamination but cannot separate an object's
surface from occluders or holes. The median is estimated camera-axis depth, not
necessarily Euclidean distance to the object or its nearest surface.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from app.types import Detection, ObjectEstimate


def classify_direction(center_x: float, width: int) -> str:
    """Divide image width into equal LEFT, CENTER, and RIGHT regions."""
    if width <= 0 or not math.isfinite(width):
        raise ValueError("Frame width must be finite and positive")
    if not math.isfinite(center_x):
        raise ValueError("Object center must be finite")
    if center_x < width / 3:
        return "LEFT"
    if center_x < 2 * width / 3:
        return "CENTER"
    return "RIGHT"


def fuse_detections(
    detections: Iterable[Detection],
    depth_map: np.ndarray,
    frame_shape: Sequence[int],
    *,
    central_ratio: float = 0.5,
    min_valid_fraction: float = 0.2,
    min_depth_m: float = 0.05,
    max_depth_m: float = 20.0,
) -> list[ObjectEstimate]:
    """Use the median of valid central-box pixels as each object's depth.

    ``central_ratio`` applies separately to box width and height; 0.5 uses the
    central quarter of the box area. Boxes are clipped to the frame; malformed,
    nonfinite, reversed, or entirely out-of-frame boxes are discarded. Depth must
    already have exactly the source image's height and width: silent rescaling
    here could associate pixels with the wrong object.
    """
    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("Frame dimensions must be positive")
    if not math.isfinite(central_ratio) or not 0 < central_ratio <= 1:
        raise ValueError("central_ratio must be greater than zero and at most one")
    if not math.isfinite(min_valid_fraction) or not 0 <= min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be between zero and one")
    if not all(math.isfinite(value) for value in (min_depth_m, max_depth_m)):
        raise ValueError("Depth bounds must be finite")
    if not 0 < min_depth_m < max_depth_m:
        raise ValueError("Depth bounds must satisfy 0 < min_depth_m < max_depth_m")
    depth = np.asarray(depth_map)
    if depth.shape != (height, width):
        raise ValueError(
            f"Depth map shape {depth.shape} is not aligned with frame {(height, width)}"
        )
    if not np.issubdtype(depth.dtype, np.number) or np.iscomplexobj(depth):
        raise ValueError("Depth map must contain real numeric values in meters")

    estimates: list[ObjectEstimate] = []
    for detection in detections:
        try:
            if len(detection.bbox) != 4:
                continue
            left, top, right, bottom = (float(value) for value in detection.bbox)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (left, top, right, bottom)):
            continue
        left, right = max(0.0, left), min(float(width), right)
        top, bottom = max(0.0, top), min(float(height), bottom)
        if right <= left or bottom <= top:
            continue

        margin_x = (right - left) * (1 - central_ratio) / 2
        margin_y = (bottom - top) * (1 - central_ratio) / 2
        roi_left = max(0, min(width - 1, math.floor(left + margin_x)))
        roi_right = max(roi_left + 1, min(width, math.ceil(right - margin_x)))
        roi_top = max(0, min(height - 1, math.floor(top + margin_y)))
        roi_bottom = max(roi_top + 1, min(height, math.ceil(bottom - margin_y)))
        roi = depth[roi_top:roi_bottom, roi_left:roi_right]
        valid_mask = (
            np.isfinite(roi)
            & (roi >= min_depth_m)
            & (roi <= max_depth_m)
            & (roi > 0)
        )
        sample_count = int(np.count_nonzero(valid_mask))
        valid_fraction = sample_count / roi.size
        distance = None
        if sample_count and valid_fraction >= min_valid_fraction:
            distance = float(np.median(roi[valid_mask]))
        estimates.append(
            ObjectEstimate(
                bbox=(left, top, right, bottom),
                label=detection.label,
                confidence=float(detection.confidence),
                class_id=int(detection.class_id),
                direction=classify_direction((left + right) / 2, width),
                distance_m=distance,
                valid_fraction=valid_fraction,
                depth_samples=sample_count,
            )
        )
    return estimates
