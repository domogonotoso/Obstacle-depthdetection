"""OpenCV presentation; drawing never mutates the image used for inference."""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.types import InferenceResult
from app.alerts import visible_objects


def distance_color(distance_m: float | None, danger_m: float = 1.0,
                   warning_m: float = 2.0) -> tuple[int, int, int]:
    """Arbitrary display thresholds only, in OpenCV BGR order."""
    if distance_m is None or not math.isfinite(distance_m) or distance_m <= 0:
        return (180, 180, 180)
    if distance_m <= danger_m:
        return (50, 50, 255)
    if distance_m <= warning_m:
        return (0, 230, 255)
    return (60, 220, 60)


def _text(image: np.ndarray, text: str, position: tuple[int, int],
          color: tuple[int, int, int] = (255, 255, 255), scale: float = 0.5) -> None:
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def render_frame(
    frame: np.ndarray,
    result: InferenceResult | None = None,
    *,
    fps: float = 0.0,
    age_ms: float = 0.0,
    show_zones: bool = True,
    status: str = "",
    selected_index: int = 0,
    max_display_distance_m: float = 3.0,
    warning_m: float = 2.0,
    danger_m: float = 1.0,
    alert_level: str = "Unknown",
    sound_enabled: bool = False,
    nearest_distance_m: float | None = None,
    invalid_depth_count: int = 0,
) -> np.ndarray:
    image = frame.copy()
    height, width = image.shape[:2]
    if result is not None and (result.width != width or result.height != height):
        raise ValueError("Inference result and display frame dimensions must match.")
    if show_zones:
        for x in (width // 3, 2 * width // 3):
            cv2.line(image, (x, 76), (x, height - 1), (130, 130, 130), 1)
        for index, direction in enumerate(("LEFT", "CENTER", "RIGHT")):
            _text(image, direction, (int((index + 0.5) * width / 3) - 28, height - 12),
                  (210, 210, 210), 0.45)
    annotations = []
    if result is not None:
        for index, obj in visible_objects(result, max_display_distance_m):
            x1, y1, x2, y2 = obj.bbox
            left = max(0, min(width - 1, round(x1)))
            top = max(0, min(height - 1, round(y1)))
            right = max(0, min(width - 1, round(x2)))
            bottom = max(0, min(height - 1, round(y2)))
            color = distance_color(obj.distance_m, danger_m, warning_m)
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            distance = f"Est. {obj.distance_m:.2f}m" if obj.distance_m is not None else "Est. N/A"
            marker = "*" if index == selected_index else ""
            lines = [f"{index}{marker} {obj.label.title()} {obj.confidence:.0%}",
                     f"{obj.direction} {distance}"]
            annotations.append((left, top, lines, color))
    # Labels are laid out after every box is drawn so overlapping boxes cannot
    # overwrite text. Prefer free vertical space near each object's anchor.
    occupied: list[tuple[int, int, int, int]] = []
    for left, top, lines, color in annotations:
        block_width = min(width - 4, max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                                       0.5, 1)[0][0] for line in lines) + 10)
        block_height = 40
        x = max(2, min(left, width - block_width - 2))
        max_y = max(76, height - block_height - 25)
        preferred_y = min(max(76, top - block_height - 4), max_y)
        y = preferred_y
        for candidate in [preferred_y, *range(76, max_y + 1, block_height + 5)]:
            rectangle = (x, candidate, x + block_width, candidate + block_height)
            if all(rectangle[2] <= used[0] or rectangle[0] >= used[2]
                   or rectangle[3] <= used[1] or rectangle[1] >= used[3] for used in occupied):
                y = candidate
                break
        occupied.append((x, y, x + block_width, y + block_height))
        cv2.rectangle(image, (x, y), (x + block_width, y + block_height), (24, 24, 24), -1)
        for line_index, line in enumerate(lines):
            _text(image, line, (x + 4, y + 16 + line_index * 18), color)
    cv2.rectangle(image, (0, 0), (width - 1, 73), (25, 25, 25), -1)
    _text(image, f"FPS {fps:.1f} | age {age_ms:.0f}ms | Q quit  M mute  S record", (8, 20))
    level_color = {"Normal": (60, 220, 60), "Warning": (0, 230, 255),
                   "Danger": (50, 50, 255)}.get(alert_level, (180, 180, 180))
    nearest = f" | Est. {nearest_distance_m:.2f}m" if nearest_distance_m is not None else ""
    _text(image, f"Sound {'ON' if sound_enabled else 'OFF'} | Alert {alert_level}{nearest}",
          (8, 42), level_color)
    _text(image, f"Estimated Distance | max {max_display_distance_m:g}m | invalid depth {invalid_depth_count}",
          (8, 64), (210, 210, 210), 0.44)
    if status:
        _text(image, status[:max(20, width // 7)], (8, max(70, height - 34)), (0, 200, 255))
    return image
