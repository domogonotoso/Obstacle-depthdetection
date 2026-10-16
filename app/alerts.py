"""Client-side distance smoothing and alert decisions, independent of models/audio.

Distances remain estimates. IoU association is deliberately lightweight: it is not
an identity tracker and cannot reliably associate crossing or occluded objects.
Only fresh, valid results can produce an alert event.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from statistics import median
import time

from app.settings import AlertSettings
from app.types import InferenceResult, ObjectEstimate


@dataclass(frozen=True)
class AlertSnapshot:
    level: str
    nearest_distance_m: float | None
    sound_event: str | None
    result: InferenceResult | None
    invalid_depth_count: int
    sound_enabled: bool


@dataclass
class _Track:
    class_id: int
    bbox: tuple[float, float, float, float]
    distances: deque[float]
    last_seen: float


def valid_distance(value: float | None) -> bool:
    """Missing, zero, negative, NaN and infinite depth never becomes a distance."""
    return value is not None and math.isfinite(value) and value > 0


def visible_objects(
    result: InferenceResult, max_distance_m: float
) -> list[tuple[int, ObjectEstimate]]:
    """Return displayable objects with their original measurement-selection indices."""
    return [
        (index, obj)
        for index, obj in enumerate(result.objects)
        if valid_distance(obj.distance_m) and obj.distance_m <= max_distance_m
    ]


def is_central(obj: ObjectEstimate, width: int, minimum_overlap: float) -> bool:
    """Accept a CENTER centroid or sufficient horizontal overlap with the middle third.

    Overlap is divided by the object's (image-clipped) width. A wide object with
    its center outside CENTER can therefore still occupy the forward corridor.
    """
    if obj.direction == "CENTER":
        return True
    if width <= 0 or not all(math.isfinite(v) for v in obj.bbox):
        return False
    left, _, right, _ = obj.bbox
    left, right = max(0.0, left), min(float(width), right)
    if right <= left:
        return False
    overlap = max(0.0, min(right, width * 2 / 3) - max(left, width / 3))
    return overlap / (right - left) >= minimum_overlap


def _iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    if not all(math.isfinite(value) for value in (*first, *second)):
        return 0.0
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (
        max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        + max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        - intersection
    )
    return intersection / union if union > 0 else 0.0


class AlertController:
    """Associate boxes, smooth estimates, debounce levels and schedule sound events.

    Call ``update`` once per newly completed inference, and ``tick`` between
    results. Consume each returned ``sound_event`` once. ``invalidate`` stops
    alerts immediately after an inference/transport failure. The controller does
    not play audio, retain video frames, or block inference.
    """

    def __init__(self, settings: AlertSettings):
        self.settings = settings
        self.sound_enabled = settings.sound_enabled
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 0
        self._dimensions: tuple[int, int] | None = None
        self._result: InferenceResult | None = None
        self._last_result_at: float | None = None
        self._level = "Unknown"
        self._nearest: float | None = None
        self._invalid_count = 0
        self._pending_level: str | None = None
        self._pending_count = 0
        self._last_sound_at: float | None = None

    def set_sound_enabled(self, enabled: bool) -> bool:
        self.sound_enabled = bool(enabled)
        return self.sound_enabled

    def toggle_sound(self) -> bool:
        return self.set_sound_enabled(not self.sound_enabled)

    def update(self, result: InferenceResult, now: float | None = None) -> AlertSnapshot:
        current = time.monotonic() if now is None else now
        self._expire(current)
        dimensions = (result.width, result.height)
        if dimensions != self._dimensions:
            self._tracks.clear()
            self._dimensions = dimensions
        objects = self._smooth(result.objects, current)
        self._result = replace(result, objects=objects)
        self._last_result_at = current
        self._invalid_count = sum(not valid_distance(obj.distance_m) for obj in objects)
        central = [
            obj for obj in objects
            if is_central(obj, result.width, self.settings.center_overlap_ratio)
        ]
        distances = [obj.distance_m for obj in central if valid_distance(obj.distance_m)]
        self._nearest = min(distances) if distances else None
        desired = self._desired_level(self._nearest) if distances else (
            "Unknown" if central else "Normal"
        )
        previous = self._level
        self._transition(desired)
        escalation = previous == "Warning" and self._level == "Danger"
        return self._snapshot(self._sound_event(current, escalation=escalation))

    def tick(self, now: float | None = None) -> AlertSnapshot:
        current = time.monotonic() if now is None else now
        self._expire(current)
        return self._snapshot(self._sound_event(current))

    def invalidate(self, now: float | None = None) -> AlertSnapshot:
        """Clear stale estimates and confirmation history; preserve sound cooldown."""
        self._tracks.clear()
        self._result = None
        self._last_result_at = None
        self._level = "Unknown"
        self._nearest = None
        self._invalid_count = 0
        self._pending_level = None
        self._pending_count = 0
        return self._snapshot(None)

    def _expire(self, now: float) -> None:
        if (
            self._last_result_at is not None
            and now >= self._last_result_at + self.settings.result_timeout_s
        ):
            self.invalidate(now)

    def _smooth(self, objects: list[ObjectEstimate], now: float) -> list[ObjectEstimate]:
        self._tracks = {
            identifier: track for identifier, track in self._tracks.items()
            if now < track.last_seen + self.settings.track_ttl_s
        }
        # Global descending-IoU assignment prevents list reordering from mixing
        # histories. Each prior track and each current box can be used only once.
        pairs = sorted(
            (
                (_iou(obj.bbox, track.bbox), index, identifier)
                for index, obj in enumerate(objects)
                for identifier, track in self._tracks.items()
                if obj.class_id == track.class_id
            ),
            reverse=True,
        )
        matched: dict[int, int] = {}
        used_tracks: set[int] = set()
        for overlap, index, identifier in pairs:
            if (
                overlap >= self.settings.track_iou_threshold
                and index not in matched and identifier not in used_tracks
            ):
                matched[index] = identifier
                used_tracks.add(identifier)
        smoothed: list[ObjectEstimate] = []
        for index, obj in enumerate(objects):
            identifier = matched.get(index)
            if identifier is None:
                identifier = self._next_track_id
                self._next_track_id += 1
                self._tracks[identifier] = _Track(
                    obj.class_id, obj.bbox, deque(maxlen=self.settings.smoothing_window), now
                )
            track = self._tracks[identifier]
            track.bbox, track.last_seen = obj.bbox, now
            if valid_distance(obj.distance_m):
                track.distances.append(float(obj.distance_m))
                smoothed.append(replace(obj, distance_m=float(median(track.distances))))
            else:
                # A missing current estimate never reuses the previous distance.
                track.distances.clear()
                smoothed.append(replace(obj, distance_m=None))
        return smoothed

    def _desired_level(self, distance: float) -> str:
        if distance <= self.settings.danger_m:
            return "Danger"
        if self._level == "Danger" and distance <= self.settings.danger_m + self.settings.hysteresis_m:
            return "Danger"
        if distance <= self.settings.warning_m:
            return "Warning"
        if (
            self._level in {"Warning", "Danger"}
            and distance <= self.settings.warning_m + self.settings.hysteresis_m
        ):
            return "Warning"
        return "Normal"

    def _transition(self, desired: str) -> None:
        if desired == "Unknown":
            self._level = "Unknown"
            self._pending_level = None
            self._pending_count = 0
        elif desired == self._level:
            self._pending_level = None
            self._pending_count = 0
        else:
            if desired != self._pending_level:
                self._pending_level, self._pending_count = desired, 1
            else:
                self._pending_count += 1
            if self._pending_count >= self.settings.confirm_frames:
                self._level = desired
                self._pending_level, self._pending_count = None, 0

    def _sound_event(self, now: float, *, escalation: bool = False) -> str | None:
        if not self.sound_enabled or self._level not in {"Warning", "Danger"}:
            return None
        if self._last_result_at is None or now >= self._last_result_at + self.settings.result_timeout_s:
            return None
        # Current missing depth or no central object stops sound immediately,
        # even if a Normal demotion is awaiting its confirmation frames.
        if self._nearest is None:
            return None
        exit_distance = (
            self.settings.danger_m if self._level == "Danger" else self.settings.warning_m
        ) + self.settings.hysteresis_m
        if self._nearest > exit_distance:
            return None
        interval = (
            self.settings.danger_repeat_s if self._level == "Danger"
            else self.settings.warning_repeat_s
        )
        if escalation or self._last_sound_at is None or now >= self._last_sound_at + interval:
            self._last_sound_at = now
            return self._level
        return None

    def _snapshot(self, event: str | None) -> AlertSnapshot:
        return AlertSnapshot(
            self._level, self._nearest, event, self._result,
            self._invalid_count, self.sound_enabled,
        )
