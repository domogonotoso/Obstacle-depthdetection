"""Hardware-free TOML settings shared by the detector and Windows client."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tomllib

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.toml"


@dataclass(frozen=True)
class DetectionSettings:
    class_names: dict[int, str]

    def __post_init__(self) -> None:
        if not self.class_names:
            raise ValueError("detection.classes must contain at least one COCO class")
        for class_id, name in self.class_names.items():
            if type(class_id) is not int or not 0 <= class_id <= 79:
                raise ValueError("detection.classes IDs must be integers in [0, 79]")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("detection.classes names must be nonempty strings")


@dataclass(frozen=True)
class AlertSettings:
    display_max_m: float = 3.0
    warning_m: float = 2.0
    danger_m: float = 1.0
    warning_repeat_s: float = 3.0
    danger_repeat_s: float = 1.5
    sound_enabled: bool = True
    smoothing_window: int = 5
    confirm_frames: int = 2
    hysteresis_m: float = 0.15
    center_overlap_ratio: float = 0.35
    track_iou_threshold: float = 0.3
    track_ttl_s: float = 1.0
    result_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        numeric = ("display_max_m", "warning_m", "danger_m", "warning_repeat_s", "danger_repeat_s",
                   "hysteresis_m", "center_overlap_ratio", "track_iou_threshold", "track_ttl_s",
                   "result_timeout_s")
        for name in numeric:
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"alerts.{name} must be a finite number")
        if not 0 < self.danger_m <= self.warning_m <= self.display_max_m <= 20:
            raise ValueError("alerts distances must satisfy 0 < danger_m <= warning_m <= display_max_m <= 20")
        if min(self.warning_repeat_s, self.danger_repeat_s, self.track_ttl_s, self.result_timeout_s) <= 0:
            raise ValueError("alerts repeat intervals and timeouts must be positive")
        if not 0 <= self.hysteresis_m < self.danger_m:
            raise ValueError("alerts.hysteresis_m must be nonnegative and smaller than danger_m")
        if not 0 < self.center_overlap_ratio <= 1 or not 0 < self.track_iou_threshold <= 1:
            raise ValueError("alerts overlap and IoU thresholds must be in (0, 1]")
        if type(self.sound_enabled) is not bool:
            raise ValueError("alerts.sound_enabled must be true or false")
        if type(self.smoothing_window) is not int or not 1 <= self.smoothing_window <= 31:
            raise ValueError("alerts.smoothing_window must be an integer in [1, 31]")
        if type(self.confirm_frames) is not int or not 1 <= self.confirm_frames <= 30:
            raise ValueError("alerts.confirm_frames must be an integer in [1, 30]")


@dataclass(frozen=True)
class AppSettings:
    detection: DetectionSettings
    alerts: AlertSettings
    path: Path


def load_settings(path: str | Path | None = None) -> AppSettings:
    settings_path = Path(path or os.getenv("OBSTACLE_CONFIG") or DEFAULT_SETTINGS_PATH)
    try:
        with settings_path.open("rb") as stream:
            data = tomllib.load(stream)
        if set(data) != {"detection", "alerts"} or set(data["detection"]) != {"classes"}:
            raise ValueError("Settings must contain [detection.classes] and [alerts] tables")
        classes = data["detection"]["classes"]
        if not isinstance(classes, dict):
            raise ValueError("detection.classes must map IDs to model.names labels")
        class_names = {}
        for key, name in classes.items():
            if not key.isascii() or not key.isdecimal():
                raise ValueError(f"Invalid COCO class ID: {key}")
            class_id = int(key)
            if class_id in class_names:
                raise ValueError(f"Duplicate COCO class ID: {class_id}")
            class_names[class_id] = name
        return AppSettings(DetectionSettings(class_names), AlertSettings(**data["alerts"]), settings_path)
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"Cannot load settings {settings_path}: {exc}") from exc
