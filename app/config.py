"""Validated settings shared by the local runner and the inference service."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re

from app.settings import DetectionSettings, load_settings


def _default_target_classes() -> tuple[tuple[int, str], ...]:
    return tuple(load_settings().detection.class_names.items())


@dataclass(frozen=True)
class VisionConfig:
    device: str = "cuda:0"
    model_dir: Path = Path("models")
    yolo_size: int = 640
    depth_size: int = 518
    confidence: float = 0.35
    central_ratio: float = 0.5
    min_valid_fraction: float = 0.2
    min_depth_m: float = 0.05
    max_depth_m: float = 20.0
    target_classes: tuple[tuple[int, str], ...] = field(default_factory=_default_target_classes)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_dir", Path(self.model_dir))
        names = dict(self.target_classes)
        DetectionSettings(names)
        if len(names) != len(self.target_classes):
            raise ValueError("target_classes contains duplicate class IDs")
        if not isinstance(self.device, str) or re.fullmatch(r"cpu|cuda(?::\d+)?", self.device) is None:
            raise ValueError("device must be cpu, cuda, or cuda:<index>")
        if (type(self.yolo_size) is not int or self.yolo_size < 32
                or self.yolo_size > 2048 or self.yolo_size % 32):
            raise ValueError("yolo_size must be a multiple of 32 in [32, 2048]")
        if type(self.depth_size) is not int or not 56 <= self.depth_size <= 1036 or self.depth_size % 14:
            raise ValueError("depth_size must be a multiple of 14 in [56, 1036]")
        for name in ("confidence", "central_ratio", "min_valid_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite in (0, 1]")
        if not (math.isfinite(self.min_depth_m) and math.isfinite(self.max_depth_m)
                and 0 < self.min_depth_m < self.max_depth_m <= 20):
            raise ValueError("depth validity bounds must satisfy 0 < min < max <= 20m")

    @classmethod
    def from_env(cls) -> "VisionConfig":
        return cls(
            device=os.getenv("DEVICE", "cuda:0"),
            model_dir=Path(os.getenv("MODEL_DIR", "models")),
            yolo_size=int(os.getenv("YOLO_SIZE", "640")),
            depth_size=int(os.getenv("DEPTH_SIZE", "518")),
            confidence=float(os.getenv("CONFIDENCE", "0.35")),
            central_ratio=float(os.getenv("CENTRAL_RATIO", "0.5")),
            min_valid_fraction=float(os.getenv("MIN_VALID_FRACTION", "0.2")),
            min_depth_m=float(os.getenv("MIN_DEPTH_M", "0.05")),
            max_depth_m=float(os.getenv("MAX_DEPTH_M", "20.0")),
        )
