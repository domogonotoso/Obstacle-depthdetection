"""Small, model-independent data contracts shared by local and remote clients."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class Detection:
    """An image-space box in ``(left, top, right, bottom)`` coordinates."""

    bbox: tuple[float, float, float, float]
    label: str
    confidence: float
    class_id: int


@dataclass(frozen=True)
class ObjectEstimate(Detection):
    direction: str
    distance_m: float | None
    valid_fraction: float
    depth_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [float(value) for value in self.bbox],
            "label": self.label,
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "direction": self.direction,
            "distance_m": None if self.distance_m is None else float(self.distance_m),
            "valid_fraction": float(self.valid_fraction),
            "depth_samples": int(self.depth_samples),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObjectEstimate:
        bbox = tuple(float(value) for value in data["bbox"])
        if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
            raise ValueError("Object bbox must contain four finite coordinates")
        distance = data.get("distance_m")
        if distance is not None:
            distance = float(distance)
            if not math.isfinite(distance) or distance <= 0:
                raise ValueError("Object distance_m must be finite and positive, or null")
        confidence = float(data["confidence"])
        valid_fraction = float(data["valid_fraction"])
        depth_samples = int(data["depth_samples"])
        direction = str(data["direction"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Object confidence must be between zero and one")
        if not math.isfinite(valid_fraction) or not 0 <= valid_fraction <= 1:
            raise ValueError("Object valid_fraction must be between zero and one")
        if depth_samples < 0:
            raise ValueError("Object depth_samples cannot be negative")
        if direction not in {"LEFT", "CENTER", "RIGHT"}:
            raise ValueError("Object direction must be LEFT, CENTER, or RIGHT")
        return cls(
            bbox=bbox,  # type: ignore[arg-type]
            label=str(data["label"]),
            confidence=confidence,
            class_id=int(data["class_id"]),
            direction=direction,
            distance_m=distance,
            valid_fraction=valid_fraction,
            depth_samples=depth_samples,
        )


@dataclass(frozen=True)
class InferenceResult:
    width: int
    height: int
    objects: list[ObjectEstimate]
    timings_ms: dict[str, float]
    device: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "objects": [obj.to_dict() for obj in self.objects],
            "timings_ms": {name: float(value) for name, value in self.timings_ms.items()},
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InferenceResult:
        width, height = int(data["width"]), int(data["height"])
        if width <= 0 or height <= 0:
            raise ValueError("Inference image dimensions must be positive")
        timings = {str(name): float(value) for name, value in data["timings_ms"].items()}
        if any(not math.isfinite(value) or value < 0 for value in timings.values()):
            raise ValueError("Inference timings must be finite and nonnegative")
        return cls(
            width=width,
            height=height,
            objects=[ObjectEstimate.from_dict(obj) for obj in data["objects"]],
            timings_ms=timings,
            device=str(data["device"]),
        )
