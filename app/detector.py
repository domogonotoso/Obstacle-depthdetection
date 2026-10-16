"""Official COCO-pretrained Ultralytics YOLO11 nano detector."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from app.config import VisionConfig
from app.model_assets import ensure_asset
from app.runtime import resolve_device
from app.types import Detection


class ObjectDetector:
    def __init__(self, config: VisionConfig):
        # Keep Ultralytics settings writable and local to the model cache.
        settings_dir = Path(os.environ.get("YOLO_CONFIG_DIR",
                            str(Path(config.model_dir).resolve() / "ultralytics")))
        settings_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_dir))
        from ultralytics import YOLO

        self.config = config
        self.device = resolve_device(config.device)
        self.weights_path = ensure_asset("yolo", config.model_dir)
        try:
            self.model = YOLO(str(self.weights_path), task="detect")
            self.model.to(self.device)
        except Exception as exc:
            raise RuntimeError("Cannot load official YOLO11n weights; check the cached checkpoint") from exc
        # Verify the configured COCO IDs against the checkpoint actually loaded.
        self.class_names = {}
        for class_id, expected_name in config.target_classes:
            try:
                actual = self.model.names[class_id]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(f"Configured class ID {class_id} is missing from model.names") from exc
            if actual.strip().casefold() != expected_name.strip().casefold():
                raise ValueError(
                    f"Class ID {class_id} mismatch: configured {expected_name!r}, model.names has {actual!r}"
                )
            self.class_names[class_id] = actual

    @torch.inference_mode()
    def infer(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=frame, imgsz=self.config.yolo_size,
            conf=self.config.confidence, device=str(self.device),
            classes=list(self.class_names),
            verbose=False, half=False, save=False, max_det=100,
        )
        output: list[Detection] = []
        if not results or results[0].boxes is None:
            return output
        result = results[0]
        # xyxy is already mapped by Ultralytics to the original frame coordinates.
        # Copy the complete small detection table to CPU only once.
        rows = result.boxes.data.detach().cpu().numpy()
        for row in rows:
            left, top, right, bottom, confidence, class_id = row[:6]
            output.append(Detection(
                bbox=(float(left), float(top), float(right), float(bottom)),
                label=result.names[int(class_id)], confidence=float(confidence), class_id=int(class_id),
            ))
        return output
