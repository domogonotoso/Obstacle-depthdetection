"""Same-frame detection, metric depth, and image-coordinate fusion."""
from __future__ import annotations

import time

import numpy as np
import torch

from app.config import VisionConfig
from app.depth_estimator import MetricDepthEstimator
from app.detector import ObjectDetector
from app.fusion import fuse_detections
from app.model_assets import sha256_file
from app.runtime import gpu_diagnostics, resolve_device
from app.types import InferenceResult


class VisionPipeline:
    def __init__(self, config: VisionConfig | None = None):
        self.config = config or VisionConfig()
        self.device = resolve_device(self.config.device)
        self.detector = ObjectDetector(self.config)
        self.depth_estimator = MetricDepthEstimator(self.config)
        self._weight_hashes = {
            "yolo11n": sha256_file(self.detector.weights_path),
            "metric_hypersim_small": sha256_file(self.depth_estimator.weights_path),
        }

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def infer(self, frame: np.ndarray) -> InferenceResult:
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0):
            raise ValueError("Input must be a nonempty HxWx3 uint8 BGR frame")
        self._sync()
        started = time.perf_counter()
        detections = self.detector.infer(frame)
        self._sync()
        detected = time.perf_counter()
        depth = self.depth_estimator.infer(frame)
        self._sync()
        estimated = time.perf_counter()
        objects = fuse_detections(
            detections, depth, frame.shape,
            central_ratio=self.config.central_ratio,
            min_valid_fraction=self.config.min_valid_fraction,
            min_depth_m=self.config.min_depth_m, max_depth_m=self.config.max_depth_m,
        )
        finished = time.perf_counter()
        return InferenceResult(
            width=frame.shape[1], height=frame.shape[0], objects=objects,
            timings_ms={
                "yolo": (detected - started) * 1000,
                "depth": (estimated - detected) * 1000,
                "fusion": (finished - estimated) * 1000,
                "total": (finished - started) * 1000,
            }, device=str(self.device),
        )

    def diagnostics(self) -> dict:
        return {
            **gpu_diagnostics(self.device),
            "yolo_model": "yolo11n", "depth_model": "Metric-Depth-Anything-V2-Hypersim-Small",
            "depth_unit": "meters", "depth_model_max_m": 20.0,
            "yolo_size": self.config.yolo_size, "depth_size": self.config.depth_size,
            "target_classes": dict(self.config.target_classes),
            "detector_device": str(next(self.detector.model.model.parameters()).device),
            "depth_device": str(next(self.depth_estimator.model.parameters()).device),
            "weights_sha256": self._weight_hashes,
        }
