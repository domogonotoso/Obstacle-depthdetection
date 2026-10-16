"""Official Metric Depth Anything V2 Small/Hypersim, never relative depth."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose
from torch.nn import functional as F

from app.config import VisionConfig
from app.model_assets import ensure_asset
from app.runtime import resolve_device
from third_party.depth_anything_v2_metric.dpt import DepthAnythingV2
from third_party.depth_anything_v2_metric.util.transform import NormalizeImage, PrepareForNet, Resize


class MetricDepthEstimator:
    MAX_DEPTH_M = 20.0

    def __init__(self, config: VisionConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.weights_path = ensure_asset("depth", config.model_dir)
        self.model = DepthAnythingV2(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384],
            max_depth=self.MAX_DEPTH_M,
        )
        try:
            weights = torch.load(self.weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(weights, strict=True)
        except Exception as exc:
            raise RuntimeError("Cannot load the verified Metric Hypersim Small checkpoint") from exc
        self.model.to(self.device).eval()
        # Identical transforms to the official metric model image2tensor().
        # Device selection is explicit here; upstream auto-selects cuda:0/cpu.
        self.transform = Compose([
            Resize(width=config.depth_size, height=config.depth_size,
                   resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
                   resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    @torch.inference_mode()
    def infer(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        image = self.transform({"image": image})["image"]
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        # Official metric head already returns meters via sigmoid * max_depth.
        depth = self.model(tensor)
        depth = F.interpolate(depth[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0]
        return depth.float().cpu().numpy()
