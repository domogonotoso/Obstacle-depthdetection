"""Real image-based YOLO / Metric Depth / fusion checks and measured timings."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
import torch

from app.config import VisionConfig
from app.runtime import gpu_diagnostics, resolve_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--component", choices=("yolo", "depth", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--yolo-size", type=int, default=640)
    parser.add_argument("--depth-size", type=int, default=518)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke"))
    args = parser.parse_args()
    if args.runs <= 0 or args.warmup < 0:
        parser.error("runs must be positive; warmup cannot be negative")
    try:
        image = cv2.imread(str(args.image))
        if image is None:
            raise ValueError(f"Cannot read image: {args.image}")
        config = VisionConfig(device=args.device, model_dir=args.model_dir,
                              yolo_size=args.yolo_size, depth_size=args.depth_size)
        device = resolve_device(config.device)
        if args.component == "yolo":
            from app.detector import ObjectDetector
            model = ObjectDetector(config)
        elif args.component == "depth":
            from app.depth_estimator import MetricDepthEstimator
            model = MetricDepthEstimator(config)
        else:
            from app.pipeline import VisionPipeline
            model = VisionPipeline(config)
        timings = []
        output = None
        with torch.inference_mode():
            for index in range(args.warmup + args.runs):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                output = model.infer(image)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if index >= args.warmup:
                    timings.append((time.perf_counter() - started) * 1000)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "component": args.component, "image": args.image.name,
            "width": image.shape[1], "height": image.shape[0],
            "yolo_size": config.yolo_size, "depth_size": config.depth_size,
            "warmup": args.warmup, "runs": args.runs,
            "timings_ms": timings, "mean_ms": statistics.mean(timings),
            "median_ms": statistics.median(timings),
            "measured_sequential_fps": 1000 / statistics.mean(timings),
            "measurement_scope": "Repeated still image, excludes camera/JPEG/network/rendering; not webcam FPS",
            "gpu": gpu_diagnostics(device),
        }
        if args.component == "all":
            report["result"] = output.to_dict()
            from app.visualization import render_frame
            cv2.imwrite(str(args.output_dir / "overlay.jpg"), render_frame(image, output))
        elif args.component == "yolo":
            report["detections"] = [asdict(detection) for detection in output]
        else:
            if output.shape != image.shape[:2] or not np.isfinite(output).all():
                raise RuntimeError("Metric depth shape/finite-value validation failed")
            if np.any(output < 0) or np.any(output > 20):
                raise RuntimeError("Metric depth output outside official 0..20m range")
            np.save(args.output_dir / "depth_m.npy", output)
            report["depth"] = {"unit": "meters", "min": float(output.min()),
                               "median": float(np.median(output)), "max": float(output.max())}
            # Fixed 0..20m display scale; never used to compute object distances.
            preview = cv2.applyColorMap((np.clip(output / 20, 0, 1) * 255).astype(np.uint8),
                                       cv2.COLORMAP_TURBO)
            cv2.imwrite(str(args.output_dir / "depth_preview.jpg"), preview)
        target = args.output_dir / f"{args.component}_report.json"
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Saved {target}")
    except Exception as exc:
        print(f"Model smoke test failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
