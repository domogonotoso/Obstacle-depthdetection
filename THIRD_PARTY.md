# Model and source provenance

* Detector: official Ultralytics YOLO11n, COCO pretrained detection model.
  https://docs.ultralytics.com/models/yolo11/
  https://github.com/ultralytics/assets/releases/tag/v8.3.0
  Ultralytics software/weights have AGPL-3.0 / commercial licensing options.
* Depth: **Metric** Depth Anything V2 Small, Hypersim indoor checkpoint.
  https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth
  Source commit: `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`.
  `third_party/depth_anything_v2_metric` is an unmodified copy of Python files
  from `metric_depth/depth_anything_v2` in that commit. It is deliberately given
  a distinct package name to prevent importing the relative-depth implementation.
  Upstream license files are preserved alongside it.
* Checkpoint repository: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small
  Revision: `3bc65d4e14a6786a61acec16453c50e12bf5f338`.
  File: `depth_anything_v2_metric_hypersim_vits.pth` (99,222,290 bytes).
  SHA256: `b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545`.
  Small checkpoint model card: Apache-2.0; see upstream model card and bundled
  source licenses before redistribution. Larger variants have different terms.

The depth wrapper calls the official preprocessing and forward implementation.
It explicitly places the preprocessed tensor on the configured device rather
than using upstream `infer_image`, which auto-selects a device independently.
No relative depth normalization or metric conversion is applied to the output.
Official head: sigmoid * 20m; output is resized back to the original image size.

No upstream model weights are committed. The downloader verifies the pinned
Hypersim checkpoint SHA256 and rejects an incomplete or changed file. YOLO
weights are retrieved from the fixed official release URL, cached locally,
and verified against SHA256
`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
Both observed hashes are included by diagnostics.
