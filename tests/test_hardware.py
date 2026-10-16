"""Explicit opt-in checks; skipped cases never imply hardware validation."""
import os
import time

import pytest

pytestmark = [pytest.mark.hardware,
              pytest.mark.skipif(os.getenv("RUN_HARDWARE_TESTS") != "1",
                                 reason="Set RUN_HARDWARE_TESTS=1 for real CUDA and webcam tests")]


def test_real_cuda_tensor():
    from app.runtime import gpu_diagnostics
    info = gpu_diagnostics()
    assert info["cuda_available"]
    assert info["cuda_tensor_test"]


def test_real_webcam_pipeline():
    from app.camera import LatestFrameCamera
    from app.config import VisionConfig
    from app.pipeline import VisionPipeline
    pipeline = VisionPipeline(VisionConfig())
    with LatestFrameCamera(index=int(os.getenv("CAMERA_INDEX", "0"))) as camera:
        deadline = time.monotonic() + 8
        frame = None
        while frame is None and time.monotonic() < deadline:
            if camera.error:
                pytest.fail(camera.error)
            frame = camera.latest()
            if frame is None:
                time.sleep(0.05)
        assert frame is not None, "No webcam frame arrived"
        result = pipeline.infer(frame.image)
        assert result.width == frame.image.shape[1]
        assert result.height == frame.image.shape[0]
        assert result.device.startswith("cuda")
