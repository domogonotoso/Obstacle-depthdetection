"""Exercise the real pipeline/fusion path with fake models, never model weights."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import VisionConfig
from app.types import Detection, InferenceResult

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pipeline_module = pytest.importorskip("app.pipeline")


class FakeDetector:
    def __init__(self, config):
        self.config = config
        self.weights_path = config.model_dir / "fake-yolo.pt"
        self.model = SimpleNamespace(model=torch.nn.Linear(1, 1))
        self.frames = []
        self.outputs = [Detection((0, 0, 4, 4), "person", 0.9, 0)]

    def infer(self, frame):
        assert torch.is_inference_mode_enabled()
        self.frames.append(frame)
        return self.outputs


class FakeDepth:
    def __init__(self, config):
        self.config = config
        self.weights_path = config.model_dir / "fake-metric.pth"
        self.model = torch.nn.Linear(1, 1)
        self.frames = []
        self.output = None

    def infer(self, frame):
        assert torch.is_inference_mode_enabled()
        self.frames.append(frame)
        if self.output is not None:
            return self.output
        return np.full(frame.shape[:2], 2.0, dtype=np.float32)


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_module, "ObjectDetector", FakeDetector)
    monkeypatch.setattr(pipeline_module, "MetricDepthEstimator", FakeDepth)
    monkeypatch.setattr(pipeline_module, "resolve_device", lambda requested: torch.device("cpu"))
    monkeypatch.setattr(pipeline_module, "sha256_file", lambda path: "f" * 64)
    return pipeline_module.VisionPipeline(VisionConfig(device="cpu", model_dir=tmp_path))


def test_pipeline_uses_same_unmodified_frame_and_original_coordinates(pipeline):
    frame = np.arange(4 * 12 * 3, dtype=np.uint8).reshape(4, 12, 3)
    original = frame.copy()
    result = pipeline.infer(frame)
    assert pipeline.detector.frames[0] is frame
    assert pipeline.depth_estimator.frames[0] is frame
    assert np.array_equal(frame, original)
    assert result.width == 12 and result.height == 4
    assert result.objects[0].bbox == (0, 0, 4, 4)
    assert result.objects[0].direction == "LEFT"
    assert result.objects[0].distance_m == 2
    assert result.objects[0].confidence == 0.9
    assert result.device == "cpu"
    assert all(value >= 0 for value in result.timings_ms.values())
    assert InferenceResult.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False))) == result


@pytest.mark.parametrize(
    "invalid_frame",
    [
        None, [[1]], np.zeros((4, 8), dtype=np.uint8),
        np.zeros((4, 8, 4), dtype=np.uint8), np.zeros((4, 8, 3), dtype=np.float32),
        np.zeros((0, 8, 3), dtype=np.uint8), np.zeros((4, 0, 3), dtype=np.uint8),
    ],
)
def test_invalid_input_is_rejected_before_any_model_call(pipeline, invalid_frame):
    with pytest.raises(ValueError, match="uint8 BGR"):
        pipeline.infer(invalid_frame)
    assert not pipeline.detector.frames
    assert not pipeline.depth_estimator.frames


def test_misaligned_depth_is_rejected_instead_of_silently_resized(pipeline):
    pipeline.depth_estimator.output = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="not aligned"):
        pipeline.infer(np.zeros((4, 8, 3), dtype=np.uint8))


def test_empty_detections_remain_valid_serializable_result(pipeline):
    pipeline.detector.outputs = []
    result = pipeline.infer(np.zeros((4, 8, 3), dtype=np.uint8))
    assert result.objects == []
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False))["objects"] == []


def test_invalid_depth_preserves_object_without_nonfinite_json(pipeline):
    pipeline.depth_estimator.output = np.full((4, 8), np.nan, dtype=np.float32)
    result = pipeline.infer(np.zeros((4, 8, 3), dtype=np.uint8))
    assert result.objects[0].distance_m is None
    assert result.objects[0].depth_samples == 0
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False))["objects"][0]["distance_m"] is None


def test_gpu_timing_synchronizes_without_requiring_a_real_gpu(pipeline, monkeypatch):
    pipeline.device = torch.device("cuda:0")
    synchronizations = []
    monkeypatch.setattr(torch.cuda, "synchronize", synchronizations.append)
    result = pipeline.infer(np.zeros((4, 8, 3), dtype=np.uint8))
    assert synchronizations == [pipeline.device] * 3
    assert result.device == "cuda:0"


def test_diagnostics_identifies_both_model_devices_and_checkpoint_hashes(pipeline, monkeypatch):
    monkeypatch.setattr(pipeline_module, "gpu_diagnostics", lambda device: {"device": str(device)})
    diagnostics = pipeline.diagnostics()
    assert diagnostics["detector_device"] == diagnostics["depth_device"] == "cpu"
    assert diagnostics["depth_unit"] == "meters"
    assert diagnostics["depth_model_max_m"] == 20
    assert diagnostics["weights_sha256"] == {"yolo11n": "f" * 64, "metric_hypersim_small": "f" * 64}
