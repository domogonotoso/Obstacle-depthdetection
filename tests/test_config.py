"""Configuration rejects settings that cannot reach the selected model backends."""

from pathlib import Path

import pytest

from app.config import VisionConfig


def test_defaults_select_metric_indoor_range_and_convert_model_path():
    config = VisionConfig(model_dir="test-models")
    assert config.device == "cuda:0"
    assert config.model_dir == Path("test-models")
    assert config.yolo_size % 32 == 0
    assert config.depth_size % 14 == 0
    assert config.max_depth_m == 20


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "cuda:2"])
def test_recognized_device_syntax_does_not_probe_hardware(device):
    assert VisionConfig(device=device).device == device


@pytest.mark.parametrize("device", ["mps", "gpu", "cudabogus", "cuda:", "cuda:-1", "cuda:1:2"])
def test_rejects_malformed_or_unsupported_device_syntax(device):
    with pytest.raises(ValueError, match="device"):
        VisionConfig(device=device)


@pytest.mark.parametrize(
    "options",
    [
        {"yolo_size": 31}, {"yolo_size": 641}, {"yolo_size": 2080},
        {"yolo_size": 640.0}, {"depth_size": 42}, {"depth_size": 519},
        {"depth_size": 1050}, {"depth_size": 518.0},
    ],
)
def test_rejects_invalid_model_input_dimensions(options):
    with pytest.raises(ValueError):
        VisionConfig(**options)


@pytest.mark.parametrize("field", ["confidence", "central_ratio", "min_valid_fraction"])
@pytest.mark.parametrize("value", [0, -0.1, 1.01, float("nan"), float("inf")])
def test_fraction_settings_must_be_finite_positive_probabilities(field, value):
    with pytest.raises(ValueError, match=field):
        VisionConfig(**{field: value})


@pytest.mark.parametrize(
    "options",
    [
        {"min_depth_m": 0}, {"min_depth_m": -1},
        {"min_depth_m": 20}, {"max_depth_m": 21},
        {"min_depth_m": float("nan")}, {"max_depth_m": float("inf")},
    ],
)
def test_depth_validity_range_is_within_metric_hypersim_model_range(options):
    with pytest.raises(ValueError, match="depth"):
        VisionConfig(**options)


def test_from_env_reads_all_public_inference_settings(monkeypatch):
    values = {
        "DEVICE": "cpu", "MODEL_DIR": "my-model-cache", "YOLO_SIZE": "320",
        "DEPTH_SIZE": "392", "CONFIDENCE": "0.5", "CENTRAL_RATIO": "0.75",
        "MIN_VALID_FRACTION": "0.6", "MIN_DEPTH_M": "0.1", "MAX_DEPTH_M": "10",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    config = VisionConfig.from_env()
    assert config == VisionConfig(
        device="cpu", model_dir=Path("my-model-cache"), yolo_size=320, depth_size=392,
        confidence=0.5, central_ratio=0.75, min_valid_fraction=0.6,
        min_depth_m=0.1, max_depth_m=10,
    )


def test_invalid_environment_setting_fails_before_loading_models(monkeypatch):
    monkeypatch.setenv("DEPTH_SIZE", "519")
    with pytest.raises(ValueError, match="depth_size"):
        VisionConfig.from_env()
