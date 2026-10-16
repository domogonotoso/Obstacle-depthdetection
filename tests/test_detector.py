"""YOLO class filtering is verified against the loaded model without downloads."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from app.config import VisionConfig
import app.detector as detector_module


TARGET_CLASSES = (
    (0, "person"), (1, "bicycle"), (13, "bench"), (16, "dog"),
    (24, "backpack"), (28, "suitcase"), (56, "chair"), (57, "couch"),
    (58, "potted plant"), (60, "dining table"),
)


@pytest.fixture
def model_factory(monkeypatch, tmp_path):
    """Replace only hardware/network boundaries; exercise real adapter logic."""
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "ultralytics"))
    monkeypatch.setattr(detector_module, "resolve_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(detector_module, "ensure_asset", lambda *args: tmp_path / "yolo11n.pt")

    def build(names=None, rows=()):
        names = dict(TARGET_CLASSES) if names is None else names
        data = torch.tensor(rows, dtype=torch.float32).reshape(-1, 6)
        model = SimpleNamespace(
            names=names, to=Mock(),
            predict=Mock(return_value=[SimpleNamespace(
                boxes=SimpleNamespace(data=data), names=names,
            )]),
        )
        constructor = Mock(return_value=model)
        monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=constructor))
        return model, constructor

    return build


def test_all_ten_configured_classes_are_passed_to_official_predict(model_factory, tmp_path):
    model, constructor = model_factory(rows=[(10, 20, 30, 40, 0.9, 56)])
    config = VisionConfig(device="cpu", model_dir=tmp_path, target_classes=TARGET_CLASSES)
    detector = detector_module.ObjectDetector(config)
    frame = np.zeros((64, 96, 3), dtype=np.uint8)

    detections = detector.infer(frame)

    constructor.assert_called_once_with(str(tmp_path / "yolo11n.pt"), task="detect")
    options = model.predict.call_args.kwargs
    assert options["source"] is frame
    assert options["classes"] == [0, 1, 13, 16, 24, 28, 56, 57, 58, 60]
    assert len(detections) == 1
    assert detections[0].class_id == 56
    assert detections[0].label == "chair"
    assert detections[0].bbox == (10, 20, 30, 40)


@pytest.mark.parametrize("bad_names", [
    {class_id: ("car" if class_id == 56 else name) for class_id, name in TARGET_CLASSES},
    {class_id: name for class_id, name in TARGET_CLASSES if class_id != 60},
])
def test_class_id_and_name_mismatch_fails_before_inference(model_factory, tmp_path, bad_names):
    model, _ = model_factory(names=bad_names)
    config = VisionConfig(device="cpu", model_dir=tmp_path, target_classes=TARGET_CLASSES)

    with pytest.raises(ValueError, match=r"(?i)(class|names|chair|dining table)"):
        detector_module.ObjectDetector(config)

    model.predict.assert_not_called()


def test_configuration_can_select_other_model_classes(model_factory, tmp_path):
    model, _ = model_factory(names={2: "car"}, rows=[(1, 2, 3, 4, 0.8, 2)])
    config = VisionConfig(device="cpu", model_dir=tmp_path, target_classes=((2, "car"),))
    detector = detector_module.ObjectDetector(config)

    detections = detector.infer(np.zeros((32, 32, 3), dtype=np.uint8))

    assert model.predict.call_args.kwargs["classes"] == [2]
    assert [item.class_id for item in detections] == [2]


def test_empty_filtered_results_remain_a_valid_empty_detection_list(model_factory, tmp_path):
    model_factory()
    detector = detector_module.ObjectDetector(VisionConfig(
        device="cpu", model_dir=tmp_path, target_classes=TARGET_CLASSES,
    ))

    assert detector.infer(np.zeros((32, 32, 3), dtype=np.uint8)) == []
