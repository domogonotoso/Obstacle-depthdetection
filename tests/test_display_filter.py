"""Screen annotations respect the distance boundary and original object indices."""
import numpy as np

from app.types import InferenceResult, ObjectEstimate
import app.visualization as visualization


def obj(distance, label="person"):
    return ObjectEstimate((30, 120, 100, 220), label, 0.9, 0, "LEFT", distance, 1.0, 100)


def test_only_valid_distances_up_to_inclusive_limit_draw_boxes(monkeypatch):
    boxes = []
    text = []
    monkeypatch.setattr(visualization.cv2, "rectangle", lambda image, a, b, color, thickness: boxes.append((a, b, thickness)))
    monkeypatch.setattr(visualization, "_text", lambda image, label, *args: text.append(label))
    objects = [obj(3.1, "far"), obj(3.0, "boundary"), obj(None, "missing"),
               obj(float("nan"), "nan"), obj(float("inf"), "infinite"), obj(0, "zero")]
    result = InferenceResult(640, 480, objects, {}, "cpu")
    visualization.render_frame(np.zeros((480, 640, 3), np.uint8), result, show_zones=False)
    assert sum(thickness == 2 for _, _, thickness in boxes) == 1
    assert any(label.startswith("1 Boundary") for label in text)
    assert not any(label.startswith("0 Far") for label in text)
    assert any("Est. 3.00m" in label for label in text)


def test_header_exposes_effective_sound_state_level_and_invalid_count(monkeypatch):
    text = []
    monkeypatch.setattr(visualization, "_text", lambda image, label, *args: text.append(label))
    visualization.render_frame(np.zeros((480, 640, 3), np.uint8), sound_enabled=True,
                               alert_level="Danger", nearest_distance_m=0.8, invalid_depth_count=2)
    assert "Sound ON | Alert Danger | Est. 0.80m" in text
    assert any("invalid depth 2" in line for line in text)
