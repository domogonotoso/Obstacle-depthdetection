"""Synthetic arrays exercise coordinate and invalid-value behavior, not accuracy."""

import json

import numpy as np
import pytest

from app.fusion import classify_direction, fuse_detections
from app.types import Detection, InferenceResult


def detection(bbox=(0, 0, 8, 8)):
    return Detection(bbox=bbox, label="person", confidence=0.9, class_id=0)


@pytest.mark.parametrize(
    ("center", "expected"),
    [(0, "LEFT"), (99, "LEFT"), (100, "CENTER"), (199, "CENTER"), (200, "RIGHT"), (300, "RIGHT")],
)
def test_direction_thirds(center, expected):
    assert classify_direction(center, 300) == expected


def test_center_crop_excludes_background_and_uses_robust_median():
    depth = np.full((8, 8), 10.0, dtype=np.float32)
    depth[2:6, 2:6] = 2.0
    depth[2, 2] = 15.0
    obj = fuse_detections([detection()], depth, (8, 8, 3))[0]
    assert obj.distance_m == 2.0
    assert obj.valid_fraction == 1
    assert obj.depth_samples == 16


def test_invalid_depth_is_excluded_and_threshold_hides_sparse_estimates():
    depth = np.array([[np.nan, np.inf, 0, -1], [21, 0.01, 2, 4]], dtype=float)
    kwargs = dict(central_ratio=1, max_depth_m=20)
    obj = fuse_detections([detection((0, 0, 4, 2))], depth, (2, 4), **kwargs)[0]
    assert obj.distance_m == 3
    assert obj.depth_samples == 2
    assert obj.valid_fraction == 0.25
    sparse = fuse_detections(
        [detection((0, 0, 4, 2))], depth, (2, 4), min_valid_fraction=0.3, **kwargs
    )[0]
    assert sparse.distance_m is None
    assert sparse.depth_samples == 2


def test_all_invalid_depth_still_preserves_detection():
    obj = fuse_detections([detection()], np.zeros((8, 8)), (8, 8))[0]
    assert obj.distance_m is None
    assert obj.valid_fraction == 0
    assert obj.depth_samples == 0


def test_depth_range_boundaries_are_accepted():
    obj = fuse_detections(
        [detection((0, 0, 2, 1))], np.array([[0.05, 20.0]]), (1, 2), central_ratio=1
    )[0]
    assert obj.depth_samples == 2
    assert obj.distance_m == pytest.approx(10.025)


def test_clips_box_and_rejects_malformed_boxes():
    boxes = [(-5, -5, 5, 5), (4, 0, 2, 4), (9, 1, 12, 3), (np.nan, 0, 2, 2), (0, 0, 0, 3)]
    objects = fuse_detections([detection(box) for box in boxes], np.ones((8, 8)), (8, 8))
    assert len(objects) == 1
    assert objects[0].bbox == (0, 0, 5, 5)
    assert objects[0].direction == "LEFT"


def test_tiny_fractional_box_has_at_least_one_sample():
    obj = fuse_detections([detection((1.2, 1.3, 1.4, 1.6))], np.full((8, 8), 3), (8, 8))[0]
    assert obj.distance_m == 3
    assert obj.depth_samples == 1


def test_empty_detection_list_and_alignment_validation():
    assert fuse_detections([], np.ones((8, 8)), (8, 8)) == []
    with pytest.raises(ValueError, match="not aligned"):
        fuse_detections([], np.ones((4, 4)), (8, 8))


@pytest.mark.parametrize("options", [{"central_ratio": 0}, {"central_ratio": 2}, {"min_valid_fraction": 2}, {"max_depth_m": 0}])
def test_rejects_invalid_configuration(options):
    with pytest.raises(ValueError):
        fuse_detections([], np.ones((8, 8)), (8, 8), **options)


def test_inference_result_roundtrips_through_json_including_missing_depth():
    objects = fuse_detections([detection()], np.zeros((8, 8)), (8, 8))
    result = InferenceResult(width=8, height=8, objects=objects, timings_ms={"total": 12.3}, device="cuda:0")
    assert InferenceResult.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False))) == result


def test_remote_result_rejects_nonfinite_distance():
    objects = fuse_detections([detection()], np.ones((8, 8)), (8, 8))
    data = InferenceResult(8, 8, objects, {}, "cpu").to_dict()
    data["objects"][0]["distance_m"] = float("nan")
    with pytest.raises(ValueError, match="distance_m"):
        InferenceResult.from_dict(data)
