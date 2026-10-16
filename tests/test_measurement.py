import csv
from datetime import datetime

import pytest

from app.measurement import CSV_FIELDS, MeasurementRecorder
from app.types import InferenceResult, ObjectEstimate


def result(distance=2.5):
    obj = ObjectEstimate((1, 2, 3, 4), "chair", 0.8, 56, "CENTER", distance, 0.9, 20)
    return InferenceResult(640, 480, [obj], {"total": 10}, "cuda:0")


def test_records_ground_truth_comparison_and_appends_one_header(tmp_path):
    output = tmp_path / "measurements" / "distance.csv"
    recorder = MeasurementRecorder(output, truth_m=2)
    row = recorder.record(result())
    assert row["signed_error_m"] == 0.5
    assert row["absolute_error_m"] == 0.5
    assert row["relative_error_pct"] == 25
    assert datetime.fromisoformat(row["timestamp_utc"]).utcoffset().total_seconds() == 0
    recorder.record(result(1.5))
    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(CSV_FIELDS)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[1]["signed_error_m"] == "-0.5"
    assert rows[1]["absolute_error_m"] == "0.5"
    assert rows[0]["label"] == "chair"


@pytest.mark.parametrize("distance", [None, 0, -1, float("nan"), float("inf")])
def test_invalid_depth_is_logged_without_fabricating_an_error(tmp_path, distance):
    row = MeasurementRecorder(tmp_path / "invalid.csv", 1).record(result(distance))
    assert row["status"] == "invalid_depth"
    assert row["estimated_distance_m"] == ""
    assert row["absolute_error_m"] == ""


@pytest.mark.parametrize("truth", [0, -1, float("nan"), float("inf")])
def test_truth_must_be_a_positive_finite_measurement(tmp_path, truth):
    with pytest.raises(ValueError, match="Ground-truth"):
        MeasurementRecorder(tmp_path / "bad.csv", truth)


def test_no_objects_does_not_create_a_measurement(tmp_path):
    path = tmp_path / "empty.csv"
    recorder = MeasurementRecorder(path, 1)
    assert recorder.record(InferenceResult(640, 480, [], {}, "cpu")) is None
    assert not path.exists()


@pytest.mark.parametrize("index", [-1, 1])
def test_invalid_object_selection_is_explicit(tmp_path, index):
    with pytest.raises(IndexError):
        MeasurementRecorder(tmp_path / "index.csv", 1).record(result(), index)


def test_refuses_to_append_to_incompatible_csv(tmp_path):
    path = tmp_path / "existing.csv"
    path.write_text("some,other,header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible header"):
        MeasurementRecorder(path, 1).record(result())
