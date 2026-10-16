"""Virtual frames/timestamps verify alert policy without a camera, GPU or speaker."""

from dataclasses import replace

import pytest

from app.alerts import AlertController, is_central, valid_distance, visible_objects
from app.settings import AlertSettings
from app.types import InferenceResult, ObjectEstimate


def obj(distance=1.5, *, bbox=(110, 10, 180, 100), direction="CENTER", class_id=0):
    return ObjectEstimate(
        bbox, "person", 0.9, class_id, direction, distance, 1.0, 100
    )


def frame(*objects):
    return InferenceResult(300, 200, list(objects), {"total": 20.0}, "cuda:0")


def controller(**options):
    return AlertController(replace(AlertSettings(), **options))


def confirmed(control, distance, start=0.0):
    first = control.update(frame(obj(distance)), now=start)
    assert first.sound_event is None
    return control.update(frame(obj(distance)), now=start + 0.1)


@pytest.mark.parametrize("distance,expected", [(2.01, "Normal"), (2, "Warning"), (1.01, "Warning"), (1, "Danger"), (0.4, "Danger")])
def test_exact_thresholds_and_two_fresh_frame_confirmation(distance, expected):
    control = controller()
    result = confirmed(control, distance)
    assert result.level == expected
    assert result.nearest_distance_m == distance
    assert result.sound_event == (None if expected == "Normal" else expected)


def test_display_filter_preserves_original_indices_and_three_meter_boundary():
    data = frame(obj(3.001), obj(3), obj(0), obj(None), obj(float("nan")), obj(1.5))
    assert [(i, o.distance_m) for i, o in visible_objects(data, 3)] == [(1, 3), (5, 1.5)]


@pytest.mark.parametrize("distance", [None, 0, -1, float("nan"), float("inf"), -float("inf")])
def test_invalid_values_never_reuse_previous_valid_depth(distance):
    control = controller()
    confirmed(control, 0.5)
    result = control.update(frame(obj(distance)), now=0.2)
    assert result.level == "Unknown"
    assert result.sound_event is None
    assert result.nearest_distance_m is None
    assert result.result.objects[0].distance_m is None
    assert result.invalid_depth_count == 1
    assert not valid_distance(distance)
    # A new valid value starts a new median rather than keeping old near samples.
    result = control.update(frame(obj(4)), now=0.3)
    assert result.result.objects[0].distance_m == 4


def test_left_and_right_objects_do_not_warn_but_center_overlap_does():
    control = controller(smoothing_window=1)
    side = frame(
        obj(0.3, bbox=(0, 0, 70, 100), direction="LEFT"),
        obj(0.5, bbox=(220, 0, 300, 100), direction="RIGHT"),
    )
    control.update(side, now=0)
    assert control.update(side, now=0.1).level == "Normal"
    overlap = obj(1.4, bbox=(20, 0, 160, 100), direction="LEFT")
    assert is_central(overlap, 300, 0.35)
    control.update(frame(overlap), now=0.2)
    result = control.update(frame(overlap), now=0.3)
    assert result.level == "Warning"
    assert result.sound_event == "Warning"


def test_insufficient_or_invalid_overlap_is_not_central():
    assert not is_central(obj(bbox=(0, 0, 110, 10), direction="LEFT"), 300, 0.35)
    assert not is_central(obj(bbox=(100, 0, 100, 10), direction="LEFT"), 300, 0.35)
    assert not is_central(obj(bbox=(float("nan"), 0, 100, 10), direction="LEFT"), 300, 0.35)
    assert not is_central(obj(direction="LEFT"), 0, 0.35)


def test_nearest_central_object_takes_priority():
    control = controller()
    data = frame(obj(1.7), obj(0.8, bbox=(150, 20, 190, 90)), obj(0.1, bbox=(0, 0, 30, 30), direction="LEFT"))
    control.update(data, now=0)
    result = control.update(data, now=0.1)
    assert result.level == "Danger"
    assert result.nearest_distance_m == 0.8


def test_distance_histories_follow_same_class_boxes_when_detection_order_changes():
    control = controller()
    near = obj(0.8, bbox=(101, 10, 140, 90))
    far = obj(2.8, bbox=(160, 10, 199, 90))
    control.update(frame(near, far), now=0)
    result = control.update(frame(replace(far, distance_m=2.6), replace(near, distance_m=1)), now=0.1)
    assert [o.distance_m for o in result.result.objects] == pytest.approx([2.7, 0.9])
    assert [o.bbox for o in result.result.objects] == [far.bbox, near.bbox]


def test_new_class_and_new_location_never_inherit_old_distance_history():
    control = controller()
    control.update(frame(obj(0.5)), now=0)
    new_class = control.update(frame(obj(3.5, class_id=56)), now=0.1)
    assert new_class.result.objects[0].distance_m == 3.5
    new_place = control.update(frame(obj(4, bbox=(5, 5, 40, 50), direction="LEFT")), now=0.2)
    assert new_place.result.objects[0].distance_m == 4


def test_one_prior_track_is_not_shared_by_two_new_boxes():
    control = controller()
    control.update(frame(obj(0.4)), now=0)
    result = control.update(frame(obj(0.6), obj(4)), now=0.1)
    # Exactly one current box can inherit the previous sample.
    values = [o.distance_m for o in result.result.objects]
    assert sum(distance in {0.5, 2.2} for distance in values) == 1


def test_lost_tracks_and_resolution_change_reset_median_history():
    control = controller(result_timeout_s=10)
    control.update(frame(obj(0.5)), now=0)
    result = control.update(frame(obj(4)), now=1.1)
    assert result.result.objects[0].distance_m == 4
    resized = replace(frame(obj(2)), width=600, height=400)
    result = control.update(resized, now=1.2)
    assert result.result.objects[0].distance_m == 2


def test_moving_median_rejects_an_isolated_near_outlier():
    control = controller()
    for index, distance in enumerate([2.5, 2.5, 2.5, 0.3, 2.5]):
        result = control.update(frame(obj(distance)), now=index / 10)
    assert result.result.objects[0].distance_m == 2.5
    assert result.level == "Normal"
    assert result.sound_event is None


def test_warning_repeat_interval_under_continuous_new_results():
    control = controller(smoothing_window=1)
    events = []
    for index in range(63):
        result = control.update(frame(obj(1.5)), now=index / 10)
        if result.sound_event:
            events.append(index / 10)
    assert events == pytest.approx([0.1, 3.1, 6.1])


def test_danger_repeat_interval_and_tick_does_not_accumulate_events():
    control = controller(result_timeout_s=20)
    assert confirmed(control, 0.8).sound_event == "Danger"
    assert control.tick(now=1.59).sound_event is None
    assert control.tick(now=1.6).sound_event == "Danger"
    # A long polling gap produces one current event, never a backlog.
    assert control.tick(now=8).sound_event == "Danger"
    assert control.tick(now=8).sound_event is None


def test_warning_to_danger_bypasses_cooldown_after_confirmation():
    control = controller(smoothing_window=1)
    assert confirmed(control, 1.5).sound_event == "Warning"
    first = control.update(frame(obj(0.8)), now=0.2)
    assert first.level == "Warning"
    assert first.sound_event is None
    danger = control.update(frame(obj(0.8)), now=0.3)
    assert danger.level == "Danger"
    assert danger.sound_event == "Danger"


def test_danger_demotion_does_not_bypass_warning_cooldown():
    control = controller(smoothing_window=1)
    confirmed(control, 0.8)
    assert control.update(frame(obj(1.5)), now=0.2).sound_event is None
    result = control.update(frame(obj(1.5)), now=0.3)
    assert result.level == "Warning"
    assert result.sound_event is None


def test_normal_reentry_does_not_bypass_sound_cooldown():
    control = controller(smoothing_window=1)
    confirmed(control, 1.5)
    control.update(frame(), now=0.2)
    assert control.update(frame(), now=0.3).level == "Normal"
    control.update(frame(obj(1.5)), now=0.4)
    result = control.update(frame(obj(1.5)), now=0.5)
    assert result.level == "Warning"
    assert result.sound_event is None


def test_invalid_reentry_preserves_cooldown_and_requires_fresh_confirmation():
    control = controller(smoothing_window=1)
    confirmed(control, 0.8)
    control.invalidate(now=0.2)
    assert control.update(frame(obj(0.8)), now=0.3).level == "Unknown"
    result = control.update(frame(obj(0.8)), now=0.4)
    assert result.level == "Danger"
    assert result.sound_event is None


def test_pending_demotion_never_repeats_old_risk_for_a_now_far_object():
    control = controller(smoothing_window=1, result_timeout_s=10)
    confirmed(control, 0.8)
    result = control.update(frame(obj(3)), now=2)
    assert result.level == "Danger"
    assert result.sound_event is None


def test_hysteresis_keeps_level_near_both_thresholds_until_clear_exit():
    control = controller(smoothing_window=1)
    confirmed(control, 0.9)
    for index, distance in enumerate([1.01, 0.99, 1.1, 1.14], start=2):
        assert control.update(frame(obj(distance)), now=index / 10).level == "Danger"
    control.update(frame(obj(1.2)), now=0.6)
    assert control.update(frame(obj(1.2)), now=0.7).level == "Warning"
    for index, distance in enumerate([2.01, 1.99, 2.1, 2.14], start=8):
        assert control.update(frame(obj(distance)), now=index / 10).level == "Warning"
    control.update(frame(obj(2.2)), now=1.2)
    assert control.update(frame(obj(2.2)), now=1.3).level == "Normal"


def test_alternating_single_frames_do_not_change_confirmed_state():
    control = controller(smoothing_window=1)
    confirmed(control, 2.5)
    for index, distance in enumerate([1.5, 2.5, 1.5, 2.5, 1.5, 2.5], start=2):
        result = control.update(frame(obj(distance)), now=index / 10)
        assert result.level == "Normal"
        assert result.sound_event is None


def test_fresh_no_objects_is_normal_and_silent_even_while_demotion_pending():
    control = controller(result_timeout_s=10)
    confirmed(control, 0.8)
    assert control.update(frame(), now=2).sound_event is None
    result = control.update(frame(), now=2.1)
    assert result.level == "Normal"
    assert result.result.objects == []


def test_stale_result_stops_alerts_and_resets_smoothing_and_confirmation():
    control = controller()
    confirmed(control, 0.8)
    assert control.tick(now=1.099).level == "Danger"
    stale = control.tick(now=1.1)
    assert stale.level == "Unknown"
    assert stale.result is None
    assert stale.nearest_distance_m is None
    assert stale.sound_event is None
    fresh = control.update(frame(obj(3)), now=1.2)
    assert fresh.result.objects[0].distance_m == 3
    assert fresh.level == "Unknown"


def test_failure_invalidation_stops_sound_and_does_not_fake_safe_distance():
    control = controller()
    confirmed(control, 0.8)
    failed = control.invalidate(now=0.2)
    assert failed.level == "Unknown"
    assert failed.result is None
    assert control.tick(now=4).sound_event is None
    assert control.update(frame(obj(4)), now=4.1).result.objects[0].distance_m == 4


def test_mute_does_not_disable_risk_state_and_unmute_has_no_queued_events():
    control = controller(sound_enabled=False, result_timeout_s=20)
    result = confirmed(control, 0.8)
    assert result.level == "Danger"
    assert not result.sound_enabled
    assert result.sound_event is None
    assert control.toggle_sound() is True
    assert control.tick(now=0.2).sound_event == "Danger"
    assert control.toggle_sound() is False
    assert control.tick(now=4).sound_event is None
    assert control.toggle_sound() is True
    assert control.tick(now=4.1).sound_event == "Danger"
    assert control.tick(now=4.1).sound_event is None


def test_tick_never_counts_as_a_confirmation_frame():
    control = controller()
    control.update(frame(obj(0.5)), now=0)
    for instant in [0.1, 0.2, 0.3]:
        result = control.tick(now=instant)
        assert result.level == "Unknown"
        assert result.sound_event is None
