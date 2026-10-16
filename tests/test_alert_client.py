"""Client orchestration with virtual time, fake transport and no camera/speaker."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from app.camera import CapturedFrame
from app.main import CompletedInference
from app.settings import load_settings
from app.sound import SoundPlayer
from app.types import InferenceResult, ObjectEstimate
import app.main as main_module


def estimate(distance=1.5, *, class_id=56, direction="CENTER"):
    obj = ObjectEstimate((55, 30, 105, 110), "chair", 0.9, class_id, direction, distance, 1.0, 100)
    return InferenceResult(160, 120, [obj], {"total": 2.0}, "cuda:0")


class SoundSpy:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.events = []
        self.toggles = []
        self.cancel_count = 0
        self.closed = False

    def play(self, level):
        self.events.append(level)
        return self.enabled

    def cancel(self):
        self.cancel_count += 1

    def set_enabled(self, enabled):
        self.toggles.append(enabled)
        self.enabled = bool(enabled)

    def close(self):
        self.closed = True
        return True


def install_client(monkeypatch, outcomes, *, keys=None, stop_when=None, sound_type=SoundSpy,
                   capture_age=0.0):
    """Replace external boundaries; preserve main's actual alert controller and GUI loop."""
    harness = SimpleNamespace(now=100.0, renders=[], sound=None, windows_closed=False)
    settings = load_settings()
    settings = replace(settings, alerts=replace(settings.alerts, confirm_frames=1, smoothing_window=1))
    monkeypatch.setattr(main_module, "load_settings", lambda path: settings)

    def advance(_seconds):
        harness.now += 0.1

    monkeypatch.setattr(main_module, "time", SimpleNamespace(monotonic=lambda: harness.now, sleep=advance))

    class Camera:
        error = None
        closed = False
        sequence = 0

        def latest(self, after_id=-1):
            self.sequence += 1
            return CapturedFrame(np.zeros((120, 160, 3), np.uint8), self.sequence,
                                 harness.now - capture_age)

        def close(self):
            self.closed = True

    class Backend:
        attempts = 0
        closed = False

        def infer(self, image):
            outcome = outcomes[min(self.attempts, len(outcomes) - 1)]
            self.attempts += 1
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        def close(self):
            self.closed = True

    class InlineWorker:
        # InferenceWorker's thread/frame contract has separate event-based tests.
        # This deterministic boundary makes retries and key events independent of
        # scheduler timing while exercising the actual main-loop decisions.
        def __init__(self, backend):
            self.backend = backend
            self.completed = None

        def submit(self, frame):
            assert self.completed is None
            try:
                result, error = self.backend.infer(frame.image), None
            except Exception as exc:
                result, error = None, f"{type(exc).__name__}: {exc}"
            self.completed = CompletedInference(frame, result, 2.0, harness.now, error)
            return True

        def take_completed(self):
            completed, self.completed = self.completed, None
            return completed

        def close(self, timeout=1.0):
            self.backend.close()
            return True

    def create_sound(enabled):
        harness.sound = sound_type(enabled)
        return harness.sound

    def render(frame, result=None, **kwargs):
        harness.renders.append({"result": result, **kwargs})
        return frame

    key_iterator = iter(keys) if keys is not None else None

    def wait_key(_delay):
        if key_iterator is not None:
            return next(key_iterator, ord("q"))
        return ord("q") if stop_when and stop_when(harness) else -1

    harness.camera, harness.backend = Camera(), Backend()
    monkeypatch.setattr(main_module, "LatestFrameCamera", lambda *args: harness.camera)
    monkeypatch.setattr(main_module, "_build_backend", lambda args: harness.backend)
    monkeypatch.setattr(main_module, "InferenceWorker", InlineWorker)
    monkeypatch.setattr(main_module, "SoundPlayer", create_sound)
    monkeypatch.setattr(main_module, "render_frame", render)
    monkeypatch.setattr(main_module.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(main_module.cv2, "waitKey", wait_key)
    monkeypatch.setattr(main_module.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(main_module.cv2, "destroyAllWindows",
                        lambda: setattr(harness, "windows_closed", True))
    return harness


def assert_released(harness):
    assert harness.camera.closed
    assert harness.backend.closed
    assert harness.sound.closed
    assert harness.windows_closed


def test_m_key_calls_real_sound_setter_and_display_uses_effective_state(monkeypatch):
    class RealSoundSpy(SoundPlayer):
        def __init__(self, enabled):
            self.toggles = []
            self.closed = False
            super().__init__(enabled=enabled, player=lambda level: None)

        def set_enabled(self, enabled):
            self.toggles.append(enabled)
            super().set_enabled(enabled)

        def close(self, timeout=1.0):
            self.closed = True
            return super().close(timeout)

    harness = install_client(monkeypatch, [estimate(4.0)],
                             keys=[ord("m"), -1, ord("M"), -1, ord("q")], sound_type=RealSoundSpy)
    assert main_module.main(["--mode", "remote", "--max-seconds", "5"]) == 0
    assert harness.sound.toggles == [False, True]
    assert [row["sound_enabled"] for row in harness.renders] == [True, False, False, True, True]
    assert_released(harness)


def test_mute_argument_starts_muted_and_m_turns_it_on(monkeypatch):
    harness = install_client(monkeypatch, [estimate()], keys=[ord("M"), -1, ord("q")])
    assert main_module.main(["--mode", "remote", "--mute", "--max-seconds", "5"]) == 0
    assert harness.renders[0]["sound_enabled"] is False
    assert harness.renders[1]["sound_enabled"] is True
    assert harness.sound.toggles == [True]
    assert_released(harness)


def test_default_failure_limit_retries_seven_failures_and_recovers(monkeypatch, capsys):
    harness = install_client(
        monkeypatch, [RuntimeError("transport timeout")] * 7 + [estimate()],
        stop_when=lambda state: any(row["alert_level"] == "Warning" for row in state.renders),
    )
    assert main_module.build_parser().parse_args([]).failure_limit == 0
    assert main_module.main(["--mode", "remote", "--max-seconds", "30"]) == 0
    assert harness.backend.attempts >= 8
    assert capsys.readouterr().err.count("transport timeout") == 7
    assert any(row["result"] is not None and row["alert_level"] == "Warning" for row in harness.renders)
    assert harness.sound.events == ["Warning"]
    assert_released(harness)


def test_explicit_failure_limit_still_stops_cleanly(monkeypatch, capsys):
    harness = install_client(monkeypatch, [RuntimeError("offline")])
    assert main_module.main(["--mode", "remote", "--failure-limit", "2", "--max-seconds", "10"]) == 3
    assert harness.backend.attempts == 2
    assert capsys.readouterr().err.count("offline") == 2
    assert not harness.sound.events
    assert_released(harness)


@pytest.mark.parametrize("invalid_depth", [None, 0.0, float("nan"), float("inf")])
def test_transport_failure_and_invalid_depth_clear_alerts_without_reusing_old_distance(
    monkeypatch, capsys, invalid_depth
):
    harness = install_client(
        monkeypatch, [estimate(0.8), RuntimeError("connection lost"), estimate(invalid_depth)],
        stop_when=lambda state: any(row.get("invalid_depth_count") == 1 for row in state.renders),
    )
    assert main_module.main(["--mode", "remote", "--max-seconds", "10"]) == 0
    assert "connection lost" in capsys.readouterr().err
    assert harness.sound.events == ["Danger"]
    danger_index = next(index for index, row in enumerate(harness.renders) if row["alert_level"] == "Danger")
    after_failure = harness.renders[danger_index + 1:]
    assert after_failure
    assert all(row["alert_level"] == "Unknown" for row in after_failure)
    invalid = after_failure[-1]
    assert invalid["nearest_distance_m"] is None
    assert invalid["result"].objects[0].distance_m is None
    assert harness.sound.cancel_count >= 2
    assert_released(harness)


def test_stale_completed_frame_never_schedules_sound(monkeypatch):
    harness = install_client(monkeypatch, [estimate(0.8)], capture_age=2.0,
                             stop_when=lambda state: state.backend.attempts >= 3)
    assert main_module.main(["--mode", "remote", "--max-seconds", "5"]) == 0
    assert not harness.sound.events
    assert all(row["result"] is None and row["alert_level"] == "Unknown" for row in harness.renders)
    assert_released(harness)


def test_remote_unconfigured_classes_are_removed_before_display_and_alerting(monkeypatch):
    harness = install_client(monkeypatch, [estimate(0.5, class_id=2)],
                             stop_when=lambda state: any(row["result"] is not None for row in state.renders))
    assert main_module.main(["--mode", "remote", "--max-seconds", "5"]) == 0
    results = [row["result"] for row in harness.renders if row["result"] is not None]
    assert results and all(not result.objects for result in results)
    assert not harness.sound.events
    assert_released(harness)


def test_capture_age_expires_while_next_inference_is_stalled(monkeypatch):
    harness = install_client(monkeypatch, [estimate(0.8)], capture_age=0.8,
                             stop_when=lambda state: state.now >= 100.5)
    worker_type = main_module.InferenceWorker

    class StalledWorker(worker_type):
        submitted = False

        def submit(self, frame):
            if self.submitted:
                return False
            self.submitted = True
            return super().submit(frame)

    monkeypatch.setattr(main_module, "InferenceWorker", StalledWorker)
    assert main_module.main(["--max-seconds", "3"]) == 0
    levels = [row["alert_level"] for row in harness.renders]
    assert "Danger" in levels
    assert levels[-1] == "Unknown"
    assert harness.renders[-1]["result"] is None
    assert harness.sound.events == ["Danger"]
    assert_released(harness)


def test_departed_central_object_cancels_sound_during_normal_confirmation(monkeypatch):
    harness = install_client(monkeypatch, [estimate(0.8), estimate(0.8),
                                          InferenceResult(160, 120, [], {}, "cpu")],
                             stop_when=lambda state: state.backend.attempts >= 4)
    settings = main_module.load_settings(None)
    monkeypatch.setattr(main_module, "load_settings",
                        lambda path: replace(settings, alerts=replace(settings.alerts, confirm_frames=2)))
    assert main_module.main(["--max-seconds", "5"]) == 0
    assert any(row["alert_level"] == "Danger" and row["nearest_distance_m"] is None
               for row in harness.renders)
    assert harness.sound.cancel_count >= 2
    assert_released(harness)
