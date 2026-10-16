"""Exercise host sound scheduling without touching a real speaker."""

from __future__ import annotations

import logging
import sys
import threading
from types import SimpleNamespace

import pytest

from app.sound import SoundPlayer


def wait_idle(sound: SoundPlayer) -> None:
    with sound._condition:
        assert sound._condition.wait_for(
            lambda: sound._active is None and sound._pending is None, timeout=2
        ), "sound worker did not become idle"


def test_play_returns_while_injected_playback_is_blocked():
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    def blocking_player(level):
        entered.set()
        release.wait(2)

    sound = SoundPlayer(player=blocking_player)
    accepted = []

    def request():
        accepted.append(sound.play("Warning"))
        returned.set()

    request_thread = threading.Thread(target=request, daemon=True)
    try:
        request_thread.start()
        assert returned.wait(1), "play() waited for sound playback"
        assert entered.wait(1)
        assert accepted == [True]
        assert not release.is_set()
    finally:
        release.set()
        request_thread.join(2)
        assert sound.close()


def test_danger_has_priority_and_repeated_events_do_not_build_a_queue():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def player(level):
        calls.append(level)
        if level == "Warning":
            entered.set()
            release.wait(2)

    sound = SoundPlayer(player=player)
    try:
        assert sound.play("Warning")
        assert entered.wait(1)
        for _ in range(100):
            assert not sound.play("Warning")
        assert sound.play("Danger")
        for _ in range(100):
            assert not sound.play("Warning")
            assert not sound.play("Danger")
        release.set()
        wait_idle(sound)
        assert calls == ["Warning", "Danger"]
    finally:
        release.set()
        assert sound.close()


def test_danger_replaces_pending_warning():
    calls = []
    with SoundPlayer(player=calls.append) as sound:
        # Hold the scheduler lock so the worker cannot consume the first event.
        with sound._condition:
            assert sound.play("Warning")
            assert sound.play("Danger")
        wait_idle(sound)
        assert calls == ["Danger"]


def test_muted_player_does_not_schedule_and_toggle_discards_old_events():
    calls = []
    with SoundPlayer(enabled=False, player=calls.append) as sound:
        assert not sound.enabled
        assert not sound.play("Danger")
        sound.set_enabled(True)
        assert sound.enabled
        with sound._condition:
            assert sound.play("Warning")
            sound.set_enabled(False)
            sound.set_enabled(True)
        wait_idle(sound)
        assert calls == []
        assert sound.play("danger")
        wait_idle(sound)
        assert calls == ["Danger"]


@pytest.mark.parametrize("cancel_method", ["cancel", "normal", "mute"])
def test_cancellation_discards_pending_event(cancel_method):
    calls = []
    with SoundPlayer(player=calls.append) as sound:
        with sound._condition:
            assert sound.play("Danger")
            if cancel_method == "normal":
                assert not sound.play("Normal")
            elif cancel_method == "mute":
                sound.set_enabled(False)
            else:
                sound.cancel()
        wait_idle(sound)
        assert calls == []


def test_callback_failure_is_logged_once_and_worker_recovers(caplog):
    calls = []

    def player(level):
        calls.append(level)
        if len(calls) < 3:
            raise RuntimeError("simulated speaker failure")

    with caplog.at_level(logging.WARNING, logger="app.sound"):
        with SoundPlayer(player=player) as sound:
            for _ in range(3):
                assert sound.play("Warning")
                wait_idle(sound)
    assert calls == ["Warning"] * 3
    assert len([record for record in caplog.records if record.name == "app.sound"]) == 1


def test_default_windows_patterns_are_distinct_without_real_beeps(monkeypatch):
    calls = []
    monkeypatch.setattr("app.sound.sys.platform", "win32")
    monkeypatch.setitem(sys.modules, "winsound", SimpleNamespace(Beep=lambda *args: calls.append(args)))
    with SoundPlayer() as sound:
        assert sound.play("Warning")
        wait_idle(sound)
        assert calls == [(660, 120)]
        calls.clear()
        assert sound.play("Danger")
        wait_idle(sound)
        assert calls == [(1000, 140), (1000, 140)]


@pytest.mark.parametrize("operation", ["mute", "cancel", "close"])
def test_default_pattern_cancels_between_beeps(monkeypatch, operation):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def fake_beep(*args):
        calls.append(args)
        entered.set()
        release.wait(2)

    monkeypatch.setattr("app.sound.sys.platform", "win32")
    monkeypatch.setitem(sys.modules, "winsound", SimpleNamespace(Beep=fake_beep))
    sound = SoundPlayer()
    try:
        assert sound.play("Danger")
        assert entered.wait(1)
        if operation == "mute":
            sound.set_enabled(False)
        elif operation == "close":
            assert not sound.close(timeout=0)
        else:
            sound.cancel()
        release.set()
        if operation != "close":
            wait_idle(sound)
        assert sound.close()
        assert calls == [(1000, 140)]
    finally:
        release.set()
        sound.close()


def test_missing_winsound_disables_sound_and_keeps_video_callers_running(monkeypatch, caplog):
    monkeypatch.setattr("app.sound.sys.platform", "win32")

    def missing_module(name):
        raise ImportError("winsound unavailable")

    monkeypatch.setattr("app.sound.importlib.import_module", missing_module)
    with SoundPlayer() as sound:
        assert sound.play("Danger")
        wait_idle(sound)
        assert not sound.enabled
        sound.set_enabled(True)
        assert not sound.enabled
        assert not sound.play("Danger")
    assert "video processing will continue" in caplog.text


def test_non_windows_default_is_disabled_without_importing_winsound(monkeypatch):
    monkeypatch.setattr("app.sound.sys.platform", "linux")

    def unexpected_import(name):
        pytest.fail("unsupported sound backend must not be imported")

    monkeypatch.setattr("app.sound.importlib.import_module", unexpected_import)
    with SoundPlayer() as sound:
        assert not sound.enabled
        sound.set_enabled(True)
        assert not sound.enabled
        assert not sound.play("Danger")


def test_close_is_idempotent_discards_pending_and_prevents_restarting():
    calls = []
    sound = SoundPlayer(player=calls.append)
    with sound._condition:
        assert sound.play("Warning")
        # The lock keeps playback pending until close clears it.
        # timeout=0 avoids waiting while intentionally holding that lock.
        sound.close(timeout=0)
    assert sound.close()
    assert sound.close()
    sound.set_enabled(True)
    assert not sound.enabled
    assert not sound.play("Danger")
    assert calls == []


def test_unknown_levels_are_ignored():
    calls = []
    with SoundPlayer(player=calls.append) as sound:
        assert not sound.play("Unavailable")
        assert not sound.play("")
        assert calls == []
