"""Host-only warning sounds with a bounded, asynchronous playback worker.

Distance thresholds and repeat timing belong to the alert controller. This module
only plays its approved events; neither inference nor the GUI waits for a beep.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib
import logging
import sys
import threading


LOGGER = logging.getLogger(__name__)
_PRIORITY = {"Warning": 1, "Danger": 2}


class SoundPlayer:
    """Keep at most one pending sound, giving Danger priority over Warning.

    An injected ``player(level)`` runs synchronously *inside the worker*, which
    makes testing possible without a speaker. Custom callbacks cannot be forcibly
    interrupted; default Windows playback checks cancellation between beeps.
    """

    def __init__(self, enabled: bool = True, player: Callable[[str], None] | None = None) -> None:
        self._player = player
        self._supported = player is not None or sys.platform == "win32"
        self._enabled = bool(enabled and self._supported)
        self._condition = threading.Condition()
        self._pending: str | None = None
        self._active: str | None = None
        self._active_cancel: threading.Event | None = None
        self._closed = False
        self._error_logged = False
        self._thread = threading.Thread(target=self._run, name="warning-sound", daemon=True)
        self._thread.start()

    @property
    def enabled(self) -> bool:
        with self._condition:
            return self._enabled and not self._closed

    def set_enabled(self, enabled: bool) -> None:
        """Mute immediately, discarding pending events instead of replaying them later."""
        with self._condition:
            self._enabled = bool(enabled and self._supported and not self._closed)
            if not self._enabled:
                self._cancel_locked()

    def play(self, level: str) -> bool:
        """Schedule Warning/Danger without blocking; return whether it was accepted.

        Repeated or lower-priority events do not queue behind an active sound.
        A Danger request replaces a pending Warning and cancels an active Warning
        between beeps. Normal cancels outstanding sounds; unknown levels are ignored.
        """
        normalized = str(level).strip().title()
        if normalized == "Normal":
            self.cancel()
            return False
        if normalized not in _PRIORITY:
            return False
        with self._condition:
            if not self._enabled or self._closed:
                return False
            priority = _PRIORITY[normalized]
            if self._active is not None and _PRIORITY[self._active] >= priority:
                return False
            if self._pending is not None and _PRIORITY[self._pending] >= priority:
                return False
            self._pending = normalized
            if self._active_cancel is not None:
                self._active_cancel.set()
            self._condition.notify()
            return True

    def cancel(self) -> None:
        """Discard stale sounds after Normal, invalid depth, or communication loss."""
        with self._condition:
            self._cancel_locked()

    def _cancel_locked(self) -> None:
        self._pending = None
        if self._active_cancel is not None:
            self._active_cancel.set()
        self._condition.notify_all()

    def close(self, timeout: float = 1.0) -> bool:
        """Cancel work and wait at most timeout seconds; return whether it stopped.

        Windows Beep may finish its current short tone. A blocked third-party
        injected callback may outlive the timeout, so the worker is also a daemon.
        """
        with self._condition:
            self._closed = True
            self._enabled = False
            self._cancel_locked()
        if threading.current_thread() is self._thread:
            return False
        self._thread.join(max(0.0, timeout))
        return not self._thread.is_alive()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                level = self._pending
                self._pending = None
                if level is None or not self._enabled:
                    continue
                cancelled = threading.Event()
                self._active = level
                self._active_cancel = cancelled
            try:
                if not cancelled.is_set():
                    if self._player is None:
                        self._play_windows_pattern(level, cancelled)
                    else:
                        self._player(level)
            except Exception:
                # A missing device or a failing callback must not take down video.
                # Only the first failure is logged to avoid per-frame log flooding.
                if not self._error_logged:
                    LOGGER.warning("Warning sound failed; video processing will continue.", exc_info=True)
                    self._error_logged = True
            finally:
                with self._condition:
                    self._active = None
                    self._active_cancel = None
                    self._condition.notify_all()

    def _play_windows_pattern(self, level: str, cancelled: threading.Event) -> None:
        # No winsound import on the server, on import of this module, or when muted.
        try:
            winsound = importlib.import_module("winsound")
        except ImportError:
            with self._condition:
                self._supported = False
                self._enabled = False
                self._cancel_locked()
            raise
        pattern = ((660, 120),) if level == "Warning" else ((1000, 140), (1000, 140))
        for index, (frequency, duration_ms) in enumerate(pattern):
            if cancelled.is_set():
                return
            if index and cancelled.wait(0.06):
                return
            winsound.Beep(frequency, duration_ms)

    def __enter__(self) -> SoundPlayer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
