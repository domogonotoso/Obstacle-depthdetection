"""Camera acquisition with a single latest-frame slot (no growing queue)."""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time

import cv2
import numpy as np


class CameraError(RuntimeError):
    """The requested camera could not be opened or stopped delivering frames."""


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray
    frame_id: int
    captured_at: float


class LatestFrameCamera:
    """Read continuously and retain only the freshest image.

    Frames are treated as immutable after capture. Consumers drawing on an image
    must copy it, so an inference result can stay paired with its original frame.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        backend: str = "auto",
        failure_limit: int = 10,
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0 or failure_limit <= 0:
            raise ValueError("Camera dimensions, FPS and failure_limit must be positive.")
        backends = {"auto": cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY,
                    "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
        if backend not in backends:
            raise ValueError(f"Unsupported camera backend: {backend}")
        self._capture = cv2.VideoCapture(index, backends[backend])
        if not self._capture.isOpened():
            self._capture.release()
            raise CameraError(
                f"Cannot open camera {index}. Check Windows camera permissions, "
                "close other camera apps, or try --camera-backend msmf."
            )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Backend may ignore this hint.
        self._failure_limit = failure_limit
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest: CapturedFrame | None = None
        self._error: str | None = None
        self._thread = threading.Thread(target=self._read_loop, name="camera-capture", daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        sequence = 0
        failures = 0
        try:
            while not self._stop.is_set():
                success, image = self._capture.read()
                captured_at = time.monotonic()
                if not success or image is None or image.size == 0:
                    failures += 1
                    if failures >= self._failure_limit:
                        raise CameraError("Camera stopped returning frames. Check the connection.")
                    self._stop.wait(0.05)
                    continue
                failures = 0
                with self._lock:
                    self._latest = CapturedFrame(image, sequence, captured_at)
                sequence += 1
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
        finally:
            self._capture.release()

    def latest(self, after_id: int = -1) -> CapturedFrame | None:
        with self._lock:
            current = self._latest
            return current if current is not None and current.frame_id > after_id else None

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def close(self, timeout: float = 1.0) -> None:
        self._stop.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            # Some Windows backends block inside read(); release unblocks them.
            self._capture.release()
            self._thread.join(timeout)

    def __enter__(self) -> LatestFrameCamera:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
