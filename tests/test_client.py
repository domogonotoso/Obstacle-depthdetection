"""Host-side tests use fake cameras/network and never need models or CUDA."""

from __future__ import annotations

import threading
import time
import csv

import cv2
import numpy as np
import pytest
import requests

from app.camera import CameraError, CapturedFrame, LatestFrameCamera
from app.main import InferenceWorker
from app.remote import RemoteClient, RemoteError
from app.types import InferenceResult, ObjectEstimate
from app.visualization import render_frame
import app.main as main_module


def empty_result(width: int = 160, height: int = 120) -> InferenceResult:
    return InferenceResult(width, height, [], {"total": 2.0}, "cuda:0")


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    trust_env = True

    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.call = None

    def post(self, url, **kwargs):
        self.call = (url, kwargs)
        if self.error:
            raise self.error
        return FakeResponse(self.payload)

    def close(self):
        pass


def test_remote_binary_jpeg_preserves_frame_coordinate_contract():
    client = RemoteClient("http://127.0.0.1:8000", timeout=2.0)
    client._session.close()
    session = FakeSession(empty_result().to_dict())
    client._session = session
    result = client.infer(np.zeros((120, 160, 3), dtype=np.uint8))
    url, options = session.call
    assert url == "http://127.0.0.1:8000/infer"
    assert options["headers"] == {"Content-Type": "image/jpeg"}
    assert options["timeout"] == (2.0, 2.0)
    assert cv2.imdecode(np.frombuffer(options["data"], dtype=np.uint8), cv2.IMREAD_COLOR).shape == (120, 160, 3)
    assert (result.width, result.height) == (160, 120)
    client.close()


def test_remote_rejects_misaligned_result_and_wraps_timeout():
    client = RemoteClient("http://127.0.0.1:8000")
    client._session.close()
    client._session = FakeSession(empty_result(width=80).to_dict())
    with pytest.raises(RemoteError, match="dimensions"):
        client.infer(np.zeros((120, 160, 3), dtype=np.uint8))
    client._session = FakeSession(error=requests.Timeout("timed out"))
    with pytest.raises(RemoteError, match="timed out"):
        client.infer(np.zeros((120, 160, 3), dtype=np.uint8))
    client.close()


def test_worker_has_no_backlog_and_preserves_exact_inference_frame():
    entered, release = threading.Event(), threading.Event()

    class Backend:
        def infer(self, image):
            entered.set()
            assert release.wait(2)
            return empty_result(image.shape[1], image.shape[0])

    worker = InferenceWorker(Backend())
    first = CapturedFrame(np.zeros((120, 160, 3), np.uint8), 1, time.monotonic())
    next_frame = CapturedFrame(np.ones((120, 160, 3), np.uint8), 2, time.monotonic())
    try:
        assert worker.submit(first)
        assert entered.wait(1)
        assert not worker.submit(next_frame)
        release.set()
        deadline = time.monotonic() + 2
        completed = None
        while completed is None and time.monotonic() < deadline:
            completed = worker.take_completed()
            time.sleep(0.005)
        assert completed is not None
        assert completed.frame is first
        assert completed.error is None
        assert completed.result.width == 160
    finally:
        release.set()
        assert worker.close()


def test_camera_open_failure_releases_capture(monkeypatch):
    class ClosedCapture:
        released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    capture = ClosedCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: capture)
    with pytest.raises(CameraError, match="Cannot open camera"):
        LatestFrameCamera()
    assert capture.released


def test_camera_retains_latest_only_and_reports_disconnect(monkeypatch):
    class Capture:
        released = False
        count = 0

        def isOpened(self):
            return True

        def set(self, *args):
            return True

        def read(self):
            self.count += 1
            if self.count <= 2:
                return True, np.full((4, 6, 3), self.count, dtype=np.uint8)
            return False, None

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: capture)
    camera = LatestFrameCamera(failure_limit=1)
    try:
        deadline = time.monotonic() + 1
        while camera.error is None and time.monotonic() < deadline:
            time.sleep(0.005)
        latest = camera.latest()
        assert latest.frame_id == 1
        assert latest.image[0, 0, 0] == 2
        assert camera.latest(after_id=1) is None
        assert "stopped returning frames" in camera.error
    finally:
        camera.close()
    assert capture.released


def test_render_is_nonmutating_and_rejects_stale_frame_dimensions():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    obj = ObjectEstimate((5, 60, 80, 110), "chair", 0.8, 56, "LEFT", 1.8, 1.0, 100)
    result = InferenceResult(160, 120, [obj], {}, "cuda:0")
    rendered = render_frame(image, result)
    assert rendered.shape == image.shape
    assert np.any(rendered)
    assert not np.any(image)
    with pytest.raises(ValueError, match="dimensions"):
        render_frame(image, empty_result(width=80))


def test_headless_loop_records_measured_result_and_releases_resources(monkeypatch, tmp_path):
    frame = CapturedFrame(np.zeros((120, 160, 3), np.uint8), 0, time.monotonic())

    class Camera:
        error = None
        closed = False

        def latest(self, after_id=-1):
            return frame if after_id < 0 else None

        def close(self):
            self.closed = True

    class Backend:
        closed = False

        def infer(self, image):
            return empty_result()

        def close(self):
            self.closed = True

    camera, backend = Camera(), Backend()
    monkeypatch.setattr(main_module, "LatestFrameCamera", lambda *args: camera)
    monkeypatch.setattr(main_module, "_build_backend", lambda args: backend)
    # Headless execution must work with opencv-python-headless as well.
    def forbidden_gui_call():
        raise AssertionError("A headless run must not call GUI cleanup")

    monkeypatch.setattr(cv2, "destroyAllWindows", forbidden_gui_call)
    path = tmp_path / "timings.csv"
    assert main_module.main(["--headless", "--max-frames", "1", "--max-seconds", "2",
                             "--benchmark-csv", str(path)]) == 0
    assert camera.closed and backend.closed
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["frame_id"] == "0"
    assert float(rows[0]["client_inference_ms"]) >= 0
    assert rows[0]["completed_fps"] == "0.0000"


def test_q_key_exits_and_releases_camera_and_windows(monkeypatch):
    frame = CapturedFrame(np.zeros((120, 160, 3), np.uint8), 0, time.monotonic())

    class Camera:
        error = None
        closed = False

        def latest(self, after_id=-1):
            return frame if after_id < 0 else None

        def close(self):
            self.closed = True

    class Backend:
        closed = False

        def infer(self, image):
            return empty_result()

        def close(self):
            self.closed = True

    camera, backend = Camera(), Backend()
    windows_closed = []
    monkeypatch.setattr(main_module, "LatestFrameCamera", lambda *args: camera)
    monkeypatch.setattr(main_module, "_build_backend", lambda args: backend)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *args: ord("q"))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: windows_closed.append(True))
    assert main_module.main(["--max-seconds", "2"]) == 0
    assert camera.closed and backend.closed and windows_closed
