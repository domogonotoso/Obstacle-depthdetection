"""Service checks use an injected pipeline: no network, camera, GPU or weights."""

from concurrent.futures import ThreadPoolExecutor
import asyncio
import threading

import cv2
from fastapi.testclient import TestClient
import httpx
import numpy as np
import pytest

from server.api import create_app
from app.types import InferenceResult


class FakePipeline:
    def __init__(self):
        self.calls = 0
        self.closed = False
        self.raise_inference = False
        self.entered = None
        self.release = None

    def infer(self, frame):
        self.calls += 1
        if self.entered is not None:
            self.entered.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("Test did not release inference")
        if self.raise_inference:
            raise RuntimeError("private model path or other internal detail")
        assert frame.dtype == np.uint8 and frame.shape[2] == 3
        return InferenceResult(
            width=frame.shape[1], height=frame.shape[0], objects=[],
            timings_ms={"total": 5.0}, device="cuda:0",
        )

    def diagnostics(self):
        return {"device": "cuda:0", "cuda_available": True, "gpu_name": "Fake GPU"}

    def close(self):
        self.closed = True


@pytest.fixture
def jpeg():
    ok, encoded = cv2.imencode(".jpg", np.zeros((24, 32, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def post_jpeg(client, data):
    return client.post("/infer", content=data, headers={"Content-Type": "image/jpeg"})


def test_health_diagnostics_and_result_contract(jpeg):
    pipeline = FakePipeline()
    with TestClient(create_app(lambda: pipeline)) as client:
        assert client.get("/health").json() == {"status": "ok", "ready": True}
        assert client.get("/diagnostics").json() == {
            "ready": True, "device": "cuda:0", "cuda_available": True, "gpu_name": "Fake GPU"
        }
        result = post_jpeg(client, jpeg)
        assert result.status_code == 200
        assert result.json() == {
            "width": 32, "height": 24, "objects": [],
            "timings_ms": {"total": 5.0}, "device": "cuda:0",
        }
        assert InferenceResult.from_dict(result.json()).objects == []
        assert pipeline.calls == 1
    assert pipeline.closed


@pytest.mark.parametrize("body", [b"", b"not jpeg", b"\xff\xd8\xff\xd9", b"\xff\xd8\xff\xe0\x00\x01"])
def test_invalid_jpeg_returns_400(body):
    pipeline = FakePipeline()
    with TestClient(create_app(lambda: pipeline)) as client:
        assert post_jpeg(client, body).status_code == 400
        assert pipeline.calls == 0


def test_wrong_content_type_returns_415(jpeg):
    with TestClient(create_app(FakePipeline)) as client:
        response = client.post("/infer", content=jpeg, headers={"Content-Type": "image/png"})
        assert response.status_code == 415


def test_jpeg_byte_limit_and_streaming_limit(monkeypatch, jpeg):
    monkeypatch.setenv("API_MAX_JPEG_BYTES", "100")
    pipeline = FakePipeline()
    with TestClient(create_app(lambda: pipeline)) as client:
        assert post_jpeg(client, jpeg).status_code == 413
        response = client.post(
            "/infer", content=iter((jpeg[:70], jpeg[70:])), headers={"Content-Type": "image/jpeg"}
        )
        assert response.status_code == 413
        assert pipeline.calls == 0


def test_pixel_limit_before_model_inference(monkeypatch, jpeg):
    monkeypatch.setenv("API_MAX_PIXELS", "100")
    pipeline = FakePipeline()
    with TestClient(create_app(lambda: pipeline)) as client:
        assert post_jpeg(client, jpeg).status_code == 413
        assert pipeline.calls == 0


def test_busy_inference_rejected_and_health_remains_responsive(jpeg):
    pipeline = FakePipeline()
    pipeline.entered = threading.Event()
    pipeline.release = threading.Event()
    with TestClient(create_app(lambda: pipeline)) as client:
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(post_jpeg, client, jpeg)
            try:
                assert pipeline.entered.wait(timeout=3)
                assert client.get("/health").status_code == 200
                second = post_jpeg(client, jpeg)
                assert second.status_code == 429
                assert second.headers["retry-after"] == "1"
                assert client.get("/diagnostics").status_code == 429
            finally:
                pipeline.release.set()
            assert first.result(timeout=3).status_code == 200
        assert pipeline.calls == 1


def test_inference_error_is_redacted_and_slot_is_released(jpeg):
    pipeline = FakePipeline()
    pipeline.raise_inference = True
    with TestClient(create_app(lambda: pipeline)) as client:
        failed = post_jpeg(client, jpeg)
        assert failed.status_code == 500
        assert "private" not in failed.text
        pipeline.raise_inference = False
        assert post_jpeg(client, jpeg).status_code == 200


def test_startup_error_is_visible():
    def failing_factory():
        raise RuntimeError("Missing CUDA or failed download")

    with pytest.raises(RuntimeError, match="Model initialization failed"):
        with TestClient(create_app(failing_factory)):
            pass


def test_cancelled_request_keeps_gpu_slot_until_worker_finishes(jpeg):
    pipeline = FakePipeline()
    pipeline.entered = threading.Event()
    pipeline.release = threading.Event()
    application = create_app(lambda: pipeline)

    async def exercise():
        async with application.router.lifespan_context(application):
            transport = httpx.ASGITransport(app=application)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                first = asyncio.create_task(client.post(
                    "/infer", content=jpeg, headers={"Content-Type": "image/jpeg"}
                ))
                try:
                    assert await asyncio.to_thread(pipeline.entered.wait, 3)
                    first.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await first
                    assert application.state.inference_lock.locked()
                    assert (await client.get("/health")).status_code == 200
                    assert (await client.get("/diagnostics")).status_code == 429
                    second = await client.post(
                        "/infer", content=jpeg, headers={"Content-Type": "image/jpeg"}
                    )
                    assert second.status_code == 429
                    assert pipeline.calls == 1
                finally:
                    pipeline.release.set()
                active = application.state.operation_task
                if active is not None:
                    await active
                assert not application.state.inference_lock.locked()
                assert (await client.get("/diagnostics")).status_code == 200

    asyncio.run(exercise())
