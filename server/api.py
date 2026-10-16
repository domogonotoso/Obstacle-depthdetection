"""Single-worker GPU inference API accepting bounded raw JPEG requests."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from typing import Any, Callable

import cv2
from fastapi import FastAPI, HTTPException, Request
import numpy as np
from starlette.concurrency import run_in_threadpool

LOGGER = logging.getLogger(__name__)
_SOF_MARKERS = frozenset((0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF))


def _default_pipeline_factory() -> Any:
    # Importing this module and running unit tests never loads model weights.
    from app.config import VisionConfig
    from app.pipeline import VisionPipeline

    return VisionPipeline(VisionConfig.from_env())


def _positive_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _jpeg_dimensions(data: bytes | bytearray) -> tuple[int, int]:
    """Read SOF before decoding so an oversized JPEG cannot allocate its pixels."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise HTTPException(400, "Invalid JPEG image")
    offset = 2
    while offset < len(data):
        if data[offset] != 0xFF:
            raise HTTPException(400, "Invalid JPEG header")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in (0xD9, 0xDA):
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            continue
        if offset + 2 > len(data):
            break
        segment_length = int.from_bytes(data[offset:offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(data):
            break
        if marker in _SOF_MARKERS:
            if segment_length < 8:
                break
            height = int.from_bytes(data[offset + 3:offset + 5], "big")
            width = int.from_bytes(data[offset + 5:offset + 7], "big")
            if width > 0 and height > 0:
                return width, height
            break
        offset += segment_length
    raise HTTPException(400, "JPEG dimensions could not be read")


def _decode_frame(data: bytearray, max_pixels: int) -> np.ndarray:
    width, height = _jpeg_dimensions(data)
    if width * height > max_pixels:
        raise HTTPException(413, f"Image exceeds {max_pixels} pixels")
    try:
        frame = cv2.imdecode(
            np.frombuffer(data, dtype=np.uint8),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
    except cv2.error as exc:
        raise HTTPException(400, "JPEG decoding failed") from exc
    if frame is None or frame.size == 0:
        raise HTTPException(400, "JPEG decoding failed")
    if frame.shape[0] * frame.shape[1] > max_pixels:
        raise HTTPException(413, f"Image exceeds {max_pixels} pixels")
    if (frame.shape[1], frame.shape[0]) != (width, height):
        raise HTTPException(400, "JPEG dimensions are inconsistent")
    return frame


def _infer_jpeg(pipeline: Any, data: bytearray, max_pixels: int) -> dict[str, Any]:
    frame = _decode_frame(data, max_pixels)
    return pipeline.infer(frame).to_dict()


def create_app(pipeline_factory: Callable[[], Any] | None = None) -> FastAPI:
    """Create an app; injectable factory lets tests run without CUDA or downloads."""
    factory = pipeline_factory or _default_pipeline_factory
    max_bytes = _positive_env("API_MAX_JPEG_BYTES", 5 * 1024 * 1024)
    max_pixels = _positive_env("API_MAX_PIXELS", 1920 * 1080)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.ready = False
        application.state.inference_lock = asyncio.Lock()
        application.state.operation_task = None
        try:
            pipeline = await run_in_threadpool(factory)
        except Exception:
            LOGGER.exception("Model initialization failed")
            raise RuntimeError("Model initialization failed; see server logs") from None
        application.state.pipeline = pipeline
        application.state.ready = True
        try:
            yield
        finally:
            application.state.ready = False
            # A disconnected HTTP client cannot stop its CUDA worker. Drain it
            # before model cleanup, including after Uvicorn cancels the request.
            active = application.state.operation_task
            if active is not None:
                await asyncio.shield(asyncio.gather(active, return_exceptions=True))
            close = getattr(pipeline, "close", None)
            if callable(close):
                await run_in_threadpool(close)

    application = FastAPI(title="Obstacle Vision inference", version="1.0.0", lifespan=lifespan)
    application.state.ready = False

    async def claim_slot() -> asyncio.Lock:
        lock = application.state.inference_lock
        # Lock.acquire() completes without yielding when the lock is free.
        # Uvicorn must run with one worker so there is only one model instance.
        if lock.locked():
            raise HTTPException(429, "GPU busy; send a fresh request later", headers={"Retry-After": "1"})
        await lock.acquire()
        return lock

    def submit_claimed(lock: asyncio.Lock, function: Callable, *args: Any) -> asyncio.Task:
        task = asyncio.create_task(run_in_threadpool(function, *args))
        application.state.operation_task = task

        def finished(completed: asyncio.Task) -> None:
            # Completion owns the slot: cancelling the HTTP request must never
            # permit another inference while its worker thread still uses CUDA.
            lock.release()
            if application.state.operation_task is completed:
                application.state.operation_task = None
            if not completed.cancelled():
                error = completed.exception()  # Observe errors even if the client left.
                if error is not None and not isinstance(error, HTTPException):
                    LOGGER.error("GPU operation failed", exc_info=(type(error), error, error.__traceback__))

        task.add_done_callback(finished)
        return task

    @application.get("/health")
    async def health() -> dict[str, Any]:
        ready = application.state.ready
        if not ready:
            raise HTTPException(503, "Models are not ready")
        return {"status": "ok", "ready": True}

    @application.get("/diagnostics")
    async def diagnostics() -> dict[str, Any]:
        if not application.state.ready:
            raise HTTPException(503, "Models are not ready")
        lock = await claim_slot()
        task = submit_claimed(lock, application.state.pipeline.diagnostics)
        try:
            result = await asyncio.shield(task)
        except Exception:
            raise HTTPException(500, "Diagnostics failed; see server logs") from None
        return {**result, "ready": True}

    @application.post("/infer")
    async def infer(request: Request) -> dict[str, Any]:
        if not application.state.ready:
            raise HTTPException(503, "Models are not ready")
        media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media_type != "image/jpeg":
            raise HTTPException(415, "Content-Type must be image/jpeg")
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                length = int(content_length)
            except ValueError:
                raise HTTPException(400, "Invalid Content-Length") from None
            if length < 0:
                raise HTTPException(400, "Invalid Content-Length")
            if length > max_bytes:
                raise HTTPException(413, f"JPEG exceeds {max_bytes} bytes")

        lock = await claim_slot()
        owns_slot = True
        try:
            data = bytearray()
            async for chunk in request.stream():
                if len(data) + len(chunk) > max_bytes:
                    raise HTTPException(413, f"JPEG exceeds {max_bytes} bytes")
                data.extend(chunk)
            if not data:
                raise HTTPException(400, "JPEG body is empty")
            task = submit_claimed(lock, _infer_jpeg, application.state.pipeline, data, max_pixels)
            owns_slot = False
            return await asyncio.shield(task)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(500, "Inference failed; see server logs") from None
        finally:
            if owns_slot:
                lock.release()

    return application


app = create_app()
