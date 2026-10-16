"""Small binary-JPEG client for the localhost inference service."""

from __future__ import annotations

import cv2
import numpy as np
import requests

from app.types import InferenceResult


class RemoteError(RuntimeError):
    """Inference service was unavailable or returned an invalid response."""


class RemoteClient:
    def __init__(self, server: str, timeout: float = 30.0, jpeg_quality: int = 90) -> None:
        if timeout <= 0:
            raise ValueError("Request timeout must be positive.")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100.")
        if not server.startswith(("http://", "https://")):
            raise ValueError("Server must start with http:// or https://.")
        self.server = server.rstrip("/")
        self.timeout = (min(3.0, timeout), timeout)
        self.jpeg_quality = jpeg_quality
        self._session = requests.Session()
        # Localhost inference must not be redirected through a system HTTP proxy.
        self._session.trust_env = False

    def infer(self, frame: np.ndarray) -> InferenceResult:
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not success:
            raise RemoteError("OpenCV could not encode the camera frame as JPEG.")
        try:
            response = self._session.post(
                f"{self.server}/infer", data=encoded.tobytes(),
                headers={"Content-Type": "image/jpeg"}, timeout=self.timeout,
            )
            response.raise_for_status()
            result = InferenceResult.from_dict(response.json())
            if (result.width, result.height) != (frame.shape[1], frame.shape[0]):
                raise ValueError("Result dimensions do not match the submitted frame.")
            return result
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            raise RemoteError(f"Inference request failed ({self.server}): {exc}") from exc

    def diagnostics(self) -> dict:
        try:
            response = self._session.get(f"{self.server}/diagnostics", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RemoteError(f"Diagnostics request failed: {exc}") from exc

    def close(self) -> None:
        self._session.close()
