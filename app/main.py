"""Windows camera host, with local or Docker-backed inference.

Run ``python -m app.main --help`` for the available camera/model options.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any

import cv2

from app.camera import CameraError, CapturedFrame, LatestFrameCamera
from app.alerts import AlertController, visible_objects
from app.settings import load_settings
from app.sound import SoundPlayer
from app.types import InferenceResult
from app.visualization import render_frame


@dataclass(frozen=True)
class CompletedInference:
    frame: CapturedFrame
    result: InferenceResult | None
    elapsed_ms: float
    completed_at: float
    error: str | None = None


class InferenceWorker:
    """A single in-flight inference, without an input/result backlog.

    The worker owns the backend. Closing the viewer remains responsive during
    an inference; network requests still have a bounded connect/read timeout.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._condition = threading.Condition()
        self._pending: CapturedFrame | None = None
        self._completed: CompletedInference | None = None
        self._busy = False
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="model-inference", daemon=True)
        self._thread.start()

    def submit(self, frame: CapturedFrame) -> bool:
        with self._condition:
            if self._stop or self._busy or self._completed is not None:
                return False
            self._busy = True
            self._pending = frame
            self._condition.notify()
            return True

    def take_completed(self) -> CompletedInference | None:
        with self._condition:
            result = self._completed
            self._completed = None
            return result

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stop or self._pending is not None)
                    if self._stop:
                        return
                    frame = self._pending
                    self._pending = None
                assert frame is not None
                started = time.monotonic()
                result = None
                error = None
                try:
                    result = self._backend.infer(frame.image)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                finished = time.monotonic()
                completed = CompletedInference(frame, result, (finished - started) * 1000,
                                               finished, error)
                with self._condition:
                    self._completed = completed
                    self._busy = False
        finally:
            close = getattr(self._backend, "close", None)
            if close is not None:
                close()

    def close(self, timeout: float = 1.0) -> bool:
        with self._condition:
            self._stop = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout)
        return not self._thread.is_alive()


class BenchmarkWriter:
    """Append observed frame timings; no synthetic FPS estimates."""

    FIELDS = ["timestamp_utc", "frame_id", "width", "height", "device", "object_count",
              "completed_fps", "capture_to_result_ms", "client_inference_ms", "timings_ms_json"]

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists() or path.stat().st_size == 0
        if not needs_header:
            with path.open("r", encoding="utf-8", newline="") as existing:
                if next(csv.reader(existing), []) != self.FIELDS:
                    raise ValueError(f"Benchmark CSV has an incompatible header: {path}")
        self._file = path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        if needs_header:
            self._writer.writeheader()
            self._file.flush()

    def record(self, completed: CompletedInference, fps: float) -> None:
        result = completed.result
        assert result is not None
        self._writer.writerow({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "frame_id": completed.frame.frame_id, "width": result.width, "height": result.height,
            "device": result.device, "object_count": len(result.objects),
            "completed_fps": f"{fps:.4f}",
            "capture_to_result_ms": f"{(completed.completed_at - completed.frame.captured_at) * 1000:.3f}",
            "client_inference_ms": f"{completed.elapsed_ms:.3f}",
            "timings_ms_json": json.dumps(result.timings_ms, sort_keys=True),
        })
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Webcam object detection and Estimated Distance viewer")
    parser.add_argument("--mode", choices=("local", "remote"), default="local")
    parser.add_argument("--config", type=Path, help="TOML detection/display/alert settings (default: config/settings.toml)")
    parser.add_argument("--mute", action="store_true", help="Start with warning sound disabled; M toggles it")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("auto", "dshow", "msmf", "any"), default="auto")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda:0", help="Local inference device; CUDA is required by default")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Local model cache directory")
    parser.add_argument("--yolo-size", type=int, default=640, help="Local YOLO input size (remote: server YOLO_SIZE)")
    parser.add_argument("--depth-size", type=int, default=518, help="Local depth input size (remote: server DEPTH_SIZE)")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--central-ratio", type=float, default=0.5)
    parser.add_argument("--min-valid-fraction", type=float, default=0.2)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=20.0)
    parser.add_argument("--request-timeout", type=float, default=30.0,
                        help="Remote connect/read timeout; seconds")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--hide-zones", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Capture/infer without an OpenCV window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many successful results; 0 = unlimited")
    parser.add_argument("--max-seconds", type=float, default=0, help="Stop capture after this wall time; 0 = unlimited")
    parser.add_argument("--failure-limit", type=int, default=0,
                        help="Stop after consecutive inference failures; 0 = keep retrying")
    parser.add_argument("--benchmark-csv", type=Path, help="Append measured per-result timings")
    parser.add_argument("--truth-m", type=float, help="Tape-measured distance in metres for S-key CSV recording")
    parser.add_argument("--measurement-csv", type=Path, help="Append distance validation records")
    parser.add_argument("--object-index", type=int, default=0, help="Zero-based detection to record with S (starred on screen)")
    parser.add_argument("--diagnostics", action="store_true", help="Print backend/GPU diagnostics and exit without opening a camera")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ("width", "height", "yolo_size", "depth_size", "camera_fps", "request_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.failure_limit < 0:
        parser.error("--failure-limit cannot be negative")
    if args.max_frames < 0 or not math.isfinite(args.max_seconds) or args.max_seconds < 0 or args.object_index < 0:
        parser.error("--max-frames, --max-seconds and --object-index cannot be negative")
    if bool(args.measurement_csv) != (args.truth_m is not None):
        parser.error("--truth-m and --measurement-csv must be supplied together")
    if args.truth_m is not None and (not math.isfinite(args.truth_m) or args.truth_m <= 0):
        parser.error("--truth-m must be finite and positive")
    if args.measurement_csv and args.headless:
        parser.error("Distance recording uses the S key; omit --headless for --measurement-csv")


def _build_backend(args: argparse.Namespace) -> Any:
    if args.mode == "remote":
        from app.remote import RemoteClient
        return RemoteClient(args.server, args.request_timeout, args.jpeg_quality)
    # The Docker host needs no torch, Ultralytics, or depth model dependencies.
    from app.config import VisionConfig
    from app.pipeline import VisionPipeline
    return VisionPipeline(VisionConfig(
        device=args.device, model_dir=args.model_dir,
        yolo_size=args.yolo_size, depth_size=args.depth_size, confidence=args.confidence,
        central_ratio=args.central_ratio, min_valid_fraction=args.min_valid_fraction,
        min_depth_m=args.min_depth_m, max_depth_m=args.max_depth_m,
        target_classes=tuple(args._settings.detection.class_names.items()),
    ))


def run(args: argparse.Namespace) -> int:
    args._settings = load_settings(args.config)
    settings = args._settings.alerts
    backend = _build_backend(args)
    if args.diagnostics:
        try:
            print(json.dumps(backend.diagnostics(), ensure_ascii=False, indent=2))
        finally:
            close = getattr(backend, "close", None)
            if close is not None:
                close()
        return 0
    measurement = None
    if args.measurement_csv:
        from app.measurement import MeasurementRecorder
        measurement = MeasurementRecorder(args.measurement_csv, args.truth_m)
    benchmark = BenchmarkWriter(args.benchmark_csv) if args.benchmark_csv else None
    worker = InferenceWorker(backend)
    sound = SoundPlayer(enabled=settings.sound_enabled and not args.mute)
    alerts = AlertController(settings)
    alerts.set_sound_enabled(sound.enabled)
    alert_snapshot = alerts.tick()
    camera = None
    completion_times: deque[float] = deque(maxlen=30)
    elapsed_total_ms = 0.0
    last_submitted = -1
    displayed: CompletedInference | None = None
    success_count = 0
    consecutive_failures = 0
    status = "Waiting for the first inference..."
    retry_after = 0.0
    camera_started = time.monotonic()
    exit_code = 0
    try:
        camera = LatestFrameCamera(args.camera, args.width, args.height,
                                   args.camera_fps, args.camera_backend)
        while True:
            now = time.monotonic()
            if args.max_seconds and now - camera_started >= args.max_seconds:
                if success_count == 0:
                    exit_code = 4
                break
            if camera.error:
                print(f"Camera error: {camera.error}", file=sys.stderr)
                status = camera.error
                if not args.headless and displayed is not None:
                    sound.cancel()
                    cv2.imshow("Obstacle Vision", render_frame(displayed.frame.image, None, status=status,
                                                               alert_level="Unknown", sound_enabled=sound.enabled))
                    cv2.waitKey(1)
                exit_code = 2
                break
            completed = worker.take_completed()
            if completed is not None:
                if completed.error is not None:
                    consecutive_failures += 1
                    status = "Inference failed; check terminal (retrying)"
                    print(completed.error, file=sys.stderr)
                    displayed = None
                    alert_snapshot = alerts.invalidate(now)
                    sound.cancel()
                    retry_after = now + 1.0
                    if args.failure_limit and consecutive_failures >= args.failure_limit:
                        exit_code = 3
                        break
                else:
                    consecutive_failures = 0
                    # Remote responses also pass the host's configured class list.
                    # All remaining indices are preserved through display filtering.
                    allowed = args._settings.detection.class_names
                    filtered = replace(completed.result,
                                       objects=[obj for obj in completed.result.objects if obj.class_id in allowed])
                    if now >= completed.frame.captured_at + settings.result_timeout_s:
                        alert_snapshot = alerts.invalidate(now)
                    else:
                        alert_snapshot = alerts.update(filtered, now)
                    displayed = replace(completed, result=alert_snapshot.result)
                    if alert_snapshot.sound_event:
                        sound.play(alert_snapshot.sound_event)
                    completion_times.append(completed.completed_at)
                    elapsed_total_ms += completed.elapsed_ms
                    success_count += 1
                    fps = _observed_fps(completion_times)
                    if benchmark:
                        benchmark.record(completed, fps)
                    shown = len(visible_objects(alert_snapshot.result, settings.display_max_m)) if alert_snapshot.result else 0
                    status = f"{shown} shown | invalid depth {alert_snapshot.invalid_depth_count}"
                    if args.headless:
                        print(f"frame={completed.frame.frame_id} objects={len(completed.result.objects)} "
                              f"shown={shown} alert={alert_snapshot.level} "
                              f"fps={fps:.2f} inference_ms={completed.elapsed_ms:.1f}", flush=True)
                    if args.max_frames and success_count >= args.max_frames:
                        break
            # Only tick after consuming a new result, so an old frame cannot
            # schedule a repeat immediately before a fresh danger escalation.
            if displayed is not None and now >= displayed.frame.captured_at + settings.result_timeout_s:
                alerts.invalidate(now)
            alerts.set_sound_enabled(sound.enabled)
            alert_snapshot = alerts.tick(now)
            if alert_snapshot.sound_event:
                sound.play(alert_snapshot.sound_event)
            active_limit = (settings.danger_m if alert_snapshot.level == "Danger"
                            else settings.warning_m) + settings.hysteresis_m
            if (alert_snapshot.level in ("Normal", "Unknown")
                    or alert_snapshot.nearest_distance_m is None
                    or alert_snapshot.nearest_distance_m > active_limit):
                sound.cancel()
            newest = camera.latest(last_submitted)
            if newest is not None and now >= retry_after and worker.submit(newest):
                last_submitted = newest.frame_id
            if newest is None and last_submitted < 0 and now - camera_started > 8:
                raise CameraError("No camera frame arrived within 8 seconds.")
            if not args.headless:
                if displayed is not None:
                    frame = displayed.frame
                    result = alert_snapshot.result
                else:
                    frame = camera.latest()
                    result = None
                if frame is not None:
                    image = render_frame(frame.image, result, fps=_observed_fps(completion_times),
                                         age_ms=max(0.0, (now - frame.captured_at) * 1000),
                                         show_zones=not args.hide_zones, status=status,
                                         selected_index=args.object_index,
                                         max_display_distance_m=settings.display_max_m,
                                         warning_m=settings.warning_m, danger_m=settings.danger_m,
                                         alert_level=alert_snapshot.level, sound_enabled=sound.enabled,
                                         nearest_distance_m=alert_snapshot.nearest_distance_m,
                                         invalid_depth_count=alert_snapshot.invalid_depth_count)
                    cv2.imshow("Obstacle Vision", image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("m"), ord("M")):
                    sound.set_enabled(not sound.enabled)
                    alerts.set_sound_enabled(sound.enabled)
                    status = f"Sound {'ON' if sound.enabled else 'OFF'}"
                if key in (ord("s"), ord("S")) and measurement is not None and alert_snapshot.result is not None:
                    try:
                        row = measurement.record(alert_snapshot.result, object_index=args.object_index)
                        status = (f"Saved measurement #{args.object_index}: {row['status']}"
                                  if row is not None else "No object to record")
                        print(status, flush=True)
                    except (ValueError, IndexError, OSError) as exc:
                        status = f"Measurement failed: {exc}"
                        print(status, file=sys.stderr)
                if frame is not None:
                    try:
                        if cv2.getWindowProperty("Obstacle Vision", cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except cv2.error:
                        break  # The native window may already be destroyed by X.
            # Event.wait also keeps CPU usage low while inference is running.
            time.sleep(0.005)
    finally:
        sound.close()
        if camera is not None:
            camera.close()
        if not worker.close(timeout=1.0):
            print("Inference is finishing in the background; remote requests retain their configured timeout.",
                  file=sys.stderr)
        if benchmark:
            benchmark.close()
        close_measurement = getattr(measurement, "close", None)
        if close_measurement is not None:
            close_measurement()
        if not args.headless:
            cv2.destroyAllWindows()
        if success_count:
            mean_ms = elapsed_total_ms / success_count
            print(f"Measured: {success_count} results; rolling completed FPS={_observed_fps(completion_times):.2f}; "
                  f"mean client inference={mean_ms:.1f} ms", flush=True)
        else:
            print("No successful inference results were measured.", flush=True)
    return exit_code


def _observed_fps(times: deque[float]) -> float:
    if len(times) < 2 or times[-1] <= times[0]:
        return 0.0
    return (len(times) - 1) / (times[-1] - times[0])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
