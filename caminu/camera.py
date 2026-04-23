"""OAK-D W camera: persistent pipeline + on-demand frame grab.

A background thread keeps a DepthAI pipeline running continuously and
holds the latest frame in memory. Capture now is just "pick the most
recent frame and JPEG-encode it" — no pipeline open/close per call.

Eliminates the 5-8s per-vision-turn cost of building the pipeline and
waiting for AE/AWB to converge.
"""
from __future__ import annotations
import base64
import threading
import time
from typing import Optional

import numpy as np

from .config import (
    CAMERA_CAPTURE_WIDTH,
    CAMERA_CAPTURE_HEIGHT,
    CAMERA_JPEG_QUALITY,
)
from .log import log


_worker: Optional["_CameraWorker"] = None
_lock = threading.Lock()


class _CameraWorker:
    """Background thread that keeps a DepthAI pipeline open and publishes
    the latest RGB frame into a thread-safe slot.
    """

    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error: Optional[str] = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="oakd-cam")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        # don't join — thread is daemon; process exit will clean up

    def get_frame(self, timeout_s: float = 5.0) -> Optional[np.ndarray]:
        """Block until a frame is available, then return the most recent one."""
        if not self._ready.wait(timeout=timeout_s):
            return None
        with self._frame_lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    @property
    def error(self) -> Optional[str]:
        return self._error

    def _run(self) -> None:
        try:
            import depthai as dai
        except ImportError as e:
            self._error = f"depthai import failed: {e}"
            log(f"camera: {self._error}")
            return

        try:
            pipeline = dai.Pipeline()
            cam = pipeline.create(dai.node.Camera).build(
                boardSocket=dai.CameraBoardSocket.CAM_A
            )
            out = cam.requestOutput(
                size=(CAMERA_CAPTURE_WIDTH, CAMERA_CAPTURE_HEIGHT),
                type=dai.ImgFrame.Type.NV12,
            )
            q = out.createOutputQueue(maxSize=2, blocking=False)
            pipeline.start()
            log("camera: pipeline running, warming up")
        except Exception as e:
            self._error = f"pipeline init failed: {e}"
            log(f"camera: {self._error}")
            return

        warmup_deadline = time.time() + 3.0
        try:
            while not self._stop.is_set():
                f = q.tryGet()
                if f is not None:
                    img = f.getCvFrame()
                    with self._frame_lock:
                        self._frame = img
                    if not self._ready.is_set() and time.time() > warmup_deadline:
                        self._ready.set()
                        log("camera: ready (warmup complete)")
                else:
                    time.sleep(0.01)
        except Exception as e:
            self._error = f"pipeline crashed: {e}"
            log(f"camera: {self._error}")
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass


def start() -> None:
    """Start the background camera worker if not already running. Non-blocking."""
    global _worker
    with _lock:
        if _worker is not None:
            return
        _worker = _CameraWorker()
        _worker.start()


def stop() -> None:
    global _worker
    with _lock:
        if _worker is not None:
            _worker.stop()
            _worker = None


def grab_jpeg_b64() -> Optional[str]:
    """Return the latest frame as a base64 JPEG. Starts the worker on demand
    if it wasn't pre-started. Returns None on failure.
    """
    global _worker
    if _worker is None:
        start()
    assert _worker is not None
    if _worker.error:
        log(f"camera: worker in error state: {_worker.error}")
        return None

    img = _worker.get_frame(timeout_s=5.0)
    if img is None:
        log("camera: no frame available (worker not warm yet?)")
        return None

    try:
        import cv2
    except ImportError as e:
        log(f"camera: cv2 unavailable: {e}")
        return None
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, CAMERA_JPEG_QUALITY])
    if not ok:
        log("camera: JPEG encode failed")
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")
