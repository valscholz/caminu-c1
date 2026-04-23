"""One-shot frame grab from the OAK-D W.

Opens a DepthAI pipeline, warms up the sensor so AE/AWB converge, grabs
one frame, encodes to JPEG, returns base64. Pipeline closes immediately
after — we don't keep the camera hot between turns.
"""
import base64
import time
from typing import Optional

from .config import (
    CAMERA_WARMUP_S,
    CAMERA_CAPTURE_WIDTH,
    CAMERA_CAPTURE_HEIGHT,
    CAMERA_JPEG_QUALITY,
)
from .log import log


def grab_jpeg_b64() -> Optional[str]:
    """Capture one RGB frame and return base64-encoded JPEG. None on failure."""
    try:
        import cv2
        import depthai as dai
    except ImportError as e:
        log(f"camera: depthai/cv2 not available: {e}")
        return None

    try:
        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.Camera).build(
            boardSocket=dai.CameraBoardSocket.CAM_A
        )
        out = cam.requestOutput(
            size=(CAMERA_CAPTURE_WIDTH, CAMERA_CAPTURE_HEIGHT),
            type=dai.ImgFrame.Type.NV12,
        )
        q = out.createOutputQueue(maxSize=4, blocking=False)
        pipeline.start()

        last = None
        t0 = time.time()
        while time.time() - t0 < CAMERA_WARMUP_S:
            f = q.tryGet()
            if f is not None:
                last = f
        if last is None:
            last = q.get()
        img = last.getCvFrame()
        pipeline.stop()
    except Exception as e:
        log(f"camera: pipeline failed: {e}")
        return None

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, CAMERA_JPEG_QUALITY])
    if not ok:
        log("camera: JPEG encode failed")
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")
