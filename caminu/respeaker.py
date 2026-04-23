"""ReSpeaker Mic Array v3 tuning interface (USB HID control).

The XMOS XVF3000 on the v3 exposes runtime parameters over USB control
transfers — including DOA (direction of arrival) in degrees (0-359), VAD
state, speech detection, and AEC parameters.

We use it for DOA-gated follow-up mode: record the angle when the user
says "Hey Jarvis", then during the follow-up window reject audio whose
DOA differs by more than FOLLOW_UP_DOA_TOLERANCE_DEG. Stops ambient
audio (TV, laptop video) from triggering follow-ups.

Based on Seeed's reference: https://github.com/respeaker/usb_4_mic_array

Thread-safe: a lock serializes USB transfers.
"""
from __future__ import annotations
import struct
import threading
from typing import Optional

from .log import log

# ReSpeaker Mic Array v3 USB identifiers
VENDOR_ID = 0x2886
PRODUCT_ID = 0x0018

# A small subset of XVF3000 parameters we actually use. Each entry is:
#   name: (id, offset, dtype)  where dtype is 'int' or 'float'.
# Full table lives in Seeed's respeaker_python_library PARAMETERS dict;
# we copy only what we need to avoid the dependency.
PARAMS = {
    "DOAANGLE":   (21, 0, "int"),    # 0..359 degrees
    "VOICEACTIVITY": (19, 32, "int"),  # 1 if voice currently detected
    "SPEECHDETECTED": (19, 22, "int"),  # 1 if speech ever detected
}


class _ReSpeakerTuning:
    """Lazy singleton. First access opens the USB device; subsequent calls
    reuse the handle. Silently no-ops if the device isn't attached or
    python-usb isn't installed — DOA gating becomes a passthrough."""

    def __init__(self) -> None:
        self._dev = None
        self._lock = threading.Lock()
        self._tried = False

    def _open(self) -> None:
        if self._tried:
            return
        self._tried = True
        try:
            import usb.core  # type: ignore[import-not-found]
        except ImportError:
            log("respeaker: python-usb not installed — DOA disabled")
            return
        try:
            dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
            if dev is None:
                log("respeaker: device not found on USB — DOA disabled")
                return
            self._dev = dev
            log("respeaker: tuning interface opened")
        except Exception as e:
            log(f"respeaker: open failed ({e}) — DOA disabled")

    def _read_param(self, key: str) -> Optional[int]:
        with self._lock:
            if self._dev is None:
                self._open()
            if self._dev is None:
                return None
            pid, offset, dtype = PARAMS[key]
            # 0xC0 = vendor IN, 0x40 = vendor OUT (for setters; unused here)
            # Control transfer returns 8 bytes: int32 value, int32 type (0=int, 1=float).
            try:
                data = self._dev.ctrl_transfer(
                    0x80 | 0x40,   # USB IN | Vendor
                    0,             # bRequest
                    0x80 | offset, # wValue: read + offset
                    pid,           # wIndex: parameter id
                    8,             # wLength
                    10000,         # timeout ms
                )
            except Exception as e:
                log(f"respeaker: ctrl_transfer failed on {key}: {e}")
                return None
            if not data or len(data) < 8:
                return None
            if dtype == "int":
                return struct.unpack("i", bytes(data[:4]))[0]
            # we don't need float params today
            return None

    def doa_angle(self) -> Optional[int]:
        """Return current DOA in degrees (0..359), or None if unavailable."""
        return self._read_param("DOAANGLE")

    def voice_active(self) -> Optional[bool]:
        v = self._read_param("VOICEACTIVITY")
        return None if v is None else bool(v)


_singleton: Optional[_ReSpeakerTuning] = None


def get_tuning() -> _ReSpeakerTuning:
    global _singleton
    if _singleton is None:
        _singleton = _ReSpeakerTuning()
    return _singleton


def doa() -> Optional[int]:
    """Shortcut: current DOA, None if unavailable."""
    return get_tuning().doa_angle()


def angle_diff(a: int, b: int) -> int:
    """Smallest angular distance between two 0..359 angles. 0..180."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d
