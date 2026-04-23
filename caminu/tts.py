"""Kokoro TTS -> PulseAudio paplay.

Applies a digital pre-gain so the small Monk Makes speaker is loud enough,
then streams to the ReSpeaker's analog output sink.
"""
from __future__ import annotations
import subprocess
from typing import Optional

import numpy as np

from .config import (
    KOKORO_DIR,
    KOKORO_PREGAIN_DB,
    KOKORO_SPEED,
    KOKORO_VOICE,
    PULSE_OUTPUT_SINK,
)
from .log import log


_tts = None


def _get_tts():
    global _tts
    if _tts is not None:
        return _tts
    from kokoro_onnx import Kokoro
    model_path = KOKORO_DIR / "kokoro-v1.0.onnx"
    voices_path = KOKORO_DIR / "voices-v1.0.bin"
    log(f"tts: loading Kokoro from {model_path}")
    _tts = Kokoro(str(model_path), str(voices_path))
    return _tts


def _apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    if gain_db == 0:
        return audio
    scale = 10 ** (gain_db / 20)
    return np.clip(audio * scale, -1.0, 1.0)


def speak(text: str) -> None:
    """Synthesize `text` with Kokoro and play through PULSE_OUTPUT_SINK."""
    text = text.strip()
    if not text:
        return
    log(f"tts: speak {text!r}")

    tts = _get_tts()
    samples, sr = tts.create(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang="en-us")
    samples = _apply_gain_db(samples.astype(np.float32), KOKORO_PREGAIN_DB)
    pcm16 = (samples * 32767.0).astype(np.int16).tobytes()

    cmd = [
        "paplay",
        f"--device={PULSE_OUTPUT_SINK}",
        "--raw",
        "--format=s16le",
        f"--rate={sr}",
        "--channels=1",
    ]
    subprocess.run(cmd, input=pcm16, check=False)
