"""faster-whisper wrapper with lazy load and graceful fallback."""
from __future__ import annotations
import io
import wave
from typing import Optional

import numpy as np

from .config import WHISPER_COMPUTE, WHISPER_DEVICE, WHISPER_MODEL
from .log import log


_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model
    from faster_whisper import WhisperModel
    log(f"stt: loading {WHISPER_MODEL} on {WHISPER_DEVICE} ({WHISPER_COMPUTE})")
    _model = WhisperModel(
        WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE
    )
    return _model


def transcribe_pcm16(pcm16: bytes, sample_rate: int = 16000) -> Optional[str]:
    """Transcribe raw s16le PCM mono audio. Returns text or None if empty."""
    if not pcm16:
        return None
    audio_i16 = np.frombuffer(pcm16, dtype=np.int16)
    if audio_i16.size == 0:
        return None
    audio = audio_i16.astype(np.float32) / 32768.0

    model = _get_model()
    segments, _info = model.transcribe(
        audio,
        language="en",
        vad_filter=False,  # our pipeline already does endpointing
        beam_size=1,       # speed over accuracy; base.en is small
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    log(f"stt: {text!r}")
    return text or None


def pcm16_to_wav_bytes(pcm16: bytes, sample_rate: int = 16000) -> bytes:
    """Utility for debug: wrap raw PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()
