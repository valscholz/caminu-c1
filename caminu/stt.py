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


# Whisper base.en's well-known hallucinations on short / silent / noisy audio.
# These come from its training data (YouTube transcripts). If the audio is
# short and STT returns one of these, the user almost certainly didn't say it.
_WHISPER_HALLUCINATIONS = {
    "thanks for watching",
    "thanks for watching!",
    "thank you for watching",
    "thank you for watching.",
    "please subscribe",
    "please like and subscribe",
    "and we'll see you in the next one",
    "and we'll see you in the next one.",
    "see you in the next one",
    "subscribe to my channel",
    "thanks for watching bye",
    "you",
    "bye",
    ".",
}


def _is_hallucination(text: str, audio_duration_s: float) -> bool:
    """Heuristic: reject common Whisper hallucinations on short / nothing clips."""
    normed = text.lower().strip(" .!?")
    if not normed:
        return True
    # If it matches a known hallucination phrase and the clip was short-ish
    if normed in _WHISPER_HALLUCINATIONS and audio_duration_s < 3.0:
        return True
    # Very short single-word "you" / "bye" on a <1s clip is almost always noise
    if audio_duration_s < 1.0 and len(normed.split()) <= 1:
        return True
    return False


MAX_AUDIO_S = 10.0  # cap on input audio. Longer clips send exponentially
# worse on CPU int8 Whisper; a 15s noisy clip can take 45s+ to decode
# while the user is already annoyed. We truncate to the last MAX_AUDIO_S
# so the user perceives "didn't quite catch everything" instead of a
# minute-long silence.


def transcribe_pcm16(pcm16: bytes, sample_rate: int = 16000) -> Optional[str]:
    """Transcribe raw s16le PCM mono audio. Returns text or None if empty
    or if the result looks like a Whisper hallucination on silence."""
    if not pcm16:
        return None
    audio_i16 = np.frombuffer(pcm16, dtype=np.int16)
    if audio_i16.size == 0:
        return None
    audio = audio_i16.astype(np.float32) / 32768.0

    # Truncate to MAX_AUDIO_S (keep the most recent chunk — the end of a
    # long recording is typically where the real speech was).
    max_samples = int(MAX_AUDIO_S * sample_rate)
    if len(audio) > max_samples:
        log(f"stt: clipping {len(audio)/sample_rate:.1f}s -> {MAX_AUDIO_S:.0f}s")
        audio = audio[-max_samples:]
    duration_s = len(audio) / sample_rate

    model = _get_model()
    segments, _info = model.transcribe(
        audio,
        language="en",
        vad_filter=False,  # our pipeline already does endpointing
        beam_size=1,       # speed over accuracy; base.en is small
    )
    text = " ".join(s.text.strip() for s in segments).strip()

    if _is_hallucination(text, duration_s):
        log(f"stt: dropping hallucination {text!r} (dur={duration_s:.2f}s)")
        return None

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
