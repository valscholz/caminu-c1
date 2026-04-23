"""Speech-to-text. Parakeet TDT 0.6B on CUDA (primary), tiny-Whisper CPU fallback.

Parakeet-TDT is NVIDIA's RNN-Transducer ASR. It scales linearly in audio
length, runs on GPU via ONNX Runtime, and transcribes a 3s utterance in
~100ms warm. Huge upgrade over Whisper base/tiny on CPU, which stalls on
long utterances.

If Parakeet fails to load (VRAM pressure, missing ONNX, whatever), we
fall back to Whisper CPU int8 so the agent still functions.
"""
from __future__ import annotations
import os
from typing import Optional

import numpy as np

from .config import STT_BACKEND, WHISPER_COMPUTE, WHISPER_DEVICE, WHISPER_MODEL
from .log import log

# Disable TensorRT entirely: Parakeet's preprocessor has dynamic shapes TRT
# can't handle, and TRT's eager arena allocation fragments CUDA memory so
# the Parakeet encoder can't even load when Gemma is resident.
os.environ.setdefault("ORT_LOG_LEVEL", "3")


# Cap on input audio. Parakeet handles 20s+ fine (linear in length);
# Whisper fallback still benefits from a cap via VAD chunking inside the
# model.
MAX_AUDIO_S = 20.0

_model = None          # either a Parakeet onnx_asr model or a faster-whisper WhisperModel
_backend: str = "?"    # "parakeet" | "whisper"


# Whisper hallucinations to drop on short/silent clips.
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
    normed = text.lower().strip(" .!?")
    if not normed:
        return True
    if normed in _WHISPER_HALLUCINATIONS and audio_duration_s < 3.0:
        return True
    if audio_duration_s < 1.0 and len(normed.split()) <= 1:
        return True
    return False


def _load_parakeet():
    import onnx_asr
    log("stt: loading Parakeet TDT 0.6B on CUDA (+ CPU fallback, skipping TensorRT)")
    return onnx_asr.load_model(
        "nemo-parakeet-tdt-0.6b-v3",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def _load_whisper():
    from faster_whisper import WhisperModel
    log(f"stt: loading Whisper {WHISPER_MODEL} on {WHISPER_DEVICE} ({WHISPER_COMPUTE})")
    return WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)


def _get_model():
    global _model, _backend
    if _model is not None:
        return _model
    if STT_BACKEND == "parakeet":
        try:
            _model = _load_parakeet()
            _backend = "parakeet"
            return _model
        except Exception as e:
            log(f"stt: Parakeet load failed ({type(e).__name__}: {str(e)[:120]}); falling back to Whisper")
    _model = _load_whisper()
    _backend = "whisper"
    return _model


def transcribe_pcm16(pcm16: bytes, sample_rate: int = 16000) -> Optional[str]:
    """Transcribe raw s16le PCM mono audio. Returns text or None if empty
    or if it looks like a Whisper hallucination on silence."""
    if not pcm16:
        return None
    audio_i16 = np.frombuffer(pcm16, dtype=np.int16)
    if audio_i16.size == 0:
        return None
    audio = audio_i16.astype(np.float32) / 32768.0

    max_samples = int(MAX_AUDIO_S * sample_rate)
    if len(audio) > max_samples:
        log(f"stt: clipping {len(audio)/sample_rate:.1f}s -> {MAX_AUDIO_S:.0f}s")
        audio = audio[-max_samples:]
    duration_s = len(audio) / sample_rate

    model = _get_model()

    if _backend == "parakeet":
        # onnx-asr accepts a numpy array of float32 mono samples at 16 kHz.
        text = model.recognize(audio).strip()
    else:
        # faster-whisper path
        segments, _info = model.transcribe(
            audio,
            language="en",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 200},
            beam_size=1,
        )
        text = " ".join(s.text.strip() for s in segments).strip()

    if _is_hallucination(text, duration_s):
        log(f"stt: dropping hallucination {text!r} (dur={duration_s:.2f}s)")
        return None

    log(f"stt[{_backend}]: {text!r}")
    return text or None
