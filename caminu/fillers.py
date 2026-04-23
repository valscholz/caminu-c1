"""Acknowledgement 'fillers' — tiny spoken interjections played while C1 is
thinking, so the user hears something within ~200 ms of finishing their turn
instead of a silent 1-2 seconds.

We pre-synthesize a pool of short phrases with Kokoro at startup, cache the
raw PCM, and play a random one through the same PulseAudio sink as regular
TTS. A filler is only played when the LLM hasn't produced its first content
token within FILLER_AFTER_MS — fast turns get no filler at all.
"""
from __future__ import annotations
import random
import subprocess
import threading
from typing import Optional

import numpy as np

from .config import (
    FILLER_PHRASES,
    FILLER_VOLUME_DB,
    KOKORO_SPEED,
    KOKORO_VOICE,
    PULSE_OUTPUT_SINK,
)
from .log import log


# Pre-synth cache: phrase -> (pcm_bytes, sample_rate)
_cache: dict[str, tuple[bytes, int]] = {}
_lock = threading.Lock()


def _apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    if gain_db == 0:
        return audio
    scale = 10 ** (gain_db / 20)
    return np.clip(audio * scale, -1.0, 1.0)


def preload() -> None:
    """Synthesize every filler phrase once and cache the PCM. Call once at
    agent startup after Kokoro is loaded.
    """
    from .tts import _get_tts  # lazy to avoid import cycle at module load
    tts = _get_tts()
    for phrase in FILLER_PHRASES:
        if phrase in _cache:
            continue
        try:
            samples, sr = tts.create(
                phrase, voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang="en-us"
            )
            samples = _apply_gain_db(samples.astype(np.float32), FILLER_VOLUME_DB)
            pcm = (samples * 32767.0).astype(np.int16).tobytes()
            _cache[phrase] = (pcm, sr)
        except Exception as e:
            log(f"fillers: failed to preload {phrase!r}: {e}")
    log(f"fillers: preloaded {len(_cache)} phrases")


def play_random() -> None:
    """Play one cached filler phrase, non-blocking as a fire-and-forget subprocess."""
    if not _cache:
        return
    with _lock:
        phrase = random.choice(list(_cache.keys()))
        pcm, sr = _cache[phrase]
    log(f"fillers: {phrase!r}")
    try:
        # fire-and-forget: we don't wait for it to finish
        subprocess.Popen(
            [
                "paplay",
                f"--device={PULSE_OUTPUT_SINK}",
                "--raw",
                "--format=s16le",
                f"--rate={sr}",
                "--channels=1",
            ],
            stdin=subprocess.PIPE,
        ).communicate(input=pcm, timeout=3)
    except Exception as e:
        log(f"fillers: play failed: {e}")
