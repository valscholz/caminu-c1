"""Acknowledgement fillers — tiny spoken interjections played while C1 is
thinking. Cached PCM is synthesized once at startup; playback is as fast as
possible (no subprocess spawn per call — we keep one aplay alive).
"""
from __future__ import annotations
import atexit
import random
import subprocess
import threading
from typing import Optional

import numpy as np

from .config import (
    ALSA_OUTPUT_DEVICE,
    FILLER_PHRASES,
    FILLER_VOLUME_DB,
    KOKORO_SPEED,
    KOKORO_VOICE,
)
from .log import log


# Pre-synth cache: phrase -> (pcm_bytes, sample_rate)
_cache: dict[str, tuple[bytes, int]] = {}
_cache_lock = threading.Lock()

# Persistent aplay process (one per session). Reused across calls so we
# avoid the 100-200 ms subprocess startup penalty per filler.
_player: Optional[subprocess.Popen] = None
_player_sr: Optional[int] = None
_player_lock = threading.Lock()


def _apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    if gain_db == 0:
        return audio
    scale = 10 ** (gain_db / 20)
    return np.clip(audio * scale, -1.0, 1.0)


def _spawn_player(sr: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "aplay",
            "-q",
            "-D", ALSA_OUTPUT_DEVICE,
            "-t", "raw",
            "-f", "S16_LE",
            "-r", str(sr),
            "-c", "1",
        ],
        stdin=subprocess.PIPE,
    )


def _get_player(sr: int) -> subprocess.Popen:
    """Return a live aplay process matching sample rate `sr`.
    Respawns if the prior process died or rate changed."""
    global _player, _player_sr
    with _player_lock:
        if (
            _player is None
            or _player.poll() is not None
            or _player_sr != sr
            or _player.stdin is None
        ):
            if _player is not None:
                try:
                    if _player.stdin:
                        _player.stdin.close()
                    _player.terminate()
                except Exception:
                    pass
            _player = _spawn_player(sr)
            _player_sr = sr
        return _player


def preload() -> None:
    """Synthesize every filler phrase once and cache the PCM."""
    from .tts import _get_tts
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

    # Deliberately do NOT pre-warm a persistent aplay here — see comment in
    # play_random(). Conflicts with the main TTS pipeline's aplay.


def play_random() -> None:
    """Play one cached filler phrase. Spawns a short-lived aplay, pipes the
    pre-synthesized PCM, closes. We deliberately don't keep aplay alive
    across calls because the main TTS path (SentenceSpeaker) also opens
    the same ALSA device; a persistent filler aplay would block it.
    Subprocess spawn is ~100 ms — noticeable but unavoidable without a
    shared-device audio server.
    """
    if not _cache:
        return
    with _cache_lock:
        phrase = random.choice(list(_cache.keys()))
        pcm, sr = _cache[phrase]
    log(f"fillers: {phrase!r}")
    try:
        proc = _spawn_player(sr)
        assert proc.stdin is not None
        proc.stdin.write(pcm)
        proc.stdin.close()
        # Wait for the filler to finish so the device is free before the
        # main TTS pipeline tries to grab it. Small blocking cost (~100-200 ms)
        # but prevents device-busy on the next sentence.
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    except (BrokenPipeError, ValueError, OSError) as e:
        log(f"fillers: write failed: {e}")


def _shutdown() -> None:
    global _player
    with _player_lock:
        if _player is not None:
            try:
                if _player.stdin:
                    _player.stdin.close()
                _player.wait(timeout=1)
            except Exception:
                try:
                    _player.kill()
                except Exception:
                    pass
            _player = None


atexit.register(_shutdown)
