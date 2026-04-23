"""Kokoro TTS -> PulseAudio paplay.

Two ways to use:
- speak(text): one-shot blocking synthesis + playback.
- SentenceSpeaker(): stream-friendly. Feed token chunks via feed(); it
  emits complete sentences to Kokoro as soon as they land, writing the
  resulting PCM to a persistent paplay subprocess. Call flush() at
  end-of-stream and close() when the turn is fully over.

Pre-gain is applied before int16 conversion so the small Monk Makes
speaker gets meaningful volume without clipping.
"""
from __future__ import annotations
import re
import subprocess
import threading
from queue import Queue
from typing import Optional

import numpy as np

from .config import (
    ALSA_OUTPUT_DEVICE,
    KOKORO_DIR,
    KOKORO_MODEL_FILENAME,
    KOKORO_PREGAIN_DB,
    KOKORO_SPEED,
    KOKORO_USE_CUDA,
    KOKORO_VOICE,
    KOKORO_VOICES_FILENAME,
)
from .log import log


_tts = None


def _get_tts():
    global _tts
    if _tts is not None:
        return _tts
    from kokoro_onnx import Kokoro
    model_path = KOKORO_DIR / KOKORO_MODEL_FILENAME
    voices_path = KOKORO_DIR / KOKORO_VOICES_FILENAME

    providers = None
    if KOKORO_USE_CUDA:
        try:
            import onnxruntime
            available = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            providers = None

    log(f"tts: loading Kokoro from {model_path} (providers={providers or 'cpu default'})")
    try:
        # Newer kokoro-onnx accepts a providers kwarg; older builds don't.
        _tts = Kokoro(str(model_path), str(voices_path), providers=providers) \
            if providers is not None else Kokoro(str(model_path), str(voices_path))
    except TypeError:
        # kwarg not supported in this kokoro-onnx version; manually set the session
        _tts = Kokoro(str(model_path), str(voices_path))
        if providers is not None:
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(str(model_path), providers=providers)
                _tts.sess = sess  # type: ignore[attr-defined]
                log(f"tts: swapped Kokoro session to providers={providers}")
            except Exception as e:
                log(f"tts: CUDA provider swap failed, staying on CPU: {e}")
    return _tts


def _apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    if gain_db == 0:
        return audio
    scale = 10 ** (gain_db / 20)
    return np.clip(audio * scale, -1.0, 1.0)


def _synthesize(text: str) -> tuple[bytes, int]:
    """Kokoro synthesize -> (int16 mono PCM bytes, sample_rate)."""
    tts = _get_tts()
    samples, sr = tts.create(
        text, voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang="en-us"
    )
    samples = _apply_gain_db(samples.astype(np.float32), KOKORO_PREGAIN_DB)
    return (samples * 32767.0).astype(np.int16).tobytes(), sr


def _aplay_process(sample_rate: int) -> subprocess.Popen:
    """aplay reads raw s16le mono from stdin and writes to the ReSpeaker ALSA
    device directly. We bypass PulseAudio for output because PA loses the
    card from its sink list once sounddevice has the ALSA device open
    exclusively for input."""
    return subprocess.Popen(
        [
            "aplay",
            "-q",                   # quiet: no startup banner
            "-D", ALSA_OUTPUT_DEVICE,
            "-t", "raw",
            "-f", "S16_LE",
            "-r", str(sample_rate),
            "-c", "1",
        ],
        stdin=subprocess.PIPE,
    )


# Backwards compat alias — older code called this _paplay_process.
_paplay_process = _aplay_process


# ---------------- blocking one-shot (retained for convenience) -----------------

def speak(text: str) -> None:
    """Synthesize `text` with Kokoro and play through the speaker. Blocking."""
    text = text.strip()
    if not text:
        return
    log(f"tts: speak {text!r}")
    pcm, sr = _synthesize(text)
    proc = _paplay_process(sr)
    assert proc.stdin is not None
    proc.stdin.write(pcm)
    proc.stdin.close()
    proc.wait()


# ---------------- streaming speaker --------------------------------------------

# Split on sentence-final punctuation followed by a boundary (whitespace or EOL).
# We keep the punctuation with the preceding sentence by using a lookbehind-ish trick.
_SENT_END = re.compile(r"([.!?][\"')\]]*\s+|\n+)")

# Avoid tiny "sentences" like single-letter abbreviations — wait for at least this
# many non-whitespace characters before we consider emitting a sentence.
_MIN_SENTENCE_CHARS = 6


class SentenceSpeaker:
    """Buffers streamed tokens, emits sentences to Kokoro -> paplay as they land.

    A worker thread owns the Kokoro/paplay pipeline so the caller (LLM
    callback) never blocks on synthesis.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._queue: Queue[Optional[str]] = Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._proc: Optional[subprocess.Popen] = None
        self._proc_sr: Optional[int] = None
        self._started = False
        self._first_audio_logged = False

    # ---- producer side (called from the LLM token callback) ----

    def feed(self, chunk: str) -> None:
        """Append streamed tokens; emit any completed sentences."""
        if not self._started:
            self._started = True
            self._worker.start()

        self._buffer += chunk
        while True:
            m = _SENT_END.search(self._buffer)
            if not m:
                return
            end = m.end()
            sentence = self._buffer[:end].strip()
            self._buffer = self._buffer[end:]
            if len(sentence) >= _MIN_SENTENCE_CHARS:
                self._queue.put(sentence)

    def flush(self) -> None:
        """Emit any trailing partial text as a final 'sentence'."""
        tail = self._buffer.strip()
        self._buffer = ""
        if tail:
            self._queue.put(tail)

    def close(self) -> None:
        """Signal end-of-turn; block until all audio has played."""
        if not self._started:
            return
        self._queue.put(None)  # sentinel
        self._worker.join()
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait()
            except Exception:
                pass
            self._proc = None
            self._proc_sr = None

    # ---- consumer side (worker thread) ----

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                pcm, sr = _synthesize(item)
            except Exception as e:
                log(f"tts: synth failed on {item!r}: {e}")
                continue

            if self._proc is None or self._proc_sr != sr:
                # Close any prior process (different sample rate) first.
                if self._proc is not None:
                    try:
                        if self._proc.stdin:
                            self._proc.stdin.close()
                        self._proc.wait()
                    except Exception:
                        pass
                self._proc = _paplay_process(sr)
                self._proc_sr = sr

            if not self._first_audio_logged:
                log(f"tts: first audio (sentence={item!r})")
                self._first_audio_logged = True

            assert self._proc.stdin is not None
            try:
                self._proc.stdin.write(pcm)
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                log("tts: paplay pipe broken — respawning on next sentence")
                self._proc = None
                self._proc_sr = None
