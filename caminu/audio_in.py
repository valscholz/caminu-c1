"""ReSpeaker capture + wake word detection + VAD-based endpointing.

Opens a sounddevice InputStream on the ReSpeaker's 6-channel input, extracts
channel 0 (the DSP-processed mono output), feeds frames through openWakeWord
for trigger detection, and records until VAD silence endpoints the turn.
"""
from __future__ import annotations
import collections
import queue
import threading
import time
from typing import Optional

import numpy as np

from .config import (
    MAX_UTTERANCE_S,
    MIC_BLOCK_MS,
    MIC_CHANNELS,
    MIC_DEVICE_SUBSTRING,
    MIC_SAMPLE_RATE,
    MIC_USE_CHANNEL,
    VAD_AGGRESSIVENESS,
    VAD_MIN_SPEECH_MS,
    VAD_SILENCE_END_MS,
    WAKE_COOLDOWN_S,
    WAKE_MODEL,
    WAKE_THRESHOLD,
)
from .log import log


def _release_pa_input(substring: str) -> None:
    """Ask PulseAudio to suspend any input source matching substring.

    PA normally holds the ALSA device exclusively; suspending it releases
    the hardware so we can open hw:0,0 directly and preserve the native
    channel ordering (ch0 = DSP-processed beamformed mono) that we need
    for STT.
    """
    import subprocess
    try:
        out = subprocess.check_output(["pactl", "list", "short", "sources"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and substring.lower() in parts[1].lower():
            name = parts[1]
            log(f"audio_in: suspending PA source {name}")
            subprocess.run(["pactl", "suspend-source", name, "1"], check=False)


def _find_mic_index() -> int:
    """Resolve a mic device index for sounddevice pointing at ReSpeaker hw:0,0.

    PulseAudio holds the ReSpeaker exclusively by default (PaErrorCode -9985).
    We release PA's grip before opening the device so we get the native 6ch
    ALSA ordering where channel 0 is the DSP-processed beamformed output.
    """
    _release_pa_input("SEEED")
    import sounddevice as sd
    devices = list(enumerate(sd.query_devices()))
    for i, dev in devices:
        if (
            dev.get("max_input_channels", 0) >= MIC_CHANNELS
            and MIC_DEVICE_SUBSTRING.lower() in dev.get("name", "").lower()
        ):
            log(f"audio_in: mic device [{i}] {dev['name']}")
            return i
    listing = "\n".join(f"  [{i}] {d['name']} (in={d['max_input_channels']})" for i, d in devices)
    raise RuntimeError(
        f"No mic matching '{MIC_DEVICE_SUBSTRING}'. Available:\n{listing}"
    )


class AudioInput:
    """Runs the mic stream and exposes a blocking wake+record API."""

    def __init__(self) -> None:
        import sounddevice as sd
        from openwakeword.model import Model as WakeWordModel
        import webrtcvad

        self._sd = sd
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._ww = WakeWordModel(wakeword_models=[WAKE_MODEL])
        self._frames_per_block = int(MIC_SAMPLE_RATE * MIC_BLOCK_MS / 1000)

        self._mic_index = _find_mic_index()
        self._stream: Optional[sd.InputStream] = None
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self._stop_wake = threading.Event()

        # webrtcvad supports 10/20/30 ms frames at 8/16/32/48 kHz; we use 20 ms @ 16 kHz
        assert MIC_BLOCK_MS in (10, 20, 30), "webrtcvad requires 10/20/30 ms blocks"

    def start(self) -> None:
        def cb(indata, frames, time_info, status):
            if status:
                # Overruns are common during TTS; ignore
                pass
            # indata shape: (frames, channels); extract our channel as int16 bytes
            ch = indata[:, MIC_USE_CHANNEL]
            try:
                self._q.put_nowait(ch.copy())
            except queue.Full:
                pass

        self._stream = self._sd.InputStream(
            device=self._mic_index,
            channels=MIC_CHANNELS,
            samplerate=MIC_SAMPLE_RATE,
            dtype="int16",
            blocksize=self._frames_per_block,
            callback=cb,
        )
        self._stream.start()
        log("audio_in: stream started")

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _drain_queue(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def _next_block(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    # ---------------------- public API ----------------------

    def wait_for_wake_word(self) -> None:
        """Block until the wake word fires. Drains any backlog first."""
        self._drain_queue()
        self._ww.reset()
        log(f"audio_in: waiting for wake word ({WAKE_MODEL})...")
        last_trigger = 0.0
        while not self._stop_wake.is_set():
            block = self._next_block()
            if block is None:
                continue
            scores = self._ww.predict(block)
            score = scores.get(WAKE_MODEL, 0.0)
            if score >= WAKE_THRESHOLD and time.time() - last_trigger > WAKE_COOLDOWN_S:
                log(f"audio_in: WAKE (score={score:.2f})")
                last_trigger = time.time()
                return

    def record_utterance(self, prebuffer: bytes = b"") -> bytes:
        """Record PCM until VAD silence endpoint or MAX_UTTERANCE_S.

        If `prebuffer` is supplied (e.g. from wait_for_speech when it detected
        the user already starting to speak), it's prepended to the captured
        audio so the first syllables aren't lost.
        """
        log("audio_in: recording utterance...")
        self._drain_queue()
        chunks: list[bytes] = [prebuffer] if prebuffer else []
        speech_ms = 0
        silence_ms = 0
        total_ms = 0
        speaking_started = bool(prebuffer)
        if prebuffer:
            # Assume the prebuffer already contains a chunk of speech, so we're
            # past the min-speech threshold.
            speech_ms = VAD_MIN_SPEECH_MS

        while total_ms < MAX_UTTERANCE_S * 1000:
            block = self._next_block()
            if block is None:
                continue
            pcm = block.astype(np.int16).tobytes()
            chunks.append(pcm)
            is_speech = self._vad.is_speech(pcm, MIC_SAMPLE_RATE)

            if is_speech:
                speech_ms += MIC_BLOCK_MS
                silence_ms = 0
                speaking_started = speaking_started or (speech_ms >= VAD_MIN_SPEECH_MS)
            else:
                silence_ms += MIC_BLOCK_MS
                if speaking_started and silence_ms >= VAD_SILENCE_END_MS:
                    break
            total_ms += MIC_BLOCK_MS

        pcm_all = b"".join(chunks)
        log(f"audio_in: captured {total_ms} ms (speech {speech_ms} ms)")
        return pcm_all

    def wait_for_speech(self, window_s: float) -> Optional[bytes]:
        """Listen for up to `window_s` seconds waiting for the user to start
        speaking. Returns the initial PCM chunk that triggered detection (so
        callers can prepend it before calling record_utterance()), or None on
        timeout.

        Used by follow-up mode: after C1 finishes speaking, we keep the mic
        open briefly — if the user starts talking, we treat it as the next
        turn without a new wake-word.
        """
        self._drain_queue()
        deadline = time.time() + window_s
        speech_ms = 0
        while time.time() < deadline:
            block = self._next_block(timeout=min(1.0, max(0.1, deadline - time.time())))
            if block is None:
                continue
            pcm = block.astype(np.int16).tobytes()
            if self._vad.is_speech(pcm, MIC_SAMPLE_RATE):
                speech_ms += MIC_BLOCK_MS
                if speech_ms >= VAD_MIN_SPEECH_MS:
                    log(f"audio_in: follow-up speech detected after {speech_ms} ms")
                    return pcm
            else:
                # a non-speech block resets the counter so random clicks don't accumulate
                speech_ms = 0
        log(f"audio_in: follow-up window expired ({window_s:.1f}s)")
        return None
