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
    FOLLOW_UP_MIN_SPEECH_MS,
    MAX_UTTERANCE_S,
    MIC_BLOCK_MS,
    MIC_CHANNELS,
    MIC_DEVICE_SUBSTRING,
    MIC_SAMPLE_RATE,
    MIC_USE_CHANNEL,
    SEMANTIC_ENDPOINT_ENABLED,
    SEMANTIC_MIN_AUDIO_MS,
    SEMANTIC_POLL_MS,
    SEMANTIC_STABLE_MS,
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
        import webrtcvad
        from .config import WAKE_MODE

        self._sd = sd
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self._wake_mode = WAKE_MODE
        # Lazy-load openWakeWord only if we're actually going to use it.
        # Saves ~150 MB RAM in VAD-only mode.
        if WAKE_MODE == "wake_word":
            from openwakeword.model import Model as WakeWordModel
            self._ww = WakeWordModel(wakeword_models=[WAKE_MODEL])
        else:
            self._ww = None
        self._frames_per_block = int(MIC_SAMPLE_RATE * MIC_BLOCK_MS / 1000)

        self._mic_index = _find_mic_index()
        self._stream: Optional[sd.InputStream] = None
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self._stop_wake = threading.Event()
        self._muted = False
        # DOA of the last wake word, used for gated follow-up.
        self.last_wake_doa: Optional[int] = None

        # webrtcvad supports 10/20/30 ms frames at 8/16/32/48 kHz; we use 20 ms @ 16 kHz
        assert MIC_BLOCK_MS in (10, 20, 30), "webrtcvad requires 10/20/30 ms blocks"

    def start(self) -> None:
        def cb(indata, frames, time_info, status):
            if status:
                # Overruns are common during TTS; ignore
                pass
            # Muted: drop audio completely — no wake, no follow-up trigger,
            # no STT feed. Prevents feedback loops where Kokoro's output
            # leaks into the mic and the agent hears itself.
            if self._muted:
                return
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

    def mute_input(self, muted: bool) -> None:
        """Drop all incoming audio while muted. Called during TTS playback
        so C1 doesn't hear her own voice echoing back. The stream stays
        open (cheap), we just discard frames at the callback."""
        self._muted = muted
        if muted:
            self._drain_queue()

    def watch_for_bargein(
        self,
        stop: "threading.Event",
        abort: "threading.Event",
        min_speech_ms: int = 400,
        min_rms: float = 60.0,
    ) -> bytes:
        """Run in a worker thread during TTS playback. Detects if the user
        starts speaking while C1 is mid-reply.

        Listens on the same mic stream. If it sees `min_speech_ms` of
        sustained speech above `min_rms`, sets `abort` to signal the LLM
        and TTS to stop, and returns the captured PCM so main.py can
        immediately continue recording the user's interrupting utterance.

        Note: C1's own voice leaks into the ReSpeaker. XVF3000 AEC reduces
        but doesn't eliminate it. We rely on (a) the user speaking louder
        than the echo and (b) 400 ms of sustained real speech to avoid
        false triggers. If echo becomes a problem, add XVF3000
        voice_active check here.

        Returns b"" if `stop` is set (C1 finished speaking naturally
        before any barge-in).
        """
        self._drain_queue()
        speech_ms = 0
        peak_rms = 0.0
        chunks: list[bytes] = []
        max_seen_rms = 0.0
        n_blocks = 0
        # Cache XVF3000 voice-active readings: checked every ~100 ms
        # (USB HID round-trip, not per-block). The chip's voice detector
        # distinguishes "human speech" from "my own output that AEC half-
        # suppressed", which raw RMS can't.
        last_va_check_ms = -100
        chip_voice_active = False
        try:
            from . import respeaker
            tuning = respeaker.get_tuning()
        except Exception:
            tuning = None

        while not stop.is_set():
            block = self._next_block(timeout=0.1)
            if block is None:
                continue
            n_blocks += 1
            pcm = block.astype(np.int16).tobytes()
            is_speech = self._vad.is_speech(pcm, MIC_SAMPLE_RATE)
            block_rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
            max_seen_rms = max(max_seen_rms, block_rms)

            # Poll the chip's voice-active bit every ~100 ms.
            cur_ms = n_blocks * MIC_BLOCK_MS
            if tuning is not None and cur_ms - last_va_check_ms >= 100:
                va = tuning.voice_active()
                chip_voice_active = bool(va) if va is not None else False
                last_va_check_ms = cur_ms

            # Accept as user-speech only if all three hold:
            #  - webrtcvad thinks it's speechy
            #  - RMS is above the echo floor
            #  - the XVF3000 chip agrees it's real voice (if available)
            gate_ok = is_speech and block_rms >= min_rms
            if tuning is not None:
                gate_ok = gate_ok and chip_voice_active

            if gate_ok:
                speech_ms += MIC_BLOCK_MS
                peak_rms = max(peak_rms, block_rms)
                chunks.append(pcm)
                if speech_ms >= min_speech_ms:
                    log(f"audio_in: BARGE-IN (rms={peak_rms:.0f}, speech={speech_ms}ms)")
                    abort.set()
                    return b"".join(chunks)
            else:
                speech_ms = 0
                chunks = []
                peak_rms = 0.0
        # On natural end, log what max RMS we ever saw so we can tune the floor.
        log(
            f"audio_in: bargein-watcher end (blocks={n_blocks}, "
            f"max_rms_seen={max_seen_rms:.0f}, floor={min_rms:.0f})"
        )
        return b""

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
        """Block until wake activation fires. Drains any backlog first.

        In "wake_word" mode: waits for openWakeWord to detect the phrase.
        In "vad" mode: waits for sustained voice (RMS + VAD) above threshold.
        """
        self._drain_queue()

        if self._wake_mode == "vad":
            # VAD-only wake: wait for sustained real speech.
            from .config import FOLLOW_UP_MIN_SPEECH_MS
            from .config import FOLLOW_UP_MIN_RMS
            log(f"audio_in: waiting for speech (vad mode)...")
            speech_ms = 0
            peak_rms = 0.0
            while not self._stop_wake.is_set():
                block = self._next_block()
                if block is None:
                    continue
                pcm = block.astype(np.int16).tobytes()
                is_speech = self._vad.is_speech(pcm, MIC_SAMPLE_RATE)
                if is_speech:
                    speech_ms += MIC_BLOCK_MS
                    block_rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
                    peak_rms = max(peak_rms, block_rms)
                    if speech_ms >= FOLLOW_UP_MIN_SPEECH_MS and peak_rms >= FOLLOW_UP_MIN_RMS:
                        # Capture DOA so follow-up mode can still filter later.
                        try:
                            from . import respeaker
                            self.last_wake_doa = respeaker.doa()
                        except Exception:
                            self.last_wake_doa = None
                        doa_str = f" doa={self.last_wake_doa}°" if self.last_wake_doa is not None else ""
                        log(f"audio_in: WAKE (vad, rms={peak_rms:.0f}){doa_str}")
                        return
                else:
                    speech_ms = 0
                    peak_rms = 0.0
            return

        # wake_word mode (openWakeWord)
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
                # Capture the ReSpeaker's direction-of-arrival at this moment
                # so follow-up mode can later reject audio from other angles.
                try:
                    from . import respeaker
                    self.last_wake_doa = respeaker.doa()
                except Exception:
                    self.last_wake_doa = None
                doa_str = f" doa={self.last_wake_doa}°" if self.last_wake_doa is not None else ""
                log(f"audio_in: WAKE (score={score:.2f}){doa_str}")
                last_trigger = time.time()
                return

    def record_utterance(
        self,
        prebuffer: bytes = b"",
        transcribe_fn=None,
    ) -> bytes:
        """Record PCM until we detect the user is done.

        End conditions, any of which stops recording:
        - VAD silence >= VAD_SILENCE_END_MS after speech started (classic path)
        - Semantic endpoint: transcribe_fn(pcm_so_far) returns the same text
          for SEMANTIC_STABLE_MS (enabled via transcribe_fn + SEMANTIC_ENDPOINT_ENABLED)
        - MAX_UTTERANCE_S hard cap

        If `transcribe_fn` is provided and SEMANTIC_ENDPOINT_ENABLED, we
        re-transcribe the rolling buffer every SEMANTIC_POLL_MS and stop
        when the transcript stops growing. This is much faster than
        VAD silence and robust to ambient noise that VAD flags as speech.
        """
        log("audio_in: recording utterance...")
        self._drain_queue()
        chunks: list[bytes] = [prebuffer] if prebuffer else []
        speech_ms = 0
        silence_ms = 0
        total_ms = 0
        speaking_started = bool(prebuffer)
        if prebuffer:
            speech_ms = VAD_MIN_SPEECH_MS

        # Semantic endpointing state
        use_semantic = transcribe_fn is not None and SEMANTIC_ENDPOINT_ENABLED
        last_text = ""
        last_change_ms = 0
        last_poll_ms = 0

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
                    log(f"audio_in: end-by-vad (silence {silence_ms}ms)")
                    break
            total_ms += MIC_BLOCK_MS

            # Semantic endpointing: re-transcribe on a schedule.
            if (
                use_semantic
                and speaking_started
                and total_ms >= SEMANTIC_MIN_AUDIO_MS
                and total_ms - last_poll_ms >= SEMANTIC_POLL_MS
            ):
                last_poll_ms = total_ms
                try:
                    text = transcribe_fn(b"".join(chunks)) or ""
                except Exception as e:
                    log(f"audio_in: semantic transcribe failed: {e}")
                    continue
                text = text.strip()
                if text != last_text:
                    last_text = text
                    last_change_ms = total_ms
                elif text and (total_ms - last_change_ms) >= SEMANTIC_STABLE_MS:
                    log(
                        f"audio_in: end-by-semantic (stable {total_ms - last_change_ms}ms, "
                        f"text={text!r})"
                    )
                    break

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
        turn without a new wake-word. We require sustained continuous speech
        (FOLLOW_UP_MIN_SPEECH_MS) to avoid breathing/clicks/ambient triggering
        false follow-ups.
        """
        self._drain_queue()
        deadline = time.time() + window_s
        speech_ms = 0
        chunks: list[bytes] = []
        while time.time() < deadline:
            block = self._next_block(timeout=min(1.0, max(0.1, deadline - time.time())))
            if block is None:
                continue
            pcm = block.astype(np.int16).tobytes()
            if self._vad.is_speech(pcm, MIC_SAMPLE_RATE):
                speech_ms += MIC_BLOCK_MS
                chunks.append(pcm)
                if speech_ms >= FOLLOW_UP_MIN_SPEECH_MS:
                    log(f"audio_in: follow-up speech detected after {speech_ms} ms")
                    # Return all accumulated speech so the first word isn't lost.
                    return b"".join(chunks)
            else:
                # Non-speech block resets so random clicks don't accumulate.
                speech_ms = 0
                chunks = []
        log(f"audio_in: follow-up window expired ({window_s:.1f}s)")
        return None
