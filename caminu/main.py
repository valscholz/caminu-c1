"""Event loop: wake word -> record -> STT -> streamed LLM -> streaming TTS."""
from __future__ import annotations
import signal
import sys
import threading
import time

from . import camera, fillers, llm, stt, tts
from .audio_in import AudioInput
from .config import FILLER_AFTER_MS, HISTORY_MAX_TURNS, HISTORY_TTL_S
from .log import log
from .tts import SentenceSpeaker


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep system prompt + last N turns to cap context size on 8 GB Orin."""
    if not history:
        return history
    sys_msg = [m for m in history[:1] if m.get("role") == "system"]
    rest = history[len(sys_msg):]
    keep: list[dict] = []
    turns = 0
    for m in reversed(rest):
        keep.append(m)
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            turns += 1
            if turns >= HISTORY_MAX_TURNS:
                break
    keep.reverse()
    return sys_msg + keep


def main() -> int:
    log("caminu-c1 starting")

    if not llm.wait_for_server(timeout_s=60):
        log("FATAL: llama-server not reachable. Is run.sh starting it correctly?")
        return 1
    log("llm: llama-server is up")

    # Preload heavy models so the first user turn doesn't pay cold-load latency.
    log("main: preloading STT, TTS, fillers, and camera")
    stt._get_model()          # warm faster-whisper
    tts._get_tts()            # warm Kokoro
    fillers.preload()         # pre-synth filler PCM
    camera.start()            # start OAK-D background thread
    log("main: ready")

    audio = AudioInput()
    audio.start()

    def _shutdown(_signum, _frame):
        log("caminu-c1 shutdown")
        audio.stop()
        camera.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    history: list[dict] = []
    last_turn = 0.0

    while True:
        try:
            audio.wait_for_wake_word()

            if time.time() - last_turn > HISTORY_TTL_S:
                history = []

            pcm = audio.record_utterance()
            text = stt.transcribe_pcm16(pcm)
            if not text:
                log("main: empty transcription, back to wake-word")
                continue

            speaker = SentenceSpeaker()
            first_audio_event = threading.Event()

            # Filler timer: play an acknowledgement phrase if no content arrives
            # within FILLER_AFTER_MS of STT completion.
            filler_timer = threading.Timer(
                FILLER_AFTER_MS / 1000.0,
                lambda: (not first_audio_event.is_set()) and fillers.play_random(),
            )
            filler_timer.daemon = True
            filler_timer.start()

            def on_text(chunk: str) -> None:
                if not first_audio_event.is_set():
                    first_audio_event.set()
                    filler_timer.cancel()
                speaker.feed(chunk)

            t0 = time.time()
            try:
                reply, history = llm.chat_turn(text, history, on_text=on_text)
                speaker.flush()
                log(f"main: reply={reply!r}  llm_elapsed={time.time()-t0:.2f}s")
            finally:
                filler_timer.cancel()
                speaker.close()

            history = _trim_history(history)
            last_turn = time.time()

        except KeyboardInterrupt:
            _shutdown(None, None)
        except Exception as e:
            log(f"main: error: {e}")
            time.sleep(0.5)


if __name__ == "__main__":
    sys.exit(main())
