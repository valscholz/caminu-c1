"""Event loop: wake word -> record -> STT -> streamed LLM -> streaming TTS."""
from __future__ import annotations
import signal
import sys
import time

from . import llm, stt, tts
from .audio_in import AudioInput
from .config import HISTORY_MAX_TURNS, HISTORY_TTL_S
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
    # Whisper ~5s (CPU int8), Kokoro ~6s (ONNX + voice embeddings).
    log("main: preloading STT and TTS models")
    stt._get_model()   # warm faster-whisper
    tts._get_tts()     # warm Kokoro
    log("main: models preloaded")

    audio = AudioInput()
    audio.start()

    def _shutdown(_signum, _frame):
        log("caminu-c1 shutdown")
        audio.stop()
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

            # Streaming TTS: tokens from Gemma → sentence buffer → Kokoro → paplay.
            # Kokoro/paplay live in a worker thread so the LLM callback never blocks.
            speaker = SentenceSpeaker()
            t0 = time.time()

            def on_text(chunk: str) -> None:
                speaker.feed(chunk)

            try:
                reply, history = llm.chat_turn(text, history, on_text=on_text)
                speaker.flush()
                log(f"main: reply={reply!r}  llm_elapsed={time.time()-t0:.2f}s")
            finally:
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
