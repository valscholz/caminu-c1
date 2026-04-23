"""Event loop: wake word -> record -> STT -> LLM -> TTS."""
from __future__ import annotations
import signal
import sys
import time

from . import llm, stt, tts
from .audio_in import AudioInput
from .config import HISTORY_MAX_TURNS, HISTORY_TTL_S
from .log import log


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep system prompt + last N turns to cap context size on 8 GB Orin."""
    if not history:
        return history
    sys_msg = [m for m in history[:1] if m.get("role") == "system"]
    rest = history[len(sys_msg):]
    # Count user turns from the end; keep last HISTORY_MAX_TURNS
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
                history = []  # fresh conversation

            pcm = audio.record_utterance()
            text = stt.transcribe_pcm16(pcm)
            if not text:
                log("main: empty transcription, back to wake-word")
                continue

            reply, history = llm.chat_turn(text, history)
            history = _trim_history(history)
            if reply:
                tts.speak(reply)
            last_turn = time.time()

        except KeyboardInterrupt:
            _shutdown(None, None)
        except Exception as e:
            log(f"main: error: {e}")
            time.sleep(0.5)


if __name__ == "__main__":
    sys.exit(main())
