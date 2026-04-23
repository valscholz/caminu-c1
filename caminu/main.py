"""Event loop: wake word -> record -> STT -> streamed LLM -> streaming TTS."""
from __future__ import annotations
import signal
import sys
import threading
import time

from . import camera, fillers, llm, memory, stt, tts
from .audio_in import AudioInput
from .config import FILLER_AFTER_MS, FOLLOW_UP_WINDOW_S, HISTORY_MAX_TURNS, HISTORY_TTL_S
from .log import log
from .tts import SentenceSpeaker


def _strip_old_images(history: list[dict]) -> list[dict]:
    """Replace bulky base64 image parts in completed turns with a text stub.

    Each OAK-D photo becomes ~256 vision tokens + ~100 KB of base64 in the
    message. After Gemma has replied about an image, the raw pixels add
    nothing — her text description stays in history as an assistant message.
    Keeping the pixels causes two problems: the KV cache bloats (~25% of
    context on a multi-vision conversation) and the mmproj compute buffer
    can OOM llama-server ('failed to allocate compute pp buffers').

    We only strip *completed* turns: the last user message keeps its image
    since Gemma may still be mid-turn processing it.
    """
    if not history:
        return history
    # Find the last user message — that one is "current", everything else is past
    last_user_idx = max(
        (i for i, m in enumerate(history) if m.get("role") == "user"),
        default=-1,
    )
    out = []
    for i, msg in enumerate(history):
        content = msg.get("content")
        if (
            isinstance(content, list)
            and i != last_user_idx
            and any(p.get("type") == "image_url" for p in content if isinstance(p, dict))
        ):
            # collapse to a simple text placeholder
            texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            stub = " ".join(t for t in texts if t) or "[photo shown to Caminu]"
            out.append({**msg, "content": stub + " [photo removed from context to save memory]"})
        else:
            out.append(msg)
    return out


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
    log("main: preloading STT, TTS, fillers, memory, and camera")
    stt._get_model()          # warm faster-whisper
    tts._get_tts()            # warm Kokoro
    fillers.preload()         # pre-synth filler PCM
    memory.preload()          # warm embedder + index conversation log
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
    follow_up_prebuffer: bytes = b""
    in_follow_up = False

    while True:
        try:
            if in_follow_up:
                # We're continuing a conversation: skip wake-word and use the
                # prebuffered speech chunk that triggered the follow-up.
                log("main: follow-up turn")
            else:
                audio.wait_for_wake_word()

            if time.time() - last_turn > HISTORY_TTL_S:
                history = []

            pcm = audio.record_utterance(prebuffer=follow_up_prebuffer)
            follow_up_prebuffer = b""
            text = stt.transcribe_pcm16(pcm)
            if not text:
                log("main: empty transcription, back to wake-word")
                in_follow_up = False
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

            history = _strip_old_images(history)
            history = _trim_history(history)
            memory.log_turn(text, reply)
            last_turn = time.time()

            # Follow-up: keep the mic open briefly. If the user starts speaking,
            # flow straight into the next turn without requiring another wake
            # word. Otherwise return to wake-word mode.
            prebuf = audio.wait_for_speech(FOLLOW_UP_WINDOW_S)
            if prebuf is not None:
                follow_up_prebuffer = prebuf
                in_follow_up = True
            else:
                in_follow_up = False

        except KeyboardInterrupt:
            _shutdown(None, None)
        except Exception as e:
            log(f"main: error: {e}")
            time.sleep(0.5)


if __name__ == "__main__":
    sys.exit(main())
