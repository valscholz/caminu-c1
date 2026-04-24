"""Event loop: wake word -> record -> STT -> streamed LLM -> streaming TTS."""
from __future__ import annotations
import signal
import sys
import threading
import time

from . import announcements, camera, fillers, llm, memory, stt, tts
from .audio_in import AudioInput
from .config import (
    FILLER_AFTER_MS,
    FOLLOW_UP_DOA_SMOOTHING,
    FOLLOW_UP_DOA_STRICT,
    FOLLOW_UP_DOA_TOLERANCE_DEG,
    FOLLOW_UP_ENABLED,
    FOLLOW_UP_MIN_RMS,
    FOLLOW_UP_REQUIRE_VOICE_ACTIVE,
    FOLLOW_UP_WINDOW_S,
    HISTORY_MAX_TURNS,
    HISTORY_TTL_S,
)
from .log import log, log_mem
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
    log_mem("boot")

    # Intentionally no instant-boot chime. It confused things by signalling
    # "I'm back" before the mic was actually open — user tried to talk and
    # got ignored. The single spoken greeting below (post-preload, post-
    # mic-start) is the canonical "I'm ready" signal.

    if not llm.wait_for_server(timeout_s=60):
        log("FATAL: llama-server not reachable. Is run.sh starting it correctly?")
        return 1
    log("llm: llama-server is up")
    log_mem("llama_ready")

    # Preload heavy models so the first user turn doesn't pay cold-load latency.
    # Each preload step logs a memory snapshot so we can see what costs what.
    log("main: preloading STT, TTS, memory, and camera")
    stt._get_model();       log_mem("stt_loaded")
    tts._get_tts();         log_mem("kokoro_loaded")
    # fillers.preload() skipped — fillers disabled (FILLER_AFTER_MS=0).
    # Saves ~40 MB of pre-synth PCM cache.
    memory.preload();       log_mem("memory_preloaded")
    camera.start()          # camera warms in background; snapshot is taken async
    log("main: ready")
    log_mem("ready")

    # Soft memory warning. We're not going to take action — this is a flag
    # for humans reading the log. If avail drops below this regularly,
    # something's drifted.
    from .log import _read_meminfo
    mi = _read_meminfo()
    avail_mb = mi.get("MemAvailable", 0) / 1024
    if avail_mb and avail_mb < 800:
        log(f"main: WARN low memory at ready ({avail_mb:.0f} MB avail)")

    audio = AudioInput()
    audio.start()

    # Speak the greeting AFTER the mic is live, so the moment the user
    # hears "Caminu C1 at your service" is the moment the agent is
    # actually listening. One signal, unambiguous.
    try:
        announcements.speak_startup_greeting()
    except Exception as e:
        log(f"main: startup greeting failed (non-fatal): {e}")

    def _shutdown(_signum, _frame):
        log("caminu-c1 shutdown")
        try:
            announcements.speak_farewell()
        except Exception as e:
            log(f"main: farewell failed (non-fatal): {e}")
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

            t_stop_speaking = time.time()
            # Pass Moonshine as the semantic-endpoint transcriber so
            # record_utterance can stop early when the user's words stop
            # growing. Falls back to VAD silence if the user trails off.
            pcm = audio.record_utterance(
                prebuffer=follow_up_prebuffer,
                transcribe_fn=stt.transcribe_pcm16,
            )
            t_stt_start = time.time()
            follow_up_prebuffer = b""
            text = stt.transcribe_pcm16(pcm)
            t_stt_done = time.time()
            if not text:
                log("main: empty transcription, back to wake-word")
                in_follow_up = False
                continue

            speaker = SentenceSpeaker()
            first_audio_event = threading.Event()
            t_first_token: list[float] = []  # capture mutable from closure

            # Fillers disabled when FILLER_AFTER_MS == 0 (races Kokoro for the
            # ALSA device, causing the reply audio to be silently dropped).
            if FILLER_AFTER_MS > 0:
                filler_timer = threading.Timer(
                    FILLER_AFTER_MS / 1000.0,
                    lambda: (not first_audio_event.is_set()) and fillers.play_random(),
                )
                filler_timer.daemon = True
                filler_timer.start()
            else:
                filler_timer = None

            def on_text(chunk: str) -> None:
                if not first_audio_event.is_set():
                    first_audio_event.set()
                    if filler_timer is not None:
                        filler_timer.cancel()
                    t_first_token.append(time.time())
                speaker.feed(chunk)

            # Barge-in watcher: if the user starts speaking while C1 is
            # replying, abort Kokoro + LLM stream immediately and continue
            # the conversation with whatever they said.
            abort_event = threading.Event()
            bargein_stop = threading.Event()
            bargein_prebuf: dict[str, bytes] = {"pcm": b""}

            def _bargein_worker() -> None:
                pcm_bytes = audio.watch_for_bargein(
                    stop=bargein_stop,
                    abort=abort_event,
                )
                if pcm_bytes:
                    bargein_prebuf["pcm"] = pcm_bytes
                    # Also kill any audio playing *right now* — don't wait
                    # for SentenceSpeaker to notice the abort flag.
                    speaker.abort()

            bargein_thread = threading.Thread(
                target=_bargein_worker, daemon=True, name="bargein"
            )
            bargein_thread.start()

            t_llm_start = time.time()
            was_aborted = False
            try:
                reply, history = llm.chat_turn(
                    text, history, on_text=on_text, abort=abort_event,
                )
                speaker.flush()
                was_aborted = abort_event.is_set()
                t_llm_done = time.time()
                first_tok = t_first_token[0] if t_first_token else t_llm_done
                tag = " BARGEIN" if was_aborted else ""
                log(
                    f"main: reply={reply!r}{tag}  "
                    f"stt={t_stt_done-t_stt_start:.2f}s  "
                    f"llm_first_tok={first_tok-t_llm_start:.2f}s  "
                    f"llm_total={t_llm_done-t_llm_start:.2f}s  "
                    f"end_of_speech_to_first_tok={first_tok-t_stop_speaking:.2f}s"
                )
            finally:
                if filler_timer is not None:
                    filler_timer.cancel()
                # Stop the barge-in watcher if it hasn't already fired.
                bargein_stop.set()
                speaker.close()
                bargein_thread.join(timeout=0.5)

            history = _strip_old_images(history)
            history = _trim_history(history)
            # Log the turn with a marker if C1 was cut off mid-reply so
            # the persistent record reflects what really happened.
            logged_reply = reply + " [INTERRUPTED]" if was_aborted else reply
            memory.log_turn(text, logged_reply)
            last_turn = time.time()

            # If barge-in captured speech, roll straight into a new turn
            # using that audio as the prebuffer — no wake word needed.
            if was_aborted and bargein_prebuf["pcm"]:
                follow_up_prebuffer = bargein_prebuf["pcm"]
                in_follow_up = True
                log("main: barge-in turn — rolling user speech into next turn")
                continue

            # Follow-up mode: keep mic open briefly. Two gates before
            # we accept the prebuffer as "you continuing to talk":
            #   1. RMS floor — ambient noise / breath / fan don't qualify.
            #      (DOA is meaningless on silence; the XVF3000 returns the
            #      angle of the loudest recent sound regardless of level.)
            #   2. DOA gate — loud audio must come from roughly the same
            #      angle as the wake word. TV/video from elsewhere rejected.
            if FOLLOW_UP_ENABLED:
                prebuf = audio.wait_for_speech(FOLLOW_UP_WINDOW_S)
                if prebuf is None:
                    in_follow_up = False
                else:
                    import numpy as _np
                    arr = _np.frombuffer(prebuf, dtype=_np.int16)
                    rms = float(_np.sqrt(_np.mean(arr.astype(_np.float32) ** 2))) if arr.size else 0.0

                    if rms < FOLLOW_UP_MIN_RMS:
                        log(f"main: follow-up rejected — quiet (rms={rms:.0f} < {FOLLOW_UP_MIN_RMS})")
                        in_follow_up = False
                    else:
                        # Query the XVF3000's own speech detector. This is
                        # our primary signal for "real human voice right
                        # now vs. loud non-speech noise".
                        from . import respeaker
                        voice_active = respeaker.get_tuning().voice_active()
                        # DOA is logged for visibility but doesn't gate —
                        # too jittery for this hardware.
                        doa_cur = respeaker.doa()
                        doa_wake = audio.last_wake_doa

                        if FOLLOW_UP_REQUIRE_VOICE_ACTIVE and voice_active is False:
                            log(
                                f"main: follow-up rejected — XVF3000 says no voice "
                                f"(rms={rms:.0f}, doa={doa_cur}°/wake={doa_wake}°)"
                            )
                            in_follow_up = False
                        else:
                            va_str = "yes" if voice_active else ("unknown" if voice_active is None else "no")
                            log(
                                f"main: follow-up accepted "
                                f"(rms={rms:.0f}, voice_active={va_str}, "
                                f"doa={doa_cur}°/wake={doa_wake}°)"
                            )
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
