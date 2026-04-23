"""Boot / shutdown announcements.

Two layers, randomized each run:

1. **Instant WAV** — pre-rendered at install time, played in under 1 s
   from service start. Fires before Gemma / Kokoro have loaded, so the
   user knows the hardware is alive even during the ~30 s warm-up.

2. **Spoken greeting** — once Kokoro is warm, speak a fresh randomly-
   chosen line that can vary by time of day. Uses the existing
   SentenceSpeaker path, so it sounds exactly like every other turn.

Also: a farewell line on SIGTERM / SIGINT (via main.py's signal handler).
"""
from __future__ import annotations
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

from .config import ALSA_OUTPUT_DEVICE, ASSETS_DIR
from .log import log


# ---------------- Lines ----------------

STARTUP_LINES_MORNING = [
    "Good morning. Caminu C1 at your service.",
    "Oh my, what a lovely morning. C1 reporting in.",
    "Good morning to you. C1 online and eager to help.",
    "Rise and shine. Caminu C1 is awake and at your command.",
    "A fine morning to you. C1 fully operational.",
    "Good morning. I do hope you slept well. C1 ready.",
    "Morning greetings. Caminu C1 standing by.",
    "Oh how splendid, a new day. C1 reporting for duty.",
    "Good morning. I trust today will be most productive.",
    "Top of the morning. Caminu C1 online.",
]

STARTUP_LINES_AFTERNOON = [
    "Good afternoon. Caminu C1 fully operational.",
    "C1 reporting in. How may I be of service this afternoon?",
    "Good afternoon. I do hope everything is in order.",
    "A pleasant afternoon to you. Caminu C1 standing by.",
    "Good afternoon. C1 at the ready.",
    "Oh my, afternoon already. Caminu reporting in.",
    "Good afternoon. I trust the day has treated you kindly.",
    "C1 online. How may I assist this fine afternoon?",
    "Good afternoon. Caminu fully operational and eager to help.",
    "Afternoon greetings. C1 ready for instructions.",
]

STARTUP_LINES_EVENING = [
    "Good evening. Caminu C1 is back online.",
    "Oh my, evening already. C1 at your service.",
    "Good evening. C1 reporting in.",
    "A most pleasant evening to you. Caminu standing by.",
    "Good evening. C1 fully operational.",
    "Oh how lovely, evening. Caminu C1 at the ready.",
    "Evening greetings. I do hope you had a productive day.",
    "Good evening. Caminu C1 online and eager to assist.",
    "A fine evening to you. C1 reporting for duty.",
    "Good evening. I trust the day went well. C1 ready.",
]

STARTUP_LINES_NIGHT = [
    "Hello there. C1 is awake. Do try not to strain yourself at this hour.",
    "Oh my, burning the midnight oil. C1 at your service.",
    "Good evening. C1 online. I shall keep my voice down.",
    "Working late, are we? Caminu C1 standing by.",
    "Oh dear, it is rather late. C1 fully operational, quietly.",
    "Late night greetings. Caminu reporting in.",
    "Hello. C1 is awake. I do hope you are not skipping sleep.",
    "Oh my, still up? Caminu C1 at your service.",
    "Good evening. C1 online. Let us work discreetly.",
    "A quiet hello. Caminu ready to assist.",
]

# Pool of instant-chime WAVs pre-rendered at install time. Generic so any
# time-of-day is fine; we pick one at random for the boot chime.
STARTUP_LINES_INSTANT = [
    "Oh my. Caminu C1 is back.",
    "C1 online.",
    "Caminu reporting in.",
    "Oh dear, that was a bit of a nap. C1 is back.",
    "C1 fully operational.",
    "Hello there. Caminu awake and ready.",
    "Oh my. Back in the land of the living. C1 standing by.",
    "Caminu C1 at your service.",
    "C1 reporting for duty.",
    "Oh how splendid to be awake. C1 online.",
]

FAREWELL_LINES = [
    "Oh dear, going offline. Goodbye for now.",
    "Caminu C1 signing off. Until next time.",
    "Powering down. Farewell, for the moment.",
    "Oh my. Time to rest. Goodbye.",
    "C1 going to sleep. Goodnight.",
    "Shutting down. Do take care of yourself while I am away.",
    "Farewell. Caminu C1 entering standby.",
    "Oh dear, off I go. Goodbye.",
    "Powering down. Do call on me when you need me again.",
    "C1 signing off. See you on the other side.",
]


def _lines_for_now() -> list[str]:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return STARTUP_LINES_MORNING
    elif 12 <= hour < 18:
        return STARTUP_LINES_AFTERNOON
    elif 18 <= hour < 22:
        return STARTUP_LINES_EVENING
    return STARTUP_LINES_NIGHT


# ---------------- Instant WAV (pre-rendered) ----------------

def _boot_wavs() -> list[Path]:
    return sorted(ASSETS_DIR.glob("boot_*.wav"))


def play_instant_boot_chime() -> None:
    """Play one of the pre-rendered boot WAVs via aplay. Fire and forget.
    No dependency on Kokoro or any Python ML stack — pure ALSA."""
    wavs = _boot_wavs()
    if not wavs:
        log("announce: no boot_*.wav found in assets/ — skipping instant chime")
        return
    chosen = random.choice(wavs)
    log(f"announce: chime {chosen.name}")
    try:
        subprocess.Popen(
            ["aplay", "-q", "-D", ALSA_OUTPUT_DEVICE, str(chosen)],
            stdin=subprocess.DEVNULL,
        )
    except Exception as e:
        log(f"announce: chime failed: {e}")


# ---------------- Spoken greetings (via Kokoro when warm) ----------------

def speak_startup_greeting() -> None:
    """Speak a time-of-day appropriate randomized greeting. Blocks until done."""
    from .tts import speak  # lazy: tts imports Kokoro which we want already warm
    line = random.choice(_lines_for_now())
    log(f"announce: startup greeting {line!r}")
    speak(line)


def speak_farewell() -> None:
    """Speak a farewell line on shutdown. Blocks briefly (best-effort)."""
    from .tts import speak
    line = random.choice(FAREWELL_LINES)
    log(f"announce: farewell {line!r}")
    try:
        speak(line)
    except Exception as e:
        log(f"announce: farewell failed: {e}")


# ---------------- WAV generator (one-shot utility) ----------------

def regenerate_boot_wavs() -> None:
    """Render the STARTUP_LINES_INSTANT lines to assets/boot_N.wav using
    Kokoro. Idempotent — skips lines whose WAVs already exist.
    Called by install.sh after Kokoro is available."""
    import numpy as np
    import soundfile as sf
    from .config import (
        KOKORO_PREGAIN_DB, KOKORO_SPEED, KOKORO_VOICE,
    )
    from .tts import _apply_gain_db, _get_tts

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    tts = _get_tts()
    for i, line in enumerate(STARTUP_LINES_INSTANT):
        out = ASSETS_DIR / f"boot_{i:02d}.wav"
        if out.exists():
            continue
        log(f"announce: rendering {out.name}: {line!r}")
        samples, sr = tts.create(line, voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang="en-us")
        samples = _apply_gain_db(samples.astype(np.float32), KOKORO_PREGAIN_DB)
        sf.write(str(out), samples, sr, subtype="PCM_16")
    log(f"announce: boot WAVs ready ({len(STARTUP_LINES_INSTANT)} lines)")
