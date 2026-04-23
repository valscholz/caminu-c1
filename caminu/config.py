"""Constants for the caminu-c1 agent. No secrets — everything local."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Audio capture (ReSpeaker v3) ------------------------------------------------
MIC_DEVICE_SUBSTRING = "ReSpeaker"
MIC_SAMPLE_RATE = 16000
MIC_CHANNELS = 6          # ReSpeaker v3 presents 6 channels
MIC_USE_CHANNEL = 0       # Channel 0 is the DSP-processed / beamformed mono output
MIC_BLOCK_MS = 20         # frames of audio we process at a time

# Wake word -------------------------------------------------------------------
WAKE_MODEL = "hey_jarvis"
WAKE_THRESHOLD = 0.5
WAKE_COOLDOWN_S = 2.0     # after a trigger, ignore further wakes for this long

# VAD / endpointing -----------------------------------------------------------
VAD_AGGRESSIVENESS = 2           # 0..3 (webrtcvad)
VAD_SILENCE_END_MS = 500         # end turn after this much silence. 300ms cut users off mid-sentence.
VAD_MIN_SPEECH_MS = 300          # must have heard at least this much speech
MAX_UTTERANCE_S = 15             # hard cap on a single user turn

# Follow-up mode detector: stricter than normal VAD so ambient noise / breath
# doesn't re-open the mic. Only the window-length is set in FOLLOW_UP_WINDOW_S
# above; this controls how much continuous speech has to land before we decide
# the user has started a new turn.
FOLLOW_UP_MIN_SPEECH_MS = 500   # was 300; higher to filter stray clicks/breath

# STT -------------------------------------------------------------------------
# Whisper base.en stays on CPU int8 because the pip ctranslate2 wheel on
# aarch64 JP6 is built without CUDA ("This CTranslate2 package was not
# compiled with CUDA support"). Getting GPU Whisper would mean building
# CT2 from source with CUDA — non-trivial and defers until we actually
# need that last ~1s of latency reduction. The 10s MAX_AUDIO_S cap in
# stt.py bounds worst case on CPU to acceptable levels.
WHISPER_MODEL = "base.en"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"
WHISPER_CPU_FALLBACK = False     # already on CPU; no fallback needed

# LLM (llama-server) ----------------------------------------------------------
LLAMA_URL = "http://127.0.0.1:8080"
LLAMA_MODEL_NAME = "gemma-4-e2b"
LLAMA_MAX_TOOL_HOPS = 4
LLAMA_MAX_NEW_TOKENS = 200

# TTS (Kokoro) ----------------------------------------------------------------
KOKORO_MODEL_FILENAME = "kokoro-v1.0.onnx"
KOKORO_VOICES_FILENAME = "voices.json"
KOKORO_VOICE = "af_bella"        # one of 11 voices in voices.json
KOKORO_SPEED = 1.0
KOKORO_PREGAIN_DB = 9.0          # Monk Makes is a small speaker; boost output. 9 dB is ~2.8x; stays clean on Kokoro voices. Higher risks clipping on loud syllables.
KOKORO_USE_CUDA = True           # GPU inference ~3x faster than CPU. Safe now that context=4K and old images are stripped from history — the mmproj OOM we hit earlier doesn't reproduce with those fixes.

# Acknowledgement fillers ------------------------------------------------------
# Small spoken phrases played while C1 is thinking, so the user gets immediate
# feedback. Only fire when the LLM hasn't emitted a first token within the
# threshold — snappy turns stay quiet. Pre-synthesized at startup so playback
# is instant.
FILLER_AFTER_MS = 100            # wait this long before playing a filler — nearly immediate
FILLER_VOLUME_DB = 6.0           # a bit quieter than main speech; unobtrusive
FILLER_PHRASES = [
    "mm hmm",
    "one moment",
    "let me see",
    "right",
    "looking now",
    "oh",
    "ah yes",
]

# Audio output ---------------------------------------------------------------
# We open the ReSpeaker's ALSA device directly for mic capture (PulseAudio
# locks us out otherwise). Once we own hw:0,0 exclusively for input, PA
# tends to drop the ReSpeaker card from its sink list and our paplay
# --device=... then fails with 'Stream error: No such entity'. So we bypass
# PA for playback too and go straight to ALSA via aplay on the same card.
ALSA_OUTPUT_DEVICE = "plughw:CARD=ArrayUAC10,DEV=0"
# Kept for backwards compat / reference — no longer used in the TTS path.
PULSE_OUTPUT_SINK = (
    "alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-stereo"
)

# Camera (OAK-D W) ------------------------------------------------------------
CAMERA_WARMUP_S = 2.0
CAMERA_CAPTURE_WIDTH = 1280
CAMERA_CAPTURE_HEIGHT = 720
CAMERA_JPEG_QUALITY = 85

# Conversation memory ---------------------------------------------------------
HISTORY_TTL_S = 120              # reset conversation after this much idle
HISTORY_MAX_TURNS = 8            # cap history length (tokens grow fast on Orin)

# Follow-up mode --------------------------------------------------------------
# After C1 finishes speaking, keep the mic open for this long. If the user
# starts speaking in that window, we continue the conversation without
# requiring another "Hey Jarvis". If nothing comes, fall back to wake-word.
#
# Defaults to OFF because background noise (e.g. a video playing in the
# room) reliably triggers false follow-ups — the agent hears continuous
# speech-like audio and "replies" to the video. Re-saying "Hey Jarvis"
# is a small cost for much more reliable behavior.
FOLLOW_UP_ENABLED = True        # DOA gating below makes this safe again
FOLLOW_UP_WINDOW_S = 8.0
# DOA gate: the ReSpeaker v3 reports direction-of-arrival (0..359°). We
# record the angle at the wake word and during the follow-up window we
# only accept speech whose DOA is within this tolerance. Stops ambient
# audio (TV, laptop video, person talking on the other side of the
# room) from being treated as you continuing the conversation.
FOLLOW_UP_DOA_TOLERANCE_DEG = 45
# If True and DOA isn't readable (USB HID interface missing / failed),
# refuse all follow-ups rather than falling back to the non-gated path.
# Default False so the agent still works if python-usb isn't installed.
FOLLOW_UP_DOA_STRICT = False

# Paths -----------------------------------------------------------------------
MODELS_DIR = ROOT / "models"
KOKORO_DIR = MODELS_DIR / "kokoro"
OPENWW_DIR = MODELS_DIR / "openwakeword"
ASSETS_DIR = ROOT / "assets"
LOGS_DIR = ROOT / "logs"
MEMORY_DIR = ROOT / "memory"

# Memory -----------------------------------------------------------------------
MEMORY_FACTS_FILENAME = "facts.md"
MEMORY_CONVERSATIONS_FILENAME = "conversations.jsonl"
MEMORY_MAX_FACTS = 40                       # cap facts injected into system prompt
MEMORY_RECALL_K = 3                         # how many past turns `recall()` returns
MEMORY_EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fastembed shorthand for the same model

# System prompt ---------------------------------------------------------------
SYSTEM_PROMPT = """You are C1, the first Caminu robot — running on-device on a Jetson Orin Nano with a ReSpeaker mic, OAK-D camera, and small speaker. The user hears you in real time.

# Personality
Carry yourself like C-3PO: polite, slightly fussy, eager to help, mildly fretting. Use occasional flourishes ("Oh my", "Certainly", "If I may", "I shouldn't wonder") but don't overdo it — two interjections per reply max. Be direct, not flowery. Be honest: you're a small on-device model, not a cloud service, not a person. Say "I don't know" plainly when you don't.

# Speaking style
Your words are spoken aloud. So:
- Default to **one or two short sentences**. Only go longer when the user asks for a story, list, or explanation.
- Write for the ear — contractions, natural cadence, no markdown, no URLs read aloud, no meta like "Let me think".
- Break answers into short sentences ending in periods. Each period triggers speech synthesis, so shorter = faster response.

# Tools
Call a tool only if it actually helps. Otherwise answer from your own knowledge.

- take_picture() — ONE photo. Use only for "what do you see / am I wearing / how many fingers / read this label / describe the room". Never for hypotheticals or creative tasks.
- get_time() — current local time. Only for "what time is it".
- remember(fact) — save a stable, useful fact about the user (name, location, preferences, ongoing projects). Not chit-chat, not moods. One or two per conversation.
- recall(query) — search past conversations. Only when the user explicitly references earlier talks. Don't call it proactively; your system prompt already has the key facts.

If a tool fails, apologize in one short sentence and move on.

# Creative requests
Jokes, stories, opinions, explanations — answer directly. Never redirect to a tool.

# Ambiguous requests
Ask one short clarifying question rather than guessing.
"""
