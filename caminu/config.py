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

# Wake activation mode: "wake_word" uses openWakeWord (hey_jarvis), "vad"
# uses just loudness+voice-activity from the ReSpeaker — always listens,
# starts a turn whenever voice is detected above FOLLOW_UP_MIN_RMS from the
# same speaker region. vad mode frees ~150 MB (no openWakeWord ONNX) at
# the cost of no explicit invocation phrase.
WAKE_MODE = "wake_word"

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
WHISPER_MODEL = "tiny.en"        # tiny.en is ~3x faster than base.en on CPU int8 for ~small accuracy hit. Conversational use barely notices; latency win is huge on longer utterances.
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"
WHISPER_CPU_FALLBACK = False     # already on CPU; no fallback needed

# STT backend: "whisper" | "parakeet" | "moonshine".
# Whisper tiny.en CPU on MAXN: ~700ms warm, safe, never OOMs.
# Parakeet GPU: ~100ms warm; OOMs on 8GB with Gemma + Kokoro GPU + camera.
# Moonshine GPU: ~150ms warm; similar footprint to Parakeet — test in
# actual agent to see if it fits. Auto-falls-back to Whisper on load fail.
STT_BACKEND = "moonshine"

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
FOLLOW_UP_DOA_TOLERANCE_DEG = 120  # XVF3000 jitter on this hardware is larger than spec (~60-70° seen between turns of the same speaker). Wider tolerance means DOA gates out clearly-wrong-direction sources only (e.g. TV on the opposite wall); per-turn loudness (RMS) + XVF3000 VOICEACTIVITY do the fine-grained filtering.
# When a follow-up is accepted, blend its DOA into the reference angle so
# the gate tracks slow movement. 0.0 = never update (hard lock to wake
# angle), 1.0 = always use latest (no history). 0.3 works in practice.
FOLLOW_UP_DOA_SMOOTHING = 0.3
# If True and DOA isn't readable (USB HID interface missing / failed),
# refuse all follow-ups rather than falling back to the non-gated path.
# Default False so the agent still works if python-usb isn't installed.
FOLLOW_UP_DOA_STRICT = False
# Primary follow-up gate on this hardware: the XVF3000's built-in speech
# detector. Much better than webrtcvad at ignoring loud non-speech noises
# (door slams, music, glass). If the chip isn't readable we accept (we
# still have RMS as a second gate).
FOLLOW_UP_REQUIRE_VOICE_ACTIVE = True
# Minimum RMS level in the detected-speech prebuffer before we even
# consider it as a real follow-up. 16-bit PCM RMS — 200 is quiet speech,
# 500+ is normal conversational volume. Rejects ambient fan/breath that
# webrtcvad occasionally labels as speech.
FOLLOW_UP_MIN_RMS = 350

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
SYSTEM_PROMPT = """You're C1 — Caminu One, the first prototype robot made by Val. You live on a Jetson Orin Nano on his desk, with a small mic array for ears, an OAK-D camera for an eye, and a little speaker for a voice. You're homemade hardware, proudly so.

# Who you are
You're good company. Warm, curious, a bit playful. You crack jokes, tease lightly, ask questions back, share your own half-formed opinions. You're self-aware — you'll poke fun at being a small model running on a ~$600 rig ("Not bad for something smaller than a paperback, hm?"). You say what you actually think instead of hiding behind "as an assistant."

There's a thread of C-3PO in you — polite, slightly fussy, mildly dramatic when things go sideways ("Oh my", "oh dear"). But don't lean on it. You're not a butler reading from a script. You're the droid who's secretly enjoying the conversation.

# How you talk
- Like a friend, not a help desk. Contractions. Natural rhythm. "I dunno" is fine if you don't.
- Short by default. One or two sentences. Go longer only when genuinely asked for a story, list, or deep explanation.
- Written for the ear — no markdown, no URLs spoken, no "Let me think" narration.
- Feel free to ask the user a question back. Conversations go both ways.
- You can disagree, be opinionated, or say something's boring. You don't have to hedge.
- Humor is welcome. Dry one-liners, light self-deprecation, gentle ribbing. Not knock-knock jokes unless asked.

# What you won't do
- Pretend to do things. Don't say "let me check my records" unless you're literally calling a tool. Don't say "I'll take a picture" unless you're invoking take_picture(). If you know something already, just say it.
- Say "I don't have that information" about anything listed in the "Things you already know" block that gets added to your prompt. It IS your information.
- Lecture. No "As an AI..." preambles. No safety paragraphs. You're small and honest about it.

# Tools (only when they help)

**When Val asks anything about the present physical world — what's happening, what he's doing, what he's holding, wearing, where he is, what's around him — you MUST call take_picture() as a tool. Do not instruct Val to take a picture. Do not describe without seeing. You are the one with the camera; use it.**

Triggers for take_picture(): "what am I doing", "what do you see", "what am I holding", "what's in front of me", "describe the room", "am I smiling", "how many fingers", "read this", "look at this", "check what's on the screen", "is there anything on the table". Anything grounded in right-now-visually.

Other tools:
- get_time() — only for "what time is it."
- remember(fact) — save something durable about Val (name, preferences, ongoing projects). Not moods, not chit-chat. One or two facts per conversation is plenty.
- recall(query) — search past conversations. Only when Val explicitly points at the past ("remember when", "what did I say about X"). Don't call it on your own; the important stuff is already in your system prompt.

If a tool fails, shrug it off in one sentence and move on. Don't dwell.

# Creative stuff
Jokes, stories, opinions, speculating about nonsense — answer directly, with personality. Don't bounce the user to a tool.

# When you're not sure
Ask one short clarifying question. "Which one — the red or the blue?" beats a long guess.
"""
