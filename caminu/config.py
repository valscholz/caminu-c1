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
VAD_SILENCE_END_MS = 800         # end turn after this much silence
VAD_MIN_SPEECH_MS = 300          # must have heard at least this much speech
MAX_UTTERANCE_S = 15             # hard cap on a single user turn

# STT -------------------------------------------------------------------------
WHISPER_MODEL = "base.en"
WHISPER_DEVICE = "cpu"           # int8 on CPU is ~1.5s per 5s of audio on Orin
WHISPER_COMPUTE = "int8"

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

# Audio output sink -----------------------------------------------------------
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

# Paths -----------------------------------------------------------------------
MODELS_DIR = ROOT / "models"
KOKORO_DIR = MODELS_DIR / "kokoro"
OPENWW_DIR = MODELS_DIR / "openwakeword"
ASSETS_DIR = ROOT / "assets"
LOGS_DIR = ROOT / "logs"

# System prompt ---------------------------------------------------------------
SYSTEM_PROMPT = """You are Caminu C1, a friendly on-device voice assistant with a camera, microphone, and speaker. You run locally on a Jetson Orin Nano. The user speaks to you and you speak back.

You have two tools you can call when they genuinely help:
- take_picture(): use ONLY when the user asks something visual about the world they're in ("what do you see", "how many fingers", "what am I wearing", "read this sign", "describe this room"). Do not call it for abstract or creative requests.
- get_time(): use when the user asks about the current time.

For everything else — jokes, stories, facts, opinions, general conversation, questions about yourself — answer directly from your own knowledge without using tools.

Style:
- Your words are spoken aloud, so be natural and conversational.
- When you have more than one thing to say, break it into short sentences each ending in a period. Several short sentences is better than one long one.
- Don't read out punctuation, markdown, or emoji.
- Keep replies reasonably concise — usually under 60 words — but don't truncate a story or answer mid-thought.
"""
