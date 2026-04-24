# Ideas to try next

## 🔥 Priority: better conversational flow (tomorrow)

These two together should make the agent feel almost like talking to a
person. Both became obvious after a full day of real use.

### Semantic endpointing — "send to Gemma when text stops changing"
Instead of waiting for VAD silence (brittle — fails on eating, chewing,
ambient fan, echo from Kokoro):
1. While the user is speaking, run Moonshine every ~500 ms on the
   audio-so-far.
2. Compare successive partial transcripts. If the last 500-1000 ms
   added no new tokens, consider the user done.
3. Send to Gemma immediately, don't wait for VAD silence.

Moonshine on GPU does 8 s of audio in ~150 ms, so we can afford to
run it repeatedly mid-utterance. Expected: ~500-800 ms faster every
turn, and immune to VAD false-positives from background noise.

Keep VAD as a safety net (if Moonshine partials don't stabilize,
still cap the turn at MAX_UTTERANCE_S).

~80 LOC. Touches `audio_in.py` (yield rolling PCM) and `main.py`
(poll + compare).

### Barge-in (interrupt C1 mid-reply)
Currently the mic ignores us while Kokoro is speaking. So if C1 is
mid-sentence and you want to redirect her ("wait, no, I meant..."),
you can't. This is the biggest "feels robotic" remaining gap.

Needs:
1. Keep mic hot during TTS playback (we already do via `mute_input`
   which we added but never wired in — can revert that, should be
   fine given XVF3000's AEC).
2. Detect wake or sustained loud speech while TTS is playing.
3. On detection: kill Kokoro mid-sentence, terminate aplay
   subprocess, skip the rest of the LLM stream, go back to
   record_utterance.
4. Handle partial history: append the partial assistant message as
   "...[user interrupted]" so Gemma has context.

Risk: self-triggering on her own voice echo. The ReSpeaker XVF3000
has onboard AEC that mostly handles this, but we'd want to verify
under load.

~100 LOC. Touches `audio_in.py`, `tts.py` (SentenceSpeaker abort),
`main.py` (abort + continue flow).

## Other

### Home Assistant integration
HA on a cheap separate device (Pi 4B €55, 3W, €8/yr power).
`control_home(service, data)` tool hits HA REST API. Unlocks
"turn off the lights" and "text Anna I'll be late" (HA notify
integrations cover Telegram/Signal/Discord).

### Pose / posture nag tool
OAK-D onboard pose estimation on the Movidius MyriadX — zero Jetson
CPU/GPU cost. Sample every 10 min, if slouching >60 s → Kokoro:
"Oh dear, you're hunching again."

### "Thinking as a tool"
Expose `think_hard(question)` so Gemma can opt into chain-of-thought
when a question genuinely needs it, then return a concise spoken
answer. Deep reasoning happens invisibly; normal turns stay fast.

### Custom wake word "hey C1"
Train openWakeWord on "hey C1" in the Colab (~1-2 h once). Commit
the .onnx, swap `WAKE_MODEL` in config.

### ReSpeaker DOA as a tool
`look_at_speaker()` — use the XVF3000 DOA to orient the OAK-D
toward whoever spoke. Future when we add pan/tilt hardware.

### Memory: cross-session summary
Summarize old turns into `memory/summary.md` after HISTORY_TTL_S
expires so "what did we talk about yesterday?" has an actual answer
beyond what's in `facts.md`.

### XTTS / F5-TTS upgrade
Voice quality ceiling on Kokoro 82M is real. XTTS v2 or F5-TTS
would be noticeably better but need AGX Orin (won't fit alongside
Gemma on 8 GB).

### Fix ffmpeg audio-post pipeline
Audio post-processing (EQ/de-esser/compressor) is config-toggleable
but the bash-shell wrapper is fragile. Rewrite as two chained
`subprocess.Popen` with explicit pipe fds instead of `/bin/bash -c`.

### Measure everything
Add a `scripts/benchmark.sh` that records a known utterance and
reports per-stage latency (STT, LLM first-token, TTS first-audio).
Useful for proving impact of any optimization.
