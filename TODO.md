# Ideas to try after v1 is working

## "Thinking as a tool"
Expose a `think_hard(question)` tool to Gemma. The system prompt instructs
Gemma to call it for genuinely complex questions. Our loop re-invokes
Gemma on that sub-query with `chat_template_kwargs.enable_thinking=true`
and returns only the final `content` as the tool result. User hears a
concise spoken answer; the reasoning happens invisibly and only when
it actually helps. Combines fast responses for simple turns with deep
reasoning for hard ones — no perceptible latency penalty on the common
case.

## Streaming TTS
Pipe Gemma's token stream through Kokoro sentence-by-sentence so the first
words start speaking before the full reply finishes.

## Custom wake word "hey C1"
Train an openWakeWord model on "hey C1" in the Colab (~1-2h once). Commit
the .onnx to repo, swap `WAKE_MODEL` in config.

## ReSpeaker DOA as a tool
ReSpeaker v3 exposes direction-of-arrival. Add a `look_at_speaker()` tool
that points the OAK-D toward wherever the voice came from before grabbing
a frame — better selfies than "the camera's wherever it's mounted."

## Barge-in / interruption
Listen during TTS so the user can cut the agent off mid-sentence.
Requires AEC (the ReSpeaker already does some) and lower wake threshold
during TTS.

## Memory
Short-term summary after long conversations, persisted to a sqlite file.
Makes "what did we talk about yesterday?" possible.
