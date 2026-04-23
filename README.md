# caminu-c1

*The first Caminu Robot called C1.*

A voice + vision AI agent running on an NVIDIA Jetson Orin Nano Super Developer Kit.

The agent:
- **hears** through a ReSpeaker Mic Array v3 (USB)
- **sees** through an OAK-D W (Luxonis, USB)
- **speaks** through a Monk Makes Speaker v2 (analog, powered from GPIO)
- **thinks** using Gemma 4 E2B (on-device, via llama.cpp + CUDA)

## Bill of materials

- Jetson Orin Nano Super Developer Kit (8 GB, JetPack 6.2)
- Seeed ReSpeaker Mic Array v3 (USB)
- Luxonis OAK-D W (USB-C)
- Monk Makes Speaker v2 (3.5 mm audio in, 5 V DC power)
- 3.5 mm ground-loop isolator (recommended — eliminates hum between ReSpeaker and speaker)

## Install on the Jetson

```bash
git clone https://github.com/valscholz/caminu-c1.git
cd caminu-c1
./install.sh         # idempotent: builds llama.cpp + CUDA, installs deps, downloads models
./run.sh             # starts llama-server + the agent loop
```

Say **"Hey Jarvis, what time is it?"** to try a non-visual turn.
Say **"Hey Jarvis, how many fingers am I holding up?"** to trigger a camera tool call.

## Architecture

```
wake word ─► STT ─► Gemma 4 E2B ─► TTS ─► speaker
(openWW)  (Whisper) (llama.cpp)  (Kokoro)
                        │
                        └─► tool calls: take_picture(), get_time()
                                            │
                                            └─► OAK-D frame ──► Gemma
```

Gemma 4 E2B is natively multimodal with tool calling. Rather than attaching
a camera frame to every turn (expensive on 8 GB), we expose `take_picture()` as
a tool. Gemma decides when it needs to see — on visual questions it emits a
tool call, we grab one OAK-D frame, and Gemma answers using the image.

## Hardware notes

- **ReSpeaker v3** enumerates as `card 0 ArrayUAC10` — a 6-channel capture
  device. Channel 0 is the beamformed / AEC-processed output; that's the
  one used for STT. Its 3.5 mm jack is the audio source for the Monk Makes.
- **Monk Makes** takes its signal from the ReSpeaker's 3.5 mm out and its
  power (5 V, GND) from the Jetson 40-pin header.
- **OAK-D W** uses the Luxonis udev rule (`/etc/udev/rules.d/80-movidius.rules`).

## License

MIT
