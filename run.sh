#!/usr/bin/env bash
# run.sh — start llama-server + the Python agent loop.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p logs
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}

GGUF="models/gguf/gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ="models/gguf/mmproj-F16.gguf"

for f in "$GGUF" "$MMPROJ" llama.cpp/build/bin/llama-server; do
  [ -e "$f" ] || { echo "Missing: $f — run ./install.sh first"; exit 1; }
done

# Ensure default PulseAudio sink is the ReSpeaker (so TTS goes to the Monk Makes)
pactl set-default-sink \
  alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-stereo 2>/dev/null || true
pactl set-sink-volume \
  alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-stereo 100% 2>/dev/null || true

# Start llama-server in the background
echo ">>> Starting llama-server"
./llama.cpp/build/bin/llama-server \
  -m "$GGUF" \
  --mmproj "$MMPROJ" \
  --jinja \
  -ngl 99 \
  -c 4096 \
  -np 1 \
  --host 127.0.0.1 \
  --port 8080 \
  >> logs/llama-server.log 2>&1 &
LLAMA_PID=$!
trap 'kill $LLAMA_PID 2>/dev/null || true' EXIT

# Wait for /health
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8080/health >/dev/null; then
    echo ">>> llama-server ready after ${i}s"
    break
  fi
  sleep 1
done

echo ">>> Launching caminu agent"
exec python -m caminu.main
