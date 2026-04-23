#!/usr/bin/env bash
# install.sh — idempotent bring-up for caminu-c1 on Jetson Orin Nano Super (JP6.2, aarch64).
# Re-runnable; skips steps already done.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

say() { printf "\n\033[1;34m>>> %s\033[0m\n" "$*"; }

# 1. Preflight -----------------------------------------------------------------
say "Preflight"
[ "$(uname -m)" = "aarch64" ] || { echo "This script is for aarch64 Jetson only."; exit 1; }
[ -f /etc/nv_tegra_release ] || { echo "No /etc/nv_tegra_release — is this a Jetson?"; exit 1; }
head -1 /etc/nv_tegra_release

# 2. apt deps ------------------------------------------------------------------
say "Installing apt packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential cmake ninja-build pkg-config \
  python3-venv python3-pip python3-dev \
  portaudio19-dev libportaudio2 libasound2-dev alsa-utils \
  libusb-1.0-0-dev libudev-dev \
  ffmpeg sox curl git wget \
  openssh-server avahi-daemon \
  cuda-toolkit-12-6

# 3. CUDA PATH -----------------------------------------------------------------
say "Ensuring CUDA on PATH in ~/.bashrc"
if ! grep -q "cuda-12.6/bin" "$HOME/.bashrc"; then
  cat >> "$HOME/.bashrc" <<'EOF'

# CUDA 12.6 on PATH (caminu-c1)
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}
EOF
fi
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}
nvcc --version | tail -1

# 4. User groups ---------------------------------------------------------------
say "Adding user to audio/video/plugdev/dialout groups"
sudo usermod -aG audio,video,plugdev,dialout "$USER" || true

# 5. OAK-D udev rule -----------------------------------------------------------
say "Installing OAK-D (Luxonis / Movidius) udev rule"
RULE='/etc/udev/rules.d/80-movidius.rules'
if ! sudo test -f "$RULE"; then
  echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' \
    | sudo tee "$RULE" >/dev/null
  sudo udevadm control --reload-rules
  sudo udevadm trigger
fi

# 6. SSH + avahi (for remote control) -----------------------------------------
say "Ensuring ssh + avahi are running"
sudo systemctl enable --now ssh avahi-daemon 2>/dev/null || true

# 7. Swap ----------------------------------------------------------------------
say "Ensuring 8 GB swap is available"
if [ "$(free -g | awk '/Swap/ {print $2}')" -lt 7 ]; then
  if [ ! -f /swapfile ]; then
    sudo fallocate -l 8G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
  fi
  sudo swapon /swapfile 2>/dev/null || true
  if ! grep -q "/swapfile" /etc/fstab; then
    echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
  fi
fi
free -h | head -2

# 8. Build llama.cpp with CUDA -------------------------------------------------
say "Building llama.cpp with CUDA for Orin (SM 87)"
if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
fi
if [ ! -x llama.cpp/build/bin/llama-server ]; then
  cmake -S llama.cpp -B llama.cpp/build -G Ninja \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=ON
  cmake --build llama.cpp/build --config Release -j"$(nproc)"
fi
llama.cpp/build/bin/llama-cli --version | head -1 || true

# 9. Python venv + deps --------------------------------------------------------
say "Creating Python venv"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip wheel
pip install --quiet -r requirements.txt

# 10. Model downloads ----------------------------------------------------------
say "Downloading Gemma 4 E2B Q4_K_M GGUF + mmproj"
mkdir -p models/gguf models/kokoro models/openwakeword
dl() {
  local url="$1" dest="$2"
  [ -s "$dest" ] && { echo "  skip $(basename "$dest") (already present)"; return; }
  echo "  fetching $(basename "$dest")"
  wget -q --show-progress -O "$dest.part" "$url"
  mv "$dest.part" "$dest"
}
dl https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf \
   models/gguf/gemma-4-E2B-it-Q4_K_M.gguf
dl https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf \
   models/gguf/mmproj-F16.gguf

say "Downloading Kokoro TTS"
dl https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
   models/kokoro/kokoro-v1.0.onnx
# NOTE: kokoro-onnx 0.3.3 wants the older voices.json (11 voices, JSON format),
# not the v1.0 .bin bundle (which targets a newer unreleased kokoro-onnx build).
dl https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json \
   models/kokoro/voices.json

say "Downloading openWakeWord helper models (hey_jarvis is prebuilt, mel+embedding fetched on first use)"
python3 -c "import openwakeword.utils as u; u.download_models(['hey_jarvis'])"

say "Warming faster-whisper base.en (one-time)"
python3 -c "from faster_whisper import WhisperModel; m = WhisperModel('base.en', device='cpu', compute_type='int8'); print('ok')"

# 11. Verification -------------------------------------------------------------
say "Verification"
arecord -l 2>/dev/null | grep -i respeaker && echo "  [ok] ReSpeaker detected" || echo "  [WARN] ReSpeaker NOT detected"
python3 -c "import depthai as dai; d = dai.Device.getAllAvailableDevices(); print(f'  [ok] OAK-D: {len(d)} device(s)')" \
  || echo "  [WARN] OAK-D not detected"

say "Done. Log out and back in if this was your first run (group changes). Then:  ./run.sh"
