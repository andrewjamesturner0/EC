#!/usr/bin/env bash
set -euo pipefail

WHISPER_DIR="$HOME/whisper.cpp"
MODEL="large-v3"

# Install only missing packages (avoids failing when sudo needs a password
# and everything is already present)
install_if_missing() {
    local pkgs=()
    for pkg in "$@"; do
        dpkg -s "$pkg" &>/dev/null || pkgs+=("$pkg")
    done
    if [ ${#pkgs[@]} -gt 0 ]; then
        echo "==> Installing missing packages: ${pkgs[*]}"
        sudo apt-get update -qq
        sudo apt-get install -y "${pkgs[@]}"
    else
        echo "==> All required packages already installed"
    fi
}

echo "==> Checking build dependencies..."
# libcurl4-openssl-dev is optional; we disable curl in cmake if absent
install_if_missing build-essential git cmake curl wget pkg-config ffmpeg

# Find CUDA installation and add to PATH if needed
CUDA_HOME_CANDIDATE=""
for d in /usr/local/cuda /usr/local/cuda-12.9 /usr/local/cuda-12; do
    if [ -x "$d/bin/nvcc" ]; then
        CUDA_HOME_CANDIDATE="$d"
        break
    fi
done

if command -v nvcc &>/dev/null; then
    echo "==> nvcc found in PATH: $(which nvcc)"
elif [ -n "$CUDA_HOME_CANDIDATE" ]; then
    echo "==> nvcc found at $CUDA_HOME_CANDIDATE/bin/nvcc, adding to PATH..."
    export PATH="$CUDA_HOME_CANDIDATE/bin:$PATH"
    export CUDA_HOME="$CUDA_HOME_CANDIDATE"
    export LD_LIBRARY_PATH="$CUDA_HOME_CANDIDATE/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    echo "==> Installing CUDA toolkit..."
    sudo apt-get install -y nvidia-cuda-toolkit
fi

echo "==> Cloning whisper.cpp..."
if [ -d "$WHISPER_DIR" ]; then
    echo "    Already exists, pulling latest..."
    git -C "$WHISPER_DIR" pull
else
    git clone https://github.com/ggerganov/whisper.cpp.git "$WHISPER_DIR"
fi

echo "==> Building whisper.cpp with CUDA support..."
cd "$WHISPER_DIR"

# Disable curl model fetching if libcurl-dev headers are not installed
CMAKE_EXTRA_FLAGS=""
if ! dpkg -s libcurl4-openssl-dev &>/dev/null; then
    echo "    libcurl4-openssl-dev not found, disabling curl support in build"
    CMAKE_EXTRA_FLAGS="-DGGML_CURL=OFF"
fi

cmake -B build -DGGML_CUDA=ON $CMAKE_EXTRA_FLAGS
cmake --build build -j$(nproc) --config Release

echo "==> Downloading ${MODEL} model..."
./models/download-ggml-model.sh "$MODEL"

MODEL_PATH="$WHISPER_DIR/models/ggml-${MODEL}.bin"
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

echo "==> Verifying GPU is detected..."
if nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found, GPU may not be available"
fi

echo "==> Testing with a short transcription..."
if command -v ffmpeg &>/dev/null; then
    # Generate 5 seconds of silence as a test
    ffmpeg -y -f lavfi -i anullsrc=r=16000:cl=mono -t 5 -ar 16000 /tmp/test_whisper.wav 2>/dev/null
    ./build/bin/whisper-cli -m "$MODEL_PATH" -f /tmp/test_whisper.wav --no-prints 2>&1 | tail -1
    rm -f /tmp/test_whisper.wav
else
    echo "    ffmpeg not available, skipping transcription test"
fi

cat <<EOF

============================================
whisper.cpp is ready!

Binary:  $WHISPER_DIR/build/bin/whisper-cli
Model:   $MODEL_PATH

Example usage:
  # Single file (auto-converts to 16kHz WAV):
  ffmpeg -i input.mp3 -ar 16000 -ac 1 /tmp/audio.wav
  $WHISPER_DIR/build/bin/whisper-cli -m $MODEL_PATH -f /tmp/audio.wav -of output -otxt

  # Batch transcribe all episodes:
  for f in episodes/*.mp3; do
    name=\$(basename "\$f" .mp3)
    ffmpeg -y -i "\$f" -ar 16000 -ac 1 /tmp/whisper_in.wav
    $WHISPER_DIR/build/bin/whisper-cli -m $MODEL_PATH -f /tmp/whisper_in.wav -of "transcripts/\$name" -otxt
    echo "Done: \$name"
  done
============================================
EOF
