#!/usr/bin/env bash
# OmniVoice TTS — installation script
# Creates venv, installs dependencies, downloads model, and sets up systemd --user service.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
VOICES_DIR="$HOME/.omnivoice/voices"

echo "=== OmniVoice TTS Setup ==="

# 1. Create virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
fi

# 2. Install dependencies
echo "[2/5] Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip
pip install -r "$PROJECT_DIR/app/requirements.txt"

# 3. Download model (first time)
echo "[3/5] Downloading OmniVoice model (first run only, ~5-10 GB)..."
python3 -c "
from huggingface_hub import snapshot_download
print('Downloading k2-fsa/OmniVoice...')
snapshot_download(repo_id='k2-fsa/OmniVoice', cache_dir='$HOME/.cache/huggingface')
print('Model downloaded.')
"

# 4. Create voices directory
echo "[4/5] Creating voices directory..."
mkdir -p "$VOICES_DIR"
echo "Voice files go in: $VOICES_DIR"
echo "Place reference audio (.wav/.mp3/.flac) and matching .txt files there."

# 5. Set up systemd --user service
echo "[5/5] Setting up systemd --user service..."

SYSTEMD_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_DIR"

cat > "$SYSTEMD_DIR/omnivoice-tts.service" <<EOF
[Unit]
Description=OmniVoice TTS Service (systemd user)
After=network.target

[Service]
Type=simple
ExecStart=$VENV_DIR/bin/python $PROJECT_DIR/app/server.py
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=$PROJECT_DIR/.env
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=default.target
EOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Place reference audio files in: $VOICES_DIR"
echo "  2. Enable the service:"
echo "     systemctl --user daemon-reload"
echo "     systemctl --user enable --now omnivoice-tts.service"
echo ""
echo "  3. Check status:"
echo "     systemctl --user status omnivoice-tts.service"
echo ""
echo "  4. API available at: http://localhost:8001"
echo "     - /health          — health check"
echo "     - /v1/models       — list models (OpenAI compatible)"
echo "     - /v1/voices       — list registered voices"
echo "     - POST /v1/audio/speech  — generate speech"
echo "     - POST /v1/voices/register — register a voice"
echo "     - DELETE /v1/voices/{name} — delete a voice"
echo ""
echo "  5. OpenAPI docs at: http://localhost:8001/docs"
echo ""
