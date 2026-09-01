#!/usr/bin/env bash
# Idempotent install/upgrade for OmniVoice TTS.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename -- "$PROJECT_DIR")"
VENV_DIR="${VENV_DIR:-$HOME/venv/$PROJECT_NAME}"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_PATH="$UNIT_DIR/$PROJECT_NAME.service"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv installation did not provide an executable" >&2
    exit 1
fi

echo "Installing/upgrading $PROJECT_NAME in $VENV_DIR"
uv python install 3.12
uv venv --python 3.12 --allow-existing "$VENV_DIR"
uv pip install --python "$VENV_DIR/bin/python" \
    --upgrade-package omnivoice --upgrade-package fastapi --upgrade-package uvicorn \
    -r "$PROJECT_DIR/app/requirements.txt"

if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    TOKEN="$($VENV_DIR/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
    sed -i "s|^OMNIVOICE_API_TOKEN=.*|OMNIVOICE_API_TOKEN=$TOKEN|" "$PROJECT_DIR/.env"
    chmod 600 "$PROJECT_DIR/.env"
    echo "Generated API token in $PROJECT_DIR/.env"
fi
mkdir -p "${VOICES_DIR:-$HOME/.omnivoice/voices}" "$UNIT_DIR"

cat >"$UNIT_PATH" <<EOF
[Unit]
Description=OmniVoice TTS ($PROJECT_NAME)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR
Environment=VENV_DIR=$VENV_DIR
Environment=OMNIVOICE_HOST=0.0.0.0
Environment=OMNIVOICE_PORT=8001
EnvironmentFile=-$PROJECT_DIR/.env
ExecStart=$PROJECT_DIR/run.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

if command -v systemctl >/dev/null 2>&1 && systemctl --user daemon-reload; then
    if systemctl --user is-active --quiet "$PROJECT_NAME.service"; then
        systemctl --user restart "$PROJECT_NAME.service"
    fi
else
    echo "warning: systemd user manager unavailable; use ./run.sh" >&2
fi

echo "Installed. Start with: systemctl --user enable --now $PROJECT_NAME.service"
echo "Or run directly:    ./run.sh [IP] [PORT]"
