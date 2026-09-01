#!/usr/bin/env bash
# Run directly as: ./run.sh [IP] [PORT]. If PORT is omitted, choose a free port.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename -- "$PROJECT_DIR")"
VENV_DIR="${VENV_DIR:-$HOME/venv/$PROJECT_NAME}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "error: not installed; run $PROJECT_DIR/install.sh first" >&2
    exit 1
fi

HOST="${1:-${OMNIVOICE_HOST:-0.0.0.0}}"
if [[ -n "${2:-}" ]]; then
    PORT="$2"
elif [[ -n "${OMNIVOICE_PORT:-}" ]]; then
    PORT="$OMNIVOICE_PORT"
else
    PORT="$($VENV_DIR/bin/python - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)"
fi

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then
    echo "error: PORT must be an integer from 1 to 65535" >&2
    exit 2
fi

echo "OmniVoice TTS listening on http://$HOST:$PORT"
ENV_ARGS=()
[[ -f "$PROJECT_DIR/.env" ]] && ENV_ARGS=(--env-file "$PROJECT_DIR/.env")
exec "$VENV_DIR/bin/python" -m uvicorn app.server:app "${ENV_ARGS[@]}" --host "$HOST" --port "$PORT"
