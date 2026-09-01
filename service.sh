#!/usr/bin/env bash
# Quick start: reload systemd, start/stop/restart Omnivoice TTS service

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMD_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SYSTEMD_DIR/omnivoice-tts.service"

reload_systemd() {
    systemctl --user daemon-reload 2>/dev/null || true
}

case "${1:-status}" in
    start)
        echo "Starting OmniVoice TTS..."
        reload_systemd
        systemctl --user enable --now omnivoice-tts.service
        echo "Started. Check with: systemctl --user status omnivoice-tts.service"
        ;;
    stop)
        echo "Stopping OmniVoice TTS..."
        systemctl --user stop omnivoice-tts.service
        echo "Stopped."
        ;;
    restart)
        echo "Restarting OmniVoice TTS..."
        reload_systemd
        systemctl --user restart omnivoice-tts.service
        echo "Restarted."
        ;;
    status)
        systemctl --user status omnivoice-tts.service
        ;;
    logs)
        journalctl --user -u omnivoice-tts.service -f --no-pager
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
