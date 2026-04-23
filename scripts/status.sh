#!/usr/bin/env bash
# Quick health check — run locally on the Jetson.
set -euo pipefail

say() { printf "\n\033[1;34m%s\033[0m\n" "$*"; }

say "== processes =="
pgrep -af "caminu.main|llama-server" | grep -v grep || echo "(not running)"

say "== memory =="
free -h | head -2

say "== GPU / VRAM =="
if command -v tegrastats >/dev/null; then
  timeout 1 tegrastats | head -1
fi

say "== /health =="
curl -sf http://127.0.0.1:8080/health && echo || echo "(llama-server not responding)"

say "== latest agent log (last 15 lines) =="
LATEST=$(ls -t ~/caminu-c1/logs/agent*.log 2>/dev/null | head -1 || true)
[ -n "$LATEST" ] && tail -15 "$LATEST" || echo "(no log found)"

say "== caminu.service status =="
systemctl --user status caminu --no-pager 2>&1 | head -6 || echo "(service not installed; run scripts/install_service.sh)"
