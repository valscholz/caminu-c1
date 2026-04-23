#!/usr/bin/env bash
# Install and enable the caminu systemd user service on this Jetson.
# Run this ONCE after ./install.sh has completed successfully.
#
# After this, caminu starts automatically on every boot and restarts
# on crash. Useful commands:
#
#   systemctl --user status caminu        # check if running
#   journalctl --user -u caminu -f        # tail the live log
#   systemctl --user restart caminu       # restart cleanly
#   systemctl --user stop caminu          # stop without disabling
#   systemctl --user disable caminu       # no longer start on boot
#
# Boot-time start requires lingering so the user session starts before login:
#   loginctl enable-linger $USER  (we do this below)
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
UNIT="$here/systemd/caminu.service"

if [ ! -f "$UNIT" ]; then
  echo "Missing $UNIT"; exit 1
fi

mkdir -p "$HOME/.config/systemd/user"
cp "$UNIT" "$HOME/.config/systemd/user/caminu.service"

systemctl --user daemon-reload
systemctl --user enable caminu.service

# Allow the user's systemd session to run without an active login so the
# service starts on boot, not on next SSH login.
sudo loginctl enable-linger "$USER"

echo ""
echo "caminu.service installed + enabled."
echo ""
echo "To start it now:"
echo "  systemctl --user start caminu"
echo ""
echo "On next boot it will start automatically (~40s after power-on)."
