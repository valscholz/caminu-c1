#!/usr/bin/env python3
"""One-shot: render the STARTUP_LINES_INSTANT to assets/boot_*.wav.

Run once after install (install.sh does this automatically). Re-run
after changing the STARTUP_LINES_INSTANT pool in announcements.py to
refresh the cached chimes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from caminu.announcements import regenerate_boot_wavs

if __name__ == "__main__":
    regenerate_boot_wavs()
