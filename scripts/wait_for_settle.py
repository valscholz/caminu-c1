#!/usr/bin/env python3
"""Wait for the system to finish its boot-time memory churn before we start.

Polls /proc/meminfo. When MemAvailable has been roughly stable for
SETTLE_WINDOW_S (±STABLE_DELTA_MB), we declare the system settled and exit 0.
Hard ceiling of MAX_WAIT_S so a weird boot never blocks caminu forever.

Called by systemd as ExecStartPre so caminu.service only launches once
Linux has stopped churning. On a fast boot this returns in ~5s; on a
slow boot it may take 30-60s. Either way better than a blind sleep.
"""
from __future__ import annotations
import sys
import time

POLL_INTERVAL_S = 2.0      # how often we re-check
SETTLE_WINDOW_S = 10.0     # memory must be stable for this long
STABLE_DELTA_MB = 50       # MemAvailable may only wiggle this much during the window
MAX_WAIT_S = 90            # hard ceiling so we never block forever
MIN_WAIT_S = 3             # always wait at least this long (catches too-early runs)


def mem_available_mb() -> float:
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def main() -> int:
    t0 = time.time()
    history: list[tuple[float, float]] = []
    print(f"wait_for_settle: polling, target window={SETTLE_WINDOW_S:.0f}s "
          f"delta={STABLE_DELTA_MB}MB max={MAX_WAIT_S}s", flush=True)

    while True:
        now = time.time()
        elapsed = now - t0
        avail = mem_available_mb()
        history.append((now, avail))
        # drop samples older than the settle window
        cutoff = now - SETTLE_WINDOW_S
        history = [(t, v) for (t, v) in history if t >= cutoff]

        if elapsed >= MAX_WAIT_S:
            print(f"wait_for_settle: hit MAX_WAIT_S={MAX_WAIT_S}s, starting anyway "
                  f"(avail={avail:.0f}MB)", flush=True)
            return 0

        if elapsed >= MIN_WAIT_S and history:
            mn = min(v for _t, v in history)
            mx = max(v for _t, v in history)
            if (history[0][0] <= now - SETTLE_WINDOW_S + POLL_INTERVAL_S
                    and (mx - mn) < STABLE_DELTA_MB):
                print(f"wait_for_settle: settled after {elapsed:.0f}s "
                      f"(avail={avail:.0f}MB, delta={mx-mn:.0f}MB)", flush=True)
                return 0

        print(f"wait_for_settle: waiting… t={elapsed:.0f}s avail={avail:.0f}MB", flush=True)
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    sys.exit(main())
