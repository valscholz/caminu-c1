"""Tiny timestamped stdout logger. Zero deps."""
import sys
import time


def log(*args) -> None:
    t = time.strftime("%H:%M:%S")
    print(f"[{t}]", *args, file=sys.stderr, flush=True)
