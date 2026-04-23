"""Tiny timestamped stdout logger. Zero deps."""
import sys
import time


def log(*args) -> None:
    t = time.strftime("%H:%M:%S")
    print(f"[{t}]", *args, file=sys.stderr, flush=True)


def _read_meminfo() -> dict[str, int]:
    """Parse /proc/meminfo into {key: kb}. Linux only."""
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                value = parts[1].strip().split()
                if value and value[0].isdigit():
                    out[key] = int(value[0])
    except OSError:
        pass
    return out


def mem_snapshot() -> str:
    """Return a short human-readable RAM snapshot: 'avail=X.XG used=Y.YG total=Z.ZG'."""
    m = _read_meminfo()
    total = m.get("MemTotal", 0) / 1024 / 1024
    avail = m.get("MemAvailable", 0) / 1024 / 1024
    used = total - avail
    return f"avail={avail:.1f}G used={used:.1f}G total={total:.1f}G"


def log_mem(tag: str) -> None:
    """Log a memory snapshot line tagged with the lifecycle stage."""
    log(f"[mem:{tag}] {mem_snapshot()}")
