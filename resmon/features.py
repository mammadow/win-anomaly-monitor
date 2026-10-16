"""
1 Hz samples → 10 s windows → ML feature vectors.

Why windows instead of scoring every second (as v2 did):
  * Interesting problems are sustained (a runaway process, a leak, a hot GPU),
    not 1-second blips. Averaging removes most of the per-second noise that made
    v2 alert ~once a minute while the laptop sat idle.
  * It cuts the training set 10×, so a full week of history (every regime you
    use — idle, gaming, battery…) fits in the model instead of just 4 hours.

Byte-rate features are log-scaled: bytes/s span 0 … 1.2 GB/s on this
machine, and 1 → 100 MB/s matters as much as 100 MB/s → 10 GB/s would.
log10(1 + rate / 64 KB/s) keeps trickles (0 vs 35 KB/s) from looking like
huge jumps while staying logarithmic above ~1 MB/s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

USER_ACTIVE_IDLE_SEC = 60.0   # input within the last minute → user is at the machine
RATE_KNEE_BPS = 65_536.0      # byte rates below ~64 KB/s all count as "about zero"

WINDOW_COLUMNS = [
    "ts", "n",
    "cpu_pct", "cpu_max_pct", "core_max_pct", "core_max_max_pct",
    "ram_pct", "commit_pct",
    "disk_read_bps", "disk_write_bps", "net_sent_bps", "net_recv_bps",
    "on_ac", "battery_pct", "user_active",
    "gpu_awake", "gpu_util_pct", "gpu_vram_pct", "gpu_temp_c", "gpu_power_w", "gpu_clock_mhz",
    "handles", "processes",
]


@dataclass(frozen=True)
class Feature:
    column: str
    label: str
    unit: str                   # %, bps, W, C, MHz, state
    log: bool = False
    states: tuple[str, str] = ("", "")   # for unit == "state": (label at 0, label at 1)


ML_FEATURES: list[Feature] = [
    Feature("cpu_pct", "CPU", "%"),
    Feature("core_max_pct", "busiest core", "%"),
    Feature("ram_pct", "RAM", "%"),
    Feature("commit_pct", "commit charge", "%"),
    Feature("disk_read_bps", "disk read", "bps", log=True),
    Feature("disk_write_bps", "disk write", "bps", log=True),
    Feature("net_sent_bps", "upload", "bps", log=True),
    Feature("net_recv_bps", "download", "bps", log=True),
    Feature("gpu_util_pct", "GPU load", "%"),
    Feature("gpu_power_w", "GPU power", "W"),
    Feature("gpu_temp_c", "GPU temp", "C"),
    Feature("gpu_vram_pct", "VRAM", "%"),
    Feature("gpu_clock_mhz", "GPU clock", "MHz"),
    # Regime context: "similar moments" must match on these too, so "gaming on AC while
    # I'm here" and "idle on battery while I'm away" are each compared with their own kind.
    Feature("gpu_awake", "dGPU", "state", states=("asleep", "awake")),
    Feature("on_ac", "power", "state", states=("battery", "AC")),
    Feature("user_active", "user", "state", states=("away", "active")),
]
FEATURE_NAMES = [f.column for f in ML_FEATURES]
FEATURE_BY_COLUMN = {f.column: f for f in ML_FEATURES}
_TEMP, _AWAKE = FEATURE_NAMES.index("gpu_temp_c"), FEATURE_NAMES.index("gpu_awake")


def _mean(vals: list[float]) -> float:
    good = [v for v in vals if not math.isnan(v)]
    return sum(good) / len(good) if good else math.nan


def _max(vals: list[float]) -> float:
    good = [v for v in vals if not math.isnan(v)]
    return max(good) if good else math.nan


class WindowAggregator:
    """Collects samples into wall-clock-aligned windows (…:00, :10, :20 …)."""

    def __init__(self, window_sec: int, sample_interval_sec: float = 1.0):
        self.window_sec = window_sec
        self.min_samples = max(2, int(0.6 * window_sec / sample_interval_sec))
        self._bucket: float | None = None
        self._samples: list[dict] = []

    def reset(self) -> None:
        self._bucket, self._samples = None, []

    def add(self, s: dict) -> dict | None:
        """Add a sample; returns the previous window when a new one starts."""
        bucket = math.floor(s["ts"] / self.window_sec) * self.window_sec
        done = None
        if self._bucket is not None and bucket != self._bucket:
            done = self._finish()
        if bucket != self._bucket:
            self._bucket, self._samples = bucket, []
        self._samples.append(s)
        return done

    def _finish(self) -> dict | None:
        ss = self._samples
        if len(ss) < self.min_samples:    # partial window (start-up, stall) — skip
            return None
        col = lambda k: [s[k] for s in ss]
        idle = [v for v in col("user_idle_sec") if not math.isnan(v)]
        return {
            "ts": self._bucket,
            "n": len(ss),
            "cpu_pct": _mean(col("cpu_pct")),
            "cpu_max_pct": _max(col("cpu_pct")),
            "core_max_pct": _mean(col("core_max_pct")),
            "core_max_max_pct": _max(col("core_max_pct")),
            "ram_pct": _mean(col("ram_pct")),
            "commit_pct": _mean(col("commit_pct")),
            "disk_read_bps": _mean(col("disk_read_bps")),
            "disk_write_bps": _mean(col("disk_write_bps")),
            "net_sent_bps": _mean(col("net_sent_bps")),
            "net_recv_bps": _mean(col("net_recv_bps")),
            "on_ac": _mean(col("on_ac")),
            "battery_pct": ss[-1]["battery_pct"],
            "user_active": (sum(v < USER_ACTIVE_IDLE_SEC for v in idle) / len(idle)) if idle else math.nan,
            "gpu_awake": _mean(col("gpu_awake")),
            "gpu_util_pct": _mean(col("gpu_util_pct")),
            "gpu_vram_pct": _mean(col("gpu_vram_pct")),
            "gpu_temp_c": _mean(col("gpu_temp_c")),      # mean over awake samples only
            "gpu_power_w": _mean(col("gpu_power_w")),
            "gpu_clock_mhz": _mean(col("gpu_clock_mhz")),
            "handles": ss[-1]["handles"],
            "processes": ss[-1]["processes"],
        }


def to_matrix(rows: np.ndarray | list[dict]) -> np.ndarray:
    """
    Window rows → model input. Accepts a list of dicts or a 2-D array whose
    columns are FEATURE_NAMES in order. NaN (GPU asleep/absent, unknown) → 0.
    """
    if isinstance(rows, np.ndarray):
        X = rows.astype(np.float64, copy=True)
    else:
        X = np.array([[r.get(c, math.nan) for c in FEATURE_NAMES] for r in rows], dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    for j, f in enumerate(ML_FEATURES):
        if f.log:
            X[:, j] = np.log10(1.0 + np.clip(X[:, j], 0, None) / RATE_KNEE_BPS)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Window temperature is averaged over awake samples only; weight it by the awake
    # fraction so it behaves like the other GPU features (which count asleep as 0) and
    # a window where the dGPU woke for 1 s isn't "45 °C while 90 % asleep".
    X[:, _TEMP] *= X[:, _AWAKE]
    return X


# ── Human-readable formatting ────────────────────────────────────────

def fmt_bytes(n: float, per_sec: bool = False) -> str:
    suffix = "/s" if per_sec else ""
    if n is None or math.isnan(n):
        return "n/a"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1000 or unit == "GB":
            return f"{n:.0f} {unit}{suffix}" if unit == "B" else f"{n:.1f} {unit}{suffix}"
        n /= 1000
    return f"{n:.1f} TB{suffix}"


def fmt_feature(f: Feature, v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a" if f.column != "gpu_temp_c" else "n/a (GPU asleep)"
    match f.unit:
        case "%":
            return f"{v:.0f}%"
        case "bps":
            return fmt_bytes(v, per_sec=True)
        case "W":
            return f"{v:.0f} W"
        case "C":
            return f"{v:.0f}°C"
        case "MHz":
            return f"{v:.0f} MHz"
        case "state":
            if v <= 0.05:
                return f.states[0]
            if v >= 0.95:
                return f.states[1]
            return f"{f.states[1]} {v * 100:.0f}% of the time"
    return f"{v:g}"
