"""
Every tunable knob in one place.

Defaults live in `MonitorConfig`. To override any of them, create `config.toml`
next to monitor.py (see config.example.toml) — only the keys you set change.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEVERITIES = ("info", "warning", "critical")


@dataclass
class MonitorConfig:
    # ── Paths ─────────────────────────────────────────────────────────
    data_dir: str = "data"                  # relative paths resolve against the project root

    # ── Sampling ──────────────────────────────────────────────────────
    sample_interval_sec: float = 1.0        # system-wide telemetry rate
    window_sec: int = 10                    # samples are averaged into windows; the ML scores windows
    process_interval_sec: float = 5.0       # per-process sweep (≈5 ms on Windows)

    # ── Storage / retention ───────────────────────────────────────────
    raw_retention_hours: float = 48         # 1 Hz samples (≈20 MB/day)
    window_retention_days: float = 14       # 10 s windows — the ML training set
    event_retention_days: float = 180

    # ── ML anomaly detection (kNN distance to your own history) ──────
    #  alert_quantile = how rare a window must be (vs. your own history) to
    #  count as anomalous. 0.001 → rarer than 99.9 % of past windows.
    #    more alerts  ← 0.005 · 0.002 · 0.001 · 0.0005 →  fewer alerts
    #  min_consecutive_windows then requires the rarity to *persist*
    #  (3 × 10 s = 30 s), which is what removes nearly all noise.
    #  `python monitor.py tune` shows what each setting would have done on your history.
    alert_quantile: float = 0.001
    min_consecutive_windows: int = 3
    close_after_normal_windows: int = 6     # 60 s of normality ends an episode (merges flapping)
    max_event_minutes: float = 30           # longer "anomalies" are a new normal; stop and cool down
    ml_cooldown_min: float = 10             # quiet period after an ML episode ends
    training_days: float = 7                # history the model learns from (covers all your regimes)
    max_training_windows: int = 30_000      # random subsample cap (training cost grows with n²)
    min_training_hours: float = 1.0         # data needed before ML detection starts
    retrain_interval_min: float = 60
    grace_period_sec: float = 120           # ML stays quiet after start-up / resume from sleep
    ml_mature_hours: float = 24             # ML events stay "info" until the model has this much history

    # ── Rule-based detectors ──────────────────────────────────────────
    ram_pressure_percent: float = 90
    commit_pressure_percent: float = 90     # Windows commit charge (RAM + pagefile)
    memory_pressure_sec: float = 60

    low_disk_free_gb: float = 20            # system drive: warn when less than this is free…
    low_disk_free_percent: float = 10       # …or less than this share of the drive (0 turns a test off)

    gpu_temp_warn_c: float = 87             # RTX 4070 Laptop target max is 87 °C
    gpu_throttle_min_temp_c: float = 75     # driver "thermal" flags below this are ignored (spurious at idle)
    gpu_thermal_sec: float = 30             # sustained thermal throttling / overheat

    background_cpu_cores: float = 0.8       # a non-foreground process averaging ≥ this many logical cores…
    background_cpu_minutes: float = 5       # …for this long → "background CPU hog"

    leak_min_history_hours: float = 1.0     # need this much history per process before judging
    leak_history_hours: float = 6.0         # trend window
    leak_min_growth_mb: float = 500         # private memory growth to call it a leak
    handle_leak_min_growth: int = 3_000     # handle growth to call it a leak
    handle_absolute_limit: int = 100_000    # any process above this is leaking, full stop
    handle_growth_min_count: int = 10_000   # growth alone only matters once a process holds this many

    gpu_awake_idle_minutes: float = 10      # on battery: dGPU awake but idle → battery drain

    upload_burst_mb: float = 20             # this much sent within…  (0 turns the rule off)
    upload_burst_sec: float = 30            # …this many seconds is an upload burst (≈1/day here)

    ui_freeze_probe: bool = True            # time how fast the foreground app answers window messages
    ui_freeze_sec: float = 1.0              # an app (or the desktop) not answering for this long has frozen
    ui_probe_interval_sec: float = 0.5      # how often it is asked (each probe costs microseconds)

    ignore_processes: list[str] = field(default_factory=list)  # names (case-insensitive) never blamed

    # ── Notifications ─────────────────────────────────────────────────
    notify: bool = False                    # Windows toast notifications
    notify_min_severity: str = "warning"

    # ── AI / Conversational Diagnostics ──────────────────────────────
    ai_provider: str = "auto"              # "auto", "ollama", "openai", "offline"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = ""

    # ── Behaviour ─────────────────────────────────────────────────────
    efficiency_mode: bool = True            # run on E-cores at low priority (Windows 11 EcoQoS)
    log_level: str = "INFO"

    # Problems found while loading config.toml (logged once logging is up).
    warnings: list[str] = field(default_factory=list, init=False, repr=False)

    # ── Derived ───────────────────────────────────────────────────────
    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def validate(self) -> None:
        if not 0 < self.alert_quantile < 0.5:
            raise ValueError("alert_quantile must be in (0, 0.5)")
        if self.window_sec < 2 * self.sample_interval_sec:
            raise ValueError("window_sec must hold at least two samples")
        if self.min_consecutive_windows < 1 or self.close_after_normal_windows < 1:
            raise ValueError("window counts must be ≥ 1")
        if self.low_disk_free_gb < 0 or not 0 <= self.low_disk_free_percent < 100:
            raise ValueError("low_disk_free_gb must be ≥ 0 and low_disk_free_percent in [0, 100)")
        if self.upload_burst_mb < 0 or not 2 * self.sample_interval_sec <= self.upload_burst_sec <= 600:
            raise ValueError("upload_burst_mb must be ≥ 0 and upload_burst_sec in [2 samples, 600]")
        if not 0.25 <= self.ui_freeze_sec <= 60 or not 0.1 <= self.ui_probe_interval_sec <= 5:
            raise ValueError("ui_freeze_sec must be in [0.25, 60] and ui_probe_interval_sec in [0.1, 5]")
        if self.notify_min_severity not in SEVERITIES:
            raise ValueError(f"notify_min_severity must be one of {SEVERITIES}")
        if self.ai_provider not in ("auto", "ollama", "openai", "offline"):
            raise ValueError("ai_provider must be one of: 'auto', 'ollama', 'openai', 'offline'")
        self.ignore_processes = [n.lower() for n in self.ignore_processes]


def load_config(path: Path | None = None, **overrides) -> MonitorConfig:
    """Defaults ← config.toml (if present) ← keyword overrides."""
    path = path or PROJECT_ROOT / "config.toml"
    values: dict = {}
    if path.exists():
        with open(path, "rb") as f:
            values = tomllib.load(f)
    values.update({k: v for k, v in overrides.items() if v is not None})

    known = {f.name for f in fields(MonitorConfig) if f.init}
    unknown = sorted(set(values) - known)
    cfg = MonitorConfig(**{k: v for k, v in values.items() if k in known})
    if unknown:
        cfg.warnings.append(f"Ignoring unknown keys in {path.name}: {', '.join(unknown)}")
    cfg.validate()
    return cfg
