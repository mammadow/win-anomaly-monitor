"""
Replay recorded windows through the ML pipeline offline.

Used by `python monitor.py tune` to answer "how many alerts per day would
setting X have produced on *my* history?", and by the tests. It runs the
exact production episode logic (rules.AnomalyEpisodes) with periodic
retraining, so its output matches what the live monitor would have done.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .config import MonitorConfig
from .detector import AnomalyModel, fit_model
from .events import Event
from .features import FEATURE_NAMES, to_matrix
from .rules import AnomalyEpisodes


class _RecordingEvents:
    """EventManager stand-in that keeps events in memory."""

    def __init__(self):
        self.events: list[Event] = []
        self.open_events: dict = {}

    def open(self, kind, severity, title, ts, details):
        ev = Event(kind, severity, title, ts, ts, dict(details), id=len(self.events) + 1)
        self.events.append(ev)
        return ev

    def update(self, ev, ts, *, title=None, severity=None, details=None):
        ev.ts_end = max(ev.ts_end, ts)
        if title:
            ev.title = title
        if severity == "warning":
            ev.severity = severity
        if details:
            ev.details.update(details)

    def reopen(self, ev, ts):
        ev.is_open = True
        ev.ts_end = max(ev.ts_end, ts)
        ev.details["recurrences"] = ev.details.get("recurrences", 0) + 1

    def close(self, ev, ts, *, title=None, severity=None, details=None):
        self.update(ev, ts, title=title, severity=severity, details=details)
        ev.is_open = False


class _ReplayMonitor:
    def __init__(self, cfg: MonitorConfig):
        self.cfg = cfg
        self.model: AnomalyModel | None = None
        self.events = _RecordingEvents()

    @staticmethod
    def activity(since: float) -> dict:
        return {"top_cpu": [], "top_io": [], "top_memory": [], "gpu_clients": [],
                "foreground": None, "user_idle": False}


@dataclass
class ReplayResult:
    events: list[Event]
    days: float
    scored_windows: int
    anomalous_windows: int

    @property
    def per_day(self) -> float:
        return len(self.events) / self.days if self.days > 0 else 0.0


def replay(windows: list[dict], cfg: MonitorConfig, *, seed: int = 0) -> ReplayResult:
    """`windows` must be time-ordered dicts with 'ts' and the FEATURE_NAMES columns."""
    if not windows:
        return ReplayResult([], 0.0, 0, 0)
    mon = _ReplayMonitor(cfg)
    episodes = AnomalyEpisodes(mon)
    ts = np.array([w["ts"] for w in windows])
    raw = np.array([[w.get(c, math.nan) for c in FEATURE_NAMES] for w in windows], dtype=np.float64)
    X = to_matrix(raw)
    min_windows = int(cfg.min_training_hours * 3600 / cfg.window_sec)
    retrain_every = cfg.retrain_interval_min * 60
    next_train = ts[0]
    grace_until = ts[0] + cfg.grace_period_sec
    rng = np.random.default_rng(seed)
    scored = flagged = 0

    for i, w in enumerate(windows):
        now = w["ts"] + cfg.window_sec
        if i and w["ts"] - windows[i - 1]["ts"] > 5 * cfg.window_sec:     # machine was off/asleep
            grace_until = now + cfg.grace_period_sec
            episodes.reset()
        if now >= next_train:
            lo = np.searchsorted(ts, now - cfg.training_days * 86400)
            hist = np.arange(lo, i)
            if len(hist) >= min_windows:
                if len(hist) > cfg.max_training_windows:
                    hist = rng.choice(hist, cfg.max_training_windows, replace=False)
                mon.model = fit_model(raw[hist], ts[hist], alert_quantile=cfg.alert_quantile,
                                      window_sec=cfg.window_sec)
                next_train = now + retrain_every
            else:
                next_train = now + 300
        w = dict(w)
        x = None
        if mon.model is not None and now >= grace_until:
            x = X[i]
            score, anomalous = mon.model.evaluate(x)
            w["score"], w["anomalous"] = float(score[0]), int(anomalous[0])
            scored += 1
            flagged += w["anomalous"]
        episodes.on_window(w, x, now)

    covered_days = len(windows) * cfg.window_sec / 86400
    return ReplayResult(mon.events.events, covered_days, scored, flagged)


def tune_grid(windows: list[dict], cfg: MonitorConfig,
              quantiles=(0.0005, 0.001, 0.002, 0.005), streaks=(2, 3, 6)) -> list[dict]:
    rows = []
    for q in quantiles:
        for k in streaks:
            r = replay(windows, replace(cfg, alert_quantile=q, min_consecutive_windows=k))
            rows.append({"alert_quantile": q, "min_consecutive_windows": k, "events": len(r.events),
                         "per_day": round(r.per_day, 2), "days": round(r.days, 2)})
    return rows
