"""
Detectors. Each one watches a stream (1 Hz samples, 5 s process sweeps or
10 s scored windows) and opens/updates/closes events through the
EventManager.

  AnomalyEpisodes      ML: sustained statistically-unusual system behaviour
  MemoryPressure       RAM or commit charge near exhaustion
  GpuThermal           GPU thermal throttling / overheating
  GpuAwakeOnBattery    dGPU kept powered while idle on battery (drains it)
  LowDiskSpace         system drive nearly full (also stops the page file growing)
  BackgroundCpu        a non-foreground process burning CPU for minutes
  Leaks                per-process private-memory and handle growth trends

The ML catches the unknown unknowns; the rules catch known-bad conditions
with a precise, actionable explanation and are active from the first second
(no training needed).
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from .events import Event, fmt_duration
from .features import FEATURE_BY_COLUMN, ML_FEATURES, fmt_bytes, fmt_feature, to_matrix
from .sampler import (SYSTEM_DRIVE, THERMAL_THROTTLE_MASK, THROTTLE_HW_SLOWDOWN, THROTTLE_HW_THERMAL,
                      THROTTLE_SW_THERMAL)

if TYPE_CHECKING:
    from .engine import Monitor
    from .processes import ProcStat


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


class Sustained:
    """
    Debounced condition. Becomes active once `cond` has held for `on_sec`;
    gaps shorter than `off_sec` neither reset the timer nor end the episode.
    """

    def __init__(self, on_sec: float, off_sec: float):
        self.on_sec, self.off_sec = on_sec, off_sec
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.since: float | None = None
        self.last_true: float | None = None

    def update(self, t: float, cond: bool) -> str | None:
        if cond:
            self.last_true = t
            if self.since is None:
                self.since = t
            if not self.active and t - self.since >= self.on_sec:
                self.active = True
                return "start"
        elif self.since is not None and t - self.last_true >= self.off_sec:
            was_active = self.active
            self.reset()
            if was_active:
                return "end"
        return None


class Detector:
    def __init__(self, mon: "Monitor"):
        self.mon = mon
        self.cfg = mon.cfg

    def on_sample(self, s: dict, now: float) -> None: ...
    def on_sweep(self, stats: list["ProcStat"], now: float) -> None: ...
    def on_window(self, w: dict, x: np.ndarray | None, now: float) -> None: ...
    def reset(self) -> None: ...

    def ignored(self, name: str) -> bool:
        return name.lower() in self.cfg.ignore_processes


# ════════════════════════════════════════════════════════════════════
#  ML anomaly episodes
# ════════════════════════════════════════════════════════════════════

_SNAPSHOT_FEATURES = [f for f in ML_FEATURES if f.unit != "MHz"]


class AnomalyEpisodes(Detector):
    kind = "anomaly"
    REPEAT_MERGE_SEC = 3600   # same kind of anomaly within an hour → reopen the earlier event

    def __init__(self, mon):
        super().__init__(mon)
        self.reset()

    def reset(self) -> None:
        self.streak: list[tuple[dict, np.ndarray]] = []
        self.event: Event | None = None
        self.last_closed: Event | None = None
        self.normal_run = 0
        self.cooldown_until = 0.0
        self.peak_score = -math.inf
        self.peak_severe = False
        self.user_driven = False

    def on_window(self, w, x, now):
        if not w.get("anomalous"):
            self.streak.clear()
            if self.event is not None:
                self.normal_run += 1
                if self.normal_run >= self.cfg.close_after_normal_windows:
                    self._close(now, "returned to normal")
            return

        w_end = w["ts"] + self.cfg.window_sec
        if self.event is not None:
            self.normal_run = 0
            ev = self.event
            ev.details["anomalous_windows"] += 1
            title, details = self._describe(w, x) if w["score"] > self.peak_score else (None, None)
            self.mon.events.update(ev, w_end, title=title, details=details, severity=self._severity())
            if w_end - ev.ts_start >= self.cfg.max_event_minutes * 60:
                self._close(now, f"still unusual after {self.cfg.max_event_minutes:g} min; "
                                 "treating it as a new normal until the model retrains")
                self.cooldown_until = now + max(self.cfg.ml_cooldown_min, self.cfg.retrain_interval_min) * 60
            return

        self.streak.append((w, x))
        if len(self.streak) < self.cfg.min_consecutive_windows or now < self.cooldown_until:
            return
        start = self.streak[0][0]["ts"]
        peak_w, peak_x = max(self.streak, key=lambda wx: wx[0]["score"])
        n = len(self.streak)
        self.streak.clear()

        prev = self.last_closed
        prev_peak = prev.details.get("peak_score", -math.inf) if prev is not None else -math.inf
        title, details = self._describe(peak_w, peak_x)
        if prev is not None and start - prev.ts_end <= self.REPEAT_MERGE_SEC \
                and self._signature(prev.details) & self._signature(details):
            # Same story as an event that just ended: continue it rather than alert again.
            self.event = prev
            if peak_w["score"] <= prev_peak:            # earlier peak stays the headline
                self.peak_score = prev_peak
                title, details = None, None
            prev.details["anomalous_windows"] += n
            self.mon.events.reopen(prev, w_end)
            self.mon.events.update(prev, w_end, title=title, details=details, severity=self._severity())
            return
        details["anomalous_windows"] = n
        self.event = self.mon.events.open(self.kind, self._severity(), title, start, details)
        self.mon.events.update(self.event, w_end)

    @staticmethod
    def _signature(details: dict) -> set[str]:
        return {d["feature"] for d in details.get("why", [])[:2]}

    def _describe(self, w: dict, x: np.ndarray | None) -> tuple[str, dict]:
        """Title + details for the most anomalous window seen so far in the episode."""
        model = self.mon.model
        self.peak_score = w["score"]
        self.peak_severe = model is not None and w["score"] > model.severe_threshold
        why = model.explain(x if x is not None else to_matrix([w])[0], w) if model else []
        act = self.mon.activity(since=w["ts"] - 60)
        rarity = model.rarity(w["score"]) if model else 1.0
        one_in = int(1 / max(rarity, 1 / max(model.n_windows, 1))) if model else 0
        parts = [f"{d['label']} {fmt_feature(FEATURE_BY_COLUMN[d['feature']], w.get(d['feature'], math.nan))}"
                 for d in why[:2]]
        title = "Unusual system behaviour"
        if parts:
            title += ": " + ", ".join(parts)

        # Who is behind it? If it's the app you're actively using, it's probably intentional.
        culprit = blurb = None
        if act["top_cpu"] and act["top_cpu"][0]["avg_cores"] >= 0.25:
            top = act["top_cpu"][0]
            culprit, blurb = top["name"], f"{top['avg_cores']:.1f} cores"
        elif act["top_io"] and act["top_io"][0]["avg_mb_s"] >= 1:
            top = act["top_io"][0]
            culprit, blurb = top["name"], f"{top['avg_mb_s']:.0f} MB/s I/O"
        elif act["gpu_clients"] and act["gpu_clients"][0]["sm_util_pct"] >= 10:
            top = act["gpu_clients"][0]
            culprit, blurb = top["name"], f"{top['sm_util_pct']}% GPU"
        fg = act["foreground"]
        self.user_driven = bool(fg and culprit == fg and not act["user_idle"])
        if culprit and self.user_driven:
            title += f" — from the app in use, {culprit} ({blurb})"
        elif culprit:
            title += f" — top: {culprit} ({blurb})"

        return title, {
            "why": why,
            "peak_score": round(w["score"], 2),
            "threshold": round(model.threshold, 2) if model else None,
            "rarity": f"about 1 in {one_in:,} windows of your history" if model else None,
            "at_peak": {f.label: fmt_feature(f, w.get(f.column, math.nan)) for f in _SNAPSHOT_FEATURES},
            "busiest_core": f"{w.get('core_max_max_pct', math.nan):.0f}%",
            "user_driven": self.user_driven,
            **act,
            "model": {"hours_of_data": round(model.hours_of_data, 1), "windows": model.n_windows,
                      "learning": model.hours_of_data < self.cfg.ml_mature_hours} if model else None,
        }

    def _severity(self) -> str:
        # Rare-but-explained-by-what-you're-doing stays informational, and so does
        # everything while the model is still young (its first day is the noisiest).
        model = self.mon.model
        mature = model is not None and model.hours_of_data >= self.cfg.ml_mature_hours
        return "warning" if self.peak_severe and not self.user_driven and mature else "info"

    def _close(self, now: float, why: str) -> None:
        ev = self.event
        ev.details["ended"] = why
        self.mon.events.close(ev, ev.ts_end)
        self.last_closed = ev
        self.event = None
        self.normal_run = 0
        self.cooldown_until = now + self.cfg.ml_cooldown_min * 60

    def shutdown(self, now: float) -> None:
        if self.event is not None:
            self._close(now, "monitor stopped")


# ════════════════════════════════════════════════════════════════════
#  System-level rules (1 Hz samples)
# ════════════════════════════════════════════════════════════════════

class MemoryPressure(Detector):
    kind = "memory_pressure"

    def __init__(self, mon):
        super().__init__(mon)
        self.state = Sustained(self.cfg.memory_pressure_sec, 30)
        self.event: Event | None = None
        self._peak = (0.0, 0.0)
        self._last_update = 0.0

    def reset(self):
        self.state.reset()
        self.event, self._peak = None, (0.0, 0.0)

    def on_sample(self, s, now):
        ram, commit = s["ram_pct"], s["commit_pct"]
        commit_hit = not _isnan(commit) and commit >= self.cfg.commit_pressure_percent
        tr = self.state.update(now, ram >= self.cfg.ram_pressure_percent or commit_hit)
        if self.state.active:
            self._peak = (max(self._peak[0], ram), max(self._peak[1], 0 if _isnan(commit) else commit))

        if tr == "start":
            top = self.mon.procs.top_memory(5)
            sev = "critical" if ram >= 97 or (commit_hit and commit >= 97) else "warning"
            title = f"Memory pressure: RAM {ram:.0f}%" + ("" if _isnan(commit) else f", commit {commit:.0f}%")
            if top:
                title += f"; largest {top[0]['name']} ({top[0]['private_mb']:,} MB)"
            self.event = self.mon.events.open(self.kind, sev, title, self.state.since,
                                              {"top_memory": top, "foreground": self.mon.procs.foreground_name})
            self._last_update = now
        elif tr == "end" and self.event is not None:
            self.mon.events.close(self.event, now, details=self._peak_details())
            self.event, self._peak = None, (0.0, 0.0)
        elif self.event is not None and now - self._last_update >= 60:
            self._last_update = now
            self.mon.events.update(self.event, now, details=self._peak_details())

    def _peak_details(self) -> dict:
        return {"peak_ram_pct": round(self._peak[0], 1), "peak_commit_pct": round(self._peak[1], 1),
                "top_memory": self.mon.procs.top_memory(5)}


_THROTTLE_NAMES = {
    THROTTLE_HW_SLOWDOWN: "hardware slowdown",
    THROTTLE_SW_THERMAL: "driver thermal limit",
    THROTTLE_HW_THERMAL: "hardware thermal limit",
}


class GpuThermal(Detector):
    kind = "gpu_thermal"

    def __init__(self, mon):
        super().__init__(mon)
        self.state = Sustained(self.cfg.gpu_thermal_sec, 30)
        self.event: Event | None = None
        self._reasons: set[str] = set()
        self._peak_temp = 0.0
        self._peak_power = 0.0
        self._last_update = 0.0

    def reset(self):
        self.state.reset()
        self.event = None
        self._reasons, self._peak_temp, self._peak_power = set(), 0.0, 0.0

    def on_sample(self, s, now):
        if s["gpu_awake"] != 1.0:
            cond, temp, power = False, math.nan, math.nan
        else:
            temp, power = s["gpu_temp_c"], s["gpu_power_w"]
            throttle = int(s["gpu_throttle"] or 0) & THERMAL_THROTTLE_MASK
            # The laptop driver raises its "thermal" flag at 32–52 °C while idling (seen here);
            # only believe it when the GPU really is warm.
            if _isnan(temp) or temp < self.cfg.gpu_throttle_min_temp_c:
                throttle = 0
            hot = not _isnan(temp) and temp >= self.cfg.gpu_temp_warn_c
            cond = bool(throttle) or hot
            if cond:
                self._reasons.update(n for bit, n in _THROTTLE_NAMES.items() if throttle & bit)
                if hot:
                    self._reasons.add(f"≥{self.cfg.gpu_temp_warn_c:g}°C")
                if not _isnan(temp):
                    self._peak_temp = max(self._peak_temp, temp)
                if not _isnan(power):
                    self._peak_power = max(self._peak_power, power)

        tr = self.state.update(now, cond)
        if tr == "start":
            limit = self.mon.gpu.slowdown_temp_c if self.mon.gpu else None
            critical = limit is not None and self._peak_temp >= limit - 2
            what = "thermal throttling" if self._reasons - {f"≥{self.cfg.gpu_temp_warn_c:g}°C"} else "running hot"
            title = f"GPU {what}: {self._peak_temp:.0f}°C at {self._peak_power:.0f} W ({', '.join(sorted(self._reasons))})"
            self.event = self.mon.events.open(self.kind, "critical" if critical else "warning", title,
                                              self.state.since, self._details())
            self._last_update = now
        elif tr == "end" and self.event is not None:
            self.mon.events.close(self.event, now, details=self._details())
            self.event = None
            self._reasons, self._peak_temp, self._peak_power = set(), 0.0, 0.0
        elif tr is None and not self.state.active and self.state.since is None:
            self._reasons, self._peak_temp, self._peak_power = set(), 0.0, 0.0
        elif self.event is not None and now - self._last_update >= 60:
            self._last_update = now
            self.mon.events.update(self.event, now, details=self._details())

    def _details(self) -> dict:
        return {"peak_temp_c": self._peak_temp, "peak_power_w": round(self._peak_power, 1),
                "reasons": sorted(self._reasons), "gpu_clients": self.mon.gpu_clients(),
                "foreground": self.mon.procs.foreground_name}


class GpuAwakeOnBattery(Detector):
    kind = "gpu_awake_on_battery"

    def __init__(self, mon):
        super().__init__(mon)
        self.state = Sustained(self.cfg.gpu_awake_idle_minutes * 60, 60)
        self.event: Event | None = None
        self._power: list[float] = []

    def reset(self):
        self.state.reset()
        self.event, self._power = None, []

    def on_sample(self, s, now):
        cond = s["on_ac"] == 0.0 and s["gpu_awake"] == 1.0 and s["gpu_util_pct"] < 5
        if cond and not _isnan(s["gpu_power_w"]):
            self._power = (self._power + [s["gpu_power_w"]])[-600:]
        tr = self.state.update(now, cond)
        if tr == "start":
            clients = self.mon.gpu_clients()
            watts = sum(self._power) / len(self._power) if self._power else math.nan
            title = (f"dGPU awake but idle on battery for {self.cfg.gpu_awake_idle_minutes:g} min "
                     f"(~{watts:.0f} W wasted)")
            if clients:
                title += f"; GPU clients: {', '.join(c['name'] for c in clients[:3])}"
            self.event = self.mon.events.open(self.kind, "info", title, self.state.since,
                                              {"gpu_clients": clients, "avg_power_w": round(watts, 1)})
        elif tr == "end" and self.event is not None:
            self.mon.events.close(self.event, now)
            self.event, self._power = None, []


class LowDiskSpace(Detector):
    """
    The system drive nearly full. An auto-managed page file can only grow into free
    space on its drive, so this also keeps the commit limit near RAM size (commit hit
    99 % here during a game, with a 2.3 GB page file and 15 GB free on C:).
    """
    kind = "low_disk_space"
    REMIND_SEC = 7 * 86400    # a drive that stays full is mentioned again after a week…
    REMIND_FACTOR = 2.0       # …or as soon as free space has halved since the last warning

    def __init__(self, mon):
        super().__init__(mon)
        # (free GB when last reported, when). Restored from recent events, so a restarted
        # monitor doesn't re-announce a drive it has already warned about.
        self.reported: tuple[float, float] | None = None
        for ev in mon.events.recent([self.kind], days=self.REMIND_SEC / 86400):
            free = ev["details"].get("free_gb")
            if free is not None and (self.reported is None or ev["ts"] > self.reported[1]):
                self.reported = (free, ev["ts"])

    def on_sample(self, s, now):
        free, total = s.get("sysdrive_free_gb"), s.get("sysdrive_total_gb")
        if _isnan(free) or _isnan(total) or total <= 0:
            return
        pct = 100.0 * free / total
        last = self.reported
        if free >= self.cfg.low_disk_free_gb and pct >= self.cfg.low_disk_free_percent:
            if last is not None and free >= self.REMIND_FACTOR * last[0]:
                self.reported = None          # clearly recovered: running low again is news
            return
        if last is None or now - last[1] >= self.REMIND_SEC or free < last[0] / self.REMIND_FACTOR:
            self._report(s, free, total, pct, now)

    def _report(self, s, free, total, pct, now):
        self.reported = (free, now)
        drive = SYSTEM_DRIVE.rstrip("\\")
        pagefile, limit, commit = s.get("pagefile_gb"), s.get("commit_limit_gb"), s.get("commit_pct")
        title = f"System drive {drive} nearly full: {free:.1f} GB free of {total:,.0f} GB ({pct:.1f}%)"
        if not _isnan(limit) and not _isnan(pagefile) and free < limit - pagefile:
            title += f" — page file can't grow much, commit limit stays ~{limit:.0f} GB"   # free < RAM
        self.mon.events.point(self.kind, "warning", title, now, {
            "drive": drive, "free_gb": round(free, 1), "total_gb": round(total, 1), "free_pct": round(pct, 1),
            "pagefile_gb": None if _isnan(pagefile) else round(pagefile, 1),
            "commit_limit_gb": None if _isnan(limit) else round(limit, 1),
            "commit_pct": None if _isnan(commit) else round(commit, 1),
            "note": "An auto-managed page file can only grow into free space on its drive. When that is "
                    "less than RAM, the commit limit (RAM + page file) stays near RAM size and a heavy "
                    "load can run out of commit (apps crash, Windows warns of low memory) before RAM "
                    "is full. Free up space, or set a fixed page-file size."})


# ════════════════════════════════════════════════════════════════════
#  Per-process rules (process sweeps)
# ════════════════════════════════════════════════════════════════════

class BackgroundCpu(Detector):
    """A process that is *not* the app you're using, burning ≥ N cores for minutes."""
    kind = "background_cpu"
    COOLDOWN_SEC = 3600

    def __init__(self, mon):
        super().__init__(mon)
        self.window = self.cfg.background_cpu_minutes * 60
        self.need = max(2, int(0.9 * self.window / self.cfg.process_interval_sec))
        self.hist: dict[tuple, deque] = {}
        self.open: dict[tuple, tuple[Event, list[float]]] = {}
        self.cooldown: dict[tuple, float] = {}

    def reset(self):
        self.hist.clear()
        self.open.clear()

    def on_sweep(self, stats, now):
        thr = self.cfg.background_cpu_cores
        seen = set()
        for s in stats:
            seen.add(s.key)
            h = self.hist.get(s.key)
            if h is None:
                if s.cores < 0.05:
                    continue
                h = self.hist[s.key] = deque()
            h.append((now, s.cores, s.foreground))
            while h and h[0][0] < now - self.window:
                h.popleft()
            if s.key not in self.open and max(c for _, c, _ in h) < 0.05:
                del self.hist[s.key]
                continue

            if s.key in self.open:
                self._track(s, h, now, thr)
            elif (len(h) >= self.need and now >= self.cooldown.get(s.key, 0) and not self.ignored(s.name)):
                mean = sum(c for _, c, _ in h) / len(h)
                fg_share = sum(fg for _, _, fg in h) / len(h)
                if mean >= thr and fg_share <= 0.2:
                    self._open(s, mean, h[0][0], now)

        for key in [k for k in self.hist if k not in seen]:   # process exited
            if key in self.open:
                self._close(key, now, "process exited")
            del self.hist[key]
        self.cooldown = {k: t for k, t in self.cooldown.items() if t > now}

    def _open(self, s, mean, start, now):
        minutes = (now - start) / 60
        sev = "warning" if mean >= 3 else "info"
        title = f"{s.name} used {mean:.1f} CPU cores in the background for {minutes:.0f}+ min"
        fg = self.mon.procs.foreground_name
        details = {"process": s.name, "pid": s.pid, "avg_cores": round(mean, 2),
                   "foreground_app": fg, "user_idle": self.mon.user_idle}
        ev = self.mon.events.open(self.kind, sev, title, start, details)
        self.open[s.key] = (ev, [mean])

    def _track(self, s, h, now, thr):
        ev, cores = self.open[s.key]
        cores.append(s.cores)
        recent = [c for t, c, _ in h if t >= now - 60]
        recent_fg = [fg for t, _, fg in h if t >= now - 60]
        if sum(recent) / len(recent) < thr / 2:
            self._close(s.key, now, "CPU use dropped")
        elif sum(recent_fg) / len(recent_fg) > 0.5:
            self._close(s.key, now, "brought to the foreground")
        elif len(cores) % 12 == 0:   # ~once a minute
            avg = sum(cores) / len(cores)
            self.mon.events.update(ev, now, details={"avg_cores": round(avg, 2)},
                                   title=f"{s.name} used {avg:.1f} CPU cores in the background "
                                         f"for {fmt_duration(now - ev.ts_start)}")

    def _close(self, key, now, why):
        ev, cores = self.open.pop(key)
        avg = sum(cores) / len(cores)
        self.mon.events.close(ev, now, details={"avg_cores": round(avg, 2), "ended": why},
                              title=f"{ev.details['process']} used {avg:.1f} CPU cores in the background "
                                    f"for {fmt_duration(now - ev.ts_start)}")
        self.cooldown[key] = now + self.COOLDOWN_SEC


# Kernel/pseudo processes whose memory isn't "owned" by an app.
_NOT_MEMORY_LEAKS = {"memory compression", "registry", "system", "secure system", "vmmem", "vmmemwsl"}


def trend(t: np.ndarray, y: np.ndarray) -> tuple[float, float, bool]:
    """(slope per hour, R², monotone-by-thirds) of a series."""
    th = (t - t[0]) / 3600.0
    slope, intercept = np.polyfit(th, y, 1)
    resid = y - (slope * th + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0
    n = len(y)
    a, b, c = np.median(y[: n // 3]), np.median(y[n // 3: 2 * n // 3]), np.median(y[2 * n // 3:])
    return float(slope), r2, bool(a < b < c)


class Leaks(Detector):
    """Steady growth of a process's private memory or handle count (classic leaks)."""
    SAMPLE_SEC = 120
    CHECK_SEC = 300
    MIN_R2 = 0.8
    REMIND_SEC = 30 * 86400   # a known leaker is mentioned again after a month…
    REMIND_FACTOR = 2.0       # …or as soon as it is twice as bad as when last reported

    def __init__(self, mon):
        super().__init__(mon)
        self.hist: dict[tuple, deque] = {}
        self.names: dict[tuple, str] = {}
        # (kind, executable name) → (value when last reported, when). Keyed by name, so a
        # restarted service — or a restarted monitor — doesn't re-announce a known leak.
        self.reported: dict[tuple[str, str], tuple[float, float]] = {}
        self._next_sample = 0.0
        self._next_check = 0.0
        for ev in mon.events.recent(["memory_leak", "handle_leak"], days=self.REMIND_SEC / 86400):
            d = ev["details"]
            if "process" not in d:
                continue
            value = d.get("handles", 0) if ev["kind"] == "handle_leak" else d.get("private_mb", 0) * 2**20
            k = (ev["kind"], d["process"].lower())
            if k not in self.reported or ev["ts"] > self.reported[k][1]:
                self.reported[k] = (value, ev["ts"])

    def on_sweep(self, stats, now):
        if now < self._next_sample:
            return
        self._next_sample = now + self.SAMPLE_SEC
        horizon = now - self.cfg.leak_history_hours * 3600
        seen = set()
        for s in stats:
            seen.add(s.key)
            h = self.hist.setdefault(s.key, deque())
            h.append((now, s.private_bytes, s.handles))
            self.names[s.key] = s.name
            while h[0][0] < horizon:
                h.popleft()
        for key in [k for k in self.hist if k not in seen]:
            self.hist.pop(key, None)
            self.names.pop(key, None)
        if now >= self._next_check:
            self._next_check = now + self.CHECK_SEC
            for key, h in self.hist.items():
                name = self.names[key]
                if self.ignored(name):
                    continue
                self._check_handles(key, name, h, now)
                if name.lower() not in _NOT_MEMORY_LEAKS:
                    self._check_memory(key, name, h, now)

    def _due(self, kind: str, name: str, value: float, now: float) -> bool:
        last = self.reported.get((kind, name.lower()))
        return last is None or now - last[1] >= self.REMIND_SEC or value >= self.REMIND_FACTOR * last[0]

    def _report(self, kind: str, name: str, value: float, now: float, title: str, details: dict) -> None:
        self.reported[(kind, name.lower())] = (value, now)
        self.mon.events.point(kind, "warning", title, now, details)

    def _series(self, h: deque, idx: int) -> tuple[np.ndarray, np.ndarray, float]:
        t = np.fromiter((x[0] for x in h), float, len(h))
        y = np.fromiter((x[idx] for x in h), float, len(h))
        return t, y, (t[-1] - t[0]) / 3600 if len(t) > 1 else 0.0

    def _check_memory(self, key, name, h, now):
        if len(h) < 6:
            return
        min_growth = self.cfg.leak_min_growth_mb * 2**20
        t, y, span_h = self._series(h, 1)
        if span_h < self.cfg.leak_min_history_hours or y[-1] - y[0] < 0.75 * min_growth:
            return
        if not self._due("memory_leak", name, y[-1], now):
            return
        slope, r2, monotone = trend(t, y)
        growth = slope * span_h
        if growth >= min_growth and r2 >= self.MIN_R2 and monotone:
            self._report(
                "memory_leak", name, y[-1], now,
                f"{name} memory growing steadily: +{growth / 2**20:,.0f} MB in {span_h:.1f} h "
                f"({slope / 2**20:,.0f} MB/h), now {y[-1] / 2**20:,.0f} MB",
                {"process": name, "pid": key[0], "create_time": key[1], "private_mb": round(y[-1] / 2**20),
                 "growth_mb": round(growth / 2**20), "mb_per_hour": round(slope / 2**20, 1),
                 "r2": round(r2, 3), "hours_observed": round(span_h, 2)})

    def _check_handles(self, key, name, h, now):
        current = h[-1][2]
        if current < self.cfg.handle_growth_min_count:
            return                     # small handle tables (Discord at 4,600) aren't worth a warning
        t, y, span_h = self._series(h, 2)
        rate = None
        if len(h) >= 6 and span_h >= self.cfg.leak_min_history_hours:
            slope, r2, monotone = trend(t, y)
            if r2 >= self.MIN_R2 and monotone and slope * span_h >= self.cfg.handle_leak_min_growth:
                rate = slope
        if current >= self.cfg.handle_absolute_limit:
            why = f"holds {current:,.0f} handles (typical apps use < 10,000)"
        elif rate is not None:
            why = f"handle count growing steadily, now {current:,.0f}"
        else:
            return
        if not self._due("handle_leak", name, current, now):
            return
        if rate is not None:
            why += f" (+{rate:,.0f}/h)"
        self._report(
            "handle_leak", name, current, now, f"{name} {why}",
            {"process": name, "pid": key[0], "create_time": key[1], "handles": int(current),
             "handles_per_hour": None if rate is None else round(rate),
             "hours_observed": round(span_h, 2),
             "note": "Handle leaks slowly exhaust kernel memory; restarting the process frees them."})


def summarize_rates(s: dict) -> str:
    """One-line human summary of a sample (used by `status`)."""
    gpu = "absent" if _isnan(s.get("gpu_awake")) else (
        "asleep" if not s.get("gpu_awake") else
        f"{s['gpu_util_pct']:.0f}% @ {s.get('gpu_temp_c') or 0:.0f}°C, {s.get('gpu_power_w') or 0:.0f} W")
    return (f"CPU {s['cpu_pct']:.0f}% (busiest core {s['core_max_pct']:.0f}%), RAM {s['ram_pct']:.0f}%, "
            f"commit {s['commit_pct'] or 0:.0f}%, disk {fmt_bytes(s['disk_read_bps'], True)} read / "
            f"{fmt_bytes(s['disk_write_bps'], True)} write, net {fmt_bytes(s['net_recv_bps'], True)} down / "
            f"{fmt_bytes(s['net_sent_bps'], True)} up, GPU {gpu}, {'AC' if s['on_ac'] else 'battery'}")
