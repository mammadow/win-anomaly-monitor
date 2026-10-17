"""
The orchestrator: one `tick()` per second.

  sample ─┬─ store (1 Hz)
          ├─ rules (memory pressure, GPU thermal, dGPU-on-battery, low disk space)
          ├─ UI freezes timed by the responsiveness probe → explained from the samples
          ├─ every 5 s: process sweep → rules (background CPU, leaks)
          └─ every 10 s: window → ML score → anomaly episodes → store

  The responsiveness probe runs in a helper process (not throttled by the
  Efficiency Mode the main loop runs in) and hands finished freezes over.

Lifecycle
  * Starts collecting immediately. Rules work from the first second.
  * ML activates once `min_training_hours` of windows exist (immediately on
    restarts — history is in SQLite) and retrains hourly on up to a week.
  * Sleep/hibernate is detected as a gap between ticks: rate baselines are
    reset and the ML stays quiet for a grace period (resume = burst of
    updates/sync that isn't worth an alert).
  * One failing tick is logged and skipped; it never kills the monitor
    (v2 died on a negative time.sleep() after 4 days).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

from . import __version__, winapi
from .config import MonitorConfig
from .detector import AnomalyModel, Trainer
from .events import EventManager, Notifier, fmt_duration
from .features import WindowAggregator, to_matrix
from .processes import ProcessTracker
from .responsiveness import ProbeProcess
from .rules import (AnomalyEpisodes, BackgroundCpu, GpuAwakeOnBattery, GpuThermal, Leaks,
                    LowDiskSpace, MemoryPressure, UiFreeze)
from .sampler import GpuSampler, SystemSampler
from .storage import Storage, load_training_windows

DB_NAME = "metrics.db"
EVENTS_NAME = "events.jsonl"
STATE_NAME = "state.json"
LOCK_NAME = "monitor.lock"
LOG_NAME = "monitor.log"

GPU_CLIENT_TTL = 300  # seconds a GPU-using process stays in the "clients" list


class InstanceLock:
    """OS-level lock so only one monitor writes to the data directory."""

    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def acquire(self) -> bool:
        fh = open(self.path, "a+")
        try:
            fh.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover
                import fcntl
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self._fh.close()
            self._fh = None

    @classmethod
    def is_held(cls, path: Path) -> bool:
        if not path.exists():
            return False
        probe = cls(path)
        if probe.acquire():
            probe.release()
            return False
        return True


class Monitor:
    def __init__(self, cfg: MonitorConfig, *, sampler=None, procs=None, gpu: GpuSampler | None | bool = True):
        self.cfg = cfg
        self.data = cfg.data_path
        self.data.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data / DB_NAME

        self.gpu = GpuSampler() if gpu is True else (gpu or None)
        self.sampler = sampler or SystemSampler(self.gpu)
        self.procs = procs or ProcessTracker()
        self.storage = Storage(self.db_path)
        self.events = EventManager(self.storage, self.data / EVENTS_NAME,
                                   Notifier() if cfg.notify else None, cfg.notify_min_severity)
        self.aggregator = WindowAggregator(cfg.window_sec, cfg.sample_interval_sec)
        self.trainer = Trainer(self.data)
        self.model: AnomalyModel | None = None
        self.ui_probe: ProbeProcess | None = None   # helper process, started by run()

        self.episodes = AnomalyEpisodes(self)
        self.detectors = [
            self.episodes,
            MemoryPressure(self),
            GpuThermal(self),
            GpuAwakeOnBattery(self),
            LowDiskSpace(self),
            UiFreeze(self),
            BackgroundCpu(self),
            Leaks(self),
        ]

        self.started_at = time.time()
        self.grace_until = self.started_at + cfg.grace_period_sec
        self.user_idle = False
        self.last_sample: dict | None = None
        self.n_samples = self.n_windows = 0
        self._gpu_clients: dict[int, tuple[int, float]] = {}
        self._last_tick: float | None = None
        self._next_sweep = 0.0
        self._next_flush = 0.0
        self._next_train = 0.0
        self._next_prune = 0.0
        self._next_state = 0.0
        self._running = False
        self._stop = threading.Event()

    # ════════════════════════════════════════════════════════════════
    #  One iteration
    # ════════════════════════════════════════════════════════════════

    def tick(self, now: float | None = None, mono: float | None = None) -> None:
        now = time.time() if now is None else now
        mono = time.monotonic() if mono is None else mono

        if self._last_tick is not None:
            gap = now - self._last_tick
            if gap > max(5.0, 5 * self.cfg.sample_interval_sec) or gap < -1.0:
                self._last_tick = now
                self._on_gap(now, gap)
                return
        self._last_tick = now

        s = self.sampler.sample(now)
        self.storage.add_sample(s)
        self.last_sample = s
        self.n_samples += 1
        idle = s.get("user_idle_sec")
        self.user_idle = idle is not None and idle == idle and idle >= 300

        if mono >= self._next_sweep:
            self._next_sweep = mono + self.cfg.process_interval_sec
            stats = self.procs.sweep(now, mono)
            self._update_gpu_clients(now)
            for d in self.detectors:
                d.on_sweep(stats, now)

        for d in self.detectors:
            d.on_sample(s, now)

        w = self.aggregator.add(s)
        if w is not None:
            self._on_window(w, now)

        self._maintenance(now, mono)

    def _on_window(self, w: dict, now: float) -> None:
        x = None
        if self.model is not None and now >= self.grace_until:
            x = to_matrix([w])[0]
            score, anomalous = self.model.evaluate(x)
            w["score"] = float(score[0])
            w["anomalous"] = int(anomalous[0])
        top = self.procs.top_activity(since=w["ts"], n=3)["cpu"]
        w["top_procs"] = [[p["name"], p["avg_cores"]] for p in top]
        w["fg_name"] = self.procs.foreground_name
        self.storage.add_window(w)
        self.n_windows += 1
        for d in self.detectors:
            d.on_window(w, x, now)

    def _on_gap(self, now: float, gap: float) -> None:
        logging.info("Clock gap of %s (sleep/hibernate?) — re-baselining; ML quiet for %ds",
                     fmt_duration(abs(gap)), self.cfg.grace_period_sec)
        self.events.close_all(now, "machine slept or clock jumped")
        for d in self.detectors:
            d.reset()
        self.aggregator.reset()
        self.sampler.reset_rates()
        self.procs.reset()
        self.grace_until = now + self.cfg.grace_period_sec

    # ════════════════════════════════════════════════════════════════
    #  Periodic work
    # ════════════════════════════════════════════════════════════════

    def _maintenance(self, now: float, mono: float) -> None:
        if mono >= self._next_flush:
            self._next_flush = mono + self.cfg.window_sec
            self.storage.flush()

        try:
            model = self.trainer.poll()
        except Exception:
            logging.exception("Model training failed; will retry")
            self._next_train = now + 300
            model = None
        if model is not None:
            first = self.model is None
            self.model = model
            logging.info("%s anomaly model on %s windows (%.1f h of data); alert threshold %.2f",
                         "Trained" if first else "Retrained", f"{model.n_windows:,}",
                         model.hours_of_data, model.threshold)

        if now >= self._next_train and not self.trainer.busy:
            self._start_training(now)

        if now >= self._next_prune:
            self._next_prune = now + 3600
            try:
                n = self.storage.prune(now, self.cfg.raw_retention_hours,
                                       self.cfg.window_retention_days, self.cfg.event_retention_days)
                if n:
                    logging.info("Pruned %s old rows", f"{n:,}")
            except Exception:
                logging.exception("Prune failed")

        if now >= self._next_state:
            self._next_state = now + 30
            self.write_state(now)

    def _start_training(self, now: float) -> None:
        cfg = self.cfg
        self.storage.flush()   # the trainer reads the database through its own connection
        min_windows = int(cfg.min_training_hours * 3600 / cfg.window_sec)
        since = now - cfg.training_days * 86400

        def load():
            ts, raw = load_training_windows(self.db_path, since, cfg.max_training_windows)
            if len(raw) < min_windows:
                logging.info("Learning your baseline: %d/%d windows (~%d min until ML detection starts)",
                             len(raw), min_windows, (min_windows - len(raw)) * cfg.window_sec // 60 + 1)
                return None
            return ts, raw

        self.trainer.start(load, alert_quantile=cfg.alert_quantile, window_sec=cfg.window_sec)
        # Until a model exists, check back every minute (a cheap query) rather than once an hour.
        self._next_train = now + (cfg.retrain_interval_min * 60 if self.model else 60)

    # ════════════════════════════════════════════════════════════════
    #  Context for explanations
    # ════════════════════════════════════════════════════════════════

    def _update_gpu_clients(self, now: float) -> None:
        if self.gpu is None:
            return
        for pid, util in self.gpu.process_utilization().items():
            self._gpu_clients[pid] = (util, now)
        self._gpu_clients = {p: v for p, v in self._gpu_clients.items() if now - v[1] <= GPU_CLIENT_TTL}

    def gpu_clients(self) -> list[dict]:
        rows = [{"name": self.procs.name_of(pid), "pid": pid, "sm_util_pct": util}
                for pid, (util, _) in self._gpu_clients.items()]
        return sorted(rows, key=lambda r: r["sm_util_pct"], reverse=True)[:5]

    def activity(self, since: float) -> dict:
        top = self.procs.top_activity(since)
        return {
            "top_cpu": top["cpu"],
            "top_io": top["io"],
            "top_memory": self.procs.top_memory(5),
            "gpu_clients": self.gpu_clients(),
            "foreground": self.procs.foreground_name,
            "user_idle": self.user_idle,
        }

    def write_state(self, now: float, stopped: bool = False) -> None:
        m = self.model
        state = {
            "version": __version__,
            "pid": os.getpid(),
            "started": datetime.fromtimestamp(self.started_at).isoformat(timespec="seconds"),
            "updated": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "stopped": stopped,
            "samples_this_run": self.n_samples,
            "windows_this_run": self.n_windows,
            "gpu": self.gpu.name if self.gpu and self.gpu.available else None,
            "model": None if m is None else {
                "trained": datetime.fromtimestamp(m.trained_at).isoformat(timespec="seconds"),
                "windows": m.n_windows,
                "hours_of_data": round(m.hours_of_data, 1),
                "threshold": round(m.threshold, 3),
            },
            "open_events": [{"id": e.id, "kind": e.kind, "title": e.title}
                            for e in self.events.open_events.values()],
            "ml_quiet_until": datetime.fromtimestamp(self.grace_until).isoformat(timespec="seconds")
            if now < self.grace_until else None,
            "ui_probe": self.ui_probe.snapshot() if self.ui_probe else None,
            "self_cpu_seconds": round(sum(psutil.Process().cpu_times()[:2]), 1),
            "self_memory_mb": round(psutil.Process().memory_info().rss / 2**20),
        }
        path = self.data / STATE_NAME
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            logging.debug("Could not write state file: %s", e)

    # ════════════════════════════════════════════════════════════════
    #  Real-time loop
    # ════════════════════════════════════════════════════════════════

    def stop(self, *_args) -> None:
        self._running = False
        self._stop.set()

    def run(self) -> None:
        cfg = self.cfg
        previous = {}
        for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGBREAK", None)):
            if sig is not None:
                try:
                    previous[sig] = signal.signal(sig, self.stop)
                except (ValueError, OSError):   # not the main thread
                    pass

        if cfg.efficiency_mode:
            try:
                psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
            except (psutil.Error, AttributeError):
                pass
            eco = winapi.enable_efficiency_mode()
        else:
            eco = False

        if cfg.ui_freeze_probe and winapi.IS_WINDOWS:
            self.ui_probe = ProbeProcess(cfg.ui_freeze_sec, cfg.ui_probe_interval_sec)
            self.ui_probe.start()

        orphans = self.storage.close_orphaned_events()
        logging.info("═" * 64)
        logging.info("  Resource Monitor v%s", __version__)
        logging.info("  sampling %gs · windows %ds · process sweep %gs · ML alert quantile %g",
                     cfg.sample_interval_sec, cfg.window_sec, cfg.process_interval_sec, cfg.alert_quantile)
        logging.info("  data: %s%s%s%s", self.data, " · efficiency mode" if eco else "",
                     f" · UI freeze probe every {cfg.ui_probe_interval_sec:g}s" if self.ui_probe else "",
                     f" · closed {orphans} event(s) left open by last run" if orphans else "")
        logging.info("═" * 64)

        self._running = True
        self.model = self.trainer.load_saved(max_age_sec=cfg.training_days * 86400)
        if self.model is not None:
            logging.info("Loaded saved model (%s windows, %.1f h of data); refreshing it in 5 min",
                         f"{self.model.n_windows:,}", self.model.hours_of_data)
            self._next_train = time.time() + 300
        interval = cfg.sample_interval_sec
        next_t = time.monotonic()
        failures = 0
        try:
            while self._running:
                try:
                    self.tick()
                    failures = 0
                except Exception:
                    failures += 1
                    logging.exception("Tick failed (%d in a row)", failures)
                    if failures >= 10:
                        self._stop.wait(30)   # something is badly wrong; don't spin
                next_t += interval
                delay = next_t - time.monotonic()
                if delay < -interval:          # fell behind (sleep, debugger, stall): resync
                    next_t = time.monotonic()
                elif delay > 0:
                    self._stop.wait(delay)
        finally:
            self.shutdown()
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def shutdown(self) -> None:
        now = time.time()
        logging.info("Shutting down…")
        if self.ui_probe is not None:
            self.ui_probe.stop()
        try:
            self.episodes.shutdown(now)
            self.events.close_all(now, "monitor stopped")
        except Exception:
            logging.exception("Error closing events")
        self.trainer.join(timeout=10)
        try:
            self.storage.close()
        except Exception:
            logging.exception("Error closing database")
        self.write_state(now, stopped=True)
        if self.gpu is not None:
            self.gpu.shutdown()
        logging.info("Monitor stopped after %s.", fmt_duration(now - self.started_at))
