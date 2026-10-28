import dataclasses
import json
import logging
import math
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import make_sample
from resmon import winapi
from resmon.cli import main
from resmon.detector import fit_model
from resmon.engine import InstanceLock, Monitor
from resmon.events import EventManager
from resmon.processes import ProcessTracker
from resmon.rules import Detector
from resmon.storage import (Storage, db_summary, load_training_windows, load_windows,
                            query_events)
from resmon.winapi import ProcInfo


class SimSampler:
    """Idle laptop with small noise; `overrides` simulate incidents."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.overrides: dict = {}
        self.fail_next = False
        self.resets = 0

    def sample(self, ts):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("simulated sensor failure")
        r = self.rng
        s = make_sample(ts, cpu_pct=float(r.gamma(2, 1.5)), core_max_pct=float(r.normal(20, 5)),
                        disk_write_bps=float(r.lognormal(11, 1)), net_recv_bps=float(r.lognormal(8, 1)),
                        user_idle_sec=float(r.choice([2.0, 600.0])))
        s.update(self.overrides)
        return s

    def reset_rates(self):
        self.resets += 1


class SimProcs(ProcessTracker):
    """Real ProcessTracker logic over a synthetic process table."""

    def __init__(self):
        super().__init__()
        self.table = {
            100: ProcInfo(100, "explorer.exe", 1.0, 0.0, 200 << 20, 150 << 20, 0, 0, 40, 3000),
            200: ProcInfo(200, "backup.exe", 1.0, 0.0, 100 << 20, 80 << 20, 0, 0, 10, 400),
        }

    def snapshot(self):
        return [dataclasses.replace(p) for p in self.table.values()]   # fresh objects, like the OS

    def advance(self, pid, dt, cores=0.0, write_bps=0.0, other_bps=0.0):
        p = self.table[pid]
        p.cpu_time += cores * dt
        p.write_bytes += int(write_bps * dt)
        p.other_bytes += int(other_bps * dt)


def simulate(m, sampler, procs, t0, seconds, on_tick=None):
    t = t0
    for i in range(int(seconds)):
        if on_tick:
            on_tick(t)
        procs.advance(100, 1.0, cores=0.02)
        m.tick(now=t, mono=t)
        t += 1.0
    return t


@pytest.fixture
def sim(cfg, monkeypatch):
    sampler, procs = SimSampler(), SimProcs()
    m = Monitor(cfg, sampler=sampler, procs=procs, gpu=None)
    monkeypatch.setattr(m, "_start_training", lambda now: None)   # tests drive training directly
    yield m, sampler, procs
    m.storage.close()


def test_end_to_end_disk_storm_is_detected_and_attributed(sim, cfg):
    m, sampler, procs = sim
    t0 = time.time()
    t = simulate(m, sampler, procs, t0, 2 * 3600)                 # 2 h of quiet history
    m.storage.flush()
    ts, raw = load_training_windows(m.db_path, 0, 100_000)
    assert len(raw) >= 700
    m.model = fit_model(raw, ts, alert_quantile=cfg.alert_quantile, window_sec=cfg.window_sec)

    t = simulate(m, sampler, procs, t, 300)                        # still quiet: nothing fires
    assert not query_events(m.db_path, 0, ["anomaly"])

    sampler.overrides = {"disk_write_bps": 300e6}
    t = simulate(m, sampler, procs, t, 120,
                 on_tick=lambda _: procs.advance(200, 1.0, cores=0.6, write_bps=300e6))
    sampler.overrides = {}
    simulate(m, sampler, procs, t, 180)
    m.storage.flush()

    [ev] = query_events(m.db_path, 0, ["anomaly"])
    assert not ev["is_open"]
    assert "disk write" in ev["title"] and "backup.exe" in ev["title"]
    assert ev["details"]["why"][0]["feature"] == "disk_write_bps"
    assert 90 <= ev["ts_end"] - ev["ts_start"] <= 150
    line = json.loads((cfg.data_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert line["id"] == ev["id"] and line["kind"] == "anomaly"


def test_end_to_end_upload_burst_names_the_uploader(sim, cfg):
    m, sampler, procs = sim
    procs.table[300] = ProcInfo(300, "ServiceModelReg.exe", 1.0, 0.0, 60 << 20, 50 << 20, 0, 0, 12, 400)
    t = simulate(m, sampler, procs, time.time(), 60)
    sampler.overrides = {"net_sent_bps": 13e6}                  # 39 MB in 3 s
    t = simulate(m, sampler, procs, t, 3, on_tick=lambda _: procs.advance(300, 1.0, cores=0.3, other_bps=10e6))
    sampler.overrides = {}
    simulate(m, sampler, procs, t, 60)
    m.storage.flush()

    [ev] = query_events(m.db_path, 0, ["upload_burst"])
    assert ev["severity"] == "warning" and not ev["is_open"]
    assert ev["title"].startswith("Uploaded 39.0 MB in 3s (peak 13.0 MB/s) — top: ServiceModelReg.exe (≈30 MB")
    assert [p["name"] for p in ev["details"]["top_net"]] == ["ServiceModelReg.exe"]
    assert not query_events(m.db_path, 0, ["anomaly"])          # the ML can't see it (no model, and too short)


def test_event_severity_only_escalates_unless_the_rule_revises_it(tmp_path):
    toasts = []
    em = EventManager(Storage(tmp_path / "m.db"), tmp_path / "events.jsonl",
                      notifier=SimpleNamespace(send=lambda title, body: toasts.append(body)))
    a = em.open("memory_pressure", "warning", "a", 100.0, {})
    em.close(a, 110.0, severity="info")
    b = em.open("upload_burst", "info", "b?", 200.0, {})
    em.close(b, 210.0, severity="warning", title="b!")             # escalating at close warns too
    c = em.open("upload_burst", "warning", "c", 300.0, {})
    em.update(c, 305.0, severity="info", allow_downgrade=True)
    em.close(c, 310.0)
    em.close(em.open("upload_burst", "warning", "d", 400.0, {}), 410.0, severity="info", allow_downgrade=True)
    em.storage.close()
    assert [e["severity"] for e in query_events(tmp_path / "m.db", 0)] == ["warning", "warning", "info", "info"]
    assert toasts == ["a", "b!", "c", "d"]


class FakeClocks:
    """winapi.sleep_clocks() that follows the simulated wall clock."""

    def __init__(self):
        self.up = self.asleep = self.wall_offset = 0.0

    def follow(self, t):
        self.up = t - self.wall_offset

    def __call__(self):
        return self.up, self.up - self.asleep


@pytest.fixture
def clocks(monkeypatch):
    c = FakeClocks()
    monkeypatch.setattr(winapi, "sleep_clocks", c)
    return c


class ResetSpy(Detector):
    resets = 0

    def reset(self):
        self.resets += 1


@pytest.mark.parametrize("cause", ["slept", "clock set forward", "clock set back", "not windows"])
def test_sleep_gap_rebaselines_without_alerts(sim, clocks, monkeypatch, cause):
    m, sampler, procs = sim
    if cause == "not windows":                                      # no clocks: every gap counts as sleep
        monkeypatch.setattr(winapi, "sleep_clocks", lambda: None)
    t = simulate(m, sampler, procs, time.time(), 60, on_tick=clocks.follow)
    ev = m.events.open("test", "info", "ongoing", t, {})
    jump = -3600 if cause == "clock set back" else 3600
    if cause == "slept":
        clocks.asleep += jump
    elif cause.startswith("clock"):
        clocks.wall_offset += jump
    t = simulate(m, sampler, procs, t + jump, 30, on_tick=clocks.follow)   # an hour later by the wall clock
    assert sampler.resets == 1
    assert m.grace_until > t - 30
    assert not ev.is_open


def test_starved_loop_is_logged_as_a_stall_not_sleep(sim, clocks, caplog):
    m, sampler, procs = sim
    spy = ResetSpy(m)
    m.detectors.append(spy)
    t = simulate(m, sampler, procs, time.time(), 60, on_tick=clocks.follow)
    ev = m.events.open("test", "info", "ongoing", t, {})
    grace, n = m.grace_until, m.n_samples
    with caplog.at_level(logging.INFO):
        simulate(m, sampler, procs, t + 12, 30, on_tick=clocks.follow)   # held up 13 s, machine awake
    assert "Main loop stalled for 13s" in caplog.text
    assert "sleep/hibernate" not in caplog.text
    assert ev.is_open and spy.resets == 0 and sampler.resets == 0
    assert m.grace_until == grace
    assert m.n_samples == n + 30                                    # the late tick still took its sample


def test_samples_windows_and_nan_roundtrip(sim):
    m, sampler, procs = sim
    simulate(m, sampler, procs, 1_000_000.0, 35)
    m.storage.flush()
    s = db_summary(m.db_path)
    assert s["samples"]["n"] == 35 and s["windows"]["n"] == 3
    assert s["latest_sample"]["gpu_temp_c"] is None                 # NaN stored as NULL
    rows = load_windows(m.db_path, 0)
    assert [r["ts"] for r in rows] == sorted(r["ts"] for r in rows)
    assert math.isnan(rows[0]["gpu_temp_c"])


def test_reader_can_query_while_monitor_holds_the_db(tmp_path):
    st = Storage(tmp_path / "m.db")
    st.add_sample(make_sample(1.0))
    st.flush()
    assert db_summary(tmp_path / "m.db")["samples"]["n"] == 1     # separate read-only connection
    st.close()


def test_prune_respects_retention(tmp_path):
    st = Storage(tmp_path / "m.db")
    now = 10_000_000.0
    for ts in (now - 50 * 3600, now - 3600):
        st.add_sample(make_sample(ts))
    st.flush()
    eid = st.insert_event("x", "info", now - 400 * 86400, "old", {})
    st.update_event(eid, severity="info", ts_end=now - 400 * 86400, is_open=False, title="old", details={})
    assert st.prune(now, raw_hours=48, window_days=14, event_days=180) == 2
    st.close()


def test_run_loop_survives_errors_and_stops_cleanly(tmp_path):
    from resmon.config import MonitorConfig
    cfg = MonitorConfig(data_dir=str(tmp_path), sample_interval_sec=0.05, window_sec=1,
                        process_interval_sec=0.2, efficiency_mode=False)
    sampler = SimSampler()
    m = Monitor(cfg, sampler=sampler, procs=SimProcs(), gpu=None)
    sampler.fail_next = True
    threading.Timer(1.5, m.stop).start()
    m.run()
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["stopped"] and state["samples_this_run"] > 10
    if sys.platform == "win32":                        # the probe helper ran and was stopped with it
        assert state["ui_probe"]["pid"] and not state["ui_probe"]["running"]


def test_background_training_produces_and_saves_a_model(cfg):
    m = Monitor(cfg, sampler=SimSampler(), procs=SimProcs(), gpu=None)
    # Keep the scheduler from training mid-simulation: a fit still in flight on partial
    # history would make the trainer busy, so the explicit start below would be ignored.
    m._next_train = math.inf
    simulate(m, m.sampler, m.procs, 2_000_000.0, 400 * 10)          # 400 windows
    m.storage.flush()
    m.cfg.min_training_hours = 0.5
    m._start_training(2_000_000.0 + 4000)
    deadline = time.time() + 60
    model = None
    while model is None and time.time() < deadline:
        time.sleep(0.2)
        model = m.trainer.poll()
    m.storage.close()
    assert model is not None and model.n_windows >= 390
    assert (cfg.data_path / "model.npz").exists()


def test_instance_lock_is_exclusive(tmp_path):
    a = InstanceLock(tmp_path / "x.lock")
    assert a.acquire()
    assert InstanceLock.is_held(tmp_path / "x.lock")
    a.release()
    assert not InstanceLock.is_held(tmp_path / "x.lock")


def test_cli_status_report_reset(sim, cfg, capsys):
    m, sampler, procs = sim
    simulate(m, sampler, procs, time.time(), 30)
    m.events.point("handle_leak", "warning", "leaky.exe holds 500,000 handles", time.time(), {"handles": 500000})
    m.storage.close()
    d = str(cfg.data_path)

    assert main(["--data-dir", d, "status"]) == 0
    out = capsys.readouterr().out
    assert "NOT RUNNING" in out and "learning baseline" in out and "1 event" in out

    assert main(["--data-dir", d, "report"]) == 0
    out = capsys.readouterr().out
    assert "leaky.exe holds 500,000 handles" in out and "WARNING" in out

    assert main(["--data-dir", d, "report", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["kind"] == "handle_leak"

    m.storage = Storage(cfg.data_path / "metrics.db")
    m.events.storage = m.storage
    t = time.time()
    m.events.point("ui_freeze", "warning", "brave.exe stopped responding for 2.4 s — dGPU powered off 0.8 s into it",
                   t, {"clues": ["dGPU powered off 0.8 s into it", "heavy disk I/O (150.0 MB/s)"],
                       "ended": "responded", "foreground": "brave.exe"}, ts_end=t + 2.4)
    m.storage.close()
    assert main(["--data-dir", d, "report", "-k", "ui_freeze"]) == 0
    out = capsys.readouterr().out
    assert "ui_freeze  lasted 2s" in out and "also  : heavy disk I/O" in out and "ended" not in out

    assert main(["--data-dir", d, "reset", "--yes"]) == 0
    assert not (cfg.data_path / "metrics.db").exists()
    m.storage = Storage(cfg.data_path / "metrics.db")             # let the fixture close something


def test_leak_alerts_are_not_repeated_after_a_monitor_restart(cfg):
    procs = SimProcs()
    procs.table[300] = ProcInfo(300, "AcerSystemCentralService.exe", 5.0, 0.0, 1 << 20, 1 << 20, 0, 0, 5, 900_000)
    for run in range(2):
        m = Monitor(cfg, sampler=SimSampler(), procs=procs, gpu=None)
        simulate(m, m.sampler, procs, time.time() + run * 100, 5)
        m.storage.close()
    events = query_events(cfg.data_path / "metrics.db", 0, ["handle_leak"])
    assert len(events) == 1 and "900,000 handles" in events[0]["title"]


def test_low_disk_warning_is_not_repeated_after_a_monitor_restart(cfg):
    for run in range(2):
        sampler = SimSampler()
        sampler.overrides = {"sysdrive_free_gb": 15.0, "sysdrive_total_gb": 953.0}
        m = Monitor(cfg, sampler=sampler, procs=SimProcs(), gpu=None)
        simulate(m, sampler, m.procs, time.time() + run * 100, 5)
        m.storage.close()
    [ev] = query_events(cfg.data_path / "metrics.db", 0, ["low_disk_space"])
    assert "15.0 GB free of 953 GB" in ev["title"]
