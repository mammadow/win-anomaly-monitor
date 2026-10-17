import math

import pytest

from conftest import make_sample
from resmon.processes import ProcStat
from resmon.rules import (AnomalyEpisodes, BackgroundCpu, GpuAwakeOnBattery, GpuThermal, Leaks,
                          LowDiskSpace, MemoryPressure, Sustained)
from resmon.sampler import THROTTLE_HW_THERMAL


# ── Sustained ─────────────────────────────────────────────────────────

def test_sustained_needs_duration_and_tolerates_short_gaps():
    s = Sustained(on_sec=10, off_sec=5)
    assert [s.update(t, True) for t in range(0, 10)] == [None] * 10
    assert s.update(10, True) == "start"
    assert s.update(12, False) is None           # 2 s blip doesn't end it
    assert s.update(13, True) is None and s.active
    assert s.update(16, False) is None           # 3 s since last true
    assert s.update(18, False) == "end"          # 5 s → over


def test_sustained_resets_if_condition_drops_before_activation():
    s = Sustained(on_sec=10, off_sec=2)
    for t in range(0, 8):
        s.update(t, True)
    s.update(11, False)                          # gone for 4 s > off_sec → timer reset
    assert s.update(12, True) is None
    assert s.update(21, True) is None
    assert s.update(22, True) == "start"


# ── System-level rules ────────────────────────────────────────────────

def test_memory_pressure_episode(mon):
    r = MemoryPressure(mon)
    for t in range(0, 70):
        r.on_sample(make_sample(t, ram_pct=95.0), t)
    for t in range(70, 110):
        r.on_sample(make_sample(t), t)
    [ev] = mon.events.of_kind("memory_pressure")
    assert ev.ts_start == 0 and not ev.is_open
    assert "RAM 95%" in ev.title and "big.exe" in ev.title
    assert ev.details["peak_ram_pct"] == 95.0


def test_short_memory_spike_is_ignored(mon):
    r = MemoryPressure(mon)
    for t in range(0, 30):
        r.on_sample(make_sample(t, ram_pct=95.0), t)
    for t in range(30, 100):
        r.on_sample(make_sample(t), t)
    assert not mon.events.events


def test_gpu_thermal_throttling(mon):
    r = GpuThermal(mon)
    gaming = dict(gpu_awake=1.0, gpu_util_pct=99.0, gpu_temp_c=86.0, gpu_power_w=110.0)
    for t in range(0, 40):
        r.on_sample(make_sample(t, gpu_throttle=THROTTLE_HW_THERMAL, **gaming), t)
    for t in range(40, 80):
        r.on_sample(make_sample(t, gpu_throttle=0, **gaming), t)
    [ev] = mon.events.of_kind("gpu_thermal")
    assert "throttling" in ev.title and "hardware thermal limit" in ev.title and "86°C" in ev.title
    assert not ev.is_open


def test_spurious_driver_thermal_flag_on_a_cool_gpu_is_ignored(mon):
    # Seen on this laptop: flag 0x20 at 32-52 °C, ≤ 16 W, idle — e.g. on the lock screen.
    r = GpuThermal(mon)
    for t in range(0, 120):
        r.on_sample(make_sample(t, gpu_awake=1.0, gpu_temp_c=38.0, gpu_power_w=15.0, gpu_throttle=0x24), t)
    assert not mon.events.events


def test_gpu_hot_without_throttle_flag(mon):
    r = GpuThermal(mon)
    for t in range(0, 40):
        r.on_sample(make_sample(t, gpu_awake=1.0, gpu_temp_c=90.0, gpu_power_w=100.0), t)
    [ev] = mon.events.of_kind("gpu_thermal")
    assert "running hot" in ev.title


def test_dgpu_awake_on_battery(mon):
    r = GpuAwakeOnBattery(mon)
    for t in range(0, 11 * 60):
        r.on_sample(make_sample(t, on_ac=0.0, gpu_awake=1.0, gpu_util_pct=0.0, gpu_power_w=14.0), t)
    [ev] = mon.events.of_kind("gpu_awake_on_battery")
    assert "~14 W" in ev.title


def test_dgpu_awake_on_ac_is_fine(mon):
    r = GpuAwakeOnBattery(mon)
    for t in range(0, 15 * 60):
        r.on_sample(make_sample(t, on_ac=1.0, gpu_awake=1.0, gpu_power_w=14.0), t)
    assert not mon.events.events


def drive(free_gb, total_gb=953.0):
    """This laptop on 2026-10-17: 2.3 GB auto-managed page file, commit limit 34 GB on 31.7 GB RAM."""
    return dict(sysdrive_free_gb=free_gb, sysdrive_total_gb=total_gb, pagefile_gb=2.3, commit_limit_gb=34.0)


def test_nearly_full_system_drive_names_the_commit_risk(mon):
    r = LowDiskSpace(mon)
    r.on_sample(make_sample(0, commit_pct=99.2, **drive(15.0)), 0.0)
    [ev] = mon.events.of_kind("low_disk_space")
    assert ev.severity == "warning" and not ev.is_open
    assert "15.0 GB free of 953 GB (1.6%)" in ev.title
    assert "page file can't grow much" in ev.title and "~34 GB" in ev.title
    d = ev.details
    assert (d["free_gb"], d["total_gb"], d["pagefile_gb"], d["commit_limit_gb"], d["commit_pct"]) == \
        (15.0, 953.0, 2.3, 34.0, 99.2)


def test_low_disk_thresholds(mon):
    r = LowDiskSpace(mon)
    r.on_sample(make_sample(0, **drive(300.0)), 0.0)             # 31 % free: fine
    r.on_sample(make_sample(1, **drive(25.0, total_gb=100.0)), 1.0)   # small drive, > 20 GB: fine
    r.on_sample(make_sample(2, **drive(math.nan)), 2.0)           # unknown: no verdict
    r.on_sample(make_sample(3), 3.0)                              # no drive reading at all
    assert not mon.events.events
    r.on_sample(make_sample(4, **drive(90.0)), 4.0)               # 9.4 % free: below 10 %
    [ev] = mon.events.of_kind("low_disk_space")
    assert "90.0 GB free" in ev.title and "page file" not in ev.title    # more free than RAM


def test_low_disk_thresholds_are_configurable(mon):
    mon.cfg.low_disk_free_gb, mon.cfg.low_disk_free_percent = 10, 0
    r = LowDiskSpace(mon)
    r.on_sample(make_sample(0, **drive(15.0)), 0.0)
    assert not mon.events.events
    r.on_sample(make_sample(1, **drive(9.0)), 1.0)
    assert len(mon.events.of_kind("low_disk_space")) == 1


def test_low_disk_space_is_repeated_only_when_twice_as_bad(mon):
    r = LowDiskSpace(mon)
    t = 0.0
    for free in [15.0] * 60 + [10.0] * 60 + [7.6] * 60:          # an hour at each level
        r.on_sample(make_sample(t, **drive(free)), t)
        t += 60
    assert len(mon.events.of_kind("low_disk_space")) == 1
    r.on_sample(make_sample(t, **drive(7.4)), t)                  # free space halved: speak up
    assert len(mon.events.of_kind("low_disk_space")) == 2
    r.on_sample(make_sample(t + 86_400, **drive(7.4)), t + 86_400)
    assert len(mon.events.of_kind("low_disk_space")) == 2
    r.on_sample(make_sample(t + 7 * 86_400, **drive(7.4)), t + 7 * 86_400)
    assert len(mon.events.of_kind("low_disk_space")) == 3         # weekly reminder


def test_drive_that_recovered_warns_afresh(mon):
    r = LowDiskSpace(mon)
    r.on_sample(make_sample(0, **drive(15.0)), 0.0)
    r.on_sample(make_sample(60, **drive(25.0, total_gb=100.0)), 60.0)    # not low, not twice as free
    r.on_sample(make_sample(120, **drive(19.0)), 120.0)
    assert len(mon.events.of_kind("low_disk_space")) == 1
    r.on_sample(make_sample(180, **drive(120.0)), 180.0)                 # cleaned up
    r.on_sample(make_sample(240, **drive(19.0)), 240.0)                  # filled up again: news
    assert len(mon.events.of_kind("low_disk_space")) == 2


# ── Per-process rules ─────────────────────────────────────────────────

def proc(name="miner.exe", pid=42, cores=0.0, fg=False, private=500 * 2**20, handles=300):
    return ProcStat(pid=pid, name=name, create_time=1.0, cores=cores, read_bps=0, write_bps=0,
                    private_bytes=private, working_set=private, handles=handles, foreground=fg)


def test_background_cpu_hog_detected_and_closed(mon):
    r = BackgroundCpu(mon)
    t = 0.0
    while t <= 6 * 60:
        r.on_sweep([proc(cores=1.5), proc("idle.exe", pid=7, cores=0.01)], t)
        t += 5
    [ev] = mon.events.of_kind("background_cpu")
    assert ev.is_open and "miner.exe" in ev.title and "1.5 CPU cores" in ev.title
    for _ in range(20):
        r.on_sweep([proc(cores=0.0)], t)
        t += 5
    assert not ev.is_open and ev.details["ended"] == "CPU use dropped"


def test_foreground_app_is_not_a_background_hog(mon):
    r = BackgroundCpu(mon)
    for i in range(100):
        r.on_sweep([proc("game.exe", cores=6.0, fg=True)], i * 5.0)
    assert not mon.events.events


def test_ignored_process_is_never_blamed(mon):
    mon.cfg.ignore_processes = ["miner.exe"]
    r = BackgroundCpu(mon)
    for i in range(100):
        r.on_sweep([proc(cores=3.0)], i * 5.0)
    assert not mon.events.events


def test_steady_memory_leak_detected_once(mon):
    r = Leaks(mon)
    for i in range(0, 61):                       # every 2 min for 2 h, +600 MB/h
        t = i * 120.0
        r.on_sweep([proc(private=int(1e9 + 600 * 2**20 * t / 3600) + (i % 3) * 2**20)], t)
    leaks = mon.events.of_kind("memory_leak")
    assert len(leaks) == 1
    assert "miner.exe" in leaks[0].title and leaks[0].details["mb_per_hour"] == pytest.approx(600, rel=0.05)


def test_step_change_is_not_a_leak(mon):
    r = Leaks(mon)
    for i in range(0, 61):
        t = i * 120.0
        r.on_sweep([proc(private=int(1e9 if t < 1800 else 3e9))], t)
    assert not mon.events.of_kind("memory_leak")


def test_handle_hoarder_flagged_immediately(mon):
    r = Leaks(mon)
    r.on_sweep([proc("AcerSystemCentralService.exe", handles=908_806)], 0.0)
    [ev] = mon.events.of_kind("handle_leak")
    assert "908,806 handles" in ev.title
    for i in range(1, 40):                       # keeps growing slowly → no repeat alerts
        r.on_sweep([proc("AcerSystemCentralService.exe", handles=908_806 + i * 200)], i * 120.0)
    assert len(mon.events.of_kind("handle_leak")) == 1


def test_small_slow_handle_growth_is_not_reported(mon):
    r = Leaks(mon)                               # Discord: 4,663 handles, +505/h
    for i in range(0, 91):
        r.on_sweep([proc("Discord.exe", handles=int(2_400 + 505 * i * 120 / 3600))], i * 120.0)
    assert not mon.events.events


def test_known_leaker_is_not_reannounced_after_a_restart(mon):
    r = Leaks(mon)
    r.on_sweep([proc("AcerCentralService.exe", pid=1, handles=2_134_208)], 0.0)
    for i in range(1, 120):                      # service restarted: new pid, leaks again
        r.on_sweep([proc("AcerCentralService.exe", pid=2, handles=150_000 + i * 700)], i * 120.0)
    assert len(mon.events.of_kind("handle_leak")) == 1
    r.on_sweep([proc("AcerCentralService.exe", pid=3, handles=190_000)], 7 * 86_400.0)
    assert len(mon.events.of_kind("handle_leak")) == 1          # managed (guard restarts it): quiet
    r.on_sweep([proc("AcerCentralService.exe", pid=3, handles=4_500_000)], 8 * 86_400.0)
    assert len(mon.events.of_kind("handle_leak")) == 2          # twice as bad: speak up
    r.on_sweep([proc("AcerCentralService.exe", pid=4, handles=150_000)], 40 * 86_400.0)
    assert len(mon.events.of_kind("handle_leak")) == 3          # monthly reminder


def test_handle_growth_trend(mon):
    r = Leaks(mon)
    for i in range(0, 61):
        t = i * 120.0
        r.on_sweep([proc(handles=int(2_000 + 5_000 * t / 3600))], t)
    [ev] = mon.events.of_kind("handle_leak")
    assert "growing steadily" in ev.title and "/h" in ev.title


# ── ML episodes ───────────────────────────────────────────────────────

class StubModel:
    threshold = 2.5
    severe_threshold = 4.0
    n_windows = 20000
    hours_of_data = 55.0

    def rarity(self, score):
        return 0.0004

    def explain(self, x, raw, top=3):
        return [{"feature": "disk_write_bps", "label": "disk write", "value": raw.get("disk_write_bps"),
                 "typical": 6e4, "impact": 0.1, "text": "disk write …"}]


def window(t, score, **kw):
    w = {"ts": t, "score": score, "anomalous": int(score > StubModel.threshold),
         "disk_write_bps": 3e8, "core_max_max_pct": 50.0}
    w.update(kw)
    return w


@pytest.fixture
def episodes(mon):
    mon.model = StubModel()
    mon.model.hours_of_data = StubModel.hours_of_data
    return AnomalyEpisodes(mon)


def feed(ep, scores, t0=0.0):
    t = t0
    for s in scores:
        ep.on_window(window(t, s), None, t + 10)
        t += 10
    return t


def test_isolated_anomalous_windows_never_alert(episodes, mon):
    feed(episodes, [3.0, 1.0, 1.0, 3.0, 3.0, 1.0] * 20)
    assert not mon.events.events


def test_sustained_anomaly_opens_one_episode(episodes, mon):
    t = feed(episodes, [1.0] * 5 + [3.0] * 8 + [1.0] * 10)
    [ev] = mon.events.events
    assert ev.ts_start == 50 and ev.ts_end == 130 and not ev.is_open
    assert ev.details["anomalous_windows"] == 8
    assert "disk write 300.0 MB/s" in ev.title
    assert ev.severity == "info"                  # never below the severe threshold
    assert t > ev.ts_end


def test_severe_background_anomaly_is_a_warning(episodes, mon):
    mon.act["top_cpu"] = [{"name": "updater.exe", "avg_cores": 3.0, "processes": 1}]
    feed(episodes, [5.0] * 4)
    [ev] = mon.events.events
    assert ev.severity == "warning" and "top: updater.exe (3.0 cores)" in ev.title


def test_young_model_only_raises_info(episodes, mon):
    mon.model.hours_of_data = 3.0
    mon.act["top_cpu"] = [{"name": "updater.exe", "avg_cores": 3.0, "processes": 1}]
    feed(episodes, [5.0] * 4)
    [ev] = mon.events.events
    assert ev.severity == "info" and ev.details["model"]["learning"]


def test_anomaly_caused_by_app_in_use_stays_info(episodes, mon):
    mon.act["top_cpu"] = [{"name": "game.exe", "avg_cores": 6.0, "processes": 1}]
    mon.act["foreground"] = "game.exe"
    feed(episodes, [5.0] * 4)
    [ev] = mon.events.events
    assert ev.severity == "info" and "from the app in use, game.exe" in ev.title


def test_recurrence_reopens_the_same_event(episodes, mon):
    t = feed(episodes, [3.0] * 4 + [1.0] * 6)             # episode 1 ends
    t = feed(episodes, [1.0] * 70, t)                     # wait out the 10 min cooldown
    feed(episodes, [3.1] * 4 + [1.0] * 6, t)              # same signature again
    [ev] = mon.events.events
    assert ev.details["recurrences"] == 1 and ev.details["anomalous_windows"] == 8


def test_endless_anomaly_is_capped_and_cooled_down(episodes, mon):
    feed(episodes, [3.0] * (6 * 60))                      # one hour straight
    [ev] = mon.events.events
    assert ev.duration <= 30 * 60 + 10
    assert "new normal" in ev.details["ended"]


def test_unscored_windows_count_as_normal(episodes, mon):
    episodes.on_window({"ts": 0.0}, None, 10.0)
    assert not mon.events.events and episodes.peak_score == -math.inf
