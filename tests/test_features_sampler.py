import math
import sys

import numpy as np
import pytest

from conftest import make_sample
from resmon import sampler as sampler_mod
from resmon import winapi
from resmon.features import (FEATURE_NAMES, ML_FEATURES, WindowAggregator, fmt_bytes, fmt_feature,
                             to_matrix)
from resmon.sampler import SAMPLE_COLUMNS, GpuSampler, SystemSampler, _rate


def test_windows_align_to_wall_clock_and_skip_partials():
    agg = WindowAggregator(window_sec=10)
    out = [agg.add(make_sample(t)) for t in np.arange(1000.0, 1035.0)]
    done = [w for w in out if w]
    assert [w["ts"] for w in done] == [1000.0, 1010.0, 1020.0]
    assert all(w["n"] == 10 for w in done)

    agg = WindowAggregator(window_sec=10)
    out = [agg.add(make_sample(t)) for t in (1007.0, 1008.0, 1009.0, 1010.0)]
    assert not any(out)                      # 3 samples < 60 % of a window → dropped


def test_window_aggregates():
    agg = WindowAggregator(window_sec=10)
    samples = [make_sample(1000.0 + i, cpu_pct=float(i), core_max_pct=100.0 if i == 3 else 10.0,
                           user_idle_sec=5.0 if i < 5 else 500.0,
                           gpu_awake=1.0 if i < 2 else 0.0, gpu_temp_c=50.0 if i < 2 else math.nan)
               for i in range(10)]
    for s in samples:
        agg.add(s)
    w = agg.add(make_sample(1010.0))
    assert w["cpu_pct"] == pytest.approx(4.5)
    assert w["cpu_max_pct"] == 9.0
    assert w["core_max_max_pct"] == 100.0
    assert w["user_active"] == pytest.approx(0.5)
    assert w["gpu_awake"] == pytest.approx(0.2)
    assert w["gpu_temp_c"] == pytest.approx(50.0)      # mean over awake samples only


def test_to_matrix_logs_rates_and_zero_fills_nan():
    row = {c: 1.0 for c in FEATURE_NAMES}
    row["disk_write_bps"] = 65_536.0 * 99
    row["disk_read_bps"] = 30_000.0                     # a trickle stays near zero
    row["gpu_temp_c"] = math.nan
    X = to_matrix([row])
    assert X[0, FEATURE_NAMES.index("disk_write_bps")] == pytest.approx(2.0)
    assert X[0, FEATURE_NAMES.index("disk_read_bps")] < 0.2
    assert X[0, FEATURE_NAMES.index("gpu_temp_c")] == 0.0
    row["gpu_temp_c"], row["gpu_awake"] = 60.0, 0.5      # temp weighted by awake fraction
    assert to_matrix([row])[0, FEATURE_NAMES.index("gpu_temp_c")] == pytest.approx(30.0)
    assert np.isfinite(X).all()


def test_formatting():
    assert fmt_bytes(1_500_000, per_sec=True) == "1.5 MB/s"
    assert fmt_bytes(512) == "512 B"
    state = next(f for f in ML_FEATURES if f.column == "on_ac")
    assert fmt_feature(state, 0.0) == "battery"
    assert fmt_feature(state, 1.0) == "AC"
    assert fmt_feature(state, 0.5) == "AC 50% of the time"


def test_rate_clamps_counter_resets():
    assert _rate(1_000, 400, 2.0) == 300.0
    assert _rate(400, 1_000, 2.0) == 0.0     # NIC vanished → v2 recorded -78 MB/s here


def test_gpu_sampler_without_nvml(monkeypatch):
    monkeypatch.setattr(sampler_mod, "pynvml", None)
    g = GpuSampler()
    s = g.sample()
    assert not g.available and math.isnan(s["gpu_awake"])
    assert g.process_utilization() == {}


def test_gpu_asleep_is_a_state_not_zero_degrees(monkeypatch):
    class FakeNVMLError(Exception):
        pass

    class FakeNVML:
        NVMLError = FakeNVMLError

        @staticmethod
        def nvmlDeviceGetUtilizationRates(h):
            raise FakeNVMLError("GPU is powered down")

    g = GpuSampler.__new__(GpuSampler)
    g._handle, g.awake = object(), True
    monkeypatch.setattr(sampler_mod, "pynvml", FakeNVML)
    s = g.sample()
    assert s["gpu_awake"] == 0.0 and math.isnan(s["gpu_temp_c"]) and not g.awake


def test_real_system_sample_has_every_column():
    s = SystemSampler(gpu=None).sample(123.0)
    assert set(SAMPLE_COLUMNS) <= set(s)
    assert 0 <= s["cpu_pct"] <= 100 and s["core_max_pct"] >= s["cpu_pct"] - 1e-9
    assert s["ts"] == 123.0
    assert 0 < s["sysdrive_free_gb"] <= s["sysdrive_total_gb"]


def test_system_drive_space_is_read_once_a_minute(monkeypatch):
    calls = []
    real = sampler_mod.psutil.disk_usage
    monkeypatch.setattr(sampler_mod.psutil, "disk_usage", lambda p: calls.append(p) or real(p))
    sm = SystemSampler(gpu=None)
    first, second = sm.sample(1.0), sm.sample(2.0)
    assert calls == [sampler_mod.SYSTEM_DRIVE]
    assert second["sysdrive_free_gb"] == first["sysdrive_free_gb"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows API")
def test_native_process_snapshot_matches_psutil():
    import os

    import psutil
    procs = {p.pid: p for p in winapi.ProcessSnapshotter().snapshot()}
    me = procs[os.getpid()]
    ps = psutil.Process()
    assert me.name.lower().startswith("python")
    assert me.cpu_time == pytest.approx(sum(ps.cpu_times()[:2]), abs=1.0)
    assert me.working_set == pytest.approx(ps.memory_info().rss, rel=0.35)
    assert me.create_time == pytest.approx(ps.create_time(), abs=1.0)
    assert len(procs) == pytest.approx(len(psutil.pids()), abs=50)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows API")
def test_performance_info_is_sane():
    pi = winapi.performance_info()
    assert pi.processes > 10 and pi.handles > 1000 and 0 < pi.commit_percent < 100
    assert 0 < pi.physical_bytes <= pi.commit_limit_bytes and 0 <= pi.pagefile_bytes <= pi.commit_limit_bytes
    idle = winapi.user_idle_seconds()
    assert idle is None or idle >= 0
