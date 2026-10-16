import re

import numpy as np
import pytest

from resmon.detector import AnomalyModel, fit_model, knn
from resmon.features import FEATURE_NAMES, to_matrix

COL = {c: i for i, c in enumerate(FEATURE_NAMES)}


def idle_history(n=3000, seed=0):
    """Windows of a quiet laptop: low CPU, small disk/network, dGPU mostly asleep."""
    rng = np.random.default_rng(seed)
    cols = {c: np.zeros(n) for c in FEATURE_NAMES}
    cols["cpu_pct"] = rng.gamma(2, 1.5, n)
    cols["core_max_pct"] = np.clip(rng.normal(20, 6, n), 0, 100)
    cols["ram_pct"] = rng.normal(40, 2, n)
    cols["commit_pct"] = rng.normal(50, 2, n)
    cols["disk_read_bps"] = rng.lognormal(8, 2, n) * (rng.random(n) < 0.3)
    cols["disk_write_bps"] = rng.lognormal(11, 1, n)
    cols["net_sent_bps"] = rng.lognormal(7, 1, n)
    cols["net_recv_bps"] = rng.lognormal(8, 1.5, n)
    awake = rng.random(n) < 0.2
    cols["gpu_awake"] = awake.astype(float)
    cols["gpu_util_pct"] = np.where(awake, rng.integers(0, 5, n), 0)
    cols["gpu_power_w"] = np.where(awake, rng.normal(6, 1, n), 0)
    cols["gpu_temp_c"] = np.where(awake, rng.normal(45, 2, n), np.nan)
    cols["gpu_vram_pct"] = np.where(awake, 3.0, 0)
    cols["gpu_clock_mhz"] = np.where(awake, 210, 0)
    cols["on_ac"] = np.ones(n)
    cols["user_active"] = (rng.random(n) < 0.7).astype(float)
    return np.column_stack([cols[c] for c in FEATURE_NAMES])


@pytest.fixture(scope="module")
def model():
    raw = idle_history()
    ts = 1_000_000.0 + 10.0 * np.arange(len(raw))
    return fit_model(raw, ts, alert_quantile=0.001, window_sec=10)


def incident(seed, **values):
    raw = idle_history(50, seed=seed)
    raw[:, COL["gpu_awake"]] = 0.0
    raw[:, COL["gpu_temp_c"]] = np.nan
    for k, v in values.items():
        raw[:, COL[k]] = v
    return raw


def flagged(model, raw):
    return np.mean(model.evaluate(to_matrix(raw))[1])


def test_threshold_is_calibrated_to_history(model):
    assert flagged(model, idle_history(3000, seed=7)) < 0.01   # ~0.1 % expected


@pytest.mark.parametrize("values", [
    {"ram_pct": 92.0},
    {"cpu_pct": 60.0},
    {"net_sent_bps": 20e6},
    {"disk_write_bps": 300e6},
    {"gpu_awake": 1.0, "gpu_util_pct": 99.0, "gpu_power_w": 90.0, "gpu_temp_c": 75.0,
     "gpu_clock_mhz": 2100.0, "gpu_vram_pct": 30.0},        # GPU miner on an idle machine
], ids=["ram", "cpu", "upload", "disk-write", "gpu-miner"])
def test_single_metric_incidents_are_caught(model, values):
    # Isolation Forest (v2) caught ~0 % of these on this laptop's real history.
    assert flagged(model, incident(11, **values)) >= 0.9


def test_trickles_do_not_matter(model):
    assert flagged(model, incident(12, disk_read_bps=30_000.0)) == 0.0


@pytest.mark.parametrize("values", [{"ram_pct": 20.0}, {"commit_pct": 20.0}, {"cpu_pct": 0.0},
                                    {"on_ac": 0.0}, {"user_active": 0.0}],
                         ids=["low-ram", "low-commit", "zero-cpu", "on-battery", "user-away"])
def test_using_less_or_changing_context_is_not_a_problem(model, values):
    assert flagged(model, incident(15, **values)) == 0.0


def test_explanations_only_list_what_is_higher(model):
    raw = incident(16, ram_pct=20.0, cpu_pct=70.0)[0]
    why = model.explain(to_matrix(raw[None, :])[0], dict(zip(FEATURE_NAMES, raw)))
    assert [d["feature"] for d in why] == ["cpu_pct"]


def test_explanation_names_the_feature_and_similar_moments(model):
    raw = incident(13, ram_pct=92.0)[0]
    why = model.explain(to_matrix(raw[None, :])[0], dict(zip(FEATURE_NAMES, raw)))
    assert why[0]["feature"] == "ram_pct"
    assert "RAM 92%" in why[0]["text"]
    assert re.search(r"similar moments: 4\d%", why[0]["text"])   # the closest history has ~40-46 %
    assert "higher than 100.00% of history" in why[0]["text"]
    assert why[0]["units_off"] > 9


def test_rarity_and_severity_ordering(model):
    mild = model.score(to_matrix(incident(14, cpu_pct=30.0)[:1]))[0]
    wild = model.score(to_matrix(incident(14, cpu_pct=95.0)[:1]))[0]
    assert wild > mild > model.threshold
    assert model.rarity(wild) <= model.rarity(mild) <= 0.001


def test_knn_exclusions():
    T = np.arange(10, dtype=np.float32)[:, None]
    t_sq = (T * T).sum(1)
    d, i = knn(T[[3]], T, t_sq, 2, self_idx=np.array([3]))
    assert set(i[0]) == {2, 4} and np.allclose(d, 1)
    ts = np.arange(10) * 100.0                 # neighbours within 300 s are excluded too
    d, i = knn(T[[3]], T, t_sq, 2, self_idx=np.array([3]), z_ts=ts[[3]], t_ts=ts)
    assert set(i[0]) == {0, 6}


def test_save_load_roundtrip(model, tmp_path):
    p = tmp_path / "model.npz"
    model.save(p)
    loaded = AnomalyModel.load(p)
    X = to_matrix(idle_history(200, seed=3))
    np.testing.assert_array_equal(loaded.score(X), model.score(X))
    assert loaded.threshold == model.threshold and loaded.n_windows == model.n_windows


def test_tiny_or_degenerate_history_still_trains():
    raw = idle_history(40)
    raw[:, COL["commit_pct"]] = np.nan                        # e.g. non-Windows
    m = fit_model(raw, 10.0 * np.arange(40), alert_quantile=0.001, window_sec=10)
    assert np.isfinite(m.threshold) and m.k <= 39
    assert np.isnan(m.raw_medians[COL["commit_pct"]])
    same = np.repeat(idle_history(1), 30, axis=0)              # all-identical windows
    m = fit_model(same, None, alert_quantile=0.001, window_sec=10)
    assert np.isfinite(m.score(to_matrix(same[:1]))).all()
