"""
Anomaly detection over 10 s windows: "how far is this moment from the most
similar moments in my history, compared with how far those moments usually
are from *their* neighbours?" (a k-nearest-neighbour, LOF-style score).

Why not Isolation Forest (v2)?
  Tested on this laptop's real history with injected incidents, an Isolation
  Forest detected 0–1 % of: a 20 MB/s upload, CPU at 60 %, RAM at 92 %, and
  a GPU at 85 °C while drawing only 20 W. Its splits are drawn inside the
  range seen in training, so a value far *beyond* anything seen isolates no
  faster than the most extreme training point, and one extreme metric is
  diluted among 16 features. This detector catches 100 % of those (and of a
  GPU crypto-miner) while flagging no more windows of real held-out history.

How it works
  * Features are measured in fixed "meaningful units": 5 % CPU, 5 % RAM,
    ×3 bytes/s, 5 °C, 10 W, 300 MHz, a power/GPU/user state flip = 2 units…
    so distances read as "how many meaningful differences apart".
  * The k most similar windows of the past week are found with all
    features (so "similar" means same power source, user presence, GPU
    state, load…). The window is then measured by how far it sits *above*
    them: using less RAM or CPU than usual is never a problem, and power /
    presence only choose the comparison, they don't count as anomalies.
  * Busy, varied regimes (gaming) are naturally sparser than idle, so that
    distance alone would flag every rare-but-normal regime. The score divides
    by the neighbours' own typical spread (floored at 1.5 units so
    near-duplicate idle windows don't make the tiniest wiggle look huge) —
    the idea behind Local Outlier Factor.
  * A window is anomalous only if the score crosses the threshold *and* at
    least one metric is ≥ 2 units above similar moments (10 % CPU, 10 % RAM,
    ×10 bytes/s, 10 °C…) — rare-but-immaterial wobbles don't count.
  * The threshold is the `alert_quantile` of the same score computed for
    every historical window against the rest of history, ignoring its own
    ±5 minutes (adjacent windows are near-duplicates and would make the
    threshold unrealistically tight).
  * Explanation: the features that differ most from those k similar moments,
    e.g. "disk write 310 MB/s (similar moments: 70 KB/s; higher than 99.98 %
    of all history)".

Pure numpy (no scikit-learn): training on 30 000 windows takes a few seconds
on a background thread; scoring a window takes ~0.1 ms.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .features import FEATURE_NAMES, ML_FEATURES, fmt_feature, to_matrix

K_NEIGHBOURS = 10
LOCAL_FLOOR = 1.5                   # units; see "Busy, varied regimes" above
CALIBRATION_EXCLUDE_SEC = 300       # ignore a window's own ±5 min when calibrating
MIN_EXPLAIN_UNITS = 1.0             # a feature must differ by ≥ 1 unit to be called out
MIN_ALERT_UNITS = 2.0               # …and something must be ≥ 2 units up for a window to count
MODEL_FORMAT = 2                    # bump when scoring changes; older saved models are retrained

# One "meaningful difference" per feature, in to_matrix() units (log10 for byte rates).
UNITS = {
    "cpu_pct": 5.0, "core_max_pct": 15.0, "ram_pct": 5.0, "commit_pct": 5.0,
    "disk_read_bps": 0.5, "disk_write_bps": 0.5, "net_sent_bps": 0.5, "net_recv_bps": 0.5,
    "gpu_util_pct": 10.0, "gpu_power_w": 10.0, "gpu_temp_c": 5.0, "gpu_vram_pct": 5.0,
    "gpu_clock_mhz": 300.0, "gpu_awake": 0.5, "on_ac": 0.5, "user_active": 0.5,
}
_UNITS = np.array([UNITS[c] for c in FEATURE_NAMES])
# Context only selects comparable moments; being on battery or away is never itself the problem.
CONTEXT_FEATURES = {"on_ac", "user_active"}
_COUNTS = np.array([0.0 if c in CONTEXT_FEATURES else 1.0 for c in FEATURE_NAMES], dtype=np.float32)


def excess(Z: np.ndarray, T: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(m, k, f): how many units each query row sits above each of its neighbours, per feature."""
    return np.maximum(np.asarray(Z, np.float32)[:, None, :] - T[idx], 0) * _COUNTS


def _excess_dist(Z: np.ndarray, T: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Mean over neighbours of the one-sided ("only what's higher") distance, chunked."""
    out = np.empty(len(Z))
    for i in range(0, len(Z), 4096):
        ex = excess(Z[i:i + 4096], T, idx[i:i + 4096])
        out[i:i + 4096] = np.sqrt((ex.astype(np.float64) ** 2).sum(axis=2)).mean(axis=1)
    return out


def _pct(share: float) -> str:
    """0.99973 → '99.97%'; keeps enough digits to distinguish rare tails."""
    p = share * 100
    if p >= 99.9 or p <= 0.1:
        return f"{p:.2f}%"
    return f"{p:.1f}%"


def knn(Z: np.ndarray, T: np.ndarray, t_sq: np.ndarray, k: int, *,
        self_idx: np.ndarray | None = None, z_ts: np.ndarray | None = None,
        t_ts: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Distances and indices of the k nearest rows of T for each row of Z, nearest first.
    `self_idx` excludes each query's own row; `z_ts`/`t_ts` exclude history within
    CALIBRATION_EXCLUDE_SEC of the query (both used only when calibrating on T itself).
    """
    Z = np.asarray(Z, dtype=np.float32)
    k = min(k, len(T) - (1 if self_idx is not None else 0))
    dist = np.empty((len(Z), k), dtype=np.float64)
    idx = np.empty((len(Z), k), dtype=np.int64)
    for i in range(0, len(Z), 64):              # 64 × 30k × 4 B ≈ 8 MB per chunk
        c = Z[i:i + 64]
        d2 = (c * c).sum(1)[:, None] + t_sq[None, :] - 2.0 * (c @ T.T)
        np.maximum(d2, 0, out=d2)
        if z_ts is not None:
            near = np.abs(z_ts[i:i + 64, None] - t_ts[None, :]) < CALIBRATION_EXCLUDE_SEC
            near[near.sum(1) > len(T) - k - 1] = False    # tiny history: don't starve the search
            d2[near] = np.inf
        if self_idx is not None:
            d2[np.arange(len(c)), self_idx[i:i + 64]] = np.inf
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        pd = np.take_along_axis(d2, part, axis=1)
        order = np.argsort(pd, axis=1)
        idx[i:i + 64] = np.take_along_axis(part, order, axis=1)
        dist[i:i + 64] = np.sqrt(np.take_along_axis(pd, order, axis=1))
    return dist, idx


@dataclass
class AnomalyModel:
    train: np.ndarray            # (n, f) float32 history in units
    train_raw: np.ndarray        # (n, f) float32 display values (for explanations)
    local: np.ndarray            # (n,) each history window's own mean kNN distance
    threshold: float             # score above this → anomalous
    severe_threshold: float      # …above this → severe
    calib_scores: np.ndarray     # sorted scores of history, for rarity
    feature_sorted: np.ndarray   # (f, n) sorted to_matrix() columns, for global percentiles
    raw_medians: np.ndarray
    n_windows: int
    hours_of_data: float
    k: int = K_NEIGHBOURS
    trained_at: float = field(default_factory=time.time)

    def __post_init__(self):
        self._t_sq = (self.train * self.train).sum(1)

    def _neighbours(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z = (np.atleast_2d(X) / _UNITS).astype(np.float32)
        return Z, knn(Z, self.train, self._t_sq, self.k)[1]

    def evaluate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(scores, anomalous) per row. Score: how far above its similar moments, relative to
        their usual spread (≈0.5 for ordinary windows; higher = more unusual)."""
        Z, idx = self._neighbours(X)
        scores = _excess_dist(Z, self.train, idx) / np.maximum(self.local[idx].mean(axis=1), LOCAL_FLOOR)
        material = excess(Z, self.train, idx).mean(axis=1).max(axis=1) >= MIN_ALERT_UNITS
        return scores, (scores > self.threshold) & material

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.evaluate(X)[0]

    def rarity(self, score: float) -> float:
        """Fraction of historical windows that scored at least this high."""
        n = len(self.calib_scores)
        return float(n - np.searchsorted(self.calib_scores, score, side="left")) / n

    def explain(self, x: np.ndarray, raw: dict, top: int = 3) -> list[dict]:
        """Which features are furthest above the most similar moments in history."""
        Z, idx = self._neighbours(x)
        diff = excess(Z, self.train, idx)[0].astype(np.float64).mean(axis=0)
        out = []
        n = self.feature_sorted.shape[1]
        cutoff = max(MIN_EXPLAIN_UNITS, 0.25 * diff.max())   # skip weak also-rans
        for j in np.argsort(-diff)[:top]:
            if diff[j] < cutoff:
                break
            f = ML_FEATURES[j]
            value = raw.get(f.column, float("nan"))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                similar = float(np.nanmedian(self.train_raw[idx[0], j]))
            col, v = self.feature_sorted[j], np.float32(x[j])
            if f.unit == "state":
                where = f"only {_pct(float(np.mean(np.abs(col - v) < 0.25)))} of history"
            else:
                where = f"higher than {_pct(np.searchsorted(col, v, side='left') / n)} of history"
            out.append({
                "feature": f.column,
                "label": f.label,
                "value": None if np.isnan(value) else round(float(value), 3),
                "similar_moments": None if np.isnan(similar) else round(similar, 3),
                "units_off": round(float(diff[j]), 1),
                "text": f"{f.label} {fmt_feature(f, value)} "
                        f"(similar moments: {fmt_feature(f, similar)}; {where})",
            })
        return out

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        meta = {"format": MODEL_FORMAT, "features": FEATURE_NAMES, "units": _UNITS.tolist(), "threshold": self.threshold,
                "severe_threshold": self.severe_threshold, "k": self.k, "n_windows": self.n_windows,
                "hours_of_data": self.hours_of_data, "trained_at": self.trained_at}
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, meta=np.array(json.dumps(meta)), train=self.train, train_raw=self.train_raw,
                 local=self.local, calib_scores=self.calib_scores, feature_sorted=self.feature_sorted,
                 raw_medians=self.raw_medians)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "AnomalyModel":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if (meta.get("format") != MODEL_FORMAT or meta["features"] != FEATURE_NAMES
                    or meta["units"] != _UNITS.tolist()):
                raise ValueError("saved by an older version of the detector")
            return cls(train=z["train"], train_raw=z["train_raw"], local=z["local"],
                       threshold=meta["threshold"], severe_threshold=meta["severe_threshold"],
                       calib_scores=z["calib_scores"], feature_sorted=z["feature_sorted"],
                       raw_medians=z["raw_medians"], n_windows=meta["n_windows"],
                       hours_of_data=meta["hours_of_data"], k=meta["k"], trained_at=meta["trained_at"])


def fit_model(raw: np.ndarray, ts: np.ndarray | None = None, *, alert_quantile: float,
              window_sec: float, k: int = K_NEIGHBOURS) -> AnomalyModel:
    """`raw`: (n_windows, n_features) display values, columns = FEATURE_NAMES; `ts`: window times."""
    X = to_matrix(raw)
    T = (X / _UNITS).astype(np.float32)
    t_sq = (T * T).sum(1)
    n = len(T)
    # Every history window against the rest of history (minus its own ±5 min).
    dist, idx = knn(T, T, t_sq, k, self_idx=np.arange(n), z_ts=ts, t_ts=ts)
    local = dist.mean(axis=1)          # each window's own spread: the local density scale
    calib = np.sort(_excess_dist(T, T, idx) / np.maximum(local[idx].mean(axis=1), LOCAL_FLOOR))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN column (e.g. no GPU)
        raw_medians = np.nanmedian(raw, axis=0)
    return AnomalyModel(
        train=T,
        train_raw=raw.astype(np.float32),
        local=local.astype(np.float32),
        threshold=float(np.quantile(calib, 1 - alert_quantile)),
        severe_threshold=float(np.quantile(calib, 1 - alert_quantile / 10)),
        calib_scores=calib,
        feature_sorted=np.sort(X, axis=0).T.astype(np.float32),
        raw_medians=raw_medians,
        n_windows=n,
        hours_of_data=n * window_sec / 3600,
        k=min(k, n - 1),
    )


class Trainer:
    """Fits a model on a background thread so sampling never stalls."""

    def __init__(self, data_dir: Path):
        self.model_path = data_dir / "model.npz"
        self._thread: threading.Thread | None = None
        self._result: AnomalyModel | None = None
        self._error: BaseException | None = None
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, load: Callable[[], tuple[np.ndarray, np.ndarray] | None], **fit_kwargs) -> bool:
        if self.busy:
            return False

        def work():
            try:
                t0 = time.perf_counter()
                data = load()
                if data is None:
                    return
                ts, raw = data
                model = fit_model(raw, ts, **fit_kwargs)
                model.save(self.model_path)
                logging.debug("Model fit in %.1fs", time.perf_counter() - t0)
                with self._lock:
                    self._result = model
            except BaseException as e:  # surfaced via poll() on the main thread
                with self._lock:
                    self._error = e

        self._thread = threading.Thread(target=work, name="trainer", daemon=True)
        self._thread.start()
        return True

    def poll(self) -> AnomalyModel | None:
        """A freshly trained model once (raises the worker's exception, if any)."""
        with self._lock:
            model, err = self._result, self._error
            self._result = self._error = None
        if err is not None:
            raise err
        return model

    def load_saved(self, max_age_sec: float) -> AnomalyModel | None:
        """The model from the previous run, if recent enough to trust."""
        if not self.model_path.exists():
            return None
        try:
            model = AnomalyModel.load(self.model_path)
        except Exception as e:
            logging.info("Ignoring saved model (%s)", e)
            return None
        return model if time.time() - model.trained_at <= max_age_sec else None

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
