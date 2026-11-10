//! Adaptive LOF-style k-Nearest-Neighbour anomaly detector (`AnomalyModel`)
//! with Rayon multi-core parallel training and background `Trainer`.

use crate::features::{
    fmt_feature, get_metric, nanmedian, to_matrix, FEATURE_NAMES, ML_FEATURES, MetricMap,
    NUM_FEATURES,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

pub const K_NEIGHBOURS: usize = 10;
pub const LOCAL_FLOOR: f64 = 1.5;
pub const CALIBRATION_EXCLUDE_SEC: f64 = 300.0;
pub const MIN_EXPLAIN_UNITS: f64 = 1.0;
pub const MIN_ALERT_UNITS: f64 = 2.0;
pub const MODEL_FORMAT: u32 = 2;

/// One "meaningful difference" per feature in `to_matrix()` units (log10 for byte rates).
pub const UNITS: [f64; NUM_FEATURES] = [
    5.0,   // cpu_pct
    15.0,  // core_max_pct
    5.0,   // ram_pct
    5.0,   // commit_pct
    0.5,   // disk_read_bps (log10)
    0.5,   // disk_write_bps (log10)
    0.5,   // net_sent_bps (log10)
    0.5,   // net_recv_bps (log10)
    10.0,  // gpu_util_pct
    10.0,  // gpu_power_w
    5.0,   // gpu_temp_c
    5.0,   // gpu_vram_pct
    300.0, // gpu_clock_mhz
    0.5,   // gpu_awake
    0.5,   // on_ac
    0.5,   // user_active
];

/// Context features (`on_ac`, `user_active`) select comparable moments in history
/// but do not themselves count as resource anomalies.
pub const COUNTS: [f32; NUM_FEATURES] = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
];

#[inline]
pub fn scale_to_units(x: &[f64; NUM_FEATURES]) -> [f32; NUM_FEATURES] {
    let mut z = [0.0f32; NUM_FEATURES];
    for j in 0..NUM_FEATURES {
        z[j] = (x[j] / UNITS[j]) as f32;
    }
    z
}

/// Scalar squared Euclidean distance between two 16-feature unit vectors.
#[inline]
pub fn sq_dist_scalar(a: &[f32; NUM_FEATURES], b: &[f32; NUM_FEATURES]) -> f32 {
    let mut sum = 0.0f32;
    for j in 0..NUM_FEATURES {
        let d = a[j] - b[j];
        sum += d * d;
    }
    sum.max(0.0)
}

/// Hardware-accelerated AVX2 squared Euclidean distance across two 256-bit YMM registers
/// (`16` `f32` elements = `2 × 8` lanes).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sq_dist_avx2(a: &[f32; NUM_FEATURES], b: &[f32; NUM_FEATURES]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let a0 = _mm256_loadu_ps(a.as_ptr());
    let b0 = _mm256_loadu_ps(b.as_ptr());
    let d0 = _mm256_sub_ps(a0, b0);
    let sq0 = _mm256_mul_ps(d0, d0);

    let a1 = _mm256_loadu_ps(a.as_ptr().add(8));
    let b1 = _mm256_loadu_ps(b.as_ptr().add(8));
    let d1 = _mm256_sub_ps(a1, b1);
    let sq1 = _mm256_mul_ps(d1, d1);

    let sum8 = _mm256_add_ps(sq0, sq1);
    let hi128 = _mm256_extractf128_ps(sum8, 1);
    let lo128 = _mm256_castps256_ps128(sum8);
    let sum4 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum4);
    let sums = _mm_add_ps(sum4, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let total = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(total).max(0.0)
}

#[inline]
pub fn has_avx2() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
pub fn sq_dist(a: &[f32; NUM_FEATURES], b: &[f32; NUM_FEATURES], use_avx2: bool) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_avx2 {
        // SAFETY: `use_avx2` is only true when `is_x86_feature_detected!("avx2")` succeeded.
        return unsafe { sq_dist_avx2(a, b) };
    }
    let _ = use_avx2;
    sq_dist_scalar(a, b)
}

#[inline]
fn push_best(best: &mut Vec<(f32, usize)>, k: usize, worst_d2: &mut f32, d2: f32, m: usize) {
    if best.len() < k {
        let pos = best.partition_point(|&(bd, _)| bd <= d2);
        best.insert(pos, (d2, m));
        if best.len() == k {
            *worst_d2 = best[k - 1].0;
        }
    } else if d2 < *worst_d2 {
        let pos = best.partition_point(|&(bd, _)| bd <= d2);
        best.insert(pos, (d2, m));
        best.truncate(k);
        *worst_d2 = best[k - 1].0;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn scan_range(
    z: &[f32; NUM_FEATURES],
    t: &[[f32; NUM_FEATURES]],
    start: usize,
    end: usize,
    self_idx: Option<usize>,
    k: usize,
    best: &mut Vec<(f32, usize)>,
    worst_d2: &mut f32,
    use_avx2: bool,
) {
    if start >= end {
        return;
    }
    if let Some(s) = self_idx.filter(|&s| s >= start && s < end) {
        for (off, tm) in t[start..s].iter().enumerate() {
            let d2 = sq_dist(z, tm, use_avx2);
            if best.len() < k || d2 < *worst_d2 {
                push_best(best, k, worst_d2, d2, start + off);
            }
        }
        for (off, tm) in t[s + 1..end].iter().enumerate() {
            let d2 = sq_dist(z, tm, use_avx2);
            if best.len() < k || d2 < *worst_d2 {
                push_best(best, k, worst_d2, d2, s + 1 + off);
            }
        }
    } else {
        for (off, tm) in t[start..end].iter().enumerate() {
            let d2 = sq_dist(z, tm, use_avx2);
            if best.len() < k || d2 < *worst_d2 {
                push_best(best, k, worst_d2, d2, start + off);
            }
        }
    }
}

/// Find the `k` nearest neighbours in `t` for a single query point `z`, sorted nearest first.
#[allow(clippy::too_many_arguments)]
fn knn_single(
    z: &[f32; NUM_FEATURES],
    t: &[[f32; NUM_FEATURES]],
    k: usize,
    self_idx: Option<usize>,
    z_ts: Option<f64>,
    t_ts: Option<&[f64]>,
    ts_sorted: bool,
    use_avx2: bool,
) -> (Vec<f64>, Vec<usize>) {
    let k = k.min(t.len().saturating_sub(usize::from(self_idx.is_some())));
    if k == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut best: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    let mut worst_d2 = f32::INFINITY;

    match (z_ts, t_ts) {
        (Some(zt), Some(tts)) if ts_sorted && tts.len() == t.len() => {
            let lo = tts.partition_point(|&tt| tt <= zt - CALIBRATION_EXCLUDE_SEC);
            let hi = tts.partition_point(|&tt| tt < zt + CALIBRATION_EXCLUDE_SEC);
            let outside_count = lo + (t.len() - hi);
            let self_outside = self_idx.map(|s| s < lo || s >= hi).unwrap_or(false);
            if outside_count >= k + usize::from(self_outside) {
                scan_range(z, t, 0, lo, self_idx, k, &mut best, &mut worst_d2, use_avx2);
                scan_range(
                    z,
                    t,
                    hi,
                    t.len(),
                    self_idx,
                    k,
                    &mut best,
                    &mut worst_d2,
                    use_avx2,
                );
            } else {
                scan_range(
                    z,
                    t,
                    0,
                    t.len(),
                    self_idx,
                    k,
                    &mut best,
                    &mut worst_d2,
                    use_avx2,
                );
            }
        }
        (Some(zt), Some(tts)) if tts.len() == t.len() => {
            for (m, tm) in t.iter().enumerate() {
                if self_idx == Some(m) || (zt - tts[m]).abs() < CALIBRATION_EXCLUDE_SEC {
                    continue;
                }
                let d2 = sq_dist(z, tm, use_avx2);
                if best.len() < k || d2 < worst_d2 {
                    push_best(&mut best, k, &mut worst_d2, d2, m);
                }
            }
            if best.len() < k {
                best.clear();
                worst_d2 = f32::INFINITY;
                scan_range(
                    z,
                    t,
                    0,
                    t.len(),
                    self_idx,
                    k,
                    &mut best,
                    &mut worst_d2,
                    use_avx2,
                );
            }
        }
        _ => {
            scan_range(
                z,
                t,
                0,
                t.len(),
                self_idx,
                k,
                &mut best,
                &mut worst_d2,
                use_avx2,
            );
        }
    }

    let dists = best.iter().map(|&(d2, _)| (d2 as f64).sqrt()).collect();
    let idxs = best.iter().map(|&(_, idx)| idx).collect();
    (dists, idxs)
}

/// Parallel kNN search for a batch of query rows `z_batch` against `t`.
pub fn knn(
    z_batch: &[[f32; NUM_FEATURES]],
    t: &[[f32; NUM_FEATURES]],
    k: usize,
    use_self_idx: bool,
    z_ts: Option<&[f64]>,
    t_ts: Option<&[f64]>,
) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
    let ts_sorted = t_ts
        .map(|tts| tts.windows(2).all(|w| w[0] <= w[1]))
        .unwrap_or(false);
    let avx2 = has_avx2();
    let results: Vec<(Vec<f64>, Vec<usize>)> = if z_batch.len() > 16 {
        z_batch
            .par_iter()
            .enumerate()
            .map(|(i, z)| {
                let s_idx = if use_self_idx { Some(i) } else { None };
                let zt = z_ts.map(|s| s[i]);
                knn_single(z, t, k, s_idx, zt, t_ts, ts_sorted, avx2)
            })
            .collect()
    } else {
        z_batch
            .iter()
            .enumerate()
            .map(|(i, z)| {
                let s_idx = if use_self_idx { Some(i) } else { None };
                let zt = z_ts.map(|s| s[i]);
                knn_single(z, t, k, s_idx, zt, t_ts, ts_sorted, avx2)
            })
            .collect()
    };

    let mut dists = Vec::with_capacity(results.len());
    let mut idxs = Vec::with_capacity(results.len());
    for (d, ix) in results {
        dists.push(d);
        idxs.push(ix);
    }
    (dists, idxs)
}

/// Compute `(mean_excess_dist, mean_excess_per_feature)` for one query row `z` and its `k` neighbours.
fn directional_excess(
    z: &[f32; NUM_FEATURES],
    t: &[[f32; NUM_FEATURES]],
    nbr_idx: &[usize],
) -> (f64, [f64; NUM_FEATURES]) {
    if nbr_idx.is_empty() {
        return (0.0, [0.0; NUM_FEATURES]);
    }
    let mut sum_dist = 0.0f64;
    let mut mean_ex = [0.0f64; NUM_FEATURES];
    let k_f = nbr_idx.len() as f64;

    for &idx in nbr_idx {
        let tn = &t[idx];
        let mut sq = 0.0f64;
        for j in 0..NUM_FEATURES {
            let ex = ((z[j] - tn[j]).max(0.0) * COUNTS[j]) as f64;
            sq += ex * ex;
            mean_ex[j] += ex / k_f;
        }
        sum_dist += sq.sqrt();
    }
    (sum_dist / k_f, mean_ex)
}

fn pct_str(share: f64) -> String {
    let p = share * 100.0;
    if p >= 99.9 || p <= 0.1 {
        format!("{:.2}%", p)
    } else {
        format!("{:.1}%", p)
    }
}

/// Linear-interpolation quantile over a sorted slice (`q` in `[0.0, 1.0]`).
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * ((sorted.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Trained LOF-style kNN anomaly model.
#[derive(Debug, Clone)]
pub struct AnomalyModel {
    pub train: Vec<[f32; NUM_FEATURES]>,
    pub train_raw: Vec<[f32; NUM_FEATURES]>,
    pub local: Vec<f32>,
    pub threshold: f64,
    pub severe_threshold: f64,
    pub calib_scores: Vec<f64>,
    pub feature_sorted: Vec<Vec<f32>>,
    pub raw_medians: [f64; NUM_FEATURES],
    pub n_windows: usize,
    pub hours_of_data: f64,
    pub k: usize,
    pub trained_at: f64,
}

#[derive(Serialize, Deserialize)]
struct ModelMeta {
    format: u32,
    features: Vec<String>,
    units: Vec<f64>,
    threshold: f64,
    severe_threshold: f64,
    k: usize,
    n_windows: usize,
    hours_of_data: f64,
    trained_at: f64,
}

impl AnomalyModel {
    /// Evaluate a single ML-space feature vector `x` (output of `to_matrix_row`),
    /// returning `(score, is_material)`.
    pub fn evaluate_raw_row(&self, x: &[f64; NUM_FEATURES]) -> (f64, bool) {
        let z = scale_to_units(x);
        let (_, idx) = knn_single(&z, &self.train, self.k, None, None, None, false, has_avx2());
        let (ex_dist, mean_ex) = directional_excess(&z, &self.train, &idx);
        let nbr_local_mean = if idx.is_empty() {
            LOCAL_FLOOR
        } else {
            let s: f64 = idx.iter().map(|&i| self.local[i] as f64).sum();
            (s / (idx.len() as f64)).max(LOCAL_FLOOR)
        };
        let score = ex_dist / nbr_local_mean;
        let max_ex = mean_ex.iter().copied().fold(0.0f64, f64::max);
        (score, max_ex >= MIN_ALERT_UNITS)
    }

    /// Threshold for a given alert quantile derived from `calib_scores`.
    pub fn threshold_for_quantile(&self, alert_quantile: f64) -> f64 {
        quantile_sorted(&self.calib_scores, 1.0 - alert_quantile)
    }

    /// Evaluate a single ML-space feature vector `x` (output of `to_matrix_row`),
    /// returning `(score, is_anomalous)`.
    pub fn evaluate_row(&self, x: &[f64; NUM_FEATURES]) -> (f64, bool) {
        let (score, material) = self.evaluate_raw_row(x);
        (score, score > self.threshold && material)
    }

    /// Evaluate a batch of ML-space rows, returning `(scores, anomalous_flags)`.
    pub fn evaluate_batch(&self, x_batch: &[[f64; NUM_FEATURES]]) -> (Vec<f64>, Vec<bool>) {
        let mut scores = Vec::with_capacity(x_batch.len());
        let mut flags = Vec::with_capacity(x_batch.len());
        for x in x_batch {
            let (s, a) = self.evaluate_row(x);
            scores.push(s);
            flags.push(a);
        }
        (scores, flags)
    }

    pub fn score_row(&self, x: &[f64; NUM_FEATURES]) -> f64 {
        self.evaluate_row(x).0
    }

    /// Fraction of historical windows that scored at least this high.
    pub fn rarity(&self, score: f64) -> f64 {
        let n = self.calib_scores.len();
        if n == 0 {
            return 0.0;
        }
        let pos = self.calib_scores.partition_point(|&v| v < score);
        ((n - pos) as f64) / (n as f64)
    }

    /// Explain which features are furthest above the most similar moments in history.
    pub fn explain(&self, x: &[f64; NUM_FEATURES], raw: &MetricMap, top: usize) -> Vec<Value> {
        let z = scale_to_units(x);
        let (_, idx) = knn_single(&z, &self.train, self.k, None, None, None, false, has_avx2());
        let (_, diff) = directional_excess(&z, &self.train, &idx);

        let max_diff = diff.iter().copied().fold(0.0f64, f64::max);
        let cutoff = MIN_EXPLAIN_UNITS.max(0.25 * max_diff);

        let mut order: Vec<usize> = (0..NUM_FEATURES).collect();
        order.sort_by(|&a, &b| {
            diff[b]
                .partial_cmp(&diff[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n = self.n_windows.max(1);
        let mut out = Vec::new();
        for j in order.into_iter().take(top) {
            if diff[j] < cutoff {
                break;
            }
            let f = &ML_FEATURES[j];
            let value = get_metric(raw, f.column);
            let nbr_vals: Vec<f64> = idx
                .iter()
                .map(|&i| self.train_raw[i][j] as f64)
                .filter(|v| v.is_finite())
                .collect();
            let similar = nanmedian(&nbr_vals);

            let col = &self.feature_sorted[j];
            let v = x[j] as f32;
            let where_str = if f.unit == "state" {
                let match_cnt = col.iter().filter(|&&c| (c - v).abs() < 0.25).count();
                format!("only {} of history", pct_str((match_cnt as f64) / (n as f64)))
            } else {
                let pos = col.partition_point(|&c| c < v);
                format!("higher than {} of history", pct_str((pos as f64) / (n as f64)))
            };

            let val_json = if value.is_finite() {
                json!((value * 1000.0).round() / 1000.0)
            } else {
                Value::Null
            };
            let sim_json = if similar.is_finite() {
                json!((similar * 1000.0).round() / 1000.0)
            } else {
                Value::Null
            };

            out.push(json!({
                "feature": f.column,
                "label": f.label,
                "value": val_json,
                "similar_moments": sim_json,
                "units_off": (diff[j] * 10.0).round() / 10.0,
                "text": format!("{} {} (similar moments: {}; {})",
                    f.label, fmt_feature(f, value), fmt_feature(f, similar), where_str),
            }));
        }
        out
    }

    /// Persist model atomically to `path` (`data/model.bin`).
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let meta = ModelMeta {
            format: MODEL_FORMAT,
            features: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
            units: UNITS.to_vec(),
            threshold: self.threshold,
            severe_threshold: self.severe_threshold,
            k: self.k,
            n_windows: self.n_windows,
            hours_of_data: self.hours_of_data,
            trained_at: self.trained_at,
        };
        let meta_bytes = serde_json::to_vec(&meta).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("tmp.bin");
        {
            let mut f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
            f.write_all(b"RSM2").map_err(|e| e.to_string())?;
            let meta_len = meta_bytes.len() as u32;
            f.write_all(&meta_len.to_le_bytes())
                .map_err(|e| e.to_string())?;
            f.write_all(&meta_bytes).map_err(|e| e.to_string())?;

            // Write train, train_raw, local, calib_scores, feature_sorted, raw_medians
            for row in &self.train {
                for &v in row {
                    f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
                }
            }
            for row in &self.train_raw {
                for &v in row {
                    f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
                }
            }
            for &v in &self.local {
                f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
            }
            for &v in &self.calib_scores {
                f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
            }
            for col in &self.feature_sorted {
                for &v in col {
                    f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
                }
            }
            for &v in &self.raw_medians {
                f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
            }
            f.sync_all().map_err(|e| e.to_string())?;
        }
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Load a previously saved model from `path`.
    pub fn load(path: &Path) -> Result<Self, String> {
        let mut buf = Vec::new();
        std::fs::File::open(path)
            .map_err(|e| e.to_string())?
            .read_to_end(&mut buf)
            .map_err(|e| e.to_string())?;
        if buf.len() < 8 || &buf[0..4] != b"RSM2" {
            return Err("invalid model header".to_string());
        }
        let meta_len = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
        if buf.len() < 8 + meta_len {
            return Err("truncated model metadata".to_string());
        }
        let meta: ModelMeta =
            serde_json::from_slice(&buf[8..8 + meta_len]).map_err(|e| e.to_string())?;
        if meta.format != MODEL_FORMAT
            || meta.features != FEATURE_NAMES
            || meta.units != UNITS
        {
            return Err("saved by an incompatible detector version".to_string());
        }

        let n = meta.n_windows;
        let expected_floats = n * NUM_FEATURES * 2 + n + n * NUM_FEATURES;
        let expected_bytes = 8 + meta_len + expected_floats * 4 + (n + NUM_FEATURES) * 8;
        if buf.len() != expected_bytes {
            return Err("model binary payload size mismatch".to_string());
        }

        let mut offset = 8 + meta_len;
        let read_f32 = |off: &mut usize| -> f32 {
            let v = f32::from_le_bytes([buf[*off], buf[*off + 1], buf[*off + 2], buf[*off + 3]]);
            *off += 4;
            v
        };
        let read_f64 = |off: &mut usize| -> f64 {
            let v = f64::from_le_bytes([
                buf[*off],
                buf[*off + 1],
                buf[*off + 2],
                buf[*off + 3],
                buf[*off + 4],
                buf[*off + 5],
                buf[*off + 6],
                buf[*off + 7],
            ]);
            *off += 8;
            v
        };

        let mut train = vec![[0.0f32; NUM_FEATURES]; n];
        for row in &mut train {
            for cell in row.iter_mut() {
                *cell = read_f32(&mut offset);
            }
        }
        let mut train_raw = vec![[0.0f32; NUM_FEATURES]; n];
        for row in &mut train_raw {
            for cell in row.iter_mut() {
                *cell = read_f32(&mut offset);
            }
        }
        let mut local = vec![0.0f32; n];
        for cell in &mut local {
            *cell = read_f32(&mut offset);
        }
        let mut calib_scores = vec![0.0f64; n];
        for cell in &mut calib_scores {
            *cell = read_f64(&mut offset);
        }
        let mut feature_sorted = vec![vec![0.0f32; n]; NUM_FEATURES];
        for col in &mut feature_sorted {
            for cell in col.iter_mut() {
                *cell = read_f32(&mut offset);
            }
        }
        let mut raw_medians = [f64::NAN; NUM_FEATURES];
        for cell in &mut raw_medians {
            *cell = read_f64(&mut offset);
        }

        Ok(Self {
            train,
            train_raw,
            local,
            threshold: meta.threshold,
            severe_threshold: meta.severe_threshold,
            calib_scores,
            feature_sorted,
            raw_medians,
            n_windows: n,
            hours_of_data: meta.hours_of_data,
            k: meta.k,
            trained_at: meta.trained_at,
        })
    }
}

/// Fit an `AnomalyModel` on `raw` historical windows (`(n_windows, 16)` display values).
pub fn fit_model(
    raw: &[[f64; NUM_FEATURES]],
    ts: Option<&[f64]>,
    alert_quantile: f64,
    window_sec: f64,
    k: usize,
) -> AnomalyModel {
    let n = raw.len();
    let x_mat = to_matrix(raw);
    let t_mat: Vec<[f32; NUM_FEATURES]> = x_mat.iter().map(scale_to_units).collect();
    let train_raw: Vec<[f32; NUM_FEATURES]> = raw
        .iter()
        .map(|r| {
            let mut row = [0.0f32; NUM_FEATURES];
            for j in 0..NUM_FEATURES {
                row[j] = r[j] as f32;
            }
            row
        })
        .collect();

    let (dists, idxs) = knn(&t_mat, &t_mat, k, true, ts, ts);
    let local: Vec<f32> = dists
        .iter()
        .map(|d| {
            if d.is_empty() {
                LOCAL_FLOOR as f32
            } else {
                (d.iter().sum::<f64>() / (d.len() as f64)) as f32
            }
        })
        .collect();

    let mut calib: Vec<f64> = if n > 64 {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let (ex_dist, _) = directional_excess(&t_mat[i], &t_mat, &idxs[i]);
                let nbr_local = if idxs[i].is_empty() {
                    LOCAL_FLOOR
                } else {
                    let s: f64 = idxs[i].iter().map(|&m| local[m] as f64).sum();
                    (s / (idxs[i].len() as f64)).max(LOCAL_FLOOR)
                };
                ex_dist / nbr_local
            })
            .collect()
    } else {
        (0..n)
            .map(|i| {
                let (ex_dist, _) = directional_excess(&t_mat[i], &t_mat, &idxs[i]);
                let nbr_local = if idxs[i].is_empty() {
                    LOCAL_FLOOR
                } else {
                    let s: f64 = idxs[i].iter().map(|&m| local[m] as f64).sum();
                    (s / (idxs[i].len() as f64)).max(LOCAL_FLOOR)
                };
                ex_dist / nbr_local
            })
            .collect()
    };
    calib.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let threshold = quantile_sorted(&calib, 1.0 - alert_quantile);
    let severe_threshold = quantile_sorted(&calib, 1.0 - alert_quantile / 10.0);

    let mut feature_sorted = Vec::with_capacity(NUM_FEATURES);
    let mut raw_medians = [f64::NAN; NUM_FEATURES];
    for (j, raw_med) in raw_medians.iter_mut().enumerate() {
        let mut col: Vec<f32> = x_mat.iter().map(|r| r[j] as f32).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        feature_sorted.push(col);

        let raw_col: Vec<f64> = raw
            .iter()
            .map(|r| r[j])
            .filter(|v| v.is_finite())
            .collect();
        *raw_med = nanmedian(&raw_col);
    }

    AnomalyModel {
        train: t_mat,
        train_raw,
        local,
        threshold,
        severe_threshold,
        calib_scores: calib,
        feature_sorted,
        raw_medians,
        n_windows: n,
        hours_of_data: (n as f64) * window_sec / 3600.0,
        k: k.min(n.saturating_sub(1)).max(1),
        trained_at: crate::sampler::unix_now(),
    }
}

/// Fits an `AnomalyModel` on a background thread so 1 Hz sampling never stalls.
pub struct Trainer {
    pub model_path: PathBuf,
    handle: Option<JoinHandle<()>>,
    result: Arc<Mutex<Option<Result<AnomalyModel, String>>>>,
}

impl Trainer {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            model_path: data_dir.join("model.bin"),
            handle: None,
            result: Arc::new(Mutex::new(None)),
        }
    }

    pub fn busy(&self) -> bool {
        self.handle
            .as_ref()
            .map(|h| !h.is_finished())
            .unwrap_or(false)
    }

    pub fn start<F>(&mut self, load: F, alert_quantile: f64, window_sec: f64) -> bool
    where
        F: FnOnce() -> Option<(Vec<f64>, Vec<[f64; NUM_FEATURES]>)> + Send + 'static,
    {
        if self.busy() {
            return false;
        }
        let path = self.model_path.clone();
        let res_slot = Arc::clone(&self.result);
        self.handle = Some(std::thread::spawn(move || {
            let _ = crate::winapi::set_current_thread_eco_qos();
            let t0 = Instant::now();
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let data = load()?;
                let (ts, raw) = data;
                let model = fit_model(&raw, Some(&ts), alert_quantile, window_sec, K_NEIGHBOURS);
                let _ = model.save(&path);
                let _elapsed = t0.elapsed();
                Some(model)
            }));
            match outcome {
                Ok(Some(model)) => {
                    if let Ok(mut g) = res_slot.lock() {
                        *g = Some(Ok(model));
                    }
                }
                Ok(None) => {}
                Err(_) => {
                    if let Ok(mut g) = res_slot.lock() {
                        *g = Some(Err("background trainer panicked".to_string()));
                    }
                }
            }
        }));
        true
    }

    pub fn poll(&mut self) -> Result<Option<AnomalyModel>, String> {
        let mut g = self.result.lock().map_err(|e| e.to_string())?;
        match g.take() {
            Some(Ok(m)) => Ok(Some(m)),
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }

    pub fn load_saved(&self, max_age_sec: f64) -> Option<AnomalyModel> {
        if !self.model_path.exists() {
            return None;
        }
        let model = AnomalyModel::load(&self.model_path).ok()?;
        if crate::sampler::unix_now() - model.trained_at <= max_age_sec {
            Some(model)
        } else {
            None
        }
    }

    pub fn join(&mut self) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
