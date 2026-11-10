//! Feature definitions, 10-second window aggregation, and human formatting.
//!
//! Raw 1 Hz samples are noisy (a single frame hitch or Defender file open spikes
//! one tick). Every `window_sec` seconds (default 10 s), samples are collapsed
//! into one summary row with 16 features for the kNN anomaly detector plus
//! auxiliary columns (`cpu_p95`, `disk_free_gb`).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Metadata for one of the 16 ML features.
#[derive(Debug, Clone, Copy)]
pub struct Feature {
    pub column: &'static str,
    pub source: &'static str,
    pub agg: &'static str,
    pub label: &'static str,
    pub unit: &'static str,
    pub log_scale: bool,
}

pub const NUM_FEATURES: usize = 16;

/// Order matters: saved models store feature vectors in this exact column order.
pub const ML_FEATURES: [Feature; NUM_FEATURES] = [
    Feature {
        column: "cpu_pct",
        source: "cpu_pct",
        agg: "mean",
        label: "CPU",
        unit: "%",
        log_scale: false,
    },
    Feature {
        column: "core_max_pct",
        source: "core_max_pct",
        agg: "max",
        label: "busiest core",
        unit: "%",
        log_scale: false,
    },
    Feature {
        column: "ram_pct",
        source: "ram_pct",
        agg: "mean",
        label: "RAM",
        unit: "%",
        log_scale: false,
    },
    Feature {
        column: "commit_pct",
        source: "commit_pct",
        agg: "mean",
        label: "commit",
        unit: "%",
        log_scale: false,
    },
    Feature {
        column: "disk_read_bps",
        source: "disk_read_bps",
        agg: "mean",
        label: "disk read",
        unit: "B/s",
        log_scale: true,
    },
    Feature {
        column: "disk_write_bps",
        source: "disk_write_bps",
        agg: "mean",
        label: "disk write",
        unit: "B/s",
        log_scale: true,
    },
    Feature {
        column: "net_sent_bps",
        source: "net_sent_bps",
        agg: "mean",
        label: "net upload",
        unit: "B/s",
        log_scale: true,
    },
    Feature {
        column: "net_recv_bps",
        source: "net_recv_bps",
        agg: "mean",
        label: "net download",
        unit: "B/s",
        log_scale: true,
    },
    Feature {
        column: "gpu_util_pct",
        source: "gpu_util_pct",
        agg: "mean",
        label: "GPU load",
        unit: "%",
        log_scale: false,
    },
    Feature {
        column: "gpu_power_w",
        source: "gpu_power_w",
        agg: "mean",
        label: "GPU power",
        unit: "W",
        log_scale: false,
    },
    Feature {
        column: "gpu_temp_c",
        source: "gpu_temp_c",
        agg: "max",
        label: "GPU temp",
        unit: "°C",
        log_scale: false,
    },
    Feature {
        column: "gpu_vram_pct",
        source: "gpu_vram_pct",
        agg: "mean",
        label: "VRAM",
        unit: "%",
        log_scale: false,
    },
    Feature {
        column: "gpu_clock_mhz",
        source: "gpu_clock_mhz",
        agg: "mean",
        label: "GPU clock",
        unit: "MHz",
        log_scale: false,
    },
    Feature {
        column: "gpu_awake",
        source: "gpu_awake",
        agg: "max",
        label: "GPU state",
        unit: "state",
        log_scale: false,
    },
    Feature {
        column: "on_ac",
        source: "on_ac",
        agg: "last",
        label: "power source",
        unit: "state",
        log_scale: false,
    },
    Feature {
        column: "user_active",
        source: "user_active",
        agg: "max",
        label: "user activity",
        unit: "state",
        log_scale: false,
    },
];

pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "cpu_pct",
    "core_max_pct",
    "ram_pct",
    "commit_pct",
    "disk_read_bps",
    "disk_write_bps",
    "net_sent_bps",
    "net_recv_bps",
    "gpu_util_pct",
    "gpu_power_w",
    "gpu_temp_c",
    "gpu_vram_pct",
    "gpu_clock_mhz",
    "gpu_awake",
    "on_ac",
    "user_active",
];

/// Rates below 64 KB/s are background trickle; measuring from here stops
/// 0 -> 20 KB/s looking like a 4-decade jump on a log scale.
pub const RATE_FLOOR: f64 = 65_536.0;

pub const IDX_GPU_TEMP: usize = 10;
pub const IDX_GPU_AWAKE: usize = 13;

/// Convert a single 16-feature raw display vector into ML space.
///
/// - NaNs (e.g. GPU asleep or absent) become `0.0`; `gpu_awake` disambiguates
///   "0 because asleep" from "awake at 0 %".
/// - Byte/s rates use `log10(1 + rate / 64 KB/s)` so 0 B/s and 64 MB/s live on
///   a comparable scale.
/// - `gpu_temp_c` is multiplied by `gpu_awake` so an awake GPU at 40 °C does not
///   sit 40 °C away from an asleep GPU (`0`).
pub fn to_matrix_row(raw: &[f64; NUM_FEATURES]) -> [f64; NUM_FEATURES] {
    let mut x = [0.0f64; NUM_FEATURES];
    for j in 0..NUM_FEATURES {
        let v = raw[j];
        let clean = if v.is_finite() { v } else { 0.0 };
        if ML_FEATURES[j].log_scale {
            x[j] = (1.0 + clean.max(0.0) / RATE_FLOOR).log10();
        } else {
            x[j] = clean;
        }
    }
    x[IDX_GPU_TEMP] *= x[IDX_GPU_AWAKE];
    x
}

/// Convert a batch of raw feature rows into ML space.
pub fn to_matrix(raw: &[[f64; NUM_FEATURES]]) -> Vec<[f64; NUM_FEATURES]> {
    raw.iter().map(to_matrix_row).collect()
}

/// Extract a `[f64; 16]` feature array from a dynamic key-value map.
pub fn map_to_feature_array(map: &BTreeMap<String, f64>) -> [f64; NUM_FEATURES] {
    let mut out = [f64::NAN; NUM_FEATURES];
    for (i, f) in ML_FEATURES.iter().enumerate() {
        if let Some(&v) = map.get(f.column) {
            out[i] = v;
        }
    }
    out
}

/// A 1 Hz system sample or 10 s window represented as a key-value map of metric names to values.
pub type MetricMap = BTreeMap<String, f64>;

/// Helper to get a metric or `f64::NAN` if absent.
#[inline]
pub fn get_metric(m: &MetricMap, key: &str) -> f64 {
    m.get(key).copied().unwrap_or(f64::NAN)
}

/// Helper to get a metric or a default if absent or NaN.
#[inline]
pub fn get_metric_or(m: &MetricMap, key: &str, default: f64) -> f64 {
    match m.get(key) {
        Some(&v) if v.is_finite() => v,
        _ => default,
    }
}

/// Linear-interpolation percentile matching `numpy.percentile(vals, q)`.
pub fn percentile(vals: &[f64], q: f64) -> f64 {
    let mut clean: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
    if clean.is_empty() {
        return f64::NAN;
    }
    clean.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if clean.len() == 1 {
        return clean[0];
    }
    let pos = (q / 100.0).clamp(0.0, 1.0) * ((clean.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    clean[lo] * (1.0 - frac) + clean[hi] * frac
}

/// Median of finite values in `vals`.
pub fn nanmedian(vals: &[f64]) -> f64 {
    percentile(vals, 50.0)
}

/// Collects 1 s samples and emits a completed window dict every `window_sec` seconds.
#[derive(Debug, Clone)]
pub struct WindowAggregator {
    pub window_sec: i64,
    pub min_samples: usize,
    bucket: Option<i64>,
    samples: Vec<MetricMap>,
}

/// Completed window returned by `WindowAggregator::add`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedWindow {
    pub row: MetricMap,
    pub raw_features: [f64; NUM_FEATURES],
    pub ml_features: [f64; NUM_FEATURES],
}

impl WindowAggregator {
    pub fn new(window_sec: i64) -> Self {
        let min_samples = (window_sec / 2).max(1) as usize;
        Self {
            window_sec,
            min_samples,
            bucket: None,
            samples: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.bucket = None;
        self.samples.clear();
    }

    pub fn add(&mut self, sample: MetricMap) -> Option<CompletedWindow> {
        let ts = get_metric(&sample, "ts");
        let w = self.window_sec as f64;
        let b = ((ts / w).floor() as i64) * self.window_sec;
        if self.bucket.is_none() {
            self.bucket = Some(b);
        }
        let mut finished = None;
        if self.bucket != Some(b) {
            if self.samples.len() >= self.min_samples {
                finished = Some(self.flush(self.bucket.unwrap()));
            }
            self.bucket = Some(b);
            self.samples.clear();
        }
        self.samples.push(sample);
        finished
    }

    fn flush(&self, bucket_ts: i64) -> CompletedWindow {
        let mut row = MetricMap::new();
        row.insert("ts".to_string(), bucket_ts as f64);

        let mut raw_vec = [f64::NAN; NUM_FEATURES];
        for (i, f) in ML_FEATURES.iter().enumerate() {
            let vals: Vec<f64> = self
                .samples
                .iter()
                .map(|s| get_metric(s, f.source))
                .filter(|v| v.is_finite())
                .collect();
            let v = if vals.is_empty() {
                f64::NAN
            } else if f.agg == "mean" {
                vals.iter().sum::<f64>() / (vals.len() as f64)
            } else if f.agg == "max" {
                vals.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            } else {
                *vals.last().unwrap()
            };
            row.insert(f.column.to_string(), v);
            raw_vec[i] = v;
        }

        // Auxiliary columns used by rules & status
        let cpu_vals: Vec<f64> = self
            .samples
            .iter()
            .map(|s| get_metric(s, "cpu_pct"))
            .filter(|v| v.is_finite())
            .collect();
        row.insert("cpu_p95".to_string(), percentile(&cpu_vals, 95.0));

        let disk_free = self
            .samples
            .iter()
            .rev()
            .map(|s| get_metric(s, "disk_free_gb"))
            .find(|v| v.is_finite())
            .unwrap_or(f64::NAN);
        row.insert("disk_free_gb".to_string(), disk_free);

        let ml_vec = to_matrix_row(&raw_vec);
        CompletedWindow {
            row,
            raw_features: raw_vec,
            ml_features: ml_vec,
        }
    }
}

/// Format byte counts or byte/s rates (`fmt_bytes(1048576.0, true)` -> `"1.0 MB/s"`).
pub fn fmt_bytes(n: f64, per_sec: bool) -> String {
    if !n.is_finite() {
        return "n/a".to_string();
    }
    let suffix = if per_sec { "/s" } else { "" };
    let mut val = n;
    for unit in ["B", "KB", "MB", "GB"] {
        if val.abs() < 1024.0 {
            if unit == "B" {
                return format!("{:.0} {}{}", val, unit, suffix);
            } else {
                return format!("{:.1} {}{}", val, unit, suffix);
            }
        }
        val /= 1024.0;
    }
    format!("{:.1} TB{}", val, suffix)
}

/// Format a feature value for human explanations (`detector::explain`).
pub fn fmt_feature(f: &Feature, v: f64) -> String {
    if !v.is_finite() {
        return if f.column.starts_with("gpu_") {
            "GPU asleep".to_string()
        } else {
            "n/a".to_string()
        };
    }
    match f.unit {
        "B/s" => fmt_bytes(v, true),
        "%" => format!("{:.0}%", v),
        "°C" => format!("{:.0}°C", v),
        "W" => format!("{:.0} W", v),
        "MHz" => format!("{:.0} MHz", v),
        "state" => match f.column {
            "gpu_awake" => {
                if v >= 0.5 {
                    "awake".to_string()
                } else {
                    "asleep".to_string()
                }
            }
            "on_ac" => {
                if v >= 0.5 {
                    "AC".to_string()
                } else {
                    "battery".to_string()
                }
            }
            "user_active" => {
                if v >= 0.5 {
                    "active".to_string()
                } else {
                    "idle".to_string()
                }
            }
            _ => format!("{:.0}", v),
        },
        _ => format!("{:.1}", v),
    }
}
