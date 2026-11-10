//! Offline historical replay (`replay`) and sensitivity grid evaluation (`tune_grid`).

use crate::config::MonitorConfig;
use crate::detector::{fit_model, AnomalyModel, K_NEIGHBOURS};
use crate::events::{merge_details, severity_rank, Event, EventSink};
use crate::features::{get_metric_or, map_to_feature_array, to_matrix, MetricMap, NUM_FEATURES};
use crate::rules::{AnomalyEpisodes, MonitorContext};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde_json::{json, Value};

#[derive(Default)]
pub struct RecordingEvents {
    pub events: Vec<Event>,
}

impl EventSink for RecordingEvents {
    fn open(
        &mut self,
        kind: &str,
        severity: &str,
        title: &str,
        ts: f64,
        details: Value,
        _notify: bool,
    ) -> Event {
        let id = (self.events.len() + 1) as i64;
        let ev = Event {
            id,
            kind: kind.to_string(),
            severity: severity.to_string(),
            title: title.to_string(),
            ts_start: ts,
            ts_end: ts,
            details,
            is_open: true,
        };
        self.events.push(ev.clone());
        ev
    }

    fn update(
        &mut self,
        ev: &mut Event,
        ts: f64,
        title: Option<&str>,
        severity: Option<&str>,
        details: Option<Value>,
        _notify_on_escalate: bool,
    ) {
        ev.ts_end = ev.ts_end.max(ts);
        if let Some(t) = title {
            ev.title = t.to_string();
        }
        if let Some(sev) = severity {
            if severity_rank(sev) > severity_rank(&ev.severity) {
                ev.severity = sev.to_string();
            }
        }
        if let Some(d) = details {
            merge_details(&mut ev.details, d);
        }
        if let Some(slot) = self.events.iter_mut().find(|e| e.id == ev.id) {
            *slot = ev.clone();
        }
    }

    fn reopen(&mut self, ev: &mut Event, ts: f64) {
        ev.is_open = true;
        ev.ts_end = ev.ts_end.max(ts);
        let rec = ev
            .details
            .get("recurrences")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            + 1;
        merge_details(&mut ev.details, json!({ "recurrences": rec }));
        if let Some(slot) = self.events.iter_mut().find(|e| e.id == ev.id) {
            *slot = ev.clone();
        }
    }

    fn close(
        &mut self,
        ev: &mut Event,
        ts: f64,
        title: Option<&str>,
        severity: Option<&str>,
        details: Option<Value>,
    ) {
        self.update(ev, ts, title, severity, details, false);
        ev.is_open = false;
        if let Some(slot) = self.events.iter_mut().find(|e| e.id == ev.id) {
            *slot = ev.clone();
        }
    }

    fn point(&mut self, kind: &str, severity: &str, title: &str, ts: f64, details: Value) -> Event {
        let mut ev = self.open(kind, severity, title, ts, details, false);
        ev.is_open = false;
        if let Some(slot) = self.events.iter_mut().find(|e| e.id == ev.id) {
            *slot = ev.clone();
        }
        ev
    }
}

pub struct ReplayMonitor {
    pub cfg: MonitorConfig,
    pub model: Option<AnomalyModel>,
    pub events: RecordingEvents,
}

impl ReplayMonitor {
    pub fn new(cfg: MonitorConfig) -> Self {
        Self {
            cfg,
            model: None,
            events: RecordingEvents::default(),
        }
    }
}

impl MonitorContext for ReplayMonitor {
    fn cfg(&self) -> &MonitorConfig {
        &self.cfg
    }

    fn model(&self) -> Option<&AnomalyModel> {
        self.model.as_ref()
    }

    fn activity(&self, _since: f64) -> Value {
        json!({
            "top_cpu": [],
            "top_io": [],
            "top_memory": [],
            "gpu_clients": [],
            "foreground": null,
            "user_idle": false,
        })
    }

    fn cpu_seconds(&self, _pid: u32) -> Option<f64> {
        None
    }

    fn name_for_pid(&self, _pid: u32) -> Option<String> {
        None
    }

    fn foreground_pid(&self) -> Option<u32> {
        None
    }

    fn with_events<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut dyn EventSink) -> R,
    {
        f(&mut self.events)
    }
}

#[derive(Debug, Clone)]
pub struct ReplayResult {
    pub events: Vec<Event>,
    pub days: f64,
    pub scored_windows: usize,
    pub anomalous_windows: usize,
}

impl ReplayResult {
    pub fn per_day(&self) -> f64 {
        if self.days > 0.0 {
            (self.events.len() as f64) / self.days
        } else {
            0.0
        }
    }
}

/// Replay time-ordered windows through the exact production `AnomalyEpisodes` pipeline.
pub fn replay(windows: &[MetricMap], cfg: &MonitorConfig, seed: u64) -> ReplayResult {
    if windows.is_empty() {
        return ReplayResult {
            events: Vec::new(),
            days: 0.0,
            scored_windows: 0,
            anomalous_windows: 0,
        };
    }

    let mut mon = ReplayMonitor::new(cfg.clone());
    let mut episodes = AnomalyEpisodes::new();
    let ts: Vec<f64> = windows
        .iter()
        .map(|w| get_metric_or(w, "ts", 0.0))
        .collect();
    let raw: Vec<[f64; NUM_FEATURES]> = windows.iter().map(map_to_feature_array).collect();
    let x_mat = to_matrix(&raw);

    let win_sec = cfg.window_sec as f64;
    let min_windows = ((cfg.min_training_hours * 3600.0) / win_sec).max(1.0) as usize;
    let retrain_every = cfg.retrain_interval_min * 60.0;
    let mut next_train = ts[0];
    let mut grace_until = ts[0] + cfg.grace_period_sec;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut scored = 0usize;
    let mut flagged = 0usize;

    for (i, w_orig) in windows.iter().enumerate() {
        let now = ts[i] + win_sec;
        if i > 0 && ts[i] - ts[i - 1] > 5.0 * win_sec {
            grace_until = now + cfg.grace_period_sec;
            episodes.reset();
        }
        if now >= next_train {
            let cutoff = now - (cfg.training_days as f64) * 86400.0;
            let lo = ts[..i].partition_point(|&t| t < cutoff);
            let hist_len = i.saturating_sub(lo);
            if hist_len >= min_windows {
                let mut indices: Vec<usize> = (lo..i).collect();
                if indices.len() > cfg.max_training_windows {
                    indices.shuffle(&mut rng);
                    indices.truncate(cfg.max_training_windows);
                    indices.sort_unstable();
                }
                let sub_raw: Vec<[f64; NUM_FEATURES]> =
                    indices.iter().map(|&idx| raw[idx]).collect();
                let sub_ts: Vec<f64> = indices.iter().map(|&idx| ts[idx]).collect();
                mon.model = Some(fit_model(
                    &sub_raw,
                    Some(&sub_ts),
                    cfg.alert_quantile,
                    win_sec,
                    K_NEIGHBOURS,
                ));
                next_train = now + retrain_every;
            } else {
                next_train = now + 300.0;
            }
        }

        let mut w = w_orig.clone();
        let mut x_opt: Option<[f64; NUM_FEATURES]> = None;
        if let Some(ref model) = mon.model {
            if now >= grace_until {
                let x = x_mat[i];
                let (score, anom) = model.evaluate_row(&x);
                w.insert("score".to_string(), score);
                w.insert("anomalous".to_string(), if anom { 1.0 } else { 0.0 });
                scored += 1;
                if anom {
                    flagged += 1;
                }
                x_opt = Some(x);
            }
        }
        episodes.on_window(&mut mon, &w, x_opt.as_ref(), now);
    }

    let covered_days = (windows.len() as f64) * win_sec / 86400.0;
    ReplayResult {
        events: mon.events.events,
        days: covered_days,
        scored_windows: scored,
        anomalous_windows: flagged,
    }
}

struct SharedModelContext<'a> {
    cfg: &'a MonitorConfig,
    model: Option<&'a AnomalyModel>,
    events: &'a mut RecordingEvents,
}

impl<'a> MonitorContext for SharedModelContext<'a> {
    fn cfg(&self) -> &MonitorConfig {
        self.cfg
    }

    fn model(&self) -> Option<&AnomalyModel> {
        self.model
    }

    fn activity(&self, _since: f64) -> Value {
        json!({
            "top_cpu": [],
            "top_io": [],
            "top_memory": [],
            "gpu_clients": [],
            "foreground": null,
            "user_idle": false,
        })
    }

    fn cpu_seconds(&self, _pid: u32) -> Option<f64> {
        None
    }

    fn name_for_pid(&self, _pid: u32) -> Option<String> {
        None
    }

    fn foreground_pid(&self) -> Option<u32> {
        None
    }

    fn with_events<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut dyn EventSink) -> R,
    {
        f(self.events)
    }
}

/// Evaluate a grid of `(alert_quantile, min_consecutive_windows)` over recorded history
/// in a single training pass so `fit_model` and kNN scoring are shared across all grid points.
pub fn tune_grid(
    windows: &[MetricMap],
    cfg: &MonitorConfig,
    quantiles: &[f64],
    streaks: &[usize],
) -> Vec<Value> {
    if quantiles.is_empty() || streaks.is_empty() {
        return Vec::new();
    }
    if windows.is_empty() {
        let mut rows = Vec::with_capacity(quantiles.len() * streaks.len());
        for &q in quantiles {
            for &k in streaks {
                rows.push(json!({
                    "alert_quantile": q,
                    "min_consecutive_windows": k,
                    "events": 0,
                    "per_day": 0.0,
                    "days": 0.0,
                }));
            }
        }
        return rows;
    }

    let ts: Vec<f64> = windows
        .iter()
        .map(|w| get_metric_or(w, "ts", 0.0))
        .collect();
    let raw: Vec<[f64; NUM_FEATURES]> = windows.iter().map(map_to_feature_array).collect();
    let x_mat = to_matrix(&raw);

    let win_sec = cfg.window_sec as f64;
    let min_windows = ((cfg.min_training_hours * 3600.0) / win_sec).max(1.0) as usize;
    let retrain_every = cfg.retrain_interval_min * 60.0;
    let mut next_train = ts[0];
    let mut grace_until = ts[0] + cfg.grace_period_sec;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    let mut base_model: Option<AnomalyModel> = None;
    let mut q_thresholds: Vec<(f64, f64)> = vec![(0.0, 0.0); quantiles.len()];

    struct GridCell {
        q_idx: usize,
        cfg: MonitorConfig,
        episodes: AnomalyEpisodes,
        events: RecordingEvents,
    }

    let mut cells = Vec::with_capacity(quantiles.len() * streaks.len());
    for (q_idx, &q) in quantiles.iter().enumerate() {
        for &k in streaks {
            let mut sub_cfg = cfg.clone();
            sub_cfg.alert_quantile = q;
            sub_cfg.min_consecutive_windows = k;
            cells.push(GridCell {
                q_idx,
                cfg: sub_cfg,
                episodes: AnomalyEpisodes::new(),
                events: RecordingEvents::default(),
            });
        }
    }

    for (i, w_orig) in windows.iter().enumerate() {
        let now = ts[i] + win_sec;
        if i > 0 && ts[i] - ts[i - 1] > 5.0 * win_sec {
            grace_until = now + cfg.grace_period_sec;
            for cell in &mut cells {
                cell.episodes.reset();
            }
        }
        if now >= next_train {
            let cutoff = now - (cfg.training_days as f64) * 86400.0;
            let lo = ts[..i].partition_point(|&t| t < cutoff);
            let hist_len = i.saturating_sub(lo);
            if hist_len >= min_windows {
                let mut indices: Vec<usize> = (lo..i).collect();
                if indices.len() > cfg.max_training_windows {
                    indices.shuffle(&mut rng);
                    indices.truncate(cfg.max_training_windows);
                    indices.sort_unstable();
                }
                let sub_raw: Vec<[f64; NUM_FEATURES]> =
                    indices.iter().map(|&idx| raw[idx]).collect();
                let sub_ts: Vec<f64> = indices.iter().map(|&idx| ts[idx]).collect();
                let m = fit_model(
                    &sub_raw,
                    Some(&sub_ts),
                    cfg.alert_quantile,
                    win_sec,
                    K_NEIGHBOURS,
                );
                for (q_idx, &q) in quantiles.iter().enumerate() {
                    q_thresholds[q_idx] = (
                        m.threshold_for_quantile(q),
                        m.threshold_for_quantile(q / 10.0),
                    );
                }
                base_model = Some(m);
                next_train = now + retrain_every;
            } else {
                next_train = now + 300.0;
            }
        }

        if let Some(ref mut model) = base_model {
            if now >= grace_until {
                let x = x_mat[i];
                let (score, material) = model.evaluate_raw_row(&x);
                let mut w = w_orig.clone();
                w.insert("score".to_string(), score);

                // Step cells grouped by quantile so model.threshold / severe_threshold stay exact
                let mut cur_q = usize::MAX;
                for cell in &mut cells {
                    if cell.q_idx != cur_q {
                        cur_q = cell.q_idx;
                        let (t_warn, t_sev) = q_thresholds[cur_q];
                        model.threshold = t_warn;
                        model.severe_threshold = t_sev;
                        let anom = material && score > t_warn;
                        w.insert("anomalous".to_string(), if anom { 1.0 } else { 0.0 });
                    }
                    let mut ctx = SharedModelContext {
                        cfg: &cell.cfg,
                        model: Some(&*model),
                        events: &mut cell.events,
                    };
                    cell.episodes.on_window(&mut ctx, &w, Some(&x), now);
                }
            }
        }
    }

    let covered_days = (windows.len() as f64) * win_sec / 86400.0;
    cells
        .into_iter()
        .map(|cell| {
            let per_day = if covered_days > 0.0 {
                (cell.events.events.len() as f64) / covered_days
            } else {
                0.0
            };
            json!({
                "alert_quantile": cell.cfg.alert_quantile,
                "min_consecutive_windows": cell.cfg.min_consecutive_windows,
                "events": cell.events.events.len(),
                "per_day": (per_day * 100.0).round() / 100.0,
                "days": (covered_days * 100.0).round() / 100.0,
            })
        })
        .collect()
}
