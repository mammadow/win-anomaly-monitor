//! All 9 resource & anomaly detectors (`AnomalyEpisodes`, `MemoryPressure`,
//! `GpuThermal`, `GpuAwakeOnBattery`, `LowDiskSpace`, `UploadBurst`, `UiFreeze`,
//! `BackgroundCpu`, `Leaks`) with episode recurrence coalescing and OLS leak trend fitting.

use crate::config::MonitorConfig;
use crate::detector::AnomalyModel;
use crate::events::{fmt_duration, Event, EventSink};
use crate::features::{
    fmt_bytes, get_metric, get_metric_or, map_to_feature_array, nanmedian, MetricMap, NUM_FEATURES,
};
use crate::processes::{ProcKey, ProcStats};
use crate::responsiveness::Freeze;
use crate::storage::EventRecord;
use crate::winapi;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};

/// Interface provided by the monitor (or test/replay harness) to detectors.
pub trait MonitorContext {
    fn cfg(&self) -> &MonitorConfig;
    fn model(&self) -> Option<&AnomalyModel>;
    fn activity(&self, since: f64) -> Value;
    fn cpu_seconds(&self, pid: u32) -> Option<f64>;
    fn name_for_pid(&self, pid: u32) -> Option<String>;
    fn foreground_pid(&self) -> Option<u32>;
    fn with_events<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut dyn EventSink) -> R;
}

pub fn is_ignored(cfg: &MonitorConfig, proc_name: &str) -> bool {
    let lower = proc_name.to_ascii_lowercase();
    let base = lower.split(" [").next().unwrap_or(&lower);
    cfg.ignore_processes.iter().any(|ig| {
        let ig_l = ig.to_ascii_lowercase();
        ig_l == lower || ig_l == base
    })
}

fn fmt_top_procs(act: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(top_cpu) = act.get("top_cpu").and_then(|v| v.as_array()) {
        if let Some(c0) = top_cpu.first() {
            let cores = c0.get("cores").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let name = c0.get("name").and_then(|v| v.as_str()).unwrap_or("?");
            if cores >= 0.25 {
                parts.push(format!("top CPU: {} ({:.1} cores)", name, cores));
            }
        }
    }
    if let Some(top_io) = act.get("top_io").and_then(|v| v.as_array()) {
        if let Some(io0) = top_io.first() {
            let r = io0.get("read_mb_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let w = io0
                .get("write_mb_s")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let name = io0.get("name").and_then(|v| v.as_str()).unwrap_or("?");
            if r + w >= 1.0 {
                parts.push(format!("top I/O: {} ({:.1}R/{:.1}W MB/s)", name, r, w));
            }
        }
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!(" — {}", parts.join(", "))
    }
}

// ── 1. AnomalyEpisodes (ML kNN detector over 10 s windows) ────────────────

pub struct AnomalyEpisodes {
    streak: Vec<MetricMap>,
    gap: usize,
    pub open_ev: Option<Event>,
    peak_score: f64,
    peak_w: Option<MetricMap>,
    peak_x: Option<[f64; NUM_FEATURES]>,
    notified: bool,
    last_closed: Option<Event>,
}

impl Default for AnomalyEpisodes {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyEpisodes {
    pub const TOAST_SUSTAIN_SEC: f64 = 60.0;
    pub const REPEAT_MERGE_SEC: f64 = 600.0;

    pub fn new() -> Self {
        Self {
            streak: Vec::new(),
            gap: 0,
            open_ev: None,
            peak_score: 0.0,
            peak_w: None,
            peak_x: None,
            notified: false,
            last_closed: None,
        }
    }

    pub fn reset(&mut self) {
        self.streak.clear();
        self.gap = 0;
    }

    pub fn on_window(
        &mut self,
        ctx: &mut impl MonitorContext,
        w: &MetricMap,
        x_opt: Option<&[f64; NUM_FEATURES]>,
        now: f64,
    ) {
        let Some(x) = x_opt else {
            return;
        };
        if ctx.model().is_none() {
            return;
        }
        let anomalous = get_metric_or(w, "anomalous", 0.0) >= 0.5;
        let score = get_metric_or(w, "score", 0.0);
        let min_consec = ctx.cfg().min_consecutive_windows;
        let max_gap = ctx.cfg().episode_gap_windows;
        let win_sec = ctx.cfg().window_sec as f64;

        if self.open_ev.is_none() {
            if anomalous {
                self.streak.push(w.clone());
                if self.streak.len() >= min_consec {
                    self.start_episode(ctx, w, now);
                }
            } else {
                self.streak.clear();
            }
            return;
        }

        if anomalous {
            self.gap = 0;
            if score >= self.peak_score {
                self.peak_score = score;
                self.peak_w = Some(w.clone());
                self.peak_x = Some(*x);
            }
            self.refresh_episode(ctx, now, false);
        } else {
            self.gap += 1;
            if self.gap >= max_gap {
                let end_ts = get_metric_or(w, "ts", now) - ((self.gap - 1) as f64) * win_sec;
                self.refresh_episode(ctx, end_ts, true);
                self.last_closed = self.open_ev.take();
                self.streak.clear();
                self.gap = 0;
            }
        }
    }

    fn start_episode(&mut self, ctx: &mut impl MonitorContext, w: &MetricMap, now: f64) {
        let ts_start = self
            .streak
            .first()
            .map(|s| get_metric_or(s, "ts", now))
            .unwrap_or(now);
        let peak_w = self
            .streak
            .iter()
            .max_by(|a, b| {
                get_metric_or(a, "score", 0.0)
                    .partial_cmp(&get_metric_or(b, "score", 0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .unwrap_or_else(|| w.clone());
        self.peak_score = get_metric_or(&peak_w, "score", 0.0);
        let peak_arr = map_to_feature_array(&peak_w);
        self.peak_x = Some(crate::features::to_matrix_row(&peak_arr));
        self.peak_w = Some(peak_w);

        if let Some(ref mut prev) = self.last_closed {
            if ts_start - prev.ts_end <= Self::REPEAT_MERGE_SEC {
                ctx.with_events(|es| es.reopen(prev, now));
                self.open_ev = self.last_closed.take();
                self.refresh_episode(ctx, now, false);
                return;
            }
        }

        self.notified = false;
        let (sev, title, details) = self.describe(ctx, ts_start, now);
        let mut ev =
            ctx.with_events(|es| es.open("anomaly", &sev, &title, ts_start, details, false));
        ctx.with_events(|es| es.update(&mut ev, now, None, None, None, false));
        self.open_ev = Some(ev);
        self.maybe_notify(ctx, now);
    }

    fn refresh_episode(&mut self, ctx: &mut impl MonitorContext, ts: f64, closing: bool) {
        let Some(ref ev_snapshot) = self.open_ev else {
            return;
        };
        let ts_start = ev_snapshot.ts_start;
        let (sev, title, details) = self.describe(ctx, ts_start, ts);
        if let Some(ref mut ev) = self.open_ev {
            if closing {
                ctx.with_events(|es| es.close(ev, ts, Some(&title), Some(&sev), Some(details)));
            } else {
                ctx.with_events(|es| {
                    es.update(ev, ts, Some(&title), Some(&sev), Some(details), false)
                });
                self.maybe_notify(ctx, ts);
            }
        }
    }

    fn maybe_notify(&mut self, ctx: &mut impl MonitorContext, now: f64) {
        if self.notified {
            return;
        }
        let Some(ref mut ev) = self.open_ev else {
            return;
        };
        if now - ev.ts_start >= Self::TOAST_SUSTAIN_SEC || ev.severity == "warning" {
            ev.severity = "warning".to_string();
            ctx.with_events(|es| es.update(ev, now, None, Some("warning"), None, true));
            self.notified = true;
        }
    }

    fn describe(
        &self,
        ctx: &impl MonitorContext,
        ts_start: f64,
        ts_end: f64,
    ) -> (String, String, Value) {
        let mut reasons = Vec::new();
        let mut rarity = 0.0;
        let mut severe = false;
        if let (Some(model), Some(ref px), Some(pw)) =
            (ctx.model(), self.peak_x, self.peak_w.as_ref())
        {
            reasons = model.explain(px, pw, 3);
            rarity = model.rarity(self.peak_score);
            severe = self.peak_score >= model.severe_threshold;
        }
        let act = ctx.activity(ts_start);
        let dur = fmt_duration(ts_end - ts_start);
        let headline = if !reasons.is_empty() {
            reasons
                .iter()
                .take(2)
                .filter_map(|r| r.get("text").and_then(|v| v.as_str()))
                .collect::<Vec<_>>()
                .join("; ")
        } else {
            format!(
                "unusual metric combination (rarer than {:.2}% of history)",
                (1.0 - rarity) * 100.0
            )
        };
        let title = format!(
            "Unusual activity ({}): {}{}",
            dur,
            headline,
            fmt_top_procs(&act)
        );
        let sev = if severe || (ts_end - ts_start >= Self::TOAST_SUSTAIN_SEC) {
            "warning".to_string()
        } else {
            "info".to_string()
        };
        let details = json!({
            "peak_score": (self.peak_score * 100.0).round() / 100.0,
            "rarity_pct": ((1.0 - rarity) * 10000.0).round() / 100.0,
            "deviated_features": reasons,
            "activity": act,
        });
        (sev, title, details)
    }
}

// ── Sustained Rule State Helper (with 10-min Recurrence Coalescing) ───────

#[derive(Default)]
struct SustainedState {
    since: Option<f64>,
    open_ev: Option<Event>,
    peak: f64,
    last_closed: Option<Event>,
}

impl SustainedState {
    const REPEAT_MERGE_SEC: f64 = 600.0;

    fn reset(&mut self) {
        self.since = None;
    }

    #[allow(clippy::too_many_arguments)]
    fn open_or_reopen(
        &mut self,
        ctx: &mut impl MonitorContext,
        kind: &str,
        sev: &str,
        title: &str,
        since: f64,
        now: f64,
        details: Value,
    ) {
        if let Some(ref mut prev) = self.last_closed {
            if since - prev.ts_end <= Self::REPEAT_MERGE_SEC {
                ctx.with_events(|es| {
                    es.reopen(prev, now);
                    es.update(prev, now, Some(title), Some(sev), Some(details), true);
                });
                self.open_ev = self.last_closed.take();
                return;
            }
        }
        let mut ev = ctx.with_events(|es| es.open(kind, sev, title, since, details, true));
        ctx.with_events(|es| es.update(&mut ev, now, None, None, None, false));
        self.open_ev = Some(ev);
    }

    fn close_episode(
        &mut self,
        ctx: &mut impl MonitorContext,
        now: f64,
        title: &str,
        sev: &str,
        details: Value,
    ) {
        if let Some(mut ev) = self.open_ev.take() {
            ctx.with_events(|es| es.close(&mut ev, now, Some(title), Some(sev), Some(details)));
            self.last_closed = Some(ev);
        }
        self.since = None;
        self.peak = 0.0;
    }
}

// ── 2. MemoryPressure ─────────────────────────────────────────────────────

#[derive(Default)]
pub struct MemoryPressure {
    state: SustainedState,
    peak_hard_faults_ps: f64,
}

impl MemoryPressure {
    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub fn on_sample(&mut self, ctx: &mut impl MonitorContext, s: &MetricMap, now: f64) {
        let ram = get_metric_or(s, "ram_pct", 0.0);
        let commit = get_metric_or(s, "commit_pct", 0.0);
        let hf = get_metric_or(s, "hard_faults_ps", 0.0);
        let ram_hi = ctx.cfg().ram_high_pct;
        let com_hi = ctx.cfg().commit_high_pct;
        let sustain = ctx.cfg().memory_sustain_sec;

        let active = ram >= ram_hi || commit >= com_hi;
        let clear = ram < ram_hi - 4.0 && commit < com_hi - 4.0;
        let val = ram.max(commit);

        if active {
            let since = *self.state.since.get_or_insert(now);
            self.state.peak = self.state.peak.max(val);
            self.peak_hard_faults_ps = self.peak_hard_faults_ps.max(hf);

            if self.state.open_ev.is_none() && now - since >= sustain {
                let (sev, title, details) = self.describe(ctx, s, since, now);
                self.state.open_or_reopen(
                    ctx,
                    "memory_pressure",
                    &sev,
                    &title,
                    since,
                    now,
                    details,
                );
            } else if self.state.open_ev.is_some() {
                let (sev, title, details) = self.describe(ctx, s, since, now);
                if let Some(ref mut ev) = self.state.open_ev {
                    ctx.with_events(|es| {
                        es.update(ev, now, Some(&title), Some(&sev), Some(details), true)
                    });
                }
            }
        } else if clear {
            if let Some(ref ev) = self.state.open_ev {
                let since = ev.ts_start;
                let (sev, title, details) = self.describe(ctx, s, since, now);
                self.state.close_episode(ctx, now, &title, &sev, details);
                self.peak_hard_faults_ps = 0.0;
            } else {
                self.state.since = None;
                self.state.peak = 0.0;
                self.peak_hard_faults_ps = 0.0;
            }
        }
    }

    fn describe(
        &self,
        ctx: &impl MonitorContext,
        s: &MetricMap,
        ts_start: f64,
        ts_end: f64,
    ) -> (String, String, Value) {
        let act = ctx.activity(ts_start);
        let top = act
            .get("top_memory")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let hogs_str = top
            .iter()
            .take(3)
            .map(|p| {
                let n = p.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                let mb = p.get("private_mb").and_then(|v| v.as_i64()).unwrap_or(0);
                format!("{} ({} MB)", n, mb)
            })
            .collect::<Vec<_>>()
            .join(", ");

        let ram = get_metric_or(s, "ram_pct", 0.0);
        let commit = get_metric_or(s, "commit_pct", 0.0);
        let sev = if ram >= 97.0 || commit >= 96.0 || self.state.peak >= 97.0 {
            "critical"
        } else {
            "warning"
        };
        let swap_tag = if self.peak_hard_faults_ps >= 50.0 {
            format!(
                " [swap thrashing: {:.0} hard faults/s]",
                self.peak_hard_faults_ps
            )
        } else {
            String::new()
        };
        let hogs_suffix = if hogs_str.is_empty() {
            String::new()
        } else {
            format!(" — largest: {}", hogs_str)
        };
        let title = format!(
            "Memory pressure ({}): RAM {:.0}%, commit {:.0}% (peak {:.0}%){}{}",
            fmt_duration(ts_end - ts_start),
            ram,
            commit,
            self.state.peak,
            swap_tag,
            hogs_suffix
        );
        let details = json!({
            "ram_pct": (ram * 10.0).round() / 10.0,
            "commit_pct": (commit * 10.0).round() / 10.0,
            "peak_pct": (self.state.peak * 10.0).round() / 10.0,
            "peak_hard_faults_ps": (self.peak_hard_faults_ps * 10.0).round() / 10.0,
            "top_memory": top,
        });
        (sev.to_string(), title, details)
    }
}

// ── 3. GpuThermal ─────────────────────────────────────────────────────────

#[derive(Default)]
pub struct GpuThermal {
    state: SustainedState,
}

impl GpuThermal {
    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub fn on_sample(&mut self, ctx: &mut impl MonitorContext, s: &MetricMap, now: f64) {
        let awake = get_metric_or(s, "gpu_awake", 0.0) >= 0.5;
        let t = get_metric(s, "gpu_temp_c");
        let throttle = get_metric_or(s, "gpu_throttle", 0.0) as u64;
        // HW slowdown (0x08) | SW thermal (0x20) | HW thermal (0x40), guarded against idle 32-52C driver quirks
        let thermal_throttle = awake && t.is_finite() && t >= 65.0 && (throttle & 0x68) != 0;
        let warn_c = ctx.cfg().gpu_temp_warn_c;
        let sustain = ctx.cfg().gpu_temp_sustain_sec;
        let active = thermal_throttle || (t.is_finite() && t >= warn_c);
        let clear = (!t.is_finite() || t < warn_c - 3.0) && !thermal_throttle;

        if active {
            let since = *self.state.since.get_or_insert(now);
            self.state.peak = self.state.peak.max(t);
            if self.state.open_ev.is_none() && now - since >= sustain {
                let (sev, title, details) = self.describe(ctx, s, since, now);
                self.state
                    .open_or_reopen(ctx, "gpu_thermal", &sev, &title, since, now, details);
            } else if self.state.open_ev.is_some() {
                let (sev, title, details) = self.describe(ctx, s, since, now);
                if let Some(ref mut ev) = self.state.open_ev {
                    ctx.with_events(|es| {
                        es.update(ev, now, Some(&title), Some(&sev), Some(details), true)
                    });
                }
            }
        } else if clear {
            if let Some(ref ev) = self.state.open_ev {
                let since = ev.ts_start;
                let (sev, title, details) = self.describe(ctx, s, since, now);
                self.state.close_episode(ctx, now, &title, &sev, details);
            } else {
                self.state.since = None;
                self.state.peak = 0.0;
            }
        }
    }

    fn describe(
        &self,
        ctx: &impl MonitorContext,
        s: &MetricMap,
        ts_start: f64,
        ts_end: f64,
    ) -> (String, String, Value) {
        let act = ctx.activity(ts_start);
        let crit_c = ctx.cfg().gpu_temp_crit_c;
        let sev = if self.state.peak >= crit_c {
            "critical"
        } else {
            "warning"
        };
        let clients: Vec<String> = act
            .get("gpu_clients")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let who = if clients.is_empty() {
            String::new()
        } else {
            format!(" — GPU clients: {}", clients.join(", "))
        };
        let pwr = get_metric_or(s, "gpu_power_w", 0.0);
        let clk = get_metric_or(s, "gpu_clock_mhz", 0.0);
        let title = format!(
            "GPU hot ({}): peak {:.0}°C @ {:.0} W, {:.0} MHz{}",
            fmt_duration(ts_end - ts_start),
            self.state.peak,
            pwr,
            clk,
            who
        );
        let details = json!({
            "peak_temp_c": self.state.peak,
            "gpu_power_w": pwr,
            "gpu_clock_mhz": clk,
            "gpu_clients": clients,
        });
        (sev.to_string(), title, details)
    }
}

// ── 4. GpuAwakeOnBattery ──────────────────────────────────────────────────

#[derive(Default)]
pub struct GpuAwakeOnBattery {
    state: SustainedState,
}

impl GpuAwakeOnBattery {
    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub fn on_sample(&mut self, ctx: &mut impl MonitorContext, s: &MetricMap, now: f64) {
        let on_ac = get_metric_or(s, "on_ac", 1.0) >= 0.5;
        let gpu_awake = get_metric_or(s, "gpu_awake", 0.0) >= 0.5;
        let gpu_util = get_metric_or(s, "gpu_util_pct", 0.0);
        let gpu_pwr = get_metric_or(s, "gpu_power_w", 0.0);
        let sustain = ctx.cfg().gpu_battery_wake_sustain_sec;

        let active = !on_ac && gpu_awake && gpu_util < 5.0;
        let clear = on_ac || !gpu_awake || gpu_util >= 15.0;

        if active {
            let since = *self.state.since.get_or_insert(now);
            self.state.peak = self.state.peak.max(gpu_pwr);
            if self.state.open_ev.is_none() && now - since >= sustain {
                let (sev, title, details) = self.describe(ctx, s, since, now);
                self.state.open_or_reopen(
                    ctx,
                    "gpu_battery_drain",
                    &sev,
                    &title,
                    since,
                    now,
                    details,
                );
            } else if self.state.open_ev.is_some() {
                let (sev, title, details) = self.describe(ctx, s, since, now);
                if let Some(ref mut ev) = self.state.open_ev {
                    ctx.with_events(|es| {
                        es.update(ev, now, Some(&title), Some(&sev), Some(details), true)
                    });
                }
            }
        } else if clear {
            if let Some(ref ev) = self.state.open_ev {
                let since = ev.ts_start;
                let (sev, title, details) = self.describe(ctx, s, since, now);
                self.state.close_episode(ctx, now, &title, &sev, details);
            } else {
                self.state.since = None;
                self.state.peak = 0.0;
            }
        }
    }

    fn describe(
        &self,
        ctx: &impl MonitorContext,
        s: &MetricMap,
        ts_start: f64,
        ts_end: f64,
    ) -> (String, String, Value) {
        let act = ctx.activity(ts_start);
        let clients: Vec<String> = act
            .get("gpu_clients")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let holder = if clients.is_empty() {
            "unknown process".to_string()
        } else {
            clients.join(", ")
        };
        let pwr = get_metric_or(s, "gpu_power_w", self.state.peak);
        let title = format!(
            "Discrete GPU kept awake on battery ({}, ~{:.0} W idle drain) — held by {}",
            fmt_duration(ts_end - ts_start),
            pwr,
            holder
        );
        let details = json!({
            "gpu_power_w": pwr,
            "gpu_clients": clients,
        });
        ("warning".to_string(), title, details)
    }
}

// ── 5. LowDiskSpace ───────────────────────────────────────────────────────

pub struct LowDiskSpace {
    alerted_severity: Option<String>,
    last_alert_at: f64,
}

impl Default for LowDiskSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl LowDiskSpace {
    pub const COOLDOWN_SEC: f64 = 12.0 * 3600.0;

    pub fn new() -> Self {
        Self {
            alerted_severity: None,
            last_alert_at: 0.0,
        }
    }

    pub fn on_sample(&mut self, ctx: &mut impl MonitorContext, s: &MetricMap, now: f64) {
        let free = get_metric(s, "disk_free_gb");
        if !free.is_finite() {
            return;
        }
        let warn_gb = ctx.cfg().disk_free_warn_gb;
        let crit_gb = ctx.cfg().disk_free_crit_gb;

        if free > warn_gb + 2.0 {
            self.alerted_severity = None;
            return;
        }
        let sev = if free <= crit_gb {
            "critical"
        } else if free <= warn_gb {
            "warning"
        } else {
            return;
        };

        let is_escalation =
            self.alerted_severity.as_deref() == Some("warning") && sev == "critical";
        if self.alerted_severity.is_none()
            || is_escalation
            || (now - self.last_alert_at >= Self::COOLDOWN_SEC)
        {
            self.alerted_severity = Some(sev.to_string());
            self.last_alert_at = now;
            let title = format!("Low disk space on system drive: {:.1} GB free", free);
            let details = json!({ "disk_free_gb": (free * 100.0).round() / 100.0 });
            ctx.with_events(|es| {
                es.point("low_disk_space", sev, &title, now, details);
            });
        }
    }
}

// ── 6. UploadBurst ────────────────────────────────────────────────────────

#[derive(Default)]
pub struct UploadBurst {
    state: SustainedState,
    sent_bytes: f64,
    last_ts: Option<f64>,
}

impl UploadBurst {
    pub fn reset(&mut self) {
        self.state.reset();
        self.last_ts = None;
    }

    pub fn on_sample(&mut self, ctx: &mut impl MonitorContext, s: &MetricMap, now: f64) {
        let dt = self
            .last_ts
            .map(|lt| (now - lt).clamp(0.0, 10.0))
            .unwrap_or(1.0);
        self.last_ts = Some(now);

        let floor = ctx.cfg().net_upload_floor_mbps * 1_048_576.0;
        let sustain = ctx.cfg().net_upload_sustain_sec;
        let rate = get_metric_or(s, "net_sent_bps", 0.0);

        if rate >= floor {
            self.sent_bytes += rate * dt;
            let since = *self.state.since.get_or_insert(now);
            self.state.peak = self.state.peak.max(rate);
            if self.state.open_ev.is_none() && now - since >= sustain {
                let (sev, title, details) = self.describe(ctx, since, now);
                self.state
                    .open_or_reopen(ctx, "upload_burst", &sev, &title, since, now, details);
            } else if self.state.open_ev.is_some() {
                let (sev, title, details) = self.describe(ctx, since, now);
                if let Some(ref mut ev) = self.state.open_ev {
                    ctx.with_events(|es| {
                        es.update(ev, now, Some(&title), Some(&sev), Some(details), true)
                    });
                }
            }
        } else if rate < floor * 0.25 {
            if let Some(ref ev) = self.state.open_ev {
                let since = ev.ts_start;
                let (sev, title, details) = self.describe(ctx, since, now);
                self.state.close_episode(ctx, now, &title, &sev, details);
            } else {
                self.state.since = None;
                self.state.peak = 0.0;
            }
            self.sent_bytes = 0.0;
        }
    }

    fn describe(
        &self,
        ctx: &impl MonitorContext,
        ts_start: f64,
        ts_end: f64,
    ) -> (String, String, Value) {
        let act = ctx.activity(ts_start);
        let title = format!(
            "Sustained upload ({}): peak {}, ~{} sent{}",
            fmt_duration(ts_end - ts_start),
            fmt_bytes(self.state.peak, true),
            fmt_bytes(self.sent_bytes, false),
            fmt_top_procs(&act)
        );
        let details = json!({
            "peak_bps": self.state.peak.round(),
            "total_mb": (self.sent_bytes / 1_048_576.0 * 10.0).round() / 10.0,
            "activity": act,
        });
        ("warning".to_string(), title, details)
    }
}

// ── 7. UiFreeze (Responsiveness stalls + Kernel DPC/ISR storms) ───────────

pub struct UiFreeze {
    recent_samples: VecDeque<MetricMap>,
    open_ev: HashMap<(String, usize), Event>,
    start_proc_cpu: HashMap<(String, usize), Option<f64>>,
    dpc_since: Option<f64>,
    dpc_peak: f64,
    dpc_ev: Option<Event>,
}

impl Default for UiFreeze {
    fn default() -> Self {
        Self::new()
    }
}

impl UiFreeze {
    pub const SAMPLE_RING: usize = 30;

    pub fn new() -> Self {
        Self {
            recent_samples: VecDeque::with_capacity(Self::SAMPLE_RING + 2),
            open_ev: HashMap::new(),
            start_proc_cpu: HashMap::new(),
            dpc_since: None,
            dpc_peak: 0.0,
            dpc_ev: None,
        }
    }

    pub fn reset(&mut self) {
        self.dpc_since = None;
        self.dpc_peak = 0.0;
    }

    pub fn on_sample(&mut self, ctx: &mut impl MonitorContext, s: &MetricMap, now: f64) {
        if self.recent_samples.len() >= Self::SAMPLE_RING {
            self.recent_samples.pop_front();
        }
        self.recent_samples.push_back(s.clone());
        self.check_dpc_storm(ctx, s, now);
    }

    pub fn on_freezes(
        &mut self,
        ctx: &mut impl MonitorContext,
        completed: &[Freeze],
        ongoing: &[Freeze],
        now: f64,
    ) {
        let min_dur = ctx.cfg().ui_freeze_sustain_sec;
        let mut live_keys = HashSet::new();

        for f in ongoing {
            let key = (f.source.clone(), f.hwnd);
            live_keys.insert(key.clone());
            self.start_proc_cpu
                .entry(key.clone())
                .or_insert_with(|| ctx.cpu_seconds(f.pid));
            if self.worth_reporting(ctx, f, min_dur) {
                let (sev, title, details) = self.describe(ctx, f);
                if let Some(ev) = self.open_ev.get_mut(&key) {
                    ctx.with_events(|es| {
                        es.update(ev, f.ts_end, Some(&title), Some(&sev), Some(details), true)
                    });
                } else {
                    let ev = ctx.with_events(|es| {
                        es.open("ui_freeze", &sev, &title, f.ts_start, details, true)
                    });
                    self.open_ev.insert(key, ev);
                }
            }
        }

        for f in completed {
            let key = (f.source.clone(), f.hwnd);
            if let Some(mut ev) = self.open_ev.remove(&key) {
                let (sev, title, details) = self.describe(ctx, f);
                ctx.with_events(|es| {
                    es.close(&mut ev, f.ts_end, Some(&title), Some(&sev), Some(details))
                });
            } else if self.worth_reporting(ctx, f, min_dur) {
                let (sev, title, details) = self.describe(ctx, f);
                let mut ev = ctx.with_events(|es| {
                    es.open("ui_freeze", &sev, &title, f.ts_start, details, true)
                });
                ctx.with_events(|es| es.close(&mut ev, f.ts_end, None, None, None));
            }
            self.start_proc_cpu.remove(&key);
        }

        let stale_keys: Vec<(String, usize)> = self
            .open_ev
            .keys()
            .filter(|k| !live_keys.contains(*k))
            .cloned()
            .collect();
        for key in stale_keys {
            if let Some(mut ev) = self.open_ev.remove(&key) {
                ctx.with_events(|es| es.close(&mut ev, now, None, None, None));
            }
            self.start_proc_cpu.remove(&key);
        }
    }

    fn worth_reporting(&self, ctx: &impl MonitorContext, f: &Freeze, min_dur: f64) -> bool {
        if !f.process.is_empty() && is_ignored(ctx.cfg(), &f.process) {
            return false;
        }
        f.timed_out || f.hung || f.duration() >= min_dur
    }

    fn describe(&self, ctx: &impl MonitorContext, f: &Freeze) -> (String, String, Value) {
        let dur = f.duration();
        let sev = if dur >= ctx.cfg().ui_severe_freeze_sec
            || f.source == "taskbar"
            || f.source == "scheduler"
        {
            "critical"
        } else {
            "warning"
        };
        let (cause_code, cause_text, metrics) = self.classify_cause(ctx, f);

        let who = match f.source.as_str() {
            "taskbar" => "Windows shell (explorer.exe taskbar)".to_string(),
            "compositor" => "Desktop compositor (dwm.exe)".to_string(),
            "scheduler" => "System scheduler (OS-wide stall)".to_string(),
            _ => {
                let mut w = if f.process.is_empty() {
                    format!("PID {}", f.pid)
                } else {
                    f.process.clone()
                };
                if !f.title.is_empty() {
                    let short: String = f.title.chars().take(40).collect();
                    w = format!("{} ({:?})", w, short);
                }
                w
            }
        };

        let mut flags = Vec::new();
        if f.hung {
            flags.push("marked Not Responding");
        }
        if f.timed_out {
            flags.push("ping timed out");
        }
        let flag_str = if flags.is_empty() {
            String::new()
        } else {
            format!(" [{}]", flags.join(", "))
        };
        let title = format!(
            "UI freeze ({}): {} stalled (peak {:.0} ms){} — likely cause: {}",
            fmt_duration(dur),
            who,
            f.max_latency_ms,
            flag_str,
            cause_text
        );
        let details = json!({
            "source": f.source,
            "process": f.process,
            "pid": f.pid,
            "tid": f.tid,
            "window_title": f.title,
            "duration_sec": (dur * 100.0).round() / 100.0,
            "max_latency_ms": (f.max_latency_ms * 10.0).round() / 10.0,
            "pings": f.pings,
            "timed_out": f.timed_out,
            "hung": f.hung,
            "cause": cause_code,
            "cause_explanation": cause_text,
            "concurrent": metrics,
        });
        (sev.to_string(), title, details)
    }

    fn classify_cause(&self, ctx: &impl MonitorContext, f: &Freeze) -> (String, String, Value) {
        let mut window: Vec<&MetricMap> = self
            .recent_samples
            .iter()
            .filter(|s| {
                let ts = get_metric_or(s, "ts", 0.0);
                ts >= f.ts_start - 2.0 && ts <= f.ts_end + 2.0
            })
            .collect();
        if window.is_empty() {
            window = self.recent_samples.iter().rev().take(5).collect();
        }
        let peak = |col: &str| -> f64 {
            window
                .iter()
                .map(|s| get_metric(s, col))
                .filter(|v| v.is_finite())
                .fold(0.0f64, f64::max)
        };

        let dpc = peak("dpc_pct");
        let isr = peak("isr_pct");
        let dpc_core = peak("dpc_max_core_pct");
        let cpu = peak("cpu_pct");
        let core_max = peak("core_max_pct");
        let ram = peak("ram_pct");
        let commit = peak("commit_pct");
        let disk_r = peak("disk_read_bps");
        let disk_w = peak("disk_write_bps");
        let hf = peak("hard_faults_ps");
        let gpu = peak("gpu_util_pct");

        let act = ctx.activity(f.ts_start - 5.0);

        let key = (f.source.clone(), f.hwnd);
        let c0 = self.start_proc_cpu.get(&key).copied().flatten();
        let c1 = ctx.cpu_seconds(f.pid);
        let proc_cores = match (c0, c1) {
            (Some(a), Some(b)) if b >= a => Some((b - a) / f.duration().max(0.25)),
            _ => None,
        };

        let snap = json!({
            "cpu_pct": (cpu * 10.0).round() / 10.0,
            "core_max_pct": (core_max * 10.0).round() / 10.0,
            "dpc_pct": (dpc * 10.0).round() / 10.0,
            "isr_pct": (isr * 10.0).round() / 10.0,
            "dpc_max_core_pct": (dpc_core * 10.0).round() / 10.0,
            "ram_pct": (ram * 10.0).round() / 10.0,
            "commit_pct": (commit * 10.0).round() / 10.0,
            "disk_read_mb_s": (disk_r / 1_048_576.0 * 10.0).round() / 10.0,
            "disk_write_mb_s": (disk_w / 1_048_576.0 * 10.0).round() / 10.0,
            "hard_faults_ps": (hf * 10.0).round() / 10.0,
            "gpu_util_pct": (gpu * 10.0).round() / 10.0,
            "stalled_proc_cores": proc_cores.map(|c| (c * 100.0).round() / 100.0),
            "activity": act,
        });

        let mut reasons: Vec<(String, String)> = Vec::new();
        if dpc + isr >= ctx.cfg().dpc_storm_pct || dpc_core >= 50.0 {
            reasons.push((
                "dpc_storm".to_string(),
                format!(
                    "kernel driver DPC/ISR storm ({:.0}% CPU, busiest core {:.0}%)",
                    dpc + isr,
                    dpc_core
                ),
            ));
        }
        if f.source == "scheduler" {
            reasons.push((
                "scheduler_starvation".to_string(),
                format!(
                    "high-priority probe thread delayed {:.0} ms by kernel/driver activity",
                    f.max_latency_ms
                ),
            ));
        }
        if ram >= ctx.cfg().ram_high_pct || commit >= ctx.cfg().commit_high_pct || hf >= 150.0 {
            let hogs = act
                .get("top_memory")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .take(2)
                        .filter_map(|p| p.get("name").and_then(|v| v.as_str()))
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            let extra = if hogs.is_empty() {
                String::new()
            } else {
                format!("; top RAM: {}", hogs)
            };
            let hf_note = if hf >= 50.0 {
                format!(", {:.0} hard page faults/s", hf)
            } else {
                String::new()
            };
            reasons.push((
                "memory_paging".to_string(),
                format!(
                    "memory pressure / pagefile thrashing (RAM {:.0}%, commit {:.0}%{}{})",
                    ram, commit, hf_note, extra
                ),
            ));
        }
        if disk_r + disk_w >= 50.0 * 1_048_576.0 {
            let io_hog = act
                .get("top_io")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|p| p.get("name"))
                .and_then(|v| v.as_str())
                .map(|n| format!(" by {}", n))
                .unwrap_or_default();
            reasons.push((
                "disk_io_stall".to_string(),
                format!(
                    "heavy disk I/O ({} R / {} W{})",
                    fmt_bytes(disk_r, true),
                    fmt_bytes(disk_w, true),
                    io_hog
                ),
            ));
        }
        if f.source == "compositor" || gpu >= 95.0 {
            reasons.push((
                "gpu_compositor_stall".to_string(),
                format!("GPU / DWM composition stall (GPU load {:.0}%)", gpu),
            ));
        }
        if cpu >= 85.0 {
            let cpu_hog = act
                .get("top_cpu")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .map(|c| {
                    format!(
                        " — top: {} ({:.1} cores)",
                        c.get("name").and_then(|v| v.as_str()).unwrap_or("?"),
                        c.get("cores").and_then(|v| v.as_f64()).unwrap_or(0.0)
                    )
                })
                .unwrap_or_default();
            reasons.push((
                "cpu_saturation".to_string(),
                format!("system-wide CPU saturation ({:.0}%{})", cpu, cpu_hog),
            ));
        }

        if reasons.is_empty() && (f.source == "foreground" || f.source == "taskbar") {
            let who = if f.process.is_empty() {
                "UI thread"
            } else {
                f.process.as_str()
            };
            if let Some(pc) = proc_cores {
                if pc >= 0.7 {
                    reasons.push((
                        "ui_thread_cpu_spin".to_string(),
                        format!(
                            "{} burning CPU on its UI thread ({:.2} cores used during stall, single core {:.0}%)",
                            who, pc, core_max
                        ),
                    ));
                } else if pc < 0.05 {
                    reasons.push((
                        "ui_thread_blocked_wait".to_string(),
                        format!(
                            "{} UI thread sleeping on a lock, synchronous file/registry call, or network request (0 CPU used during stall)",
                            who
                        ),
                    ));
                }
            }
            if reasons.is_empty() {
                if core_max >= 90.0 {
                    reasons.push((
                        "ui_thread_cpu_spin".to_string(),
                        format!(
                            "{} main thread pegging one CPU core ({:.0}% on busiest core, system CPU {:.0}%)",
                            who, core_max, cpu
                        ),
                    ));
                } else {
                    reasons.push((
                        "ui_thread_blocked".to_string(),
                        format!(
                            "{} main thread blocked (system CPU {:.0}%, RAM {:.0}%, disk I/O low)",
                            who, cpu, ram
                        ),
                    ));
                }
            }
        }

        if reasons.is_empty() {
            return (
                "unclassified".to_string(),
                "transient stall".to_string(),
                snap,
            );
        }
        let code = reasons[0].0.clone();
        let text = reasons
            .into_iter()
            .map(|(_, t)| t)
            .collect::<Vec<_>>()
            .join("; plus ");
        (code, text, snap)
    }

    fn check_dpc_storm(&mut self, ctx: &mut impl MonitorContext, s: &MetricMap, now: f64) {
        let dpc = get_metric_or(s, "dpc_pct", 0.0);
        let isr = get_metric_or(s, "isr_pct", 0.0);
        let core = get_metric_or(s, "dpc_max_core_pct", 0.0);
        let total = dpc + isr;
        let storm_thr = ctx.cfg().dpc_storm_pct;
        let active = total >= storm_thr || core >= 50.0;
        let clear = total < storm_thr * 0.5 && core < 25.0;

        if active {
            let since = *self.dpc_since.get_or_insert(now);
            self.dpc_peak = self.dpc_peak.max(total);
            let dur = now - since;
            if self.dpc_ev.is_none() && dur >= 3.0 {
                let drivers: Vec<String> = winapi::loaded_third_party_drivers()
                    .into_iter()
                    .take(8)
                    .collect();
                let title = format!(
                    "Kernel DPC/ISR storm ({}): {:.0}% CPU in driver interrupts (busiest core {:.0}%)",
                    fmt_duration(dur),
                    self.dpc_peak,
                    core
                );
                let details = json!({
                    "dpc_pct": (dpc * 10.0).round() / 10.0,
                    "isr_pct": (isr * 10.0).round() / 10.0,
                    "peak_pct": (self.dpc_peak * 10.0).round() / 10.0,
                    "dpc_max_core_pct": (core * 10.0).round() / 10.0,
                    "loaded_third_party_drivers": drivers,
                    "note": "High DPC/ISR time comes from a kernel driver (network, storage, GPU, audio) and stalls mouse/audio/UI regardless of free CPU.",
                });
                let ev = ctx.with_events(|es| {
                    es.open("dpc_storm", "warning", &title, since, details, true)
                });
                self.dpc_ev = Some(ev);
            } else if let Some(ref mut ev) = self.dpc_ev {
                let sev = if dur >= 15.0 || self.dpc_peak >= 35.0 {
                    "critical"
                } else {
                    "warning"
                };
                let title = format!(
                    "Kernel DPC/ISR storm ({}): peak {:.0}% CPU in driver interrupts (busiest core {:.0}%)",
                    fmt_duration(dur),
                    self.dpc_peak,
                    core
                );
                let details = json!({ "peak_pct": (self.dpc_peak * 10.0).round() / 10.0 });
                ctx.with_events(|es| {
                    es.update(ev, now, Some(&title), Some(sev), Some(details), true)
                });
            }
        } else if clear {
            if let Some(mut ev) = self.dpc_ev.take() {
                ctx.with_events(|es| es.close(&mut ev, now, None, None, None));
            }
            self.dpc_since = None;
            self.dpc_peak = 0.0;
        }
    }
}

// ── 8. BackgroundCpu ──────────────────────────────────────────────────────

pub struct BackgroundCpu {
    hist: HashMap<ProcKey, VecDeque<(f64, f64, bool)>>,
    open_ev: HashMap<ProcKey, (Event, Vec<f64>)>,
    cooldown: HashMap<ProcKey, f64>,
}

impl Default for BackgroundCpu {
    fn default() -> Self {
        Self::new()
    }
}

impl BackgroundCpu {
    pub const COOLDOWN_SEC: f64 = 1800.0;

    pub fn new() -> Self {
        Self {
            hist: HashMap::new(),
            open_ev: HashMap::new(),
            cooldown: HashMap::new(),
        }
    }

    pub fn on_sweep(&mut self, ctx: &mut impl MonitorContext, stats: &[ProcStats], now: f64) {
        let win = ctx.cfg().background_cpu_sustain_sec;
        let thr = ctx.cfg().background_cpu_cores;
        let min_samples = (win / ctx.cfg().process_interval_sec * 0.7).max(3.0) as usize;

        let act = ctx.activity(now - 30.0);
        let mut active_names: HashSet<String> = HashSet::new();
        if let Some(arr) = act.get("gpu_clients").and_then(|v| v.as_array()) {
            for v in arr {
                if let Some(s) = v.as_str() {
                    active_names.insert(s.to_ascii_lowercase());
                }
            }
        }
        for s in stats {
            if s.foreground {
                active_names.insert(s.name.to_ascii_lowercase());
                active_names.insert(s.base_name.to_ascii_lowercase());
            }
        }

        let mut seen = HashSet::with_capacity(stats.len());
        for s in stats {
            seen.insert(s.key);
            if is_ignored(ctx.cfg(), &s.name) {
                continue;
            }
            let h = self.hist.entry(s.key).or_default();
            let fg_like = s.foreground
                || active_names.contains(&s.name.to_ascii_lowercase())
                || active_names.contains(&s.base_name.to_ascii_lowercase());
            h.push_back((now, s.cpu_cores, fg_like));
            while let Some(&(t0, _, _)) = h.front() {
                if t0 < now - win {
                    h.pop_front();
                } else {
                    break;
                }
            }

            if self.open_ev.contains_key(&s.key) {
                let h_snap = h.clone();
                self.update_open(ctx, s, &h_snap, now, thr);
                continue;
            }
            if h.len() < min_samples || h.front().unwrap().0 > now - win * 0.8 {
                continue;
            }
            if let Some(&cd) = self.cooldown.get(&s.key) {
                if now < cd {
                    continue;
                }
            }

            let n_h = h.len() as f64;
            let avg_cores: f64 = h.iter().map(|x| x.1).sum::<f64>() / n_h;
            let fg_share: f64 = h.iter().filter(|x| x.2).count() as f64 / n_h;
            let busy_share: f64 = h.iter().filter(|x| x.1 >= thr * 0.7).count() as f64 / n_h;

            if fg_share == 0.0 && avg_cores >= thr && busy_share >= 0.8 {
                let start = h.front().unwrap().0;
                let title = format!(
                    "{} using {:.1} CPU cores in the background for {}",
                    s.name,
                    avg_cores,
                    fmt_duration(now - start)
                );
                let details = json!({
                    "process": s.name,
                    "pid": s.pid(),
                    "avg_cores": (avg_cores * 100.0).round() / 100.0,
                });
                let mut ev = ctx.with_events(|es| {
                    es.open("background_cpu", "warning", &title, start, details, true)
                });
                ctx.with_events(|es| es.update(&mut ev, now, None, None, None, false));
                let cores_vec: Vec<f64> = h.iter().map(|x| x.1).collect();
                self.open_ev.insert(s.key, (ev, cores_vec));
            }
        }

        let dead_hist: Vec<ProcKey> = self
            .hist
            .keys()
            .filter(|k| !seen.contains(*k))
            .copied()
            .collect();
        for k in dead_hist {
            self.hist.remove(&k);
        }
        let dead_open: Vec<ProcKey> = self
            .open_ev
            .keys()
            .filter(|k| !seen.contains(*k))
            .copied()
            .collect();
        for k in dead_open {
            self.close_key(ctx, k, now, "process exited");
        }
    }

    fn update_open(
        &mut self,
        ctx: &mut impl MonitorContext,
        s: &ProcStats,
        h: &VecDeque<(f64, f64, bool)>,
        now: f64,
        thr: f64,
    ) {
        let recent_len = h.len().clamp(1, 6);
        let recent_sum: f64 = h.iter().rev().take(6).map(|x| x.1).sum();
        let recent_fg: usize = h.iter().rev().take(6).filter(|x| x.2).count();

        if recent_sum / (recent_len as f64) < thr / 2.0 {
            self.close_key(ctx, s.key, now, "CPU use dropped");
        } else if (recent_fg as f64) / (recent_len as f64) > 0.5 {
            self.close_key(ctx, s.key, now, "brought to the foreground");
        } else if let Some((ref mut ev, ref mut cores)) = self.open_ev.get_mut(&s.key) {
            cores.push(s.cpu_cores);
            if cores.len() % 12 == 0 {
                let avg = cores.iter().sum::<f64>() / (cores.len() as f64);
                let title = format!(
                    "{} used {:.1} CPU cores in the background for {}",
                    s.name,
                    avg,
                    fmt_duration(now - ev.ts_start)
                );
                let details = json!({ "avg_cores": (avg * 100.0).round() / 100.0 });
                ctx.with_events(|es| es.update(ev, now, Some(&title), None, Some(details), false));
            }
        }
    }

    fn close_key(&mut self, ctx: &mut impl MonitorContext, key: ProcKey, now: f64, why: &str) {
        if let Some((mut ev, cores)) = self.open_ev.remove(&key) {
            let avg = if cores.is_empty() {
                0.0
            } else {
                cores.iter().sum::<f64>() / (cores.len() as f64)
            };
            let proc_name = ev
                .details
                .get("process")
                .and_then(|v| v.as_str())
                .unwrap_or("process")
                .to_string();
            let title = format!(
                "{} used {:.1} CPU cores in the background for {}",
                proc_name,
                avg,
                fmt_duration(now - ev.ts_start)
            );
            let details = json!({
                "avg_cores": (avg * 100.0).round() / 100.0,
                "ended": why,
            });
            ctx.with_events(|es| es.close(&mut ev, now, Some(&title), None, Some(details)));
            self.cooldown.insert(key, now + Self::COOLDOWN_SEC);
        }
    }
}

// ── 9. Leaks (Memory & Handle OLS Linear Regression) ──────────────────────

const NOT_MEMORY_LEAKS: [&str; 6] = [
    "memory compression",
    "registry",
    "system",
    "secure system",
    "vmmem",
    "vmmemwsl",
];

/// Compute `(slope_per_hour, r2, monotone_by_thirds)` of series `(t, y)`.
pub fn trend(t: &[f64], y: &[f64]) -> (f64, f64, bool) {
    let n = t.len().min(y.len());
    if n < 3 {
        return (0.0, 0.0, false);
    }
    let t0 = t[0];
    let th: Vec<f64> = t[..n].iter().map(|&ti| (ti - t0) / 3600.0).collect();
    let mean_x = th.iter().sum::<f64>() / (n as f64);
    let mean_y = y[..n].iter().sum::<f64>() / (n as f64);

    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..n {
        let dx = th[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        den += dx * dx;
    }
    let slope = if den > 1e-12 { num / den } else { 0.0 };
    let intercept = mean_y - slope * mean_x;

    let mut ss_res = 0.0f64;
    let mut ss_tot = 0.0f64;
    for i in 0..n {
        let pred = slope * th[i] + intercept;
        let r = y[i] - pred;
        let dy = y[i] - mean_y;
        ss_res += r * r;
        ss_tot += dy * dy;
    }
    let r2 = if ss_tot > 0.0 {
        1.0 - (ss_res / ss_tot)
    } else {
        0.0
    };

    let third = n / 3;
    let a = nanmedian(&y[..third]);
    let b = nanmedian(&y[third..2 * third]);
    let c = nanmedian(&y[2 * third..n]);
    (slope, r2, a < b && b < c)
}

pub struct Leaks {
    hist: HashMap<ProcKey, VecDeque<(f64, f64, f64)>>,
    names: HashMap<ProcKey, (String, String)>, // (enriched_name, base_name)
    pub reported: HashMap<(String, String), (f64, f64)>,
    next_sample: f64,
    next_check: f64,
}

impl Default for Leaks {
    fn default() -> Self {
        Self::new(&[])
    }
}

impl Leaks {
    pub const SAMPLE_SEC: f64 = 120.0;
    pub const CHECK_SEC: f64 = 300.0;
    pub const MIN_R2: f64 = 0.8;
    pub const REMIND_SEC: f64 = 30.0 * 86400.0;
    pub const REMIND_FACTOR: f64 = 2.0;

    pub fn new(recent_leak_events: &[EventRecord]) -> Self {
        let mut reported: HashMap<(String, String), (f64, f64)> = HashMap::new();
        for ev in recent_leak_events {
            let Some(proc_name) = ev.details.get("process").and_then(|v| v.as_str()) else {
                continue;
            };
            let val = if ev.kind == "handle_leak" {
                ev.details
                    .get("handles")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
            } else {
                ev.details
                    .get("private_mb")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
                    * 1_048_576.0
            };
            let k = (ev.kind.clone(), proc_name.to_ascii_lowercase());
            match reported.get(&k) {
                Some(&(_, prev_ts)) if prev_ts >= ev.ts => {}
                _ => {
                    reported.insert(k, (val, ev.ts));
                }
            }
        }
        Self {
            hist: HashMap::new(),
            names: HashMap::new(),
            reported,
            next_sample: 0.0,
            next_check: 0.0,
        }
    }

    pub fn on_sweep(&mut self, ctx: &mut impl MonitorContext, stats: &[ProcStats], now: f64) {
        if now < self.next_sample {
            return;
        }
        self.next_sample = now + Self::SAMPLE_SEC;
        let horizon = now - ctx.cfg().leak_history_hours * 3600.0;

        let mut seen = HashSet::with_capacity(stats.len());
        for s in stats {
            seen.insert(s.key);
            let h = self.hist.entry(s.key).or_default();
            h.push_back((now, s.private_bytes as f64, s.handles as f64));
            self.names
                .insert(s.key, (s.name.clone(), s.base_name.clone()));
            while let Some(&(t0, _, _)) = h.front() {
                if t0 < horizon {
                    h.pop_front();
                } else {
                    break;
                }
            }
        }

        let dead: Vec<ProcKey> = self
            .hist
            .keys()
            .filter(|k| !seen.contains(*k))
            .copied()
            .collect();
        for k in dead {
            self.hist.remove(&k);
            self.names.remove(&k);
        }

        if now >= self.next_check {
            self.next_check = now + Self::CHECK_SEC;
            let keys: Vec<ProcKey> = self.hist.keys().copied().collect();
            for key in keys {
                let Some((name, base_name)) = self.names.get(&key).cloned() else {
                    continue;
                };
                if is_ignored(ctx.cfg(), &name) {
                    continue;
                }
                let h = self.hist.get(&key).unwrap().clone();
                self.check_handles(ctx, key, &name, &base_name, &h, now);
                if !NOT_MEMORY_LEAKS.contains(&base_name.to_ascii_lowercase().as_str()) {
                    self.check_memory(ctx, key, &name, &base_name, &h, now);
                }
            }
        }
    }

    fn due(&self, kind: &str, name: &str, base_name: &str, value: f64, now: f64) -> bool {
        let last = self
            .reported
            .get(&(kind.to_string(), name.to_ascii_lowercase()))
            .or_else(|| {
                self.reported
                    .get(&(kind.to_string(), base_name.to_ascii_lowercase()))
            });
        match last {
            None => true,
            Some(&(prev_val, prev_ts)) => {
                now - prev_ts >= Self::REMIND_SEC || value >= Self::REMIND_FACTOR * prev_val
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn report(
        &mut self,
        ctx: &mut impl MonitorContext,
        kind: &str,
        name: &str,
        value: f64,
        now: f64,
        title: String,
        details: Value,
    ) {
        self.reported
            .insert((kind.to_string(), name.to_ascii_lowercase()), (value, now));
        ctx.with_events(|es| {
            es.point(kind, "warning", &title, now, details);
        });
    }

    fn check_memory(
        &mut self,
        ctx: &mut impl MonitorContext,
        key: ProcKey,
        name: &str,
        base_name: &str,
        h: &VecDeque<(f64, f64, f64)>,
        now: f64,
    ) {
        if h.len() < 6 {
            return;
        }
        let min_growth = ctx.cfg().leak_min_growth_mb * 1_048_576.0;
        let t: Vec<f64> = h.iter().map(|x| x.0).collect();
        let y: Vec<f64> = h.iter().map(|x| x.1).collect();
        let span_h = (t[t.len() - 1] - t[0]) / 3600.0;
        let cur_val = y[y.len() - 1];

        if span_h < ctx.cfg().leak_min_history_hours || cur_val - y[0] < 0.75 * min_growth {
            return;
        }
        if !self.due("memory_leak", name, base_name, cur_val, now) {
            return;
        }
        let (slope, r2, monotone) = trend(&t, &y);
        let growth = slope * span_h;
        if growth >= min_growth && r2 >= Self::MIN_R2 && monotone {
            let title = format!(
                "{} memory growing steadily: +{:.0} MB in {:.1} h ({:.0} MB/h), now {:.0} MB",
                name,
                growth / 1_048_576.0,
                span_h,
                slope / 1_048_576.0,
                cur_val / 1_048_576.0
            );
            let details = json!({
                "process": name,
                "pid": key.0,
                "create_time": key.1,
                "private_mb": (cur_val / 1_048_576.0).round() as i64,
                "growth_mb": (growth / 1_048_576.0).round() as i64,
                "mb_per_hour": (slope / 1_048_576.0 * 10.0).round() / 10.0,
                "r2": (r2 * 1000.0).round() / 1000.0,
                "hours_observed": (span_h * 100.0).round() / 100.0,
            });
            self.report(ctx, "memory_leak", name, cur_val, now, title, details);
        }
    }

    fn check_handles(
        &mut self,
        ctx: &mut impl MonitorContext,
        key: ProcKey,
        name: &str,
        base_name: &str,
        h: &VecDeque<(f64, f64, f64)>,
        now: f64,
    ) {
        let Some(&(_, _, current)) = h.back() else {
            return;
        };
        if current < ctx.cfg().handle_growth_min_count {
            return;
        }
        let t: Vec<f64> = h.iter().map(|x| x.0).collect();
        let y: Vec<f64> = h.iter().map(|x| x.2).collect();
        let span_h = if t.len() > 1 {
            (t[t.len() - 1] - t[0]) / 3600.0
        } else {
            0.0
        };

        let mut rate = None;
        if h.len() >= 6 && span_h >= ctx.cfg().leak_min_history_hours {
            let (slope, r2, monotone) = trend(&t, &y);
            if r2 >= Self::MIN_R2 && monotone && slope * span_h >= ctx.cfg().handle_leak_min_growth
            {
                rate = Some(slope);
            }
        }

        let mut why = if current >= ctx.cfg().handle_absolute_limit {
            format!("holds {:.0} handles (typical apps use < 10,000)", current)
        } else if rate.is_some() {
            format!("handle count growing steadily, now {:.0}", current)
        } else {
            return;
        };

        if !self.due("handle_leak", name, base_name, current, now) {
            return;
        }
        if let Some(r) = rate {
            why.push_str(&format!(" (+{:.0}/h)", r));
        }
        let title = format!("{} {}", name, why);
        let details = json!({
            "process": name,
            "pid": key.0,
            "create_time": key.1,
            "handles": current.round() as i64,
            "handles_per_hour": rate.map(|r| r.round() as i64),
            "hours_observed": (span_h * 100.0).round() / 100.0,
            "note": "Handle leaks slowly exhaust kernel memory; restarting the process frees them.",
        });
        self.report(ctx, "handle_leak", name, current, now, title, details);
    }
}

/// Bundle of all 9 detectors for the live engine.
pub struct RuleEngine {
    pub anomaly: AnomalyEpisodes,
    pub mem_pressure: MemoryPressure,
    pub gpu_thermal: GpuThermal,
    pub gpu_battery: GpuAwakeOnBattery,
    pub low_disk: LowDiskSpace,
    pub upload_burst: UploadBurst,
    pub ui_freeze: UiFreeze,
    pub bg_cpu: BackgroundCpu,
    pub leaks: Leaks,
}

impl RuleEngine {
    pub fn new(recent_leak_events: &[EventRecord]) -> Self {
        Self {
            anomaly: AnomalyEpisodes::new(),
            mem_pressure: MemoryPressure::default(),
            gpu_thermal: GpuThermal::default(),
            gpu_battery: GpuAwakeOnBattery::default(),
            low_disk: LowDiskSpace::new(),
            upload_burst: UploadBurst::default(),
            ui_freeze: UiFreeze::new(),
            bg_cpu: BackgroundCpu::new(),
            leaks: Leaks::new(recent_leak_events),
        }
    }

    pub fn reset(&mut self) {
        self.anomaly.reset();
        self.mem_pressure.reset();
        self.gpu_thermal.reset();
        self.gpu_battery.reset();
        self.upload_burst.reset();
        self.ui_freeze.reset();
    }

    pub fn on_sample(&mut self, ctx: &mut impl MonitorContext, s: &MetricMap, now: f64) {
        self.mem_pressure.on_sample(ctx, s, now);
        self.gpu_thermal.on_sample(ctx, s, now);
        self.gpu_battery.on_sample(ctx, s, now);
        self.low_disk.on_sample(ctx, s, now);
        self.upload_burst.on_sample(ctx, s, now);
        self.ui_freeze.on_sample(ctx, s, now);
    }

    pub fn on_window(
        &mut self,
        ctx: &mut impl MonitorContext,
        w: &MetricMap,
        x_opt: Option<&[f64; NUM_FEATURES]>,
        now: f64,
    ) {
        self.anomaly.on_window(ctx, w, x_opt, now);
    }

    pub fn on_sweep(&mut self, ctx: &mut impl MonitorContext, stats: &[ProcStats], now: f64) {
        self.bg_cpu.on_sweep(ctx, stats, now);
        self.leaks.on_sweep(ctx, stats, now);
    }

    pub fn on_freezes(
        &mut self,
        ctx: &mut impl MonitorContext,
        completed: &[Freeze],
        ongoing: &[Freeze],
        now: f64,
    ) {
        self.ui_freeze.on_freezes(ctx, completed, ongoing, now);
    }
}

/// One-line human summary of a sample (used by `status` and `state.json`).
pub fn summarize_rates(s: &MetricMap) -> String {
    let gpu_awake = get_metric(s, "gpu_awake");
    let gpu = if !gpu_awake.is_finite() {
        "absent".to_string()
    } else if gpu_awake < 0.5 {
        "asleep".to_string()
    } else {
        format!(
            "{:.0}% @ {:.0}°C, {:.0} W",
            get_metric_or(s, "gpu_util_pct", 0.0),
            get_metric_or(s, "gpu_temp_c", 0.0),
            get_metric_or(s, "gpu_power_w", 0.0)
        )
    };
    let pwr = if get_metric_or(s, "on_ac", 1.0) >= 0.5 {
        "AC"
    } else {
        "battery"
    };
    let hf = get_metric_or(s, "hard_faults_ps", 0.0);
    let hf_str = if hf >= 10.0 {
        format!(", {:.0} page faults/s", hf)
    } else {
        String::new()
    };
    format!(
        "CPU {:.0}% (busiest core {:.0}%), RAM {:.0}%, commit {:.0}%{}, disk {} read / {} write, net {} down / {} up, GPU {}, {}",
        get_metric_or(s, "cpu_pct", 0.0),
        get_metric_or(s, "core_max_pct", 0.0),
        get_metric_or(s, "ram_pct", 0.0),
        get_metric_or(s, "commit_pct", 0.0),
        hf_str,
        fmt_bytes(get_metric_or(s, "disk_read_bps", 0.0), true),
        fmt_bytes(get_metric_or(s, "disk_write_bps", 0.0), true),
        fmt_bytes(get_metric_or(s, "net_recv_bps", 0.0), true),
        fmt_bytes(get_metric_or(s, "net_sent_bps", 0.0), true),
        gpu,
        pwr
    )
}

/// Helper to evaluate a `MetricMap` window with `AnomalyModel` if present.
pub fn score_window(model: Option<&AnomalyModel>, w: &mut MetricMap) -> [f64; NUM_FEATURES] {
    let raw_arr = map_to_feature_array(w);
    let x = crate::features::to_matrix_row(&raw_arr);
    if let Some(m) = model {
        let (score, anom) = m.evaluate_row(&x);
        w.insert("score".to_string(), score);
        w.insert("anomalous".to_string(), if anom { 1.0 } else { 0.0 });
    }
    x
}
