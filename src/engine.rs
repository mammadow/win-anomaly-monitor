//! Main 1 Hz monitoring engine (`Monitor`), single-instance lock (`InstanceLock`),
//! and atomic `state.json` writer.

use crate::config::MonitorConfig;
use crate::detector::{AnomalyModel, Trainer};
use crate::events::{EventManager, EventSink, LiveEventSink};
use crate::features::{get_metric_or, MetricMap, WindowAggregator};
use crate::processes::ProcessTracker;
use crate::responsiveness::ResponsivenessProbe;
use crate::rules::{summarize_rates, MonitorContext, RuleEngine};
use crate::sampler::{mono_now, unix_now, SystemSampler};
use crate::storage::Storage;
use crate::tray::TrayController;
use crate::winapi;
use chrono::{Local, TimeZone};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Prevents two monitor daemons from writing to the same `data/` directory.
pub struct InstanceLock {
    pub path: PathBuf,
    held: bool,
}

impl InstanceLock {
    pub fn new(path: PathBuf) -> Self {
        Self { path, held: false }
    }

    pub fn acquire(&mut self) -> Result<(), String> {
        if self.path.exists() {
            if let Ok(txt) = std::fs::read_to_string(&self.path) {
                if let Ok(pid) = txt.trim().parse::<u32>() {
                    if pid != winapi::current_pid() && winapi::pid_exists(pid) {
                        return Err(format!(
                            "Monitor is already running (PID {}). Stop it first or inspect with `resmon status`.",
                            pid
                        ));
                    }
                }
            }
        }
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        std::fs::write(&self.path, winapi::current_pid().to_string())
            .map_err(|e| format!("Failed to write lock file {}: {}", self.path.display(), e))?;
        self.held = true;
        Ok(())
    }

    pub fn release(&mut self) {
        if self.held {
            let _ = std::fs::remove_file(&self.path);
            self.held = false;
        }
    }
}

impl Drop for InstanceLock {
    fn drop(&mut self) {
        self.release();
    }
}

/// Core state accessible to detectors via `MonitorContext`.
pub struct MonitorCore {
    pub cfg: MonitorConfig,
    pub storage: Storage,
    pub sampler: SystemSampler,
    pub aggregator: WindowAggregator,
    pub procs: ProcessTracker,
    pub events: EventManager,
    pub trainer: Trainer,
    pub model: Option<AnomalyModel>,
    pub ui_probe: Option<ResponsivenessProbe>,
    pub last_sample: Option<MetricMap>,
}

impl MonitorContext for MonitorCore {
    fn cfg(&self) -> &MonitorConfig {
        &self.cfg
    }

    fn model(&self) -> Option<&AnomalyModel> {
        self.model.as_ref()
    }

    fn activity(&self, since: f64) -> Value {
        let (top_cpu, top_io) = self.procs.top_activity(since, 4);
        let top_mem = self.procs.top_memory(5);
        let mut gpu_clients = Vec::new();
        for pid in self.sampler.gpu.client_pids() {
            let name = self
                .procs
                .name_for_pid(pid)
                .unwrap_or_else(|| format!("PID {}", pid));
            if !gpu_clients.contains(&name) {
                gpu_clients.push(name);
            }
        }
        let user_idle = self
            .last_sample
            .as_ref()
            .map(|s| get_metric_or(s, "user_active", 1.0) < 0.5)
            .unwrap_or(false);
        json!({
            "top_cpu": top_cpu,
            "top_io": top_io,
            "top_memory": top_mem,
            "gpu_clients": gpu_clients,
            "foreground": self.procs.foreground_name(),
            "user_idle": user_idle,
        })
    }

    fn cpu_seconds(&self, pid: u32) -> Option<f64> {
        self.procs.cpu_seconds(pid)
    }

    fn name_for_pid(&self, pid: u32) -> Option<String> {
        self.procs.name_for_pid(pid)
    }

    fn foreground_pid(&self) -> Option<u32> {
        self.procs.foreground_pid
    }

    fn with_events<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut dyn EventSink) -> R,
    {
        let mut sink = LiveEventSink {
            storage: &mut self.storage,
            events: &mut self.events,
        };
        f(&mut sink)
    }
}

pub struct Monitor {
    pub core: MonitorCore,
    pub rules: RuleEngine,
    pub stop: Arc<AtomicBool>,
    tray: Option<TrayController>,
    started_at: f64,
    grace_until: f64,
    last_tick_ts: Option<f64>,
    next_proc_sweep: f64,
    next_flush: f64,
    next_train: f64,
    next_prune: f64,
}

impl Monitor {
    pub const FLUSH_EVERY_SEC: f64 = 10.0;
    pub const PRUNE_EVERY_SEC: f64 = 3600.0;

    pub fn new(cfg: MonitorConfig) -> Result<Self, String> {
        let storage = Storage::new(&cfg.db_path())?;
        let sampler = SystemSampler::new(&cfg);
        let aggregator = WindowAggregator::new(cfg.window_sec);
        let proc_hist_sec = (cfg.window_sec as f64 * 6.0).max(120.0);
        let mut procs = ProcessTracker::new(proc_hist_sec);
        procs.cmdline_attribution = cfg.cmdline_attribution;
        let events = EventManager::new(&cfg);
        let trainer = Trainer::new(&cfg.data_dir);
        let model = trainer.load_saved(cfg.retrain_interval_min * 60.0 * 4.0);

        let recent_leaks = events.recent(&storage, &["memory_leak", "handle_leak"], 30.0);
        let rules = RuleEngine::new(&recent_leaks);

        let ui_probe = if cfg.ui_probe_enabled {
            Some(ResponsivenessProbe::new(
                cfg.ui_probe_interval_ms,
                cfg.ui_ping_timeout_ms,
                cfg.ui_stall_warn_ms,
            ))
        } else {
            None
        };

        let now = unix_now();
        let grace_until = now + cfg.grace_period_sec;
        let next_train = match model {
            Some(ref m) => m.trained_at + cfg.retrain_interval_min * 60.0,
            None => 0.0,
        };

        Ok(Self {
            core: MonitorCore {
                cfg,
                storage,
                sampler,
                aggregator,
                procs,
                events,
                trainer,
                model,
                ui_probe,
                last_sample: None,
            },
            rules,
            stop: Arc::new(AtomicBool::new(false)),
            tray: None,
            started_at: now,
            grace_until,
            last_tick_ts: None,
            next_proc_sweep: 0.0,
            next_flush: 0.0,
            next_train,
            next_prune: 0.0,
        })
    }

    pub fn run(&mut self, enable_tray: bool) -> Result<(), String> {
        let mut lock = InstanceLock::new(self.core.cfg.lock_path());
        lock.acquire()?;
        winapi::install_ctrl_handler(Arc::clone(&self.stop));
        // Keep the 1 Hz sampler thread in Windows Efficiency Mode (EcoQoS) so foreground
        // work and HighQoS `probe-*` UI responsiveness threads keep P-core priority.
        let _ = winapi::set_current_thread_eco_qos();

        if enable_tray || self.core.cfg.tray_enabled {
            self.tray = Some(TrayController::start(Arc::clone(&self.stop)));
        }

        let now = unix_now();
        self.started_at = now;
        self.grace_until = now + self.core.cfg.grace_period_sec;
        let _ = self.core.procs.sweep(now);
        if let Some(ref mut probe) = self.core.ui_probe {
            probe.start();
        }
        if self.core.model.is_none() || now >= self.next_train {
            self.maybe_train(now);
        }

        let interval = self.core.cfg.sample_interval_sec;
        let mut next_t = mono_now();

        while !self.stop.load(Ordering::Relaxed) {
            self.tick();
            next_t += interval;
            let sleep_for = next_t - mono_now();
            if sleep_for > 0.0 {
                std::thread::sleep(Duration::from_secs_f64(sleep_for));
            } else {
                next_t = mono_now();
            }
        }

        self.shutdown();
        lock.release();
        Ok(())
    }

    pub fn tick(&mut self) {
        if let Ok(Some(fresh)) = self.core.trainer.poll() {
            self.core.model = Some(fresh);
        }

        if let Some(ref tray) = self.tray {
            self.core.events.notifier.paused_until = tray.paused_until();
        }

        let s = self.core.sampler.sample();
        let now = get_metric_or(&s, "ts", unix_now());

        if let Some(last_ts) = self.last_tick_ts {
            let sleep_gap = (5.0 * self.core.cfg.sample_interval_sec).max(30.0);
            if now - last_ts > sleep_gap {
                self.core.sampler.prime();
                self.core.aggregator.reset();
                self.core.procs.reset();
                self.rules.reset();
                self.grace_until = now + self.core.cfg.grace_period_sec;
                self.last_tick_ts = Some(now);
                return;
            }
        }
        self.last_tick_ts = Some(now);
        self.core.last_sample = Some(s.clone());

        if now >= self.next_proc_sweep {
            self.next_proc_sweep = now + self.core.cfg.process_interval_sec;
            let stats = self.core.procs.sweep(now);
            self.rules.on_sweep(&mut self.core, &stats, now);
        }

        self.core.storage.add_sample(&s);
        self.rules.on_sample(&mut self.core, &s, now);

        if let Some(ref mut probe) = self.core.ui_probe {
            let (completed, ongoing) = probe.poll(now);
            self.rules
                .on_freezes(&mut self.core, &completed, &ongoing, now);
        }

        if let Some(mut cw) = self.core.aggregator.add(s.clone()) {
            let mut x_opt = None;
            if let Some(ref model) = self.core.model {
                if now >= self.grace_until {
                    let (score, anom) = model.evaluate_row(&cw.ml_features);
                    cw.row.insert("score".to_string(), score);
                    cw.row
                        .insert("anomalous".to_string(), if anom { 1.0 } else { 0.0 });
                    x_opt = Some(cw.ml_features);
                }
            }
            self.core.storage.add_window(&cw.row);
            self.rules
                .on_window(&mut self.core, &cw.row, x_opt.as_ref(), now);
        }

        if now >= self.next_flush {
            self.next_flush = now + Self::FLUSH_EVERY_SEC;
            self.core.storage.flush();
            self.write_state(&s, now);
            if let Some(ref tray) = self.tray {
                let tip = format!(
                    "resmon v4.0 — CPU {:.0}%, RAM {:.0}%, Commit {:.0}% ({} active alerts)",
                    get_metric_or(&s, "cpu_pct", 0.0),
                    get_metric_or(&s, "ram_pct", 0.0),
                    get_metric_or(&s, "commit_pct", 0.0),
                    self.core.events.open_events.len()
                );
                tray.set_tooltip(&tip);
            }
        }

        if now >= self.next_train {
            self.maybe_train(now);
        }

        if now >= self.next_prune {
            self.next_prune = now + Self::PRUNE_EVERY_SEC;
            self.core.storage.prune(
                self.core.cfg.retain_samples_days,
                self.core.cfg.retain_windows_days,
                self.core.cfg.retain_events_days,
            );
        }
    }

    fn maybe_train(&mut self, now: f64) {
        let min_windows = ((self.core.cfg.min_training_hours * 3600.0)
            / (self.core.cfg.window_sec as f64))
            .max(1.0) as usize;
        let db_path = self.core.storage.path.clone();
        let days = self.core.cfg.training_days;
        let max_rows = self.core.cfg.max_training_windows;
        let q = self.core.cfg.alert_quantile;
        let win_sec = self.core.cfg.window_sec as f64;

        let load_fn = move || {
            let (ts, raw) = Storage::load_training_windows_from_path(&db_path, days, max_rows, 0)?;
            if raw.len() < min_windows {
                return None;
            }
            Some((ts, raw))
        };

        self.core.trainer.start(load_fn, q, win_sec);
        let interval = if self.core.model.is_some() {
            self.core.cfg.retrain_interval_min * 60.0
        } else {
            300.0
        };
        self.next_train = now + interval;
    }

    pub fn shutdown(&mut self) {
        if let Some(ref mut probe) = self.core.ui_probe {
            probe.close();
        }
        if let Some(ref mut tray) = self.tray {
            tray.close();
        }
        let now = unix_now();
        let open_list: Vec<_> = self.core.events.open_events.values().cloned().collect();
        for mut ev in open_list {
            self.core.events.close_with_storage(
                &mut self.core.storage,
                &mut ev,
                now,
                None,
                None,
                None,
            );
        }
        self.core.storage.flush();
        self.core.trainer.join();
        if let Some(ref sample) = self.core.last_sample {
            self.write_state_with_stopped(sample, now, true);
        }
    }

    pub fn write_state(&self, sample: &MetricMap, now: f64) {
        self.write_state_with_stopped(sample, now, false);
    }

    fn write_state_with_stopped(&self, sample: &MetricMap, now: f64, stopped: bool) {
        let model_info = if let Some(ref m) = self.core.model {
            json!({
                "ready": true,
                "windows": m.n_windows,
                "hours_of_data": (m.hours_of_data * 10.0).round() / 10.0,
                "threshold": (m.threshold * 100.0).round() / 100.0,
                "severe_threshold": (m.severe_threshold * 100.0).round() / 100.0,
                "trained_ago_min": ((now - m.trained_at) / 60.0 * 10.0).round() / 10.0,
                "training": self.core.trainer.busy(),
            })
        } else {
            json!({
                "ready": false,
                "min_training_hours": self.core.cfg.min_training_hours,
                "training": self.core.trainer.busy(),
            })
        };

        let mut clean_sample = serde_json::Map::new();
        for (k, &v) in sample {
            if v.is_finite() {
                clean_sample.insert(k.clone(), json!((v * 100.0).round() / 100.0));
            } else {
                clean_sample.insert(k.clone(), Value::Null);
            }
        }

        let open_evs: Vec<Value> = self
            .core
            .events
            .open_events
            .values()
            .map(|e| e.to_dict())
            .collect();
        let resp_snap = self
            .core
            .ui_probe
            .as_ref()
            .map(|p| p.snapshot())
            .unwrap_or_else(|| json!({ "enabled": false }));

        let updated_iso = Local
            .timestamp_opt(now as i64, 0)
            .single()
            .map(|dt| dt.format("%Y-%m-%dT%H:%M:%S").to_string())
            .unwrap_or_default();

        let state = json!({
            "pid": winapi::current_pid(),
            "engine": "rust-v4",
            "stopped": stopped,
            "started_at": self.started_at,
            "updated_at": now,
            "updated_iso": updated_iso,
            "in_grace_period": now < self.grace_until,
            "summary": summarize_rates(sample),
            "sample": Value::Object(clean_sample),
            "activity": self.core.activity(now - 30.0),
            "responsiveness": resp_snap,
            "model": model_info,
            "open_events": open_evs,
        });

        let p = self.core.cfg.state_path();
        let tmp = p.with_extension("json.tmp");
        if let Ok(bytes) = serde_json::to_vec_pretty(&state) {
            if std::fs::write(&tmp, bytes).is_ok() {
                let _ = std::fs::rename(&tmp, &p);
            }
        }
    }
}

pub fn read_state_file(path: &Path) -> Option<Value> {
    let txt = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&txt).ok()
}
