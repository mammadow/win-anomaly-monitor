//! Configuration loading and validation (`config.toml`).
//!
//! All settings have sensible defaults so `resmon` works out of the box with no
//! config file. Supports both v4 keys and legacy v3 `config.example.toml` keys.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Complete runtime configuration for `resmon`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MonitorConfig {
    pub data_dir: PathBuf,

    // Sampling & aggregation
    pub sample_interval_sec: f64,
    pub window_sec: i64,
    pub process_interval_sec: f64,
    pub user_idle_after_sec: f64,

    // GPU polling (Optimus / D3cold safe)
    pub gpu_poll_when_awake_sec: f64,
    pub gpu_poll_when_asleep_sec: f64,

    // Anomaly detector (adaptive LOF-style kNN)
    pub training_days: i64,
    pub min_training_hours: f64,
    pub retrain_interval_min: f64,
    pub max_training_windows: usize,
    pub alert_quantile: f64,
    pub min_consecutive_windows: usize,
    pub episode_gap_windows: usize,
    pub grace_period_sec: f64,

    // Hard safety ceilings
    pub ram_high_pct: f64,
    pub commit_high_pct: f64,
    pub memory_sustain_sec: f64,
    pub gpu_temp_warn_c: f64,
    pub gpu_temp_crit_c: f64,
    pub gpu_temp_sustain_sec: f64,
    pub gpu_battery_wake_sustain_sec: f64,
    pub disk_free_warn_gb: f64,
    pub disk_free_crit_gb: f64,
    pub net_upload_floor_mbps: f64,
    pub net_upload_sustain_sec: f64,

    // UI & compositor responsiveness probe
    pub ui_probe_enabled: bool,
    pub ui_probe_interval_ms: u64,
    pub ui_ping_timeout_ms: u64,
    pub ui_stall_warn_ms: f64,
    pub ui_freeze_sustain_sec: f64,
    pub ui_severe_freeze_sec: f64,
    pub dpc_storm_pct: f64,

    // Process behavior detectors
    pub cmdline_attribution: bool,
    pub background_cpu_cores: f64,
    pub background_cpu_sustain_sec: f64,
    pub leak_history_hours: f64,
    pub leak_min_history_hours: f64,
    pub leak_min_growth_mb: f64,
    pub handle_leak_min_growth: f64,
    pub handle_growth_min_count: f64,
    pub handle_absolute_limit: f64,
    pub ignore_processes: Vec<String>,

    // Notifications & system tray
    pub toast_notifications: bool,
    pub toast_min_severity: String,
    pub notification_cooldown_sec: f64,
    pub tray_enabled: bool,

    // Retention
    pub retain_samples_days: i64,
    pub retain_windows_days: i64,
    pub retain_events_days: i64,

    // AI-assisted forensic diagnostics
    pub ai_provider: String,
    pub ai_endpoint: String,
    pub ai_model: String,
    pub ai_api_key_env: String,
    pub ai_timeout_sec: f64,
    pub ai_context_minutes: i64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("data"),
            sample_interval_sec: 1.0,
            window_sec: 10,
            process_interval_sec: 5.0,
            user_idle_after_sec: 300.0,
            gpu_poll_when_awake_sec: 2.0,
            gpu_poll_when_asleep_sec: 60.0,
            training_days: 14,
            min_training_hours: 1.0,
            retrain_interval_min: 30.0,
            max_training_windows: 30_000,
            alert_quantile: 0.001,
            min_consecutive_windows: 3,
            episode_gap_windows: 3,
            grace_period_sec: 120.0,
            ram_high_pct: 92.0,
            commit_high_pct: 90.0,
            memory_sustain_sec: 60.0,
            gpu_temp_warn_c: 85.0,
            gpu_temp_crit_c: 92.0,
            gpu_temp_sustain_sec: 30.0,
            gpu_battery_wake_sustain_sec: 180.0,
            disk_free_warn_gb: 15.0,
            disk_free_crit_gb: 5.0,
            net_upload_floor_mbps: 5.0,
            net_upload_sustain_sec: 120.0,
            ui_probe_enabled: true,
            ui_probe_interval_ms: 250,
            ui_ping_timeout_ms: 2000,
            ui_stall_warn_ms: 500.0,
            ui_freeze_sustain_sec: 1.0,
            ui_severe_freeze_sec: 5.0,
            dpc_storm_pct: 15.0,
            cmdline_attribution: true,
            background_cpu_cores: 1.5,
            background_cpu_sustain_sec: 600.0,
            leak_history_hours: 6.0,
            leak_min_history_hours: 1.0,
            leak_min_growth_mb: 500.0,
            handle_leak_min_growth: 2000.0,
            handle_growth_min_count: 10_000.0,
            handle_absolute_limit: 50_000.0,
            ignore_processes: Vec::new(),
            toast_notifications: true,
            toast_min_severity: "warning".to_string(),
            notification_cooldown_sec: 600.0,
            tray_enabled: false,
            retain_samples_days: 2,
            retain_windows_days: 30,
            retain_events_days: 180,
            ai_provider: "auto".to_string(),
            ai_endpoint: String::new(),
            ai_model: String::new(),
            ai_api_key_env: "OPENAI_API_KEY".to_string(),
            ai_timeout_sec: 45.0,
            ai_context_minutes: 15,
        }
    }
}

impl MonitorConfig {
    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("metrics.db")
    }

    pub fn events_jsonl_path(&self) -> PathBuf {
        self.data_dir.join("events.jsonl")
    }

    pub fn state_path(&self) -> PathBuf {
        self.data_dir.join("state.json")
    }

    pub fn model_path(&self) -> PathBuf {
        self.data_dir.join("model.bin")
    }

    pub fn lock_path(&self) -> PathBuf {
        self.data_dir.join("monitor.lock")
    }

    /// Validate all configuration parameters, returning `Err(String)` if any invariant is violated.
    pub fn validate(&self) -> Result<(), String> {
        let mut bad = Vec::new();
        if self.sample_interval_sec <= 0.0 {
            bad.push("sample_interval_sec must be > 0".to_string());
        }
        if (self.window_sec as f64) < self.sample_interval_sec {
            bad.push("window_sec must be >= sample_interval_sec".to_string());
        }
        if self.process_interval_sec <= 0.0 {
            bad.push("process_interval_sec must be > 0".to_string());
        }
        if !(0.0 < self.alert_quantile && self.alert_quantile < 0.5) {
            bad.push("alert_quantile must be in (0, 0.5)".to_string());
        }
        if self.min_consecutive_windows < 1 {
            bad.push("min_consecutive_windows must be >= 1".to_string());
        }
        if self.min_training_hours <= 0.0 {
            bad.push("min_training_hours must be > 0".to_string());
        }
        if self.gpu_temp_crit_c <= self.gpu_temp_warn_c {
            bad.push("gpu_temp_crit_c must be > gpu_temp_warn_c".to_string());
        }
        if self.disk_free_crit_gb >= self.disk_free_warn_gb {
            bad.push("disk_free_crit_gb must be < disk_free_warn_gb".to_string());
        }
        if self.ui_probe_interval_ms < 50 {
            bad.push("ui_probe_interval_ms must be >= 50".to_string());
        }
        if (self.ui_ping_timeout_ms as f64) < self.ui_stall_warn_ms {
            bad.push("ui_ping_timeout_ms must be >= ui_stall_warn_ms".to_string());
        }
        if self.ui_severe_freeze_sec <= self.ui_freeze_sustain_sec {
            bad.push("ui_severe_freeze_sec must be > ui_freeze_sustain_sec".to_string());
        }
        if !matches!(
            self.toast_min_severity.as_str(),
            "info" | "warning" | "critical"
        ) {
            bad.push("toast_min_severity must be 'info', 'warning', or 'critical'".to_string());
        }
        if !matches!(
            self.ai_provider.as_str(),
            "auto" | "ollama" | "openai" | "offline"
        ) {
            bad.push("ai_provider must be 'auto', 'ollama', 'openai', or 'offline'".to_string());
        }
        if self.ai_timeout_sec <= 0.0 {
            bad.push("ai_timeout_sec must be > 0".to_string());
        }
        if self.ai_context_minutes < 1 {
            bad.push("ai_context_minutes must be >= 1".to_string());
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(format!("Invalid config.toml: {}", bad.join("; ")))
        }
    }
}

/// Flatten any `[section]` tables in `config.toml` and translate legacy v3 keys.
fn flatten_and_normalize_toml(value: toml::Value) -> toml::map::Map<String, toml::Value> {
    let mut raw = toml::map::Map::new();
    if let toml::Value::Table(tbl) = value {
        for (k, v) in tbl {
            if let toml::Value::Table(sub) = v {
                for (sk, sv) in sub {
                    raw.insert(sk, sv);
                }
            } else {
                raw.insert(k, v);
            }
        }
    }

    let mut out = toml::map::Map::new();
    for (k, v) in raw {
        match k.as_str() {
            "ram_pressure_percent" => {
                out.insert("ram_high_pct".to_string(), v);
            }
            "commit_pressure_percent" => {
                out.insert("commit_high_pct".to_string(), v);
            }
            "memory_pressure_sec" => {
                out.insert("memory_sustain_sec".to_string(), v);
            }
            "low_disk_free_gb" => {
                out.insert("disk_free_warn_gb".to_string(), v);
            }
            "gpu_thermal_sec" => {
                out.insert("gpu_temp_sustain_sec".to_string(), v);
            }
            "background_cpu_minutes" => {
                if let Some(m) = v.as_float().or_else(|| v.as_integer().map(|i| i as f64)) {
                    out.insert(
                        "background_cpu_sustain_sec".to_string(),
                        toml::Value::Float(m * 60.0),
                    );
                }
            }
            "gpu_awake_idle_minutes" => {
                if let Some(m) = v.as_float().or_else(|| v.as_integer().map(|i| i as f64)) {
                    out.insert(
                        "gpu_battery_wake_sustain_sec".to_string(),
                        toml::Value::Float(m * 60.0),
                    );
                }
            }
            "ui_freeze_probe" => {
                out.insert("ui_probe_enabled".to_string(), v);
            }
            "ui_freeze_sec" => {
                out.insert("ui_freeze_sustain_sec".to_string(), v);
            }
            "ui_probe_interval_sec" => {
                if let Some(s) = v.as_float().or_else(|| v.as_integer().map(|i| i as f64)) {
                    out.insert(
                        "ui_probe_interval_ms".to_string(),
                        toml::Value::Integer((s * 1000.0).round() as i64),
                    );
                }
            }
            "notify" => {
                out.insert("toast_notifications".to_string(), v);
            }
            "notify_min_severity" => {
                out.insert("toast_min_severity".to_string(), v);
            }
            "window_retention_days" => {
                if let Some(d) = v.as_integer().or_else(|| v.as_float().map(|f| f as i64)) {
                    out.insert("retain_windows_days".to_string(), toml::Value::Integer(d));
                }
            }
            "event_retention_days" => {
                if let Some(d) = v.as_integer().or_else(|| v.as_float().map(|f| f as i64)) {
                    out.insert("retain_events_days".to_string(), toml::Value::Integer(d));
                }
            }
            "close_after_normal_windows" => {
                out.insert("episode_gap_windows".to_string(), v);
            }
            "ollama_url" | "openai_base_url" => {
                out.insert("ai_endpoint".to_string(), v);
            }
            "ollama_model" | "openai_model" => {
                out.insert("ai_model".to_string(), v);
            }
            // Ignored legacy v3 keys that have no effect in v4
            "raw_retention_hours"
            | "max_event_minutes"
            | "ml_cooldown_min"
            | "ml_mature_hours"
            | "low_disk_free_percent"
            | "gpu_throttle_min_temp_c"
            | "upload_burst_mb"
            | "upload_burst_sec"
            | "openai_api_key"
            | "efficiency_mode"
            | "log_level" => {}
            _ => {
                out.insert(k, v);
            }
        }
    }
    out
}

/// Known configuration field names for typo detection.
fn known_fields() -> HashSet<&'static str> {
    [
        "data_dir",
        "sample_interval_sec",
        "window_sec",
        "process_interval_sec",
        "user_idle_after_sec",
        "gpu_poll_when_awake_sec",
        "gpu_poll_when_asleep_sec",
        "training_days",
        "min_training_hours",
        "retrain_interval_min",
        "max_training_windows",
        "alert_quantile",
        "min_consecutive_windows",
        "episode_gap_windows",
        "grace_period_sec",
        "ram_high_pct",
        "commit_high_pct",
        "memory_sustain_sec",
        "gpu_temp_warn_c",
        "gpu_temp_crit_c",
        "gpu_temp_sustain_sec",
        "gpu_battery_wake_sustain_sec",
        "disk_free_warn_gb",
        "disk_free_crit_gb",
        "net_upload_floor_mbps",
        "net_upload_sustain_sec",
        "ui_probe_enabled",
        "ui_probe_interval_ms",
        "ui_ping_timeout_ms",
        "ui_stall_warn_ms",
        "ui_freeze_sustain_sec",
        "ui_severe_freeze_sec",
        "dpc_storm_pct",
        "cmdline_attribution",
        "background_cpu_cores",
        "background_cpu_sustain_sec",
        "leak_history_hours",
        "leak_min_history_hours",
        "leak_min_growth_mb",
        "handle_leak_min_growth",
        "handle_growth_min_count",
        "handle_absolute_limit",
        "ignore_processes",
        "toast_notifications",
        "toast_min_severity",
        "notification_cooldown_sec",
        "tray_enabled",
        "retain_samples_days",
        "retain_windows_days",
        "retain_events_days",
        "ai_provider",
        "ai_endpoint",
        "ai_model",
        "ai_api_key_env",
        "ai_timeout_sec",
        "ai_context_minutes",
    ]
    .into_iter()
    .collect()
}

/// Load `MonitorConfig` from `path`, or auto-discover `config.toml` / `data/` from CWD
/// or the project directory when invoked globally via `PATH`.
pub fn load_config(path: Option<&Path>) -> Result<MonitorConfig, String> {
    let mut cfg = MonitorConfig::default();

    let (target_config, inferred_data_dir) = if let Some(explicit) = path {
        (explicit.to_path_buf(), None)
    } else {
        let cwd_cfg = PathBuf::from("config.toml");
        let cwd_data = PathBuf::from("data");
        if cwd_cfg.exists()
            || cwd_data.join("metrics.db").exists()
            || cwd_data.join("state.json").exists()
        {
            (cwd_cfg, None)
        } else if let Some(repo_dir) = option_env!("CARGO_MANIFEST_DIR").map(PathBuf::from) {
            let repo_cfg = repo_dir.join("config.toml");
            let repo_data = repo_dir.join("data");
            if repo_cfg.exists() || repo_data.exists() {
                (repo_cfg, Some(repo_data))
            } else {
                (cwd_cfg, None)
            }
        } else {
            (cwd_cfg, None)
        }
    };

    if let Some(d) = inferred_data_dir {
        cfg.data_dir = d;
    }

    if target_config.exists() {
        let content = std::fs::read_to_string(&target_config)
            .map_err(|e| format!("Failed to read {}: {}", target_config.display(), e))?;
        let raw_val: toml::Value = toml::from_str(&content)
            .map_err(|e| format!("Failed to parse {}: {}", target_config.display(), e))?;
        let flat = flatten_and_normalize_toml(raw_val);
        let known = known_fields();
        for key in flat.keys() {
            if !known.contains(key.as_str()) {
                eprintln!(
                    "WARN: Unknown setting in {}: {:?}",
                    target_config.display(),
                    key
                );
            }
        }
        let default_fallback_data = cfg.data_dir.clone();
        let has_data_dir = flat.contains_key("data_dir");
        let flat_val = toml::Value::Table(flat);
        cfg = flat_val
            .try_into()
            .map_err(|e: toml::de::Error| format!("Invalid config values: {}", e))?;
        if !has_data_dir {
            cfg.data_dir = default_fallback_data;
        } else if cfg.data_dir.is_relative() {
            if let Some(parent) = target_config.parent() {
                if !parent.as_os_str().is_empty() && path.is_none() {
                    cfg.data_dir = parent.join(&cfg.data_dir);
                }
            }
        }
    }

    if !cfg.data_dir.exists() {
        let _ = std::fs::create_dir_all(&cfg.data_dir);
    }
    cfg.validate()?;
    Ok(cfg)
}
