//! Event lifecycle (`Event`, `EventSink`, `EventManager`), JSONL logging,
//! and non-blocking Windows 11 toast notifications (`Notifier`).

use crate::config::MonitorConfig;
use crate::sampler::mono_now;
use crate::storage::{EventRecord, Storage};
use chrono::{Local, TimeZone};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

pub fn severity_rank(sev: &str) -> i32 {
    match sev {
        "critical" => 2,
        "warning" => 1,
        _ => 0,
    }
}

pub fn fmt_duration(sec: f64) -> String {
    let sec = sec.max(0.0);
    if sec < 60.0 {
        format!("{:.0}s", sec)
    } else if sec < 3600.0 {
        format!("{:.1} min", sec / 60.0)
    } else {
        format!("{:.1} h", sec / 3600.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: i64,
    pub kind: String,
    pub severity: String,
    pub title: String,
    pub ts_start: f64,
    pub ts_end: f64,
    pub details: Value,
    pub is_open: bool,
}

impl Event {
    pub fn duration(&self) -> f64 {
        (self.ts_end - self.ts_start).max(0.0)
    }

    pub fn to_dict(&self) -> Value {
        let iso = Local
            .timestamp_opt(self.ts_start as i64, 0)
            .single()
            .map(|dt| dt.format("%Y-%m-%dT%H:%M:%S").to_string())
            .unwrap_or_default();
        json!({
            "id": self.id,
            "time": iso,
            "ts": (self.ts_start * 100.0).round() / 100.0,
            "ts_end": (self.ts_end * 100.0).round() / 100.0,
            "duration_sec": (self.duration() * 10.0).round() / 10.0,
            "open": self.is_open,
            "kind": self.kind,
            "severity": self.severity,
            "title": self.title,
            "details": self.details,
        })
    }
}

pub fn merge_details(target: &mut Value, extra: Value) {
    if let (Value::Object(ref mut t_map), Value::Object(e_map)) = (target, extra) {
        for (k, v) in e_map {
            t_map.insert(k, v);
        }
    }
}

/// Trait abstracting event recording so both live `EventManager` and offline `replay`
/// share identical episode logic.
pub trait EventSink {
    fn open(
        &mut self,
        kind: &str,
        severity: &str,
        title: &str,
        ts: f64,
        details: Value,
        notify: bool,
    ) -> Event;

    fn update(
        &mut self,
        ev: &mut Event,
        ts: f64,
        title: Option<&str>,
        severity: Option<&str>,
        details: Option<Value>,
        notify_on_escalate: bool,
    );

    fn reopen(&mut self, ev: &mut Event, ts: f64);

    fn close(
        &mut self,
        ev: &mut Event,
        ts: f64,
        title: Option<&str>,
        severity: Option<&str>,
        details: Option<Value>,
    );

    fn point(&mut self, kind: &str, severity: &str, title: &str, ts: f64, details: Value) -> Event;
}

/// Rate-limited Windows balloon/toast notifier via native Win32 `Shell_NotifyIconW` (`NIF_INFO`).
pub struct Notifier {
    pub enabled: bool,
    pub min_rank: i32,
    pub cooldown_sec: f64,
    pub paused_until: f64,
    last_sent: HashMap<String, f64>,
}

impl Notifier {
    pub fn new(enabled: bool, min_severity: &str, cooldown_sec: f64) -> Self {
        Self {
            enabled,
            min_rank: severity_rank(min_severity),
            cooldown_sec,
            paused_until: 0.0,
            last_sent: HashMap::new(),
        }
    }

    pub fn send(&mut self, ev: &Event) {
        if !self.enabled || severity_rank(&ev.severity) < self.min_rank {
            return;
        }
        let now = mono_now();
        if now < self.paused_until {
            return;
        }
        if let Some(&last) = self.last_sent.get(&ev.kind) {
            if now - last < self.cooldown_sec {
                return;
            }
        }
        self.last_sent.insert(ev.kind.clone(), now);
        let title = format!(
            "Resource Monitor — {} ({})",
            ev.kind.replace('_', " "),
            ev.severity
        );
        crate::tray::show_balloon_notification(&title, &ev.title, &ev.severity);
    }
}

pub struct EventManager {
    pub jsonl_path: PathBuf,
    pub notifier: Notifier,
    pub open_events: HashMap<i64, Event>,
    next_fallback_id: i64,
}

impl EventManager {
    pub fn new(cfg: &MonitorConfig) -> Self {
        Self {
            jsonl_path: cfg.events_jsonl_path(),
            notifier: Notifier::new(
                cfg.toast_notifications,
                &cfg.toast_min_severity,
                cfg.notification_cooldown_sec,
            ),
            open_events: HashMap::new(),
            next_fallback_id: 1,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open_with_storage(
        &mut self,
        storage: &mut Storage,
        kind: &str,
        severity: &str,
        title: &str,
        ts: f64,
        details: Value,
        notify: bool,
    ) -> Event {
        let id = storage.insert_event(ts, ts, kind, severity, title, &details);
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
        self.open_events.insert(id, ev.clone());
        self.write_jsonl(&ev);
        if notify {
            self.notifier.send(&ev);
        }
        ev
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_with_storage(
        &mut self,
        storage: &mut Storage,
        ev: &mut Event,
        ts: f64,
        title: Option<&str>,
        severity: Option<&str>,
        details: Option<Value>,
        notify_on_escalate: bool,
    ) {
        ev.ts_end = ev.ts_end.max(ts);
        if let Some(t) = title {
            ev.title = t.to_string();
        }
        if let Some(d) = details {
            merge_details(&mut ev.details, d);
        }
        let mut escalated = false;
        if let Some(sev) = severity {
            if severity_rank(sev) > severity_rank(&ev.severity) {
                ev.severity = sev.to_string();
                escalated = true;
            }
        }
        storage.update_event(
            ev.id,
            ev.ts_end,
            ev.is_open,
            &ev.severity,
            &ev.title,
            &ev.details,
        );
        if ev.is_open {
            self.open_events.insert(ev.id, ev.clone());
        }
        if escalated && notify_on_escalate {
            self.notifier.send(ev);
        }
    }

    pub fn reopen_with_storage(&mut self, storage: &mut Storage, ev: &mut Event, ts: f64) {
        ev.is_open = true;
        ev.ts_end = ev.ts_end.max(ts);
        let rec = ev
            .details
            .get("recurrences")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            + 1;
        merge_details(&mut ev.details, json!({ "recurrences": rec }));
        storage.update_event(
            ev.id,
            ev.ts_end,
            ev.is_open,
            &ev.severity,
            &ev.title,
            &ev.details,
        );
        self.open_events.insert(ev.id, ev.clone());
    }

    pub fn close_with_storage(
        &mut self,
        storage: &mut Storage,
        ev: &mut Event,
        ts: f64,
        title: Option<&str>,
        severity: Option<&str>,
        details: Option<Value>,
    ) {
        ev.is_open = false;
        self.update_with_storage(storage, ev, ts, title, severity, details, false);
        self.open_events.remove(&ev.id);
        self.write_jsonl(ev);
    }

    pub fn point_with_storage(
        &mut self,
        storage: &mut Storage,
        kind: &str,
        severity: &str,
        title: &str,
        ts: f64,
        details: Value,
    ) -> Event {
        let mut ev = self.open_with_storage(storage, kind, severity, title, ts, details, true);
        ev.is_open = false;
        storage.update_event(
            ev.id,
            ev.ts_end,
            false,
            &ev.severity,
            &ev.title,
            &ev.details,
        );
        self.open_events.remove(&ev.id);
        ev
    }

    pub fn recent(&self, storage: &Storage, kinds: &[&str], days: f64) -> Vec<EventRecord> {
        let since = crate::sampler::unix_now() - days * 86400.0;
        let rows = storage.query_events(since, None, 500);
        rows.into_iter()
            .filter(|r| kinds.contains(&r.kind.as_str()))
            .collect()
    }

    fn write_jsonl(&self, ev: &Event) {
        if let Some(parent) = self.jsonl_path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        if let Ok(mut f) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.jsonl_path)
        {
            if let Ok(line) = serde_json::to_string(&ev.to_dict()) {
                let _ = writeln!(f, "{}", line);
            }
        }
    }
}

/// Bundled `(&mut Storage, &mut EventManager)` implementing `EventSink`.
pub struct LiveEventSink<'a> {
    pub storage: &'a mut Storage,
    pub events: &'a mut EventManager,
}

impl<'a> EventSink for LiveEventSink<'a> {
    fn open(
        &mut self,
        kind: &str,
        severity: &str,
        title: &str,
        ts: f64,
        details: Value,
        notify: bool,
    ) -> Event {
        self.events
            .open_with_storage(self.storage, kind, severity, title, ts, details, notify)
    }

    fn update(
        &mut self,
        ev: &mut Event,
        ts: f64,
        title: Option<&str>,
        severity: Option<&str>,
        details: Option<Value>,
        notify_on_escalate: bool,
    ) {
        self.events.update_with_storage(
            self.storage,
            ev,
            ts,
            title,
            severity,
            details,
            notify_on_escalate,
        );
    }

    fn reopen(&mut self, ev: &mut Event, ts: f64) {
        self.events.reopen_with_storage(self.storage, ev, ts);
    }

    fn close(
        &mut self,
        ev: &mut Event,
        ts: f64,
        title: Option<&str>,
        severity: Option<&str>,
        details: Option<Value>,
    ) {
        self.events
            .close_with_storage(self.storage, ev, ts, title, severity, details);
    }

    fn point(&mut self, kind: &str, severity: &str, title: &str, ts: f64, details: Value) -> Event {
        self.events
            .point_with_storage(self.storage, kind, severity, title, ts, details)
    }
}

impl EventSink for EventManager {
    fn open(
        &mut self,
        kind: &str,
        severity: &str,
        title: &str,
        ts: f64,
        details: Value,
        notify: bool,
    ) -> Event {
        let id = self.next_fallback_id;
        self.next_fallback_id += 1;
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
        self.open_events.insert(id, ev.clone());
        if notify {
            self.notifier.send(&ev);
        }
        ev
    }

    fn update(
        &mut self,
        ev: &mut Event,
        ts: f64,
        title: Option<&str>,
        severity: Option<&str>,
        details: Option<Value>,
        notify_on_escalate: bool,
    ) {
        ev.ts_end = ev.ts_end.max(ts);
        if let Some(t) = title {
            ev.title = t.to_string();
        }
        if let Some(d) = details {
            merge_details(&mut ev.details, d);
        }
        if let Some(sev) = severity {
            if severity_rank(sev) > severity_rank(&ev.severity) {
                ev.severity = sev.to_string();
                if notify_on_escalate {
                    self.notifier.send(ev);
                }
            }
        }
        if ev.is_open {
            self.open_events.insert(ev.id, ev.clone());
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
        self.open_events.insert(ev.id, ev.clone());
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
        self.open_events.remove(&ev.id);
    }

    fn point(&mut self, kind: &str, severity: &str, title: &str, ts: f64, details: Value) -> Event {
        let mut ev = self.open(kind, severity, title, ts, details, true);
        ev.is_open = false;
        self.open_events.remove(&ev.id);
        ev
    }
}
