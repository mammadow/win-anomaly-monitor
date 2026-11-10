//! SQLite WAL persistence (`Storage`) 100% compatible with `data/metrics.db` (schema v3).

use crate::features::{get_metric, MetricMap, FEATURE_NAMES, NUM_FEATURES};
use crate::sampler::unix_now;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rusqlite::{params_from_iter, Connection, OpenFlags, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const SCHEMA_VERSION: i32 = 3;

pub const SAMPLE_COLUMNS: [&str; 22] = [
    "ts",
    "cpu_pct",
    "core_max_pct",
    "dpc_pct",
    "isr_pct",
    "dpc_max_core_pct",
    "ram_pct",
    "commit_pct",
    "disk_read_bps",
    "disk_write_bps",
    "net_sent_bps",
    "net_recv_bps",
    "disk_free_gb",
    "on_ac",
    "user_active",
    "gpu_awake",
    "gpu_util_pct",
    "gpu_power_w",
    "gpu_temp_c",
    "gpu_vram_pct",
    "gpu_clock_mhz",
    "hard_faults_ps",
];

pub const WINDOW_COLUMNS: [&str; 21] = [
    "ts",
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
    "cpu_p95",
    "disk_free_gb",
    "score",
    "anomalous",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRecord {
    pub id: i64,
    pub ts: f64,
    pub ts_end: f64,
    pub is_open: bool,
    pub kind: String,
    pub severity: String,
    pub title: String,
    pub details: Value,
}

fn clean_opt(v: f64) -> Option<f64> {
    if v.is_finite() {
        Some(v)
    } else {
        None
    }
}

pub struct Storage {
    pub path: PathBuf,
    conn: Connection,
    sample_buf: Vec<Vec<Option<f64>>>,
    window_buf: Vec<Vec<Option<f64>>>,
}

impl Storage {
    pub fn new(path: &Path) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        let conn = Connection::open(path).map_err(|e| format!("SQLite open failed: {}", e))?;
        conn.busy_timeout(Duration::from_secs(10))
            .map_err(|e| e.to_string())?;
        let _ = conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;");

        let mut st = Self {
            path: path.to_path_buf(),
            conn,
            sample_buf: Vec::new(),
            window_buf: Vec::new(),
        };
        st.migrate()?;
        Ok(st)
    }

    /// Open an existing database in read-only mode (no schema migrations or WAL writes).
    pub fn open_readonly(path: &Path) -> Result<Self, String> {
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        let conn = Connection::open_with_flags(path, flags)
            .map_err(|e| format!("SQLite read-only open failed: {}", e))?;
        conn.busy_timeout(Duration::from_secs(10))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            path: path.to_path_buf(),
            conn,
            sample_buf: Vec::new(),
            window_buf: Vec::new(),
        })
    }

    fn migrate(&mut self) -> Result<(), String> {
        let _ = self
            .conn
            .execute_batch("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);");

        let sample_cols: String = SAMPLE_COLUMNS[1..]
            .iter()
            .map(|c| format!("{} REAL", c))
            .collect::<Vec<_>>()
            .join(", ");
        let window_cols: String = WINDOW_COLUMNS[1..]
            .iter()
            .map(|c| format!("{} REAL", c))
            .collect::<Vec<_>>()
            .join(", ");

        let schema = format!(
            "CREATE TABLE IF NOT EXISTS samples (ts REAL PRIMARY KEY, {sample_cols}) WITHOUT ROWID;
             CREATE TABLE IF NOT EXISTS windows (ts REAL PRIMARY KEY, {window_cols}) WITHOUT ROWID;
             CREATE TABLE IF NOT EXISTS events (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 kind TEXT NOT NULL,
                 severity TEXT NOT NULL,
                 ts_start REAL NOT NULL,
                 ts_end REAL,
                 is_open INTEGER NOT NULL DEFAULT 0,
                 title TEXT NOT NULL,
                 details TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS events_by_start ON events(ts_start);
             INSERT OR REPLACE INTO meta VALUES ('schema_version', '{SCHEMA_VERSION}');"
        );
        self.conn
            .execute_batch(&schema)
            .map_err(|e| format!("Schema migration failed: {}", e))?;

        // Additive column migrations on existing v3 `samples` and `windows` tables
        self.ensure_columns("samples", &SAMPLE_COLUMNS[1..]);
        self.ensure_columns("windows", &WINDOW_COLUMNS[1..]);
        Ok(())
    }

    fn ensure_columns(&self, table: &str, cols: &[&str]) {
        let mut have = HashSet::new();
        if let Ok(mut stmt) = self.conn.prepare(&format!("PRAGMA table_info({})", table)) {
            if let Ok(mut rows) = stmt.query([]) {
                while let Ok(Some(r)) = rows.next() {
                    if let Ok(name) = r.get::<_, String>(1) {
                        have.insert(name);
                    }
                }
            }
        }
        for &col in cols {
            if !have.contains(col) {
                let _ = self.conn.execute(
                    &format!("ALTER TABLE {} ADD COLUMN {} REAL", table, col),
                    [],
                );
            }
        }
    }

    pub fn add_sample(&mut self, s: &MetricMap) {
        let row: Vec<Option<f64>> = SAMPLE_COLUMNS
            .iter()
            .map(|c| clean_opt(get_metric(s, c)))
            .collect();
        self.sample_buf.push(row);
    }

    pub fn add_window(&mut self, w: &MetricMap) {
        let row: Vec<Option<f64>> = WINDOW_COLUMNS
            .iter()
            .map(|c| clean_opt(get_metric(w, c)))
            .collect();
        self.window_buf.push(row);
    }

    pub fn flush(&mut self) {
        if self.sample_buf.is_empty() && self.window_buf.is_empty() {
            return;
        }
        if let Ok(tx) = self.conn.transaction() {
            if !self.sample_buf.is_empty() {
                let cols_str = SAMPLE_COLUMNS.join(", ");
                let placeholders = vec!["?"; SAMPLE_COLUMNS.len()].join(", ");
                let sql = format!(
                    "INSERT OR REPLACE INTO samples ({}) VALUES ({})",
                    cols_str, placeholders
                );
                if let Ok(mut stmt) = tx.prepare_cached(&sql) {
                    for row in self.sample_buf.drain(..) {
                        let _ = stmt.execute(params_from_iter(row.iter()));
                    }
                }
            }
            if !self.window_buf.is_empty() {
                let cols_str = WINDOW_COLUMNS.join(", ");
                let placeholders = vec!["?"; WINDOW_COLUMNS.len()].join(", ");
                let sql = format!(
                    "INSERT OR REPLACE INTO windows ({}) VALUES ({})",
                    cols_str, placeholders
                );
                if let Ok(mut stmt) = tx.prepare_cached(&sql) {
                    for row in self.window_buf.drain(..) {
                        let _ = stmt.execute(params_from_iter(row.iter()));
                    }
                }
            }
            let _ = tx.commit();
        }
    }

    pub fn insert_event(
        &mut self,
        ts: f64,
        ts_end: f64,
        kind: &str,
        severity: &str,
        title: &str,
        details: &Value,
    ) -> i64 {
        let det_str = serde_json::to_string(details).unwrap_or_else(|_| "{}".to_string());
        let _ = self.conn.execute(
            "INSERT INTO events(kind, severity, ts_start, ts_end, is_open, title, details) VALUES (?1, ?2, ?3, ?4, 1, ?5, ?6)",
            rusqlite::params![kind, severity, ts, ts_end, title, det_str],
        );
        self.conn.last_insert_rowid()
    }

    pub fn update_event(
        &mut self,
        event_id: i64,
        ts_end: f64,
        is_open: bool,
        severity: &str,
        title: &str,
        details: &Value,
    ) {
        let det_str = serde_json::to_string(details).unwrap_or_else(|_| "{}".to_string());
        let _ = self.conn.execute(
            "UPDATE events SET ts_end=?1, is_open=?2, severity=?3, title=?4, details=?5 WHERE id=?6",
            rusqlite::params![ts_end, i32::from(is_open), severity, title, det_str, event_id],
        );
    }

    /// Thread-safe static helper that opens a read connection to load `(ts, raw)` training windows.
    pub fn load_training_windows_from_path(
        path: &Path,
        days: i64,
        max_rows: usize,
        seed: u64,
    ) -> Option<(Vec<f64>, Vec<[f64; NUM_FEATURES]>)> {
        let cutoff = unix_now() - (days as f64) * 86400.0;
        let cols = FEATURE_NAMES.join(", ");
        let sql = format!(
            "SELECT ts, {} FROM windows WHERE ts >= ?1 ORDER BY ts",
            cols
        );
        let conn = Connection::open(path).ok()?;
        let _ = conn.busy_timeout(Duration::from_secs(10));
        let mut stmt = conn.prepare(&sql).ok()?;
        let mut rows_iter = stmt.query([cutoff]).ok()?;

        let mut all_rows: Vec<(f64, [f64; NUM_FEATURES])> = Vec::new();
        while let Ok(Some(r)) = rows_iter.next() {
            let ts: f64 = r.get::<_, Option<f64>>(0).ok().flatten().unwrap_or(0.0);
            let mut feat = [f64::NAN; NUM_FEATURES];
            for (j, slot) in feat.iter_mut().enumerate() {
                *slot = r
                    .get::<_, Option<f64>>(j + 1)
                    .ok()
                    .flatten()
                    .unwrap_or(f64::NAN);
            }
            all_rows.push((ts, feat));
        }
        if all_rows.is_empty() {
            return None;
        }
        if all_rows.len() > max_rows {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            all_rows.shuffle(&mut rng);
            all_rows.truncate(max_rows);
            all_rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }
        let ts_vec = all_rows.iter().map(|x| x.0).collect();
        let raw_vec = all_rows.into_iter().map(|x| x.1).collect();
        Some((ts_vec, raw_vec))
    }

    pub fn load_training_windows(
        &self,
        days: i64,
        max_rows: usize,
        seed: u64,
    ) -> Option<(Vec<f64>, Vec<[f64; NUM_FEATURES]>)> {
        Self::load_training_windows_from_path(&self.path, days, max_rows, seed)
    }

    /// Load historical windows as dicts (for `replay` / `tune`).
    pub fn load_windows(&self, days: Option<f64>) -> Vec<MetricMap> {
        let cols: Vec<&str> = std::iter::once("ts")
            .chain(FEATURE_NAMES.iter().copied())
            .collect();
        let cols_sql = cols.join(", ");
        let mut out = Vec::new();
        if let Some(d) = days {
            let cutoff = unix_now() - d * 86400.0;
            let sql = format!(
                "SELECT {} FROM windows WHERE ts >= ?1 ORDER BY ts",
                cols_sql
            );
            if let Ok(mut stmt) = self.conn.prepare(&sql) {
                if let Ok(mut rows) = stmt.query([cutoff]) {
                    while let Ok(Some(r)) = rows.next() {
                        out.push(row_to_metric_map(r, &cols));
                    }
                }
            }
        } else {
            let sql = format!("SELECT {} FROM windows ORDER BY ts", cols_sql);
            if let Ok(mut stmt) = self.conn.prepare(&sql) {
                if let Ok(mut rows) = stmt.query([]) {
                    while let Ok(Some(r)) = rows.next() {
                        out.push(row_to_metric_map(r, &cols));
                    }
                }
            }
        }
        out
    }

    pub fn query_events(&self, since: f64, kind: Option<&str>, limit: usize) -> Vec<EventRecord> {
        let mut out = Vec::new();
        if let Some(k) = kind {
            let sql = "SELECT id, ts_start, ts_end, is_open, kind, severity, title, details FROM events WHERE (ts_end >= ?1 OR ts_start >= ?1 OR is_open = 1) AND kind = ?2 ORDER BY ts_start DESC LIMIT ?3";
            if let Ok(mut stmt) = self.conn.prepare(sql) {
                if let Ok(mut rows) = stmt.query(rusqlite::params![since, k, limit as i64]) {
                    while let Ok(Some(r)) = rows.next() {
                        if let Some(ev) = parse_event_row(r) {
                            out.push(ev);
                        }
                    }
                }
            }
        } else {
            let sql = "SELECT id, ts_start, ts_end, is_open, kind, severity, title, details FROM events WHERE (ts_end >= ?1 OR ts_start >= ?1 OR is_open = 1) ORDER BY ts_start DESC LIMIT ?2";
            if let Ok(mut stmt) = self.conn.prepare(sql) {
                if let Ok(mut rows) = stmt.query(rusqlite::params![since, limit as i64]) {
                    while let Ok(Some(r)) = rows.next() {
                        if let Some(ev) = parse_event_row(r) {
                            out.push(ev);
                        }
                    }
                }
            }
        }
        out
    }

    pub fn get_event(&self, event_id: i64) -> Option<EventRecord> {
        let sql = "SELECT id, ts_start, ts_end, is_open, kind, severity, title, details FROM events WHERE id = ?1";
        self.conn
            .query_row(sql, [event_id], |r| Ok(parse_event_row(r)))
            .optional()
            .ok()
            .flatten()
            .flatten()
    }

    pub fn latest_event(&self, kind: Option<&str>) -> Option<EventRecord> {
        if let Some(k) = kind {
            let sql = "SELECT id, ts_start, ts_end, is_open, kind, severity, title, details FROM events WHERE kind = ?1 ORDER BY ts_start DESC LIMIT 1";
            self.conn
                .query_row(sql, [k], |r| Ok(parse_event_row(r)))
                .optional()
                .ok()
                .flatten()
                .flatten()
        } else {
            let sql = "SELECT id, ts_start, ts_end, is_open, kind, severity, title, details FROM events ORDER BY ts_start DESC LIMIT 1";
            self.conn
                .query_row(sql, [], |r| Ok(parse_event_row(r)))
                .optional()
                .ok()
                .flatten()
                .flatten()
        }
    }

    pub fn query_samples(&self, t0: f64, t1: f64, max_rows: usize) -> Vec<MetricMap> {
        let cols_sql = SAMPLE_COLUMNS.join(", ");
        let sql = format!(
            "SELECT {} FROM samples WHERE ts >= ?1 AND ts <= ?2 ORDER BY ts",
            cols_sql
        );
        let mut rows_vec = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare(&sql) {
            if let Ok(mut rows) = stmt.query(rusqlite::params![t0, t1]) {
                while let Ok(Some(r)) = rows.next() {
                    rows_vec.push(row_to_metric_map(r, &SAMPLE_COLUMNS));
                }
            }
        }
        if rows_vec.len() > max_rows && max_rows > 0 {
            let step = (rows_vec.len() / max_rows).max(1);
            rows_vec = rows_vec.into_iter().step_by(step).take(max_rows).collect();
        }
        rows_vec
    }

    pub fn query_windows_range(&self, t0: f64, t1: f64) -> Vec<MetricMap> {
        let cols_sql = WINDOW_COLUMNS.join(", ");
        let sql = format!(
            "SELECT {} FROM windows WHERE ts >= ?1 AND ts <= ?2 ORDER BY ts",
            cols_sql
        );
        let mut out = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare(&sql) {
            if let Ok(mut rows) = stmt.query(rusqlite::params![t0, t1]) {
                while let Ok(Some(r)) = rows.next() {
                    out.push(row_to_metric_map(r, &WINDOW_COLUMNS));
                }
            }
        }
        out
    }

    pub fn prune(&mut self, samples_days: i64, windows_days: i64, events_days: i64) {
        let now = unix_now();
        if let Ok(tx) = self.conn.transaction() {
            let _ = tx.execute(
                "DELETE FROM samples WHERE ts < ?1",
                [now - (samples_days as f64) * 86400.0],
            );
            let _ = tx.execute(
                "DELETE FROM windows WHERE ts < ?1",
                [now - (windows_days as f64) * 86400.0],
            );
            let _ = tx.execute(
                "DELETE FROM events WHERE ts_start < ?1 AND is_open = 0",
                [now - (events_days as f64) * 86400.0],
            );
            let _ = tx.commit();
        }
    }
}

fn row_to_metric_map(r: &rusqlite::Row<'_>, cols: &[&str]) -> MetricMap {
    let mut m = MetricMap::new();
    for (i, &col) in cols.iter().enumerate() {
        let v = r
            .get::<_, Option<f64>>(i)
            .ok()
            .flatten()
            .unwrap_or(f64::NAN);
        m.insert(col.to_string(), v);
    }
    m
}

fn parse_event_row(r: &rusqlite::Row<'_>) -> Option<EventRecord> {
    let id: i64 = r.get(0).ok()?;
    let ts: f64 = r.get(1).ok()?;
    let ts_end: f64 = r.get::<_, Option<f64>>(2).ok().flatten().unwrap_or(ts);
    let is_open: i32 = r.get::<_, Option<i32>>(3).ok().flatten().unwrap_or(0);
    let kind: String = r.get(4).ok()?;
    let severity: String = r.get(5).ok()?;
    let title: String = r.get(6).ok()?;
    let det_str: String = r
        .get::<_, Option<String>>(7)
        .ok()
        .flatten()
        .unwrap_or_default();
    let details: Value = serde_json::from_str(&det_str).unwrap_or_else(|_| json!({}));
    Some(EventRecord {
        id,
        ts,
        ts_end,
        is_open: is_open != 0,
        kind,
        severity,
        title,
        details,
    })
}

/// Read-only summary of `data/metrics.db` for CLI commands (`status`).
pub fn db_summary(path: &Path) -> Value {
    if !path.exists() {
        return json!({
            "exists": false,
            "samples": 0,
            "windows": 0,
            "events": 0,
            "span_hours": 0.0
        });
    }
    let Ok(conn) = Connection::open(path) else {
        return json!({
            "exists": true,
            "samples": 0,
            "windows": 0,
            "events": 0,
            "span_hours": 0.0
        });
    };
    let _ = conn.busy_timeout(Duration::from_secs(5));
    let ns: i64 = conn
        .query_row("SELECT COUNT(*) FROM samples", [], |r| r.get(0))
        .unwrap_or(0);
    let (nw, t0, t1): (i64, Option<f64>, Option<f64>) = conn
        .query_row("SELECT COUNT(*), MIN(ts), MAX(ts) FROM windows", [], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .unwrap_or((0, None, None));
    let ne: i64 = conn
        .query_row("SELECT COUNT(*) FROM events", [], |r| r.get(0))
        .unwrap_or(0);
    let span = match (t0, t1) {
        (Some(a), Some(b)) if b >= a => (b - a) / 3600.0,
        _ => 0.0,
    };
    json!({
        "exists": true,
        "samples": ns,
        "windows": nw,
        "events": ne,
        "span_hours": (span * 10.0).round() / 10.0,
    })
}
