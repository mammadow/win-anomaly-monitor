//! Process-level telemetry (`ProcessTracker`) using a single `NtQuerySystemInformation`
//! kernel sweep every `process_interval_sec` seconds, enriched with cached
//! command-line script tags (`python.exe [script.py]`), Windows Service tags
//! (`svchost.exe [SysMain]`), and per-process hard page fault rates.

use crate::sampler::mono_now;
use crate::winapi::{self, RawProc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

pub type ProcKey = (u32, i64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcStats {
    pub key: ProcKey,
    pub name: String,
    pub base_name: String,
    pub cpu_cores: f64,
    pub cpu_pct: f64,
    pub read_bps: f64,
    pub write_bps: f64,
    pub other_bps: f64,
    pub hard_faults_ps: f64,
    pub private_bytes: u64,
    pub working_set: u64,
    pub threads: u32,
    pub handles: u32,
    pub foreground: bool,
}

impl ProcStats {
    pub fn pid(&self) -> u32 {
        self.key.0
    }
}

#[derive(Debug, Clone)]
struct PrevProc {
    mono_t: f64,
    raw: RawProc,
}

pub struct ProcessTracker {
    pub history_sec: f64,
    pub n_cpus: usize,
    buf: Vec<u8>,
    prev: HashMap<ProcKey, PrevProc>,
    cmdline_cache: HashMap<ProcKey, Option<String>>,
    history: VecDeque<(f64, Arc<Vec<ProcStats>>)>,
    pub latest: Arc<Vec<ProcStats>>,
    pub foreground_pid: Option<u32>,
}

impl ProcessTracker {
    pub fn new(history_sec: f64) -> Self {
        let n_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);
        Self {
            history_sec,
            n_cpus,
            buf: vec![0u8; 1024 * 1024],
            prev: HashMap::new(),
            cmdline_cache: HashMap::new(),
            history: VecDeque::new(),
            latest: Arc::new(Vec::new()),
            foreground_pid: None,
        }
    }

    pub fn reset(&mut self) {
        self.prev.clear();
        self.history.clear();
    }

    /// Take one snapshot of all running processes and return per-process rates
    /// since the previous sweep.
    pub fn sweep(&mut self, now: f64) -> Arc<Vec<ProcStats>> {
        let t = mono_now();
        let fg_pid = winapi::foreground_window()
            .map(|h| winapi::window_owner(h).0)
            .filter(|&pid| pid != 0);
        self.foreground_pid = fg_pid;

        let raw_list = winapi::iter_processes(&mut self.buf);
        let svc_map = winapi::svchost_service_map();

        let mut cur_prev = HashMap::with_capacity(raw_list.len());
        let mut stats = Vec::with_capacity(raw_list.len());

        for p in raw_list {
            let key = (p.pid, p.create_time);
            let enriched_name = self.enrich_name(key, &p.name, &svc_map);

            if let Some(prev_entry) = self.prev.get(&key) {
                let dt = (t - prev_entry.mono_t).max(1e-3);
                let prev_p = &prev_entry.raw;
                let cores = ((p.cpu_100ns - prev_p.cpu_100ns).max(0) as f64) / 1e7 / dt;
                let cpu_pct = cores * 100.0 / (self.n_cpus as f64);
                let r_bps = ((p.io_read - prev_p.io_read).max(0) as f64) / dt;
                let w_bps = ((p.io_write - prev_p.io_write).max(0) as f64) / dt;
                let o_bps = ((p.io_other - prev_p.io_other).max(0) as f64) / dt;
                let hf_ps = (p.hard_faults.wrapping_sub(prev_p.hard_faults) as f64) / dt;

                stats.push(ProcStats {
                    key,
                    name: enriched_name,
                    base_name: p.name.clone(),
                    cpu_cores: cores,
                    cpu_pct,
                    read_bps: r_bps,
                    write_bps: w_bps,
                    other_bps: o_bps,
                    hard_faults_ps: hf_ps.max(0.0),
                    private_bytes: p.private_bytes,
                    working_set: p.working_set,
                    threads: p.threads,
                    handles: p.handles,
                    foreground: Some(p.pid) == fg_pid,
                });
            }

            cur_prev.insert(key, PrevProc { mono_t: t, raw: p });
        }

        // Prune dead processes from cmdline cache
        self.cmdline_cache.retain(|k, _| cur_prev.contains_key(k));
        self.prev = cur_prev;
        let arc_stats = Arc::new(stats);
        self.latest = Arc::clone(&arc_stats);

        self.history.push_back((now, Arc::clone(&arc_stats)));
        let cutoff = now - self.history_sec;
        while self.history.len() > 1 {
            if let Some((front_ts, _)) = self.history.front() {
                if *front_ts < cutoff {
                    self.history.pop_front();
                } else {
                    break;
                }
            }
        }
        arc_stats
    }

    fn enrich_name(
        &mut self,
        key: ProcKey,
        raw_name: &str,
        svc_map: &HashMap<u32, String>,
    ) -> String {
        if raw_name.eq_ignore_ascii_case("svchost.exe") {
            if let Some(svc_tag) = svc_map.get(&key.0) {
                return format!("svchost.exe [{}]", svc_tag);
            }
            return raw_name.to_string();
        }
        if winapi::needs_cmdline_tag(raw_name) {
            let tag_opt = self
                .cmdline_cache
                .entry(key)
                .or_insert_with(|| winapi::query_process_cmdline_tag(key.0));
            if let Some(ref tag) = tag_opt {
                if !tag.is_empty() {
                    return format!("{} [{}]", raw_name, tag);
                }
            }
        }
        raw_name.to_string()
    }

    pub fn foreground_name(&self) -> Option<String> {
        let fg = self.foreground_pid?;
        self.name_for_pid(fg)
    }

    pub fn name_for_pid(&self, pid: u32) -> Option<String> {
        for s in self.latest.iter() {
            if s.pid() == pid {
                return Some(s.name.clone());
            }
        }
        for (&(p, _), entry) in &self.prev {
            if p == pid {
                return Some(entry.raw.name.clone());
            }
        }
        winapi::process_image_name(pid)
    }

    /// Cumulative user + kernel CPU seconds for `pid` from the latest sweep.
    pub fn cpu_seconds(&self, pid: u32) -> Option<f64> {
        for (&(p, _), entry) in &self.prev {
            if p == pid {
                return Some((entry.raw.cpu_100ns as f64) / 1e7);
            }
        }
        None
    }

    /// Mean per-executable CPU and I/O over sweeps since `since`.
    /// Multiple processes with the same enriched name are summed so worker pools
    /// show their combined footprint while distinct scripts (`python.exe [a.py]`
    /// vs `python.exe [b.py]`) remain separate.
    pub fn top_activity(&self, since: f64, top_n: usize) -> (Vec<Value>, Vec<Value>) {
        let mut slices: Vec<&[ProcStats]> = self
            .history
            .iter()
            .filter(|(ts, _)| *ts >= since)
            .map(|(_, s)| s.as_slice())
            .collect();
        if slices.is_empty() {
            if let Some((_, last)) = self.history.back() {
                slices.push(last.as_slice());
            }
        }
        if slices.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let n_sweeps = slices.len() as f64;
        #[derive(Default)]
        struct Agg {
            display_name: String,
            cores: f64,
            cpu_pct: f64,
            read_bps: f64,
            write_bps: f64,
            other_bps: f64,
            hard_faults_ps: f64,
            priv_bytes: u64,
            pid: u32,
            peak_cores: f64,
        }

        let mut agg: HashMap<String, Agg> = HashMap::new();
        for sweep in slices {
            for s in sweep {
                let k = s.name.to_ascii_lowercase();
                let b = agg.entry(k).or_insert_with(|| Agg {
                    display_name: s.name.clone(),
                    pid: s.pid(),
                    ..Default::default()
                });
                b.cores += s.cpu_cores / n_sweeps;
                b.cpu_pct += s.cpu_pct / n_sweeps;
                b.read_bps += s.read_bps / n_sweeps;
                b.write_bps += s.write_bps / n_sweeps;
                b.other_bps += s.other_bps / n_sweeps;
                b.hard_faults_ps += s.hard_faults_ps / n_sweeps;
                b.priv_bytes = b.priv_bytes.max(s.private_bytes);
                if s.cpu_cores >= b.peak_cores {
                    b.peak_cores = s.cpu_cores;
                    b.pid = s.pid();
                    b.display_name = s.name.clone();
                }
            }
        }

        let mut rows: Vec<Agg> = agg.into_values().collect();

        // Top CPU (>= 0.05 cores)
        rows.sort_by(|a, b| {
            b.cores
                .partial_cmp(&a.cores)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_cpu: Vec<Value> = rows
            .iter()
            .filter(|r| r.cores >= 0.05)
            .take(top_n)
            .map(|r| {
                let mut obj = json!({
                    "name": r.display_name,
                    "pid": r.pid,
                    "cores": round2(r.cores),
                    "cpu_pct": round1(r.cpu_pct),
                    "private_mb": (r.priv_bytes as f64 / 1048576.0).round() as i64,
                });
                if r.hard_faults_ps >= 1.0 {
                    obj["hard_faults_ps"] = json!(round1(r.hard_faults_ps));
                }
                obj
            })
            .collect();

        // Top I/O (>= 64 KB/s total)
        rows.sort_by(|a, b| {
            let ta = a.read_bps + a.write_bps + a.other_bps;
            let tb = b.read_bps + b.write_bps + b.other_bps;
            tb.partial_cmp(&ta).unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_io: Vec<Value> = rows
            .iter()
            .filter(|r| r.read_bps + r.write_bps + r.other_bps >= 65_536.0)
            .take(top_n)
            .map(|r| {
                let mut obj = json!({
                    "name": r.display_name,
                    "pid": r.pid,
                    "read_mb_s": round2(r.read_bps / 1048576.0),
                    "write_mb_s": round2(r.write_bps / 1048576.0),
                    "other_mb_s": round2(r.other_bps / 1048576.0),
                });
                if r.hard_faults_ps >= 1.0 {
                    obj["hard_faults_ps"] = json!(round1(r.hard_faults_ps));
                }
                obj
            })
            .collect();

        (top_cpu, top_io)
    }

    /// Largest memory consumers right now, grouped by enriched process name.
    pub fn top_memory(&self, top_n: usize) -> Vec<Value> {
        #[derive(Default)]
        struct MemGroup {
            name: String,
            pid: u32,
            max_single_priv: u64,
            private_bytes: u64,
            working_set: u64,
            processes: usize,
            handles: u32,
            hard_faults_ps: f64,
        }

        let mut by_name: HashMap<String, MemGroup> = HashMap::new();
        for s in self.latest.iter() {
            let k = s.name.to_ascii_lowercase();
            let g = by_name.entry(k).or_insert_with(|| MemGroup {
                name: s.name.clone(),
                pid: s.pid(),
                ..Default::default()
            });
            g.private_bytes += s.private_bytes;
            g.working_set += s.working_set;
            g.processes += 1;
            g.handles += s.handles;
            g.hard_faults_ps += s.hard_faults_ps;
            if s.private_bytes >= g.max_single_priv {
                g.max_single_priv = s.private_bytes;
                g.pid = s.pid();
                g.name = s.name.clone();
            }
        }

        let mut groups: Vec<MemGroup> = by_name.into_values().collect();
        groups.sort_by(|a, b| b.private_bytes.cmp(&a.private_bytes));
        groups
            .into_iter()
            .take(top_n)
            .map(|g| {
                let mut obj = json!({
                    "name": g.name,
                    "pid": g.pid,
                    "private_mb": (g.private_bytes as f64 / 1048576.0).round() as i64,
                    "working_set_mb": (g.working_set as f64 / 1048576.0).round() as i64,
                    "processes": g.processes,
                    "handles": g.handles,
                });
                if g.hard_faults_ps >= 1.0 {
                    obj["hard_faults_ps"] = json!(round1(g.hard_faults_ps));
                }
                obj
            })
            .collect()
    }
}

#[inline]
fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

#[inline]
fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
