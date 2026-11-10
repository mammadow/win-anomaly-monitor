//! In-process HighQoS UI & compositor responsiveness probe (`ResponsivenessProbe`)
//! and freeze coalescer (`StallTracker`).
//!
//! Unlike the Python v3 helper process, Rust has no GIL: the three probe threads
//! (`foreground`, `taskbar`, `compositor`) run directly inside the single binary
//! at `THREAD_PRIORITY_HIGHEST` with `ThreadPowerThrottling` HighQoS disabled.

use crate::sampler::{mono_now, unix_now};
use crate::winapi;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// One responsiveness hiccup observed on a single probe tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stall {
    pub ts: f64,
    pub source: String, // "foreground" | "taskbar" | "compositor" | "scheduler"
    pub latency_ms: f64,
    pub timed_out: bool,
    pub hung: bool,
    pub pid: u32,
    pub tid: u32,
    pub process: String,
    pub title: String,
    pub hwnd: usize,
}

/// Coalesced episode of consecutive stalls on the same target `(source, hwnd)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Freeze {
    pub source: String,
    pub ts_start: f64,
    pub ts_end: f64,
    pub max_latency_ms: f64,
    pub timed_out: bool,
    pub hung: bool,
    pub pid: u32,
    pub tid: u32,
    pub process: String,
    pub title: String,
    pub hwnd: usize,
    pub pings: usize,
    pub last_seen: f64,
}

impl Freeze {
    pub fn duration(&self) -> f64 {
        (self.ts_end - self.ts_start).max(self.max_latency_ms / 1000.0)
    }

    pub fn absorb(&mut self, s: &Stall) {
        self.ts_end = self.ts_end.max(s.ts);
        self.last_seen = s.ts;
        self.max_latency_ms = self.max_latency_ms.max(s.latency_ms);
        self.timed_out |= s.timed_out;
        self.hung |= s.hung;
        self.pings += 1;
        if !s.process.is_empty() {
            self.process = s.process.clone();
        }
        if !s.title.is_empty() {
            self.title = s.title.clone();
        }
    }
}

/// Pure state machine that turns raw per-tick `Stall`s into `Freeze` episodes.
pub struct StallTracker {
    pub interval_ms: u64,
    pub ping_timeout_ms: u64,
    pub stall_warn_ms: f64,
    pub active: HashMap<(String, usize), Freeze>,
    pub completed: Vec<Freeze>,
    clocks: Option<(f64, f64)>,
    grace_until: f64,
}

impl StallTracker {
    pub const MERGE_GAP_SEC: f64 = 3.0;
    pub const RESUME_GRACE_SEC: f64 = 5.0;

    pub fn new(interval_ms: u64, ping_timeout_ms: u64, stall_warn_ms: f64) -> Self {
        Self {
            interval_ms,
            ping_timeout_ms,
            stall_warn_ms,
            active: HashMap::new(),
            completed: Vec::new(),
            clocks: None,
            grace_until: 0.0,
        }
    }

    /// Returns `true` if the machine slept between the previous check and now.
    pub fn check_sleep(&mut self, now: f64, clock_pair: (f64, f64)) -> bool {
        let (up, unb) = clock_pair;
        let prev = self.clocks;
        self.clocks = Some((up, unb));
        if let Some((p_up, p_unb)) = prev {
            if (up - p_up) - (unb - p_unb) > 1.0 {
                self.active.clear();
                self.grace_until = now + Self::RESUME_GRACE_SEC;
                return true;
            }
        }
        false
    }

    pub fn check_wakeup(&self, late_wakeup_ms: f64, now: f64) -> Option<Stall> {
        let floor = ((2 * self.interval_ms) as f64).max(self.stall_warn_ms);
        if late_wakeup_ms >= floor {
            Some(Stall {
                ts: now,
                source: "scheduler".to_string(),
                latency_ms: late_wakeup_ms,
                timed_out: late_wakeup_ms >= (self.ping_timeout_ms as f64),
                hung: false,
                pid: 0,
                tid: 0,
                process: String::new(),
                title: String::new(),
                hwnd: 0,
            })
        } else {
            None
        }
    }

    pub fn step(&mut self, stalls: &[Stall], now: f64, clock_pair: Option<(f64, f64)>) {
        if let Some(cp) = clock_pair {
            if self.check_sleep(now, cp) {
                return;
            }
        }
        if now >= self.grace_until {
            for s in stalls {
                let key = (s.source.clone(), s.hwnd);
                let merge_existing = self
                    .active
                    .get(&key)
                    .map(|f| s.ts - f.last_seen <= Self::MERGE_GAP_SEC)
                    .unwrap_or(false);
                if merge_existing {
                    if let Some(f) = self.active.get_mut(&key) {
                        f.absorb(s);
                    }
                } else {
                    if let Some(old) = self.active.remove(&key) {
                        self.completed.push(old);
                    }
                    let start = s.ts - (s.latency_ms / 1000.0);
                    self.active.insert(
                        key,
                        Freeze {
                            source: s.source.clone(),
                            ts_start: start,
                            ts_end: s.ts,
                            max_latency_ms: s.latency_ms,
                            timed_out: s.timed_out,
                            hung: s.hung,
                            pid: s.pid,
                            tid: s.tid,
                            process: s.process.clone(),
                            title: s.title.clone(),
                            hwnd: s.hwnd,
                            pings: 1,
                            last_seen: s.ts,
                        },
                    );
                }
            }
        }

        let expired_keys: Vec<(String, usize)> = self
            .active
            .iter()
            .filter(|(_, f)| now - f.last_seen >= Self::MERGE_GAP_SEC)
            .map(|(k, _)| k.clone())
            .collect();
        for k in expired_keys {
            if let Some(f) = self.active.remove(&k) {
                self.completed.push(f);
            }
        }
    }

    pub fn drain(&mut self) -> (Vec<Freeze>, Vec<Freeze>) {
        let done = std::mem::take(&mut self.completed);
        let ongoing: Vec<Freeze> = self.active.values().cloned().collect();
        (done, ongoing)
    }
}

#[derive(Debug, Clone)]
pub struct ThreadProbeState {
    pub source: String,
    pub ts: f64,
    pub started_at: f64,
    pub busy: bool,
    pub latency_ms: f64,
    pub timed_out: bool,
    pub hung: bool,
    pub late_ms: f64,
    pub pid: u32,
    pub tid: u32,
    pub process: String,
    pub title: String,
    pub hwnd: usize,
}

impl ThreadProbeState {
    fn new(source: &str) -> Self {
        let now = unix_now();
        Self {
            source: source.to_string(),
            ts: now,
            started_at: now,
            busy: false,
            latency_ms: 0.0,
            timed_out: false,
            hung: false,
            late_ms: 0.0,
            pid: 0,
            tid: 0,
            process: String::new(),
            title: String::new(),
            hwnd: 0,
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "source": self.source,
            "ts": self.ts,
            "started_at": self.started_at,
            "busy": self.busy,
            "latency_ms": (self.latency_ms * 10.0).round() / 10.0,
            "timed_out": self.timed_out,
            "hung": self.hung,
            "late_ms": (self.late_ms * 10.0).round() / 10.0,
            "pid": self.pid,
            "tid": self.tid,
            "process": self.process,
            "title": self.title,
            "hwnd": self.hwnd,
        })
    }
}

struct SharedProbeState {
    boosted: bool,
    threads: HashMap<String, ThreadProbeState>,
    pending_stalls: Vec<Stall>,
    pid_names: HashMap<u32, String>,
}

/// HighQoS multi-threaded UI & compositor responsiveness probe.
pub struct ResponsivenessProbe {
    pub interval_ms: u64,
    pub ping_timeout_ms: u64,
    pub stall_warn_ms: f64,
    pub tracker: StallTracker,
    stop: Arc<AtomicBool>,
    shared: Arc<Mutex<SharedProbeState>>,
    handles: Vec<JoinHandle<()>>,
}

impl ResponsivenessProbe {
    pub fn new(interval_ms: u64, ping_timeout_ms: u64, stall_warn_ms: f64) -> Self {
        let mut threads = HashMap::new();
        for src in ["foreground", "taskbar", "compositor"] {
            threads.insert(src.to_string(), ThreadProbeState::new(src));
        }
        Self {
            interval_ms,
            ping_timeout_ms,
            stall_warn_ms,
            tracker: StallTracker::new(interval_ms, ping_timeout_ms, stall_warn_ms),
            stop: Arc::new(AtomicBool::new(false)),
            shared: Arc::new(Mutex::new(SharedProbeState {
                boosted: false,
                threads,
                pending_stalls: Vec::new(),
                pid_names: HashMap::new(),
            })),
            handles: Vec::new(),
        }
    }

    pub fn start(&mut self) {
        if !self.handles.is_empty() {
            return;
        }
        self.stop.store(false, Ordering::SeqCst);
        let boosted = winapi::boost_current_thread();
        if let Ok(mut g) = self.shared.lock() {
            g.boosted = boosted;
        }

        for source in ["foreground", "taskbar", "compositor"] {
            let stop_flag = Arc::clone(&self.stop);
            let shared_ref = Arc::clone(&self.shared);
            let interval_ms = self.interval_ms;
            let timeout_ms = self.ping_timeout_ms as u32;
            let warn_ms = self.stall_warn_ms;
            let src_name = source.to_string();

            let h = thread::Builder::new()
                .name(format!("probe-{}", source))
                .spawn(move || {
                    let b = winapi::boost_current_thread();
                    if b {
                        if let Ok(mut g) = shared_ref.lock() {
                            g.boosted = true;
                        }
                    }
                    probe_worker_loop(
                        &src_name,
                        interval_ms,
                        timeout_ms,
                        warn_ms,
                        stop_flag,
                        shared_ref,
                    );
                });
            if let Ok(handle) = h {
                self.handles.push(handle);
            }
        }
    }

    pub fn close(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Probe threads may be inside SendMessageTimeoutW for up to ping_timeout_ms;
        // detach or join quickly without blocking shutdown if a target window is hung.
        self.handles.clear();
    }

    pub fn snapshot(&self) -> Value {
        if let Ok(g) = self.shared.lock() {
            let mut th_map = serde_json::Map::new();
            for (k, v) in &g.threads {
                th_map.insert(k.clone(), v.to_json());
            }
            json!({
                "alive": !self.handles.is_empty() && !self.stop.load(Ordering::Relaxed),
                "boosted": g.boosted,
                "threads": Value::Object(th_map),
            })
        } else {
            json!({ "alive": false })
        }
    }

    pub fn poll(&mut self, now: f64) -> (Vec<Freeze>, Vec<Freeze>) {
        let mut stalls = Vec::new();
        let mut max_late = 0.0f64;

        if let Ok(mut g) = self.shared.lock() {
            stalls.append(&mut g.pending_stalls);
            for st in g.threads.values() {
                max_late = max_late.max(st.late_ms);
                // Also synthesize an in-flight stall if a thread is currently blocked inside a call
                // for longer than stall_warn_ms
                if st.busy {
                    let blocked_ms = (now - st.started_at).max(0.0) * 1000.0;
                    if blocked_ms >= self.stall_warn_ms {
                        stalls.push(Stall {
                            ts: now,
                            source: st.source.clone(),
                            latency_ms: blocked_ms,
                            timed_out: blocked_ms >= (self.ping_timeout_ms as f64),
                            hung: st.hung,
                            pid: st.pid,
                            tid: st.tid,
                            process: st.process.clone(),
                            title: st.title.clone(),
                            hwnd: st.hwnd,
                        });
                    }
                }
            }
        }

        if let Some(sched) = self.tracker.check_wakeup(max_late, now) {
            stalls.push(sched);
        }
        self.tracker.step(&stalls, now, Some(winapi::sleep_clocks()));
        self.tracker.drain()
    }
}

impl Drop for ResponsivenessProbe {
    fn drop(&mut self) {
        self.close();
    }
}

fn probe_worker_loop(
    source: &str,
    interval_ms: u64,
    timeout_ms: u32,
    warn_ms: f64,
    stop: Arc<AtomicBool>,
    shared: Arc<Mutex<SharedProbeState>>,
) {
    let interval_sec = (interval_ms as f64) / 1000.0;
    let my_pid = winapi::current_pid();
    let mut next_tick = mono_now();

    while !stop.load(Ordering::Relaxed) {
        let now_mono = mono_now();
        let late_ms = ((now_mono - next_tick) * 1000.0).max(0.0);
        let wall_start = unix_now();

        if source == "compositor" {
            if let Ok(mut g) = shared.lock() {
                if let Some(st) = g.threads.get_mut(source) {
                    st.started_at = wall_start;
                    st.busy = true;
                    st.late_ms = late_ms;
                    st.process = "dwm.exe".to_string();
                }
            }
            let t0 = mono_now();
            let hr = winapi::dwm_flush();
            let lat_ms = if hr == 0 {
                (mono_now() - t0) * 1000.0
            } else {
                0.0
            };
            let wall_end = unix_now();
            if let Ok(mut g) = shared.lock() {
                if let Some(st) = g.threads.get_mut(source) {
                    st.ts = wall_end;
                    st.busy = false;
                    st.latency_ms = lat_ms;
                    st.timed_out = lat_ms >= (timeout_ms as f64);
                    st.hung = false;
                }
                if lat_ms >= warn_ms {
                    g.pending_stalls.push(Stall {
                        ts: wall_end,
                        source: source.to_string(),
                        latency_ms: lat_ms,
                        timed_out: lat_ms >= (timeout_ms as f64),
                        hung: false,
                        pid: 0,
                        tid: 0,
                        process: "dwm.exe".to_string(),
                        title: String::new(),
                        hwnd: 0,
                    });
                }
            }
        } else {
            let hwnd_opt = if source == "foreground" {
                winapi::foreground_window()
            } else {
                winapi::taskbar_window()
            };

            if let Some(mut hwnd) = hwnd_opt {
                if let Some(real) = winapi::real_window_from_ghost(hwnd) {
                    hwnd = real;
                }
                let (pid, tid) = winapi::window_owner(hwnd);
                if pid != 0 && pid != my_pid {
                    let title = winapi::window_title(hwnd);
                    let hung_before = winapi::is_hung_window(hwnd);
                    let proc_name = resolve_cached_name(&shared, pid);

                    if let Ok(mut g) = shared.lock() {
                        if let Some(st) = g.threads.get_mut(source) {
                            st.started_at = wall_start;
                            st.busy = true;
                            st.late_ms = late_ms;
                            st.hwnd = hwnd;
                            st.pid = pid;
                            st.tid = tid;
                            st.process = proc_name.clone();
                            st.title = title.clone();
                            st.hung = hung_before;
                        }
                    }

                    let t0 = mono_now();
                    let (ok, err) = winapi::send_null(hwnd, timeout_ms);
                    let mut lat_ms = (mono_now() - t0) * 1000.0;
                    let timed_out = !ok && err == 1460;
                    let hung = hung_before || winapi::is_hung_window(hwnd);
                    let alive = ok || timed_out || winapi::window_owner(hwnd).0 != 0;
                    if !alive {
                        lat_ms = 0.0;
                    }
                    let wall_end = unix_now();

                    if let Ok(mut g) = shared.lock() {
                        if let Some(st) = g.threads.get_mut(source) {
                            st.ts = wall_end;
                            st.busy = false;
                            st.latency_ms = lat_ms;
                            st.timed_out = timed_out && alive;
                            st.hung = hung && alive;
                        }
                        if alive && (lat_ms >= warn_ms || timed_out || hung) {
                            g.pending_stalls.push(Stall {
                                ts: wall_end,
                                source: source.to_string(),
                                latency_ms: lat_ms.max(if hung { warn_ms } else { 0.0 }),
                                timed_out,
                                hung,
                                pid,
                                tid,
                                process: proc_name,
                                title,
                                hwnd,
                            });
                        }
                    }
                } else {
                    mark_idle(&shared, source, wall_start, late_ms);
                }
            } else {
                mark_idle(&shared, source, wall_start, late_ms);
            }
        }

        next_tick += interval_sec;
        let now2 = mono_now();
        if next_tick < now2 {
            next_tick = now2 + interval_sec;
        }
        let sleep_s = (next_tick - now2).max(0.005);
        thread::sleep(Duration::from_secs_f64(sleep_s));
    }
}

fn mark_idle(shared: &Arc<Mutex<SharedProbeState>>, source: &str, ts: f64, late_ms: f64) {
    if let Ok(mut g) = shared.lock() {
        if let Some(st) = g.threads.get_mut(source) {
            st.ts = ts;
            st.busy = false;
            st.latency_ms = 0.0;
            st.timed_out = false;
            st.hung = false;
            st.late_ms = late_ms;
            st.hwnd = 0;
            st.pid = 0;
            st.tid = 0;
            st.process.clear();
            st.title.clear();
        }
    }
}

fn resolve_cached_name(shared: &Arc<Mutex<SharedProbeState>>, pid: u32) -> String {
    if let Ok(g) = shared.lock() {
        if let Some(name) = g.pid_names.get(&pid) {
            return name.clone();
        }
    }
    let name = winapi::process_image_name(pid).unwrap_or_default();
    if !name.is_empty() {
        if let Ok(mut g) = shared.lock() {
            if g.pid_names.len() > 512 {
                g.pid_names.clear();
            }
            g.pid_names.insert(pid, name.clone());
        }
    }
    name
}
