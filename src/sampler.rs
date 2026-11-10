//! 1 Hz system-wide telemetry sampler (`SystemSampler`) and Optimus-aware
//! NVIDIA GPU sampler (`GpuSampler`).

use crate::config::MonitorConfig;
use crate::features::MetricMap;
use crate::winapi::{
    self, CoreTickTimes, NvmlClient, NvmlTelemetry, PdhDiskSampler, SystemPerfCounters,
};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DISK_FREE_TTL_SEC: f64 = 60.0;

/// Current Unix timestamp in seconds (`f64`).
#[inline]
pub fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Monotonic clock in seconds since an arbitrary origin.
#[inline]
pub fn mono_now() -> f64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// Polls NVIDIA GPU metrics via dynamically loaded `nvml.dll` without waking
/// an Optimus laptop dGPU from D3cold.
pub struct GpuSampler {
    client: Option<NvmlClient>,
    awake_interval: f64,
    asleep_interval: f64,
    next_poll: f64,
    next_retry_init: f64,
    last: NvmlTelemetry,
}

impl GpuSampler {
    const RETRY_INIT_SEC: f64 = 300.0;

    pub fn new(poll_when_awake_sec: f64, poll_when_asleep_sec: f64) -> Self {
        let client = NvmlClient::try_new();
        let last = if client.is_some() {
            NvmlTelemetry::asleep()
        } else {
            NvmlTelemetry::absent()
        };
        let next_retry_init = if client.is_some() {
            0.0
        } else {
            mono_now() + Self::RETRY_INIT_SEC
        };
        Self {
            client,
            awake_interval: poll_when_awake_sec,
            asleep_interval: poll_when_asleep_sec,
            next_poll: 0.0,
            next_retry_init,
            last,
        }
    }

    pub fn available(&self) -> bool {
        self.client.is_some()
    }

    pub fn sample(&mut self, now: f64) -> NvmlTelemetry {
        if self.client.is_none() {
            if now >= self.next_retry_init {
                self.client = NvmlClient::try_new();
                if self.client.is_some() {
                    self.last = NvmlTelemetry::asleep();
                    self.next_poll = 0.0;
                } else {
                    self.next_retry_init = now + Self::RETRY_INIT_SEC;
                    return NvmlTelemetry::absent();
                }
            } else {
                return self.last;
            }
        }
        let Some(ref client) = self.client else {
            return self.last;
        };
        if now < self.next_poll {
            return self.last;
        }
        let t = client.query_telemetry();
        let interval = if t.gpu_awake >= 0.5 {
            self.awake_interval
        } else {
            self.asleep_interval
        };
        self.next_poll = now + interval;
        self.last = t;
        t
    }

    pub fn client_pids(&self) -> Vec<u32> {
        let Some(ref client) = self.client else {
            return Vec::new();
        };
        if self.last.gpu_awake >= 0.5 {
            client.client_pids()
        } else {
            Vec::new()
        }
    }
}

/// Collects one system-wide telemetry dict per `sample()` call (~1 Hz).
pub struct SystemSampler {
    pub user_idle_after_sec: f64,
    pub gpu: GpuSampler,
    pdh_disk: Option<PdhDiskSampler>,
    prev_t: f64,
    prev_cpu_cores: Option<Vec<CoreTickTimes>>,
    prev_dpc_cores: Option<Vec<CoreTickTimes>>,
    prev_perf: Option<SystemPerfCounters>,
    prev_net: Option<(u64, u64)>,
    disk_free_gb: f64,
    disk_free_at: f64,
}

impl SystemSampler {
    pub fn new(cfg: &MonitorConfig) -> Self {
        let mut s = Self {
            user_idle_after_sec: cfg.user_idle_after_sec,
            gpu: GpuSampler::new(cfg.gpu_poll_when_awake_sec, cfg.gpu_poll_when_asleep_sec),
            pdh_disk: PdhDiskSampler::new(),
            prev_t: mono_now(),
            prev_cpu_cores: winapi::per_core_times(),
            prev_dpc_cores: winapi::per_core_times(),
            prev_perf: winapi::system_perf_counters(),
            prev_net: winapi::network_octets(),
            disk_free_gb: f64::NAN,
            disk_free_at: 0.0,
        };
        s.prime();
        s
    }

    /// Re-prime rate baselines (called at startup and after system sleep/resume).
    pub fn prime(&mut self) {
        self.prev_t = mono_now();
        let cores = winapi::per_core_times();
        self.prev_cpu_cores = cores.clone();
        self.prev_dpc_cores = cores;
        self.prev_perf = winapi::system_perf_counters();
        self.prev_net = winapi::network_octets();
        if let Some(ref mut pdh) = self.pdh_disk {
            let _ = pdh.sample_rates();
        }
    }

    pub fn sample(&mut self) -> MetricMap {
        let now = mono_now();
        let ts = unix_now();
        let dt = (now - self.prev_t).max(1e-3);
        self.prev_t = now;

        // 1. Per-core CPU, busiest core, DPC %, ISR %, busiest DPC+ISR core %
        let (cpu_pct, core_max_pct, dpc_pct, isr_pct, dpc_max_core_pct) = self.sample_cpu_dpc();

        // 2. RAM & Commit charge
        let (ram_pct, commit_pct) = winapi::memory_and_commit_pct();

        // 3. Disk Read/Write B/s & system-wide Hard Page Faults/s
        let cur_perf = winapi::system_perf_counters();
        let (mut disk_read_bps, mut disk_write_bps, hard_faults_ps) = match (self.prev_perf, cur_perf)
        {
            (Some(p), Some(c)) => {
                let dr = c.io_read_bytes.saturating_sub(p.io_read_bytes) as f64 / dt;
                let dw = c.io_write_bytes.saturating_sub(p.io_write_bytes) as f64 / dt;
                let hf = c.hard_page_faults.wrapping_sub(p.hard_page_faults) as f64 / dt;
                (dr, dw, hf.max(0.0))
            }
            _ => (0.0, 0.0, 0.0),
        };
        self.prev_perf = cur_perf;

        if let Some(ref mut pdh) = self.pdh_disk {
            if let Some((pr, pw)) = pdh.sample_rates() {
                disk_read_bps = pr;
                disk_write_bps = pw;
            }
        }

        // 4. Network Sent/Recv B/s
        let cur_net = winapi::network_octets();
        let (net_sent_bps, net_recv_bps) = match (self.prev_net, cur_net) {
            (Some((ps, pr)), Some((cs, cr))) if cs >= ps && cr >= pr => {
                ((cs - ps) as f64 / dt, (cr - pr) as f64 / dt)
            }
            _ => (0.0, 0.0),
        };
        self.prev_net = cur_net;

        // 5. System drive free space (cached 60 s)
        if now - self.disk_free_at >= DISK_FREE_TTL_SEC || !self.disk_free_gb.is_finite() {
            self.disk_free_gb = winapi::system_drive_free_gb();
            self.disk_free_at = now;
        }

        // 6. Power & User activity
        let on_ac = winapi::on_ac_power();
        let user_active = if winapi::user_idle_seconds() < self.user_idle_after_sec {
            1.0
        } else {
            0.0
        };

        // 7. GPU
        let gpu = self.gpu.sample(now);

        let mut map = MetricMap::new();
        map.insert("ts".to_string(), ts);
        map.insert("cpu_pct".to_string(), cpu_pct);
        map.insert("core_max_pct".to_string(), core_max_pct);
        map.insert("dpc_pct".to_string(), dpc_pct);
        map.insert("isr_pct".to_string(), isr_pct);
        map.insert("dpc_max_core_pct".to_string(), dpc_max_core_pct);
        map.insert("ram_pct".to_string(), ram_pct);
        map.insert("commit_pct".to_string(), commit_pct);
        map.insert("disk_read_bps".to_string(), disk_read_bps);
        map.insert("disk_write_bps".to_string(), disk_write_bps);
        map.insert("net_sent_bps".to_string(), net_sent_bps);
        map.insert("net_recv_bps".to_string(), net_recv_bps);
        map.insert("hard_faults_ps".to_string(), hard_faults_ps);
        map.insert("disk_free_gb".to_string(), self.disk_free_gb);
        map.insert("on_ac".to_string(), on_ac);
        map.insert("user_active".to_string(), user_active);
        map.insert("gpu_awake".to_string(), gpu.gpu_awake);
        map.insert("gpu_util_pct".to_string(), gpu.gpu_util_pct);
        map.insert("gpu_power_w".to_string(), gpu.gpu_power_w);
        map.insert("gpu_temp_c".to_string(), gpu.gpu_temp_c);
        map.insert("gpu_vram_pct".to_string(), gpu.gpu_vram_pct);
        map.insert("gpu_clock_mhz".to_string(), gpu.gpu_clock_mhz);
        map.insert("gpu_throttle".to_string(), gpu.gpu_throttle as f64);
        map
    }

    fn sample_cpu_dpc(&mut self) -> (f64, f64, f64, f64, f64) {
        let Some(cur) = winapi::per_core_times() else {
            return (0.0, 0.0, f64::NAN, f64::NAN, f64::NAN);
        };

        // CPU % and Core Max %
        let mut cpu_pct = 0.0;
        let mut core_max_pct = 0.0;
        if let Some(ref prev) = self.prev_cpu_cores {
            if prev.len() == cur.len() && !cur.is_empty() {
                let mut sum_busy = 0.0;
                let mut sum_tot = 0.0;
                let mut max_core = 0.0f64;
                for (p, c) in prev.iter().zip(cur.iter()) {
                    let d_idle = (c.idle - p.idle).max(0) as f64;
                    let d_tot = ((c.kernel - p.kernel) + (c.user - p.user)).max(0) as f64;
                    if d_tot > 0.0 {
                        let busy = (d_tot - d_idle).clamp(0.0, d_tot);
                        let pct = (busy / d_tot) * 100.0;
                        max_core = max_core.max(pct);
                        sum_busy += busy;
                        sum_tot += d_tot;
                    }
                }
                if sum_tot > 0.0 {
                    cpu_pct = (sum_busy / sum_tot * 100.0).clamp(0.0, 100.0);
                    core_max_pct = max_core.clamp(0.0, 100.0);
                }
            }
        }
        self.prev_cpu_cores = Some(cur.clone());

        // DPC % and ISR % (requires >= 0.5 s window per core)
        let mut dpc_pct = f64::NAN;
        let mut isr_pct = f64::NAN;
        let mut dpc_max_core_pct = f64::NAN;
        if let Some(ref prev_dpc) = self.prev_dpc_cores {
            if prev_dpc.len() == cur.len() && !cur.is_empty() {
                let mut tot_wall = 0i64;
                let mut tot_dpc = 0i64;
                let mut tot_isr = 0i64;
                let mut max_core_dpc_isr = 0.0f64;
                for (p, c) in prev_dpc.iter().zip(cur.iter()) {
                    let wall = (c.kernel - p.kernel) + (c.user - p.user);
                    let dpc = (c.dpc - p.dpc).max(0);
                    let isr = (c.interrupt - p.interrupt).max(0);
                    tot_wall += wall;
                    tot_dpc += dpc;
                    tot_isr += isr;
                    if wall > 0 {
                        let core_pct = ((dpc + isr) as f64 / wall as f64) * 100.0;
                        max_core_dpc_isr = max_core_dpc_isr.max(core_pct);
                    }
                }
                let min_ticks = (0.5e7 * (cur.len() as f64)) as i64;
                if tot_wall >= min_ticks {
                    dpc_pct = (tot_dpc as f64 / tot_wall as f64) * 100.0;
                    isr_pct = (tot_isr as f64 / tot_wall as f64) * 100.0;
                    dpc_max_core_pct = max_core_dpc_isr;
                    self.prev_dpc_cores = Some(cur);
                }
            } else {
                self.prev_dpc_cores = Some(cur);
            }
        } else {
            self.prev_dpc_cores = Some(cur);
        }

        (cpu_pct, core_max_pct, dpc_pct, isr_pct, dpc_max_core_pct)
    }
}
