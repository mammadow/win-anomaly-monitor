"""
System-wide telemetry, sampled once per second.

Fixes over v2:
  * Per-core max CPU: on a 32-thread CPU a single runaway thread shows up as
    ~3 % total CPU — invisible — but as 100 % on its core.
  * Counter resets (network adapter / VPN going away) used to produce negative
    rates (down to -78 MB/s in the old data). Negative deltas are now clamped.
  * The dGPU being asleep (Optimus D3cold; ~80 % of the time on this laptop)
    is a state, not "temperature 0 °C". It is reported as gpu_awake=0 with
    temperature unknown, instead of feeding 0↔47 °C jumps to the model.
  * Physically impossible GPU readings (a 590 W spike was recorded) are dropped.

System-drive free space (and the page file it hosts) changes slowly, so it is
read once a minute. Those readings ride along in every sample for the rules
but are not stored.
"""

from __future__ import annotations

import logging
import math
import os
import time
import warnings

import psutil

from . import winapi

with warnings.catch_warnings():
    # The `pynvml` shim package warns on import; nvidia-ml-py provides the same module.
    warnings.simplefilter("ignore", FutureWarning)
    try:
        import pynvml
    except ImportError:  # pragma: no cover - GPU support is optional
        pynvml = None

NAN = math.nan

SAMPLE_COLUMNS = [
    "ts",
    "cpu_pct", "core_max_pct",
    "ram_pct", "ram_used_gb", "commit_pct",
    "disk_read_bps", "disk_write_bps", "net_sent_bps", "net_recv_bps",
    "on_ac", "battery_pct", "user_idle_sec",
    "gpu_awake", "gpu_util_pct", "gpu_vram_pct", "gpu_temp_c", "gpu_power_w",
    "gpu_clock_mhz", "gpu_throttle",
    "handles", "processes", "threads",
]

SYSTEM_DRIVE = os.environ.get("SystemDrive", "C:") + "\\"
DRIVE_INTERVAL_SEC = 60

# Refreshed every DRIVE_INTERVAL_SEC; present in each sample, not in SAMPLE_COLUMNS.
_DRIVE_UNKNOWN = {
    "sysdrive_free_gb": NAN, "sysdrive_total_gb": NAN, "pagefile_gb": NAN, "commit_limit_gb": NAN,
}

# NVML clock-event ("throttle") reason bits that mean heat, not workload.
THROTTLE_HW_SLOWDOWN = 0x08
THROTTLE_SW_THERMAL = 0x20
THROTTLE_HW_THERMAL = 0x40
THERMAL_THROTTLE_MASK = THROTTLE_HW_SLOWDOWN | THROTTLE_SW_THERMAL | THROTTLE_HW_THERMAL

_GPU_ABSENT = {
    "gpu_awake": NAN, "gpu_util_pct": NAN, "gpu_vram_pct": NAN, "gpu_temp_c": NAN,
    "gpu_power_w": NAN, "gpu_clock_mhz": NAN, "gpu_throttle": 0,
}
_GPU_ASLEEP = {
    "gpu_awake": 0.0, "gpu_util_pct": 0.0, "gpu_vram_pct": 0.0, "gpu_temp_c": NAN,
    "gpu_power_w": 0.0, "gpu_clock_mhz": 0.0, "gpu_throttle": 0,
}


class GpuSampler:
    """NVIDIA GPU via NVML. Never raises; missing GPU → NaNs, sleeping GPU → gpu_awake=0."""

    RETRY_INIT_SEC = 300  # the driver can still be loading when we start at logon

    def __init__(self):
        self._handle = None
        self._next_init = 0.0
        self._init_logged = False
        self._last_proc_ts = 0
        self.name: str | None = None
        self.power_limit_w: float | None = None
        self.slowdown_temp_c: float | None = None
        self.awake = False
        self._init()

    @property
    def available(self) -> bool:
        return self._handle is not None

    def _init(self) -> None:
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() == 0:
                raise pynvml.NVMLError(pynvml.NVML_ERROR_NOT_FOUND)
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(h)
            self.name = name.decode() if isinstance(name, bytes) else name
            try:
                self.power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
            except pynvml.NVMLError:
                self.power_limit_w = None
            try:
                self.slowdown_temp_c = float(pynvml.nvmlDeviceGetTemperatureThreshold(
                    h, pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN))
            except pynvml.NVMLError:
                self.slowdown_temp_c = None
            self._handle = h
            logging.info("GPU: %s (power limit %s W, slowdown at %s °C)",
                         self.name, self.power_limit_w, self.slowdown_temp_c)
        except pynvml.NVMLError as e:
            self._next_init = time.monotonic() + self.RETRY_INIT_SEC
            if not self._init_logged:
                logging.warning("NVML unavailable, GPU metrics disabled for now: %s", e)
                self._init_logged = True

    def sample(self) -> dict:
        if self._handle is None:
            if pynvml is not None and time.monotonic() >= self._next_init:
                self._init()
            if self._handle is None:
                self.awake = False
                return dict(_GPU_ABSENT)

        h, nv = self._handle, pynvml
        try:
            util = nv.nvmlDeviceGetUtilizationRates(h).gpu
        except nv.NVMLError:
            # On Optimus laptops this is what a powered-down dGPU looks like.
            self.awake = False
            return dict(_GPU_ASLEEP)
        self.awake = True

        def get(fn, *args, default=NAN):
            try:
                return fn(h, *args)
            except nv.NVMLError:
                return default

        mem = get(nv.nvmlDeviceGetMemoryInfo, default=None)
        temp = float(get(nv.nvmlDeviceGetTemperature, nv.NVML_TEMPERATURE_GPU))
        power = get(nv.nvmlDeviceGetPowerUsage) / 1000.0
        clock = float(get(nv.nvmlDeviceGetClockInfo, nv.NVML_CLOCK_GRAPHICS))
        reasons_fn = getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None) \
            or nv.nvmlDeviceGetCurrentClocksThrottleReasons
        throttle = int(get(reasons_fn, default=0))

        # Sanity limits — drop readings that are physically impossible.
        if not 1 <= temp <= 125:
            temp = NAN
        max_power = (self.power_limit_w or 250.0) * 1.5
        if not 0 <= power <= max_power:
            power = NAN

        return {
            "gpu_awake": 1.0,
            "gpu_util_pct": float(min(max(util, 0), 100)),
            "gpu_vram_pct": 100.0 * mem.used / mem.total if mem and mem.total else NAN,
            "gpu_temp_c": temp,
            "gpu_power_w": power,
            "gpu_clock_mhz": clock,
            "gpu_throttle": throttle,
        }

    def process_utilization(self) -> dict[int, int]:
        """{pid: peak SM utilisation %} for processes that used the GPU since the last call."""
        if self._handle is None or not self.awake:
            return {}
        try:
            samples = pynvml.nvmlDeviceGetProcessUtilization(self._handle, self._last_proc_ts)
        except pynvml.NVMLError:
            return {}
        out: dict[int, int] = {}
        for s in samples:
            out[s.pid] = max(out.get(s.pid, 0), int(s.smUtil))
            self._last_proc_ts = max(self._last_proc_ts, int(s.timeStamp))
        return out

    def shutdown(self) -> None:
        if self._handle is not None:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
            self._handle = None


def _rate(cur: int, prev: int, dt: float) -> float:
    d = cur - prev
    return d / dt if d >= 0 else 0.0  # negative = counter reset (NIC/disk vanished)


class SystemSampler:
    def __init__(self, gpu: GpuSampler | None = None):
        self.gpu = gpu
        self._drive = dict(_DRIVE_UNKNOWN)
        self._next_drive = 0.0
        self.reset_rates()

    def reset_rates(self) -> None:
        """Re-baseline rate counters (start-up, or after the machine slept)."""
        psutil.cpu_percent(percpu=True)
        self._prev_disk = psutil.disk_io_counters()
        self._prev_net = psutil.net_io_counters()
        self._prev_t = time.monotonic()

    def sample(self, ts: float | None = None) -> dict:
        now = time.monotonic()
        dt = max(now - self._prev_t, 1e-3)
        self._prev_t = now

        per_core = psutil.cpu_percent(percpu=True)
        mem = psutil.virtual_memory()

        disk = psutil.disk_io_counters()
        if disk is not None and self._prev_disk is not None:
            disk_r = _rate(disk.read_bytes, self._prev_disk.read_bytes, dt)
            disk_w = _rate(disk.write_bytes, self._prev_disk.write_bytes, dt)
        else:
            disk_r = disk_w = NAN
        self._prev_disk = disk

        net = psutil.net_io_counters()
        if net is not None and self._prev_net is not None:
            net_s = _rate(net.bytes_sent, self._prev_net.bytes_sent, dt)
            net_r = _rate(net.bytes_recv, self._prev_net.bytes_recv, dt)
        else:
            net_s = net_r = NAN
        self._prev_net = net

        bat = psutil.sensors_battery()
        on_ac = 1.0 if bat is None or bat.power_plugged in (True, None) else 0.0
        battery_pct = float(bat.percent) if bat is not None else NAN

        perf = winapi.performance_info()
        idle = winapi.user_idle_seconds()
        if now >= self._next_drive:
            self._next_drive = now + DRIVE_INTERVAL_SEC
            self._drive = self._drive_space(perf)

        s = {
            "ts": time.time() if ts is None else ts,
            "cpu_pct": sum(per_core) / len(per_core) if per_core else NAN,
            "core_max_pct": max(per_core) if per_core else NAN,
            "ram_pct": mem.percent,
            "ram_used_gb": mem.used / 2**30,
            "commit_pct": perf.commit_percent if perf else NAN,
            "disk_read_bps": disk_r,
            "disk_write_bps": disk_w,
            "net_sent_bps": net_s,
            "net_recv_bps": net_r,
            "on_ac": on_ac,
            "battery_pct": battery_pct,
            "user_idle_sec": idle if idle is not None else NAN,
            "handles": perf.handles if perf else NAN,
            "processes": perf.processes if perf else NAN,
            "threads": perf.threads if perf else NAN,
        }
        s.update(self._drive)
        s.update(self.gpu.sample() if self.gpu else _GPU_ABSENT)
        return s

    @staticmethod
    def _drive_space(perf: winapi.PerformanceInfo | None) -> dict:
        try:
            du = psutil.disk_usage(SYSTEM_DRIVE)
            free, total = du.free / 2**30, du.total / 2**30
        except OSError:
            free = total = NAN
        return {
            "sysdrive_free_gb": free,
            "sysdrive_total_gb": total,
            "pagefile_gb": perf.pagefile_bytes / 2**30 if perf else NAN,
            "commit_limit_gb": perf.commit_limit_bytes / 2**30 if perf else NAN,
        }
