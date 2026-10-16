"""
Per-process accounting: who is using the CPU, I/O, memory and handles.

v2 asked psutil for `cpu_percent` once per anomaly. That value is relative to
the previous call on the same Process object (often 0.0 or a 30-second-old
average), and the "System Idle Process" (3155 % on 32 threads) topped every
snapshot. Here we sweep all processes every few seconds and compute real rates
from cumulative counters, keyed by (pid, create_time) so PID reuse can't mix
two processes up. Idle and the monitor itself are excluded.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass

import psutil

from . import winapi
from .winapi import ProcInfo

IDLE_PID = 0
ACTIVE_CORES = 0.02           # below this a process is "idle" for history purposes
ACTIVE_IO_BPS = 64 * 1024


@dataclass(slots=True)
class ProcStat:
    pid: int
    name: str
    create_time: float
    cores: float              # logical cores in use (1.0 = one thread pegged)
    read_bps: float           # all I/O: disk, network and devices
    write_bps: float
    private_bytes: int
    working_set: int
    handles: int
    foreground: bool = False

    @property
    def key(self) -> tuple[int, float]:
        return self.pid, self.create_time

    @property
    def io_bps(self) -> float:
        return self.read_bps + self.write_bps


def _psutil_snapshot() -> list[ProcInfo]:
    """Portable fallback (slow on Windows; used only if the native call fails)."""
    out = []
    attrs = ["name", "create_time", "cpu_times", "memory_info", "io_counters", "num_threads"]
    for p in psutil.process_iter(attrs):
        i = p.info
        ct, mi, io = i.get("cpu_times"), i.get("memory_info"), i.get("io_counters")
        out.append(ProcInfo(
            pid=p.pid,
            name=i.get("name") or "",
            create_time=i.get("create_time") or 0.0,
            cpu_time=(ct.user + ct.system) if ct else 0.0,
            working_set=mi.rss if mi else 0,
            private_bytes=getattr(mi, "private", getattr(mi, "rss", 0)) if mi else 0,
            read_bytes=io.read_bytes if io else 0,
            write_bytes=io.write_bytes if io else 0,
            threads=i.get("num_threads") or 0,
            handles=0,
        ))
    return out


class ProcessTracker:
    def __init__(self, history_sec: float = 35 * 60):
        self._native = None
        if winapi.IS_WINDOWS:
            try:
                self._native = winapi.ProcessSnapshotter()
                self._native.snapshot()
            except OSError as e:
                logging.warning("Fast process snapshot unavailable, using psutil: %s", e)
                self._native = None
        self._self_pid = os.getpid()
        self._prev: dict[tuple[int, float], ProcInfo] = {}
        self._prev_wall: float | None = None
        self._prev_mono: float | None = None
        self._history_sec = history_sec
        # (wall_ts, dt, [(key, name, cores, io_bps), ...]) — only active processes, to stay small
        self.history: deque[tuple[float, float, list[tuple]]] = deque()
        self.latest: list[ProcStat] = []
        self.by_pid: dict[int, ProcStat] = {}
        self.foreground_name: str | None = None

    def snapshot(self) -> list[ProcInfo]:
        if self._native is not None:
            try:
                return self._native.snapshot()
            except OSError as e:
                logging.warning("Fast process snapshot failed, falling back to psutil: %s", e)
                self._native = None
        return _psutil_snapshot()

    def sweep(self, now_wall: float, now_mono: float, procs: list[ProcInfo] | None = None) -> list[ProcStat]:
        procs = self.snapshot() if procs is None else procs
        dt = (now_mono - self._prev_mono) if self._prev_mono is not None else 0.0
        fg_pid = winapi.foreground_pid()

        stats: list[ProcStat] = []
        for p in procs:
            if p.pid == IDLE_PID or p.pid == self._self_pid:
                continue
            prev = self._prev.get(p.key)
            cores = r = w = 0.0
            if prev is not None and dt > 0:
                cores = max(0.0, p.cpu_time - prev.cpu_time) / dt
                r = max(0, p.read_bytes - prev.read_bytes) / dt
                w = max(0, p.write_bytes - prev.write_bytes) / dt
            elif self._prev_wall is not None and p.create_time > self._prev_wall:
                # Started since the last sweep: everything it has done happened in this interval.
                life = max(now_wall - p.create_time, 0.5)
                cores, r, w = p.cpu_time / life, p.read_bytes / life, p.write_bytes / life
            stats.append(ProcStat(
                pid=p.pid, name=p.name, create_time=p.create_time,
                cores=cores, read_bps=r, write_bps=w,
                private_bytes=p.private_bytes, working_set=p.working_set, handles=p.handles,
            ))

        self.by_pid = {s.pid: s for s in stats}
        fg = self.by_pid.get(fg_pid) if fg_pid else None
        self.foreground_name = fg.name if fg else None
        if self.foreground_name:
            # Multi-process apps (browsers, games + helpers): same image name counts as foreground.
            for s in stats:
                s.foreground = s.name == self.foreground_name

        self._prev = {p.key: p for p in procs}
        self._prev_wall, self._prev_mono = now_wall, now_mono
        self.latest = stats

        if dt > 0:
            active = [(s.key, s.name, s.cores, s.io_bps) for s in stats
                      if s.cores >= ACTIVE_CORES or s.io_bps >= ACTIVE_IO_BPS]
            self.history.append((now_wall, dt, active))
        while self.history and self.history[0][0] < now_wall - self._history_sec:
            self.history.popleft()
        return stats

    def reset(self) -> None:
        """Forget rate baselines (after sleep the deltas would span hours)."""
        self._prev.clear()
        self._prev_wall = self._prev_mono = None

    # ── Summaries used to explain events ──────────────────────────────

    def top_activity(self, since: float, n: int = 5) -> dict[str, list[dict]]:
        """Top CPU and I/O consumers since `since`, grouped by executable name."""
        cpu_s: dict[str, float] = defaultdict(float)
        io_b: dict[str, float] = defaultdict(float)
        pids: dict[str, set[int]] = defaultdict(set)
        span = 0.0
        for t, dt, active in self.history:
            if t < since:
                continue
            span += dt
            for key, name, cores, io_bps in active:
                cpu_s[name] += cores * dt
                io_b[name] += io_bps * dt
                pids[name].add(key[0])
        if span <= 0:
            return {"cpu": [], "io": []}

        def rows(totals: dict[str, float], fmt) -> list[dict]:
            ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
            return [fmt(name, v) for name, v in ranked[:n] if v > 0]

        return {
            "cpu": rows(cpu_s, lambda name, v: {
                "name": name, "avg_cores": round(v / span, 2), "processes": len(pids[name]),
            }),
            "io": rows(io_b, lambda name, v: {
                "name": name, "avg_mb_s": round(v / span / 1e6, 2), "total_mb": round(v / 1e6, 1),
                "processes": len(pids[name]),
            }),
        }

    def top_memory(self, n: int = 5) -> list[dict]:
        """Largest private-memory users right now, grouped by executable name."""
        total: dict[str, int] = defaultdict(int)
        count: dict[str, int] = defaultdict(int)
        for s in self.latest:
            total[s.name] += s.private_bytes
            count[s.name] += 1
        ranked = sorted(total.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [{"name": k, "private_mb": round(v / 2**20), "processes": count[k]} for k, v in ranked]

    def name_of(self, pid: int) -> str:
        s = self.by_pid.get(pid)
        return s.name if s else f"pid {pid}"
