"""
UI responsiveness probe: does the app in front of you still answer?

An app "freezes" when the thread that owns its window stops pumping window
messages. Three threads time that directly, at above-normal priority, in a
small helper process (`ProbeProcess`, ~15 MB) rather than in the monitor:
the monitor runs at below-normal priority in Efficiency Mode, and when the
CPU is saturated its main thread gets preempted *while holding the GIL*, which
held in-process probe threads up for 2–4.5 s (measured); in their own process
the same threads were never late. The helper exits when the monitor does
(its stdin pipe closes) and the monitor restarts it if it dies.

  foreground  every 0.5 s, WM_NULL to the foreground window via
              SendMessageTimeout (2 s timeout; returns at once if Windows
              already considers the window hung). An answer taking ≥ 1 s, a
              timeout or a hung window is a *stall*. While it lasts the same
              window is re-probed every 0.1 s, so its end is timed closely even
              if you switch away. Stalls of one app < 3 s apart are one freeze.
  taskbar     the same against Explorer's taskbar. Stalled together with the
              app → the whole desktop froze, not just the app.
  compositor  DwmFlush, which returns at the desktop compositor's next frame
              (≈6 ms at 165 Hz). Only while you're at the keyboard. A stuck
              compositor freezes the whole screen even while every app answers.

The probe must not mistake its own delays for an app's. Each thread checks how
late it woke; a freeze overlapping a wake-up ≥ 0.25 s late is dropped, and so
is any round-trip that overran the timeout (the probe, not the app, was held
up). Sleep/hibernate (the sleep-excluding clock falls behind) or being parked
for ≥ 30 s (Modern Standby) aborts whatever was being timed and pauses
detection for 30 s after resume, when apps are still paging back in.

Finished freezes are streamed to the monitor as JSON lines; its main loop
collects them (`drain`) and rules.UiFreeze explains them with the system
samples recorded around them.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import winapi

PROBE_TIMEOUT_SEC = 2.0
LATE_SEC = 0.25           # a probe thread waking this late can't vouch for its timings
PARK_SEC = 30.0           # this late = the process was suspended (Modern Standby), not a stall
RESUME_GRACE_SEC = 30.0   # after sleep, apps page back in; their sluggishness isn't news
CHASE_SEC = 0.1           # re-probe interval while a window is stalled…
HUNG_CHASE_SEC = 0.25     # …or already marked hung (those probes return instantly)
MERGE_SEC = 3.0           # stalls of one app this close together are one freeze
MAX_STALL_SEC = 300.0     # give up on a window that stays frozen this long
DESKTOP_SLOW_SEC = 0.5    # taskbar / compositor this slow, during a freeze, = desktop-wide
USER_ACTIVE_SEC = 10.0    # the compositor probe runs only with input this recent
HISTORY_SEC = 600.0

# Probe outcomes
OK, SLOW, TIMEOUT, HUNG, GONE = "ok", "slow", "timeout", "hung", "gone"


@dataclass(slots=True)
class Target:
    """A window being probed, and who owns it."""
    hwnd: int
    pid: int
    tid: int
    exe: str = ""
    window_class: str = ""


@dataclass
class Freeze:
    start: float                 # wall clock
    end: float
    exe: str
    pid: int
    window_class: str
    scope: str = "app"           # app | desktop (taskbar or compositor too) | display (compositor only)
    hung: bool = False           # Windows itself marked the window "Not Responding"
    stalls: int = 1
    longest_sec: float = 0.0
    ended: str = "responded"
    switched_away: bool = False  # you left the frozen window before it recovered
    taskbar_slow_sec: float = 0.0
    compositor_slow_sec: float = 0.0
    tid: int = field(default=0, repr=False)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class _Stall:
    target: Target
    start: float
    hung: bool = False
    switched_away: bool = False


class StallTracker:
    """
    Probe results for one stream of windows → stalls (one window not
    answering) → freezes (stalls of one app close together).
    Not thread-safe; ResponsivenessProbe serialises access.
    """

    def __init__(self, threshold: float, merge_sec: float = MERGE_SEC, max_sec: float = MAX_STALL_SEC):
        self.threshold, self.merge_sec, self.max_sec = threshold, merge_sec, max_sec
        self.stall: _Stall | None = None
        self.freeze: Freeze | None = None      # may still absorb the next stall
        self.done: list[Freeze] = []
        self.abandoned: set[int] = set()       # windows frozen past max_sec; ignored until they answer

    @property
    def stalled(self) -> Target | None:
        return self.stall.target if self.stall else None

    def observe(self, target: Target, t_send: float, t_done: float, result: str) -> None:
        if target.hwnd in self.abandoned:
            if result in (OK, SLOW, GONE):
                self.abandoned.discard(target.hwnd)
            return
        st = self.stall
        if st is None:
            if result in (SLOW, TIMEOUT, HUNG):
                self.stall = _Stall(target, t_send, hung=result == HUNG)
                if result == SLOW:                 # answered, just late: over already
                    self._end(t_done, "responded")
        elif result in (TIMEOUT, HUNG):
            st.hung = st.hung or result == HUNG
        elif result in (OK, SLOW):
            self._end(t_done, "responded")
        elif result == GONE:
            self._end(t_done, "window closed")

    def note_foreground(self, hwnd: int | None) -> None:
        if self.stall is not None and hwnd != self.stall.target.hwnd:
            self.stall.switched_away = True

    def poll(self, now: float) -> None:
        st = self.stall
        if st is not None and now - st.start >= self.max_sec:
            self.abandoned.add(st.target.hwnd)
            self._end(now, f"still not responding after {self.max_sec / 60:g} min")
        f = self.freeze
        if f is not None and self.stall is None and now - f.end >= self.merge_sec:
            self.done.append(f)
            self.freeze = None

    def open_intervals(self, now: float) -> list[tuple[float, float]]:
        """What is still being timed: the open stall and the freeze it may join."""
        out = [(self.freeze.start, self.freeze.end)] if self.freeze else []
        if self.stall is not None:
            out.append((self.stall.start, now))
        return out

    def abort(self) -> None:
        """Sleep/resume: whatever was being timed is meaningless."""
        self.stall = self.freeze = None

    def _end(self, end: float, why: str) -> None:
        st, self.stall = self.stall, None
        dur = end - st.start
        if dur < self.threshold and not st.hung:
            return
        f = self.freeze
        if f is not None and f.pid == st.target.pid and st.start - f.end <= self.merge_sec:
            f.end = max(f.end, end)
            f.stalls += 1
            f.longest_sec = max(f.longest_sec, dur)
            f.hung = f.hung or st.hung
            f.switched_away = f.switched_away or st.switched_away
            f.ended = why
            return
        if f is not None:
            self.done.append(f)
        t = st.target
        self.freeze = Freeze(start=st.start, end=end, exe=t.exe, pid=t.pid, window_class=t.window_class,
                             tid=t.tid, hung=st.hung, longest_sec=dur, ended=why,
                             switched_away=st.switched_away)


def overlap(a0: float, a1: float, intervals) -> float:
    """Seconds of [a0, a1] covered by the (start, end) intervals (assumed not to overlap each other)."""
    return sum(max(0.0, min(a1, e) - max(a0, s)) for s, e in intervals)


class ResponsivenessProbe:
    def __init__(self, threshold: float = 1.0, interval: float = 0.5, *,
                 foreground: Callable[[], int | None] | None = None,
                 taskbar: bool = True, compositor: bool = True):
        self.threshold = threshold
        self.interval = interval
        self._pick_foreground = foreground or self._foreground_hwnd
        self._use_taskbar, self._use_compositor = taskbar, compositor
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        # The monitor has no windows of its own; tests probe windows they create.
        self._skip_pid = os.getpid() if foreground is None else 0

        self.fg = StallTracker(threshold)
        self.taskbar = StallTracker(DESKTOP_SLOW_SEC, merge_sec=0.0)
        self._taskbar_target: Target | None = None
        self._compositor: deque[tuple[float, float]] = deque()
        self._compositor_inflight: float | None = None
        self._display: list[Freeze] = []                     # compositor-only freezes, settling
        self._emitted: deque[tuple[float, float]] = deque()  # recent app freezes (display dedup)
        self._late: deque[tuple[float, float]] = deque()     # (woke at, seconds late)
        self._paused_until = 0.0
        self.stats = {"probes": 0, "freezes": 0, "dropped_probe_late": 0, "late_wakeups": 0,
                      "max_late_sec": 0.0, "last_late": None, "suspends": 0, "boosted": None}

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        loops = [("foreground", self._foreground_step)]
        if self._use_taskbar:
            loops.append(("taskbar", self._taskbar_step))
        if self._use_compositor:
            loops.append(("compositor", self._compositor_step))
        for name, step in loops:
            t = threading.Thread(target=self._run, args=(name, step), name=f"ui-probe-{name}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = PROBE_TIMEOUT_SEC + 1) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout)

    def snapshot(self) -> dict:
        with self._lock:
            s = dict(self.stats)
        s["threads"] = sum(t.is_alive() for t in self._threads)
        return s

    # ── the main loop's side ─────────────────────────────────────────

    def drain(self, now: float | None = None) -> list[Freeze]:
        """Freezes that have settled (no more stalls or desktop readings can join them)."""
        now = time.time() if now is None else now
        out: list[Freeze] = []
        with self._lock:
            self.fg.poll(now)
            self.taskbar.poll(now)
            self._prune(now)
            taskbar = [(f.start, f.end) for f in self.taskbar.done] + self.taskbar.open_intervals(now)
            compositor = list(self._compositor)
            if self._compositor_inflight is not None:
                compositor.append((self._compositor_inflight, now))
            taskbar_tid = self._taskbar_target.tid if self._taskbar_target else 0

            ready, self.fg.done = self.fg.done, []
            for f in ready:
                if self._probe_was_late(f):
                    self.stats["dropped_probe_late"] += 1
                    continue
                if f.tid != taskbar_tid:     # an Explorer freeze is not "Explorer froze too"
                    f.taskbar_slow_sec = round(overlap(f.start, f.end, taskbar), 2)
                f.compositor_slow_sec = round(overlap(f.start, f.end, compositor), 2)
                if max(f.taskbar_slow_sec, f.compositor_slow_sec) >= DESKTOP_SLOW_SEC:
                    f.scope = "desktop"
                self._emitted.append((f.start, f.end))
                out.append(f)

            busy = list(self._emitted) + self.fg.open_intervals(now)
            settled = [d for d in self._display if d.end <= now - MERGE_SEC]
            self._display = [d for d in self._display if d.end > now - MERGE_SEC]
            for d in settled:
                if overlap(d.start, d.end, busy) > 0:
                    continue                 # already explained as an app/desktop freeze
                if self._probe_was_late(d):
                    self.stats["dropped_probe_late"] += 1
                    continue
                d.taskbar_slow_sec = round(overlap(d.start, d.end, taskbar), 2)
                d.compositor_slow_sec = round(d.duration, 2)
                out.append(d)
            self.stats["freezes"] += len(out)
        return sorted(out, key=lambda f: f.start)

    def _probe_was_late(self, f: Freeze) -> bool:
        return any(t - sec <= f.end and t >= f.start for t, sec in self._late)

    def _prune(self, now: float) -> None:
        horizon = now - HISTORY_SEC
        for q in (self._compositor, self._emitted):
            while q and q[0][1] < horizon:
                q.popleft()
        while self._late and self._late[0][0] < horizon:
            self._late.popleft()
        self.taskbar.done = [iv for iv in self.taskbar.done if iv.end >= horizon]

    # ── probe threads ─────────────────────────────────────────────────

    def _run(self, name: str, step: Callable[[float], float]) -> None:
        boosted = winapi.boost_current_thread()
        with self._lock:
            self.stats["boosted"] = boosted if self.stats["boosted"] is None else self.stats["boosted"] and boosted
        clocks = winapi.sleep_clocks()
        delay, failed = 0.0, False
        while True:
            due = time.perf_counter() + delay
            if self._stop.wait(delay):
                return
            late = time.perf_counter() - due
            now = time.time()
            prev, clocks = clocks, winapi.sleep_clocks()
            slept = _slept(prev, clocks)
            if slept >= 1.0 or late >= PARK_SEC:
                self._suspend(now, name, max(slept, late))
                delay = self.interval
                continue
            if late >= LATE_SEC:
                self._note_late(now, name, late)
                delay = self.interval
                continue
            try:
                delay = step(now)
                failed = False
            except Exception:
                if not failed:   # log once per run of failures
                    logging.exception("UI probe (%s) failed; retrying", name)
                failed = True
                delay = self.interval * 4
            clocks = winapi.sleep_clocks()

    def _timed(self, name: str, call: Callable, limit: float | None):
        """(sent at, round-trip, call result), or None if the measurement can't be trusted."""
        c0, t0, p0 = winapi.sleep_clocks(), time.time(), time.perf_counter()
        value = call()
        rtt = time.perf_counter() - p0
        slept = _slept(c0, winapi.sleep_clocks())
        if slept >= 1.0:
            self._suspend(t0 + rtt, name, slept)
            return None
        if limit is not None and rtt > limit + LATE_SEC:
            self._note_late(t0 + rtt, name, rtt - limit)   # the call can't take this long; we were held up
            return None
        return t0, rtt, value

    def _suspend(self, now: float, name: str, sec: float) -> None:
        with self._lock:
            self.fg.abort()
            self.taskbar.abort()
            self._display.clear()
            self._paused_until = now + RESUME_GRACE_SEC
            self.stats["suspends"] += 1
        logging.debug("UI probe (%s): machine slept or was suspended for %.0f s; pausing %.0f s",
                      name, sec, RESUME_GRACE_SEC)

    def _note_late(self, now: float, name: str, sec: float) -> None:
        with self._lock:
            self._late.append((now, sec))
            self.stats["late_wakeups"] += 1
            if sec > self.stats["max_late_sec"]:
                self.stats["max_late_sec"] = round(sec, 2)
            self.stats["last_late"] = f"{datetime.fromtimestamp(now).isoformat(timespec='seconds')} ({sec:.2f}s)"
        log = logging.info if sec >= self.threshold else logging.debug
        log("UI probe (%s) was held up %.2f s; freezes overlapping it are ignored", name, sec)

    def _classify(self, hwnd: int, answered: bool, err: int, rtt: float, threshold: float) -> str | None:
        if answered:
            return SLOW if rtt >= threshold else OK
        if winapi.is_hung_window(hwnd):
            return HUNG
        if not winapi.is_window(hwnd):
            return GONE
        if err == winapi.ERROR_TIMEOUT or rtt >= 0.9 * PROBE_TIMEOUT_SEC:
            return TIMEOUT
        return None    # can't be probed (e.g. an elevated window, blocked by UIPI)

    def _send(self, name: str, target: Target, threshold: float):
        r = self._timed(name, lambda: winapi.send_null(target.hwnd, int(PROBE_TIMEOUT_SEC * 1000)),
                        PROBE_TIMEOUT_SEC)
        if r is None:
            return None
        t0, rtt, (answered, err) = r
        result = self._classify(target.hwnd, answered, err, rtt, threshold)
        return None if result is None else (t0, t0 + rtt, result)

    @staticmethod
    def _foreground_hwnd() -> int | None:
        hwnd = winapi.foreground_window()
        if hwnd and winapi.window_class(hwnd) == "Ghost":
            hwnd = winapi.hung_window_from_ghost(hwnd) or hwnd
        return hwnd

    def _target(self, hwnd: int | None) -> Target | None:
        if not hwnd:
            return None
        pid, tid = winapi.window_owner(hwnd)
        if not pid or pid == self._skip_pid:
            return None
        return Target(hwnd, pid, tid)

    @staticmethod
    def _describe(t: Target) -> None:
        if not t.exe:
            t.window_class = winapi.window_class(t.hwnd)
            t.exe = winapi.process_image_name(t.pid) or f"pid {t.pid}"

    def _foreground_step(self, now: float) -> float:
        with self._lock:
            target = self.fg.stalled
        fg = self._pick_foreground()
        if target is None:
            target = self._target(fg)
            if target is None:
                return self.interval
        r = self._send("foreground", target, self.threshold)
        if r is None:
            return self.interval
        t0, t1, result = r
        if result != OK:
            self._describe(target)
        with self._lock:
            self.stats["probes"] += 1
            if t0 >= self._paused_until:
                self.fg.note_foreground(fg)
                self.fg.observe(target, t0, t1, result)
            self.fg.poll(time.time())
            stalled = self.fg.stall is not None
        if result == HUNG:
            return HUNG_CHASE_SEC
        return CHASE_SEC if stalled else self.interval

    def _taskbar_step(self, now: float) -> float:
        t = self._taskbar_target
        if t is None or not winapi.is_window(t.hwnd):      # Explorer restarted
            t = self._target(winapi.taskbar_window())
            if t is not None:
                t.exe, t.window_class = "explorer.exe", "Shell_TrayWnd"
            with self._lock:
                self._taskbar_target = t
            if t is None:
                return self.interval * 10
        r = self._send("taskbar", t, DESKTOP_SLOW_SEC)
        if r is None:
            return self.interval
        t0, t1, result = r
        with self._lock:
            if t0 >= self._paused_until:
                self.taskbar.observe(t, t0, t1, result)
            self.taskbar.poll(time.time())
            stalled = self.taskbar.stall is not None
        if result == HUNG:
            return HUNG_CHASE_SEC
        return CHASE_SEC if stalled else self.interval

    def _compositor_step(self, now: float) -> float:
        idle = winapi.user_idle_seconds()
        if idle is None or idle > USER_ACTIVE_SEC:
            return self.interval        # nobody watching; don't keep the display pipeline awake
        with self._lock:
            self._compositor_inflight = now
        try:
            r = self._timed("compositor", winapi.dwm_flush, None)
        finally:
            with self._lock:
                self._compositor_inflight = None
        if r is None:
            return self.interval
        t0, rtt, ok = r
        if not ok or rtt < DESKTOP_SLOW_SEC:
            return self.interval
        t1 = t0 + rtt
        display = None
        idle = winapi.user_idle_seconds()
        if rtt >= self.threshold and idle is not None and idle <= rtt + USER_ACTIVE_SEC \
                and not winapi.fullscreen_app_running():
            target = self._target(self._pick_foreground())
            if target is not None:
                self._describe(target)
                display = Freeze(start=t0, end=t1, exe=target.exe, pid=target.pid,
                                 window_class=target.window_class, tid=target.tid, scope="display",
                                 longest_sec=rtt, ended="compositor resumed")
        with self._lock:
            if t0 >= self._paused_until:
                self._compositor.append((t0, t1))
                if display is not None:
                    self._display.append(display)
        return self.interval


def _slept(before: tuple[float, float] | None, after: tuple[float, float] | None) -> float:
    """Seconds the machine spent asleep between two sleep_clocks() readings."""
    if not before or not after:
        return 0.0
    return (after[0] - before[0]) - (after[1] - before[1])


# ════════════════════════════════════════════════════════════════════
#  Helper process
# ════════════════════════════════════════════════════════════════════

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CREATE_NO_WINDOW = 0x08000000
_NORMAL_PRIORITY_CLASS = 0x00000020   # else it inherits the monitor's below-normal class
STATS_EVERY_SEC = 30.0


class ProbeProcess:
    """The monitor's handle on the probe helper; same drain()/snapshot() interface as the probe."""

    RESTART_MIN_SEC = 30.0
    RESTART_MAX_SEC = 3600.0

    def __init__(self, threshold: float = 1.0, interval: float = 0.5, *, hwnd: int | None = None):
        self.threshold, self.interval, self._hwnd = threshold, interval, hwnd
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._started = 0.0
        self._freezes: list[Freeze] = []
        self._stats: dict = {}
        self._stopping = False
        self._backoff = self.RESTART_MIN_SEC
        self._restart_at: float | None = None
        self.restarts = 0

    def start(self) -> None:
        cmd = [sys.executable, "-m", "resmon.responsiveness",
               "--threshold", repr(self.threshold), "--interval", repr(self.interval)]
        if self._hwnd:
            cmd += ["--hwnd", str(self._hwnd)]
        flags = _CREATE_NO_WINDOW | _NORMAL_PRIORITY_CLASS if winapi.IS_WINDOWS else 0
        self._started = time.monotonic()
        self._proc = subprocess.Popen(cmd, cwd=_PROJECT_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, creationflags=flags)
        threading.Thread(target=self._read, args=(self._proc,), name="ui-probe-reader", daemon=True).start()

    def _read(self, proc: subprocess.Popen) -> None:
        for raw in proc.stdout:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if "freeze" in msg:
                with self._lock:
                    self._freezes.append(Freeze(**msg["freeze"]))
            elif "stats" in msg:
                with self._lock:
                    self._stats = msg["stats"]
            elif "log" in msg:
                level, text = msg["log"]
                logging.log(level, "UI probe: %s", text)
        rc = proc.wait()
        if self._stopping:
            return
        if time.monotonic() - self._started > 600:
            self._backoff = self.RESTART_MIN_SEC          # it had been running fine
        logging.warning("UI probe helper exited (code %s); restarting in %.0f s", rc, self._backoff)
        with self._lock:
            self._restart_at = time.monotonic() + self._backoff
        self._backoff = min(self._backoff * 2, self.RESTART_MAX_SEC)

    def drain(self, now: float | None = None) -> list[Freeze]:
        with self._lock:
            out, self._freezes = self._freezes, []
            due = self._restart_at is not None and time.monotonic() >= self._restart_at
            if due:
                self._restart_at = None
        if due and not self._stopping:
            self.restarts += 1
            try:
                self.start()
            except OSError as e:
                logging.error("Could not restart the UI probe helper: %s", e)
        return out

    def snapshot(self) -> dict:
        with self._lock:
            s = dict(self._stats)
        proc = self._proc
        s.update({"pid": proc.pid if proc else None, "running": proc is not None and proc.poll() is None,
                  "restarts": self.restarts})
        return s

    def stop(self, timeout: float = PROBE_TIMEOUT_SEC + 2) -> None:
        self._stopping = True
        proc = self._proc
        if proc is None:
            return
        try:
            proc.stdin.close()                             # the helper's cue to exit
            proc.wait(timeout)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()


class _JsonLines:
    """Thread-safe writer of one JSON object per line to the monitor."""

    def __init__(self, stream):
        self._stream, self._lock = stream, threading.Lock()

    def send(self, obj: dict) -> None:
        with self._lock:
            self._stream.write(json.dumps(obj) + "\n")
            self._stream.flush()


class _LogToMonitor(logging.Handler):
    def __init__(self, out: _JsonLines):
        super().__init__(logging.INFO)
        self._out = out

    def emit(self, record):
        try:
            self._out.send({"log": [record.levelno, self.format(record)]})
        except (OSError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    """Helper-process entry point: `python -m resmon.responsiveness` (started by ProbeProcess)."""
    ap = argparse.ArgumentParser(description="UI freeze probe helper (started by the monitor)")
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--hwnd", type=int, help="probe this window instead of the foreground one (tests)")
    args = ap.parse_args(argv)

    out = _JsonLines(sys.stdout)
    logging.getLogger().addHandler(_LogToMonitor(out))
    logging.getLogger().setLevel(logging.INFO)
    threading.excepthook = lambda a: logging.error(
        "thread %s crashed: %s: %s", a.thread.name if a.thread else "?", a.exc_type.__name__, a.exc_value)
    winapi.enable_efficiency_mode(False)

    probe = ResponsivenessProbe(args.threshold, args.interval,
                                foreground=(lambda: args.hwnd) if args.hwnd else None)
    probe.start()
    stop = threading.Event()

    def watch_monitor():
        try:
            sys.stdin.buffer.read()            # returns at EOF: the monitor stopped or died
        except (OSError, ValueError, AttributeError):
            pass
        stop.set()
    threading.Thread(target=watch_monitor, name="watch-monitor", daemon=True).start()

    next_stats = 0.0
    try:
        while not stop.wait(1.0):
            for f in probe.drain():
                out.send({"freeze": asdict(f)})
            if time.monotonic() >= next_stats:
                next_stats = time.monotonic() + STATS_EVERY_SEC
                out.send({"stats": probe.snapshot()})
    except (OSError, ValueError):              # pipe gone: the monitor is gone
        pass
    finally:
        probe.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
