import ctypes
import hashlib
import math
import sqlite3
import sys
import threading
import time

import pytest

from conftest import make_sample
from resmon import winapi
from resmon.responsiveness import (GONE, HUNG, OK, SLOW, TIMEOUT, Freeze, ProbeProcess, ResponsivenessProbe,
                                   StallTracker, Target)
from resmon.rules import UiFreeze
from resmon.sampler import SAMPLE_COLUMNS, SystemSampler
from resmon.storage import Storage, db_summary

BRAVE = Target(hwnd=10, pid=100, tid=101, exe="brave.exe", window_class="Chrome_WidgetWin_1")
CODE = Target(hwnd=20, pid=200, tid=201, exe="Code.exe", window_class="Chrome_WidgetWin_1")


def feed(tr, target, results, t=0.0, gap=0.1):
    """Run (round-trip, result) pairs through the tracker back to back; returns the end time."""
    for rtt, result in results:
        tr.observe(target, t, t + rtt, result)
        t += rtt + gap
    return t


def finish(tr, now):
    tr.poll(now)
    out, tr.done = tr.done, []
    return out


# ── Stall tracking ────────────────────────────────────────────────────

def test_quick_and_merely_sluggish_answers_are_not_freezes():
    tr = StallTracker(threshold=1.0)
    t = feed(tr, BRAVE, [(0.001, OK)] * 10 + [(0.7, OK)] + [(0.002, OK)] * 3)
    assert finish(tr, t + 10) == []


def test_one_slow_answer_is_a_freeze_of_that_length():
    tr = StallTracker(threshold=1.0)
    t = feed(tr, BRAVE, [(0.001, OK), (1.4, SLOW), (0.001, OK)], t=100.0)
    [f] = finish(tr, t + 10)
    assert f.exe == "brave.exe" and f.window_class == "Chrome_WidgetWin_1" and not f.hung
    assert f.start == pytest.approx(100.101) and f.duration == pytest.approx(1.4)


def test_timeouts_then_hung_then_answer_is_one_long_hung_freeze():
    tr = StallTracker(threshold=1.0)
    t = feed(tr, BRAVE, [(2.0, TIMEOUT), (2.0, TIMEOUT), (2.0, TIMEOUT)] + [(0.0, HUNG)] * 8 + [(0.01, OK)])
    assert tr.stalled is None
    [f] = finish(tr, t + 10)
    assert f.hung and f.stalls == 1 and f.start == 0.0
    assert f.duration == pytest.approx(t - 0.1, abs=0.01)


def test_stalls_close_together_merge_and_far_apart_dont():
    tr = StallTracker(threshold=1.0)
    t = feed(tr, BRAVE, [(1.2, SLOW), (0.01, OK), (0.01, OK)])
    t = feed(tr, BRAVE, [(1.5, SLOW)], t=t + 2.0)                 # 2 s later: same freeze
    t = feed(tr, BRAVE, [(0.01, OK)], t=t)
    t = feed(tr, BRAVE, [(1.1, SLOW)], t=t + 8.0)                 # 8 s later: a new one
    first, second = finish(tr, t + 10)
    assert first.stalls == 2 and first.longest_sec == pytest.approx(1.5)
    assert second.stalls == 1


def test_different_apps_are_different_freezes():
    tr = StallTracker(threshold=1.0)
    t = feed(tr, BRAVE, [(1.2, SLOW)])
    feed(tr, CODE, [(1.3, SLOW)], t=t + 0.5)
    assert [f.exe for f in finish(tr, t + 10)] == ["brave.exe", "Code.exe"]


def test_closed_window_ends_the_freeze():
    tr = StallTracker(threshold=1.0)
    t = feed(tr, BRAVE, [(2.0, TIMEOUT), (0.0, HUNG), (0.0, GONE)])
    [f] = finish(tr, t + 10)
    assert f.ended == "window closed"


def test_window_frozen_for_good_is_reported_once():
    tr = StallTracker(threshold=1.0, max_sec=60)
    feed(tr, BRAVE, [(0.0, HUNG)])
    tr.poll(61.0)
    for t in range(62, 200, 2):
        tr.observe(BRAVE, t, t, HUNG)
        tr.poll(t)
    [f] = finish(tr, 300)
    assert f.ended.startswith("still not responding")
    tr.observe(BRAVE, 301, 301.01, OK)                            # it recovered: watched again
    feed(tr, BRAVE, [(1.5, SLOW)], t=400)
    assert len(finish(tr, 500)) == 1


def test_switching_away_from_a_frozen_window_is_noted():
    tr = StallTracker(threshold=1.0)
    tr.observe(BRAVE, 0.0, 2.0, TIMEOUT)
    tr.note_foreground(CODE.hwnd)
    tr.observe(BRAVE, 2.1, 2.2, OK)
    [f] = finish(tr, 10)
    assert f.switched_away


# ── Classification when freezes are handed to the main loop ──────────

def probe_with(freezes, taskbar=(), compositor=(), late=(), display=()):
    p = ResponsivenessProbe(threshold=1.0)
    p.fg.done = list(freezes)
    p.taskbar.done = [Freeze(s, e, "explorer.exe", 1, "Shell_TrayWnd") for s, e in taskbar]
    p._taskbar_target = Target(1, 1, 999, "explorer.exe", "Shell_TrayWnd")
    p._compositor.extend(compositor)
    p._late.extend(late)
    p._display = list(display)
    return p


def brave_freeze(start, end):
    return Freeze(start, end, "brave.exe", 100, "Chrome_WidgetWin_1", tid=101, longest_sec=end - start)


def test_app_freeze_while_the_taskbar_answers_is_the_apps_own():
    p = probe_with([brave_freeze(100, 102)], taskbar=[(50, 51)])
    [f] = p.drain(110)
    assert f.scope == "app" and f.taskbar_slow_sec == 0


def test_taskbar_or_compositor_stuck_at_the_same_time_means_the_desktop_froze():
    p = probe_with([brave_freeze(100, 102)], taskbar=[(100.5, 101.6)])
    [f] = p.drain(110)
    assert f.scope == "desktop" and f.taskbar_slow_sec == pytest.approx(1.1)
    p = probe_with([brave_freeze(100, 102)], compositor=[(99.0, 101.0)])
    [f] = p.drain(110)
    assert f.scope == "desktop" and f.compositor_slow_sec == pytest.approx(1.0)


def test_explorer_freezing_is_not_explorer_freezing_too():
    f = Freeze(100, 102, "explorer.exe", 1, "CabinetWClass", tid=999)
    p = probe_with([f], taskbar=[(100, 102)])
    assert p.drain(110)[0].scope == "app"


def test_freeze_is_dropped_when_the_probe_itself_was_held_up():
    p = probe_with([brave_freeze(100, 102), brave_freeze(200, 201.5)], late=[(101.8, 1.2)])
    [f] = p.drain(210)
    assert f.start == 200 and p.stats["dropped_probe_late"] == 1


def test_compositor_stall_is_reported_on_its_own_only_when_no_app_froze_with_it():
    screen = Freeze(300, 302, "brave.exe", 100, "Chrome_WidgetWin_1", scope="display", tid=101)
    p = probe_with([brave_freeze(100, 102)], compositor=[(100.2, 101.9), (300, 302)],
                   display=[Freeze(100.2, 101.9, "brave.exe", 100, "x", scope="display"), screen])
    out = p.drain(310)
    assert [f.scope for f in out] == ["desktop", "display"]
    assert out[1].start == 300 and out[1].compositor_slow_sec == pytest.approx(2.0)


def test_nothing_is_handed_over_until_it_has_settled():
    p = ResponsivenessProbe(threshold=1.0)
    p.fg.observe(BRAVE, 100.0, 101.5, SLOW)
    assert p.drain(102.0) == []                                   # another stall could still join
    assert len(p.drain(105.0)) == 1


# ── Real windows ──────────────────────────────────────────────────────

class FreezableWindow(threading.Thread):
    """A hidden window whose thread can be told to freeze, or to stay busy while still pumping."""

    def __init__(self):
        super().__init__(daemon=True)
        self.ready, self.done = threading.Event(), threading.Event()
        self.freeze_for = self.busy_for = 0.0
        self.hwnd = None

    def run(self):
        from ctypes import wintypes
        u = ctypes.WinDLL("user32")
        k = ctypes.WinDLL("kernel32")

        class WNDCLASSW(ctypes.Structure):
            _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", ctypes.c_void_p), ("cbClsExtra", ctypes.c_int),
                        ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                        ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                        ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]
        u.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
        u.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                      ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                      wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
        u.CreateWindowExW.restype = wintypes.HWND
        u.PeekMessageW.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
        u.MsgWaitForMultipleObjects.argtypes = [wintypes.DWORD, ctypes.c_void_p, wintypes.BOOL, wintypes.DWORD,
                                                wintypes.DWORD]
        u.DestroyWindow.argtypes = [wintypes.HWND]
        k.GetModuleHandleW.restype = wintypes.HMODULE
        hinst = k.GetModuleHandleW(None)
        # DefWindowProcW as the window procedure: messages never need the GIL.
        wc = WNDCLASSW(lpfnWndProc=ctypes.cast(u.DefWindowProcW, ctypes.c_void_p).value,
                       hInstance=hinst, lpszClassName="ResmonFreezeTest")
        u.RegisterClassW(ctypes.byref(wc))
        self.hwnd = u.CreateWindowExW(0, "ResmonFreezeTest", "freeze test", 0, 0, 0, 50, 50, None, None, hinst, None)
        self.ready.set()
        msg = wintypes.MSG()

        def pump():
            while u.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
                u.DispatchMessageW(ctypes.byref(msg))

        data = b"x" * (1 << 20)
        while not self.done.is_set():
            if self.freeze_for:
                d, self.freeze_for = self.freeze_for, 0.0
                time.sleep(d)                                     # not pumping messages: frozen
            if self.busy_for:
                end, self.busy_for = time.perf_counter() + self.busy_for, 0.0
                while time.perf_counter() < end:                  # CPU-bound, but pumps between chunks
                    hashlib.sha256(data).digest()
                    pump()
            u.MsgWaitForMultipleObjects(0, None, False, 20, 0x04FF)
            pump()
        u.DestroyWindow(self.hwnd)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows API")
def test_real_window_busy_is_fine_frozen_is_caught():
    w = FreezableWindow()
    w.start()
    w.ready.wait(5)
    if not w.hwnd:
        pytest.skip("cannot create window in headless/service session")
    p = ResponsivenessProbe(threshold=1.0, interval=0.25, foreground=lambda: w.hwnd, compositor=False)
    p.start()
    try:
        time.sleep(0.5)
        w.busy_for = 2.5
        time.sleep(6.0)
        assert p.drain() == [], "a busy app that keeps pumping messages must not count as frozen"
        w.freeze_for = 1.6
        time.sleep(5.5)
        freezes = p.drain()
    finally:
        p.stop()
        w.done.set()
    assert p.snapshot()["probes"] >= 10
    if not freezes:
        assert p.snapshot()["dropped_probe_late"], "freeze neither recorded nor explained"
        pytest.skip("the probe itself was held up on this run")
    [f] = freezes
    assert f.scope == "app" and f.exe.lower().startswith("python") and f.window_class == "ResmonFreezeTest"
    assert 1.1 <= f.duration <= 3.5


@pytest.mark.skipif(sys.platform != "win32", reason="Windows API")
def test_helper_process_reports_freezes_and_restarts_after_a_crash(monkeypatch):
    w = FreezableWindow()
    w.start()
    w.ready.wait(5)
    if not w.hwnd:
        pytest.skip("cannot create window in headless/service session")
    monkeypatch.setattr(ProbeProcess, "RESTART_MIN_SEC", 0.0)
    p = ProbeProcess(threshold=1.0, interval=0.25, hwnd=w.hwnd)
    p.start()
    try:
        deadline = time.time() + 15
        while "threads" not in p.snapshot() and time.time() < deadline:   # first stats line arrived
            time.sleep(0.1)
        assert p.snapshot()["running"] and p.snapshot()["threads"] == 3
        w.freeze_for = 1.6
        freezes = []
        freeze_deadline = time.time() + 20
        while not freezes and time.time() < freeze_deadline:
            time.sleep(0.3)
            freezes = p.drain()
        if not freezes:
            if p.snapshot().get("dropped_probe_late"):
                pytest.skip("the probe itself was held up on this runner")
        assert freezes, "freeze was not captured within timeout"
        [f] = freezes
        assert f.exe.lower().startswith("python") and 1.1 <= f.duration <= 3.5

        first = p.snapshot()["pid"]
        p._proc.kill()                                    # the helper dies…
        restart_deadline = time.time() + 10
        while (p.snapshot()["running"] or p._restart_at is None) and time.time() < restart_deadline:
            time.sleep(0.1)
        p.drain()                                         # …and the next tick restarts it
        assert p.restarts == 1 and p.snapshot()["running"] and p.snapshot()["pid"] != first
    finally:
        p.stop()
        w.done.set()
    assert not p.snapshot()["running"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows API")
def test_probe_calls_against_real_windows():
    tb = winapi.taskbar_window()
    if tb is None:
        pytest.skip("no taskbar (e.g. a service session)")
    pid, tid = winapi.window_owner(tb)
    img = winapi.process_image_name(pid)
    if img is None:
        pytest.skip("cannot query explorer image name (insufficient permissions)")
    assert pid and tid and img.lower() == "explorer.exe"
    assert winapi.send_null(tb, 2000) == (True, 0) and not winapi.is_hung_window(tb)
    assert winapi.boost_current_thread() in (True, False)
    up, unbiased = winapi.sleep_clocks()
    # GetTickCount64 (15.6ms quantum) can lag high-precision interrupt time by up to ~20ms on fresh boot
    assert up >= unbiased - 0.05 and unbiased > 0


# ── Explaining a freeze ───────────────────────────────────────────────

class FakeProbe:
    def __init__(self, freezes):
        self.freezes = list(freezes)

    def drain(self, now):
        out, self.freezes = self.freezes, []
        return out


def run_samples(rule, t0, t1, **per_ts):
    for t in range(t0, t1):
        rule.on_sample(make_sample(float(t), **{k: v(t) for k, v in per_ts.items()}), float(t))


def test_freeze_report_waits_for_context_then_explains_it(mon):
    mon.ui_probe = FakeProbe([brave_freeze(1000.2, 1002.6)])
    mon.procs.top_activity = lambda since, n=5: {"cpu": [{"name": "brave.exe", "avg_cores": 0.4, "processes": 9}],
                                                 "io": []}
    rule = UiFreeze(mon)
    awake = lambda t: 1.0 if t < 1001 else 0.0                          # dGPU powers off during it
    run_samples(rule, 900, 1008, gpu_awake=awake)
    assert not mon.events.events                                       # not settled yet
    run_samples(rule, 1008, 1012, gpu_awake=awake)
    [ev] = mon.events.of_kind("ui_freeze")
    assert ev.ts_start == 1000.2 and ev.ts_end == 1002.6
    assert ev.title.startswith("brave.exe stopped responding for 2.4 s — dGPU powered off 0.8 s into it")
    assert ev.severity == "info"
    d = ev.details
    assert d["dgpu_transitions"] == [{"at_sec": 0.8, "dgpu": "powered off"}]
    assert d["top_cpu"][0]["name"] == "brave.exe" and d["around"]["dgpu_awake"] == [0, 1]
    assert not d["dpc_isr_spike"] and d["window_class"] == "Chrome_WidgetWin_1"


def test_dpc_storm_and_idle_app_are_named(mon):
    hung = brave_freeze(1000.0, 1006.0)
    hung.hung = True
    mon.ui_probe = FakeProbe([hung])
    mon.procs.cpu_seconds = lambda pid, start, end: (0.05, 10.0)
    rule = UiFreeze(mon)
    run_samples(rule, 900, 1015, dpc_isr_core_max_pct=lambda t: 60.0 if 1001 <= t <= 1004 else 2.0)
    [ev] = mon.events.of_kind("ui_freeze")
    assert ev.severity == "warning" and "Not Responding" in ev.title
    assert ev.details["dpc_isr_spike"] and "DPC/interrupt storm" in ev.title
    assert any("nearly idle" in c for c in ev.details["clues"])


def test_desktop_and_display_freezes_are_titled_as_such(mon):
    desk = brave_freeze(1000.0, 1001.5)
    desk.scope, desk.taskbar_slow_sec = "desktop", 1.2
    screen = Freeze(1100.0, 1102.0, "brave.exe", 100, "Chrome_WidgetWin_1", scope="display",
                    compositor_slow_sec=2.0)
    mon.ui_probe = FakeProbe([desk, screen])
    rule = UiFreeze(mon)
    run_samples(rule, 990, 1120)
    a, b = mon.events.of_kind("ui_freeze")
    assert a.title == "Desktop froze for 1.5 s: brave.exe and the taskbar stopped responding"
    assert b.title.startswith("Screen froze for 2.0 s (desktop compositor stalled) while using brave.exe")
    assert a.severity == b.severity == "warning"


def test_freeze_while_user_away_stays_info(mon):
    f = brave_freeze(1000.0, 1008.0)
    mon.ui_probe = FakeProbe([f])
    rule = UiFreeze(mon)
    run_samples(rule, 990, 1020, user_idle_sec=lambda t: 600.0)
    [ev] = mon.events.of_kind("ui_freeze")
    assert ev.severity == "info" and ev.details["user_idle"]


# ── DPC/ISR sampling and storage ──────────────────────────────────────

@pytest.mark.skipif(sys.platform != "win32", reason="Windows API")
def test_dpc_and_interrupt_time_are_sampled():
    import os
    times = winapi.processor_times()
    assert len(times) == os.cpu_count()
    assert all(total >= dpc + isr >= 0 for total, dpc, isr in times)
    sm = SystemSampler(gpu=None)
    assert math.isnan(sm.sample(0.5)["dpc_isr_core_max_pct"])     # too short to resolve 15.6 ms ticks
    time.sleep(0.8)
    s = sm.sample(1.0)
    if math.isnan(s["dpc_pct"]):
        time.sleep(0.8)
        s = sm.sample(2.0)
    assert not math.isnan(s["dpc_pct"])
    assert 0 <= s["dpc_pct"] < 50 and 0 <= s["isr_pct"] < 50
    assert s["dpc_pct"] + s["isr_pct"] <= s["dpc_isr_core_max_pct"] + 1e-9 <= 100


def test_existing_database_gains_the_new_sample_columns(tmp_path):
    db = tmp_path / "m.db"
    old_cols = [c for c in SAMPLE_COLUMNS[1:] if c not in ("dpc_pct", "isr_pct", "dpc_isr_core_max_pct")]
    conn = sqlite3.connect(db)
    conn.execute(f"CREATE TABLE samples (ts REAL PRIMARY KEY, {', '.join(c + ' REAL' for c in old_cols)}) "
                 "WITHOUT ROWID")
    conn.execute("INSERT INTO samples (ts, cpu_pct) VALUES (1.0, 5.0)")
    conn.commit()
    conn.close()
    st = Storage(db)
    st.add_sample(make_sample(2.0, dpc_isr_core_max_pct=33.0))
    st.flush()
    st.close()
    s = db_summary(db)
    assert s["samples"]["n"] == 2 and s["latest_sample"]["dpc_isr_core_max_pct"] == 33.0
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT dpc_pct FROM samples WHERE ts = 1.0").fetchone()[0] is None
    conn.close()
    assert not math.isnan(s["latest_sample"]["cpu_pct"])
