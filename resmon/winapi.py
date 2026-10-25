"""
Thin ctypes wrappers for Windows APIs that psutil either lacks or makes slow.

The important one is `ProcessSnapshotter`. Walking ~400 processes with psutil
costs ~1.3 s on this machine, because protected processes fall back to a
per-process NtQuerySystemInformation call (O(n^2)). A single
NtQuerySystemInformation(SystemProcessInformation) call returns CPU times,
memory, I/O and handle counts for *every* process in a few milliseconds.

The window-message calls at the bottom back the UI responsiveness probe
(responsiveness.py): a WM_NULL round-trip tells whether a window's thread is
still pumping messages, which is what "the app froze" means to the user.

Everything here degrades gracefully: on non-Windows platforms (or if an API
call fails) callers get None / empty results and fall back to psutil.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from dataclasses import dataclass

IS_WINDOWS = sys.platform == "win32"

_EPOCH_AS_FILETIME = 116_444_736_000_000_000   # 1970-01-01 in 100 ns ticks since 1601
_STATUS_INFO_LENGTH_MISMATCH = -1073741820     # 0xC0000004 as a signed NTSTATUS
_SYSTEM_PROCESS_INFORMATION_CLASS = 5
_SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION_CLASS = 8

WM_NULL = 0x0000
SMTO_BLOCK = 0x0001
SMTO_ABORTIFHUNG = 0x0002
ERROR_TIMEOUT = 1460


@dataclass(slots=True)
class ProcInfo:
    pid: int
    name: str
    create_time: float      # epoch seconds (0-ish for System / Idle)
    cpu_time: float         # user + kernel seconds, cumulative
    working_set: int        # bytes resident in RAM
    private_bytes: int      # bytes of committed private memory (what leaks grow)
    read_bytes: int         # cumulative bytes read: files (disk or cache), pipes, devices
    write_bytes: int        # cumulative bytes written
    threads: int
    handles: int
    other_bytes: int = 0    # cumulative "other" I/O: socket (network) traffic and device control calls

    @property
    def key(self) -> tuple[int, float]:
        """PIDs get reused; (pid, create_time) identifies one process instance."""
        return self.pid, self.create_time


if IS_WINDOWS:
    _ntdll = ctypes.WinDLL("ntdll")
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _dwmapi = ctypes.WinDLL("dwmapi")
    _shell32 = ctypes.WinDLL("shell32")

    class _UNICODE_STRING(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.USHORT),
            ("MaximumLength", wintypes.USHORT),
            ("Buffer", ctypes.c_void_p),
        ]

    class _SYSTEM_PROCESS_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("NextEntryOffset", wintypes.ULONG),
            ("NumberOfThreads", wintypes.ULONG),
            ("WorkingSetPrivateSize", ctypes.c_int64),
            ("HardFaultCount", wintypes.ULONG),
            ("NumberOfThreadsHighWatermark", wintypes.ULONG),
            ("CycleTime", ctypes.c_uint64),
            ("CreateTime", ctypes.c_int64),
            ("UserTime", ctypes.c_int64),
            ("KernelTime", ctypes.c_int64),
            ("ImageName", _UNICODE_STRING),
            ("BasePriority", ctypes.c_long),
            ("UniqueProcessId", ctypes.c_void_p),
            ("InheritedFromUniqueProcessId", ctypes.c_void_p),
            ("HandleCount", wintypes.ULONG),
            ("SessionId", wintypes.ULONG),
            ("UniqueProcessKey", ctypes.c_size_t),
            ("PeakVirtualSize", ctypes.c_size_t),
            ("VirtualSize", ctypes.c_size_t),
            ("PageFaultCount", wintypes.ULONG),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivatePageCount", ctypes.c_size_t),
            ("ReadOperationCount", ctypes.c_int64),
            ("WriteOperationCount", ctypes.c_int64),
            ("OtherOperationCount", ctypes.c_int64),
            ("ReadTransferCount", ctypes.c_int64),
            ("WriteTransferCount", ctypes.c_int64),
            ("OtherTransferCount", ctypes.c_int64),
        ]

    class _PERFORMANCE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("CommitTotal", ctypes.c_size_t),
            ("CommitLimit", ctypes.c_size_t),
            ("CommitPeak", ctypes.c_size_t),
            ("PhysicalTotal", ctypes.c_size_t),
            ("PhysicalAvailable", ctypes.c_size_t),
            ("SystemCache", ctypes.c_size_t),
            ("KernelTotal", ctypes.c_size_t),
            ("KernelPaged", ctypes.c_size_t),
            ("KernelNonpaged", ctypes.c_size_t),
            ("PageSize", ctypes.c_size_t),
            ("HandleCount", wintypes.DWORD),
            ("ProcessCount", wintypes.DWORD),
            ("ThreadCount", wintypes.DWORD),
        ]

    class _LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

    class _PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
        _fields_ = [
            ("Version", wintypes.ULONG),
            ("ControlMask", wintypes.ULONG),
            ("StateMask", wintypes.ULONG),
        ]

    _THREAD_POWER_THROTTLING_STATE = _PROCESS_POWER_THROTTLING_STATE   # same layout

    class _SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("IdleTime", ctypes.c_int64),
            ("KernelTime", ctypes.c_int64),     # includes idle, DPC and interrupt time
            ("UserTime", ctypes.c_int64),
            ("DpcTime", ctypes.c_int64),
            ("InterruptTime", ctypes.c_int64),
            ("InterruptCount", wintypes.ULONG),
        ]

    _ntdll.NtQuerySystemInformation.argtypes = [
        wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG)]
    _ntdll.NtQuerySystemInformation.restype = ctypes.c_long

    _kernel32.K32GetPerformanceInfo.argtypes = [
        ctypes.POINTER(_PERFORMANCE_INFORMATION), wintypes.DWORD]
    _kernel32.K32GetPerformanceInfo.restype = wintypes.BOOL
    _kernel32.GetTickCount.restype = wintypes.DWORD
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.SetProcessInformation.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _kernel32.SetProcessInformation.restype = wintypes.BOOL

    _user32.GetForegroundWindow.restype = wintypes.HWND
    _user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    _user32.GetLastInputInfo.argtypes = [ctypes.POINTER(_LASTINPUTINFO)]
    _user32.GetLastInputInfo.restype = wintypes.BOOL

    _kernel32.GetCurrentThread.restype = wintypes.HANDLE
    _kernel32.SetThreadInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _kernel32.SetThreadInformation.restype = wintypes.BOOL
    _kernel32.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
    _kernel32.SetThreadPriority.restype = wintypes.BOOL
    _kernel32.GetTickCount64.restype = ctypes.c_uint64
    _kernel32.QueryUnbiasedInterruptTime.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    _kernel32.QueryUnbiasedInterruptTime.restype = wintypes.BOOL
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    _kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL

    _user32.SendMessageTimeoutW.argtypes = [
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM, wintypes.UINT, wintypes.UINT,
        ctypes.POINTER(ctypes.c_size_t)]
    _user32.SendMessageTimeoutW.restype = ctypes.c_size_t
    _user32.IsHungAppWindow.argtypes = [wintypes.HWND]
    _user32.IsHungAppWindow.restype = wintypes.BOOL
    _user32.IsWindow.argtypes = [wintypes.HWND]
    _user32.IsWindow.restype = wintypes.BOOL
    _user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.GetClassNameW.restype = ctypes.c_int
    _user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    _user32.FindWindowW.restype = wintypes.HWND
    # Undocumented but long-standing: the frozen window behind a "(Not Responding)" ghost.
    _HungWindowFromGhostWindow = getattr(_user32, "HungWindowFromGhostWindow", None)
    if _HungWindowFromGhostWindow is not None:
        _HungWindowFromGhostWindow.argtypes = [wintypes.HWND]
        _HungWindowFromGhostWindow.restype = wintypes.HWND

    _dwmapi.DwmFlush.restype = ctypes.c_long
    _shell32.SHQueryUserNotificationState.argtypes = [ctypes.POINTER(ctypes.c_int)]
    _shell32.SHQueryUserNotificationState.restype = ctypes.c_long


class ProcessSnapshotter:
    """One syscall → every process's counters. Reuses its buffer between calls."""

    def __init__(self, initial_buffer: int = 1 << 21):
        if not IS_WINDOWS:
            raise OSError("ProcessSnapshotter requires Windows")
        self._buf = ctypes.create_string_buffer(initial_buffer)

    def snapshot(self) -> list[ProcInfo]:
        needed = wintypes.ULONG(0)
        for _ in range(8):
            status = _ntdll.NtQuerySystemInformation(
                _SYSTEM_PROCESS_INFORMATION_CLASS, self._buf, len(self._buf), ctypes.byref(needed))
            if status == _STATUS_INFO_LENGTH_MISMATCH:
                # Process list grew between calls; leave headroom and retry.
                self._buf = ctypes.create_string_buffer(max(needed.value, len(self._buf)) + (1 << 18))
                continue
            if status < 0:
                raise OSError(f"NtQuerySystemInformation failed: NTSTATUS 0x{status & 0xFFFFFFFF:08X}")
            break
        else:
            raise OSError("NtQuerySystemInformation: buffer never large enough")

        out: list[ProcInfo] = []
        offset = 0
        while True:
            spi = _SYSTEM_PROCESS_INFORMATION.from_buffer(self._buf, offset)
            pid = spi.UniqueProcessId or 0
            if spi.ImageName.Buffer and spi.ImageName.Length:
                name = ctypes.wstring_at(spi.ImageName.Buffer, spi.ImageName.Length // 2)
            else:
                name = "System Idle Process" if pid == 0 else ""
            out.append(ProcInfo(
                pid=pid,
                name=name,
                create_time=max(0.0, (spi.CreateTime - _EPOCH_AS_FILETIME) / 1e7),
                cpu_time=(spi.UserTime + spi.KernelTime) / 1e7,
                working_set=spi.WorkingSetSize,
                private_bytes=spi.PrivatePageCount,
                read_bytes=spi.ReadTransferCount,
                write_bytes=spi.WriteTransferCount,
                threads=spi.NumberOfThreads,
                handles=spi.HandleCount,
                other_bytes=spi.OtherTransferCount,
            ))
            if spi.NextEntryOffset == 0:
                break
            offset += spi.NextEntryOffset
        return out


@dataclass(slots=True)
class PerformanceInfo:
    commit_bytes: int
    commit_limit_bytes: int
    physical_bytes: int
    handles: int
    processes: int
    threads: int

    @property
    def commit_percent(self) -> float:
        return 100.0 * self.commit_bytes / self.commit_limit_bytes if self.commit_limit_bytes else 0.0

    @property
    def pagefile_bytes(self) -> int:
        """Page-file share of the commit limit (limit − RAM; how psutil sizes swap on Windows)."""
        return max(0, self.commit_limit_bytes - self.physical_bytes)


def performance_info() -> PerformanceInfo | None:
    """System-wide commit charge and object counts (one cheap call)."""
    if not IS_WINDOWS:
        return None
    pi = _PERFORMANCE_INFORMATION()
    pi.cb = ctypes.sizeof(pi)
    if not _kernel32.K32GetPerformanceInfo(ctypes.byref(pi), pi.cb):
        return None
    return PerformanceInfo(
        commit_bytes=pi.CommitTotal * pi.PageSize,
        commit_limit_bytes=pi.CommitLimit * pi.PageSize,
        physical_bytes=pi.PhysicalTotal * pi.PageSize,
        handles=pi.HandleCount,
        processes=pi.ProcessCount,
        threads=pi.ThreadCount,
    )


def foreground_pid() -> int | None:
    """PID owning the foreground window, or None (locked screen, no window)."""
    if not IS_WINDOWS:
        return None
    hwnd = _user32.GetForegroundWindow()
    if not hwnd:
        return None
    pid = wintypes.DWORD(0)
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value or None


def user_idle_seconds() -> float | None:
    """Seconds since the last keyboard/mouse input in this session."""
    if not IS_WINDOWS:
        return None
    lii = _LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(lii)
    if not _user32.GetLastInputInfo(ctypes.byref(lii)):
        return None
    # Both values are 32-bit millisecond tick counts; mask handles the 49.7-day wrap.
    return ((_kernel32.GetTickCount() - lii.dwTime) & 0xFFFFFFFF) / 1000.0


def enable_efficiency_mode(on: bool = True) -> bool:
    """
    Opt this process into Windows 11 "Efficiency mode" (EcoQoS): the scheduler
    prefers E-cores and low clocks for it, so the monitor never competes with
    a game for P-cores. `on=False` explicitly opts out (the UI probe helper).
    Returns False if unsupported.
    """
    if not IS_WINDOWS:
        return False
    state = _PROCESS_POWER_THROTTLING_STATE(Version=1, ControlMask=0x1, StateMask=0x1 if on else 0x0)
    ok = _kernel32.SetProcessInformation(
        _kernel32.GetCurrentProcess(), 4,  # ProcessPowerThrottling
        ctypes.byref(state), ctypes.sizeof(state))
    return bool(ok)


def processor_times() -> list[tuple[int, int, int]] | None:
    """
    Cumulative time per logical CPU as (total, DPC, interrupt), in 100 ns units.
    DPC/ISR time is driver work that runs ahead of every thread: a storm of it
    on one core stalls whatever is scheduled there (audio crackle, frozen UI).
    """
    if not IS_WINDOWS:
        return None
    n = os.cpu_count() or 1
    buf = (_SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION * n)()
    got = wintypes.ULONG(0)
    if _ntdll.NtQuerySystemInformation(_SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION_CLASS, buf,
                                       ctypes.sizeof(buf), ctypes.byref(got)) < 0:
        return None
    count = got.value // ctypes.sizeof(_SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION)
    return [(c.KernelTime + c.UserTime, c.DpcTime, c.InterruptTime) for c in buf[:count]]


def sleep_clocks() -> tuple[float, float] | None:
    """
    (seconds since boot, the same excluding sleep/hibernate). If the first
    advanced more than the second between two calls, the machine slept.
    """
    if not IS_WINDOWS:
        return None
    unbiased = ctypes.c_uint64(0)
    if not _kernel32.QueryUnbiasedInterruptTime(ctypes.byref(unbiased)):
        return None
    return _kernel32.GetTickCount64() / 1000.0, unbiased.value / 1e7


def boost_current_thread() -> bool:
    """
    Exempt the calling thread from the process's Efficiency Mode (EcoQoS) and
    raise it to normal-app priority, so a timing probe isn't parked on a slow
    E-core behind the monitor's own throttling.
    """
    if not IS_WINDOWS:
        return False
    me = _kernel32.GetCurrentThread()
    state = _THREAD_POWER_THROTTLING_STATE(Version=1, ControlMask=0x1, StateMask=0x0)  # HighQoS
    ok = _kernel32.SetThreadInformation(me, 3,  # ThreadPowerThrottling
                                        ctypes.byref(state), ctypes.sizeof(state))
    return bool(_kernel32.SetThreadPriority(me, 2)) and bool(ok)  # THREAD_PRIORITY_HIGHEST


# ── Window-message probing (responsiveness.py) ───────────────────────

def foreground_window() -> int | None:
    if not IS_WINDOWS:
        return None
    return _user32.GetForegroundWindow() or None


def window_owner(hwnd: int) -> tuple[int, int]:
    """(pid, thread id) of the thread that owns the window and pumps its messages."""
    pid = wintypes.DWORD(0)
    tid = _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value, tid


def window_class(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    return buf.value if _user32.GetClassNameW(hwnd, buf, 256) else ""


def process_image_name(pid: int) -> str | None:
    """Executable name of a process (works for most elevated ones too)."""
    h = _kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buf))
        if not _kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return None
        return os.path.basename(buf.value) or None
    finally:
        _kernel32.CloseHandle(h)


def send_null(hwnd: int, timeout_ms: int) -> tuple[bool, int]:
    """
    Send WM_NULL and wait for the window's thread to process it: (answered,
    last error). Returns at once, unanswered, if Windows already considers
    the window hung (no message pumping for 5 s).
    """
    result = ctypes.c_size_t(0)
    ok = _user32.SendMessageTimeoutW(hwnd, WM_NULL, 0, 0, SMTO_ABORTIFHUNG | SMTO_BLOCK,
                                     timeout_ms, ctypes.byref(result))
    return bool(ok), 0 if ok else ctypes.get_last_error()


def is_hung_window(hwnd: int) -> bool:
    return bool(_user32.IsHungAppWindow(hwnd))


def is_window(hwnd: int) -> bool:
    return bool(_user32.IsWindow(hwnd))


def hung_window_from_ghost(hwnd: int) -> int | None:
    """The frozen window that a "(Not Responding)" ghost window stands in for."""
    if _HungWindowFromGhostWindow is None:
        return None
    return _HungWindowFromGhostWindow(hwnd) or None


def taskbar_window() -> int | None:
    """The taskbar, owned by Explorer's main UI thread."""
    if not IS_WINDOWS:
        return None
    return _user32.FindWindowW("Shell_TrayWnd", None) or None


def dwm_flush() -> bool:
    """Block until the desktop compositor's next frame (≈ one refresh when healthy)."""
    return _dwmapi.DwmFlush() == 0


def fullscreen_app_running() -> bool:
    """A full-screen game, video or presentation owns the display (DWM may not compose)."""
    if not IS_WINDOWS:
        return False
    state = ctypes.c_int(0)
    if _shell32.SHQueryUserNotificationState(ctypes.byref(state)) != 0:
        return False
    return state.value in (2, 3, 4)   # QUNS_BUSY, QUNS_RUNNING_D3D_FULL_SCREEN, QUNS_PRESENTATION_MODE


if IS_WINDOWS:
    assert ctypes.sizeof(_SYSTEM_PROCESS_INFORMATION) == 256, "unexpected struct layout"
    assert ctypes.sizeof(_SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION) == 48, "unexpected struct layout"
