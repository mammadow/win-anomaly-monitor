"""
Thin ctypes wrappers for Windows APIs that psutil either lacks or makes slow.

The important one is `ProcessSnapshotter`. Walking ~400 processes with psutil
costs ~1.3 s on this machine, because protected processes fall back to a
per-process NtQuerySystemInformation call (O(n^2)). A single
NtQuerySystemInformation(SystemProcessInformation) call returns CPU times,
memory, I/O and handle counts for *every* process in a few milliseconds.

Everything here degrades gracefully: on non-Windows platforms (or if an API
call fails) callers get None / empty results and fall back to psutil.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass

IS_WINDOWS = sys.platform == "win32"

_EPOCH_AS_FILETIME = 116_444_736_000_000_000   # 1970-01-01 in 100 ns ticks since 1601
_STATUS_INFO_LENGTH_MISMATCH = -1073741820     # 0xC0000004 as a signed NTSTATUS
_SYSTEM_PROCESS_INFORMATION_CLASS = 5


@dataclass(slots=True)
class ProcInfo:
    pid: int
    name: str
    create_time: float      # epoch seconds (0-ish for System / Idle)
    cpu_time: float         # user + kernel seconds, cumulative
    working_set: int        # bytes resident in RAM
    private_bytes: int      # bytes of committed private memory (what leaks grow)
    read_bytes: int         # cumulative I/O bytes read (disk + network + device)
    write_bytes: int        # cumulative I/O bytes written
    threads: int
    handles: int

    @property
    def key(self) -> tuple[int, float]:
        """PIDs get reused; (pid, create_time) identifies one process instance."""
        return self.pid, self.create_time


if IS_WINDOWS:
    _ntdll = ctypes.WinDLL("ntdll")
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _user32 = ctypes.WinDLL("user32")

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
            ))
            if spi.NextEntryOffset == 0:
                break
            offset += spi.NextEntryOffset
        return out


@dataclass(slots=True)
class PerformanceInfo:
    commit_bytes: int
    commit_limit_bytes: int
    handles: int
    processes: int
    threads: int

    @property
    def commit_percent(self) -> float:
        return 100.0 * self.commit_bytes / self.commit_limit_bytes if self.commit_limit_bytes else 0.0


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


def enable_efficiency_mode() -> bool:
    """
    Opt this process into Windows 11 "Efficiency mode" (EcoQoS): the scheduler
    prefers E-cores and low clocks for it, so the monitor never competes with
    a game for P-cores. Returns False if unsupported.
    """
    if not IS_WINDOWS:
        return False
    state = _PROCESS_POWER_THROTTLING_STATE(Version=1, ControlMask=0x1, StateMask=0x1)
    ok = _kernel32.SetProcessInformation(
        _kernel32.GetCurrentProcess(), 4,  # ProcessPowerThrottling
        ctypes.byref(state), ctypes.sizeof(state))
    return bool(ok)


if IS_WINDOWS:
    assert ctypes.sizeof(_SYSTEM_PROCESS_INFORMATION) == 256, "unexpected struct layout"
