//! Native Win32 & NT kernel bindings for system telemetry, process attribution,
//! UI responsiveness probing, and dynamic NVML GPU monitoring.

#![allow(non_snake_case, non_camel_case_types, clippy::upper_case_acronyms)]

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::mem::{size_of, zeroed};
use std::os::windows::ffi::OsStringExt;
use std::ptr::{null, null_mut};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub type HANDLE = *mut core::ffi::c_void;
pub type HWND = *mut core::ffi::c_void;
pub type BOOL = i32;
pub type ULONG = u32;
pub type DWORD = u32;
pub type USHORT = u16;
pub type NTSTATUS = i32;

const SYSTEM_PERFORMANCE_INFORMATION: ULONG = 2;
const SYSTEM_PROCESS_INFORMATION: ULONG = 5;
const SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION: ULONG = 8;
const STATUS_INFO_LENGTH_MISMATCH: NTSTATUS = 0xC0000004u32 as i32;

const PROCESS_QUERY_LIMITED_INFORMATION: DWORD = 0x1000;
const PROCESS_COMMAND_LINE_INFORMATION: ULONG = 60;

const WM_NULL: u32 = 0x0000;
const SMTO_BLOCK: u32 = 0x0001;

const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
const THREAD_PRIORITY_HIGHEST: i32 = 2;
const THREAD_POWER_THROTTLING: u32 = 4;
const THREAD_POWER_THROTTLING_CURRENT_VERSION: ULONG = 1;
const THREAD_POWER_THROTTLING_EXECUTION_SPEED: ULONG = 0x1;

const SC_MANAGER_ENUMERATE_SERVICE: DWORD = 0x0004;
const SC_ENUM_PROCESS_INFO: u32 = 0;
const SERVICE_WIN32: DWORD = 0x00000030;
const SERVICE_ACTIVE: DWORD = 0x00000001;
const ERROR_MORE_DATA: DWORD = 234;

#[repr(C)]
#[derive(Copy, Clone)]
struct UNICODE_STRING {
    Length: USHORT,
    MaximumLength: USHORT,
    Buffer: *mut u16,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct SYSTEM_PROCESS_INFORMATION_RAW {
    NextEntryOffset: ULONG,
    NumberOfThreads: ULONG,
    WorkingSetPrivateSize: i64,
    HardFaultCount: ULONG,
    NumberOfThreadsHighWatermark: ULONG,
    CycleTime: u64,
    CreateTime: i64,
    UserTime: i64,
    KernelTime: i64,
    ImageName: UNICODE_STRING,
    BasePriority: i32,
    UniqueProcessId: usize,
    InheritedFromUniqueProcessId: usize,
    HandleCount: ULONG,
    SessionId: ULONG,
    UniqueProcessKey: usize,
    PeakVirtualSize: usize,
    VirtualSize: usize,
    PageFaultCount: ULONG,
    PeakWorkingSetSize: usize,
    WorkingSetSize: usize,
    QuotaPeakPagedPoolUsage: usize,
    QuotaPagedPoolUsage: usize,
    QuotaPeakNonPagedPoolUsage: usize,
    QuotaNonPagedPoolUsage: usize,
    PagefileUsage: usize,
    PeakPagefileUsage: usize,
    PrivatePageCount: usize,
    ReadOperationCount: i64,
    WriteOperationCount: i64,
    OtherOperationCount: i64,
    ReadTransferCount: i64,
    WriteTransferCount: i64,
    OtherTransferCount: i64,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION_RAW {
    IdleTime: i64,
    KernelTime: i64,
    UserTime: i64,
    DpcTime: i64,
    InterruptTime: i64,
    InterruptCount: ULONG,
}

/// Prefix of `SYSTEM_PERFORMANCE_INFORMATION` (class 2, 312+ bytes).
#[repr(C)]
#[derive(Copy, Clone)]
struct SYSTEM_PERFORMANCE_INFORMATION_PREFIX {
    IdleProcessTime: i64,
    IoReadTransferCount: i64,
    IoWriteTransferCount: i64,
    IoOtherTransferCount: i64,
    IoReadOperationCount: ULONG,
    IoWriteOperationCount: ULONG,
    IoOtherOperationCount: ULONG,
    AvailablePages: ULONG,
    CommittedPages: ULONG,
    CommitLimit: ULONG,
    PeakCommitment: ULONG,
    PageFaultCount: ULONG,
    CopyOnWriteCount: ULONG,
    TransitionCount: ULONG,
    CacheTransitionCount: ULONG,
    DemandZeroCount: ULONG,
    PageReadCount: ULONG,
    PageReadIoCount: ULONG,
    _reserved: [u8; 256],
}

#[repr(C)]
struct THREAD_POWER_THROTTLING_STATE {
    Version: ULONG,
    ControlMask: ULONG,
    StateMask: ULONG,
}

#[repr(C)]
struct MEMORYSTATUSEX {
    dwLength: DWORD,
    dwMemoryLoad: DWORD,
    ullTotalPhys: u64,
    ullAvailPhys: u64,
    ullTotalPageFile: u64,
    ullAvailPageFile: u64,
    ullTotalVirtual: u64,
    ullAvailVirtual: u64,
    ullAvailExtendedVirtual: u64,
}

#[repr(C)]
struct PERFORMANCE_INFORMATION {
    cb: DWORD,
    CommitTotal: usize,
    CommitLimit: usize,
    CommitPeak: usize,
    PhysicalTotal: usize,
    PhysicalAvailable: usize,
    SystemCache: usize,
    KernelTotal: usize,
    KernelPaged: usize,
    KernelNonpaged: usize,
    PageSize: usize,
    HandleCount: DWORD,
    ProcessCount: DWORD,
    ThreadCount: DWORD,
}

#[repr(C)]
struct SYSTEM_POWER_STATUS {
    ACLineStatus: u8,
    BatteryFlag: u8,
    BatteryLifePercent: u8,
    SystemStatusFlag: u8,
    BatteryLifeTime: DWORD,
    BatteryFullLifeTime: DWORD,
}

#[repr(C)]
struct LASTINPUTINFO {
    cbSize: u32,
    dwTime: DWORD,
}

#[repr(C)]
struct SERVICE_STATUS_PROCESS {
    dwServiceType: DWORD,
    dwCurrentState: DWORD,
    dwControlsAccepted: DWORD,
    dwWin32ExitCode: DWORD,
    dwServiceSpecificExitCode: DWORD,
    dwCheckPoint: DWORD,
    dwWaitHint: DWORD,
    dwProcessId: DWORD,
    dwServiceFlags: DWORD,
}

#[repr(C)]
struct ENUM_SERVICE_STATUS_PROCESSW {
    lpServiceName: *mut u16,
    lpDisplayName: *mut u16,
    ServiceStatusProcess: SERVICE_STATUS_PROCESS,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct SYSTEM_BATTERY_STATE {
    AcOnLine: u8,
    BatteryPresent: u8,
    Charging: u8,
    Discharging: u8,
    Spare1: [u8; 3],
    Tag: u8,
    MaxCapacity: u32,
    RemainingCapacity: u32,
    Rate: i32,
    EstimatedTime: u32,
    DefaultAlert1: u32,
    DefaultAlert2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct PROCESSOR_POWER_INFORMATION {
    Number: u32,
    MaxMhz: u32,
    CurrentMhz: u32,
    MhzLimit: u32,
    MaxIdleState: u32,
    CurrentIdleState: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct MIB_IF_ROW2 {
    InterfaceLuid: u64,
    InterfaceIndex: u32,
    InterfaceGuid: [u8; 16],
    Alias: [u16; 257],
    Description: [u16; 257],
    PhysicalAddressLength: u32,
    PhysicalAddress: [u8; 32],
    PermanentPhysicalAddress: [u8; 32],
    Mtu: u32,
    Type: u32,
    TunnelType: u32,
    MediaType: u32,
    PhysicalMediumType: u32,
    AccessType: u32,
    DirectionType: u32,
    InterfaceAndOperStatusFlags: u8,
    OperStatus: u32,
    AdminStatus: u32,
    MediaConnectState: u32,
    NetworkGuid: [u8; 16],
    ConnectionType: u32,
    TransmitLinkSpeed: u64,
    ReceiveLinkSpeed: u64,
    InOctets: u64,
    InUcastPkts: u64,
    InNUcastPkts: u64,
    InDiscards: u64,
    InErrors: u64,
    InUnknownProtos: u64,
    InUcastOctets: u64,
    InMulticastOctets: u64,
    InBroadcastOctets: u64,
    OutOctets: u64,
    OutUcastPkts: u64,
    OutNUcastPkts: u64,
    OutDiscards: u64,
    OutErrors: u64,
    OutUcastOctets: u64,
    OutMulticastOctets: u64,
    OutBroadcastOctets: u64,
    OutQLen: u64,
}

#[repr(C)]
struct MIB_IF_TABLE2 {
    NumEntries: u32,
    Table: [MIB_IF_ROW2; 1],
}

#[link(name = "ntdll")]
extern "system" {
    fn NtQuerySystemInformation(
        SystemInformationClass: ULONG,
        SystemInformation: *mut core::ffi::c_void,
        SystemInformationLength: ULONG,
        ReturnLength: *mut ULONG,
    ) -> NTSTATUS;

    fn NtQueryInformationProcess(
        ProcessHandle: HANDLE,
        ProcessInformationClass: ULONG,
        ProcessInformation: *mut core::ffi::c_void,
        ProcessInformationLength: ULONG,
        ReturnLength: *mut ULONG,
    ) -> NTSTATUS;
}

#[link(name = "kernel32")]
extern "system" {
    fn OpenProcess(dwDesiredAccess: DWORD, bInheritHandle: BOOL, dwProcessId: DWORD) -> HANDLE;
    fn CloseHandle(hObject: HANDLE) -> BOOL;
    fn QueryFullProcessImageNameW(
        hProcess: HANDLE,
        dwFlags: DWORD,
        lpExeName: *mut u16,
        lpdwSize: *mut DWORD,
    ) -> BOOL;
    fn GetCurrentThread() -> HANDLE;
    fn SetThreadPriority(hThread: HANDLE, nPriority: i32) -> BOOL;
    fn SetThreadInformation(
        hThread: HANDLE,
        ThreadInformationClass: u32,
        ThreadInformation: *const core::ffi::c_void,
        ThreadInformationSize: DWORD,
    ) -> BOOL;
    fn GetTickCount64() -> u64;
    fn GetTickCount() -> DWORD;
    fn QueryUnbiasedInterruptTime(UnbiasedTime: *mut u64) -> BOOL;
    fn GetLastError() -> DWORD;
    fn GlobalMemoryStatusEx(lpBuffer: *mut MEMORYSTATUSEX) -> BOOL;
    fn GetSystemPowerStatus(lpSystemPowerStatus: *mut SYSTEM_POWER_STATUS) -> BOOL;
    fn GetDiskFreeSpaceExW(
        lpDirectoryName: *const u16,
        lpFreeBytesAvailableToCaller: *mut u64,
        lpTotalNumberOfBytes: *mut u64,
        lpTotalNumberOfFreeBytes: *mut u64,
    ) -> BOOL;
    fn GetCurrentProcessId() -> DWORD;
    fn GetModuleHandleW(lpModuleName: *const u16) -> HANDLE;
    fn GetProcAddress(hModule: HANDLE, lpProcName: *const u8) -> *const core::ffi::c_void;
    fn SetConsoleCtrlHandler(
        HandlerRoutine: Option<unsafe extern "system" fn(DWORD) -> BOOL>,
        Add: BOOL,
    ) -> BOOL;
}

#[link(name = "psapi")]
extern "system" {
    fn GetPerformanceInfo(pPerformanceInformation: *mut PERFORMANCE_INFORMATION, cb: DWORD)
        -> BOOL;
    fn EnumDeviceDrivers(
        lpImageBase: *mut *mut core::ffi::c_void,
        cb: DWORD,
        lpcbNeeded: *mut DWORD,
    ) -> BOOL;
    fn GetDeviceDriverBaseNameW(
        ImageBase: *mut core::ffi::c_void,
        lpFilename: *mut u16,
        nSize: DWORD,
    ) -> DWORD;
}

#[link(name = "powrprof")]
extern "system" {
    fn CallNtPowerInformation(
        InformationLevel: i32,
        InputBuffer: *mut core::ffi::c_void,
        InputBufferLength: u32,
        OutputBuffer: *mut core::ffi::c_void,
        OutputBufferLength: u32,
    ) -> NTSTATUS;
}

#[link(name = "user32")]
extern "system" {
    fn GetForegroundWindow() -> HWND;
    fn FindWindowW(lpClassName: *const u16, lpWindowName: *const u16) -> HWND;
    fn GetWindowThreadProcessId(hWnd: HWND, lpdwProcessId: *mut DWORD) -> DWORD;
    fn GetWindowTextW(hWnd: HWND, lpString: *mut u16, nMaxCount: i32) -> i32;
    fn GetClassNameW(hWnd: HWND, lpClassName: *mut u16, nMaxCount: i32) -> i32;
    fn SendMessageTimeoutW(
        hWnd: HWND,
        Msg: u32,
        wParam: usize,
        lParam: isize,
        fuFlags: u32,
        uTimeout: u32,
        lpdwResult: *mut usize,
    ) -> isize;
    fn IsHungAppWindow(hWnd: HWND) -> BOOL;
    fn GetLastInputInfo(plii: *mut LASTINPUTINFO) -> BOOL;
}

#[link(name = "dwmapi")]
extern "system" {
    fn DwmFlush() -> i32;
}

#[link(name = "advapi32")]
extern "system" {
    fn OpenSCManagerW(
        lpMachineName: *const u16,
        lpDatabaseName: *const u16,
        dwDesiredAccess: DWORD,
    ) -> HANDLE;
    fn CloseServiceHandle(hSCObject: HANDLE) -> BOOL;
    fn EnumServicesStatusExW(
        hSCManager: HANDLE,
        InfoLevel: u32,
        dwServiceType: DWORD,
        dwServiceState: DWORD,
        lpServices: *mut u8,
        cbBufSize: DWORD,
        pcbBytesNeeded: *mut DWORD,
        lpServicesReturned: *mut DWORD,
        lpResumeHandle: *mut DWORD,
        pszGroupName: *const u16,
    ) -> BOOL;
}

#[link(name = "iphlpapi")]
extern "system" {
    fn GetIfTable2(Table: *mut *mut MIB_IF_TABLE2) -> DWORD;
    fn FreeMibTable(Memory: *mut core::ffi::c_void);
}

#[link(name = "pdh")]
extern "system" {
    fn PdhOpenQueryW(szDataSource: *const u16, dwUserData: usize, phQuery: *mut HANDLE) -> DWORD;
    fn PdhAddEnglishCounterW(
        hQuery: HANDLE,
        szFullCounterPath: *const u16,
        dwUserData: usize,
        phCounter: *mut HANDLE,
    ) -> DWORD;
    fn PdhCollectQueryData(hQuery: HANDLE) -> DWORD;
    fn PdhGetFormattedCounterValue(
        hCounter: HANDLE,
        dwFormat: DWORD,
        lpdwType: *mut DWORD,
        pValue: *mut PDH_FMT_COUNTERVALUE,
    ) -> DWORD;
    fn PdhCloseQuery(hQuery: HANDLE) -> DWORD;
}

const PDH_FMT_DOUBLE: DWORD = 0x00000200;

#[repr(C)]
struct PDH_FMT_COUNTERVALUE {
    CStatus: DWORD,
    doubleValue: f64,
}

/// Physical Disk I/O rate sampler using Windows Performance Data Helper (PDH).
pub struct PdhDiskSampler {
    query: HANDLE,
    read_counter: HANDLE,
    write_counter: HANDLE,
    primed: bool,
}

unsafe impl Send for PdhDiskSampler {}

impl PdhDiskSampler {
    pub fn new() -> Option<Self> {
        unsafe {
            let mut query: HANDLE = null_mut();
            if PdhOpenQueryW(null(), 0, &mut query) != 0 || query.is_null() {
                return None;
            }
            let read_path: Vec<u16> = "\\PhysicalDisk(_Total)\\Disk Read Bytes/sec\0"
                .encode_utf16()
                .collect();
            let write_path: Vec<u16> = "\\PhysicalDisk(_Total)\\Disk Write Bytes/sec\0"
                .encode_utf16()
                .collect();
            let mut read_counter: HANDLE = null_mut();
            let mut write_counter: HANDLE = null_mut();
            if PdhAddEnglishCounterW(query, read_path.as_ptr(), 0, &mut read_counter) != 0
                || PdhAddEnglishCounterW(query, write_path.as_ptr(), 0, &mut write_counter) != 0
            {
                let _ = PdhCloseQuery(query);
                return None;
            }
            let _ = PdhCollectQueryData(query);
            Some(Self {
                query,
                read_counter,
                write_counter,
                primed: true,
            })
        }
    }

    pub fn sample_rates(&mut self) -> Option<(f64, f64)> {
        unsafe {
            if PdhCollectQueryData(self.query) != 0 {
                return None;
            }
            if !self.primed {
                self.primed = true;
                return None;
            }
            let mut val_r: PDH_FMT_COUNTERVALUE = zeroed();
            let mut val_w: PDH_FMT_COUNTERVALUE = zeroed();
            let r_ok = PdhGetFormattedCounterValue(
                self.read_counter,
                PDH_FMT_DOUBLE,
                null_mut(),
                &mut val_r,
            ) == 0;
            let w_ok = PdhGetFormattedCounterValue(
                self.write_counter,
                PDH_FMT_DOUBLE,
                null_mut(),
                &mut val_w,
            ) == 0;
            if r_ok && w_ok {
                Some((val_r.doubleValue.max(0.0), val_w.doubleValue.max(0.0)))
            } else {
                None
            }
        }
    }
}

impl Drop for PdhDiskSampler {
    fn drop(&mut self) {
        if !self.query.is_null() {
            unsafe {
                let _ = PdhCloseQuery(self.query);
            }
        }
    }
}

/// Raw process entry from `NtQuerySystemInformation(SystemProcessInformation)`.
#[derive(Debug, Clone)]
pub struct RawProc {
    pub pid: u32,
    pub create_time: i64,
    pub name: String,
    pub cpu_100ns: i64,
    pub io_read: i64,
    pub io_write: i64,
    pub io_other: i64,
    pub working_set: u64,
    pub private_bytes: u64,
    pub threads: u32,
    pub handles: u32,
    pub hard_faults: u32,
}

/// Snapshot of every running process in one kernel call (~1 ms).
pub fn iter_processes(buf: &mut Vec<u8>) -> Vec<RawProc> {
    if buf.len() < 1024 * 1024 {
        buf.resize(1024 * 1024, 0);
    }
    let mut needed: ULONG = 0;
    loop {
        let status = unsafe {
            NtQuerySystemInformation(
                SYSTEM_PROCESS_INFORMATION,
                buf.as_mut_ptr() as *mut _,
                buf.len() as ULONG,
                &mut needed,
            )
        };
        if status == STATUS_INFO_LENGTH_MISMATCH {
            let new_len = (needed as usize + 65536).max(buf.len() * 2);
            buf.resize(new_len, 0);
            continue;
        }
        if status != 0 {
            return Vec::new();
        }
        break;
    }

    let mut out = Vec::with_capacity(300);
    let mut offset: usize = 0;
    while offset + size_of::<SYSTEM_PROCESS_INFORMATION_RAW>() <= buf.len() {
        let e = unsafe { &*(buf.as_ptr().add(offset) as *const SYSTEM_PROCESS_INFORMATION_RAW) };
        let pid = e.UniqueProcessId as u32;
        if pid != 0 {
            let name = if !e.ImageName.Buffer.is_null() && e.ImageName.Length > 0 {
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        e.ImageName.Buffer,
                        (e.ImageName.Length / 2) as usize,
                    )
                };
                OsString::from_wide(slice).to_string_lossy().into_owned()
            } else {
                "System".to_string()
            };
            out.push(RawProc {
                pid,
                create_time: e.CreateTime,
                name,
                cpu_100ns: e.UserTime + e.KernelTime,
                io_read: e.ReadTransferCount,
                io_write: e.WriteTransferCount,
                io_other: e.OtherTransferCount,
                working_set: e.WorkingSetSize as u64,
                private_bytes: e.PrivatePageCount as u64,
                threads: e.NumberOfThreads,
                handles: e.HandleCount,
                hard_faults: e.HardFaultCount,
            });
        }
        if e.NextEntryOffset == 0 {
            break;
        }
        offset += e.NextEntryOffset as usize;
    }
    out
}

/// Runtime/scripting host executables whose command line should be inspected to
/// disambiguate which script, module, or app they are running.
pub fn needs_cmdline_tag(exe_name: &str) -> bool {
    let lower = exe_name.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "python.exe"
            | "pythonw.exe"
            | "python3.exe"
            | "node.exe"
            | "bun.exe"
            | "deno.exe"
            | "pwsh.exe"
            | "powershell.exe"
            | "cmd.exe"
            | "java.exe"
            | "javaw.exe"
            | "dotnet.exe"
            | "ruby.exe"
            | "wsl.exe"
    )
}

/// Query a process's command line via `NtQueryInformationProcess(ProcessCommandLineInformation = 60)`
/// and extract a concise script/module/argument tag (e.g. `"uvicorn"` or `"train.py"`).
pub fn query_process_cmdline_tag(pid: u32) -> Option<String> {
    let raw = query_process_cmdline_raw(pid)?;
    extract_cmdline_tag(&raw)
}

pub fn query_process_cmdline_raw(pid: u32) -> Option<String> {
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            return None;
        }
        let mut needed: ULONG = 0;
        let mut buf = vec![0u8; 2048];
        let mut status = NtQueryInformationProcess(
            h,
            PROCESS_COMMAND_LINE_INFORMATION,
            buf.as_mut_ptr() as *mut _,
            buf.len() as ULONG,
            &mut needed,
        );
        if status == STATUS_INFO_LENGTH_MISMATCH && needed > 0 && needed < 65536 {
            buf.resize(needed as usize + 16, 0);
            status = NtQueryInformationProcess(
                h,
                PROCESS_COMMAND_LINE_INFORMATION,
                buf.as_mut_ptr() as *mut _,
                buf.len() as ULONG,
                &mut needed,
            );
        }
        let _ = CloseHandle(h);
        if status != 0 || buf.len() < size_of::<UNICODE_STRING>() {
            return None;
        }
        let ustr = &*(buf.as_ptr() as *const UNICODE_STRING);
        if ustr.Buffer.is_null() || ustr.Length == 0 {
            return None;
        }
        let slice = std::slice::from_raw_parts(ustr.Buffer, (ustr.Length / 2) as usize);
        Some(OsString::from_wide(slice).to_string_lossy().into_owned())
    }
}

/// Parse a raw Windows command line string into a short, human-meaningful tag.
pub fn extract_cmdline_tag(cmdline: &str) -> Option<String> {
    let tokens = split_windows_cmdline(cmdline);
    if tokens.len() <= 1 {
        return None;
    }
    let mut i = 1;
    while i < tokens.len() {
        let tok = tokens[i].trim();
        if tok.is_empty() {
            i += 1;
            continue;
        }
        if tok == "-m" || tok == "-jar" {
            if let Some(next) = tokens.get(i + 1) {
                return Some(shorten_token(next));
            }
        }
        if tok == "-c" || tok == "-Command" || tok == "-EncodedCommand" {
            if let Some(next) = tokens.get(i + 1) {
                return Some(shorten_token(next));
            }
            return Some(tok.to_string());
        }
        if tok.starts_with('-') || tok.starts_with('/') {
            // Skip common runtime flags
            if matches!(
                tok,
                "-File" | "--inspect" | "--max-old-space-size" | "-cp" | "-classpath"
            ) {
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        return Some(shorten_token(tok));
    }
    None
}

fn split_windows_cmdline(cmdline: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    for ch in cmdline.chars() {
        match ch {
            '"' => in_quotes = !in_quotes,
            ' ' | '\t' if !in_quotes => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(ch),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn shorten_token(tok: &str) -> String {
    let normalized = tok.replace('\\', "/");
    let parts: Vec<&str> = normalized.split('/').filter(|s| !s.is_empty()).collect();
    let label = if parts.len() >= 2 {
        let last = parts[parts.len() - 1];
        let prev = parts[parts.len() - 2];
        if matches!(
            last.to_ascii_lowercase().as_str(),
            "index.js"
                | "index.ts"
                | "main.py"
                | "__init__.py"
                | "__main__.py"
                | "server.js"
                | "cli.js"
        ) && prev != "."
        {
            format!("{}/{}", prev, last)
        } else {
            last.to_string()
        }
    } else if let Some(&last) = parts.last() {
        last.to_string()
    } else {
        tok.to_string()
    };
    if label.chars().count() > 28 {
        let s: String = label.chars().take(25).collect();
        format!("{}...", s)
    } else {
        label
    }
}

/// Map `svchost.exe` PIDs to their hosted Windows Service names, cached for 30 seconds.
pub fn svchost_service_map() -> HashMap<u32, String> {
    static CACHE: Mutex<Option<(Instant, HashMap<u32, String>)>> = Mutex::new(None);
    if let Ok(guard) = CACHE.lock() {
        if let Some((ts, ref map)) = *guard {
            if ts.elapsed() < Duration::from_secs(30) {
                return map.clone();
            }
        }
    }
    let map = query_svchost_services();
    if let Ok(mut guard) = CACHE.lock() {
        *guard = Some((Instant::now(), map.clone()));
    }
    map
}

fn query_svchost_services() -> HashMap<u32, String> {
    let mut by_pid: HashMap<u32, Vec<String>> = HashMap::new();
    unsafe {
        let scm = OpenSCManagerW(null(), null(), SC_MANAGER_ENUMERATE_SERVICE);
        if scm.is_null() {
            return HashMap::new();
        }
        let mut bytes_needed: DWORD = 0;
        let mut count: DWORD = 0;
        let mut resume: DWORD = 0;
        let _ = EnumServicesStatusExW(
            scm,
            SC_ENUM_PROCESS_INFO,
            SERVICE_WIN32,
            SERVICE_ACTIVE,
            null_mut(),
            0,
            &mut bytes_needed,
            &mut count,
            &mut resume,
            null(),
        );
        if GetLastError() == ERROR_MORE_DATA && bytes_needed > 0 && bytes_needed < 4 * 1024 * 1024 {
            let mut buf = vec![0u8; bytes_needed as usize];
            resume = 0;
            let ok = EnumServicesStatusExW(
                scm,
                SC_ENUM_PROCESS_INFO,
                SERVICE_WIN32,
                SERVICE_ACTIVE,
                buf.as_mut_ptr(),
                buf.len() as DWORD,
                &mut bytes_needed,
                &mut count,
                &mut resume,
                null(),
            );
            if ok != 0 && count > 0 {
                let entries = std::slice::from_raw_parts(
                    buf.as_ptr() as *const ENUM_SERVICE_STATUS_PROCESSW,
                    count as usize,
                );
                for entry in entries {
                    let pid = entry.ServiceStatusProcess.dwProcessId;
                    if pid != 0 && !entry.lpServiceName.is_null() {
                        let name = wide_ptr_to_string(entry.lpServiceName);
                        if !name.is_empty() {
                            by_pid.entry(pid).or_default().push(name);
                        }
                    }
                }
            }
        }
        let _ = CloseServiceHandle(scm);
    }

    let mut out = HashMap::new();
    for (pid, mut names) in by_pid {
        names.sort();
        let tag = if names.len() <= 2 {
            names.join(", ")
        } else {
            format!("{}, {} +{}", names[0], names[1], names.len() - 2)
        };
        out.insert(pid, tag);
    }
    out
}

unsafe fn wide_ptr_to_string(ptr: *const u16) -> String {
    if ptr.is_null() {
        return String::new();
    }
    let mut len = 0usize;
    while len < 512 && *ptr.add(len) != 0 {
        len += 1;
    }
    let slice = std::slice::from_raw_parts(ptr, len);
    OsString::from_wide(slice).to_string_lossy().into_owned()
}

/// Per-core 100-ns tick counters from `SystemProcessorPerformanceInformation`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CoreTickTimes {
    pub idle: i64,
    pub kernel: i64,
    pub user: i64,
    pub dpc: i64,
    pub interrupt: i64,
}

/// Query per-logical-processor CPU, DPC, and ISR cumulative 100-ns tick counters.
pub fn per_core_times() -> Option<Vec<CoreTickTimes>> {
    let n_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1);
    let mut entries = vec![SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION_RAW::default(); n_cpus + 16];
    let mut ret_len: ULONG = 0;
    let status = unsafe {
        NtQuerySystemInformation(
            SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION,
            entries.as_mut_ptr() as *mut _,
            (entries.len() * size_of::<SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION_RAW>()) as ULONG,
            &mut ret_len,
        )
    };
    if status != 0 {
        return None;
    }
    let actual_count =
        (ret_len as usize) / size_of::<SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION_RAW>();
    if actual_count == 0 {
        return None;
    }
    Some(
        entries[..actual_count]
            .iter()
            .map(|c| CoreTickTimes {
                idle: c.IdleTime,
                kernel: c.KernelTime,
                user: c.UserTime,
                dpc: c.DpcTime,
                interrupt: c.InterruptTime,
            })
            .collect(),
    )
}

/// System-wide I/O transfer counters and hard page fault count from `SystemPerformanceInformation`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemPerfCounters {
    pub io_read_bytes: u64,
    pub io_write_bytes: u64,
    pub hard_page_faults: u32,
}

pub fn system_perf_counters() -> Option<SystemPerfCounters> {
    unsafe {
        let mut info: SYSTEM_PERFORMANCE_INFORMATION_PREFIX = zeroed();
        let mut ret_len: ULONG = 0;
        let status = NtQuerySystemInformation(
            SYSTEM_PERFORMANCE_INFORMATION,
            &mut info as *mut _ as *mut _,
            size_of::<SYSTEM_PERFORMANCE_INFORMATION_PREFIX>() as ULONG,
            &mut ret_len,
        );
        // Either 0 (STATUS_SUCCESS) or STATUS_INFO_LENGTH_MISMATCH if prefix was filled
        if status == 0 || ret_len >= 88 {
            Some(SystemPerfCounters {
                io_read_bytes: info.IoReadTransferCount.max(0) as u64,
                io_write_bytes: info.IoWriteTransferCount.max(0) as u64,
                hard_page_faults: info.PageReadCount,
            })
        } else {
            None
        }
    }
}

/// Query RAM usage % and Commit charge % (RAM + pagefile).
pub fn memory_and_commit_pct() -> (f64, f64) {
    let mut ram_pct = 0.0;
    unsafe {
        let mut mem: MEMORYSTATUSEX = zeroed();
        mem.dwLength = size_of::<MEMORYSTATUSEX>() as DWORD;
        if GlobalMemoryStatusEx(&mut mem) != 0 && mem.ullTotalPhys > 0 {
            let used = mem.ullTotalPhys.saturating_sub(mem.ullAvailPhys);
            ram_pct = (used as f64) / (mem.ullTotalPhys as f64) * 100.0;
        }
    }

    let mut commit_pct = f64::NAN;
    unsafe {
        let mut pi: PERFORMANCE_INFORMATION = zeroed();
        pi.cb = size_of::<PERFORMANCE_INFORMATION>() as DWORD;
        if GetPerformanceInfo(&mut pi, pi.cb) != 0 && pi.CommitLimit > 0 {
            commit_pct = (pi.CommitTotal as f64) / (pi.CommitLimit as f64) * 100.0;
        }
    }
    (ram_pct, commit_pct)
}

/// Total network bytes sent and received across all non-loopback interfaces via `GetIfTable2`.
pub fn network_octets() -> Option<(u64, u64)> {
    unsafe {
        let mut table_ptr: *mut MIB_IF_TABLE2 = null_mut();
        if GetIfTable2(&mut table_ptr) != 0 || table_ptr.is_null() {
            return None;
        }
        let num = (*table_ptr).NumEntries as usize;
        let rows_ptr = (*table_ptr).Table.as_ptr();
        let rows = std::slice::from_raw_parts(rows_ptr, num);
        let mut sent: u64 = 0;
        let mut recv: u64 = 0;
        for row in rows {
            // Skip software loopback (IF_TYPE_SOFTWARE_LOOPBACK = 24)
            if row.Type == 24 {
                continue;
            }
            sent = sent.wrapping_add(row.OutOctets);
            recv = recv.wrapping_add(row.InOctets);
        }
        FreeMibTable(table_ptr as *mut _);
        Some((sent, recv))
    }
}

/// Free disk space in GiB on the system drive (`C:\`).
pub fn system_drive_free_gb() -> f64 {
    let drive = std::env::var("SystemDrive").unwrap_or_else(|_| "C:".to_string());
    let path_str = format!("{}\\\0", drive);
    let wide: Vec<u16> = path_str.encode_utf16().collect();
    let mut free_avail: u64 = 0;
    let mut total: u64 = 0;
    let mut total_free: u64 = 0;
    unsafe {
        if GetDiskFreeSpaceExW(wide.as_ptr(), &mut free_avail, &mut total, &mut total_free) != 0 {
            (free_avail as f64) / ((1u64 << 30) as f64)
        } else {
            f64::NAN
        }
    }
}

/// Returns `1.0` if on AC power (or desktop with no battery), `0.0` if on battery.
pub fn on_ac_power() -> f64 {
    unsafe {
        let mut st: SYSTEM_POWER_STATUS = zeroed();
        if GetSystemPowerStatus(&mut st) != 0 {
            // ACLineStatus == 0 means Offline (on battery); 1 = Online, 255 = Unknown/Desktop
            if st.ACLineStatus == 0 {
                0.0
            } else {
                1.0
            }
        } else {
            1.0
        }
    }
}

/// Live battery state and per-core CPU frequency telemetry from `powrprof.dll`
/// (`CallNtPowerInformation`), requiring zero elevation and zero kernel drivers.
#[derive(Debug, Clone, Copy)]
pub struct PowerTelemetry {
    pub battery_present: bool,
    pub battery_pct: f64,
    pub battery_rate_w: f64,
    pub battery_charging: bool,
    pub battery_discharging: bool,
    pub cpu_mhz: f64,
    pub cpu_max_mhz: f64,
    pub cpu_mhz_limit: f64,
}

impl Default for PowerTelemetry {
    fn default() -> Self {
        Self {
            battery_present: false,
            battery_pct: f64::NAN,
            battery_rate_w: f64::NAN,
            battery_charging: false,
            battery_discharging: false,
            cpu_mhz: f64::NAN,
            cpu_max_mhz: f64::NAN,
            cpu_mhz_limit: f64::NAN,
        }
    }
}

/// Query battery charge/wattage (`SystemBatteryState = 5`) and CPU frequencies (`ProcessorInformation = 11`).
pub fn power_telemetry() -> PowerTelemetry {
    const SYSTEM_BATTERY_STATE_LEVEL: i32 = 5;
    const PROCESSOR_INFORMATION_LEVEL: i32 = 11;

    let mut out = PowerTelemetry::default();
    unsafe {
        let mut bat = SYSTEM_BATTERY_STATE::default();
        let st_bat = CallNtPowerInformation(
            SYSTEM_BATTERY_STATE_LEVEL,
            null_mut(),
            0,
            (&mut bat as *mut SYSTEM_BATTERY_STATE).cast(),
            size_of::<SYSTEM_BATTERY_STATE>() as u32,
        );
        if st_bat == 0 && bat.BatteryPresent != 0 && bat.MaxCapacity > 0 {
            out.battery_present = true;
            out.battery_pct = ((bat.RemainingCapacity as f64) / (bat.MaxCapacity as f64) * 100.0)
                .clamp(0.0, 100.0);
            out.battery_charging = bat.Charging != 0;
            out.battery_discharging = bat.Discharging != 0;
            // BATTERY_UNKNOWN_RATE is i32::MIN (0x80000000); guard absurd/unknown readings.
            if bat.Rate != i32::MIN && bat.Rate.saturating_abs() <= 500_000 {
                out.battery_rate_w = (bat.Rate as f64) / 1000.0;
            }
        }

        let n_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, 512);
        let mut cpus = vec![PROCESSOR_POWER_INFORMATION::default(); n_cpus];
        let st_cpu = CallNtPowerInformation(
            PROCESSOR_INFORMATION_LEVEL,
            null_mut(),
            0,
            cpus.as_mut_ptr().cast(),
            (size_of::<PROCESSOR_POWER_INFORMATION>() * n_cpus) as u32,
        );
        if st_cpu == 0 {
            let valid: Vec<&PROCESSOR_POWER_INFORMATION> =
                cpus.iter().filter(|c| c.MaxMhz > 0).collect();
            if !valid.is_empty() {
                let cnt = valid.len() as f64;
                out.cpu_mhz = valid.iter().map(|c| c.CurrentMhz as f64).sum::<f64>() / cnt;
                out.cpu_max_mhz = valid.iter().map(|c| c.MaxMhz as f64).sum::<f64>() / cnt;
                out.cpu_mhz_limit = valid.iter().map(|c| c.MhzLimit as f64).sum::<f64>() / cnt;
            }
        }
    }
    out
}

/// Seconds since the user last touched keyboard or mouse.
pub fn user_idle_seconds() -> f64 {
    unsafe {
        let mut lii = LASTINPUTINFO {
            cbSize: size_of::<LASTINPUTINFO>() as u32,
            dwTime: 0,
        };
        if GetLastInputInfo(&mut lii) == 0 {
            return 0.0;
        }
        let now_tick = GetTickCount();
        let elapsed_ms = now_tick.wrapping_sub(lii.dwTime);
        (elapsed_ms as f64) / 1000.0
    }
}

/// Current foreground window handle (`None` if no window is active).
pub fn foreground_window() -> Option<usize> {
    let h = unsafe { GetForegroundWindow() };
    if h.is_null() {
        None
    } else {
        Some(h as usize)
    }
}

/// Explorer's taskbar window (`Shell_TrayWnd`).
pub fn taskbar_window() -> Option<usize> {
    let cls: Vec<u16> = "Shell_TrayWnd\0".encode_utf16().collect();
    let h = unsafe { FindWindowW(cls.as_ptr(), null()) };
    if h.is_null() {
        None
    } else {
        Some(h as usize)
    }
}

/// `(pid, tid)` owning `hwnd` (`(0, 0)` if invalid).
pub fn window_owner(hwnd: usize) -> (u32, u32) {
    let mut pid: DWORD = 0;
    let tid = unsafe { GetWindowThreadProcessId(hwnd as HWND, &mut pid) };
    if tid == 0 {
        (0, 0)
    } else {
        (pid, tid)
    }
}

/// Window title text (reads window struct without pumping the target thread's message queue).
pub fn window_title(hwnd: usize) -> String {
    let mut buf = [0u16; 256];
    let n = unsafe { GetWindowTextW(hwnd as HWND, buf.as_mut_ptr(), buf.len() as i32) };
    if n > 0 {
        OsString::from_wide(&buf[..n as usize])
            .to_string_lossy()
            .into_owned()
    } else {
        String::new()
    }
}

/// Executable basename (e.g. `"chrome.exe"`) for `pid` via `QueryFullProcessImageNameW`.
pub fn process_image_name(pid: u32) -> Option<String> {
    if pid == 0 {
        return None;
    }
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            return None;
        }
        let mut buf = [0u16; 520];
        let mut size = buf.len() as DWORD;
        let ok = QueryFullProcessImageNameW(h, 0, buf.as_mut_ptr(), &mut size);
        let _ = CloseHandle(h);
        if ok == 0 || size == 0 {
            return None;
        }
        let full = OsString::from_wide(&buf[..size as usize])
            .to_string_lossy()
            .into_owned();
        let base = full.rsplit(['\\', '/']).next().unwrap_or(&full).to_string();
        Some(base)
    }
}

/// Check if a PID is currently alive on the system.
pub fn pid_exists(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    process_image_name(pid).is_some()
}

/// Current process ID.
pub fn current_pid() -> u32 {
    unsafe { GetCurrentProcessId() }
}

/// Send `WM_NULL` to `hwnd` with `SMTO_BLOCK`.
/// Returns `(true, 0)` if the window processed the message in time;
/// `(false, ERROR_TIMEOUT=1460)` if its message queue was stalled for `timeout_ms`.
pub fn send_null(hwnd: usize, timeout_ms: u32) -> (bool, u32) {
    let mut res: usize = 0;
    let ok = unsafe {
        SendMessageTimeoutW(
            hwnd as HWND,
            WM_NULL,
            0,
            0,
            SMTO_BLOCK,
            timeout_ms,
            &mut res,
        )
    };
    if ok != 0 {
        (true, 0)
    } else {
        (false, unsafe { GetLastError() })
    }
}

/// `true` if Windows considers `hwnd` hung (~5 s without pumping messages).
pub fn is_hung_window(hwnd: usize) -> bool {
    unsafe { IsHungAppWindow(hwnd as HWND) != 0 }
}

/// Return the Win32 window class name (`""` on failure).
pub fn window_class_name(hwnd: usize) -> String {
    let mut buf = [0u16; 64];
    let n = unsafe { GetClassNameW(hwnd as HWND, buf.as_mut_ptr(), buf.len() as i32) };
    if n > 0 {
        OsString::from_wide(&buf[..n as usize])
            .to_string_lossy()
            .into_owned()
    } else {
        String::new()
    }
}

/// If `hwnd` is a `(Not Responding)` DWM `"Ghost"` window, resolve the real underlying hung window.
/// Checks `GetClassNameW` first so responsive non-Ghost windows never trigger `GetProcAddress`.
pub fn real_window_from_ghost(hwnd: usize) -> Option<usize> {
    if window_class_name(hwnd) != "Ghost" {
        return None;
    }
    static FN_PTR: std::sync::OnceLock<Option<unsafe extern "system" fn(HWND) -> HWND>> =
        std::sync::OnceLock::new();
    let func_opt = FN_PTR.get_or_init(|| unsafe {
        let u32_mod: Vec<u16> = "user32.dll\0".encode_utf16().collect();
        let h = GetModuleHandleW(u32_mod.as_ptr());
        if h.is_null() {
            return None;
        }
        let addr = GetProcAddress(h, c"HungWindowFromGhostWindow".as_ptr().cast::<u8>());
        if addr.is_null() {
            None
        } else {
            Some(std::mem::transmute::<
                *const core::ffi::c_void,
                unsafe extern "system" fn(HWND) -> HWND,
            >(addr))
        }
    });
    let func = (*func_opt)?;
    let real = unsafe { func(hwnd as HWND) };
    if real.is_null() {
        None
    } else {
        Some(real as usize)
    }
}

/// Wait for the Desktop Window Manager's next composition pass (`DwmFlush`).
pub fn dwm_flush() -> i32 {
    unsafe { DwmFlush() }
}

/// Boost the calling thread to `THREAD_PRIORITY_HIGHEST` and disable EcoQoS power throttling.
pub fn boost_current_thread() -> bool {
    unsafe {
        let th = GetCurrentThread();
        let prio_ok = SetThreadPriority(th, THREAD_PRIORITY_HIGHEST) != 0;
        let state = THREAD_POWER_THROTTLING_STATE {
            Version: THREAD_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: THREAD_POWER_THROTTLING_EXECUTION_SPEED,
            StateMask: 0,
        };
        let _ = SetThreadInformation(
            th,
            THREAD_POWER_THROTTLING,
            &state as *const _ as *const _,
            size_of::<THREAD_POWER_THROTTLING_STATE>() as DWORD,
        );
        prio_ok
    }
}

/// Enable Windows Efficiency Mode (`EcoQoS` + below-normal priority) on background worker
/// threads so 1 Hz sampling and kNN training favor Efficiency (E) cores without competing
/// with foreground apps or HighQoS UI probe threads.
pub fn set_current_thread_eco_qos() -> bool {
    unsafe {
        let th = GetCurrentThread();
        let prio_ok = SetThreadPriority(th, THREAD_PRIORITY_BELOW_NORMAL) != 0;
        let state = THREAD_POWER_THROTTLING_STATE {
            Version: THREAD_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: THREAD_POWER_THROTTLING_EXECUTION_SPEED,
            StateMask: THREAD_POWER_THROTTLING_EXECUTION_SPEED,
        };
        let _ = SetThreadInformation(
            th,
            THREAD_POWER_THROTTLING,
            &state as *const _ as *const _,
            size_of::<THREAD_POWER_THROTTLING_STATE>() as DWORD,
        );
        prio_ok
    }
}

static CTRL_STOP_FLAG: std::sync::Mutex<Option<std::sync::Arc<std::sync::atomic::AtomicBool>>> =
    std::sync::Mutex::new(None);

unsafe extern "system" fn console_ctrl_handler(_ctrl_type: DWORD) -> BOOL {
    if let Ok(g) = CTRL_STOP_FLAG.try_lock() {
        if let Some(ref flag) = *g {
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
            return 1;
        }
    }
    0
}

/// Register a Windows console Ctrl+C / Close handler that sets `stop` for graceful shutdown.
pub fn install_ctrl_handler(stop: std::sync::Arc<std::sync::atomic::AtomicBool>) {
    if let Ok(mut g) = CTRL_STOP_FLAG.lock() {
        let first_install = g.is_none();
        *g = Some(stop);
        if first_install {
            unsafe {
                let _ = SetConsoleCtrlHandler(Some(console_ctrl_handler), 1);
            }
        }
    }
}

/// `(wall_uptime_s, unbiased_awake_time_s)`:
/// `GetTickCount64` advances across sleep/hibernate; `QueryUnbiasedInterruptTime` pauses during sleep.
pub fn sleep_clocks() -> (f64, f64) {
    unsafe {
        let up = (GetTickCount64() as f64) / 1000.0;
        let mut ticks: u64 = 0;
        let _ = QueryUnbiasedInterruptTime(&mut ticks);
        (up, (ticks as f64) / 1e7)
    }
}

/// Enumerate loaded non-inbox kernel drivers (`.sys`) for DPC/ISR storm attribution.
pub fn loaded_third_party_drivers() -> Vec<String> {
    unsafe {
        let mut bases = vec![null_mut::<core::ffi::c_void>(); 1024];
        let mut needed: DWORD = 0;
        if EnumDeviceDrivers(
            bases.as_mut_ptr(),
            (bases.len() * size_of::<*mut core::ffi::c_void>()) as DWORD,
            &mut needed,
        ) == 0
        {
            return Vec::new();
        }
        let count = (needed as usize / size_of::<*mut core::ffi::c_void>()).min(bases.len());
        let inbox: HashSet<&str> = [
            "ntoskrnl.exe",
            "hal.dll",
            "kd.dll",
            "mcupdate_genuineintel.dll",
            "mcupdate_authenticamd.dll",
            "clfs.sys",
            "tm.sys",
            "pshed.dll",
            "bootvid.dll",
            "fltmgr.sys",
            "msrpc.sys",
            "ksecdd.sys",
            "clipsp.sys",
            "cmimcext.sys",
            "werkernel.sys",
            "ntosext.sys",
            "ci.dll",
            "cng.sys",
            "acpiex.sys",
            "mssecflt.sys",
            "acpi.sys",
            "wmilib.sys",
            "WppRecorder.sys",
            "SgrmAgent.sys",
            "pci.sys",
            "tpm.sys",
            "intelpep.sys",
            "WindowsTrustedRT.sys",
            "IntelTA.sys",
            "WindowsTrustedRTProxy.sys",
            "pcw.sys",
            "msisadrv.sys",
            "vdrvroot.sys",
            "pdc.sys",
            "cea.sys",
            "partmgr.sys",
            "spaceport.sys",
            "volmgr.sys",
            "volmgrx.sys",
            "mountmgr.sys",
            "stornvme.sys",
            "ehstorclass.sys",
            "fileinfo.sys",
            "wof.sys",
            "ntfs.sys",
            "fs_rec.sys",
            "ksecpkg.sys",
            "fwpkclnt.sys",
            "wfplwfs.sys",
            "fvevol.sys",
            "volume.sys",
            "volsnap.sys",
            "rdyboost.sys",
            "mup.sys",
            "iorate.sys",
            "disk.sys",
            "clasPNP.sys",
        ]
        .into_iter()
        .collect();

        let mut notable = Vec::new();
        let mut buf = [0u16; 260];
        for &base in &bases[..count] {
            if base.is_null() {
                continue;
            }
            let len = GetDeviceDriverBaseNameW(base, buf.as_mut_ptr(), buf.len() as DWORD);
            if len > 0 {
                let name = OsString::from_wide(&buf[..len as usize])
                    .to_string_lossy()
                    .into_owned();
                if !inbox.contains(name.as_str())
                    && !inbox.contains(name.to_ascii_lowercase().as_str())
                {
                    notable.push(name);
                }
            }
        }
        notable
    }
}

// ── Dynamic NVML Loader (Zero Hard Link Dependency) ──────────────────────

#[repr(C)]
struct NvmlUtilization {
    gpu: u32,
    memory: u32,
}

#[repr(C)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct NvmlProcessInfo {
    pid: u32,
    usedGpuMemory: u64,
    gpuInstanceId: u32,
    computeInstanceId: u32,
}

type NvmlDevice = *mut core::ffi::c_void;

/// Dynamically loaded NVIDIA Management Library (`nvml.dll`).
pub struct NvmlClient {
    _lib: libloading::Library,
    device: NvmlDevice,
    max_power_w: f64,
    fn_shutdown: unsafe extern "C" fn() -> i32,
    fn_get_util: unsafe extern "C" fn(NvmlDevice, *mut NvmlUtilization) -> i32,
    fn_get_power: unsafe extern "C" fn(NvmlDevice, *mut u32) -> i32,
    fn_get_temp: unsafe extern "C" fn(NvmlDevice, u32, *mut u32) -> i32,
    fn_get_mem: unsafe extern "C" fn(NvmlDevice, *mut NvmlMemory) -> i32,
    fn_get_clock: unsafe extern "C" fn(NvmlDevice, u32, *mut u32) -> i32,
    fn_get_throttle: Option<unsafe extern "C" fn(NvmlDevice, *mut u64) -> i32>,
    fn_get_comp_procs:
        Option<unsafe extern "C" fn(NvmlDevice, *mut u32, *mut NvmlProcessInfo) -> i32>,
    fn_get_gfx_procs:
        Option<unsafe extern "C" fn(NvmlDevice, *mut u32, *mut NvmlProcessInfo) -> i32>,
}

unsafe impl Send for NvmlClient {}
unsafe impl Sync for NvmlClient {}

#[derive(Debug, Clone, Copy)]
pub struct NvmlTelemetry {
    pub gpu_awake: f64,
    pub gpu_util_pct: f64,
    pub gpu_power_w: f64,
    pub gpu_temp_c: f64,
    pub gpu_vram_pct: f64,
    pub gpu_clock_mhz: f64,
    pub gpu_throttle: u64,
}

impl NvmlTelemetry {
    pub fn absent() -> Self {
        Self {
            gpu_awake: f64::NAN,
            gpu_util_pct: f64::NAN,
            gpu_power_w: f64::NAN,
            gpu_temp_c: f64::NAN,
            gpu_vram_pct: f64::NAN,
            gpu_clock_mhz: f64::NAN,
            gpu_throttle: 0,
        }
    }

    pub fn asleep() -> Self {
        Self {
            gpu_awake: 0.0,
            gpu_util_pct: 0.0,
            gpu_power_w: 0.0,
            gpu_temp_c: f64::NAN,
            gpu_vram_pct: 0.0,
            gpu_clock_mhz: 0.0,
            gpu_throttle: 0,
        }
    }
}

impl NvmlClient {
    pub fn try_new() -> Option<Self> {
        unsafe {
            let lib = libloading::Library::new("nvml.dll")
                .or_else(|_| {
                    let sys32 =
                        std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string());
                    libloading::Library::new(format!("{}\\System32\\nvml.dll", sys32))
                })
                .ok()?;

            let fn_init: unsafe extern "C" fn() -> i32 = *lib
                .get(b"nvmlInit_v2\0")
                .or_else(|_| lib.get(b"nvmlInit\0"))
                .ok()?;
            if fn_init() != 0 {
                return None;
            }

            let fn_shutdown: unsafe extern "C" fn() -> i32 = *lib.get(b"nvmlShutdown\0").ok()?;
            let fn_get_handle: unsafe extern "C" fn(u32, *mut NvmlDevice) -> i32 = *lib
                .get(b"nvmlDeviceGetHandleByIndex_v2\0")
                .or_else(|_| lib.get(b"nvmlDeviceGetHandleByIndex\0"))
                .ok()?;

            let mut device: NvmlDevice = null_mut();
            if fn_get_handle(0, &mut device) != 0 || device.is_null() {
                let _ = fn_shutdown();
                return None;
            }

            let max_power_w = lib
                .get::<unsafe extern "C" fn(NvmlDevice, *mut u32) -> i32>(
                    b"nvmlDeviceGetEnforcedPowerLimit\0",
                )
                .ok()
                .and_then(|f| {
                    let mut mw: u32 = 0;
                    if f(device, &mut mw) == 0 && mw > 0 {
                        Some((mw as f64) / 1000.0 * 1.5)
                    } else {
                        None
                    }
                })
                .unwrap_or(375.0);

            let fn_get_util = *lib.get(b"nvmlDeviceGetUtilizationRates\0").ok()?;
            let fn_get_power = *lib.get(b"nvmlDeviceGetPowerUsage\0").ok()?;
            let fn_get_temp = *lib.get(b"nvmlDeviceGetTemperature\0").ok()?;
            let fn_get_mem = *lib.get(b"nvmlDeviceGetMemoryInfo\0").ok()?;
            let fn_get_clock = *lib.get(b"nvmlDeviceGetClockInfo\0").ok()?;
            let fn_get_throttle = lib
                .get(b"nvmlDeviceGetCurrentClocksEventReasons\0")
                .or_else(|_| lib.get(b"nvmlDeviceGetCurrentClocksThrottleReasons\0"))
                .ok()
                .map(|s| *s);

            let fn_get_comp_procs = lib
                .get(b"nvmlDeviceGetComputeRunningProcesses_v3\0")
                .or_else(|_| lib.get(b"nvmlDeviceGetComputeRunningProcesses\0"))
                .ok()
                .map(|s| *s);
            let fn_get_gfx_procs = lib
                .get(b"nvmlDeviceGetGraphicsRunningProcesses_v3\0")
                .or_else(|_| lib.get(b"nvmlDeviceGetGraphicsRunningProcesses\0"))
                .ok()
                .map(|s| *s);

            Some(Self {
                _lib: lib,
                device,
                max_power_w,
                fn_shutdown,
                fn_get_util,
                fn_get_power,
                fn_get_temp,
                fn_get_mem,
                fn_get_clock,
                fn_get_throttle,
                fn_get_comp_procs,
                fn_get_gfx_procs,
            })
        }
    }

    pub fn client_pids(&self) -> Vec<u32> {
        let mut pids = HashSet::new();
        for func in [self.fn_get_comp_procs, self.fn_get_gfx_procs]
            .into_iter()
            .flatten()
        {
            unsafe {
                let mut count: u32 = 64;
                let mut buf = vec![NvmlProcessInfo::default(); 64];
                let rc = func(self.device, &mut count, buf.as_mut_ptr());
                if rc == 0 && count > 0 {
                    for p in &buf[..(count as usize).min(buf.len())] {
                        if p.pid != 0 {
                            pids.insert(p.pid);
                        }
                    }
                }
            }
        }
        pids.into_iter().collect()
    }

    pub fn query_telemetry(&self) -> NvmlTelemetry {
        unsafe {
            let mut util = NvmlUtilization { gpu: 0, memory: 0 };
            if (self.fn_get_util)(self.device, &mut util) != 0 {
                return NvmlTelemetry::asleep();
            }
            let mut mw: u32 = 0;
            let raw_power_w = if (self.fn_get_power)(self.device, &mut mw) == 0 {
                (mw as f64) / 1000.0
            } else {
                f64::NAN
            };
            let power_w =
                if raw_power_w.is_finite() && (0.0..=self.max_power_w).contains(&raw_power_w) {
                    raw_power_w
                } else {
                    f64::NAN
                };
            let mut temp: u32 = 0;
            let raw_temp_c = if (self.fn_get_temp)(self.device, 0, &mut temp) == 0 {
                temp as f64
            } else {
                f64::NAN
            };
            let temp_c = if raw_temp_c.is_finite() && (1.0..=125.0).contains(&raw_temp_c) {
                raw_temp_c
            } else {
                f64::NAN
            };
            let mut mem = NvmlMemory {
                total: 0,
                free: 0,
                used: 0,
            };
            let vram_pct = if (self.fn_get_mem)(self.device, &mut mem) == 0 && mem.total > 0 {
                (mem.used as f64) / (mem.total as f64) * 100.0
            } else {
                f64::NAN
            };
            let mut clk: u32 = 0;
            let clock_mhz = if (self.fn_get_clock)(self.device, 0, &mut clk) == 0 {
                clk as f64
            } else {
                f64::NAN
            };
            let mut throttle: u64 = 0;
            if let Some(f_thr) = self.fn_get_throttle {
                let _ = f_thr(self.device, &mut throttle);
            }
            NvmlTelemetry {
                gpu_awake: 1.0,
                gpu_util_pct: (util.gpu.min(100)) as f64,
                gpu_power_w: power_w,
                gpu_temp_c: temp_c,
                gpu_vram_pct: vram_pct,
                gpu_clock_mhz: clock_mhz,
                gpu_throttle: throttle,
            }
        }
    }
}

impl Drop for NvmlClient {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.fn_shutdown)();
        }
    }
}
