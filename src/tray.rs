//! Lightweight native Win32 System Tray Icon (`Shell_NotifyIconW`) for `resmon run --tray`.

#![allow(non_snake_case, non_camel_case_types, clippy::upper_case_acronyms)]

use crate::sampler::mono_now;
use crate::winapi::{BOOL, DWORD, HANDLE, HWND};
use std::collections::VecDeque;
use std::mem::{size_of, zeroed};
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

const WM_DESTROY: u32 = 0x0002;
const WM_QUERYENDSESSION: u32 = 0x0011;
const WM_ENDSESSION: u32 = 0x0016;
const WM_COMMAND: u32 = 0x0111;
const WM_USER: u32 = 0x0400;
const WM_TRAYICON: u32 = WM_USER + 1;
const WM_RBUTTONUP: u32 = 0x0205;
const WM_LBUTTONDBLCLK: u32 = 0x0203;
const CREATE_NEW_CONSOLE: u32 = 0x0000_0010;

const NIM_ADD: DWORD = 0x00000000;
const NIM_MODIFY: DWORD = 0x00000001;
const NIM_DELETE: DWORD = 0x00000002;
const NIF_MESSAGE: DWORD = 0x00000001;
const NIF_ICON: DWORD = 0x00000002;
const NIF_TIP: DWORD = 0x00000004;
const NIF_INFO: DWORD = 0x00000010;
const NIIF_INFO: DWORD = 0x00000001;
const NIIF_WARNING: DWORD = 0x00000002;
const NIIF_ERROR: DWORD = 0x00000003;

const IDI_APPLICATION: *const u16 = 32512 as *const u16;
const IMAGE_ICON: u32 = 1;
const LR_DEFAULTCOLOR: u32 = 0x0000_0000;
const LR_SHARED: u32 = 0x0000_8000;
const SM_CXICON: i32 = 11;
const SM_CYICON: i32 = 12;
const SM_CXSMICON: i32 = 49;
const SM_CYSMICON: i32 = 50;
const MF_STRING: u32 = 0x00000000;
const MF_SEPARATOR: u32 = 0x00000800;
const TPM_BOTTOMALIGN: u32 = 0x0020;
const TPM_LEFTALIGN: u32 = 0x0000;
const TPM_RETURNCMD: u32 = 0x0100;
const TPM_NONOTIFY: u32 = 0x0080;

const CMD_OPEN_TOP: usize = 1001;
const CMD_EXPLAIN_LATEST: usize = 1002;
const CMD_PAUSE_1H: usize = 1003;
const CMD_EXIT: usize = 1004;

#[repr(C)]
struct POINT {
    x: i32,
    y: i32,
}

#[repr(C)]
struct MSG {
    hwnd: HWND,
    message: u32,
    wParam: usize,
    lParam: isize,
    time: DWORD,
    pt: POINT,
}

#[repr(C)]
struct WNDCLASSEXW {
    cbSize: u32,
    style: u32,
    lpfnWndProc: unsafe extern "system" fn(HWND, u32, usize, isize) -> isize,
    cbClsExtra: i32,
    cbWndExtra: i32,
    hInstance: HANDLE,
    hIcon: HANDLE,
    hCursor: HANDLE,
    hbrBackground: HANDLE,
    lpszMenuName: *const u16,
    lpszClassName: *const u16,
    hIconSm: HANDLE,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct NOTIFYICONDATAW {
    cbSize: DWORD,
    hWnd: HWND,
    uID: u32,
    uFlags: u32,
    uCallbackMessage: u32,
    hIcon: HANDLE,
    szTip: [u16; 128],
    dwState: DWORD,
    dwStateMask: DWORD,
    szInfo: [u16; 256],
    uTimeoutOrVersion: u32,
    szInfoTitle: [u16; 64],
    dwInfoFlags: DWORD,
    guidItem: [u8; 16],
    hBalloonIcon: HANDLE,
}

#[link(name = "user32")]
extern "system" {
    fn RegisterClassExW(lpwcx: *const WNDCLASSEXW) -> u16;
    fn CreateWindowExW(
        dwExStyle: DWORD,
        lpClassName: *const u16,
        lpWindowName: *const u16,
        dwStyle: DWORD,
        X: i32,
        Y: i32,
        nWidth: i32,
        nHeight: i32,
        hWndParent: HWND,
        hMenu: HANDLE,
        hInstance: HANDLE,
        lpParam: *mut core::ffi::c_void,
    ) -> HWND;
    fn DefWindowProcW(hWnd: HWND, Msg: u32, wParam: usize, lParam: isize) -> isize;
    fn PeekMessageW(
        lpMsg: *mut MSG,
        hWnd: HWND,
        wMsgFilterMin: u32,
        wMsgFilterMax: u32,
        wRemoveMsg: u32,
    ) -> BOOL;
    fn TranslateMessage(lpMsg: *const MSG) -> BOOL;
    fn DispatchMessageW(lpMsg: *const MSG) -> isize;
    fn DestroyWindow(hWnd: HWND) -> BOOL;
    fn LoadIconW(hInstance: HANDLE, lpIconName: *const u16) -> HANDLE;
    fn LoadImageW(
        hInst: HANDLE,
        name: *const u16,
        r#type: u32,
        cx: i32,
        cy: i32,
        fuLoad: u32,
    ) -> HANDLE;
    fn GetSystemMetrics(nIndex: i32) -> i32;
    fn CreatePopupMenu() -> HANDLE;
    fn AppendMenuW(hMenu: HANDLE, uFlags: u32, uIDNewItem: usize, lpNewItem: *const u16) -> BOOL;
    fn TrackPopupMenu(
        hMenu: HANDLE,
        uFlags: u32,
        x: i32,
        y: i32,
        nReserved: i32,
        hWnd: HWND,
        prcRect: *const core::ffi::c_void,
    ) -> BOOL;
    fn DestroyMenu(hMenu: HANDLE) -> BOOL;
    fn GetCursorPos(lpPoint: *mut POINT) -> BOOL;
    fn SetForegroundWindow(hWnd: HWND) -> BOOL;
}

#[link(name = "shell32")]
extern "system" {
    fn Shell_NotifyIconW(dwMessage: DWORD, lpData: *mut NOTIFYICONDATAW) -> BOOL;
}

#[link(name = "kernel32")]
extern "system" {
    fn GetModuleHandleW(lpModuleName: *const u16) -> HANDLE;
}

unsafe fn load_app_icon(small: bool) -> HANDLE {
    let hmod = GetModuleHandleW(null());
    if !hmod.is_null() {
        let (cx, cy) = if small {
            (
                GetSystemMetrics(SM_CXSMICON).max(16),
                GetSystemMetrics(SM_CYSMICON).max(16),
            )
        } else {
            (
                GetSystemMetrics(SM_CXICON).max(32),
                GetSystemMetrics(SM_CYICON).max(32),
            )
        };
        let sized = LoadImageW(
            hmod,
            std::ptr::without_provenance::<u16>(1),
            IMAGE_ICON,
            cx,
            cy,
            LR_DEFAULTCOLOR | LR_SHARED,
        );
        if !sized.is_null() {
            return sized;
        }
        let icon = LoadIconW(hmod, std::ptr::without_provenance::<u16>(1));
        if !icon.is_null() {
            return icon;
        }
    }
    LoadIconW(null_mut(), IDI_APPLICATION)
}

pub struct TrayController {
    stop: Arc<AtomicBool>,
    tooltip: Arc<Mutex<String>>,
    pause_until: Arc<Mutex<f64>>,
    handle: Option<JoinHandle<()>>,
}

impl TrayController {
    pub fn start(stop: Arc<AtomicBool>) -> Self {
        let tooltip = Arc::new(Mutex::new("resmon v4.0 — Initializing...".to_string()));
        let pause_until = Arc::new(Mutex::new(0.0f64));

        let stop_c = Arc::clone(&stop);
        let tip_c = Arc::clone(&tooltip);
        let pause_c = Arc::clone(&pause_until);

        let handle = thread::Builder::new()
            .name("tray-ui".to_string())
            .spawn(move || {
                tray_thread_main(stop_c, tip_c, pause_c);
            })
            .ok();

        Self {
            stop,
            tooltip,
            pause_until,
            handle,
        }
    }

    pub fn set_tooltip(&self, text: &str) {
        if let Ok(mut g) = self.tooltip.lock() {
            *g = text.to_string();
        }
    }

    pub fn paused_until(&self) -> f64 {
        self.pause_until.lock().map(|g| *g).unwrap_or(0.0)
    }

    pub fn close(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for TrayController {
    fn drop(&mut self) {
        self.close();
    }
}

struct TrayActionState {
    stop: Arc<AtomicBool>,
    pause_until: Arc<Mutex<f64>>,
}

static TRAY_STATE: Mutex<Option<TrayActionState>> = Mutex::new(None);

unsafe extern "system" fn tray_wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: usize,
    lparam: isize,
) -> isize {
    if msg == WM_TRAYICON {
        let event = (lparam as u32) & 0xFFFF;
        if event == WM_RBUTTONUP {
            show_context_menu(hwnd);
            return 0;
        } else if event == WM_LBUTTONDBLCLK {
            spawn_cli_window(&["top"]);
            return 0;
        }
    } else if msg == WM_COMMAND {
        handle_menu_command(wparam & 0xFFFF);
        return 0;
    } else if msg == WM_QUERYENDSESSION {
        return 1;
    } else if msg == WM_ENDSESSION {
        if wparam != 0 {
            if let Ok(guard) = TRAY_STATE.lock() {
                if let Some(ref st) = *guard {
                    st.stop.store(true, Ordering::SeqCst);
                }
            }
        }
        return 0;
    } else if msg == WM_DESTROY {
        return 0;
    }
    DefWindowProcW(hwnd, msg, wparam, lparam)
}

unsafe fn show_context_menu(hwnd: HWND) {
    let menu = CreatePopupMenu();
    if menu.is_null() {
        return;
    }
    let add_item = |id: usize, label: &str| {
        let wide: Vec<u16> = format!("{}\0", label).encode_utf16().collect();
        let _ = AppendMenuW(menu, MF_STRING, id, wide.as_ptr());
    };
    add_item(CMD_OPEN_TOP, "Live Dashboard (resmon top)");
    add_item(CMD_EXPLAIN_LATEST, "Explain Latest Event");
    add_item(CMD_PAUSE_1H, "Pause Toast Alerts (1 hour)");
    let _ = AppendMenuW(menu, MF_SEPARATOR, 0, null());
    add_item(CMD_EXIT, "Stop Monitor");

    let mut pt = POINT { x: 0, y: 0 };
    let _ = GetCursorPos(&mut pt);
    let _ = SetForegroundWindow(hwnd);
    let cmd = TrackPopupMenu(
        menu,
        TPM_BOTTOMALIGN | TPM_LEFTALIGN | TPM_RETURNCMD | TPM_NONOTIFY,
        pt.x,
        pt.y,
        0,
        hwnd,
        null(),
    );
    let _ = DestroyMenu(menu);
    if cmd > 0 {
        handle_menu_command(cmd as usize);
    }
}

fn handle_menu_command(cmd: usize) {
    match cmd {
        CMD_OPEN_TOP => spawn_cli_window(&["top"]),
        CMD_EXPLAIN_LATEST => spawn_cli_window(&["explain"]),
        CMD_PAUSE_1H => {
            if let Ok(guard) = TRAY_STATE.lock() {
                if let Some(ref st) = *guard {
                    if let Ok(mut p) = st.pause_until.lock() {
                        *p = mono_now() + 3600.0;
                    }
                }
            }
        }
        CMD_EXIT => {
            if let Ok(guard) = TRAY_STATE.lock() {
                if let Some(ref st) = *guard {
                    st.stop.store(true, Ordering::SeqCst);
                }
            }
        }
        _ => {}
    }
}

fn resolve_cli_exe() -> Option<std::path::PathBuf> {
    let cur = std::env::current_exe().ok()?;
    let cli = cur.with_file_name("resmon.exe");
    if cli.exists() {
        Some(cli)
    } else {
        Some(cur)
    }
}

fn spawn_cli_window(args: &[&str]) {
    use std::os::windows::process::CommandExt;
    let Some(exe) = resolve_cli_exe() else {
        return;
    };
    let is_interactive = args.first().copied() == Some("top");
    if is_interactive {
        let _ = std::process::Command::new(&exe)
            .args(args)
            .creation_flags(CREATE_NEW_CONSOLE)
            .spawn();
    } else {
        let quoted_args: Vec<String> = args
            .iter()
            .map(|a| {
                if a.contains(' ') {
                    format!("\"{}\"", a)
                } else {
                    (*a).to_string()
                }
            })
            .collect();
        let cmdline = format!("/d /k \"\"{}\" {}\"", exe.display(), quoted_args.join(" "));
        let _ = std::process::Command::new("cmd.exe")
            .raw_arg(&cmdline)
            .creation_flags(CREATE_NEW_CONSOLE)
            .spawn();
    }
}

struct TrayBalloon {
    title: String,
    body: String,
    severity: String,
}

static TRAY_ACTIVE: AtomicBool = AtomicBool::new(false);
static PENDING_BALLOONS: Mutex<VecDeque<TrayBalloon>> = Mutex::new(VecDeque::new());

/// Deliver a Windows balloon/toast notification via `Shell_NotifyIconW(NIF_INFO)` with zero PowerShell spawn.
pub fn show_balloon_notification(title: &str, body: &str, severity: &str) {
    if TRAY_ACTIVE.load(Ordering::Relaxed) {
        if let Ok(mut q) = PENDING_BALLOONS.lock() {
            if q.len() < 16 {
                q.push_back(TrayBalloon {
                    title: title.to_string(),
                    body: body.to_string(),
                    severity: severity.to_string(),
                });
                return;
            }
        }
    }
    // Headless daemon fallback: lightweight ephemeral Win32 notification icon (no powershell.exe)
    let t = title.to_string();
    let b = body.to_string();
    let s = severity.to_string();
    let _ = thread::Builder::new()
        .name("win32-notify".to_string())
        .spawn(move || unsafe {
            let cls_name: Vec<u16> = "ResmonEphemeralNotifyCls\0".encode_utf16().collect();
            let wc = WNDCLASSEXW {
                cbSize: size_of::<WNDCLASSEXW>() as u32,
                style: 0,
                lpfnWndProc: DefWindowProcW,
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: GetModuleHandleW(null()),
                hIcon: load_app_icon(false),
                hCursor: null_mut(),
                hbrBackground: null_mut(),
                lpszMenuName: null(),
                lpszClassName: cls_name.as_ptr(),
                hIconSm: load_app_icon(true),
            };
            let _ = RegisterClassExW(&wc);
            let hwnd = CreateWindowExW(
                0,
                cls_name.as_ptr(),
                cls_name.as_ptr(),
                0,
                0,
                0,
                0,
                0,
                null_mut(),
                null_mut(),
                null_mut(),
                null_mut(),
            );
            if hwnd.is_null() {
                return;
            }
            let mut nid: NOTIFYICONDATAW = zeroed();
            nid.cbSize = size_of::<NOTIFYICONDATAW>() as DWORD;
            nid.hWnd = hwnd;
            nid.uID = 2;
            nid.uFlags = NIF_ICON | NIF_TIP | NIF_INFO;
            nid.hIcon = load_app_icon(true);
            fill_wide(&mut nid.szTip, "resmon v4.0");
            apply_balloon_fields(&mut nid, &t, &b, &s);
            let _ = Shell_NotifyIconW(NIM_ADD, &mut nid);
            thread::sleep(std::time::Duration::from_secs(6));
            let _ = Shell_NotifyIconW(NIM_DELETE, &mut nid);
            let _ = DestroyWindow(hwnd);
        });
}

fn fill_wide<const N: usize>(buf: &mut [u16; N], text: &str) {
    buf.fill(0);
    if N <= 1 {
        return;
    }
    let mut len = 0usize;
    for ch in text.chars() {
        let needed = ch.len_utf16();
        if len + needed >= N {
            break;
        }
        let mut tmp = [0u16; 2];
        let encoded = ch.encode_utf16(&mut tmp);
        for &u in &*encoded {
            buf[len] = u;
            len += 1;
        }
    }
}

fn apply_balloon_fields(nid: &mut NOTIFYICONDATAW, title: &str, body: &str, severity: &str) {
    fill_wide(&mut nid.szInfoTitle, title);
    fill_wide(&mut nid.szInfo, body);
    nid.dwInfoFlags = match severity {
        "critical" => NIIF_ERROR,
        "warning" => NIIF_WARNING,
        _ => NIIF_INFO,
    };
}

fn tray_thread_main(
    stop: Arc<AtomicBool>,
    tooltip: Arc<Mutex<String>>,
    pause_until: Arc<Mutex<f64>>,
) {
    if let Ok(mut guard) = TRAY_STATE.lock() {
        *guard = Some(TrayActionState {
            stop: Arc::clone(&stop),
            pause_until,
        });
    }

    unsafe {
        let cls_name: Vec<u16> = "ResmonTrayWndClass\0".encode_utf16().collect();
        let wc = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            style: 0,
            lpfnWndProc: tray_wnd_proc,
            cbClsExtra: 0,
            cbWndExtra: 0,
            hInstance: GetModuleHandleW(null()),
            hIcon: load_app_icon(false),
            hCursor: null_mut(),
            hbrBackground: null_mut(),
            lpszMenuName: null(),
            lpszClassName: cls_name.as_ptr(),
            hIconSm: load_app_icon(true),
        };
        let _ = RegisterClassExW(&wc);

        let hwnd = CreateWindowExW(
            0,
            cls_name.as_ptr(),
            cls_name.as_ptr(),
            0,
            0,
            0,
            0,
            0,
            null_mut(),
            null_mut(),
            null_mut(),
            null_mut(),
        );
        if hwnd.is_null() {
            return;
        }

        let mut nid: NOTIFYICONDATAW = zeroed();
        nid.cbSize = size_of::<NOTIFYICONDATAW>() as DWORD;
        nid.hWnd = hwnd;
        nid.uID = 1;
        nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP;
        nid.uCallbackMessage = WM_TRAYICON;
        nid.hIcon = load_app_icon(true);
        fill_wide(&mut nid.szTip, "resmon v4.0 — Running");
        let _ = Shell_NotifyIconW(NIM_ADD, &mut nid);
        TRAY_ACTIVE.store(true, Ordering::SeqCst);

        let mut last_tip = String::new();
        let mut msg: MSG = zeroed();

        while !stop.load(Ordering::Relaxed) {
            while PeekMessageW(&mut msg, null_mut(), 0, 0, 1) != 0 {
                let _ = TranslateMessage(&msg);
                let _ = DispatchMessageW(&msg);
            }
            if let Ok(mut q) = PENDING_BALLOONS.lock() {
                if let Some(balloon) = q.pop_front() {
                    nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP | NIF_INFO;
                    apply_balloon_fields(
                        &mut nid,
                        &balloon.title,
                        &balloon.body,
                        &balloon.severity,
                    );
                    let _ = Shell_NotifyIconW(NIM_MODIFY, &mut nid);
                    nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP;
                }
            }
            if let Ok(g) = tooltip.lock() {
                if *g != last_tip {
                    last_tip = g.clone();
                    fill_wide(&mut nid.szTip, &last_tip);
                    let _ = Shell_NotifyIconW(NIM_MODIFY, &mut nid);
                }
            }
            thread::sleep(std::time::Duration::from_millis(100));
        }

        TRAY_ACTIVE.store(false, Ordering::SeqCst);
        let _ = Shell_NotifyIconW(NIM_DELETE, &mut nid);
        let _ = DestroyWindow(hwnd);
    }
}
