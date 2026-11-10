//! Windows build script embedding PE `VS_VERSION_INFO`, an `asInvoker` + Win10/11
//! DPI-aware application manifest, and a crisp native `.ico` into `resmon.exe` and `resmonw.exe`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let Some(rc_exe) = find_windows_sdk_tool("rc.exe") else {
        println!("cargo:warning=Windows SDK rc.exe not found; skipping PE resource embedding");
        return;
    };

    let ico_path = out_dir.join("resmon.ico");
    let manifest_path = out_dir.join("resmon.manifest");
    let rc_path = out_dir.join("resmon.rc");
    let res_path = out_dir.join("resmon.res");

    fs::write(&ico_path, generate_resmon_ico()).expect("write resmon.ico");
    fs::write(&manifest_path, MANIFEST_XML).expect("write resmon.manifest");
    fs::write(&rc_path, RC_SCRIPT).expect("write resmon.rc");

    let status = Command::new(&rc_exe)
        .current_dir(&out_dir)
        .args(["/nologo", "/fo", "resmon.res", "resmon.rc"])
        .status();

    match status {
        Ok(s) if s.success() && res_path.exists() => {
            // Copy/link via verbatim or short DOS path if OUT_DIR contains spaces so
            // `cargo:rustc-link-arg-bins=` passes cleanly as a single linker argument.
            let link_path = to_space_safe_path(&res_path);
            println!("cargo:rustc-link-arg-bins={}", link_path.display());
        }
        Ok(s) => {
            println!(
                "cargo:warning=rc.exe exited with {}; skipping PE resources",
                s
            );
        }
        Err(e) => {
            println!(
                "cargo:warning=failed to run rc.exe ({}); skipping PE resources",
                e
            );
        }
    }
}

fn find_windows_sdk_tool(name: &str) -> Option<PathBuf> {
    if let Ok(path_var) = env::var("PATH") {
        for dir in env::split_paths(&path_var) {
            let cand = dir.join(name);
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    let arch = match env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("x86") => "x86",
        Ok("aarch64") => "arm64",
        _ => "x64",
    };
    let mut kits = Vec::new();
    if let Ok(sdk) = env::var("WindowsSdkDir") {
        kits.push(PathBuf::from(sdk).join("bin"));
    }
    kits.push(PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10\bin"));
    kits.push(PathBuf::from(r"C:\Program Files\Windows Kits\10\bin"));

    let mut versions = Vec::new();
    for root in kits {
        if let Ok(entries) = fs::read_dir(&root) {
            for ent in entries.flatten() {
                let cand = ent.path().join(arch).join(name);
                if cand.is_file() {
                    versions.push(cand);
                }
            }
        }
    }
    versions.sort();
    versions.pop()
}

fn to_space_safe_path(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    if !s.contains(' ') {
        return path.to_path_buf();
    }
    // Convert path via Windows GetShortPathNameW if available so MSVC link.exe
    // never splits space-containing `OUT_DIR` paths passed through Cargo metadata.
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::ffi::{OsStrExt, OsStringExt};
        #[link(name = "kernel32")]
        extern "system" {
            fn GetShortPathNameW(
                lpszLongPath: *const u16,
                lpszShortPath: *mut u16,
                cchBuffer: u32,
            ) -> u32;
        }
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        let mut buf = vec![0u16; 1024];
        let n = unsafe { GetShortPathNameW(wide.as_ptr(), buf.as_mut_ptr(), buf.len() as u32) };
        if n > 0 && (n as usize) < buf.len() {
            let short = std::ffi::OsString::from_wide(&buf[..n as usize]);
            let p = PathBuf::from(short);
            if !p.to_string_lossy().contains(' ') {
                return p;
            }
        }
    }
    path.to_path_buf()
}

/// Generate a crisp 32x32 32-bit ARGB ICO featuring a dark slate circle + teal ECG pulse line.
fn generate_resmon_ico() -> Vec<u8> {
    const W: usize = 32;
    const H: usize = 32;
    let mut bgra = vec![0u8; W * H * 4];

    // Background badge: rounded dark slate square (#0F172A) with cyan border (#06B6D4)
    let cx = 15.5f32;
    let cy = 15.5f32;
    for y in 0..H {
        for x in 0..W {
            let dx = (x as f32 - cx).abs();
            let dy = (y as f32 - cy).abs();
            let r = dx.max(dy);
            let idx = (y * W + x) * 4;
            if r <= 14.5 {
                let border = r >= 12.6;
                let (b, g, red, a) = if border {
                    (212u8, 182u8, 6u8, 255u8) // Cyan border
                } else {
                    (42u8, 23u8, 15u8, 255u8) // Slate body
                };
                bgra[idx] = b;
                bgra[idx + 1] = g;
                bgra[idx + 2] = red;
                bgra[idx + 3] = a;
            }
        }
    }

    // Telemetry ECG waveform across y in top-down coordinates
    let pts: [(f32, f32); 9] = [
        (4.0, 16.0),
        (9.5, 16.0),
        (12.0, 11.0),
        (14.5, 22.0),
        (17.5, 7.0),
        (20.0, 20.0),
        (22.0, 16.0),
        (27.5, 16.0),
        (27.5, 16.0),
    ];
    for seg in pts.windows(2) {
        let (x0, y0) = seg[0];
        let (x1, y1) = seg[1];
        let steps = 40;
        for step in 0..=steps {
            let t = (step as f32) / (steps as f32);
            let px = x0 + (x1 - x0) * t;
            let py = y0 + (y1 - y0) * t;
            for oy in -1i32..=1 {
                for ox in -1i32..=1 {
                    let ix = px.round() as i32 + ox;
                    let iy = py.round() as i32 + oy;
                    if (0..W as i32).contains(&ix) && (0..H as i32).contains(&iy) {
                        let dist = ox * ox + oy * oy;
                        if dist <= 1 {
                            let idx = ((iy as usize) * W + (ix as usize)) * 4;
                            // Bright cyan/emerald pulse (#34D399 / #22D3EE)
                            bgra[idx] = 238;
                            bgra[idx + 1] = 230;
                            bgra[idx + 2] = 56;
                            bgra[idx + 3] = 255;
                        }
                    }
                }
            }
        }
    }

    // ICO format stores BMP bottom-up (H rows reversed) + 1-bpp AND mask
    let dib_size = 40 + W * H * 4 + (W * H / 8);
    let mut out = Vec::with_capacity(6 + 16 + dib_size);
    // ICONDIR header
    out.extend_from_slice(&0u16.to_le_bytes()); // reserved
    out.extend_from_slice(&1u16.to_le_bytes()); // type = ICO
    out.extend_from_slice(&1u16.to_le_bytes()); // count = 1
                                                // ICONDIRENTRY
    out.push(W as u8);
    out.push(H as u8);
    out.push(0); // color count
    out.push(0); // reserved
    out.extend_from_slice(&1u16.to_le_bytes()); // planes
    out.extend_from_slice(&32u16.to_le_bytes()); // bpp
    out.extend_from_slice(&(dib_size as u32).to_le_bytes());
    out.extend_from_slice(&22u32.to_le_bytes()); // offset
                                                 // BITMAPINFOHEADER
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(W as i32).to_le_bytes());
    out.extend_from_slice(&((H * 2) as i32).to_le_bytes()); // height * 2 for XOR + AND masks
    out.extend_from_slice(&1u16.to_le_bytes()); // planes
    out.extend_from_slice(&32u16.to_le_bytes()); // bpp
    out.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB
    out.extend_from_slice(&((W * H * 4) as u32).to_le_bytes());
    out.extend_from_slice(&[0u8; 16]);
    // Bottom-up pixel rows
    for y in (0..H).rev() {
        let row = &bgra[y * W * 4..(y + 1) * W * 4];
        out.extend_from_slice(row);
    }
    // 1-bpp AND mask (all 0s; alpha channel governs transparency)
    out.extend(std::iter::repeat_n(0u8, W * H / 8));
    out
}

const MANIFEST_XML: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <assemblyIdentity version="4.0.0.0" processorArchitecture="*" name="win-anomaly-monitor.resmon" type="win32"/>
  <description>resmon v4.0 Windows Anomaly and Resource Monitor</description>
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="asInvoker" uiAccess="false"/>
      </requestedPrivileges>
    </security>
  </trustInfo>
  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">
    <application>
      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"/>
    </application>
  </compatibility>
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings>
      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2</dpiAwareness>
      <activeCodePage xmlns="http://schemas.microsoft.com/SMI/2019/WindowsSettings">UTF-8</activeCodePage>
    </windowsSettings>
  </application>
</assembly>
"#;

const RC_SCRIPT: &str = r#"1 ICON "resmon.ico"
1 24 "resmon.manifest"

1 VERSIONINFO
FILEVERSION     4,0,0,0
PRODUCTVERSION  4,0,0,0
FILEFLAGSMASK   0x3fL
FILEFLAGS       0x0L
FILEOS          0x40004L
FILETYPE        0x1L
FILESUBTYPE     0x0L
BEGIN
    BLOCK "StringFileInfo"
    BEGIN
        BLOCK "040904B0"
        BEGIN
            VALUE "CompanyName",      "win-anomaly-monitor\0"
            VALUE "FileDescription",  "Windows 11 Anomaly & Resource Monitor\0"
            VALUE "FileVersion",      "4.0.0.0\0"
            VALUE "InternalName",     "resmon\0"
            VALUE "LegalCopyright",   "MIT License\0"
            VALUE "OriginalFilename", "resmon.exe\0"
            VALUE "ProductName",      "resmon\0"
            VALUE "ProductVersion",   "4.0.0\0"
        END
    END
    BLOCK "VarFileInfo"
    BEGIN
        VALUE "Translation", 0x0409, 1200
    END
END
"#;
