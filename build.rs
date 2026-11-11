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

/// Generate a crisp multi-resolution (`16, 20, 24, 32, 48, 64, 128`) 32-bit ARGB ICO
/// featuring a 4x4 supersampled midnight-navy squircle badge, bright cyan border,
/// and glowing telemetry ECG waveform.
fn generate_resmon_ico() -> Vec<u8> {
    const SIZES: [usize; 7] = [16, 20, 24, 32, 48, 64, 128];

    struct EncodedFrame {
        size: usize,
        dib: Vec<u8>,
    }

    let frames: Vec<EncodedFrame> = SIZES
        .iter()
        .map(|&size| EncodedFrame {
            size,
            dib: encode_ico_dib(size, &render_icon_rgba(size)),
        })
        .collect();

    let count = frames.len();
    let header_bytes = 6 + count * 16;
    let total_bytes: usize = header_bytes + frames.iter().map(|f| f.dib.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total_bytes);

    // ICONDIR header
    out.extend_from_slice(&0u16.to_le_bytes()); // reserved
    out.extend_from_slice(&1u16.to_le_bytes()); // type = ICO
    out.extend_from_slice(&(count as u16).to_le_bytes());

    let mut offset = header_bytes as u32;
    for f in &frames {
        let dim = if f.size >= 256 { 0u8 } else { f.size as u8 };
        out.push(dim); // bWidth
        out.push(dim); // bHeight
        out.push(0); // bColorCount
        out.push(0); // bReserved
        out.extend_from_slice(&1u16.to_le_bytes()); // wPlanes
        out.extend_from_slice(&32u16.to_le_bytes()); // wBitCount
        out.extend_from_slice(&(f.dib.len() as u32).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += f.dib.len() as u32;
    }

    for f in &frames {
        out.extend_from_slice(&f.dib);
    }
    out
}

fn encode_ico_dib(size: usize, bgra: &[u8]) -> Vec<u8> {
    let xor_bytes = size * size * 4;
    // ICO 1-bpp AND mask scanlines MUST be 32-bit (4-byte) DWORD aligned.
    let and_stride = (size.div_ceil(32)) * 4;
    let and_bytes = and_stride * size;
    let mut dib = Vec::with_capacity(40 + xor_bytes + and_bytes);

    // BITMAPINFOHEADER
    dib.extend_from_slice(&40u32.to_le_bytes());
    dib.extend_from_slice(&(size as i32).to_le_bytes());
    dib.extend_from_slice(&((size * 2) as i32).to_le_bytes()); // height * 2 for XOR + AND masks
    dib.extend_from_slice(&1u16.to_le_bytes()); // planes
    dib.extend_from_slice(&32u16.to_le_bytes()); // bpp
    dib.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB
    dib.extend_from_slice(&(xor_bytes as u32).to_le_bytes());
    dib.extend_from_slice(&[0u8; 16]);

    // Bottom-up pixel rows
    for y in (0..size).rev() {
        dib.extend_from_slice(&bgra[y * size * 4..(y + 1) * size * 4]);
    }

    // Bottom-up 1-bpp AND mask: 1 = fully transparent pixel, 0 = opaque / alpha-blended
    let mut and_row = vec![0u8; and_stride];
    for y in (0..size).rev() {
        and_row.fill(0);
        for x in 0..size {
            let alpha = bgra[(y * size + x) * 4 + 3];
            if alpha == 0 {
                and_row[x / 8] |= 0x80 >> (x % 8);
            }
        }
        dib.extend_from_slice(&and_row);
    }
    dib
}

fn rounded_box_sdf(px: f32, py: f32, half_extent: f32, radius: f32) -> f32 {
    let qx = px.abs() - (half_extent - radius);
    let qy = py.abs() - (half_extent - radius);
    let out_dist = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
    let in_dist = qx.max(qy).min(0.0);
    out_dist + in_dist - radius
}

fn seg_dist(px: f32, py: f32, x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
    let vx = x1 - x0;
    let vy = y1 - y0;
    let len_sq = vx * vx + vy * vy;
    if len_sq <= 1e-8 {
        return ((px - x0).powi(2) + (py - y0).powi(2)).sqrt();
    }
    let t = (((px - x0) * vx + (py - y0) * vy) / len_sq).clamp(0.0, 1.0);
    let proj_x = x0 + vx * t;
    let proj_y = y0 + vy * t;
    ((px - proj_x).powi(2) + (py - proj_y).powi(2)).sqrt()
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn render_icon_rgba(size: usize) -> Vec<u8> {
    // 4x4 subpixel MSAA grid keeps edges & thin 16x16 telemetry peaks sharp
    const SUB: usize = 4;
    let mut bgra = vec![0u8; size * size * 4];
    let inv_size = 2.0 / (size as f32);
    let inv_sub = 1.0 / (SUB as f32);

    // Keep stroke & rim clearly visible even at 16x16 taskbar tray scale
    let px_unit = 2.0 / (size as f32);
    let border_w = (px_unit * 1.15).max(0.115);
    let core_r = (px_unit * 0.92).max(0.082);
    let glow_r = core_r * 2.15;

    let ecg_pts: [(f32, f32); 8] = [
        (-0.64, 0.04),
        (-0.34, 0.04),
        (-0.19, -0.22),
        (-0.04, 0.36),
        (0.11, -0.50),
        (0.26, 0.28),
        (0.39, 0.04),
        (0.64, 0.04),
    ];

    for y in 0..size {
        for x in 0..size {
            let mut acc_r = 0.0f32;
            let mut acc_g = 0.0f32;
            let mut acc_b = 0.0f32;
            let mut acc_a = 0.0f32;

            for sy in 0..SUB {
                for sx in 0..SUB {
                    let fx = (x as f32) + ((sx as f32) + 0.5) * inv_sub;
                    let fy = (y as f32) + ((sy as f32) + 0.5) * inv_sub;
                    let nx = fx * inv_size - 1.0;
                    let ny = fy * inv_size - 1.0;

                    let sd = rounded_box_sdf(nx, ny, 0.90, 0.34);
                    if sd > px_unit * 0.65 {
                        continue;
                    }
                    let shape_cov = 1.0 - smoothstep(-px_unit * 0.5, px_unit * 0.5, sd);
                    if shape_cov <= 0.0 {
                        continue;
                    }

                    // Midnight-navy top-to-bottom interior gradient
                    let grad_t = ((ny + 0.90) / 1.80).clamp(0.0, 1.0);
                    let mut r = 20.0 * (1.0 - grad_t) + 10.0 * grad_t;
                    let mut g = 33.0 * (1.0 - grad_t) + 18.0 * grad_t;
                    let mut b = 61.0 * (1.0 - grad_t) + 36.0 * grad_t;

                    // Crisp electric-cyan outer rim (#06B6D4 -> #38BDF8)
                    let rim_t =
                        smoothstep(-border_w - px_unit * 0.45, -border_w + px_unit * 0.45, sd);
                    if rim_t > 0.0 {
                        let rim_r = 34.0 * (1.0 - grad_t) + 6.0 * grad_t;
                        let rim_g = 211.0 * (1.0 - grad_t) + 182.0 * grad_t;
                        let rim_b = 238.0 * (1.0 - grad_t) + 212.0 * grad_t;
                        r = r * (1.0 - rim_t) + rim_r * rim_t;
                        g = g * (1.0 - rim_t) + rim_g * rim_t;
                        b = b * (1.0 - rim_t) + rim_b * rim_t;
                    }

                    // ECG waveform glow + luminous mint-white core
                    let mut min_d = f32::MAX;
                    for seg in ecg_pts.windows(2) {
                        min_d = min_d.min(seg_dist(nx, ny, seg[0].0, seg[0].1, seg[1].0, seg[1].1));
                    }
                    let glow_t = (1.0 - smoothstep(core_r * 0.7, glow_r, min_d)) * 0.45;
                    if glow_t > 0.0 {
                        r = r * (1.0 - glow_t) + 16.0 * glow_t;
                        g = g * (1.0 - glow_t) + 220.0 * glow_t;
                        b = b * (1.0 - glow_t) + 235.0 * glow_t;
                    }
                    let wave_t =
                        1.0 - smoothstep(core_r - px_unit * 0.35, core_r + px_unit * 0.35, min_d);
                    if wave_t > 0.0 {
                        // Subtle left-to-right cyan -> bright emerald-mint highlight
                        let wt = ((nx + 0.64) / 1.28).clamp(0.0, 1.0);
                        let wr = 110.0 * (1.0 - wt) + 150.0 * wt;
                        let wg = 242.0 * (1.0 - wt) + 255.0 * wt;
                        let wb = 255.0 * (1.0 - wt) + 225.0 * wt;
                        r = r * (1.0 - wave_t) + wr * wave_t;
                        g = g * (1.0 - wave_t) + wg * wave_t;
                        b = b * (1.0 - wave_t) + wb * wave_t;
                    }

                    acc_r += r * shape_cov;
                    acc_g += g * shape_cov;
                    acc_b += b * shape_cov;
                    acc_a += shape_cov;
                }
            }

            let samples = (SUB * SUB) as f32;
            let a = (acc_a / samples).clamp(0.0, 1.0);
            let idx = (y * size + x) * 4;
            if a > 0.004 && acc_a > 0.0 {
                // Convert back to straight (un-premultiplied) BGRA colors cleanly
                bgra[idx] = (acc_b / acc_a).round().clamp(0.0, 255.0) as u8;
                bgra[idx + 1] = (acc_g / acc_a).round().clamp(0.0, 255.0) as u8;
                bgra[idx + 2] = (acc_r / acc_a).round().clamp(0.0, 255.0) as u8;
                bgra[idx + 3] = (a * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    bgra
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
