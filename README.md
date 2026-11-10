<div align="center">

# ⚡ `resmon` v4.0 — Pure Rust Windows Anomaly & Forensic Monitor

**Adaptive LOF-kNN Anomaly Detection · Deep Script/Service Attribution · HighQoS UI Responsiveness Probe · Interactive TUI & AI Diagnostics**

[![CI](https://github.com/mammadow/win-anomaly-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/mammadow/win-anomaly-monitor/actions/workflows/ci.yml)
[![Rust 2021](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%2011-0078D6.svg)](https://www.microsoft.com/windows)
[![Overhead](https://img.shields.io/badge/Footprint-~4%20MB%20RAM%20%7C%20%3C0.1%25%20CPU-brightgreen.svg)]()
[![Storage](https://img.shields.io/badge/storage-SQLite%20WAL%20(bundled)-informational.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*An ultra-low-overhead, single-binary Windows 11 systems performance & anomaly monitor built in pure Rust (`v4.0`). It learns your personal workload baseline across 16 telemetry dimensions, disambiguates opaque runtime/service processes (`python.exe [train.py]`, `svchost.exe [SysMain]`), tracks hard page faults (`HardFaultCount`), probes UI & DWM responsiveness from HighQoS native threads, and provides instant forensic Root-Cause Analysis (RCA).*

---

</div>

## 🌟 What's New in v4.0 (Pure Rust Engine)

- **🦀 Single-Binary, GIL-Free Engine (~50 MB total w/ model vs. ~120 MB+ multi-proc Python)**:
  - Compiles both CLI [`resmon.exe`](Cargo.toml) and windowless tray daemon [`resmonw.exe`](src/bin/resmonw.rs) (`#![windows_subsystem = "windows"]`).
  - Pins the 1 Hz sampler and background kNN trainer threads to **Windows Efficiency Mode (`EcoQoS`)** while running the 3 in-process UI responsiveness probes (`foreground`, `taskbar`, `compositor`) at **`THREAD_PRIORITY_HIGHEST` (HighQoS)**.
- **🔬 Deep Process, Script & Service Attribution**:
  - **Command-Line Disambiguation**: Queries `NtQueryInformationProcess(ProcessCommandLineInformation = 60)` once per new `(pid, create_time)` for scripting/runtime hosts (`python.exe`, `node.exe`, `pwsh.exe`, `java.exe`, `dotnet.exe`, `bun.exe`, `deno.exe`) to label exact scripts/modules (e.g., `python.exe [uvicorn]`, `node.exe [src/server.js]`).
  - **Windows Service Mapping**: Queries SCM (`EnumServicesStatusExW`) to map `svchost.exe` PIDs to their active Windows service names (e.g., `svchost.exe [SysMain, wuauserv]`).
  - **Hard Page Fault Tracking (`HardFaultCount`)**: Captures per-process and system-wide hard page faults/sec (`hard_faults_ps`) at zero extra syscall cost to separate true swap/pagefile thrashing from normal file I/O.
  - **Kernel Driver Enumeration**: Lists loaded third-party `.sys` kernel drivers during `dpc_storm` incidents.
- **🚀 AVX2 SIMD + Rayon Multi-Core kNN Calibration**:
  - Uses runtime-detected 256-bit **AVX2 SIMD** across all 16 scaled features, Rayon multi-core parallelization, binary-search timestamp windowing, and lockstep `tune` grid evaluation (~73× faster full-history grid replay).
- **🔕 Smart Episode Recurrence Coalescing & Native Win32 Notifications**:
  - Merges recurring `memory_pressure`, `gpu_thermal`, `gpu_battery_drain`, and `anomaly` episodes within a 10-minute window (`REPEAT_MERGE_SEC = 600`), eliminating alert flapping when RAM/commit hovers near threshold boundaries.
  - Delivers alert balloons directly via Win32 `Shell_NotifyIconW(NIF_INFO)` with zero `powershell.exe` child process spawning.
- **📊 Interactive Terminal Dashboard (`resmon top`) & System Tray (`--tray`)**:
  - `resmon top`: Interactive `ratatui` dashboard with live CPU/Core/RAM/Commit gauges, 60-second CPU & Commit sparklines, daemon HighQoS UI ping & ML readiness badges, toggleable CPU/Memory vs Disk/Net I/O + Hard Page Faults views (`Tab`/`i`), and an in-TUI Offline RCA modal (`Up`/`Down` + `Enter`/`e`).
  - `resmonw.exe` / `resmon.exe run --tray`: Native Win32 `Shell_NotifyIconW` system tray icon with live health tooltip and quick-action context menu.
- **🔄 100% Database & Config Compatibility**:
  - Reads and writes existing `data/metrics.db` (SQLite WAL schema v3), `data/events.jsonl`, `data/state.json`, and `config.toml` without losing a single second of history.
  - The complete Python v3 implementation is preserved in [`legacy/`](legacy/).

---

## 🚀 Quick Start

### Build & Run (Rust v4.0)

```powershell
# Build optimized release binaries (`target\release\resmon.exe` & `resmonw.exe`)
cargo build --release

# Start windowless background monitor with system tray icon (suitable for shell:startup)
.\target\release\resmonw.exe run --tray

# Check live status & database summary
.\target\release\resmon.exe status

# Launch interactive live terminal dashboard (`Tab`/`i` toggles I/O, `Enter`/`e` explains selected event)
.\target\release\resmon.exe top
```

### CLI Reference

| Command | Description |
|:---|:---|
| `resmonw` / `resmon run [--tray]` | Run the 1 Hz background monitoring daemon (`resmonw.exe` runs windowless with tray by default) |
| `resmon top` | Interactive terminal UI (`Tab`/`i`: CPU↔I/O, `Up`/`Down`+`Enter`: Offline RCA modal, `r`: refresh, `q`: quit) |
| `resmon status` | Display live telemetry summary, top CPU/RAM processes, UI ping latency, and model readiness |
| `resmon report [-H 24] [-k KIND] [-v] [--json]` | List recorded incidents over the lookback window |
| `resmon explain [ID\|latest] [-q QUESTION] [--offline]` | Run forensic root-cause analysis (RCA) on a specific or latest event |
| `resmon ask "<question>" [-H 24] [--offline]` | Ask natural-language diagnostic questions over recent telemetry |
| `resmon diagnose [-H 24] [--json]` | Run a holistic system health audit and prioritized action plan |
| `resmon tune [-d DAYS]` | Replay recorded history across `(alert_quantile, min_consecutive_windows)` grid |
| `resmon reset [-y]` | Wipe recorded database and learned model to start fresh |

---

## 🏗️ Architecture & Pipeline

```text
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │                     SINGLE-BINARY RUST DATA COLLECTION                        │
 ├───────────────────────────┬────────────────────────────┬──────────────────────┤
 │ 1 Hz System Sampler       │ HighQoS In-Process Probe   │ 5 s Process Sweep    │
 │ • Per-core CPU, DPC, ISR  │ • 3 native threads @       │ • Single-call        │
 │ • RAM, Commit, HardFaults │   THREAD_PRIORITY_HIGHEST  │   NtQuerySystemInfo  │
 │ • PDH Physical Disk B/s   │ • SendMessageTimeoutW      │ • PEB Command-line   │
 │ • GetIfTable2 Network B/s │   (Foreground & Taskbar)   │   script extraction  │
 │ • Dynamic NVML GPU FFI    │ • DwmFlush Compositor      │ • SCM svchost.exe    │
 │   (Optimus D3cold safe)   │ • Sleep vs Scheduler clock │   service mapping    │
 └─────────────┬─────────────┴──────────────┬─────────────┴──────────┬───────────┘
               │                            │                        │
               ▼                            ▼                        ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │                       ANALYTICS & STORAGE (SQLite WAL)                        │
 ├──────────────────────────────────────────┬────────────────────────────────────┤
 │ 10-Second Windows & Rayon kNN Detector   │ 9 Rule Detectors + Coalescing      │
 │ • 16-D meaningful-unit scaling           │ • UI Freezes & Kernel DPC Storms   │
 │ • Directional excess LOF scoring         │ • Memory & Handle Leaks (OLS R²≥.8)│
 │ • Background thread retraining           │ • Memory Pressure + Swap Thrashing │
 │ • Atomic `data/model.bin` persistence    │ • Thermal, Battery Drain, Uploads  │
 └──────────────────────────────────────────┴────────────────────────────────────┘
                                            │
                                            ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │                   INTERFACES & FORENSIC DIAGNOSTICS                           │
 ├───────────────────────────────────────────────────────────────────────────────┤
 │ • Deterministic Offline RCA Engine (`explain`, `ask`, `diagnose`)             │
 │ • Optional Local Ollama (`/api/chat`) & OpenAI (`/v1/chat/completions`) LLM   │
 │ • Interactive Terminal UI (`resmon top`) & Win32 System Tray (`--tray`)       │
 └───────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Layout

```text
├── Cargo.toml                 # Pure Rust v4.0 package manifest
├── config.example.toml        # Documented configuration template
├── src/
│   ├── main.rs                # Console CLI entry point (`resmon.exe`)
│   ├── bin/resmonw.rs         # Windowless Win32 GUI/tray entry point (`resmonw.exe`)
│   ├── lib.rs                 # Library crate root
│   ├── winapi.rs              # Native NT/Win32 syscalls, dynamic NVML FFI, PEB & SCM queries
│   ├── sampler.rs             # 1 Hz system, DPC/ISR, hard fault, and Optimus GPU sampler
│   ├── processes.rs           # 5 s process sweep with script/service attribution
│   ├── responsiveness.rs      # HighQoS in-process UI & DWM responsiveness probe + StallTracker
│   ├── features.rs            # 16 ML features, 10 s WindowAggregator, unit scaling
│   ├── detector.rs            # Rayon parallel LOF-style kNN anomaly model & background Trainer
│   ├── rules.rs               # All 9 detectors, recurrence coalescing, OLS leak regression
│   ├── storage.rs             # Bundled SQLite WAL storage (schema v3 compatible)
│   ├── events.rs              # Event lifecycle, JSONL writer, WinRT Toast notifier
│   ├── diagnostics.rs         # Forensic RCA engine, NL Q&A, health audit, LLM client
│   ├── replay.rs              # Historical replay & sensitivity grid tuning
│   ├── tui.rs                 # Interactive `resmon top` dashboard (ratatui + crossterm)
│   ├── tray.rs                # Native Win32 system tray icon (`Shell_NotifyIconW`)
│   ├── engine.rs              # Single-instance lock, 1 Hz main loop, state.json writer
│   ├── cli.rs                 # Clap CLI command implementations
│   └── config.rs              # TOML configuration loader (v3 & v4 compatible)
├── tests/
│   └── integration_tests.rs   # Comprehensive Rust integration test suite
└── legacy/                    # Preserved Python v3 implementation & pytest suite
    ├── monitor.py
    ├── resmon/
    └── tests/
```

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE).
