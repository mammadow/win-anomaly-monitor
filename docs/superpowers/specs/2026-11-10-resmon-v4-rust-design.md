# `resmon` v4.0 — Pure Rust Single-Binary Architecture & Deep Attribution Spec

**Date:** 2026-11-10  
**Status:** Approved (Approach 2 — Native Single-Binary Rust Engine + Deep Attribution + Tray + TUI)

---

## 1. Goals & Motivation

Rebuild `win-anomaly-monitor` (`resmon`) in **pure Rust (v4.0)** while preserving the Python v3 implementation in `legacy/` and maintaining **100% compatibility** with existing `data/metrics.db` (SQLite schema v3), `data/events.jsonl`, `data/state.json`, and `config.toml`.

### Key Improvements Over Python v3
1. **Single-Process, GIL-Free Execution (~3–5 MB RAM vs. ~120 MB)**:
   - Eliminates the separate `--probe-helper` Python interpreter and pipe IPC (`resmon/responsiveness.py`).
   - Runs the 3 UI responsiveness probe threads (`foreground`, `taskbar`, `compositor`) in-process on native OS threads pinned to `THREAD_PRIORITY_HIGHEST` with `THREAD_POWER_THROTTLING_CURRENT_VERSION` HighQoS.
2. **SIMD + Rayon Parallel kNN (`detector.rs`)**:
   - Replaces single-threaded NumPy kNN calibration over 30,000 windows with multi-threaded `rayon` chunked L2 / directional-excess distance computation, cutting training time from ~3–5 seconds to <150 ms.
3. **Deep Process, Script, and Service Attribution (`winapi.rs`, `processes.rs`)**:
   - **Command-Line Disambiguation**: Queries `NtQueryInformationProcess(..., ProcessCommandLineInformation = 60)` once per new `(pid, create_time)` for scripting/runtime hosts (`python.exe`, `pythonw.exe`, `node.exe`, `pwsh.exe`, `powershell.exe`, `java.exe`, `dotnet.exe`, `cmd.exe`, `ruby.exe`, `bun.exe`, `deno.exe`) to extract the script or module name (e.g., `python.exe [train.py]`, `node.exe [vite]`).
   - **Windows Service Attribution**: Queries `EnumServicesStatusExW` via SCM to map `svchost.exe` PIDs to their hosted service names (e.g., `svchost.exe [SysMain, wuauserv]`).
   - **Per-Process Hard Page Faults (`HardFaultCount`)**: Reads `SYSTEM_PROCESS_INFORMATION.HardFaultCount` at zero extra syscall cost and computes `hard_faults_ps` per process and system-wide, distinguishing true swap/pagefile thrashing from sequential file I/O during `memory_pressure` and `ui_freeze` events.
4. **Episode Recurrence Coalescing (`rules.rs`)**:
   - Adds a 10-minute recurrence merge window (`REPEAT_MERGE_SEC = 600`) to `MemoryPressure`, `GpuThermal`, and `GpuAwakeOnBattery` (matching `AnomalyEpisodes`), eliminating the alert flapping observed when RAM/commit hovers near 90–98%.
5. **Opportunistic Kernel Driver Attribution (`winapi.rs`)**:
   - Enumerates loaded kernel driver modules (`EnumDeviceDrivers` / `GetDeviceDriverBaseNameW`) and tracks kernel DPC/ISR rates so DPC storms include actionable driver context.
6. **Interactive Terminal UI (`resmon top`) & System Tray (`--tray`)**:
   - `resmon top`: Built-in live terminal dashboard (`ratatui` + `crossterm`) displaying real-time CPU/core/DPC/ISR gauges, RAM/commit/swap fault rates, GPU state, top processes with script/service tags, live UI responsiveness probe latencies, and recent events.
   - `resmon run --tray`: Lightweight Win32 `Shell_NotifyIconW` system tray icon with live health tooltip and context menu.

---

## 2. Repository Layout

```text
win-anomaly-monitor/
├── Cargo.toml                  # Pure Rust crate manifest (resmon v4.0.0)
├── src/
│   ├── main.rs                 # Entry point & CLI dispatcher
│   ├── lib.rs                  # Library root exporting all modules for tests
│   ├── config.rs               # MonitorConfig, TOML parser, validation
│   ├── winapi.rs               # Native NT/Win32 syscalls, dynamic NVML FFI, SCM & PEB cmdline
│   ├── sampler.rs              # SystemSampler (1 Hz) & GpuSampler (D3cold-safe NVML)
│   ├── processes.rs            # ProcessTracker (5s sweep, HardFaultCount, script/service tags)
│   ├── features.rs             # 16 ML_FEATURES, WindowAggregator, to_matrix, formatting
│   ├── detector.rs             # Adaptive LOF-style kNN AnomalyModel, Rayon parallel knn, Trainer
│   ├── storage.rs              # SQLite WAL Storage (schema v3 compatible)
│   ├── events.rs               # Event, EventManager, JSONL writer, WinRT Toast Notifier
│   ├── rules.rs                # All 9 detectors + recurrence coalescing + OLS trend
│   ├── responsiveness.rs       # In-process HighQoS UI & DWM responsiveness probe + StallTracker
│   ├── replay.rs               # Historical window replay & parameter tuning grid
│   ├── diagnostics.rs          # Forensic context builder, deterministic RCA, LLM client, health audit
│   ├── engine.rs               # InstanceLock, Monitor daemon loop, state.json writer
│   ├── tui.rs                  # `resmon top` interactive terminal dashboard
│   ├── tray.rs                 # Native Win32 system tray icon (`--tray`)
│   └── cli.rs                  # Clap CLI subcommands (run, status, report, explain, ask, diagnose, tune, top, reset)
├── tests/                      # Rust integration tests
├── legacy/                     # Preserved Python v3 codebase & pytest suite
│   ├── monitor.py
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── resmon/
│   └── tests/
├── data/                       # Live user database, events.jsonl, state.json (untouched)
└── config.example.toml
```

---

## 3. Data & Schema Compatibility

- **SQLite (`data/metrics.db`)**:
  - Connects with `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;` and `busy_timeout = 10000 ms`.
  - Preserves `USER_VERSION = 3` (`PRAGMA user_version`) so existing `samples`, `windows`, `events`, and `meta` tables are never dropped.
  - Adds optional `hard_faults_ps REAL` column to `samples` via `ALTER TABLE samples ADD COLUMN hard_faults_ps REAL` if not already present (idempotent, non-breaking).
- **Model Persistence (`data/model.bin`)**:
  - Rust saves its trained model to `data/model.bin` (compact binary format with magic header `RSMN`, format v2) so it never collides with or corrupts `data/model.npz` if the legacy Python version is run.
- **State & Events (`data/state.json`, `data/events.jsonl`)**:
  - Identical JSON structure, plus enriched process names (`python.exe [script.py]`, `svchost.exe [SysMain]`) and `hard_faults_ps` telemetry.
