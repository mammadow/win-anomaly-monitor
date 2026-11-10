# Contributing to Windows Resource & Anomaly Monitor (`resmon`)

Thank you for helping improve `resmon`! This project is written in pure Rust (`v4.0`, edition 2021) for Windows 11 (`resmon.exe` CLI/TUI and windowless `resmonw.exe` background tray agent), with the earlier Python v3 implementation preserved under `legacy/`.

## Prerequisites

- **OS:** Windows 11 (x64)
- **Rust toolchain:** stable `rustc` & `cargo` with `rustfmt` and `clippy` components (`rustup component add rustfmt clippy`)
- **Windows SDK (optional but recommended):** `rc.exe` (used automatically by `build.rs` to embed PE version info, DPI-aware manifest, and app icon) and `signtool.exe` (used by `scripts/sign.ps1`)
- **Python 3.11+ (only if touching `legacy/`):** see `legacy/requirements.txt`

## Development Workflow

### 1. Run formatting, lints, and tests

Every pull request must pass the same checks enforced by `.github/workflows/ci.yml`:

```powershell
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo build --release --locked
cargo test --all-targets --locked -- --nocapture
```

If you modify anything in `legacy/`, also run the legacy Python test suite:

```powershell
cd legacy
$env:PYTHONPATH = "."
python -m pytest tests -v -ra
```

### 2. Optional local code signing

For local Windows builds stored in startup shortcuts or evaluated by endpoint security, `scripts/sign.ps1` creates (if needed) a `CN=resmon Local Publisher` code-signing certificate in `Cert:\CurrentUser\My` and SHA-256 Authenticode-signs `target\release\resmon.exe` and `target\release\resmonw.exe`:

```powershell
.\scripts\sign.ps1
```

## Architecture & Coding Guidelines

- **Zero external CLI subprocesses on hot paths:** System telemetry (`src/sampler.rs`), process/service/socket attribution (`src/processes.rs`), HighQoS UI responsiveness probing (`src/responsiveness.rs`), and Win32 notifications/tray (`src/tray.rs`) call Win32/NT/NVML APIs directly through `src/winapi.rs` and `libloading`.
- **Optimus / D3cold safety:** GPU queries in `src/winapi.rs` check device power state and skip NVML calls while the dGPU is sleeping. Keep that invariant intact so `resmon` never triggers spurious GPU wakes.
- **Schema & config compatibility:** Keep live `data/metrics.db`, `data/events.jsonl`, `data/state.json`, and legacy v3 `config.toml` keys backward-compatible. Validate new tests cleanly against `tests/integration_tests.rs` using offline mocks and temporary directories.
- **Keep private data local:** Never commit `config.toml`, `data/`, `*.log`, or local workspace instruction files listed in `.gitignore`.
- **Tests:** Add focused coverage in `tests/integration_tests.rs` alongside bug fixes or new rules/diagnostic detectors.
