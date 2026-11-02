<div align="center">

# ⚡ ML System Resource Monitor (v3.1)

**Adaptive Anomaly Detection · Windows Kernel Forensics · AI Root-Cause Diagnostics**

[![CI](https://github.com/mammadow/win-anomaly-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/mammadow/win-anomaly-monitor/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%2011-0078D6.svg)](https://www.microsoft.com/windows)
[![Overhead](https://img.shields.io/badge/CPU%20Overhead-%3C1%25%20single%20core-brightgreen.svg)]()
[![Storage](https://img.shields.io/badge/storage-SQLite%20WAL-informational.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*An intelligent, low-overhead background telemetry monitor tailored for modern high-performance Windows laptops (Intel Core i9-13900HX / RTX 4070 Laptop GPU). It learns your daily habits, detects subtle performance regressions, and diagnoses UI freezes with second-by-second forensic precision.*

---

</div>

## 🌟 Highlights

- **🧠 Self-Calibrating ML Anomaly Detection**: Learns what *normal* looks like for you across 16 telemetry dimensions (idle, gaming, on battery, away from keyboard). No arbitrary static thresholds.
- **🔍 Deep Forensic RCA (`explain`)**: Correlates incidents with $\pm10\text{s}$ raw 1 Hz telemetry, per-core DPC/ISR storms, memory commit limits, and PCIe power transitions.
- **💬 Conversational Diagnostics (`ask`)**: Ask natural language questions like *"Why did WhatsApp freeze?"* or *"Why did my laptop stutter?"* powered by local Ollama (`llama3.2`, `qwen2.5`) or an offline expert rule engine.
- **⚡ Hardware-Aware (Optimus D3cold)**: Differentiates between actual GPU work and discrete GPU sleep transitions, preventing false thermal alerts and identifying PCIe link wake-up stalls.
- **⏱️ Independent UI Freeze Probes**: Measures message pump latency (`SendMessageTimeout`) and desktop compositor stalls (`DwmFlush`) from a dedicated helper process outside the Python GIL.
- **🪶 Zero-Impact Footprint**: Runs at below-normal priority in Windows 11 EcoQoS (Efficiency Mode on E-cores), using **~1.2% of one core** and **~65 MB RAM**.

---

## 📸 Illustrative Reports

### 1. Forensic Incident Explanation (`python monitor.py explain 284`)
```text
========================================================================
🔍 FORENSIC DIAGNOSTIC REPORT: Event #284 [UI_FREEZE]
   Time     : 2026-11-03 03:14:38 (2s duration)
   Severity : INFO
   Title    : Taskmgr.exe stopped responding for 1.6 s — dGPU woke up 1.1 s into it (±1 s)
========================================================================

📌 ROOT CAUSE ANALYSIS (RCA):
  • NVIDIA dGPU Wake-up Latency (PCIe D3cold → D0 transition): The discrete GPU woke up right 
    as 'Taskmgr.exe' was interacting with the system. On NVIDIA Advanced Optimus laptops, waking 
    the dGPU from deep sleep forces a PCIe link retraining delay of 0.5s–2.0s, which stalls 
    the calling thread or DWM display pipeline.

⚙️ CONTRIBUTING FACTORS & FORENSICS:
  • Heavy Local Computation (12.3 CPU-seconds): 'Taskmgr.exe' was actively executing CPU work 
    on its UI thread without yielding to the Windows message pump.

📊 TELEMETRY AT INCIDENT:
  Peak CPU: 13.3%  ·  Busiest Core: 86.2%  ·  RAM: 64.2%  ·  Commit: 79.0%  ·  Disk: 32.0 MB/s R / 34.9 MB/s W
  Top CPU   : Taskmgr.exe (1.02 cores) · brave.exe (0.61 cores) · System (0.41 cores)

💡 RECOMMENDED ACTIONS:
  ➜ If 'Taskmgr.exe' does not need 3D acceleration, set its GPU preference to 'Power saving' 
    (integrated GPU) in Windows Settings → System → Display → Graphics.
  ➜ In hardware monitors (HWiNFO, PredatorSense), disable periodic dGPU polling to avoid waking the card.
========================================================================
```

### 2. Explained ML Anomaly (`python monitor.py report`)
```text
#7   2026-10-08 14:02:10  WARNING  anomaly  lasted 2m 30s
    Unusual system behaviour: upload 18.4 MB/s, CPU 31% — top: SyncAgent.exe (2.4 cores)
    why   : upload 18.4 MB/s (similar moments: 3.1 KB/s; higher than 100.00% of history)
    why   : CPU 31% (similar moments: 2%; higher than 99.95% of history)
    cpu   : SyncAgent.exe 2.40 cores · svchost.exe (×4) 0.31 cores
    net   : SyncAgent.exe 1,290.4 MB
    in use: chrome.exe  (user away)
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mammadow/win-anomaly-monitor.git
cd win-anomaly-monitor

# Install dependencies
pip install -r requirements.txt
```

> [!NOTE]
> `nvidia-ml-py` is optional. If installed, full GPU clocks, temperature, VRAM, and power metrics are enabled.

### Common Commands

| Command | Action |
|:---|:---|
| `pythonw monitor.py` | Run silently as a background service (Startup shortcut) |
| `python monitor.py status` | Live system telemetry, model state, and monitor overhead |
| `python monitor.py report` | Summarize detected incidents from the last 24h (`-H 168` for a week) |
| `python monitor.py explain [ID]` | Run forensic root-cause analysis on an event (defaults to latest) |
| `python monitor.py ask "<query>"` | Ask natural language diagnostic questions about freezes or high load |
| `python monitor.py diagnose` | Executive health audit of recurring bottlenecks and hardware stalls |
| `python monitor.py tune` | Replay history to preview alert sensitivity quantiles |
| `python monitor.py reset` | Wipe collected database and re-learn a fresh baseline |
| `python -m pytest tests` | Run the complete automated test suite (120+ tests) |

---

## 🏗️ Architecture & Pipeline

```text
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                           DATA COLLECTION LAYER                             │
 ├────────────────────────────┬─────────────────────────────┬──────────────────┤
 │   1 Hz System Telemetry    │    UI Responsiveness Probe  │  5 s Process     │
 │   • CPU % & Busiest Core   │    (Independent Process)    │  Sweep           │
 │   • DPC & ISR Time         │    • SendMessageTimeout     │  • Fast native   │
 │   • RAM & Commit Charge    │      (WM_NULL to foreground)│    NtQuerySystem-│
 │   • Disk I/O & Network Bps │    • Explorer Taskbar Probe │    Information   │
 │   • dGPU Power / Clocks    │    • DwmFlush Compositor    │  • Top CPU, I/O, │
 │   • Power & User Presence  │      Frame Latency          │    Net, Handles  │
 └─────────────┬──────────────┴──────────────┬──────────────┴────────┬─────────┘
               │                             │                       │
               ▼                             ▼                       ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         ANALYTICS & STORAGE (SQLite WAL)                    │
 ├──────────────────────────────────────────┬──────────────────────────────────┤
 │  10-Second Windows                       │  Rule-Based Incident Engine      │
 │  • Aggregates into 16-D points           │  • UI Freezes (±3s context)      │
 │  • Scaled by fixed meaningful units      │  • Memory & Handle Leaks (R²≥0.8)│
 │  • Local Outlier Factor (LOF) scoring    │  • Thermal & dGPU Drain          │
 │  • Hourly background retraining          │  • Upload Bursts & Drive Space   │
 └──────────────────────────────────────────┴──────────────────────────────────┘
                                             │
                                             ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                     DIAGNOSTICS & CONVERSATIONAL AI                         │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │  • Built-in Deterministic Expert Engine (Zero dependencies, instant RCA)    │
 │  • Conversational Query Interface (Ollama / Local LLM / OpenAI API)         │
 │  • Executive System Health Auditing (Stall patterns, driver DPC storms)     │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛡️ What It Detects

| Event | Mechanism | Explanation & Threshold |
|:---|:---|:---|
| `ui_freeze` | Win32 Message Pump | Foreground window or desktop compositor stopped responding for $\ge 1\text{s}$. Correlates with $\pm 3\text{s}$ dGPU transitions, DPC storms, CPU saturation, or synchronous driver I/O. |
| `anomaly` | 16-D Machine Learning | Metric combination sits higher than 99.9% of historical windows for 3 consecutive windows (30s+), naming responsible processes. |
| `memory_pressure` | Commit Limit | RAM or Windows commit charge $\ge 90\%$ for 60s; lists the largest memory consumers. |
| `low_disk_space` | Virtual Memory Safety | System drive C: has $< 20\text{ GB}$ or $< 10\%$ free. Alerts to dynamic pagefile cap risks. |
| `upload_burst` | Socket Traffic | $\ge 20\text{ MB}$ sent within 30s. Names the responsible executable using per-process I/O socket counters. |
| `gpu_thermal` | Thermal Guard | GPU $\ge 87\text{ }^\circ\text{C}$ or reporting thermal throttling while $\ge 75\text{ }^\circ\text{C}$ for 30s. |
| `gpu_awake_on_battery` | Power Saver | dGPU awake but idle on battery for 10 min (draining 10–15 W). |
| `background_cpu` | CPU Scheduler | A background application consumes $\ge 0.8$ CPU cores for 5+ min. |
| `memory_leak` | Linear Regression | A process's private memory grows steadily ($\ge 500\text{ MB}$, $R^2 \ge 0.8$) over 1+ hours. |
| `handle_leak` | Kernel Handles | A process holds $> 100,000$ handles, or $> 10,000$ and steadily climbing. |

---

## 🔬 Engineering & Technical Edge Cases Solved

### Why Not Isolation Forest?
In early development (v2), Isolation Forest was benchmarked against recorded telemetry with injected incidents:
* **Detection rate**: Isolation Forest detected only **0–1%** of major incidents (60% CPU runaway, 20 MB/s stealth upload, 92% RAM choke, GPU at 85 °C).
* **False positive flood**: With `contamination=0.01`, it flagged 1% of normal idle seconds (~1 alert every 1.5 minutes).
* **The Solution**: The v3 detector uses a calibrated **16-dimensional distance metric** (LOF-inspired) comparing points against your machine's 10 nearest neighbors. It achieves **96–100% detection** on the same benchmark while ignoring harmless "lower than usual" states.

### NVIDIA Advanced Optimus & PCIe D3cold Stalls
On modern gaming laptops, polling NVML while the discrete GPU is asleep (D3cold) can:
1. Spudiously read 0 °C, tricking models into seeing massive temperature spikes.
2. Force a PCIe link wake-up, introducing 0.5s–2.0s stutter in the active window.

`resmon` treats dGPU sleep as an explicit state (`gpu_awake = 0`), handles temperature as unknown, and correlates application freezes with PCIe power transitions.

### Fast Win32 Process Sweeps
Standard libraries like `psutil` take ~1.3 seconds per sweep across 400+ processes. `resmon` uses direct ctypes bindings to `NtQuerySystemInformation(SystemProcessInformation)`, completing full process sweeps in **~5 milliseconds**.

### Clock Divergence: Sleep vs. Loop Starvation
When $>5\text{s}$ passes between two 1 Hz ticks, `resmon` checks two clocks: one that counts during system sleep (`GetTickCount64`) and one that pauses (`QueryUnbiasedInterruptTime`). If they diverged, the laptop slept—baselines are smoothly re-initialized. If they advanced together, the monitor's low-priority loop was starved by CPU congestion, and a sampling stall is recorded as part of the incident context.

---

## 📁 Repository Structure

```text
├── monitor.py                 # CLI entry point (status, report, explain, ask, diagnose)
├── config.example.toml        # Documented configuration template
├── pyproject.toml             # Standard packaging & tool configuration
├── requirements.txt           # Minimal dependencies (psutil, numpy, nvidia-ml-py)
│
├── resmon/
│   ├── diagnostics.py         # AI & forensic diagnostic engine (Ollama/OpenAI/Offline)
│   ├── winapi.py              # Native NT APIs (NtQuerySystemInformation, ctypes)
│   ├── sampler.py             # 1 Hz system, DPC/ISR, and GPU telemetry sampler
│   ├── responsiveness.py      # Independent UI freeze probe helper (SendMessageTimeout, DwmFlush)
│   ├── rules.py               # Deterministic rule engine (thermal, leaks, bursts, freezes)
│   ├── detector.py            # 16-D LOF-style distance anomaly detector
│   ├── storage.py             # SQLite WAL storage for samples, windows, and events
│   ├── engine.py              # Main monitoring loop & drift-free scheduler
│   ├── cli.py                 # Command-line interface definitions
│   └── config.py              # Configuration validation and dataclass
│
└── tests/                     # 120+ unit and integration tests (pytest)
```

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE).
