# ML System Resource Monitor v3

A background monitor for this laptop (i9-13900HX / RTX 4070 Laptop, Windows 11).
It learns what *normal* looks like for you (idle, gaming, on battery, away from
the keyboard…) and records an **explained event** when something is genuinely
off, naming the process responsible. An illustrative `report` entry:

```
#7   2026-10-08 14:02:10  WARNING  anomaly  lasted 2m 30s
    Unusual system behaviour: upload 18.4 MB/s, CPU 31% — top: SyncAgent.exe (2.4 cores)
    why   : upload 18.4 MB/s (similar moments: 3.1 KB/s; higher than 100.00% of history)
    why   : CPU 31% (similar moments: 2%; higher than 99.95% of history)
    cpu   : SyncAgent.exe 2.40 cores · svchost.exe (×4) 0.31 cores
    i/o   : SyncAgent.exe 18.9 MB/s
    in use: chrome.exe  (user away)
```

## Usage

```bash
pythonw monitor.py            # run silently (the Startup shortcut does this)
python monitor.py             # run with console output
python monitor.py status      # running? model state, live readings, own overhead
python monitor.py report      # events from the last 24 h  (-H 168 for a week, -v for detail)
python monitor.py tune        # how many alerts each sensitivity would give on your history
python monitor.py reset       # delete all collected data and start over
python -m pytest tests        # test suite
```

Requirements: `pip install -r requirements.txt` (psutil, numpy, nvidia-ml-py).
Optional settings go in `config.toml` (copy `config.example.toml`).

## What it detects

| Event | Meaning |
|---|---|
| `anomaly` | ML: for 30 s+, something sits well *above* every similar moment of the past week. Explains which metrics and who was busy. `info` if the app you're actively using caused it or the model has < 24 h of history; `warning` if extreme and not explained by what you're doing. |
| `memory_pressure` | RAM or commit charge ≥ 90 % for a minute; lists the largest processes. |
| `gpu_thermal` | GPU ≥ 87 °C, or reporting thermal throttling while ≥ 75 °C, for 30 s; lists GPU clients. (This laptop's driver raises its "thermal" flag at 32–52 °C while idle, so the flag alone is ignored.) |
| `gpu_awake_on_battery` | dGPU powered up but idle on battery for 10 min (it drains ~10–15 W); names what keeps it awake. |
| `background_cpu` | A process that is *not* the foreground app uses ≥ 0.8 cores for 5 min. |
| `memory_leak` | A process's private memory grows steadily (≥ 500 MB, R² ≥ 0.8, monotone) over ≥ 1 h. |
| `handle_leak` | A process holds > 100 000 handles, or holds > 10 000 and keeps growing steadily. |

A leaking executable is reported once. After that it's mentioned again only when it is twice as bad,
or a month later, even if the process is restarted, so a known, managed leak doesn't nag.

Rules work from the first second. The ML needs one hour of data before it starts.
It then improves as history accumulates; the first day is the noisiest.

## How it works

```
1 Hz system sample ──┬── samples table (48 h)
                     ├── rules: memory pressure, GPU thermal, dGPU on battery
5 s process sweep ───┼── rules: background CPU hog, memory/handle leaks
                     │   (+ who-was-busy history for explanations)
10 s window ─────────┴── score vs. history ── anomaly episodes ── events (SQLite + events.jsonl)
                          ▲
hourly retrain on ≤ 30 000 windows of the past 7 days (background thread, a few seconds)
```

**Detector.** Each 10-second window is a point in 16 dimensions: CPU, busiest core,
RAM, commit, disk read/write, upload/download, GPU load/power/temp/VRAM/clock, and
context (dGPU awake, AC power, user present). Each dimension is measured in fixed
"meaningful units": 5 % CPU, 5 % RAM, ×3 bytes/s, 5 °C, 10 W, and so on. The 10 most
similar historical windows are found using all dimensions. The score is how far the
window sits *above* them, counting only resources, because using less than usual is
never a problem and battery or user-away only choose the comparison. That distance
is divided by how spread out those similar windows usually are (the idea behind
Local Outlier Factor), so sparse but normal regimes such as gaming don't trigger
alerts. A window counts as anomalous when the score is in the top 0.1 % of your
own history *and* some metric is at least 2 units above similar moments (10 % CPU,
10 % RAM, ×10 bytes/s, 10 °C…). An *event* additionally needs 3 such windows in a row.

**Why not Isolation Forest (v2)?** Tested on this laptop's recorded history with
injected incidents, the forest detected **0–1 %** of:
- a 20 MB/s upload
- CPU at 60 %
- RAM at 92 %
- a GPU at 85 °C while drawing 20 W

The new detector catches **96–100 %** of those, plus a GPU crypto-miner, while ignoring
harmless "lower than usual" moments (RAM at 20 % is flagged 6 % of the time). Replaying
the old recording end-to-end gives **2 events** where v2 wrote **390 snapshots** for the same
period. Replaying this machine's first 45 h of v3 history gives 2 events, both real heavy use.

**v2 problems fixed**
- Crashed after 4 days (`time.sleep` with a negative duration). The loop now uses
  drift-free scheduling, and a failed tick is logged and skipped.
- `contamination=0.01` flagged 1 % of all seconds by construction (~1 alert/minute
  while idle). Thresholds are now calibrated, need persistence, and become episodes
  with a start and an end.
- The sleeping dGPU (80 % of the time) was recorded as 0 °C, so every wake looked
  like a 47 °C jump. "Asleep" is now a state.
- Network counter resets produced rates as low as −78 MB/s. They are clamped.
- A 590 W GPU power reading was accepted. Impossible readings are now dropped.
- "System Idle Process" (3155 %) topped every process snapshot, and other CPU
  figures were stale. Rates are now computed from cumulative counters, excluding
  Idle and the monitor itself.
- psutil needed ~1.3 s per sweep of ~400 processes. One
  `NtQuerySystemInformation` call now takes ~5 ms.
- A pegged single thread on 32 logical CPUs shows as only ~3 % total CPU. The
  busiest core is now tracked.
- DuckDB locked the database so nothing else could read it. SQLite in WAL mode
  lets `status` and `report` run alongside the monitor.

**Footprint** (measured here): about 0.8 % of one core (0.03 % of the whole CPU),
around 70 MB of RAM and about 7 ms per tick. It runs at below-normal priority in
Windows 11 Efficiency Mode (E-cores). `status` shows the live overhead.

## Acer handle-leak guard

This laptop's `Acer Agent Service` (`AASSvc`, `AcerCentralService.exe`) and `Acer System
Monitor Service` (`ASMSvc`, `AcerSystemCentralService.exe`) leak about 20 000 handles an hour
each. By 12 Oct 2026 they held 2.2 M and 2.5 M handles. No update was available (checked
Windows Update and Acer's PHN16-71 page), and disabling them can break PredatorSense. So a
scheduled task, **ResourceMonitor - Acer handle-leak guard**, runs hourly as SYSTEM and
restarts either service once it holds more than 200 000 handles. Restarts are logged to
`C:\ProgramData\ResourceMonitor\acer-handle-guard.log`. To remove it (admin PowerShell):

```powershell
Unregister-ScheduledTask -TaskName 'ResourceMonitor - Acer handle-leak guard' -Confirm:$false
```

## Files

| Path | What |
|---|---|
| `monitor.py` | Entry point / CLI |
| `resmon/` | `winapi` (fast native calls), `sampler` (system + GPU), `processes`, `features` (windows), `detector` (model), `rules` (all detectors), `events`, `storage`, `engine` (main loop), `replay` (offline tuning), `cli` |
| `data/metrics.db` | SQLite: `samples`, `windows`, `events` |
| `data/events.jsonl` | One line per finished event (a reopened event is written again; the later line wins) |
| `data/model.npz` | Last trained model (reloaded at start, so detection resumes immediately after a reboot) |
| `data/state.json` | Heartbeat used by `status` |
| `data/monitor.log` | Rotating log (2 MB × 4) |
