"""
Command line:

  python monitor.py                run the monitor (what the startup shortcut does)
  python monitor.py status         is it running? model state, current readings
  python monitor.py report [-H N]  events from the last N hours (default 24)
  python monitor.py explain [ID]   AI & forensic root-cause analysis of an event (default: latest)
  python monitor.py ask "question" natural language query about recent freezes or high usage
  python monitor.py diagnose [-H]  system health audit of recurring hardware/software friction
  python monitor.py tune           how many alerts each sensitivity would give on your history
  python monitor.py reset [--yes]  delete all collected data and start over
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from . import __version__
from .config import load_config
from .engine import DB_NAME, EVENTS_NAME, LOCK_NAME, LOG_NAME, STATE_NAME, InstanceLock, Monitor
from .events import fmt_duration
from .rules import summarize_rates

DATA_PATTERNS = [
    f"{DB_NAME}", f"{DB_NAME}-wal", f"{DB_NAME}-shm", EVENTS_NAME, STATE_NAME, "state.tmp",
    f"{LOG_NAME}*", LOCK_NAME,
    "metrics.duckdb*", "anomaly_snapshots.jsonl*",          # v2 leftovers
]


def setup_logging(path: Path, level: str, console: bool) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if console and sys.stderr is not None:   # pythonw has no stderr
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root.addHandler(ch)
    logging.captureWarnings(True)

    def log_uncaught(exc_type, exc, tb):
        logging.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
    sys.excepthook = log_uncaught
    threading.excepthook = lambda a: logging.critical(
        "Uncaught exception in thread %s", a.thread.name if a.thread else "?",
        exc_info=(a.exc_type, a.exc_value, a.exc_traceback))


def _ago(ts: float, now: float) -> str:
    return fmt_duration(max(0.0, now - ts)) + " ago"


def _dt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# ── commands ─────────────────────────────────────────────────────────

def cmd_run(cfg, args) -> int:
    cfg.data_path.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.data_path / LOG_NAME, cfg.log_level, console=True)
    for w in cfg.warnings:
        logging.warning(w)
    lock = InstanceLock(cfg.data_path / LOCK_NAME)
    if not lock.acquire():
        logging.error("Another monitor is already running on %s — exiting.", cfg.data_path)
        return 1
    try:
        Monitor(cfg).run()
    finally:
        lock.release()
    return 0


def cmd_status(cfg, args) -> int:
    from .storage import db_summary

    data = cfg.data_path
    running = InstanceLock.is_held(data / LOCK_NAME)
    state = {}
    if (data / STATE_NAME).exists():
        try:
            state = json.loads((data / STATE_NAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    now = time.time()
    print(f"Resource Monitor v{__version__}  ·  data: {data}")

    if running:
        started = datetime.fromisoformat(state["started"]).timestamp() if state.get("started") else now
        up = now - started
        cpu = state.get("self_cpu_seconds")
        overhead = f", own overhead {100 * cpu / up:.2f}% of one core, {state.get('self_memory_mb')} MB" \
            if cpu is not None and up > 60 else ""
        print(f"  Status  : RUNNING (pid {state.get('pid', '?')}, up {fmt_duration(up)}{overhead})")
    else:
        when = f" (last stopped {state['updated']})" if state.get("stopped") else \
            (f" (last heartbeat {state['updated']} — it may have crashed; see {LOG_NAME})" if state else "")
        print(f"  Status  : NOT RUNNING{when}")

    if not (data / DB_NAME).exists():
        print("  No data yet.")
        return 0
    s = db_summary(data / DB_NAME)
    latest = s["latest_sample"]
    if latest:
        print(f"  Latest  : {summarize_rates(latest)}  ({_ago(latest['ts'], now)})")
    m = state.get("model") if running else None
    if m:
        trained = datetime.fromisoformat(m["trained"]).timestamp()
        print(f"  Model   : trained {_ago(trained, now)} on {m['windows']:,} windows "
              f"({m['hours_of_data']} h of history)")
    else:
        need = int(cfg.min_training_hours * 3600 / cfg.window_sec)
        have = s["windows"]["n"]
        if have < need:
            print(f"  Model   : learning baseline — {have:,}/{need:,} windows "
                  f"(~{(need - have) * cfg.window_sec // 60 + 1} min of running left)")
        else:
            print("  Model   : enough data; trains when the monitor runs")
    probe = state.get("ui_probe") if running else None
    if probe:
        late = f", held up {probe['late_wakeups']}× (max {probe['max_late_sec']} s)" if probe.get("late_wakeups") else ""
        print(f"  UI probe: {probe.get('threads', 0)} thread(s), {probe.get('probes', 0):,} foreground probes, "
              f"{probe.get('freezes', 0)} freeze(s) recorded{late}")
    if state.get("ml_quiet_until") and running:
        print(f"  Note    : ML alerts paused until {state['ml_quiet_until']} (start-up / resume grace)")
    print(f"  Data    : {s['samples']['n']:,} samples, {s['windows']['n']:,} windows, "
          f"{s['events']:,} events, {s['size_mb']} MB on disk")
    if running and state.get("open_events"):
        print("  Open    :")
        for e in state["open_events"]:
            print(f"    #{e['id']} [{e['kind']}] {e['title']}")

    from .storage import query_events
    recent = query_events(data / DB_NAME, now - 86400)
    if recent:
        by_sev: dict[str, int] = {}
        for e in recent:
            by_sev[e["severity"]] = by_sev.get(e["severity"], 0) + 1
        sev = ", ".join(f"{n} {k}" for k, n in sorted(by_sev.items()))
        print(f"  Last 24h: {len(recent)} event(s) ({sev}) — run `python monitor.py report` for details")
    else:
        print("  Last 24h: no events")
    return 0


def _fmt_list(rows: list[dict], kind: str) -> str:
    parts = []
    for r in rows:
        n = f" (×{r['processes']})" if r.get("processes", 1) > 1 else ""
        match kind:
            case "cpu":
                parts.append(f"{r['name']}{n} {r['avg_cores']:.2f} cores")
            case "io":
                parts.append(f"{r['name']}{n} {r['avg_mb_s']:.1f} MB/s")
            case "net":
                parts.append(f"{r['name']}{n} {r['total_mb']:,.1f} MB")
            case "mem":
                parts.append(f"{r['name']}{n} {r['private_mb']:,} MB")
            case "gpu":
                parts.append(f"{r['name']} {r['sm_util_pct']}%")
    return " · ".join(parts)


def cmd_report(cfg, args) -> int:
    from .storage import query_events

    db = cfg.data_path / DB_NAME
    if not db.exists():
        print("No data yet.")
        return 0
    now = time.time()
    since = now - args.hours * 3600
    events = query_events(db, since, args.kind)
    if args.json:
        print(json.dumps(events, indent=2, default=str, ensure_ascii=False))
        return 0
    span = f"{args.hours:g} h" if args.hours < 48 else f"{args.hours / 24:g} days"
    print(f"Events in the last {span}: {len(events)}")
    for e in events:
        d = e["details"]
        dur = "" if e["ts_end"] - e["ts_start"] < 1 else f"  lasted {fmt_duration(e['ts_end'] - e['ts_start'])}"
        state = "  (ongoing)" if e["is_open"] else ""
        print()
        print(f"#{e['id']}  {_dt(e['ts_start'])}  {e['severity'].upper():8} {e['kind']}{dur}{state}")
        print(f"    {e['title']}")
        for item in d.get("why", []):
            print(f"    why   : {item['text']}")
        if d.get("rarity"):
            print(f"    rarity: {d['rarity']}")
        for clue in d.get("clues", [])[1:] if e["kind"] == "ui_freeze" else []:
            print(f"    also  : {clue}")
        for key, label, kind in (("top_net", "net   ", "net"), ("top_cpu", "cpu   ", "cpu"),
                                 ("top_io", "i/o   ", "io"), ("top_memory", "memory", "mem"),
                                 ("gpu_clients", "gpu   ", "gpu")):
            if d.get(key) and (args.verbose or key in ("top_cpu", "top_io", "gpu_clients")
                               or (key == "top_net" and e["kind"] in ("upload_burst", "anomaly"))
                               or e["kind"] == "memory_pressure"):
                print(f"    {label}: {_fmt_list(d[key][:4], kind)}")
        if d.get("foreground"):
            print(f"    in use: {d['foreground']}" + ("  (user away)" if d.get("user_idle") else ""))
        if d.get("ended") and d["ended"] not in ("responded", "compositor resumed"):
            print(f"    ended : {d['ended']}")
        if args.verbose:
            skip = {"why", "rarity", "top_cpu", "top_io", "top_net", "top_memory", "gpu_clients", "foreground",
                    "user_idle", "ended", "clues"}
            extra = {k: v for k, v in d.items() if k not in skip}
            if extra:
                print(f"    detail: {json.dumps(extra, ensure_ascii=False, default=str)}")
    if not events:
        print("Nothing unusual detected.")
    return 0


def cmd_tune(cfg, args) -> int:
    """Replay recorded history with different sensitivities."""
    from .replay import tune_grid
    from .storage import load_windows

    db = cfg.data_path / DB_NAME
    if not db.exists():
        print("No data yet.")
        return 0
    windows = load_windows(db, time.time() - cfg.training_days * 86400)
    need = int(cfg.min_training_hours * 3600 / cfg.window_sec) * 2
    if len(windows) < need:
        print(f"Not enough history to tune yet ({len(windows):,}/{need:,} windows). "
              f"Let it run for a few hours first.")
        return 0
    print(f"Replaying {len(windows):,} windows ({len(windows) * cfg.window_sec / 3600:.1f} h of data) "
          f"with hourly retraining…")
    rows = tune_grid(windows, cfg)
    print()
    print("  alert_quantile  min_consecutive_windows   ML events   per day")
    for r in rows:
        mark = "  ← current" if (r["alert_quantile"] == cfg.alert_quantile
                                and r["min_consecutive_windows"] == cfg.min_consecutive_windows) else ""
        print(f"  {r['alert_quantile']:<14g}  {r['min_consecutive_windows']:<23d}   {r['events']:<9d}   "
              f"{r['per_day']:<6.1f}{mark}")
    print()
    print("The first hours of any history are noisy (the model is still young); judge by later days.")
    print("Set your choice in config.toml, then restart the monitor.")
    return 0


def cmd_reset(cfg, args) -> int:
    data = cfg.data_path
    if InstanceLock.is_held(data / LOCK_NAME):
        print("The monitor is running — stop it first (Task Manager → pythonw.exe running monitor.py).")
        return 1
    targets = sorted({p for pat in DATA_PATTERNS for p in data.glob(pat) if p.is_file()})
    if not targets:
        print(f"Nothing to delete in {data}.")
        return 0
    total = sum(p.stat().st_size for p in targets) / 2**20
    print(f"This deletes {len(targets)} file(s), {total:.1f} MB, from {data}:")
    for p in targets:
        print(f"  {p.name}")
    if not args.yes and input("Type 'yes' to continue: ").strip().lower() != "yes":
        print("Cancelled.")
        return 1
    for p in targets:
        p.unlink()
    print("Done — the monitor will learn a fresh baseline on its next start.")
    return 0




def cmd_explain(cfg, args) -> int:
    from .diagnostics import build_diagnostic_prompt, collect_event_forensics, explain_event
    from .storage import get_event_by_id, query_events

    db = cfg.data_path / DB_NAME
    if not db.exists():
        print("No data yet.")
        return 0

    event = None
    target_id_str = getattr(args, "event_id", "latest") or "latest"
    if target_id_str.lower() == "latest":
        recent = query_events(db, time.time() - 86400 * 7)
        if recent:
            event = recent[-1]
    else:
        try:
            eid = int(target_id_str.lstrip("#"))
            event = get_event_by_id(db, eid)
        except ValueError:
            print(f"Invalid event ID: {target_id_str}. Must be an integer or 'latest'.")
            return 1

    if not event:
        print(f"Event '{target_id_str}' not found in {db}.")
        return 1

    if getattr(args, "prompt", False):
        forensics = collect_event_forensics(db, event)
        print(build_diagnostic_prompt(forensics, user_query=getattr(args, "question", None)))
        return 0

    result = explain_event(cfg, event, user_query=getattr(args, "question", None),
                           force_offline=getattr(args, "offline", False))
    print(result)
    return 0


def cmd_ask(cfg, args) -> int:
    from .diagnostics import explain_event, parse_time_or_query_targets
    from .storage import query_events

    db = cfg.data_path / DB_NAME
    if not db.exists():
        print("No data yet.")
        return 0

    question = args.question
    events = query_events(db, time.time() - args.hours * 3600)
    if not events:
        print(f"No events found in the last {args.hours:g} hours to diagnose.")
        return 0

    event, match_reason = parse_time_or_query_targets(question, events)
    if not event:
        print("Could not match your question to any recorded incident. Try `python monitor.py report`.")
        return 0

    print(f"🔍 Analyzing {match_reason}...")
    result = explain_event(cfg, event, user_query=question,
                           force_offline=getattr(args, "offline", False))
    print(result)
    return 0


def cmd_diagnose(cfg, args) -> int:
    from .diagnostics import audit_system_health

    print(audit_system_health(cfg, hours=args.hours))
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):      # pipes default to the ANSI code page on Windows
        if stream is not None:
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass
    ap = argparse.ArgumentParser(prog="monitor.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="config file (default: config.toml next to monitor.py)")
    ap.add_argument("--data-dir", help="override data_dir")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("run", help="run the monitor in the foreground (default)")
    sub.add_parser("status", help="show whether it is running and what it knows")
    rp = sub.add_parser("report", help="list detected events")
    rp.add_argument("-H", "--hours", type=float, default=24, help="look back this many hours (default 24)")
    rp.add_argument("-k", "--kind", action="append",
                    help="only these event kinds (anomaly, memory_pressure, gpu_thermal, "
                         "gpu_awake_on_battery, low_disk_space, upload_burst, ui_freeze, background_cpu, "
                         "memory_leak, handle_leak)")
    rp.add_argument("-v", "--verbose", action="store_true", help="show every detail")
    rp.add_argument("--json", action="store_true", help="machine-readable output")
    sub.add_parser("tune", help="replay your history to see how many alerts each sensitivity gives")
    zp = sub.add_parser("reset", help="delete all collected data")
    zp.add_argument("--yes", action="store_true", help="don't ask for confirmation")

    ep = sub.add_parser("explain", help="AI & forensic explanation of a specific event")
    ep.add_argument("event_id", nargs="?", default="latest", help="event ID (or 'latest', default: latest)")
    ep.add_argument("-q", "--question", help="optional specific question about this event")
    ep.add_argument("--offline", action="store_true", help="force built-in deterministic expert engine (no LLM)")
    ep.add_argument("--prompt", action="store_true", help="output LLM prompt and payload without querying")

    ak = sub.add_parser("ask", help="ask a question about recent system behavior in natural language")
    ak.add_argument("question", help="e.g. 'Why did WhatsApp freeze?' or 'Why did my laptop stutter around 2 PM?'")
    ak.add_argument("-H", "--hours", type=float, default=24, help="look back this many hours (default 24)")
    ak.add_argument("--offline", action="store_true", help="force built-in deterministic expert engine")

    dp = sub.add_parser("diagnose", help="run an executive system health audit over recent history")
    dp.add_argument("-H", "--hours", type=float, default=24, help="look back this many hours (default 24)")

    args = ap.parse_args(argv)

    cfg = load_config(args.config, data_dir=args.data_dir)
    handler = {
        "run": cmd_run, "status": cmd_status, "report": cmd_report, "tune": cmd_tune,
        "reset": cmd_reset, "explain": cmd_explain, "ask": cmd_ask, "diagnose": cmd_diagnose,
    }
    return handler[args.cmd or "run"](cfg, args)
