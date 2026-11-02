"""
AI-Assisted Event Explanation & Conversational Diagnostics for resmon.

Provides deep root-cause analysis (RCA) and actionable recommendations for
system anomalies, UI freezes, memory leaks, and performance stalls.

Features:
  1. Offline Expert Diagnostic Engine: Zero-dependency, deterministic systems
     analysis built for Windows 11 / hybrid laptops (Optimus D3cold wakeups,
     DPC/ISR storms, commit charge limits, thread saturation). Always works,
     even without an LLM running.
  2. LLM Integration: Seamlessly pipes rich forensic telemetry (+-10s raw 1 Hz
     samples, process sweeps, hardware states) to local Ollama (llama3.2, etc.)
     or OpenAI-compatible APIs.
  3. Natural Language Question Answering: Resolves queries like "Why did WhatsApp freeze?"
     or "Why did my laptop stutter around 2 PM?".
  4. System-Wide Health Auditing: Summarizes recurring issues and hardware friction.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import MonitorConfig
from .events import fmt_duration
from .storage import connect, get_event_by_id, query_events, query_samples


def _dt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _format_rate(bytes_per_sec: float) -> str:
    if bytes_per_sec < 1024:
        return f"{bytes_per_sec:.0f} B/s"
    if bytes_per_sec < 1024 * 1024:
        return f"{bytes_per_sec / 1024:.1f} KB/s"
    return f"{bytes_per_sec / (1024 * 1024):.1f} MB/s"


# ── Forensics Gathering ───────────────────────────────────────────────────────

def collect_event_forensics(db_path: Path, event: dict, window_padding_sec: float = 10.0) -> dict[str, Any]:
    """
    Enrich an event with raw 1 Hz telemetry samples (+- window_padding_sec around it)
    and synthesize peak metrics, dGPU transitions, and hardware state shifts.
    """
    ts_start = event.get("ts_start", 0.0)
    ts_end = event.get("ts_end", ts_start)
    details = event.get("details", {})

    samples = query_samples(db_path, ts_start - window_padding_sec, ts_end + window_padding_sec)

    cpu_vals = [s["cpu_pct"] for s in samples if s.get("cpu_pct") is not None]
    core_max_vals = [s["core_max_pct"] for s in samples if s.get("core_max_pct") is not None]
    ram_vals = [s["ram_pct"] for s in samples if s.get("ram_pct") is not None]
    commit_vals = [s["commit_pct"] for s in samples if s.get("commit_pct") is not None]
    disk_read_vals = [s["disk_read_bps"] for s in samples if s.get("disk_read_bps") is not None]
    disk_write_vals = [s["disk_write_bps"] for s in samples if s.get("disk_write_bps") is not None]
    net_sent_vals = [s["net_sent_bps"] for s in samples if s.get("net_sent_bps") is not None]
    net_recv_vals = [s["net_recv_bps"] for s in samples if s.get("net_recv_bps") is not None]
    dpc_vals = [s["dpc_max_pct"] for s in samples if s.get("dpc_max_pct") is not None]
    isr_vals = [s["isr_max_pct"] for s in samples if s.get("isr_max_pct") is not None]
    gpu_awake_vals = [s["gpu_awake"] for s in samples if s.get("gpu_awake") is not None]

    # Detect dGPU wake/sleep state changes across the raw telemetry
    gpu_transitions: list[str] = []
    for i in range(1, len(gpu_awake_vals)):
        prev = gpu_awake_vals[i - 1]
        curr = gpu_awake_vals[i]
        if prev == 0 and curr == 1:
            rel_t = samples[i]["ts"] - ts_start
            gpu_transitions.append(f"woke up at {rel_t:+.1f}s")
        elif prev == 1 and curr == 0:
            rel_t = samples[i]["ts"] - ts_start
            gpu_transitions.append(f"went to sleep at {rel_t:+.1f}s")

    telemetry_summary = {
        "sample_count": len(samples),
        "peak_cpu_pct": max(cpu_vals) if cpu_vals else None,
        "peak_core_max_pct": max(core_max_vals) if core_max_vals else None,
        "peak_ram_pct": max(ram_vals) if ram_vals else None,
        "peak_commit_pct": max(commit_vals) if commit_vals else None,
        "peak_disk_read": _format_rate(max(disk_read_vals)) if disk_read_vals else "0 B/s",
        "peak_disk_write": _format_rate(max(disk_write_vals)) if disk_write_vals else "0 B/s",
        "peak_net_sent": _format_rate(max(net_sent_vals)) if net_sent_vals else "0 B/s",
        "peak_net_recv": _format_rate(max(net_recv_vals)) if net_recv_vals else "0 B/s",
        "peak_dpc_pct": max(dpc_vals) if dpc_vals else None,
        "peak_isr_pct": max(isr_vals) if isr_vals else None,
        "gpu_transitions": gpu_transitions,
    }

    return {
        "event": event,
        "details": details,
        "telemetry": telemetry_summary,
        "raw_samples": samples,
    }


# ── Built-in Expert Diagnostic Engine (Offline / Deterministic) ───────────────

def diagnose_offline(forensics: dict[str, Any]) -> str:
    """
    Expert rule-based RCA without any LLM dependencies.
    Understands Windows 11 internals, NVIDIA Optimus transitions, DPC storms,
    thread preemption, and memory/commit limits.
    """
    event = forensics["event"]
    details = forensics["details"]
    telemetry = forensics["telemetry"]

    kind = event.get("kind", "anomaly")
    severity = event.get("severity", "info").upper()
    title = event.get("title", "")
    ts_start = event.get("ts_start", 0.0)
    ts_end = event.get("ts_end", ts_start)
    duration_sec = max(0.0, ts_end - ts_start)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"🔍 FORENSIC DIAGNOSTIC REPORT: Event #{event.get('id', '?')} [{kind.upper()}]")
    lines.append(f"   Time     : {_dt(ts_start)} ({fmt_duration(duration_sec)} duration)")
    lines.append(f"   Severity : {severity}")
    lines.append(f"   Title    : {title}")
    lines.append("=" * 72)

    root_causes: list[str] = []
    contributing: list[str] = []
    recommendations: list[str] = []

    # 1. UI Freeze Diagnosis
    if kind == "ui_freeze":
        proc = details.get("process", "unknown")
        scope = details.get("scope", "app")
        app_cpu = details.get("app_cpu_sec", 0.0)
        dgpu_trans = details.get("dgpu_transitions", [])
        dpc_spike = details.get("dpc_isr_spike", False)
        around = details.get("around", {})
        loop_gap = around.get("monitor_loop_gap_sec", 1.0)

        # Check dGPU wake-up trigger: prioritize probe's recorded dgpu_transitions if present
        if "dgpu_transitions" in details:
            has_dgpu_wake = any("woke up" in str(t).lower() for t in details["dgpu_transitions"])
        else:
            has_dgpu_wake = any("woke up" in t for t in telemetry.get("gpu_transitions", []))

        if has_dgpu_wake:
            root_causes.append(
                f"NVIDIA dGPU Wake-up Latency (PCIe D3cold → D0 transition): "
                f"The discrete GPU woke up right as '{proc}' was interacting with the system. "
                f"On NVIDIA Advanced Optimus laptops, waking the dGPU from deep sleep forces a PCIe link "
                f"retraining delay of 0.5s–2.0s, which stalls the calling thread or DWM display pipeline."
            )
            recommendations.append(
                f"If '{proc}' does not need dedicated 3D acceleration (e.g. Task Manager, PredatorSense, browser), "
                f"set its GPU preference to 'Power saving' (integrated GPU) in Windows Settings → System → Display → Graphics."
            )
            recommendations.append(
                "If using hardware monitoring tools (HWiNFO, PredatorSense), disable periodic dGPU polling to avoid waking the card."
            )

        if not has_dgpu_wake and loop_gap and loop_gap >= 2.0:
            top_cpu = details.get("top_cpu", [])
            top_cpu_str = ", ".join(f"{p['name']} ({p.get('avg_cores', 0):.2f} cores)" for p in top_cpu[:3]) if top_cpu else "background processes"
            busiest = around.get("busiest_core_pct") or telemetry.get("peak_core_max_pct") or 100
            root_causes.append(
                f"CPU Scheduler Starvation & Heavy Background Congestion: The system experienced a {loop_gap:.1f}s sampling stall "
                f"while the busiest core reached {busiest:.0f}%. Background workloads ({top_cpu_str}) "
                f"saturated processor queue capacity, preventing '{proc}' from processing window messages within the timeout."
            )
            recommendations.append(
                "Verify whether Windows Update or anti-malware background processes are running without EcoQoS (Efficiency Mode)."
            )
        elif not has_dgpu_wake and app_cpu and app_cpu > 1.0:
            root_causes.append(
                f"UI Thread Blocking Computation: '{proc}' performed {app_cpu:.1f} CPU-seconds of intensive work directly on its "
                f"main message-pump thread instead of offloading it to a background worker thread."
            )

        if loop_gap and loop_gap >= 2.0 and has_dgpu_wake:
            contributing.append(
                f"System-wide Scheduler Stall ({loop_gap:.1f}s monitor sampling gap): "
                f"The monitor's own low-priority thread was preempted, indicating either severe CPU starvation, "
                f"heavy kernel lock contention, or high-priority DPC execution."
            )

        if app_cpu and app_cpu > 1.0 and (has_dgpu_wake or (loop_gap and loop_gap >= 2.0)):
            contributing.append(
                f"Heavy Local Computation ({app_cpu:.1f} CPU-seconds): '{proc}' was actively executing CPU work on its UI thread "
                f"without yielding to the Windows message pump."
            )
        elif app_cpu is not None and app_cpu < 0.5:
            contributing.append(
                f"Synchronous I/O or Lock Wait: '{proc}' was nearly idle ({app_cpu:.2f} CPU-s), indicating it stalled while "
                f"synchronously waiting on a driver, disk read, or inter-process COM/RPC lock."
            )

        if dpc_spike or (telemetry.get("peak_dpc_pct") and telemetry["peak_dpc_pct"] > 5.0):
            root_causes.append(
                f"Kernel DPC/ISR Storm (Peak DPC {telemetry.get('peak_dpc_pct', 0):.1f}%): A kernel driver held execution "
                f"at elevated IRQL, blocking user-mode thread scheduling."
            )
            recommendations.append("Run LatencyMon to identify if an audio, Wi-Fi, or GPU driver is introducing DPC spikes.")

    # 2. Upload Burst Diagnosis
    elif kind == "upload_burst":
        sent_mb = details.get("sent_mb", 0.0)
        top_net = details.get("top_net", [])
        foreground = details.get("foreground", "")
        top_name = top_net[0]["name"] if top_net else "unknown"

        if foreground and top_name.lower() == foreground.lower():
            root_causes.append(
                f"User-Initiated Large Network Upload: Uploaded {sent_mb:.1f} MB in {fmt_duration(duration_sec)} "
                f"directly from active app '{foreground}'."
            )
        else:
            root_causes.append(
                f"Background Outbound Data Burst: Uploaded {sent_mb:.1f} MB in {fmt_duration(duration_sec)} "
                f"driven primarily by '{top_name}' while foreground app was '{foreground or 'none'}'."
            )
            recommendations.append(
                f"Verify if '{top_name}' was performing a routine cloud sync, package download, or API call. "
                f"If '{top_name}' is a known trusted service, you can whitelist it under 'ignore_processes' in config.toml."
            )

    # 3. Memory Pressure / Low Disk Space Diagnosis
    elif kind in ("memory_pressure", "low_disk_space"):
        if kind == "low_disk_space":
            root_causes.append(
                "System Drive (C:) Space Exhaustion: Free space is critically low (<10% or <20 GB). "
                "Because the Windows page file automatically expands into free disk space, a full C: drive caps "
                "the total Commit Limit near physical RAM size, creating severe risk of allocation failures under load."
            )
            recommendations.append(
                "Free up at least 20–30 GB on drive C: (run Disk Cleanup, clean temp files, or move large games) "
                "to allow dynamic pagefile expansion."
            )
        else:
            root_causes.append(
                "High RAM or Commit Charge Pressure (≥90% for sustained period): System memory or virtual commit is near capacity."
            )
            top_mem = details.get("top_memory", [])
            if top_mem:
                names = ", ".join(f"{m['name']} ({m.get('private_mb', 0):,} MB)" for m in top_mem[:3])
                contributing.append(f"Top memory consumers: {names}")
            recommendations.append("Close unneeded high-memory applications or increase virtual memory paging limit.")

    # 4. Memory / Handle Leaks
    elif kind in ("memory_leak", "handle_leak"):
        proc = details.get("process", "unknown")
        growth = details.get("growth_mb", details.get("growth_handles", 0))
        root_causes.append(
            f"Resource Leak in '{proc}': Steadily accumulated {growth} over the monitored window with monotonic growth (R² ≥ 0.8)."
        )
        if "dwm.exe" in proc.lower():
            contributing.append(
                "Desktop Window Manager (DWM) leak: Known Windows 11 graphics driver interop issue with hybrid Intel/NVIDIA laptops."
            )
            recommendations.append("Update Intel Graphics and NVIDIA drivers to the latest OEM or WHQL releases.")
        else:
            recommendations.append(f"Consider restarting '{proc}' or reporting the leak to its developers.")

    # 5. Background CPU
    elif kind == "background_cpu":
        proc = details.get("process", "unknown")
        cores = details.get("avg_cores", 0.0)
        fg = details.get("foreground_app", "unknown")
        root_causes.append(
            f"Background CPU Hog: '{proc}' consumed {cores:.2f} CPU cores continuously for {fmt_duration(duration_sec)} "
            f"while user was working in '{fg}'."
        )
        if "svchost.exe" in proc.lower():
            contributing.append(
                "Windows Service Host: Typically Windows Update (wuauserv), Defender scans, or Background Intelligent Transfer (BITS)."
            )
            recommendations.append("Check Task Manager → Services or Windows Update settings if this recurs frequently.")
        else:
            recommendations.append(f"Set '{proc}' to Windows 11 Efficiency Mode (EcoQoS) or lower its priority.")

    # 6. ML Anomaly
    else:
        root_causes.append(
            f"Statistical Metric Outlier: Resource usage sat higher than 99.9% of historical windows for this machine regime."
        )
        for item in details.get("why", []):
            contributing.append(item.get("text", ""))

    # Fallback if no specific root cause was generated
    if not root_causes:
        root_causes.append(f"Identified anomalous activity: {title}")

    # Build formatted section blocks
    lines.append("\n📌 ROOT CAUSE ANALYSIS (RCA):")
    for rc in root_causes:
        lines.append(f"  • {rc}")

    if contributing:
        lines.append("\n⚙️ CONTRIBUTING FACTORS & FORENSICS:")
        for cb in contributing:
            lines.append(f"  • {cb}")

    lines.append("\n📊 TELEMETRY AT INCIDENT:")
    t_lines = []
    if telemetry.get("peak_cpu_pct") is not None:
        t_lines.append(f"Peak CPU: {telemetry['peak_cpu_pct']:.1f}%")
    if telemetry.get("peak_core_max_pct") is not None:
        t_lines.append(f"Busiest Core: {telemetry['peak_core_max_pct']:.1f}%")
    if telemetry.get("peak_ram_pct") is not None:
        t_lines.append(f"RAM: {telemetry['peak_ram_pct']:.1f}%")
    if telemetry.get("peak_commit_pct") is not None:
        t_lines.append(f"Commit: {telemetry['peak_commit_pct']:.1f}%")
    if telemetry.get("peak_disk_read") or telemetry.get("peak_disk_write"):
        t_lines.append(f"Disk: {telemetry.get('peak_disk_read')} R / {telemetry.get('peak_disk_write')} W")
    if telemetry.get("peak_dpc_pct") is not None and telemetry["peak_dpc_pct"] > 0.1:
        t_lines.append(f"Max DPC: {telemetry['peak_dpc_pct']:.2f}%")
    lines.append("  " + "  ·  ".join(t_lines))

    # Top processes
    for key, label in (("top_cpu", "Top CPU"), ("top_net", "Top Net"), ("top_memory", "Top Memory")):
        if details.get(key):
            procs = details[key][:3]
            proc_str = " · ".join(f"{p['name']} ({p.get('avg_cores', p.get('total_mb', p.get('private_mb', '')))})" for p in procs)
            lines.append(f"  {label:<10}: {proc_str}")

    if recommendations:
        lines.append("\n💡 RECOMMENDED ACTIONS:")
        for rec in recommendations:
            lines.append(f"  ➜ {rec}")

    lines.append("=" * 72)
    return "\n".join(lines)


# ── LLM Prompt Builder & Provider Client ──────────────────────────────────────

def build_diagnostic_prompt(forensics: dict[str, Any], user_query: str | None = None) -> str:
    """
    Construct a high-density, context-rich prompt for an LLM (Ollama / OpenAI).
    """
    event = forensics["event"]
    details = forensics["details"]
    telemetry = forensics["telemetry"]

    context = {
        "event_id": event.get("id"),
        "kind": event.get("kind"),
        "severity": event.get("severity"),
        "title": event.get("title"),
        "timestamp": _dt(event.get("ts_start", 0)),
        "duration_sec": event.get("ts_end", 0) - event.get("ts_start", 0),
        "telemetry_peaks": telemetry,
        "details": details,
    }

    prompt = f"""You are a Principal Windows Systems Performance Engineer analyzing real forensic data from a high-performance Windows 11 laptop (Intel Core i9-13900HX, NVIDIA RTX 4070 Laptop GPU, 32+ logical cores).

Here is the exact telemetry and event context recorded by the background resource monitor:
```json
{json.dumps(context, indent=2, default=str)}
```

Hardware & OS Architecture Context:
- NVIDIA Advanced Optimus: The discrete GPU rests in deep PCIe D3cold sleep while on desktop. Querying it or waking it causes a 500ms–2000ms PCIe bus retraining stall.
- The CPU has 32 logical threads (Performance + Efficient cores). A single peg on 1 core shows as ~3% overall CPU, but 100% on that core.
- Page file expansion relies on free disk space on C:. Low disk space caps commit limit near RAM size.

{f"USER QUESTION: {user_query}" if user_query else "TASK: Provide a comprehensive diagnostic analysis of this event."}

Provide a concise, direct, expert diagnostic explanation with:
1. **Summary**: Plain-English 1-2 sentence explanation of what happened.
2. **Root Cause Analysis (RCA)**: The primary trigger (dGPU wake, DPC storm, single-core saturation, thread lock).
3. **Contributing Factors**: Secondary processes or I/O load.
4. **Actionable Recommendations**: 1-2 concrete, technical steps to avoid or fix it.
"""
    return prompt


def query_ollama(prompt: str, url: str = "http://127.0.0.1:11434", model: str = "llama3.2", timeout_sec: int = 45) -> str:
    """Send prompt to local Ollama instance."""
    api_endpoint = f"{url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result.get("response", "").strip()


def query_openai(prompt: str, base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o-mini", api_key: str = "", timeout_sec: int = 45) -> str:
    """Send prompt to OpenAI-compatible REST API."""
    api_endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a Principal Windows Systems Performance Engineer."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(api_endpoint, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        choices = result.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return "No response content received from API."


def is_ollama_available(url: str = "http://127.0.0.1:11434", timeout_sec: float = 1.0) -> bool:
    """Quick check if Ollama server is currently listening."""
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return resp.status == 200
    except Exception:
        return False


def explain_event(cfg: MonitorConfig, event: dict, user_query: str | None = None, force_offline: bool = False) -> str:
    """
    Produce the best possible explanation for an event.
    Uses LLM (Ollama / OpenAI) if available and requested; otherwise uses the built-in Expert Engine.
    """
    forensics = collect_event_forensics(cfg.data_path / "metrics.db", event)

    if force_offline or cfg.ai_provider == "offline":
        return diagnose_offline(forensics)

    # Check provider options
    provider = cfg.ai_provider
    ollama_url = os.environ.get("OLLAMA_BASE_URL", cfg.ollama_url)
    ollama_model = os.environ.get("OLLAMA_MODEL", cfg.ollama_model)
    openai_key = os.environ.get("OPENAI_API_KEY", cfg.openai_api_key)
    openai_base = os.environ.get("OPENAI_BASE_URL", cfg.openai_base_url)

    prompt = build_diagnostic_prompt(forensics, user_query)

    # Auto mode resolution
    if provider == "auto":
        if is_ollama_available(ollama_url):
            provider = "ollama"
        elif openai_key:
            provider = "openai"
        else:
            # Fall back to offline engine with an informative header
            offline_report = diagnose_offline(forensics)
            return (
                offline_report + "\n\n"
                "ℹ️ [Note: Running in Built-in Offline Expert Mode. "
                f"To enable local LLM synthesis, start Ollama ('ollama serve') or set OPENAI_API_KEY.]"
            )

    try:
        if provider == "ollama":
            response = query_ollama(prompt, url=ollama_url, model=ollama_model)
            return (
                f"🤖 AI DIAGNOSTIC REPORT (via Ollama: {ollama_model})\n"
                f"   Event #{event.get('id', '?')} [{event.get('kind', '?')}] at {_dt(event.get('ts_start', 0))}\n"
                f"{'=' * 72}\n"
                f"{response}\n"
                f"{'=' * 72}"
            )
        elif provider == "openai":
            response = query_openai(prompt, base_url=openai_base, model=cfg.openai_model, api_key=openai_key)
            return (
                f"🤖 AI DIAGNOSTIC REPORT (via OpenAI: {cfg.openai_model})\n"
                f"   Event #{event.get('id', '?')} [{event.get('kind', '?')}] at {_dt(event.get('ts_start', 0))}\n"
                f"{'=' * 72}\n"
                f"{response}\n"
                f"{'=' * 72}"
            )
    except Exception as e:
        logging.warning("AI provider failed (%s), falling back to offline diagnostic engine", e)
        offline_report = diagnose_offline(forensics)
        return (
            offline_report + "\n\n"
            f"⚠️ [Note: {provider.capitalize()} query failed ({e}). Fell back to Built-in Offline Expert Engine.]"
        )

    return diagnose_offline(forensics)


# ── Natural Language Query & Time Resolution ──────────────────────────────────

def parse_time_or_query_targets(query: str, events: list[dict]) -> tuple[dict | None, str | None]:
    """
    Examine user question (e.g. "Why did WhatsApp freeze?", "What happened at 03:14?", "Why did it lag 10m ago?").
    Returns (matching_event, extracted_context).
    """
    if not events:
        return None, None

    query_lower = query.lower()

    # 1. Match specific event id like "#284" or "event 284"
    id_match = re.search(r"(?:event\s+|#)(\d+)", query_lower)
    if id_match:
        target_id = int(id_match.group(1))
        for e in events:
            if e.get("id") == target_id:
                return e, f"Direct event #{target_id} match"

    # 2. Match process names mentioned in query
    for e in reversed(events):
        details = e.get("details", {})
        proc = details.get("process", "").lower()
        title = e.get("title", "").lower()
        fg = details.get("foreground", "").lower()

        # Check if process executable or base name is in the query (e.g. "whatsapp", "taskmgr", "brave")
        for candidate in (proc, fg):
            if candidate:
                base = candidate.replace(".exe", "")
                if base and base in query_lower:
                    return e, f"Matched process '{candidate}' in Event #{e.get('id')}"

        # Match in event title
        if any(word in title for word in query_lower.split() if len(word) > 4):
            return e, f"Matched keywords in Event #{e.get('id')}"

    # 3. Default to the latest event if user asks generally "Why did my laptop freeze/stutter?"
    freeze_keywords = ("freeze", "stutter", "lag", "hang", "slow", "unresponsive")
    if any(k in query_lower for k in freeze_keywords):
        for e in reversed(events):
            if e.get("kind") == "ui_freeze":
                return e, f"Matched most recent UI freeze (Event #{e.get('id')})"

    return events[-1], f"Latest event (Event #{events[-1].get('id')})"


# ── System Health Audit ───────────────────────────────────────────────────────

def audit_system_health(cfg: MonitorConfig, hours: float = 24.0) -> str:
    """
    Synthesize an executive systems health audit over the specified time window.
    Detects recurrent architectural issues (Optimus stalls, leaks, disk hazards).
    """
    db_path = cfg.data_path / "metrics.db"
    if not db_path.exists():
        return "No database found. Let the monitor run to collect telemetry."

    now = time.time()
    events = query_events(db_path, now - hours * 3600)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"🏥 SYSTEM HEALTH AUDIT (Past {hours:g} Hours)")
    lines.append(f"   Evaluated at: {_dt(now)}  ·  Events Found: {len(events)}")
    lines.append("=" * 72)

    if not events:
        lines.append("✅ All Clear: No performance anomalies, UI freezes, or resource leaks detected.")
        lines.append("=" * 72)
        return "\n".join(lines)

    # Classify events
    by_kind: dict[str, list[dict]] = {}
    for e in events:
        by_kind.setdefault(e["kind"], []).append(e)

    # Detect patterns
    dgpu_freeze_count = 0
    dgpu_trigger_apps: set[str] = set()

    for e in by_kind.get("ui_freeze", []):
        d = e.get("details", {})
        trans = d.get("dgpu_transitions", [])
        if any("woke up" in str(t).lower() for t in trans):
            dgpu_freeze_count += 1
            if d.get("process"):
                dgpu_trigger_apps.add(d["process"])

    warnings: list[str] = []

    # Check 1: Optimus Stalls
    if dgpu_freeze_count > 0:
        apps_str = ", ".join(f"'{a}'" for a in sorted(dgpu_trigger_apps))
        warnings.append(
            f"⚡ NVIDIA Optimus D3cold Wake-up Latency ({dgpu_freeze_count} freeze events): "
            f"The discrete RTX 4070 GPU was awakened by hardware-probing software ({apps_str}). "
            f"Each wake causes a 0.5s–2.0s PCIe link latency that freezes desktop windows."
        )

    # Check 2: Low Disk Space
    if "low_disk_space" in by_kind:
        latest_disk = by_kind["low_disk_space"][-1]
        warnings.append(
            f"💾 System Drive C: Low Free Space Hazard: {latest_disk.get('title')}. "
            f"This restricts the dynamic page file, capping the Windows commit limit."
        )

    # Check 3: Leaks
    if "memory_leak" in by_kind or "handle_leak" in by_kind:
        leaks = by_kind.get("memory_leak", []) + by_kind.get("handle_leak", [])
        for lk in leaks:
            warnings.append(f"💧 Resource Leak: {lk.get('title')}")

    # Check 4: Upload bursts
    if "upload_burst" in by_kind:
        count = len(by_kind["upload_burst"])
        warnings.append(f"🌐 Outbound Upload Bursts ({count} recorded): Review if traffic was expected.")

    lines.append("\n📋 EXECUTIVE SUMMARY:")
    for kind, list_events in sorted(by_kind.items()):
        warn_count = sum(1 for e in list_events if e["severity"] == "warning")
        sev_tag = f" ({warn_count} warning)" if warn_count else ""
        lines.append(f"  • {kind:<20}: {len(list_events)} event(s){sev_tag}")

    if warnings:
        lines.append("\n⚠️ IDENTIFIED SYSTEMIC PATTERNS & BOTTLENECKS:")
        for w in warnings:
            lines.append(f"  ➜ {w}")

    lines.append("\n💡 STRATEGIC RECOMMENDATIONS:")
    if dgpu_freeze_count > 0:
        lines.append(
            "  1. Assign utility apps (Taskmgr, PredatorSense, browsers) to 'Power Saving' (Integrated Graphics) "
            "in Windows Settings to prevent spurious dGPU wake-ups."
        )
    if "low_disk_space" in by_kind:
        lines.append("  2. Free up 20–30 GB on C: so the dynamic pagefile can expand without hitting commit exhaustion.")
    lines.append("  3. Run `python monitor.py explain <id>` to inspect any individual incident in detail.")

    lines.append("=" * 72)
    return "\n".join(lines)
