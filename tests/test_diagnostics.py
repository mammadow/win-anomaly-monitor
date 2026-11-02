import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from conftest import make_sample

from resmon.config import MonitorConfig
from resmon.diagnostics import (
    audit_system_health,
    build_diagnostic_prompt,
    collect_event_forensics,
    diagnose_offline,
    explain_event,
    parse_time_or_query_targets,
    query_ollama,
    query_openai,
)
from resmon.storage import Storage


@pytest.fixture
def test_db(tmp_path):
    db_path = tmp_path / "metrics.db"
    s = Storage(db_path)
    now = time.time()

    # Add samples around now
    for i in range(20):
        t = now - 10 + i
        sample = make_sample(
            t,
            cpu_pct=10.0 + i,
            core_max_pct=20.0 + i * 2,
            ram_pct=60.0,
            commit_pct=70.0,
            disk_read_bps=1000000.0,
            disk_write_bps=2000000.0,
            net_sent_bps=50000.0,
            net_recv_bps=80000.0,
            gpu_awake=1.0 if i >= 10 else 0.0,
        )
        s.add_sample(sample)
    s.flush()

    # Insert an event
    eid = s.insert_event(
        "ui_freeze",
        "info",
        now,
        "TestApp.exe stopped responding for 1.5s",
        {
            "process": "TestApp.exe",
            "scope": "app",
            "app_cpu_sec": 1.2,
            "dgpu_transitions": [{"at_sec": 0.5, "dgpu": "woke up"}],
            "around": {"busiest_core_pct": 85, "monitor_loop_gap_sec": 1.0},
            "top_cpu": [{"name": "TestApp.exe", "avg_cores": 1.0}],
        },
    )
    s.update_event(
        eid,
        severity="info",
        ts_end=now + 1.5,
        is_open=False,
        title="TestApp.exe stopped responding for 1.5s",
        details={
            "process": "TestApp.exe",
            "scope": "app",
            "app_cpu_sec": 1.2,
            "dgpu_transitions": [{"at_sec": 0.5, "dgpu": "woke up"}],
            "around": {"busiest_core_pct": 85, "monitor_loop_gap_sec": 1.0},
            "top_cpu": [{"name": "TestApp.exe", "avg_cores": 1.0}],
        },
    )
    s.close()
    return db_path, eid, now


def test_collect_event_forensics(test_db):
    db_path, eid, now = test_db
    event = {
        "id": eid,
        "kind": "ui_freeze",
        "severity": "info",
        "title": "Test freeze",
        "ts_start": now,
        "ts_end": now + 1.5,
        "details": {"process": "TestApp.exe"},
    }
    forensics = collect_event_forensics(db_path, event, window_padding_sec=5.0)
    assert forensics["event"]["id"] == eid
    assert forensics["telemetry"]["sample_count"] > 0
    assert forensics["telemetry"]["peak_cpu_pct"] is not None
    assert any("woke up" in t for t in forensics["telemetry"]["gpu_transitions"])


def test_diagnose_offline_ui_freeze_dgpu(test_db):
    db_path, eid, now = test_db
    event = {
        "id": eid,
        "kind": "ui_freeze",
        "severity": "info",
        "title": "TestApp.exe stopped responding",
        "ts_start": now,
        "ts_end": now + 1.5,
        "details": {
            "process": "TestApp.exe",
            "dgpu_transitions": [{"at_sec": 0.5, "dgpu": "woke up"}],
            "around": {"busiest_core_pct": 70, "monitor_loop_gap_sec": 1.0},
            "app_cpu_sec": 0.2,
        },
    }
    forensics = collect_event_forensics(db_path, event)
    report = diagnose_offline(forensics)
    assert "NVIDIA dGPU Wake-up Latency" in report
    assert "PCIe D3cold" in report
    assert "Power saving" in report


def test_diagnose_offline_scheduler_stall(test_db):
    db_path, eid, now = test_db
    event = {
        "id": eid,
        "kind": "ui_freeze",
        "severity": "info",
        "title": "WhatsApp.exe stopped responding",
        "ts_start": now,
        "ts_end": now + 1.0,
        "details": {
            "process": "WhatsApp.exe",
            "dgpu_transitions": [],
            "around": {"busiest_core_pct": 100, "monitor_loop_gap_sec": 3.5},
            "top_cpu": [{"name": "svchost.exe", "avg_cores": 4.5}],
            "app_cpu_sec": 0.1,
        },
    }
    forensics = collect_event_forensics(db_path, event)
    report = diagnose_offline(forensics)
    assert "CPU Scheduler Starvation" in report
    assert "svchost.exe" in report


def test_diagnose_offline_upload_burst(test_db):
    db_path, eid, now = test_db
    event = {
        "id": eid,
        "kind": "upload_burst",
        "severity": "warning",
        "title": "Uploaded 80 MB",
        "ts_start": now,
        "ts_end": now + 60,
        "details": {
            "sent_mb": 80.0,
            "top_net": [{"name": "python.exe", "total_mb": 50.0}],
            "foreground": "Code.exe",
        },
    }
    forensics = collect_event_forensics(db_path, event)
    report = diagnose_offline(forensics)
    assert "Background Outbound Data Burst" in report
    assert "python.exe" in report


def test_diagnose_offline_leak_and_disk():
    now = time.time()
    event_leak = {
        "id": 10,
        "kind": "memory_leak",
        "severity": "warning",
        "title": "dwm.exe memory growing steadily",
        "ts_start": now,
        "ts_end": now,
        "details": {"process": "dwm.exe", "growth_mb": 1080},
    }
    forensics_leak = {"event": event_leak, "details": event_leak["details"], "telemetry": {}}
    report_leak = diagnose_offline(forensics_leak)
    assert "Desktop Window Manager (DWM) leak" in report_leak
    assert "Intel Graphics and NVIDIA drivers" in report_leak

    event_disk = {
        "id": 11,
        "kind": "low_disk_space",
        "severity": "warning",
        "title": "System drive C: nearly full",
        "ts_start": now,
        "ts_end": now,
        "details": {},
    }
    forensics_disk = {"event": event_disk, "details": event_disk["details"], "telemetry": {}}
    report_disk = diagnose_offline(forensics_disk)
    assert "System Drive (C:) Space Exhaustion" in report_disk
    assert "Commit Limit" in report_disk


def test_build_prompt(test_db):
    db_path, eid, now = test_db
    event = {"id": eid, "kind": "ui_freeze", "ts_start": now, "ts_end": now + 1.0, "details": {}}
    forensics = collect_event_forensics(db_path, event)
    prompt = build_diagnostic_prompt(forensics, user_query="Why did this stall?")
    assert "Principal Windows Systems Performance Engineer" in prompt
    assert "NVIDIA Advanced Optimus" in prompt
    assert "Why did this stall?" in prompt


def test_parse_time_or_query_targets():
    events = [
        {"id": 100, "kind": "background_cpu", "title": "svchost busy", "details": {"process": "svchost.exe"}},
        {"id": 101, "kind": "ui_freeze", "title": "WhatsApp froze", "details": {"process": "WhatsApp.Root.exe"}},
        {"id": 102, "kind": "ui_freeze", "title": "Taskmgr froze", "details": {"process": "Taskmgr.exe"}},
    ]

    # Match by ID
    e, reason = parse_time_or_query_targets("Explain #101", events)
    assert e["id"] == 101

    # Match by process name
    e, reason = parse_time_or_query_targets("Why did whatsapp stop responding?", events)
    assert e["id"] == 101

    # Match by freeze keyword
    e, reason = parse_time_or_query_targets("Why did my laptop stutter?", events)
    assert e["id"] == 102  # Latest freeze


def test_audit_system_health(tmp_path):
    cfg = MonitorConfig(data_dir=str(tmp_path))
    db_path = tmp_path / "metrics.db"
    s = Storage(db_path)
    now = time.time()

    s.insert_event("low_disk_space", "warning", now, "System drive C: nearly full: 54 GB free", {})
    s.insert_event("ui_freeze", "info", now, "Taskmgr freeze", {
        "process": "Taskmgr.exe",
        "dgpu_transitions": [{"dgpu": "woke up"}],
    })
    s.close()

    audit = audit_system_health(cfg, hours=24)
    assert "SYSTEM HEALTH AUDIT" in audit
    assert "Optimus D3cold" in audit
    assert "Taskmgr.exe" in audit
    assert "Low Free Space Hazard" in audit


def test_explain_event_offline(test_db, tmp_path):
    db_path, eid, now = test_db
    cfg = MonitorConfig(data_dir=str(db_path.parent), ai_provider="offline")
    event = {
        "id": eid,
        "kind": "ui_freeze",
        "severity": "info",
        "title": "TestApp.exe freeze",
        "ts_start": now,
        "ts_end": now + 1.0,
        "details": {"process": "TestApp.exe"},
    }
    result = explain_event(cfg, event, force_offline=True)
    assert "FORENSIC DIAGNOSTIC REPORT" in result
    assert "Event #" in result
