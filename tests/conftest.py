import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resmon.config import MonitorConfig  # noqa: E402
from resmon.events import Event  # noqa: E402

NAN = math.nan


@pytest.fixture
def cfg(tmp_path):
    c = MonitorConfig(data_dir=str(tmp_path), efficiency_mode=False)
    c.validate()
    return c


def make_sample(ts: float, **kw) -> dict:
    """A quiet, idle-laptop sample; override any field."""
    s = {
        "ts": ts, "cpu_pct": 2.0, "core_max_pct": 15.0, "ram_pct": 40.0, "ram_used_gb": 12.5,
        "commit_pct": 50.0, "disk_read_bps": 0.0, "disk_write_bps": 60_000.0,
        "net_sent_bps": 1_500.0, "net_recv_bps": 2_000.0, "on_ac": 1.0, "battery_pct": 80.0,
        "user_idle_sec": 5.0, "gpu_awake": 0.0, "gpu_util_pct": 0.0, "gpu_vram_pct": 0.0,
        "gpu_temp_c": NAN, "gpu_power_w": 0.0, "gpu_clock_mhz": 0.0, "gpu_throttle": 0,
        "handles": 120_000, "processes": 350, "threads": 5_000,
    }
    s.update(kw)
    return s


class RecordingEvents:
    def __init__(self):
        self.events: list[Event] = []
        self.open_events = {}

    def open(self, kind, severity, title, ts, details):
        ev = Event(kind, severity, title, ts, ts, dict(details), id=len(self.events) + 1)
        self.events.append(ev)
        self.open_events[ev.id] = ev
        return ev

    def update(self, ev, ts, *, title=None, severity=None, details=None):
        ev.ts_end = max(ev.ts_end, ts)
        if title:
            ev.title = title
        if severity:
            ev.severity = severity
        if details:
            ev.details.update(details)

    def reopen(self, ev, ts):
        ev.is_open = True
        ev.details["recurrences"] = ev.details.get("recurrences", 0) + 1
        self.open_events[ev.id] = ev
        self.update(ev, ts)

    def close(self, ev, ts, *, title=None, severity=None, details=None):
        self.update(ev, ts, title=title, severity=severity, details=details)
        ev.is_open = False
        self.open_events.pop(ev.id, None)

    def point(self, kind, severity, title, ts, details):
        ev = self.open(kind, severity, title, ts, details)
        self.close(ev, ts)
        return ev

    def recent(self, kinds, days):
        return []

    def of_kind(self, kind):
        return [e for e in self.events if e.kind == kind]


class FakeProcs:
    foreground_name = None

    def top_memory(self, n=5):
        return [{"name": "big.exe", "private_mb": 9000, "processes": 1}]

    def top_activity(self, since, n=5):
        return {"cpu": [], "io": []}

    def name_of(self, pid):
        return f"pid{pid}"


class FakeMon:
    """The slice of Monitor that detectors use."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.events = RecordingEvents()
        self.procs = FakeProcs()
        self.gpu = None
        self.model = None
        self.user_idle = False
        self.act = {"top_cpu": [], "top_io": [], "top_memory": [], "gpu_clients": [],
                    "foreground": None, "user_idle": False}

    def gpu_clients(self):
        return []

    def activity(self, since):
        return dict(self.act)


@pytest.fixture
def mon(cfg):
    return FakeMon(cfg)
