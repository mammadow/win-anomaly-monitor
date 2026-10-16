"""
Events: what the monitor found, persisted to SQLite and mirrored to JSONL.

An event is an *episode* with a start and an end (v2 wrote a separate
snapshot every 30 s for as long as a condition lasted). Each event is written to
the database when it opens and updated as it develops. When it closes, one complete
line is appended to events.jsonl. If the same problem comes back shortly after,
the event is reopened rather than duplicated; it is then written again (same id)
when it closes, and the later line supersedes the earlier one.
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from .config import SEVERITIES
from .storage import Storage


@dataclass
class Event:
    kind: str
    severity: str
    title: str
    ts_start: float
    ts_end: float
    details: dict = field(default_factory=dict)
    id: int | None = None
    is_open: bool = True

    @property
    def duration(self) -> float:
        return max(0.0, self.ts_end - self.ts_start)


def fmt_duration(sec: float) -> str:
    sec = int(round(sec))
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m {sec % 60:02d}s"
    return f"{sec // 3600}h {(sec % 3600) // 60:02d}m"


def severity_rank(s: str) -> int:
    return SEVERITIES.index(s) if s in SEVERITIES else 0


class Notifier:
    """Windows toast notifications via PowerShell's WinRT bridge (no extra packages)."""

    MIN_GAP_SEC = 120
    _APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"

    def __init__(self):
        self._last = 0.0

    def send(self, title: str, body: str) -> bool:
        if sys.platform != "win32" or time.monotonic() - self._last < self.MIN_GAP_SEC:
            return False
        self._last = time.monotonic()
        xml = (f"<toast><visual><binding template='ToastGeneric'>"
               f"<text>{escape(title[:120])}</text><text>{escape(body[:250])}</text>"
               f"</binding></visual></toast>").replace("'", "''")
        script = (
            "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null;"
            "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime] | Out-Null;"
            f"$x = New-Object Windows.Data.Xml.Dom.XmlDocument; $x.LoadXml('{xml}');"
            f"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{self._APP_ID}')"
            ".Show([Windows.UI.Notifications.ToastNotification]::new($x))"
        )
        try:
            subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
                 "-EncodedCommand", base64.b64encode(script.encode("utf-16-le")).decode()],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return True
        except OSError as e:
            logging.warning("Toast notification failed: %s", e)
            return False


class EventManager:
    def __init__(self, storage: Storage, jsonl_path: Path,
                 notifier: Notifier | None = None, notify_min_severity: str = "warning"):
        self.storage = storage
        self.jsonl_path = jsonl_path
        self.notifier = notifier
        self.notify_min = severity_rank(notify_min_severity)
        self.open_events: dict[int, Event] = {}

    def open(self, kind: str, severity: str, title: str, ts: float, details: dict) -> Event:
        ev = Event(kind, severity, title, ts, ts, details)
        ev.id = self.storage.insert_event(kind, severity, ts, title, details)
        self.open_events[ev.id] = ev
        self._log(ev, "started")
        self._maybe_notify(ev, None)
        return ev

    def update(self, ev: Event, ts: float, *, title: str | None = None,
               severity: str | None = None, details: dict | None = None) -> None:
        prev_sev = ev.severity
        self._apply(ev, ts, title, severity, details)
        self.storage.update_event(ev.id, severity=ev.severity, ts_end=ev.ts_end, is_open=True,
                                  title=ev.title, details=ev.details)
        if severity_rank(ev.severity) > severity_rank(prev_sev):
            self._log(ev, f"escalated to {ev.severity}")
            self._maybe_notify(ev, prev_sev)

    def close(self, ev: Event, ts: float, *, title: str | None = None,
              severity: str | None = None, details: dict | None = None) -> None:
        if not ev.is_open:
            return
        self._apply(ev, ts, title, severity, details)
        ev.is_open = False
        self.open_events.pop(ev.id, None)
        self.storage.update_event(ev.id, severity=ev.severity, ts_end=ev.ts_end, is_open=False,
                                  title=ev.title, details=ev.details)
        self._append_jsonl(ev)
        if ev.duration > 0:
            logging.info("■ [%s #%d] ended after %s", ev.kind, ev.id, fmt_duration(ev.duration))

    def reopen(self, ev: Event, ts: float) -> None:
        """Continue a recently closed event (the same problem came back)."""
        ev.is_open = True
        ev.ts_end = max(ev.ts_end, ts)
        ev.details["recurrences"] = ev.details.get("recurrences", 0) + 1
        ev.details.pop("ended", None)
        self.open_events[ev.id] = ev
        self.storage.update_event(ev.id, severity=ev.severity, ts_end=ev.ts_end, is_open=True,
                                  title=ev.title, details=ev.details)
        logging.info("▶ [%s #%d] recurred (×%d): %s", ev.kind, ev.id, ev.details["recurrences"], ev.title)

    def point(self, kind: str, severity: str, title: str, ts: float, details: dict) -> Event:
        """An instantaneous finding (e.g. a leak verdict): opened and closed at once."""
        ev = Event(kind, severity, title, ts, ts, details)
        ev.id = self.storage.insert_event(kind, severity, ts, title, details)
        ev.is_open = False
        self.storage.update_event(ev.id, severity=severity, ts_end=ts, is_open=False,
                                  title=title, details=details)
        self._append_jsonl(ev)
        self._log(ev, "")
        self._maybe_notify(ev, None)
        return ev

    def recent(self, kinds: list[str], days: float) -> list[dict]:
        return self.storage.recent_events(kinds, time.time() - days * 86400)

    def close_all(self, ts: float, note: str) -> None:
        for ev in list(self.open_events.values()):
            ev.details["closed_because"] = note
            self.close(ev, max(ev.ts_end, min(ts, ev.ts_end + 60)))

    # ── internals ─────────────────────────────────────────────────────

    @staticmethod
    def _apply(ev, ts, title, severity, details):
        ev.ts_end = max(ev.ts_end, ts)
        if title:
            ev.title = title
        if severity and severity_rank(severity) > severity_rank(ev.severity):
            ev.severity = severity  # severity only ever escalates within an event
        if details:
            ev.details.update(details)

    def _log(self, ev: Event, what: str) -> None:
        level = logging.WARNING if severity_rank(ev.severity) >= 1 else logging.INFO
        suffix = f" ({what})" if what else ""
        logging.log(level, "▶ [%s #%d %s] %s%s", ev.kind, ev.id, ev.severity.upper(), ev.title, suffix)

    def _maybe_notify(self, ev: Event, prev_severity: str | None) -> None:
        if self.notifier is None or severity_rank(ev.severity) < self.notify_min:
            return
        if prev_severity is not None and severity_rank(prev_severity) >= self.notify_min:
            return  # already notified at a qualifying severity
        self.notifier.send(f"Resource Monitor: {ev.kind.replace('_', ' ')}", ev.title)

    def _append_jsonl(self, ev: Event) -> None:
        rec = {
            "id": ev.id,
            "kind": ev.kind,
            "severity": ev.severity,
            "start": datetime.fromtimestamp(ev.ts_start).isoformat(timespec="seconds"),
            "end": datetime.fromtimestamp(ev.ts_end).isoformat(timespec="seconds"),
            "duration_sec": round(ev.duration),
            "title": ev.title,
            "details": ev.details,
        }
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            logging.error("Could not write %s: %s", self.jsonl_path, e)
