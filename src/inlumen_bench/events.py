"""Append-only event log per episode, plus an in-process bus for live progress.

The same events carry telemetry (requests, usage, cost) and progress (phases,
assets, repairs, budgets), so the CLI view, the dashboard and the analysis all
read one source.

Event types:
  episode_started, episode_finished
  phase_started, phase_finished        phase: export|design|codegen|validate|evaluate
  stage_started                        design-internal stage name
  asset_started, asset_finished        index, total, name
  repair_started                       attempt, max_attempts, reason
  request_started, request_progress, request_finished, request_failed
  budget_update                        cost_usd, elapsed_s, requests and their limits
  stall_warning                        seconds without bytes on a streaming request
  log                                  free-form message
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

Listener = Callable[[dict], None]


class Bus:
    def __init__(self) -> None:
        self._listeners: list[Listener] = []
        self._lock = threading.Lock()

    def subscribe(self, listener: Listener) -> None:
        with self._lock:
            self._listeners.append(listener)

    def unsubscribe(self, listener: Listener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def publish(self, event: dict) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:  # A broken view must never stop an episode.
                pass


BUS = Bus()


class EventLog:
    def __init__(self, path: Path | None, episode_id: str | None = None, bus: Bus = BUS) -> None:
        self.path = path
        self.episode_id = episode_id
        self.bus = bus
        self._lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, type: str, **fields) -> dict:
        event = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "mono": round(time.monotonic(), 4),
            "episode_id": self.episode_id,
            "type": type,
            **fields,
        }
        if self.path:
            line = json.dumps(event, default=str)
            with self._lock, self.path.open("a") as handle:
                handle.write(line + "\n")
        self.bus.publish(event)
        return event


def read_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A crash can leave a half-written last line; skip it.
            continue
    return events
