"""Live progress views fed by the event bus.

Interactive terminals get nested bars: the experiment on top and one row per
running episode. Anything else (nohup, CI, a pipe) gets plain log lines at phase
changes plus a summary every 30 seconds, so log files stay readable.
"""

from __future__ import annotations

import statistics
import sys
import threading
import time
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from .events import BUS, Bus

SUMMARY_INTERVAL_S = 30
ETA_MIN_SAMPLES = 3


@dataclass
class EpisodeView:
    episode_id: str
    group: str  # method + model, for ETA estimates
    phase: str = "queued"
    detail: str = ""
    request: str = ""
    budget: str = ""
    started: float = field(default_factory=time.monotonic)


class ProgressTracker:
    """Keeps the state both views render; also usable by the dashboard later."""

    def __init__(self, total: int, title: str, bus: Bus = BUS) -> None:
        self.total = total
        self.title = title
        self.done = 0
        self.failed = 0
        self.active: dict[str, EpisodeView] = {}
        self.durations: dict[str, list[float]] = {}
        self.lock = threading.Lock()
        self.bus = bus
        self.changes: list[str] = []
        bus.subscribe(self.on_event)

    def close(self) -> None:
        self.bus.unsubscribe(self.on_event)

    def eta_s(self) -> float | None:
        samples = [d for ds in self.durations.values() for d in ds]
        if len(samples) < ETA_MIN_SAMPLES:
            return None
        remaining = self.total - self.done - self.failed
        workers = max(1, len(self.active))
        return statistics.median(samples) * remaining / workers

    def on_event(self, e: dict) -> None:
        eid = e.get("episode_id")
        kind = e["type"]
        with self.lock:
            if kind == "episode_started":
                view = EpisodeView(eid, e.get("group", ""))
                # Monotonic time is system-wide, so replayed events keep their real start.
                view.started = e.get("mono", view.started)
                self.active[eid] = view
                self.changes.append(f"start  {eid}")
                return
            view = self.active.get(eid)
            if kind == "episode_finished":
                if view:
                    self.durations.setdefault(view.group, []).append(time.monotonic() - view.started)
                    del self.active[eid]
                if e.get("status") in ("generated", "evaluated", "designed"):
                    self.done += 1
                else:
                    self.failed += 1
                self.changes.append(f"finish {eid} → {e.get('status')}" + (f" ({e.get('error')})" if e.get("error") else ""))
                return
            if not view:
                return
            if kind == "phase_started":
                view.phase, view.detail, view.request = e["phase"], "", ""
                self.changes.append(f"phase  {eid} → {e['phase']}")
            elif kind == "stage_started":
                view.detail = e.get("stage", "")
            elif kind == "asset_started":
                view.detail = f"asset {e['index']}/{e['total']} {e.get('name', '')}"
            elif kind == "repair_started":
                view.detail = f"repair {e['attempt']}/{e['max_attempts']} ({e.get('reason', '')[:40]})"
            elif kind == "request_started":
                view.request = "waiting for first token"
            elif kind == "request_progress":
                view.request = f"↓ ≈{e['approx_tokens']:,} tok · {e['elapsed_s']:.0f}s"
            elif kind == "request_finished":
                out = (e.get("usage") or {}).get("output_tokens")
                view.request = f"done {out:,} tok" if out else "done"
            elif kind == "stall_warning":
                view.request = f"⚠ stalled {e['seconds']:.0f}s"
            elif kind == "budget_update":
                view.budget = _budget_text(e)
            elif kind == "validation_progress":
                view.detail = e.get("message", "")


def _budget_text(b: dict) -> str:
    cost = "?" if b.get("cost_incomplete") and not b.get("cost_usd") else f"${b.get('cost_usd', 0):.3f}"
    if b.get("max_cost_usd"):
        cost += f"/${b['max_cost_usd']:.2f}"
    reqs = f"{b.get('requests', 0)}" + (f"/{b['max_requests']}" if b.get("max_requests") else "")
    return f"{cost} · {reqs} req"


def _fmt_s(seconds: float | None) -> str:
    if seconds is None:
        return "estimating…"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


class RichView:
    def __init__(self, tracker: ProgressTracker, console: Console) -> None:
        self.tracker = tracker
        self.console = console
        self.bar = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[extra]}"),
            TimeElapsedColumn(),
            console=console,
        )
        self.task = self.bar.add_task(tracker.title, total=tracker.total, extra="")

    def render(self):
        t = self.tracker
        with t.lock:
            self.bar.update(
                self.task,
                completed=t.done + t.failed,
                extra=f"✗ {t.failed} · ETA {_fmt_s(t.eta_s())}",
            )
            table = Table(box=None, pad_edge=False, show_header=bool(t.active))
            for col in ("episode", "phase", "step", "request", "budget", "time"):
                table.add_column(col, no_wrap=True, overflow="ellipsis", max_width=46)
            now = time.monotonic()
            for v in sorted(t.active.values(), key=lambda v: v.started):
                table.add_row(v.episode_id, v.phase, v.detail, v.request, v.budget, _fmt_s(now - v.started))
        return Group(self.bar, table)


class PlainView:
    def __init__(self, tracker: ProgressTracker, console: Console) -> None:
        self.tracker = tracker
        self.console = console
        self.last_summary = time.monotonic()

    def tick(self) -> None:
        t = self.tracker
        with t.lock:
            changes, t.changes = t.changes, []
        for line in changes:
            self.console.print(f"{time.strftime('%H:%M:%S')} {line}", highlight=False)
        if time.monotonic() - self.last_summary >= SUMMARY_INTERVAL_S:
            self.last_summary = time.monotonic()
            with t.lock:
                self.console.print(
                    f"{time.strftime('%H:%M:%S')} progress {t.done + t.failed}/{t.total} "
                    f"(failed {t.failed}, running {len(t.active)}, ETA {_fmt_s(t.eta_s())})",
                    highlight=False,
                )


class ProgressDisplay:
    """Context manager that renders until the experiment finishes."""

    def __init__(self, total: int, title: str, *, plain: bool | None = None) -> None:
        self.console = Console(stderr=True)
        self.tracker = ProgressTracker(total, title)
        self.plain = (not sys.stderr.isatty()) if plain is None else plain
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._live: Live | None = None

    def __enter__(self) -> "ProgressDisplay":
        if self.plain:
            view = PlainView(self.tracker, self.console)

            def loop():
                while not self._stop.wait(1):
                    view.tick()
                view.tick()

            self._thread = threading.Thread(target=loop, daemon=True)
            self._thread.start()
        else:
            view = RichView(self.tracker, self.console)
            self._live = Live(view.render(), console=self.console, refresh_per_second=4, transient=False)
            self._live.__enter__()

            def loop():
                while not self._stop.wait(0.25):
                    self._live.update(view.render())
                self._live.update(view.render())

            self._thread = threading.Thread(target=loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._live:
            self._live.__exit__(*exc)
        self.tracker.close()
