"""Background work for the dashboard (fetch, setup, experiment runs) and live snapshots.

Experiment state lives in the run folders, so snapshots are rebuilt from files and
experiments started from the CLI show up as well.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from pathlib import Path

from .config import bench_paths, resolve_data_dir, restore_owner
from .events import Bus, read_events
from .progress import ProgressTracker
from .runner import Experiment, episode_state


class JobBusy(Exception):
    pass


class Jobs:
    """Threads with a short log each; one running job per label."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.stops: dict[str, threading.Event] = {}
        self.lock = threading.Lock()

    def start(self, kind: str, label: str, fn, *args) -> dict:
        with self.lock:
            if self.running_for(label):
                raise JobBusy(f"{label} is already running")
            job = {"id": uuid.uuid4().hex[:10], "kind": kind, "label": label, "status": "running",
                   "log": [], "started": time.time(), "finished": None}
            self.jobs[job["id"]] = job
            self.stops[job["id"]] = threading.Event()

        def log(message: str) -> None:
            job["log"] = (job["log"] + [f"{time.strftime('%H:%M:%S')} {message}"])[-200:]

        def target():
            try:
                fn(log, self.stops[job["id"]], *args)
                job["status"] = "stopped" if self.stops[job["id"]].is_set() else "done"
            except Exception as exc:
                log(f"error: {exc}")
                job["log"].append(traceback.format_exc()[-2000:])
                job["status"] = "failed"
            restore_owner(*bench_paths(resolve_data_dir().path))
            job["finished"] = time.time()

        threading.Thread(target=target, daemon=True, name=f"job-{kind}").start()
        return job

    def running_for(self, label: str) -> dict | None:
        return next((j for j in self.jobs.values() if j["label"] == label and j["status"] == "running"), None)

    def last_for(self, label: str) -> dict | None:
        mine = [j for j in self.jobs.values() if j["label"] == label]
        return max(mine, key=lambda j: j["started"]) if mine else None

    def stop(self, label: str) -> bool:
        job = self.running_for(label)
        if job:
            self.stops[job["id"]].set()
        return bool(job)


def snapshot(root: Path) -> dict:
    """Counts per status and the live state of every running episode."""
    exp = Experiment.load(root)
    tracker = ProgressTracker(0, exp.name, bus=Bus())
    counts: dict[str, int] = {}
    for ep in exp.episodes():
        ep_dir = root / "episodes" / ep["id"]
        state = episode_state(ep_dir) or {"status": "pending"}
        counts[state["status"]] = counts.get(state["status"], 0) + 1
        if state["status"] == "running":
            for event in read_events(ep_dir / "events.jsonl"):
                tracker.on_event(event)
    now = time.monotonic()
    active = [{"episode": v.episode_id, "phase": v.phase, "step": v.detail, "request": v.request,
               "budget": v.budget, "elapsed_s": round(now - v.started, 1)} for v in tracker.active.values()]
    return {"total": len(exp.episodes()), "counts": counts, "active": active}
