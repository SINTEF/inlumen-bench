"""Background work for the dashboard (fetch, setup, experiment runs) and live snapshots.

Experiment state lives in the run folders, so snapshots are rebuilt from files and
experiments started from the CLI show up as well.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from .config import bench_paths, resolve_data_dir, restore_owner
from .events import Bus, read_events
from .progress import ProgressTracker, request_text
from .runner import Experiment, episode_state


EXPERIMENT_ENV = "INLUMEN_BENCH_EXPERIMENT"
CONTAINER_LABEL = "inlumen-bench.experiment"


class JobBusy(Exception):
    pass


RUN_LOG = "run.log"


def log_line(experiment: Path, line: str, stamp: bool = True) -> None:
    """Appends to the experiment's run log, which outlives the dashboard process."""
    if stamp:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}"
    try:
        with (experiment / RUN_LOG).open("a") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def read_log(experiment: Path, lines: int = 60) -> list[str]:
    path = experiment / RUN_LOG
    if not path.is_file():
        return []
    start = max(0, path.stat().st_size - 64_000)
    with path.open("rb") as handle:  # Only the end: the log grows for days.
        handle.seek(start)
        text = handle.read().decode(errors="replace").splitlines()
    return (text[1:] if start else text)[-lines:]  # A mid-file start cuts the first line.


def run_cli(log, stop: threading.Event, experiment: Path, *args: str) -> None:
    """Runs `bench <args>` for one experiment in its own process group.

    The experiment's temporary files (and so every Docker mount of the validator
    and evaluator) live under `<experiment>/.tmp`, which lets Stop find and kill
    the run's containers. Stop kills the whole process group at once.
    """
    tmp = experiment / ".tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, EXPERIMENT_ENV: experiment.name, "TMPDIR": str(tmp), "PYTHONUNBUFFERED": "1",
           "COLUMNS": "160", "INLUMEN_BENCH_RUNS_DIR": str(experiment.parent)}
    proc = subprocess.Popen([sys.executable, "-m", "inlumen_bench.cli", *args], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)

    def pump() -> None:
        for line in proc.stdout:
            if line.strip():
                log(line.rstrip())
                log_line(experiment, line.rstrip(), stamp=False)

    reader = threading.Thread(target=pump, daemon=True)
    log_line(experiment, f"started: bench {' '.join(args)}")
    reader.start()
    while proc.poll() is None:
        if stop.wait(0.3):
            kill_run(proc, experiment, lambda m: (log(m), log_line(experiment, m)))
            break
    reader.join(timeout=2)
    log_line(experiment, "stopped from the dashboard" if stop.is_set() else f"ended (exit code {proc.returncode})")
    if not stop.is_set():
        shutil.rmtree(tmp, ignore_errors=True)
        if proc.returncode:
            raise RuntimeError(f"bench {args[0]} exited with code {proc.returncode}")


def kill_run(proc: subprocess.Popen | None, experiment: Path, log=print) -> None:
    """Kills a run right away: its processes, its Docker containers, and marks
    the episodes that were running as stopped (they rerun on Resume)."""
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)
    killed = kill_containers(experiment)
    stopped = mark_stopped(experiment)
    shutil.rmtree(experiment / ".tmp", ignore_errors=True)
    log(f"stopped: processes killed, {killed} containers removed, {stopped} episodes marked stopped")


def kill_containers(experiment: Path) -> int:
    """Removes running containers labelled with, or mounting folders of, this experiment."""
    try:
        ids = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True, timeout=20).stdout.split()
        if not ids:
            return 0
        info = json.loads(subprocess.run(["docker", "inspect", *ids], capture_output=True, text=True,
                                         timeout=30).stdout or "[]")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return 0
    root = str(experiment.resolve()).rstrip("/") + "/"
    mine = []
    for c in info:
        labels = (c.get("Config") or {}).get("Labels") or {}
        # Docker Desktop reports host paths with a /host_mnt prefix.
        sources = [m.get("Source", "").removeprefix("/host_mnt") + "/" for m in c.get("Mounts") or []]
        if labels.get(CONTAINER_LABEL) == experiment.name or any(s.startswith(root) for s in sources):
            mine.append(c["Id"])
    if mine:
        subprocess.run(["docker", "rm", "-f", *mine], capture_output=True, timeout=60)
    return len(mine)


def mark_stopped(experiment: Path) -> int:
    count = 0
    for state_file in (experiment / "episodes").glob("*/episode.json"):
        try:
            state = json.loads(state_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("status") == "running":
            state.update(status="stopped", error="stopped by the user")
            state_file.write_text(json.dumps(state, indent=2) + "\n")
            count += 1
    return count


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
    active = [{"episode": v.episode_id, "model": v.model, "phase": v.phase, "step": v.detail,
               "request": request_text(v, now), "budget": v.budget, "elapsed_s": round(now - v.started, 1)}
              for v in tracker.active.values()]
    return {"total": len(exp.episodes()), "counts": counts, "active": active}
