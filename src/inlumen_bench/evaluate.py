"""Scoring through ServiceCraft's own batch evaluator, run as a separate command.

Each call evaluates the generated episodes that have no evaluation yet, in a new
`evaluation/batch-NNN/` folder (ServiceCraft's resume rejects changed manifests).
Controls come from the locally requalified corpus, so their images exist here.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from .corpus import Corpus, CorpusError
from .events import EventLog
from .runner import Experiment, _write_state, episode_state

POLL_S = 2.0


def evaluate_experiment(root: Path, corpus: Corpus, log=print) -> Path | None:
    exp = Experiment.load(root)
    registry = corpus.registry(qualified=True)
    cases = {c["case_id"]: c for c in registry["cases"]}
    qroot = corpus.qualified_corpus

    entries, controls, waiting = [], {}, []
    for ep in exp.episodes():
        ep_dir = root / "episodes" / ep["id"]
        state = episode_state(ep_dir)
        if not state or state.get("status") != "generated":
            continue
        case = cases.get(ep["case_id"])
        if not case:
            raise CorpusError(f"{ep['case_id']} is not in the qualified corpus")
        control_id = f"control_{ep['case_id'][:3]}"
        case_dir = qroot / "cases" / ep["case_id"]
        controls[control_id] = {
            "profile": case["dependency_profile"],
            "candidate": str(qroot / case["control"]) if not Path(case["control"]).is_absolute() else case["control"],
            "inputs": str(case_dir / "inputs"),
            "spec": str(qroot / case["evaluation_spec"]),
        }
        entries.append({
            "id": ep["id"],
            "control": control_id,
            "candidate": str(ep_dir / "submission"),
            "inputs": str(case_dir / "inputs"),
            "spec": str(qroot / case["evaluation_spec"]),
            "metadata": {
                "method": ep["method"],
                "model": ep["model"],
                "track": ep["track"],
                "repetition": ep["repetition"],
                "attempt": 0,
                "episode_id": ep["id"],
                "checkpoint_sha256": (state.get("checkpoints") or [{}])[-1].get("sha256"),
            },
        })
        waiting.append(ep_dir)
    if not entries:
        log("nothing to evaluate")
        return None

    batches = root / "evaluation"
    batches.mkdir(exist_ok=True)
    batch = batches / f"batch-{len(list(batches.glob('batch-*'))) + 1:03d}"
    batch.mkdir()
    profiles = {name: prof for name, prof in registry["profiles"].items()
                if name in {c["profile"] for c in controls.values()}}
    manifest = {
        "schema_version": "servicecraft.experiment@1",
        "profiles": profiles,
        "controls": controls,
        "entries": entries,
        "infrastructure_retries": 1,
    }
    (batch / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    events = {Path(d).name: EventLog(Path(d) / "events.jsonl", Path(d).name) for d in waiting}
    for ev in events.values():
        ev.emit("phase_started", phase="evaluate")
    cmd, env, cwd = corpus.evaluator_command("batch", str(batch / "manifest.json"), "--output", str(batch / "output"))
    log(f"evaluating {len(entries)} submissions with {len(controls)} controls → {batch}")
    with (batch / "batch.log").open("w") as out:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT)
        _follow(proc, batch / "output", len(entries), log)
    if proc.returncode == 2 and not (batch / "output/results.jsonl").is_file():
        raise CorpusError(f"batch setup failed; see {batch / 'batch.log'}")

    rows = {}
    results = batch / "output/results.jsonl"
    if results.is_file():
        for line in results.read_text().splitlines():
            row = json.loads(line)
            rows[row.get("id") or row.get("entry_id")] = row
    for ep_dir in waiting:
        state = episode_state(ep_dir)
        row = rows.get(ep_dir.name)
        if row is None:
            continue
        (ep_dir / "evaluation.json").write_text(json.dumps(row, indent=2) + "\n")
        # Blocked or infrastructure results stay "generated" so the next call retries them.
        state.update(evaluation_batch=batch.name, scores=row.get("scores"),
                     stages=row.get("stages"), evaluation_status=row.get("status"),
                     evaluation_report=str(batch / "output" / row["report"]) if row.get("report") else None)
        if row.get("status") == "evaluated":
            state["status"] = "evaluated"
        _write_state(ep_dir, state)
        events[ep_dir.name].emit("phase_finished", phase="evaluate", scores=row.get("scores"))
    return batch


def _follow(proc: subprocess.Popen, output: Path, total: int, log) -> None:
    """Reports progress from ServiceCraft's incremental summary.json."""
    last = -1
    while proc.poll() is None:
        time.sleep(POLL_S)
        done = _processed(output)
        if done != last:
            last = done
            log(f"evaluated {done}/{total}")
    log(f"evaluated {_processed(output)}/{total} (exit {proc.returncode})")


def _processed(output: Path) -> int:
    summary = output / "summary.json"
    try:
        data = json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    for key in ("processed", "completed", "entries_processed"):
        if isinstance(data.get(key), int):
            return data[key]
    if isinstance(data.get("entries"), list):
        return len(data["entries"])
    return 0
