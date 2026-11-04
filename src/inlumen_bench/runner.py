"""Experiments: the case × method × model × track × repetition matrix.

Layout of `runs/<experiment>/`:
  experiment.json
  episodes/<episode-id>/
    episode.json        status, counters, budget, usage totals, checkpoint
    events.jsonl        progress and telemetry events
    workspace/          ServiceCraft allowlist export (what the method may see)
    contract.md         the runtime contract the method received
    submission/         the generated Dagster bundle
    work/               raw responses and method scratch files
  evaluation/batch-NNN/ ServiceCraft batch manifest and results
"""

from __future__ import annotations

import json
import random
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from . import design_score
from .budget import Budget, BudgetExceeded, Limits
from .contract import CONTRACT_VERSION, render
from .corpus import TRACKS, Corpus
from .events import EventLog, read_events
from .llm import LLMClient, LLMError
from .methods import EpisodeContext, get_method
from .providers import ModelRef, load_profiles

FINAL_STATUSES = {"generated", "generation_failed", "evaluated"}
TRACK_SHORT = {"natural_language": "nl", "fixed_graph": "fg"}


@dataclass
class Experiment:
    name: str
    cases: list[str]
    methods: list[str]
    models: list[str]
    tracks: list[str]
    repetitions: int
    seed: int = 0
    limits: dict = field(default_factory=dict)
    method_options: dict = field(default_factory=dict)  # method name → options
    dataset: dict = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION
    bench_version: str = __version__
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @classmethod
    def load(cls, root: Path) -> "Experiment":
        return cls(**json.loads((root / "experiment.json").read_text()))

    def save(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "experiment.json").write_text(json.dumps(asdict(self), indent=2) + "\n")

    def episodes(self) -> list[dict]:
        out = []
        for case in self.cases:
            for method in self.methods:
                for model in self.models:
                    for track in self.tracks:
                        for rep in range(self.repetitions):
                            ref = ModelRef.parse(model)
                            out.append({
                                "id": f"{case[:3]}-{method}-{ref.slug}-{TRACK_SHORT[track]}-r{rep}",
                                "case_id": case,
                                "method": method,
                                "model": model,
                                "track": track,
                                "repetition": rep,
                            })
        # Random order spreads order effects (cache warmth, rate limits) over cells.
        random.Random(self.seed).shuffle(out)
        return out


def create_experiment(root: Path, corpus: Corpus, **kwargs) -> Experiment:
    if (root / "experiment.json").exists():
        raise FileExistsError(f"{root} already has an experiment; use `bench run --resume`")
    for track in kwargs["tracks"]:
        if track not in TRACKS:
            raise ValueError(f"Unknown track {track!r}")
    profiles = load_profiles()
    for model in kwargs["models"]:
        ref = ModelRef.parse(model)
        if ref.profile not in profiles:
            raise ValueError(f"Unknown endpoint profile {ref.profile!r} (see `bench providers list`)")
    for method in kwargs["methods"]:
        get_method(method)
    exp = Experiment(dataset={"name": "servicecraft", **corpus.metadata()}, **kwargs)
    exp.save(root)
    return exp


def episode_state(ep_dir: Path) -> dict | None:
    path = ep_dir / "episode.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _write_state(ep_dir: Path, state: dict) -> None:
    tmp = ep_dir / "episode.json.tmp"
    tmp.write_text(json.dumps(state, indent=2, default=str) + "\n")
    tmp.replace(ep_dir / "episode.json")


def pending_episodes(root: Path, exp: Experiment) -> list[dict]:
    pending = []
    for ep in exp.episodes():
        state = episode_state(root / "episodes" / ep["id"])
        if not state or state.get("status") not in FINAL_STATUSES:
            pending.append(ep)
    return pending


def run_experiment(root: Path, corpus: Corpus, workers: int = 2, stop: threading.Event | None = None) -> None:
    exp = Experiment.load(root)
    todo = pending_episodes(root, exp)
    profiles = load_profiles()
    limits = Limits(**exp.limits)

    def task(ep: dict) -> None:
        if stop and stop.is_set():
            return
        run_episode(root, corpus, exp, ep, profiles, limits)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(task, todo))


def run_episode(root: Path, corpus: Corpus, exp: Experiment, ep: dict, profiles, limits: Limits) -> dict:
    ep_dir = root / "episodes" / ep["id"]
    if ep_dir.exists():
        # An interrupted earlier attempt is kept for audit, never reused.
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        ep_dir.rename(ep_dir.with_name(f"{ep['id']}.interrupted-{stamp}"))
    ep_dir.mkdir(parents=True)
    events = EventLog(ep_dir / "events.jsonl", ep["id"])
    budget = Budget(limits)
    method = get_method(ep["method"])
    ref = ModelRef.parse(ep["model"])
    state = {
        **ep,
        "status": "running",
        "method_version": method.version,
        "dataset_commit": exp.dataset.get("commit"),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_state(ep_dir, state)
    events.emit("episode_started", group=f"{ep['method']}·{ep['model']}", **ep)

    phase = "export"
    try:
        events.emit("phase_started", phase=phase)
        workspace = corpus.export(ep["case_id"], ep_dir / "workspace", ep["track"])
        contract_text, contract_sha = render(corpus, ep["case_id"], ep["track"])
        (ep_dir / "contract.md").write_text(contract_text)
        state["contract_sha256"] = contract_sha
        events.emit("phase_finished", phase=phase)

        phase = "generate"
        events.emit("phase_started", phase=phase)
        profile = profiles[ref.profile]
        ctx = EpisodeContext(
            case_id=ep["case_id"],
            track=ep["track"],
            workspace=workspace,
            contract=contract_text,
            submission_dir=ep_dir / "submission",
            work_dir=ep_dir / "work",
            llm=LLMClient(profile, ref.model, events, budget),
            events=events,
            budget=budget,
            asset_names=(corpus.case_info(ep["case_id"])["logical_to_asset"]
                         if ep["track"] == "fixed_graph" else {}),
            options=exp.method_options.get(ep["method"], {}),
        )
        ctx.work_dir.mkdir()
        result = method.generate(ctx)
        events.emit("phase_finished", phase=phase)
        state.update(status=result.status, error=result.error, checkpoints=result.checkpoints,
                     counters=result.counters)
        design_file = ctx.work_dir / "design.json"
        if design_file.is_file() and ep["track"] == "natural_language":
            # Operator-side scoring against the private reference graph, after generation.
            reference = json.loads((corpus.root / "corpus/pipelines" / ep["case_id"] / "task_graph.json").read_text())
            state["design_score"] = design_score.score(json.loads(design_file.read_text()), reference)
    except BudgetExceeded as exc:
        state.update(status="generation_failed", error=f"budget: {exc}", failed_phase=phase)
    except LLMError as exc:
        state.update(status="generation_failed", error=f"model request: {exc}", failed_phase=phase)
    except Exception as exc:  # Infrastructure problems are kept apart from method failures.
        state.update(status="infra_error", error=f"{type(exc).__name__}: {exc}", failed_phase=phase)
        (ep_dir / "traceback.txt").write_text(traceback.format_exc())

    state["budget"] = budget.snapshot()
    state["usage"] = usage_totals(ep_dir / "events.jsonl")
    state["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    _write_state(ep_dir, state)
    events.emit("episode_finished", status=state["status"], error=state.get("error"))
    return state


def usage_totals(events_path: Path) -> dict:
    """Sums provider-reported usage; counts requests whose usage was unavailable."""
    totals = {"requests": 0, "failed_requests": 0, "usage_unavailable": 0, "cost_unavailable": 0,
              "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "reasoning_tokens": 0,
              "cost_usd": 0.0, "request_seconds": 0.0}
    for e in read_events(events_path):
        if e["type"] == "request_failed":
            totals["failed_requests"] += 1
            totals["request_seconds"] += e.get("latency_s") or 0
        if e["type"] != "request_finished":
            continue
        totals["requests"] += 1
        totals["request_seconds"] += e.get("latency_s") or 0
        usage = e.get("usage")
        if not usage:
            totals["usage_unavailable"] += 1
        else:
            for key in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"):
                totals[key] += usage.get(key) or 0
        if e.get("cost_usd") is None:
            totals["cost_unavailable"] += 1
        else:
            totals["cost_usd"] += e["cost_usd"]
    totals["cost_usd"] = round(totals["cost_usd"], 6)
    totals["request_seconds"] = round(totals["request_seconds"], 2)
    return totals
