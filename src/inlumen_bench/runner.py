"""Experiments, in two steps: pipeline design, then code generation.

- A design experiment (`stage="design"`) turns each case's prompt into a task graph,
  per model and repetition. Designs are scored against the reference graph; nothing
  runs in Docker.
- A code experiment (`stage="code"`) generates Dagster code from a design: either
  ServiceCraft's reference graph (`design_from="reference"`, the fixed-graph track)
  or the successful designs of a design experiment (frozen in `designs` when the
  experiment is created). ServiceCraft scores the code.

Older experiments (no `stage`) are code experiments; a `natural_language` track there
means design and code in one episode.

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
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from . import design_score, metrics
from .budget import Budget, BudgetExceeded, Limits
from .contract import CONTRACT_VERSION, render
from .corpus import TRACKS, Corpus
from .events import EventLog, read_events
from .llm import LLMClient, LLMError
from .methods import EpisodeContext, get_method
from .providers import ModelRef, load_profiles

FINAL_STATUSES = {"generated", "generation_failed", "evaluated", "designed"}
TRACK_SHORT = {"natural_language": "nl", "fixed_graph": "fg"}
STAGES = ("design", "code")
DESIGN_METHOD = "inlumen"
REFERENCE = "reference"
# Before bench 0.2, `inlumen` was a standalone reimplementation of inLUMEN.
LEGACY_METHOD_ALIASES = {"inlumen": "inlumen-standalone"}
CONDITIONED_METHOD = "inlumen"


def make_conditions(stage: str, *, grounding: tuple[bool, ...] = (True,), contracts: tuple[bool, ...] = (True,),
                    repair: tuple[bool, ...] = (True,), repairs: int = 2) -> list[dict]:
    """The full factorial of the chosen support mechanisms, or [] for inLUMEN's defaults.

    Composition varies metamodel grounding; implementation varies explicit
    artifact contracts and the repair loop independently.
    """
    if stage == "design":
        if list(grounding) == [True]:
            return []
        return [{"name": "grounded" if g else "ungrounded", "slug": "g" if g else "ug",
                 "options": {"grounded": g}} for g in grounding]
    if list(contracts) == [True] and list(repair) == [True]:
        return []
    out = []
    for c in contracts:
        for r in repair:
            name = ("contracts" if c else "no contracts") + " · " + (f"repair ×{repairs}" if r else "no repair")
            out.append({"name": name, "slug": ("c" if c else "nc") + ("r" if r else "nr"),
                        "options": {"artifact_contracts": c, "repair_attempts": repairs if r else 0}})
    return out


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
    stage: str = "code"
    # Code experiments: "reference", or the design experiment the designs come from.
    design_from: str = ""
    # Designs frozen at creation: {"episode", "case_id", "model", "repetition"}.
    designs: list = field(default_factory=list)
    # Episode method name → method that runs it. Experiments made before `inlumen`
    # ran inLUMEN's own agents keep the reimplementation they started with.
    method_aliases: dict = field(default_factory=dict)
    # Experimental conditions for the ablation study: [{"name", "slug", "options"}].
    # Each applies to the `inlumen` method only (the baseline has no support to
    # switch off); empty means inLUMEN's defaults.
    conditions: list = field(default_factory=list)

    @property
    def design_source(self) -> str:
        """Short description of where the code's designs come from."""
        if self.stage == "design":
            return "—"
        if self.designs:
            return f"compositions from {self.design_from}"
        return "reference compositions" if self.tracks == ["fixed_graph"] else ", ".join(
            "reference compositions" if t == "fixed_graph" else "composed in the same run" for t in self.tracks)

    @classmethod
    def load(cls, root: Path) -> "Experiment":
        data = json.loads((root / "experiment.json").read_text())
        data.setdefault("method_aliases", dict(LEGACY_METHOD_ALIASES))
        return cls(**data)

    def method_for(self, episode_method: str) -> str:
        return self.method_aliases.get(episode_method, episode_method)

    def save(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "experiment.json").write_text(json.dumps(asdict(self), indent=2) + "\n")

    def conditions_for(self, method: str) -> list[dict | None]:
        """The conditions a method runs under (None: its defaults, no suffix in run ids)."""
        if self.conditions and self.method_for(method) == CONDITIONED_METHOD:
            return list(self.conditions)
        return [None]

    def episodes(self) -> list[dict]:
        out = []

        def add(base: dict, id_prefix: str, rep: int) -> None:
            for cond in self.conditions_for(base["method"]):
                suffix = f"-{cond['slug']}" if cond else ""
                extra = {"condition": cond["name"], "condition_options": cond["options"]} if cond else {}
                out.append({**base, **extra, "id": f"{id_prefix}{suffix}-r{rep}", "repetition": rep})

        if self.stage == "design":
            for case in self.cases:
                for model in self.models:
                    for rep in range(self.repetitions):
                        add({"case_id": case, "method": DESIGN_METHOD, "model": model,
                             "track": "natural_language", "design": "own"},
                            f"{case[:3]}-design-{ModelRef.parse(model).slug}", rep)
        elif self.designs:
            for d in self.designs:
                dslug = f"{ModelRef.parse(d['model']).slug}-r{d['repetition']}"
                for method in self.methods:
                    for model in self.models:
                        for rep in range(self.repetitions):
                            add({"case_id": d["case_id"], "method": method, "model": model,
                                 "track": "natural_language", "design": d["model"],
                                 "design_episode": d["episode"]},
                                f"{d['case_id'][:3]}-{method}-{ModelRef.parse(model).slug}-from-{dslug}", rep)
        else:
            for case in self.cases:
                for method in self.methods:
                    for model in self.models:
                        for track in self.tracks:
                            for rep in range(self.repetitions):
                                ref = ModelRef.parse(model)
                                add({"case_id": case, "method": method, "model": model, "track": track,
                                     "design": REFERENCE if track == "fixed_graph" else "own"},
                                    f"{case[:3]}-{method}-{ref.slug}-{TRACK_SHORT[track]}", rep)
        # Random order spreads order effects (cache warmth, rate limits) over cells.
        random.Random(self.seed).shuffle(out)
        return out


def successful_designs(root: Path) -> list[dict]:
    """The designs of a design experiment that passed the structural check."""
    exp = Experiment.load(root)
    if exp.stage != "design":
        raise ValueError(f"{exp.name} is not a design experiment")
    out = []
    for ep in exp.episodes():
        state = episode_state(root / "episodes" / ep["id"]) or {}
        if state.get("status") == "designed" and (root / "episodes" / ep["id"] / "work/design.json").is_file():
            out.append({k: ep[k] for k in ("case_id", "model", "repetition")} | {"episode": ep["id"]})
    return sorted(out, key=lambda d: (d["case_id"], d["model"], d["repetition"]))


def dependents(runs: Path, name: str) -> list[str]:
    """Code experiments that take their designs from experiment `name`."""
    out = []
    for path in sorted(runs.glob("*/experiment.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("design_from") == name and path.parent.name != name:
            out.append(path.parent.name)
    return out


def delete_experiment(root: Path) -> None:
    """Removes an experiment folder: episodes, evaluations and all."""
    if not (root / "experiment.json").is_file():
        raise FileNotFoundError(f"{root} is not an experiment")
    shutil.rmtree(root)


def delete_runs(root: Path, run_ids: list[str]) -> int:
    """Removes runs' folders (and their interrupted attempts); they become pending
    and run again on Resume. Scores of earlier evaluation batches stay in
    `evaluation/` for audit. Returns how many runs had a folder."""
    known = {ep["id"] for ep in Experiment.load(root).episodes()}
    removed = 0
    for run_id in run_ids:
        if run_id not in known:
            raise ValueError(f"{run_id} is not a run of {root.name}")
        folders = [root / "episodes" / run_id, *(root / "episodes").glob(f"{run_id}.interrupted-*")]
        removed += folders[0].is_dir()
        for folder in folders:
            if folder.is_dir():
                shutil.rmtree(folder)
    return removed


def create_experiment(root: Path, corpus: Corpus, *, stage: str = "code", design_from: str = REFERENCE,
                      design_models: list[str] | None = None, **kwargs) -> Experiment:
    """Creates an experiment of one step. Code experiments take designs from
    `design_from`: "reference", or a design experiment next to `root`."""
    if (root / "experiment.json").exists():
        raise FileExistsError(f"{root} already has an experiment; use `bench run --resume`")
    if stage not in STAGES:
        raise ValueError(f"Unknown step {stage!r}; use design or code")
    extra: dict = {"stage": stage}
    if stage == "design":
        kwargs.update(methods=[DESIGN_METHOD], tracks=["natural_language"])
    elif design_from == REFERENCE:
        kwargs.setdefault("tracks", ["fixed_graph"])
        extra["design_from"] = REFERENCE
    else:
        source = root.parent / design_from
        if not (source / "experiment.json").is_file():
            raise ValueError(f"No design experiment named {design_from!r}")
        wanted = set(kwargs.get("cases") or [])
        designs = [d for d in successful_designs(source)
                   if (not wanted or d["case_id"] in wanted)
                   and (not design_models or d["model"] in design_models)]
        if not designs:
            raise ValueError(f"{design_from} has no finished designs for these cases and models")
        kwargs.update(tracks=["natural_language"], cases=sorted({d["case_id"] for d in designs}))
        extra.update(design_from=design_from, designs=designs)
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
    exp = Experiment(dataset={"name": "servicecraft", **corpus.metadata()}, **extra, **kwargs)
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
    method = get_method(exp.method_for(ep["method"]))
    ref = ModelRef.parse(ep["model"])
    reference_file = corpus.root / "corpus/pipelines" / ep["case_id"] / "task_graph.json"
    reference = json.loads(reference_file.read_text()) if reference_file.is_file() else {"tasks": []}
    state = {
        **ep,
        "status": "running",
        "method_version": method.version,
        "dataset_commit": exp.dataset.get("commit"),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        # RQ3: complexity of the reference composition, and the condition of the run.
        "complexity": metrics.complexity(reference),
        "conditions": {
            "method": exp.method_for(ep["method"]), "model": ep["model"], "replicate": ep["repetition"],
            "composition_source": ("prompt" if exp.stage == "design" else ep.get("design_episode")
                                   and f"{exp.design_from}/{ep['design_episode']}" or ep.get("design")),
            "limits": dict(exp.limits),
            "options": {**exp.method_options.get(ep["method"], {}), **(ep.get("condition_options") or {})},
            **({"name": ep["condition"]} if ep.get("condition") else {}),
        },
    }
    _write_state(ep_dir, state)
    events.emit("episode_started", group=f"{ep['method']}·{ep['model']}", **ep)

    phase = "export"
    ctx = None
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
        design = None
        if ep.get("design_episode"):
            source = root.parent / exp.design_from / "episodes" / ep["design_episode"] / "work/design.json"
            if not source.is_file():
                raise FileNotFoundError(f"design {source} is missing (was {exp.design_from} deleted?)")
            design = json.loads(source.read_text())
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
            options={**exp.method_options.get(ep["method"], {}), **(ep.get("condition_options") or {})},
            design=design,
        )
        ctx.work_dir.mkdir()
        result = method.design(ctx) if exp.stage == "design" else method.generate(ctx)
        events.emit("phase_finished", phase=phase)
        state.update(status=result.status, error=result.error, checkpoints=result.checkpoints,
                     counters=result.counters)
        ctx.details.update(result.details)
        design_file = ctx.work_dir / "design.json"
        if design_file.is_file() and ep["track"] == "natural_language" and design is None:
            # Operator-side scoring against the private reference graph, after generation.
            state["design_score"] = design_score.score(json.loads(design_file.read_text()), reference)
    except BudgetExceeded as exc:
        state.update(status="generation_failed", error=f"budget: {exc}", failed_phase=phase)
    except LLMError as exc:
        state.update(status="generation_failed", error=f"model request: {exc}", failed_phase=phase)
    except Exception as exc:  # Infrastructure problems are kept apart from method failures.
        state.update(status="infra_error", error=f"{type(exc).__name__}: {exc}", failed_phase=phase)
        (ep_dir / "traceback.txt").write_text(traceback.format_exc())

    if ctx is not None:
        _merge_details(state, ctx.details)
    graph_file = ep_dir / "work" / "inlumen-graph.json"
    if graph_file.is_file() and exp.stage == "design":
        state["composition"] = metrics.composition_summary(json.loads(graph_file.read_text()))
    if state.get("error") and state["status"] == "generation_failed":
        state.setdefault("failure_category", metrics.classify_failure(state["error"]))
    state["budget"] = budget.snapshot()
    state["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    events.emit("episode_finished", status=state["status"], error=state.get("error"))
    record_usage(ep_dir, state)
    _write_state(ep_dir, state)
    return state


def _merge_details(state: dict, details: dict) -> None:
    for key, value in (details or {}).items():
        if key == "conditions":
            state["conditions"].update(value)
        else:
            state[key] = value


def recompute(root: Path, corpus: Corpus) -> int:
    """Recomputes the measures of finished runs from their files (for runs made before
    a measure existed, or after the scoring changed). Returns how many runs changed."""
    exp = Experiment.load(root)
    changed = 0
    for ep in exp.episodes():
        ep_dir = root / "episodes" / ep["id"]
        state = episode_state(ep_dir)
        if not state or state.get("status") == "running":
            continue
        before = json.dumps(state, sort_keys=True, default=str)
        reference_file = corpus.root / "corpus/pipelines" / ep["case_id"] / "task_graph.json"
        reference = json.loads(reference_file.read_text()) if reference_file.is_file() else None
        if reference is not None:
            state["complexity"] = metrics.complexity(reference)
        state.setdefault("conditions", {"method": exp.method_for(ep["method"]), "model": ep["model"],
                                        "replicate": ep["repetition"], "limits": dict(exp.limits)})
        work = ep_dir / "work"
        if exp.stage == "design" and (work / "inlumen-graph.json").is_file():
            state["composition"] = metrics.composition_summary(json.loads((work / "inlumen-graph.json").read_text()))
        if (reference is not None and (work / "design.json").is_file() and ep["track"] == "natural_language"
                and not ep.get("design_episode")):
            state["design_score"] = design_score.score(json.loads((work / "design.json").read_text()), reference)
        report = Path(state["evaluation_report"]) if state.get("evaluation_report") else None
        if report is not None and (report / "result.json").is_file():
            state["execution"] = metrics.execution_summary(json.loads((report / "result.json").read_text()), report)
        if state.get("error") and state.get("status") == "generation_failed":
            state["failure_category"] = metrics.classify_failure(state["error"])
        record_usage(ep_dir, state)
        if json.dumps(state, sort_keys=True, default=str) != before:
            _write_state(ep_dir, state)
            changed += 1
    return changed


def record_usage(ep_dir: Path, state: dict) -> None:
    """Usage totals and the split by phase, recomputed from the run's events."""
    events = read_events(ep_dir / "events.jsonl")
    state["usage"] = usage_totals(ep_dir / "events.jsonl")
    state["phases"] = metrics.phase_usage(events)


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
