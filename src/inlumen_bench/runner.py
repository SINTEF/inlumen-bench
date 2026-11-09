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
import re
import shutil
import threading
import traceback
from dataclasses import asdict, dataclass, field, replace
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

# Errors inLUMEN's agents raise as plain exceptions that are the model's failure, not the
# infrastructure's: e.g. after a tool call the model gave back no usable answer.
MODEL_FAILURES = re.compile(r"produced no valid text response", re.I)


def is_model_failure(error: str) -> bool:
    return bool(MODEL_FAILURES.search(error or ""))
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
    # Runs of one model that may run at the same time. A local server (Ollama) answers one
    # request per model at a time, so more only queue (and the wait counts as run time);
    # with 1, parallel runs go to different models, each on its own GPU. Hosted APIs can
    # take more.
    runs_per_model: int = 1

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
        # One model at a time, in random order within each model: random order spreads order
        # effects (cache warmth, rate limits) over cells, while interleaving models would make a
        # local server (Ollama) keep several loaded at once, pushing some off the GPU and timing
        # model loads instead of runs.
        random.Random(self.seed).shuffle(out)
        out.sort(key=lambda ep: self.models.index(ep["model"]) if ep["model"] in self.models else len(self.models))
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


class ModelQueue:
    """Hands out pending runs model by model.

    A profile with several `endpoints` (one Ollama server per GPU) gets one lane per
    endpoint. A lane runs one model at a time (at most `per_model` runs of it at once)
    and keeps that model while it has runs left; every lane can run the same model, so
    no GPU waits while another model finishes. Other profiles use their base URL, with
    at most `per_model` runs of one model at a time, and a worker keeps to its model.
    """

    def __init__(self, todo: list[dict], per_model: int = 1, endpoints: dict[str, list[str]] | None = None) -> None:
        self.runs: dict[str, list[dict]] = {}
        for ep in todo:  # `todo` is already in model order.
            self.runs.setdefault(ep["model"], []).append(ep)
        self.active = {m: 0 for m in self.runs}
        self.per_model = max(1, per_model)
        self.endpoints = {name: urls for name, urls in (endpoints or {}).items() if urls}
        self.lanes = {url: {"model": None, "busy": 0} for urls in self.endpoints.values() for url in urls}
        self.lock = threading.Lock()

    def _lane(self, model: str, prefer: str | None) -> str | None:
        urls = self.endpoints[model.partition(":")[0]]
        ranked = ([prefer] if prefer in urls else []) + urls
        for url in ranked:  # A lane already on this model, then an idle one (unloaded first).
            if self.lanes[url]["model"] == model and self.lanes[url]["busy"] < self.per_model:
                return url
        idle = [u for u in ranked if not self.lanes[u]["busy"]]
        return next((u for u in idle if self.lanes[u]["model"] in (None, model)), idle[0] if idle else None)

    def take(self, prefer: str | None = None, lane: str | None = None) -> tuple[dict, str | None, str | None] | None:
        """The next run for a worker: (run, endpoint or None for the base URL, model to
        unload from that endpoint first or None)."""
        with self.lock:
            order = ([prefer] if prefer in self.runs else []) + list(self.runs)
            for model in order:
                if not self.runs[model]:
                    continue
                if model.partition(":")[0] not in self.endpoints:
                    if self.active[model] < self.per_model:
                        self.active[model] += 1
                        return self.runs[model].pop(0), None, None
                    continue
                url = self._lane(model, lane)
                if url is None:
                    continue
                previous = self.lanes[url]["model"]
                self.lanes[url].update(model=model, busy=self.lanes[url]["busy"] + 1)
                self.active[model] += 1
                return self.runs[model].pop(0), url, previous if previous not in (None, model) else None
            return None  # The rest belong to models or lanes other workers are running.

    def release(self, ep: dict, url: str | None = None) -> list[str | None] | None:
        """Marks a run done. When that was the model's last run, returns where to unload it
        from (endpoints, or [None] for the base URL); otherwise None."""
        model = ep["model"]
        with self.lock:
            self.active[model] -= 1
            if url is not None:
                self.lanes[url]["busy"] -= 1
            if self.runs[model] or self.active[model]:
                return None
            if model.partition(":")[0] not in self.endpoints:
                return [None]
            holding = [u for u, lane in self.lanes.items() if lane["model"] == model and not lane["busy"]]
            for u in holding:
                self.lanes[u]["model"] = None
            return holding

    def progress(self, model: str) -> tuple[int, int]:
        with self.lock:
            return self.active[model], len(self.runs[model])


def unload_model(profiles: dict, model: str, log: EventLog, url: str | None = None, *, finished: bool = True) -> None:
    """Asks an Ollama server (`url`, or the profile's base URL) to unload a model, so the
    next model gets the whole GPU. Other servers are left alone."""
    import httpx

    ref = ModelRef.parse(model)
    profile = profiles.get(ref.profile)
    if profile is None:
        return
    root = (url or profile.base_url).rstrip("/").removesuffix("/v1")
    try:
        with httpx.Client(timeout=10) as client:
            if "version" not in client.get(f"{root}/api/version").json():
                return
            client.post(f"{root}/api/generate", json={"model": ref.model, "keep_alive": 0}, timeout=60)
        where = f" at {root}" if url else ""
        log.emit("log", message=f"{model}: " + ("all runs done; unloaded" if finished else "unloaded")
                 + f" from Ollama{where}")
    except Exception:  # noqa: BLE001, S110 - not Ollama, or not reachable: nothing to free
        pass


def run_experiment(root: Path, corpus: Corpus, workers: int = 2, stop: threading.Event | None = None) -> None:
    exp = Experiment.load(root)
    todo = pending_episodes(root, exp)
    profiles = load_profiles()
    limits = Limits(**exp.limits)
    used = {ModelRef.parse(m).profile for m in exp.models}
    endpoints = {name: p.run_endpoints for name, p in profiles.items() if name in used and p.run_endpoints}
    queue = ModelQueue(todo, exp.runs_per_model, endpoints)
    log = EventLog(None)
    errors: list[BaseException] = []
    for name, urls in endpoints.items():
        log.emit("log", message=f"{name}: {len(urls)} servers side by side: {', '.join(urls)}")

    def worker() -> None:
        model = lane = None
        while not (stop and stop.is_set()):
            taken = queue.take(prefer=model, lane=lane)
            if taken is None:
                return
            ep, url, previous = taken
            if previous:
                unload_model(profiles, previous, log, url, finished=False)
            if (ep["model"], url) != (model, lane):
                model, lane = ep["model"], url
                where = f" on {url}" if url else ""
                log.emit("log", message=f"{model}: starting its runs{where} ({sum(queue.progress(model))} left)")
            try:
                run_episode(root, corpus, exp, ep, profiles, limits, endpoint=url)
            except BaseException as exc:  # noqa: BLE001 - re-raised after the other workers stop
                errors.append(exc)
                return
            finally:
                for where in queue.release(ep, url) or []:
                    unload_model(profiles, ep["model"], log, where)

    threads = [threading.Thread(target=worker, name=f"run-worker-{i}", daemon=True)
               for i in range(max(1, workers))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def run_episode(root: Path, corpus: Corpus, exp: Experiment, ep: dict, profiles, limits: Limits,
                endpoint: str | None = None) -> dict:
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
        **({"endpoint": endpoint} if endpoint else {}),
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
        if endpoint:
            profile = replace(profile, base_url=endpoint)
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
        error = f"{type(exc).__name__}: {exc}"
        status = "generation_failed" if is_model_failure(error) else "infra_error"
        state.update(status=status, error=error, failed_phase=phase)
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
            result = json.loads((report / "result.json").read_text())
            state["execution"] = metrics.execution_summary(result, report)
            state["static"] = metrics.static_summary(result)
            state.setdefault("stages", {name: (result.get("stages") or {}).get(name, {}).get("status")
                                        for name in metrics.EVALUATION_STAGES})
        if state.get("status") == "infra_error" and is_model_failure(state.get("error")):
            state["status"] = "generation_failed"  # labelled before the bench knew it as a model failure
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
