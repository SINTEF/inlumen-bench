"""inLUMEN's own agents, from the `inlumen-agents` package (inLUMEN ADR 0005).

1. Design (step 1, `design()`): inLUMEN's pipeline design agent turns the prompt
   into an inLUMEN graph (`inlumen.design.run.design`), with its own tools,
   validation and repair round. inLUMEN does not name artifact files when it
   designs, so a separate **naming pass** (one bench request, purpose `naming`)
   maps each inLUMEN step to the input files it reads or the artifacts it writes.
   The graph's structure is kept as is; the result is checked like any design.
2. Code (step 2, `generate()`): the design (a step-1 design or ServiceCraft's
   reference graph) becomes an inLUMEN graph whose tasks declare their artifact
   file names, and inLUMEN's code generator (`inlumen.codegen.run.generate`) writes
   one Task package per task, with its own planning, repair and validation (in
   Docker, as inLUMEN runs by default). The packages are wrapped as Dagster assets
   (`package_harness`); public validation runs once and is only recorded: repairs
   are inLUMEN's own.

Every model request is recorded in the episode's events and counted against the
budget: code generation goes through the bench's client; the design agent needs
tool calls, so it uses an OpenAI-compatible client on the same endpoint.

Options (`method_options["inlumen"]`): `sandbox` ("docker" or "none"),
`repair_attempts` (inLUMEN's, default 2), `max_tool_iterations` (design, default 30).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid

from ..design import asset_name, check, from_task_graph
from ..harness import default_names
from ..package_harness import canvas_graph, submission_files
from ..validate import mandatory_artifacts, validate
from .base import EpisodeContext, InfraError, Method, MethodResult, write_bundle

DEFAULTS = {"sandbox": "docker", "repair_attempts": 2, "max_tool_iterations": 30, "naming_repairs": 1}

NAMING_SYSTEM = """\
You map the steps of an existing pipeline to the files they read and write.
Do not change the pipeline. Answer with one JSON object only."""

NAMING_FORMAT = """\
Return a JSON object in a ```json block:

{
  "sources": {"<source step id>": ["<input file name>", ...]},
  "tasks": {"<task step id>": ["<artifact file name it writes>", ...]}
}

Rules:
- Every input file is read by exactly one source step; use the file names as listed.
- Every task step writes at least one artifact. Use the mandatory artifact names exactly;
  name any other artifact a task needs to pass on (for example `features.csv`).
- Every mandatory artifact is written by exactly one task: the step whose description
  produces it.
- Use only the step ids given."""


def _require_package():
    from importlib.util import find_spec

    if find_spec("inlumen") is None or find_spec("inlumen.codegen") is None or find_spec("autogen_ext") is None:
        raise InfraError(
            "the inlumen method needs inLUMEN's agents library: "
            "pip install 'inlumen-bench[inlumen]' (see the README)")


def installed_packages(contract: str) -> list[str]:
    """`name==version` lines from the contract's "Installed packages" block."""
    match = re.search(r"## Installed packages\s*```[a-z]*\n(.*?)```", contract, re.S)
    return [line.strip() for line in match.group(1).splitlines() if line.strip()] if match else []


def _usage(event: dict) -> dict | None:
    if not (event.get("prompt_tokens") or event.get("completion_tokens")):
        return None
    return {"input_tokens": event.get("prompt_tokens"), "output_tokens": event.get("completion_tokens"),
            "cached_input_tokens": None, "reasoning_tokens": None,
            "total_tokens": (event.get("prompt_tokens") or 0) + (event.get("completion_tokens") or 0)}


class BenchCodegenModel:
    """inLUMEN's code generator talking to the model through the bench's client."""

    def __init__(self, ctx: EpisodeContext, counters: dict):
        self.ctx = ctx
        self.counters = counters
        self.name = str(ctx.llm.ref) if hasattr(ctx.llm, "ref") else "bench"

    async def complete(self, *, system_prompt, user_prompt, json_output, max_tokens):
        from inlumen.codegen.model import ModelReply

        try:
            operation = json.loads(user_prompt).get("operation") or "codegen"
        except (ValueError, AttributeError):
            operation = "codegen"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        result = await asyncio.to_thread(self.ctx.llm.chat, messages, purpose=operation, max_tokens=max_tokens)
        self.counters["model_requests"] += 1
        self.counters["code_producing_requests"] += 1
        self.counters["repairs"] += operation.startswith("repair")
        usage = result.usage or {}
        raw = {"prompt_tokens": usage.get("input_tokens"), "completion_tokens": usage.get("output_tokens"),
               "total_tokens": usage.get("total_tokens")}
        if result.cost_usd is not None:
            raw["cost"] = result.cost_usd
        return ModelReply(content=result.text, usage=raw if result.usage else None,
                          finish_reason=result.finish_reason)


class Inlumen(Method):
    name = "inlumen"
    version = "1.0"

    @staticmethod
    def _counters() -> dict:
        return {"model_requests": 0, "code_producing_requests": 0, "design_requests": 0,
                "naming_requests": 0, "node_generations": 0, "repairs": 0, "design_repairs": 0,
                "validations": 0}

    @staticmethod
    def _provenance(ctx: EpisodeContext, **extra) -> None:
        from inlumen.contracts import package_version

        (ctx.work_dir / "inlumen.json").write_text(json.dumps(
            {"package": "inlumen-agents", "version": package_version(), **extra}, indent=2))

    # ---- step 1: design ------------------------------------------------------

    def design_model(self, ctx: EpisodeContext):
        """The autogen client for inLUMEN's design agent: the episode's endpoint and model."""
        from inlumen.design.run import openai_model_client

        profile = ctx.llm.profile
        headers = dict(profile.headers)
        if profile.is_openrouter:
            headers.setdefault("X-Title", "inlumen-bench")
        return openai_model_client(ctx.llm.model, base_url=profile.base_url,
                                   api_key=profile.resolved_key() or None,
                                   default_headers=headers or None, timeout=profile.timeout_s)

    def design(self, ctx: EpisodeContext) -> MethodResult:
        _require_package()
        from inlumen.design.catalog import Catalog
        from inlumen.design.run import RecordingModelClient, design

        opts = {**DEFAULTS, **ctx.options}
        counters = self._counters()
        events, budget = ctx.events, ctx.budget
        pending: dict[str, float] = {}

        class Budgeted(RecordingModelClient):
            """Checks the budget and announces each request before it is sent."""

            async def create(self, messages, **kwargs):
                if budget is not None:
                    budget.check()
                request_id = uuid.uuid4().hex[:12]
                pending["id"], pending["started"] = request_id, time.monotonic()
                events.emit("request_started", request_id=request_id, purpose="design_agent",
                            model=str(ctx.llm.ref), attempt=0, max_tokens=None)
                return await super().create(messages, **kwargs)

        def record(event: dict) -> None:
            counters["model_requests"] += 1
            counters["design_requests"] += 1
            request_id = pending.get("id") or uuid.uuid4().hex[:12]
            if budget is not None:
                budget.charge(None)  # The design client reports tokens but no cost.
            snapshot = budget.snapshot() if budget is not None else {}
            if event.get("error"):
                events.emit("request_failed", request_id=request_id, purpose="design_agent", kind="provider",
                            message=event["error"][:2000], latency_s=event.get("latency_s"),
                            usage_provenance="unavailable", budget=snapshot)
                return
            usage = _usage(event)
            events.emit("request_finished", request_id=request_id, purpose="design_agent",
                        model=str(ctx.llm.ref), finish_reason=event.get("finish_reason"),
                        latency_s=event.get("latency_s"), usage=usage, raw_usage=None,
                        usage_provenance="provider_reported" if usage else "unavailable",
                        cost_usd=None, cost_provenance="unavailable", budget=snapshot)
            events.emit("budget_update", **snapshot)

        events.emit("phase_started", phase="design")
        events.emit("stage_started", stage="inLUMEN design agent")
        catalog = Catalog.default()
        model = Budgeted(self.design_model(ctx), lambda _event: None)
        result = asyncio.run(design(ctx.prompt, model=model, catalog=catalog, name=ctx.case_id,
                                    events=record, max_tool_iterations=opts["max_tool_iterations"]))
        counters["design_repairs"] = int(result.repaired)
        (ctx.work_dir / "inlumen-graph.json").write_text(json.dumps(result.graph, indent=2))
        (ctx.work_dir / "inlumen-design.json").write_text(json.dumps(
            {"message": result.message, "accepted": result.accepted, "repaired": result.repaired,
             "sync": result.sync}, indent=2, default=str))
        (ctx.work_dir / "project.json").write_text(json.dumps(result.project, indent=2))
        self._provenance(ctx, catalog_fingerprint=result.catalog_fingerprint)
        if not result.accepted:
            reason = result.sync.get("message") or result.sync.get("status") or "rejected"
            return MethodResult("generation_failed", error=f"inLUMEN rejected its design: {reason}",
                                counters=counters)

        events.emit("stage_started", stage="name artifacts")
        bench_design, problems = self._name_artifacts(ctx, result.graph, counters, opts)
        if bench_design is None or problems:
            if bench_design is not None:
                (ctx.work_dir / "design.rejected.json").write_text(json.dumps(bench_design, indent=2))
            return MethodResult("generation_failed", error="design problems: " + "; ".join(problems),
                                counters=counters)
        (ctx.work_dir / "design.json").write_text(json.dumps(bench_design, indent=2))
        events.emit("design_ready", tasks=len(bench_design["tasks"]), assets=[t["id"] for t in bench_design["tasks"]])
        return MethodResult("designed", counters=counters)

    def _name_artifacts(self, ctx, graph, counters, opts):
        """The naming pass: inLUMEN graph → design with artifact names (structure unchanged)."""
        steps = _steps(graph)
        tasks = [s for s in steps.values() if s["type"] in {"task", "subpipeline"}]
        if not tasks:
            return None, ["inLUMEN's design has no task steps"]
        listing = [{"id": s["id"], "type": s["type"], "label": s["label"], "description": s["description"],
                    "reads_from": s["upstream"]} for s in steps.values() if s["type"] != "destination"]
        mandatory = mandatory_artifacts(ctx.prompt)
        messages = [
            {"role": "system", "content": NAMING_SYSTEM},
            {"role": "user", "content": "\n\n".join([
                f"# Requirements\n\n{ctx.prompt}",
                "# Input files\n\n" + "\n".join(f"- {f}" for f in ctx.input_files),
                "# Mandatory artifacts\n\n" + "\n".join(f"- {a}" for a in mandatory),
                "# Pipeline steps\n\n```json\n" + json.dumps(listing, indent=1) + "\n```",
                f"# Output\n\n{NAMING_FORMAT}",
            ])},
        ]
        problems: list[str] = []
        bench_design = None
        for attempt in range(opts["naming_repairs"] + 1):
            reply = ctx.llm.chat(messages, purpose="naming" if attempt == 0 else "naming_repair")
            counters["model_requests"] += 1
            counters["naming_requests"] += 1
            (ctx.work_dir / f"naming-response-{attempt}.md").write_text(reply.text)
            try:
                naming = _parse_json(reply.text)
                bench_design = _bench_design(graph, steps, naming, ctx.input_files)
                problems = check(bench_design, mandatory, ctx.input_files)
            except (ValueError, KeyError, TypeError) as exc:
                problems = [f"the reply is not a valid naming: {exc}"]
            if not problems:
                return bench_design, []
            messages += [
                {"role": "assistant", "content": reply.text},
                {"role": "user", "content": "Fix these problems and return the complete corrected mapping. "
                 "Only the file names can change, not the steps:\n- " + "\n- ".join(problems)},
            ]
        return bench_design, problems

    # ---- step 2: code --------------------------------------------------------

    def generate(self, ctx: EpisodeContext) -> MethodResult:
        _require_package()
        from inlumen.codegen.run import generate
        from inlumen.codegen.sandbox import DockerSandbox, NoSandbox

        opts = {**DEFAULTS, **ctx.options}
        counters = self._counters()
        ctx.events.emit("phase_started", phase="design")
        if ctx.design is not None:
            ctx.events.emit("stage_started", stage="using the step 1 design")
            design = json.loads(json.dumps(ctx.design))
        elif ctx.provided_graph:
            ctx.events.emit("stage_started", stage="using the provided graph")
            design = from_task_graph(json.loads(ctx.provided_graph))
        else:
            return MethodResult("generation_failed", error="no design: run step 1 first", counters=counters)
        names = {**default_names(design), **ctx.asset_names}
        graph = canvas_graph(design, names, ctx.case_id)
        (ctx.work_dir / "design.json").write_text(json.dumps(design, indent=2))
        (ctx.work_dir / "inlumen-graph.json").write_text(json.dumps(graph, indent=2))
        ctx.events.emit("design_ready", tasks=len(design["tasks"]), assets=[names[t["id"]] for t in design["tasks"]])

        ctx.events.emit("phase_started", phase="codegen")
        ctx.events.emit("stage_started", stage="inLUMEN code generator")
        sandbox = DockerSandbox() if opts["sandbox"] == "docker" else NoSandbox()
        runtime = {"python_version": "3.11", "allowed_packages": installed_packages(ctx.contract),
                   "allow_unlisted_model_packages": False, "network_allowed": False,
                   "max_runtime_seconds": 900}
        options = {"repair_attempts": opts["repair_attempts"],
                   # The bench's budget applies; inLUMEN's own limits must not cut a run short first.
                   "max_llm_requests": 32, "max_reported_cost_usd": 100, "max_generation_seconds": 3600}
        result = asyncio.run(generate(graph, model=BenchCodegenModel(ctx, counters), sandbox=sandbox,
                                      files=ctx.workspace / "inputs", instruction=ctx.prompt,
                                      runtime=runtime, options=options))
        counters["node_generations"] = sum(names[t["id"]] in result.packages for t in design["tasks"])
        (ctx.work_dir / "inlumen-codegen.json").write_text(json.dumps(
            {"accepted": result.accepted, "validation": result.validation,
             "generation_run": result.response.generation_run.model_dump(mode="json")
             if result.response.generation_run else None}, indent=2))
        packages_dir = ctx.work_dir / "packages"
        result.write_packages(packages_dir)
        self._provenance(ctx, sandbox=opts["sandbox"])
        if not result.accepted:
            errors = "; ".join(result.validation.get("errors") or [])[:500]
            return MethodResult("generation_failed", error=f"inLUMEN rejected its code: {errors}",
                                counters=counters)

        try:
            files = submission_files(design, names, result.packages)
        except ValueError as exc:
            return MethodResult("generation_failed", error=str(exc), counters=counters)
        checkpoints = [{"kind": "initial", "sha256": write_bundle(files, ctx.submission_dir)}]
        ctx.events.emit("checkpoint", kind="initial", sha256=checkpoints[-1]["sha256"])

        ctx.events.emit("phase_started", phase="validate")
        report = validate(ctx.submission_dir, ctx.workspace / "inputs", ctx.prompt, events=ctx.events)
        counters["validations"] += 1
        (ctx.work_dir / "validation-0.json").write_text(json.dumps(report.to_dict(), indent=2))
        if report.status == "infra_error":
            raise InfraError(f"public validation could not run: {' '.join(report.errors)[:300]}")
        checkpoints.append({"kind": "final", "sha256": checkpoints[0]["sha256"],
                            "public_validation": report.status})
        return MethodResult("generated", checkpoints=checkpoints, counters=counters)


def _parse_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*\n(\{.*?\})\s*```", text, re.S)
    raw = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _steps(graph: dict) -> dict[str, dict]:
    """inLUMEN steps with their direct upstream step ids."""
    steps: dict[str, dict] = {}
    for node in graph.get("nodes", []):
        data = node.get("data") or {}
        steps[str(node["id"])] = {"id": str(node["id"]), "type": str(data.get("type") or ""),
                                  "label": str(data.get("label") or ""),
                                  "description": str(data.get("description") or ""),
                                  "template": str(data.get("template_label") or ""),
                                  "parameters": data.get("param") if isinstance(data.get("param"), dict) else {},
                                  "upstream": []}
    for edge in graph.get("edges", []):
        source, target = str(edge.get("source")), str(edge.get("target"))
        if source in steps and target in steps and source not in steps[target]["upstream"]:
            steps[target]["upstream"].append(source)
    return steps


def _bench_design(graph: dict, steps: dict[str, dict], naming: dict, input_files: list[str]) -> dict:
    """A bench design with the inLUMEN graph's structure and the naming pass's files.

    Flow steps (conditions, Parallel Map, Collect) pass their inputs through;
    Destinations are left out (the artifacts are written by the task before them).
    """
    sources = {str(k): [str(f) for f in v] for k, v in (naming.get("sources") or {}).items()}
    outputs = {str(k): [str(f) for f in v] for k, v in (naming.get("tasks") or {}).items()}
    unknown = sorted((set(sources) | set(outputs)) - set(steps))
    if unknown:
        raise ValueError(f"unknown step ids {unknown}")

    def reads(step_id: str, seen: frozenset = frozenset()) -> list[str]:
        items: list[str] = []
        for up in steps[step_id]["upstream"]:
            if up in seen:
                continue
            kind = steps[up]["type"]
            if kind == "source":
                found = [f"input:{f}" for f in sources.get(up, [])]
            elif kind in {"task", "subpipeline"}:
                found = list(outputs.get(up, []))
            else:  # flow steps pass their inputs on
                found = reads(up, seen | {step_id})
            items += [i for i in found if i not in items]
        return items

    ids: dict[str, str] = {}
    used: set[str] = set()
    for step in steps.values():
        if step["type"] not in {"task", "subpipeline"}:
            continue
        base = asset_name(step["label"] or step["id"])
        candidate, n = base, 2
        while candidate in used:
            candidate, n = f"{base}_{n}", n + 1
        used.add(candidate)
        ids[step["id"]] = candidate
    tasks = [{
        "id": ids[sid], "name": steps[sid]["label"], "operation": steps[sid]["template"] or steps[sid]["type"],
        "inputs": reads(sid), "outputs": outputs.get(sid, []),
        "requirements": [steps[sid]["description"]] if steps[sid]["description"] else [],
        "parameters": steps[sid]["parameters"], "inlumen_step": sid,
    } for sid in ids]
    return {"title": (graph.get("pipeline") or {}).get("name") or "",
            "inputs": [{"id": f"input:{f}", "path": f"inputs/{f}"} for f in input_files],
            "tasks": tasks}

