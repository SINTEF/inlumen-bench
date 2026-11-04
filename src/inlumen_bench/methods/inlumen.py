"""inLUMEN-style structured construction, reimplemented standalone.

1. Design: natural-language prompt → task graph (fixed-graph track: the provided
   graph is used as is). Structural problems are fed back for a bounded number of
   design repairs. The design is also saved as inLUMEN Project JSON.
2. Node generation: one model request per task, in dependency order. Each node is
   a `run(inputs) -> dict` function; Dagster wiring and file writing come from the
   deterministic harness.
3. Public validation and repair: failures are attributed to a node from the
   public validator's tracebacks, and only that node is regenerated.
"""

from __future__ import annotations

import json
import re

from ..design import check, from_task_graph, parse_design, to_project_json, topo_order
from ..harness import RUNTIME, default_names, definitions
from ..validate import mandatory_artifacts, validate
from .base import EpisodeContext, InfraError, Method, MethodResult, bundle_hash, parse_bundle, write_bundle

DEFAULTS = {"design_repairs": 2, "max_repairs": 3}

DESIGN_SYSTEM = """\
You are a data pipeline architect. Turn task requirements into a precise task graph.
Answer with one JSON object only."""

DESIGN_FORMAT = """\
Return the design as a JSON object in a ```json block:

{
  "title": "short title",
  "tasks": [
    {
      "id": "snake_case_id",
      "name": "Short human name",
      "operation": "e.g. source_validation, feature_engineering, feature_join, kmeans_clustering",
      "inputs": ["input:<file name>", "<artifact written by another task>"],
      "outputs": ["<artifact file name>"],
      "requirements": ["exact rules for this task copied from the requirements: formulas, parameters, ordering, validation"],
      "parameters": {"name": "value"}
    }
  ]
}

Rules:
- One task per logical task in the requirements; respect any stated task count.
- Read input files as "input:<file name>" (for example "input:data.csv").
- Use exactly the mandatory artifact file names; every mandatory artifact is written by exactly one task.
- A task depends on another only by reading one of its artifacts.
- Several input files with different roles: each file starts its own branch. Its first task reads only
  that file; never chain one branch's tasks after another's. Independent branches run in parallel.
- Branches meet at a joining task whose `inputs` list one artifact per branch (one input per branch,
  each written by a different task). Never route two branches through one artifact.
- Do not use the task id "context".

Example: "clean customers.csv and orders.csv, then merge on customer_id into customer_orders.csv":
clean_customers reads input:customers.csv → customers_clean.csv; clean_orders reads input:orders.csv
→ orders_clean.csv; merge reads ["customers_clean.csv", "orders_clean.csv"] → customer_orders.csv."""

NODE_SYSTEM = """\
You are an expert Python data engineer. Implement one pipeline task as a pure Python function.
Follow the task specification exactly; other tasks are implemented separately."""

NODE_CONTRACT = """\
Write the module `nodes/{module}.py` with:

    def run(inputs: dict) -> dict:

`inputs` has one entry per item in this task's `inputs` list (a joining task gets one entry per branch):
- "input:<file name>": a pathlib.Path to the read-only input file; read it yourself (pandas for CSV/XLSX, json for JSON).
- "<artifact>": the in-memory value the upstream task returned for that artifact
  (pandas.DataFrame for .csv, dict/list for .json, the fitted object for .pkl).
Return a dict whose keys are exactly this task's output artifact names:
- .csv → pandas.DataFrame (written with index=False; column order is kept)
- .json → JSON-serializable dict/list with finite numbers only
- .pkl → the object to pickle (see the model rules in the contract)
Do not write files and do not import Dagster; the harness handles both. Do not use the network.
The runtime contract describes the whole submission; the harness already satisfies its Dagster
wiring and file-writing rules, so follow its environment, package and model rules.
Return the module as one fenced block whose first line is `# file: nodes/{module}.py`."""


class Inlumen(Method):
    name = "inlumen"
    version = "0.2"

    def generate(self, ctx: EpisodeContext) -> MethodResult:
        opts = {**DEFAULTS, **ctx.options}
        counters = {"model_requests": 0, "code_producing_requests": 0, "design_requests": 0,
                    "node_generations": 0, "repairs": 0, "design_repairs": 0, "validations": 0}
        mandatory = mandatory_artifacts(ctx.prompt)

        design, error = self._design(ctx, mandatory, counters, opts)
        if design is None:
            return MethodResult("generation_failed", error=error, counters=counters)
        names = {**default_names(design), **{k: v for k, v in ctx.asset_names.items()}}
        (ctx.work_dir / "design.json").write_text(json.dumps(design, indent=2))
        (ctx.work_dir / "project.json").write_text(
            json.dumps(to_project_json(design, design.get("title") or ctx.case_id, ctx.case_id), indent=2))
        ctx.events.emit("design_ready", tasks=len(design["tasks"]), assets=[names[t["id"]] for t in design["tasks"]])

        order = topo_order(design)
        code: dict[str, str] = {}
        ctx.events.emit("phase_started", phase="codegen")
        for index, task in enumerate(order, start=1):
            ctx.events.emit("asset_started", index=index, total=len(order), name=names[task["id"]])
            code[task["id"]] = self._node(ctx, design, task, names, code, counters)
            ctx.events.emit("asset_finished", index=index, total=len(order), name=names[task["id"]])

        checkpoints = []
        files = self._bundle(design, names, code)
        checkpoints.append({"kind": "initial", "sha256": write_bundle(files, ctx.submission_dir)})
        ctx.events.emit("checkpoint", kind="initial", sha256=checkpoints[-1]["sha256"])

        by_asset = {names[t["id"]]: t for t in order}
        for attempt in range(opts["max_repairs"] + 1):
            ctx.events.emit("phase_started", phase="validate")
            report = validate(ctx.submission_dir, ctx.workspace / "inputs", ctx.prompt, events=ctx.events)
            counters["validations"] += 1
            (ctx.work_dir / f"validation-{attempt}.json").write_text(json.dumps(report.to_dict(), indent=2))
            if report.status == "infra_error":
                # Repairs would be skipped for a reason that has nothing to do with the model.
                raise InfraError(f"public validation could not run: {' '.join(report.errors)[:300]}")
            if report.status == "pass" or attempt == opts["max_repairs"]:
                break
            target = self._attribute(report, by_asset, order, names)
            if target is None:
                ctx.events.emit("log", level="warn", message="could not attribute the failure to a task")
                break
            counters["repairs"] += 1
            ctx.events.emit("phase_started", phase="repair")
            ctx.events.emit("repair_started", attempt=attempt + 1, max_attempts=opts["max_repairs"],
                            reason=f"{names[target['id']]}: {report.stage}")
            code[target["id"]] = self._node(ctx, design, target, names, code, counters,
                                            feedback=report.feedback(), previous=code[target["id"]])
            files = self._bundle(design, names, code)
            checkpoints.append({"kind": f"repair-{attempt + 1}", "sha256": write_bundle(files, ctx.submission_dir)})
            ctx.events.emit("checkpoint", kind=checkpoints[-1]["kind"], sha256=checkpoints[-1]["sha256"])
        checkpoints.append({"kind": "final", "sha256": bundle_hash(ctx.submission_dir),
                            "public_validation": report.status})
        return MethodResult("generated", checkpoints=checkpoints, counters=counters)

    # ---- design ------------------------------------------------------------

    def _design(self, ctx, mandatory, counters, opts):
        ctx.events.emit("phase_started", phase="design")
        if ctx.provided_graph:
            ctx.events.emit("stage_started", stage="using the provided graph")
            return from_task_graph(json.loads(ctx.provided_graph)), None
        messages = [
            {"role": "system", "content": DESIGN_SYSTEM},
            {"role": "user", "content": f"# Requirements\n\n{ctx.prompt}\n\n# Output\n\n{DESIGN_FORMAT}"},
        ]
        problems: list[str] = []
        for attempt in range(opts["design_repairs"] + 1):
            ctx.events.emit("stage_started", stage="draft design" if attempt == 0 else f"fix design ({attempt})")
            result = ctx.llm.chat(messages, purpose="design" if attempt == 0 else "design_repair")
            counters["model_requests"] += 1
            counters["design_requests"] += 1
            counters["design_repairs"] += attempt > 0
            (ctx.work_dir / f"design-response-{attempt}.md").write_text(result.text)
            try:
                design = parse_design(result.text)
                design["inputs"] = [{"id": f"input:{f}", "path": f"inputs/{f}"} for f in ctx.input_files]
                ctx.events.emit("stage_started", stage="check design")
                problems = check(design, mandatory, ctx.input_files)
            except (ValueError, json.JSONDecodeError) as exc:
                design, problems = None, [f"the reply is not a valid design: {exc}"]
            if not problems:
                return design, None
            messages += [
                {"role": "assistant", "content": result.text},
                {"role": "user", "content": "Fix these problems and return the complete corrected design:\n- "
                 + "\n- ".join(problems)},
            ]
        return None, "design problems: " + "; ".join(problems)

    # ---- nodes -------------------------------------------------------------

    def _node(self, ctx, design, task, names, code, counters, feedback=None, previous=None) -> str:
        module = names[task["id"]]
        made_by = {out: t for t in design["tasks"] for out in t["outputs"]}
        upstream = []
        for item in task["inputs"]:
            src = made_by.get(item)
            if src and src["id"] != task["id"] and src["id"] in code and src not in upstream:
                upstream.append(src)
        parts = [
            f"# Overall requirements\n\n{ctx.prompt}",
            "# Pipeline design\n\n```json\n" + json.dumps(
                [{k: t[k] for k in ("id", "name", "inputs", "outputs")} for t in design["tasks"]], indent=1) + "\n```",
            "# Task to implement\n\n```json\n" + json.dumps(task, indent=1) + "\n```",
        ]
        for src in upstream:
            parts.append(f"# Upstream task `{src['id']}` (already implemented)\n\n```python\n{code[src['id']]}```")
        parts.append(f"# Runtime contract\n\n{ctx.contract}")
        parts.append("# What to write\n\n" + NODE_CONTRACT.format(module=module))
        if previous is not None:
            parts.append(f"# Your previous version\n\n```python\n{previous}```")
            parts.append(f"# Public validation feedback\n\n{feedback}\n\nReturn the corrected complete module.")
        messages = [{"role": "system", "content": NODE_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]
        purpose = "node_repair" if previous is not None else "node"
        result = ctx.llm.chat(messages, purpose=purpose)
        counters["model_requests"] += 1
        counters["code_producing_requests"] += 1
        counters["node_generations"] += previous is None
        suffix = f"-repair{counters['repairs']}" if previous is not None else ""
        (ctx.work_dir / f"node-{module}{suffix}.md").write_text(result.text)
        files = parse_bundle(result.text)
        body = files.get(f"nodes/{module}.py") or next(
            (v for k, v in files.items() if k.endswith(f"{module}.py")), None)
        if body is None and len(files) == 1:
            body = next(iter(files.values()))
        if body is None:
            # Keep going: the validator reports it and a repair can replace it.
            body = f"def run(inputs):\n    raise RuntimeError('no code was generated for {module}')\n"
        return body

    @staticmethod
    def _bundle(design, names, code) -> dict[str, str]:
        files = {"bench_runtime.py": RUNTIME, "definitions.py": definitions(design, names),
                 "nodes/__init__.py": ""}
        for task in design["tasks"]:
            files[f"nodes/{names[task['id']]}.py"] = code[task["id"]]
        return files

    @staticmethod
    def _attribute(report, by_asset, order, names):
        """The earliest task (in dependency order) that the public report blames."""
        blamed = set(report.node_errors)
        text = "\n".join(report.errors)
        blamed |= set(re.findall(r"nodes/([A-Za-z0-9_]+)\.py", text))
        blamed |= set(re.findall(r'step "?([A-Za-z0-9_]+)"?', text))
        for task in order:
            if names[task["id"]] in blamed:
                return task
        if report.missing_artifacts:
            for task in order:
                if set(task["outputs"]) & set(report.missing_artifacts):
                    return task
        return None
