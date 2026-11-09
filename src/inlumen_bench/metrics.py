"""Measurements recorded for every run, for the three research questions.

- RQ1 (pipeline composition): `composition_summary` describes a composition graph
  (components, Source and Destination nodes, dependencies and bindings at port
  level); `design_score` compares it with the reference composition.
- RQ2 (pipeline implementation and execution): `classify_failure` sorts validation
  and repair feedback into categories; `execution_summary` reads ServiceCraft's
  result (did Dagster load and complete the pipeline, which components failed, which
  outputs were produced and which passed their checks).
- RQ3 (cost): `complexity` measures the reference composition (never the generated
  one, so a composition with missing components can't make a case look simpler);
  `phase_usage` splits tokens and time by phase.

Everything here is a pure function of files the run already wrote, so results can be
recomputed for old runs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from .design import edges

# ---- complexity (RQ3) ------------------------------------------------------------

# Levels by component count, chosen from the corpus: about a third of the 100
# cases has 4 components or fewer, half has 5–6 and the rest 7 or more.
COMPLEXITY_LEVELS = (("low", 4), ("medium", 6), ("high", None))


def complexity(task_graph: dict) -> dict:
    """Size and shape of a reference composition (ServiceCraft `task_graph.json`)."""
    tasks = [t["id"] for t in task_graph.get("tasks", [])]
    deps = {(a, b) for a, b, _ in edges(task_graph)}
    up: dict[str, set] = {t: set() for t in tasks}
    down: dict[str, set] = {t: set() for t in tasks}
    for a, b in deps:
        up[b].add(a)
        down[a].add(b)
    depth: dict[str, int] = {}

    def chain(t: str, seen: frozenset = frozenset()) -> int:
        if t not in depth:
            depth[t] = 1 + max((chain(u, seen | {t}) for u in up[t] if u not in seen), default=0)
        return depth[t]

    inputs = {i for t in task_graph.get("tasks", []) for i in t.get("inputs", []) if i.startswith("input:")}
    n = len(tasks)
    level = next(name for name, limit in COMPLEXITY_LEVELS if limit is None or n <= limit)
    return {
        "components": n,
        "dependencies": len(deps),
        "inputs": len(inputs),
        "outputs": sum(len(t.get("outputs", [])) for t in task_graph.get("tasks", [])),
        "longest_chain": max((chain(t) for t in tasks), default=0),
        "max_fan_out": max((len(v) for v in down.values()), default=0),
        "max_fan_in": max((len(v) for v in up.values()), default=0),
        # Components where the pipeline splits or joins.
        "branching_points": sum(1 for t in tasks if len(down[t]) > 1 or len(up[t]) > 1),
        "topology": task_graph.get("topology") or "",
        "level": level,
    }


# ---- composition (RQ1) ------------------------------------------------------------

def composition_summary(graph: dict) -> dict:
    """A composition graph (inLUMEN canvas graph) in countable terms.

    Components are the steps; Sources and Destinations are counted apart.
    Dependencies keep their ports; bindings are the parameter values set on
    components and the data files attached to Sources.
    """
    nodes = [n for n in graph.get("nodes") or [] if isinstance(n, dict)]
    components, by_type, bindings = [], {}, []
    for node in nodes:
        data = node.get("data") if isinstance(node.get("data"), dict) else node
        kind = str(data.get("type") or "")
        by_type[kind] = by_type.get(kind, 0) + 1
        components.append({"id": str(node.get("id")), "type": kind, "label": str(data.get("label") or ""),
                           "template": str(data.get("template_label") or data.get("definition_id") or "")})
        params = data.get("param") if isinstance(data.get("param"), dict) else {}
        for key, value in params.items():
            bindings.append({"component": str(node.get("id")), "kind": "parameter", "name": str(key),
                             "value": value if isinstance(value, (str, int, float, bool)) or value is None
                             else str(value)[:200]})
        for item in data.get("files") or []:
            name = item if isinstance(item, str) else (item or {}).get("filename") or (item or {}).get("name")
            if name:
                bindings.append({"component": str(node.get("id")), "kind": "file", "name": str(name)})
    dependencies = [{"source": str(e.get("source")), "target": str(e.get("target")),
                     "source_port": str(e.get("sourceHandle") or e.get("source_port") or ""),
                     "target_port": str(e.get("targetHandle") or e.get("target_port") or "")}
                    for e in graph.get("edges") or [] if isinstance(e, dict)]
    return {
        "components": len(components),
        "by_type": by_type,
        "sources": [c["id"] for c in components if c["type"] == "source"],
        "destinations": [c["id"] for c in components if c["type"] == "destination"],
        "dependencies": len(dependencies),
        "bindings": len(bindings),
        "component_list": components,
        "dependency_list": dependencies,
        "binding_list": bindings,
    }


# ---- implementation (RQ2) ---------------------------------------------------------

FAILURE_CATEGORIES = ("budget", "syntax", "format", "contract", "dependency", "sandbox", "deployment", "runtime",
                      "provider", "other")
# Matched in this order against inLUMEN's validation messages (inlumen.codegen
# validation), the bench's public validator, ServiceCraft's checks and run errors.
_CATEGORY_RULES: list[tuple[str, re.Pattern]] = [
    ("budget", re.compile(r"^budget:|limit .*reached", re.I)),
    ("syntax", re.compile(r"invalid syntax|syntaxerror|indentationerror|parse error|unexpected eof", re.I)),
    # The model's reply could not be read (not the JSON or files asked for).
    ("format", re.compile(r"invalid json|jsondecodeerror|expecting .*delimiter|unterminated string|"
                          r"not a valid (graph|naming)|reply is not|no definitions\.py|one json object|"
                          r"no valid text response", re.I)),
    ("provider", re.compile(r"^model request:|http \d{3}|request failed after", re.I)),
    ("dependency", re.compile(r"no module named|modulenotfounderror|importerror|cannot import name|"
                              r"allowed package|requirement|reviewed dependency|pip install", re.I)),
    ("contract", re.compile(r"contract|artifact|expected output|missing output|output file|did not write|"
                            r"declare|required (csv )?columns?|columns? missing|manifest|cardinality|"
                            r"required generated file|published artifact", re.I)),
    ("sandbox", re.compile(r"sandbox|docker|container|timed out|timeout|oom|out of memory|killed|"
                           r"banned (module|function)|network", re.I)),
    ("deployment", re.compile(r"dagster|definitions\.py|definition graph|defs\b|asset graph|materializ|"
                              r"compilation failed|deploy", re.I)),
    ("runtime", re.compile(r"traceback|error|exception|failed|exit(ed)? (with )?(code|status)", re.I)),
]


def classify_failure(message: str) -> str:
    """One category for a validation or execution error message.

    Rules are checked in order, so an error naming both a missing module and an
    artifact counts as a dependency failure.
    """
    for category, pattern in _CATEGORY_RULES:
        if pattern.search(message or ""):
            return category
    return "other"


def categorize(messages: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for message in messages:
        category = classify_failure(message)
        counts[category] = counts.get(category, 0) + 1
    return counts


# ---- tokens and time by phase (RQ3) -------------------------------------------------

# Request purpose → phase of pipeline construction.
def request_phase(purpose: str) -> str:
    p = (purpose or "").lower()
    if p.startswith(("design", "naming", "composition")):
        return "composition"  # its repairs included: they repair the composition
    if "repair" in p:
        return "repair"
    if p.startswith(("generate", "node", "codegen")):
        return "implementation"
    return "other"


# Bench phase events → reported phase. The method's `validate` phase is the
# public check of the bundle, and `bundle` its deterministic assembly.
PHASE_NAMES = {"export": "setup", "generate": "setup", "design":"composition", "codegen": "implementation", "repair": "repair",
               "bundle": "bundle", "validate": "validation", "evaluate": "execution"}


def phase_usage(events: list[dict]) -> dict:
    """Requests, tokens and model time per phase, and wall time per phase.

    Wall time comes from phase events: a phase lasts until the next phase starts,
    or until it finishes. Failed requests are counted (their time too).
    """
    usage: dict[str, dict] = {}
    for e in events:
        if e.get("type") not in ("request_finished", "request_failed"):
            continue
        phase = request_phase(e.get("purpose", ""))
        row = usage.setdefault(phase, {"requests": 0, "failed_requests": 0, "input_tokens": 0,
                                       "output_tokens": 0, "cost_usd": 0.0, "model_seconds": 0.0})
        row["requests"] += 1
        row["failed_requests"] += e["type"] == "request_failed"
        row["model_seconds"] += e.get("latency_s") or 0
        tokens = e.get("usage") or {}
        row["input_tokens"] += tokens.get("input_tokens") or 0
        row["output_tokens"] += tokens.get("output_tokens") or 0
        row["cost_usd"] += e.get("cost_usd") or 0
    for row in usage.values():
        row["cost_usd"] = round(row["cost_usd"], 6)
        row["model_seconds"] = round(row["model_seconds"], 2)

    wall: dict[str, float] = {}
    open_phase, started = None, None
    for e in events:
        kind, mono = e.get("type"), e.get("mono")
        if mono is None:
            continue
        if kind == "phase_started" or (kind in ("phase_finished", "episode_finished") and open_phase):
            if open_phase is not None and started is not None:
                name = PHASE_NAMES.get(open_phase, open_phase)
                wall[name] = wall.get(name, 0.0) + max(0.0, mono - started)
            open_phase, started = (e.get("phase"), mono) if kind == "phase_started" else (None, None)
    # A phase that only marks a step (e.g. loading a given composition) is left out.
    return {"by_phase": usage, "seconds": {k: round(v, 2) for k, v in wall.items() if v >= 0.01}}


# ---- execution (RQ2) ---------------------------------------------------------------

_STEP_TIME = re.compile(r"Finished execution of step \"?([^\" ]+)\"? in ([\d.]+)(ms|s)")


def execution_summary(result: dict, report_dir: Path | None = None) -> dict:
    """What happened when ServiceCraft ran the pipeline in Dagster."""
    stages = result.get("stages") or {}
    graph = stages.get("definition_graph") or {}
    execution = stages.get("execution") or {}
    contracts = stages.get("contracts") or {}
    oracle = stages.get("oracle") or {}
    events = execution.get("events") or []
    failed = sorted({e.get("step_key") for e in events if e.get("type") == "STEP_FAILURE" and e.get("step_key")})
    step_seconds = 0.0
    for e in events:
        match = _STEP_TIME.search(e.get("message") or "")
        if match:
            step_seconds += float(match.group(2)) / (1000 if match.group(3) == "ms" else 1)
    produced = []
    if report_dir is not None and (report_dir / "artifacts").is_dir():
        produced = sorted(str(p.relative_to(report_dir / "artifacts"))
                          for p in (report_dir / "artifacts").rglob("*") if p.is_file())
    checks = contracts.get("checks") or []
    oracle_checks = [c for c in oracle.get("checks") or [] if "id" in c]
    errors = [str(c.get("error")) for c in checks if c.get("error")]
    return {
        "loaded": (graph.get("actual") or {}).get("status") == "pass" or graph.get("status") == "pass",
        "definition_graph_score": graph.get("structure_score_01"),
        "missing_components": graph.get("missing_nodes") or [],
        "unexpected_components": graph.get("unexpected_nodes") or [],
        "completed": bool(execution.get("run_success")),
        "completed_components": execution.get("completed_steps") or [],
        "failed_components": failed,
        "materialized": execution.get("materialized") or [],
        "outputs_produced": produced,
        "outputs_checked": len(checks),
        "outputs_valid": sum(1 for c in checks if c.get("status") == "pass"),
        "outputs_invalid": [c.get("artifact") for c in checks if c.get("status") != "pass"],
        "outputs_matching_reference": sum(1 for c in oracle_checks if c.get("status") == "pass"),
        "outputs_compared": len(oracle_checks),
        "output_errors": errors,
        "static_errors": [str(e)[:300] for e in (stages.get("static") or {}).get("syntax_errors") or []][:20],
        "evaluation_seconds": result.get("elapsed_seconds"),
        "dagster_step_seconds": round(step_seconds, 3),
        "failure_stage": next((name for name in ("submission", "static", "definition_graph", "execution",
                                                 "contracts", "oracle")
                               if (stages.get(name) or {}).get("status") not in (None, "pass")), None),
    }
