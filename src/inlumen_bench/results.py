"""Results per run and aggregated per cell, shared by the CLI, the dashboard and exports.

A *run* is one case × method × model × condition × replicate (an "episode" in the
run folders). A *cell* groups runs by method, model, composition source and
condition. Denominators are explicit: every scheduled run counts. Generation
failures count as failures; pending runs and infrastructure errors are reported
apart and left out of rates, as ServiceCraft recommends. Failed runs are always
included in cost.
"""

from __future__ import annotations

import csv
import io
import json
import statistics
from math import comb
from pathlib import Path

from .metrics import EVALUATION_STAGES, FAILURE_CATEGORIES, STATIC_TOOLS
from .runner import Experiment, episode_state

SCORE_KEYS = ("dagster_executable", "output_contract_valid", "behaviorally_correct", "complete_success")
DESIGN_KEYS = ("artifact_coverage", "task_recall", "task_one_to_one", "dependency_recall", "binding_recall",
               "design_correct")
STATIC_KEYS = ("files", "physical_lines", "syntax_valid",
               *(f"{t}_{k}" for t in STATIC_TOOLS for k in ("findings", "flagged_lines", "line_fraction")),
               "pylint_error", "pylint_warning", "pylint_refactor", "pylint_convention",
               "bandit_high", "bandit_medium", "bandit_low",
               "radon_blocks", "radon_mean_cc", "radon_max_cc", "radon_mi", "radon_mi_01")
COUNTER_KEYS = ("code_producing_requests", "design_requests", "naming_requests", "node_generations",
                "design_repairs")
# Seconds of pipeline construction: everything before Dagster runs the pipeline.
CONSTRUCTION_PHASES = ("composition", "implementation", "repair", "bundle", "validation")
DEFAULT_GROUPS = ("method", "model", "design", "condition")


def condition_label(row: dict) -> str:
    """The experimental condition in a few words, e.g. "contracts · repair ×2"."""
    c = row.get("conditions") or {}
    if row.get("stage", "code") == "design" or c.get("phase") == "composition":
        return "ungrounded" if c.get("grounded") is False else "grounded"
    if c.get("baseline") or row.get("method") == "direct":
        return "baseline"
    if not c:  # runs from before conditions were recorded
        return "default"
    contracts = "contracts" if c.get("artifact_contracts", True) else "no contracts"
    repair = f"repair ×{c.get('max_repair_attempts')}" if c.get("repair", True) else "no repair"
    return f"{contracts} · {repair}"


def episodes(root: Path) -> list[dict]:
    exp = Experiment.load(root)
    rows = []
    for ep in exp.episodes():
        state = episode_state(root / "episodes" / ep["id"]) or {**ep, "status": "pending"}
        # The method that ran: experiments from before 0.2 ran `inlumen-standalone` as `inlumen`.
        row = {**ep, **state, "method": exp.method_for(ep["method"]), "stage": exp.stage}
        row["condition"] = row.get("condition") or condition_label(row)
        row["complexity_level"] = (row.get("complexity") or {}).get("level")
        rows.append(row)
    return rows


def _outcome(row: dict) -> str:
    """scored | failed (generation) | pending | infra."""
    status = row.get("status")
    if status in ("evaluated", "designed"):
        return "scored"
    if status == "generation_failed":
        return "failed"
    if status == "infra_error" or row.get("evaluation_status") in ("infra_error", "blocked_control"):
        return "infra"
    return "pending"


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased pass@k from n samples with c successes (Chen et al., 2021)."""
    if n < k or k < 1:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _rate(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(bool(v) for v in values) / len(values), 4) if values else None


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def _median(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 2) if values else None


def construction_seconds(row: dict) -> float | None:
    seconds = (row.get("phases") or {}).get("seconds") or {}
    if not seconds:
        return (row.get("budget") or {}).get("elapsed_s")
    return round(sum(seconds.get(p, 0) for p in CONSTRUCTION_PHASES), 2)


def summarize(root: Path, group_by: tuple[str, ...] = DEFAULT_GROUPS) -> list[dict]:
    rows = episodes(root)
    exp = Experiment.load(root)
    cells: dict[tuple, list[dict]] = {}
    for row in rows:
        row["category"] = "A" if int(row["case_id"][:3]) <= 13 else "B"
        cells.setdefault(tuple(row.get(k) for k in group_by), []).append(row)

    out = []
    for key, members in sorted(cells.items(), key=lambda kv: [str(x) for x in kv[0]]):
        outcomes = [_outcome(r) for r in members]
        decided = [r for r, o in zip(members, outcomes) if o in ("scored", "failed")]
        cell = dict(zip(group_by, key))
        cell.update(
            episodes=len(members),
            scored=outcomes.count("scored"),
            generation_failed=outcomes.count("failed"),
            pending=outcomes.count("pending"),
            infra=outcomes.count("infra"),
            decided=len(decided),
        )
        # RQ1: the share of runs that produced a structurally valid composition.
        cell["valid_design"] = (round(sum(r.get("status") == "designed" for r in decided) / len(decided), 4)
                                if decided else None)
        for score in SCORE_KEYS:
            hits = sum(1 for r in decided if (r.get("scores") or {}).get(score) == 1)
            cell[score] = round(hits / len(decided), 4) if decided else None

        # Success rate per replicate (over cases), to show spread across replicates.
        by_rep: dict[int, list[int]] = {}
        for r in decided:
            by_rep.setdefault(r["repetition"], []).append(int((r.get("scores") or {}).get("complete_success") == 1))
        rep_rates = [sum(v) / len(v) for v in by_rep.values() if v]
        cell["success_mean"] = round(statistics.mean(rep_rates), 4) if rep_rates else None
        cell["success_std"] = round(statistics.stdev(rep_rates), 4) if len(rep_rates) > 1 else None

        # pass@k per case over its replicates, averaged over cases.
        by_case: dict[str, list[int]] = {}
        for r in decided:
            by_case.setdefault(r["case_id"], []).append(int((r.get("scores") or {}).get("complete_success") == 1))
        for k in sorted({1, exp.repetitions}):
            values = [p for v in by_case.values() if (p := pass_at_k(len(v), sum(v), k)) is not None]
            cell[f"pass@{k}"] = round(statistics.mean(values), 4) if values else None

        # RQ2: first attempt vs final, from the method's own validation trace.
        traced = [r for r in decided if (r.get("implementation") or {}).get("final_valid") is not None]
        initially_failing = [r for r in traced if not r["implementation"].get("initial_valid")]
        cell["initial_valid"] = _rate([r["implementation"].get("initial_valid") for r in traced])
        cell["final_valid"] = _rate([r["implementation"].get("final_valid") for r in traced])
        cell["recovered_by_repair"] = _rate([r["implementation"].get("recovered_by_repair")
                                             for r in initially_failing])
        cell["mean_attempts"] = _mean([r["implementation"].get("attempts") for r in traced])
        # Over finished runs like every other rate: a run that never reached ServiceCraft did not load.
        executed = any(r.get("execution") for r in decided)
        cell["dagster_loaded"] = (_rate([bool((r.get("execution") or {}).get("loaded")) for r in decided])
                                  if executed else None)
        cell["dagster_completed"] = (_rate([bool((r.get("execution") or {}).get("completed")) for r in decided])
                                     if executed else None)

        # RQ3: cost and time, failed runs included.
        costs = [(r.get("usage") or {}).get("cost_usd") or 0 for r in members]
        tokens = [((r.get("usage") or {}).get("input_tokens") or 0) + ((r.get("usage") or {}).get("output_tokens") or 0)
                  for r in members]
        successes = sum(1 for r in decided if (r.get("scores") or {}).get("complete_success") == 1)
        if exp.stage == "design":
            successes = sum(1 for r in decided if r.get("status") == "designed")
        cell["cost_usd"] = round(sum(costs), 4)
        cell["cost_incomplete"] = any((r.get("usage") or {}).get("cost_unavailable") for r in members)
        cell["cost_per_run"] = round(sum(costs) / len(members), 4) if members else None
        cell["cost_per_success"] = round(sum(costs) / successes, 4) if successes else None
        cell["output_tokens"] = sum((r.get("usage") or {}).get("output_tokens") or 0 for r in members)
        cell["input_tokens"] = sum((r.get("usage") or {}).get("input_tokens") or 0 for r in members)
        cell["tokens_per_run"] = round(sum(tokens) / len(members)) if members else None
        cell["tokens_per_success"] = round(sum(tokens) / successes) if successes else None
        times = [(r.get("budget") or {}).get("elapsed_s") for r in members if (r.get("budget") or {}).get("elapsed_s")]
        cell["median_generation_s"] = round(statistics.median(times), 1) if times else None
        cell["median_construction_s"] = _median([construction_seconds(r) for r in members if r.get("budget")])
        cell["median_execution_s"] = _median([(r.get("execution") or {}).get("evaluation_seconds") for r in members])
        designs = [r["design_score"] for r in members if r.get("design_score")]
        for k in DESIGN_KEYS:
            values = [d[k] for d in designs if d.get(k) is not None]
            cell[f"design_{k}"] = round(statistics.mean(values), 4) if values else None
        cell["mean_operations"] = _mean([(r.get("composition_trace") or {}).get("operations") for r in members])
        out.append(cell)
    return out


# ---- one row per run, for analysis ------------------------------------------------

def run_row(experiment: str, row: dict) -> dict:
    """A run as one flat record: conditions, complexity and every RQ1–RQ3 measure."""
    c = row.get("conditions") or {}
    cx = row.get("complexity") or {}
    comp = row.get("composition") or {}
    score = row.get("design_score") or {}
    trace = row.get("composition_trace") or {}
    impl = row.get("implementation") or {}
    ex = row.get("execution") or {}
    usage = row.get("usage") or {}
    phases = (row.get("phases") or {}).get("by_phase") or {}
    seconds = (row.get("phases") or {}).get("seconds") or {}
    stage_s = impl.get("stage_seconds") or {}
    scores = row.get("scores") or {}
    stages = row.get("stages") or {}
    static = row.get("static") or {}
    counters = row.get("counters") or {}
    out = {
        "experiment": experiment, "run": row.get("id"),
        "phase": "composition" if row.get("stage") == "design" else "implementation",
        "case": row.get("case_id"), "case_number": str(row.get("case_id", ""))[:3],
        "method": row.get("method"), "model": row.get("model"), "replicate": row.get("repetition"),
        "condition": row.get("condition"), "composition_source": c.get("composition_source") or row.get("design"),
        "status": row.get("status"), "outcome": _outcome(row), "error": row.get("error") or "",
        "failure_category": row.get("failure_category") or impl.get("final_failure_category") or "",
        # conditions
        "grounded": c.get("grounded"), "artifact_contracts": c.get("artifact_contracts"),
        "repair": c.get("repair"), "max_repair_attempts": c.get("max_repair_attempts"),
        "sandbox": c.get("sandbox"), "max_cost_usd": (c.get("limits") or {}).get("max_cost_usd"),
        "max_seconds": (c.get("limits") or {}).get("max_seconds"),
        "max_requests": (c.get("limits") or {}).get("max_requests"),
        # complexity of the reference composition
        **{f"ref_{k}": cx.get(k) for k in ("level", "components", "dependencies", "inputs", "outputs",
                                           "longest_chain", "max_fan_out", "max_fan_in", "branching_points",
                                           "topology")},
        # RQ1: composition
        "components": comp.get("components"), "sources": len(comp.get("sources") or []) if comp else None,
        "destinations": len(comp.get("destinations") or []) if comp else None,
        "dependencies": comp.get("dependencies"), "bindings": comp.get("bindings"),
        "design_components": score.get("design_tasks"),
        **{k: score.get(k) for k in DESIGN_KEYS},
        "missing_components": len(score.get("missing_components") or []) if score else None,
        "extra_components": len(score.get("unmatched_design_tasks") or []) if score else None,
        "split_components": len(score.get("split_tasks") or []) if score else None,
        "merged_components": len(score.get("merged_tasks") or []) if score else None,
        "missing_dependencies": len(score.get("missing_dependencies") or []) if score else None,
        "extra_dependencies": score.get("dependency_extra"),
        "missing_bindings": len(score.get("missing_bindings") or []) if score else None,
        "extra_bindings": len(score.get("extra_bindings") or []) if score else None,
        "composition_operations": trace.get("operations"),
        "failed_operations": trace.get("failed_operations"),
        "composition_repaired": trace.get("repaired"),
        # RQ2: implementation
        "attempts": impl.get("attempts"), "repairs": impl.get("repairs"),
        "first_passing_attempt": impl.get("first_passing_attempt"),
        "initial_valid": impl.get("initial_valid"), "final_valid": impl.get("final_valid"),
        "recovered_by_repair": impl.get("recovered_by_repair"),
        "final_failure_reason": impl.get("final_failure_reason") or "",
        **{f"feedback_{k}": (impl.get("feedback_categories") or {}).get(k, 0) if impl else None
           for k in FAILURE_CATEGORIES},
        "public_validation": impl.get("public_validation"),
        # RQ2: execution
        "dagster_loaded": ex.get("loaded"), "dagster_completed": ex.get("completed"),
        "failed_components": ";".join(ex.get("failed_components") or []),
        "outputs_produced": len(ex.get("outputs_produced") or []) if ex else None,
        "outputs_checked": ex.get("outputs_checked"), "outputs_valid": ex.get("outputs_valid"),
        "outputs_compared": ex.get("outputs_compared"),
        "outputs_matching_reference": ex.get("outputs_matching_reference"),
        "execution_failure_stage": ex.get("failure_stage"),
        "definition_graph_score": ex.get("definition_graph_score"),
        **{f"stage_{k}": stages.get(k) for k in EVALUATION_STAGES},
        **{k: scores.get(k) for k in SCORE_KEYS},
        # RQ2: static diagnostics of the submission (ServiceCraft's static stage)
        **{f"static_{k}": static.get(k) for k in STATIC_KEYS},
        # RQ3: tokens, cost and time
        "requests": usage.get("requests"), "failed_requests": usage.get("failed_requests"),
        "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
        "cached_input_tokens": usage.get("cached_input_tokens"), "reasoning_tokens": usage.get("reasoning_tokens"),
        "usage_unavailable_requests": usage.get("usage_unavailable"),
        "model_seconds": usage.get("request_seconds"),
        # work performed
        **{k: counters.get(k) for k in COUNTER_KEYS},
        "cost_usd": usage.get("cost_usd"), "cost_unavailable_requests": usage.get("cost_unavailable"),
        **{f"{p}_{k}": (phases.get(p) or {}).get(k) for p in ("composition", "implementation", "repair")
           for k in ("requests", "input_tokens", "output_tokens", "model_seconds")},
        **{f"{p}_seconds": seconds.get(p) for p in ("composition", "implementation", "repair", "bundle",
                                                    "validation", "execution")},
        "generation_seconds": stage_s.get("pipeline_generation"),
        "validation_and_repair_seconds": stage_s.get("pipeline_validation"),
        "compilation_seconds": stage_s.get("node_compilation"),
        "bundle_seconds": impl.get("bundle_seconds"),
        "construction_seconds": construction_seconds(row),
        "dagster_step_seconds": ex.get("dagster_step_seconds"),
        "evaluation_seconds": ex.get("evaluation_seconds"),
        "wall_seconds": (row.get("budget") or {}).get("elapsed_s"),
    }
    return out


def export_rows(roots: list[Path]) -> list[dict]:
    return [run_row(root.name, row) for root in roots for row in episodes(root)]


def to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def to_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, default=str) + "\n" for r in rows)
