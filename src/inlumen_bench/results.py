"""Aggregated results per method × model × track, shared by the CLI and dashboard.

Denominators are explicit: every scheduled episode counts. Generation failures
count as failures; pending episodes and infrastructure errors are reported apart
and left out of rates, as ServiceCraft recommends.
"""

from __future__ import annotations

import statistics
from math import comb
from pathlib import Path

from .runner import Experiment, episode_state

SCORE_KEYS = ("dagster_executable", "output_contract_valid", "behaviorally_correct", "complete_success")
DESIGN_KEYS = ("artifact_coverage", "task_recall", "task_one_to_one", "dependency_recall", "design_correct")


def episodes(root: Path) -> list[dict]:
    exp = Experiment.load(root)
    rows = []
    for ep in exp.episodes():
        state = episode_state(root / "episodes" / ep["id"]) or {**ep, "status": "pending"}
        rows.append({**ep, **state})
    return rows


def _outcome(row: dict) -> str:
    """scored | failed (generation) | pending | infra."""
    status = row.get("status")
    if status == "evaluated":
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


def summarize(root: Path, group_by: tuple[str, ...] = ("method", "model", "track")) -> list[dict]:
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
        for score in SCORE_KEYS:
            hits = sum(1 for r in decided if (r.get("scores") or {}).get(score) == 1)
            cell[score] = round(hits / len(decided), 4) if decided else None

        # Success rate per repetition index (over cases), to show spread across repetitions.
        by_rep: dict[int, list[int]] = {}
        for r in decided:
            by_rep.setdefault(r["repetition"], []).append(int((r.get("scores") or {}).get("complete_success") == 1))
        rep_rates = [sum(v) / len(v) for v in by_rep.values() if v]
        cell["success_mean"] = round(statistics.mean(rep_rates), 4) if rep_rates else None
        cell["success_std"] = round(statistics.stdev(rep_rates), 4) if len(rep_rates) > 1 else None

        # pass@k per case over its repetitions, averaged over cases.
        by_case: dict[str, list[int]] = {}
        for r in decided:
            by_case.setdefault(r["case_id"], []).append(int((r.get("scores") or {}).get("complete_success") == 1))
        for k in sorted({1, exp.repetitions}):
            values = [p for v in by_case.values() if (p := pass_at_k(len(v), sum(v), k)) is not None]
            cell[f"pass@{k}"] = round(statistics.mean(values), 4) if values else None

        costs = [(r.get("usage") or {}).get("cost_usd") or 0 for r in members]
        successes = sum(1 for r in decided if (r.get("scores") or {}).get("complete_success") == 1)
        cell["cost_usd"] = round(sum(costs), 4)
        cell["cost_incomplete"] = any((r.get("usage") or {}).get("cost_unavailable") for r in members)
        cell["cost_per_success"] = round(sum(costs) / successes, 4) if successes else None
        cell["output_tokens"] = sum((r.get("usage") or {}).get("output_tokens") or 0 for r in members)
        cell["input_tokens"] = sum((r.get("usage") or {}).get("input_tokens") or 0 for r in members)
        times = [(r.get("budget") or {}).get("elapsed_s") for r in members if (r.get("budget") or {}).get("elapsed_s")]
        cell["median_generation_s"] = round(statistics.median(times), 1) if times else None
        designs = [r["design_score"] for r in members if r.get("design_score")]
        for k in DESIGN_KEYS:
            values = [d[k] for d in designs if d.get(k) is not None]
            cell[f"design_{k}"] = round(statistics.mean(values), 4) if values else None
        out.append(cell)
    return out
