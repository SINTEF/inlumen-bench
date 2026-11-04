"""Design scores against ServiceCraft's reference task graph (operator side only).

ServiceCraft has no reviewed semantic task mapping yet, so this uses a simple,
deterministic and inspectable rule: a generated task matches a reference task
when it writes at least one of the same artifacts. Mandatory artifact names are
given in the prompt, so this anchors tasks without any judgement calls. The
scores are labelled provisional until a reviewed mapping procedure exists.

The reference graph is read here, after generation; it is never shown to a method.
"""

from __future__ import annotations

from .design import edges

METHOD = "artifact-anchored@1 (provisional)"


def score(design: dict, reference: dict) -> dict:
    ref_outputs = {t["id"]: set(t["outputs"]) for t in reference["tasks"]}
    gen_outputs = {t["id"]: set(t.get("outputs", [])) for t in design.get("tasks", [])}

    matches = {g: sorted(r for r, routs in ref_outputs.items() if gouts & routs)
               for g, gouts in gen_outputs.items()}
    ref_to_gen: dict[str, list[str]] = {r: [] for r in ref_outputs}
    for g, refs in matches.items():
        for r in refs:
            ref_to_gen[r].append(g)
    one_to_one = {r: gs[0] for r, gs in ref_to_gen.items() if len(gs) == 1 and len(matches[gs[0]]) == 1}

    ref_edges = {(s, d) for s, d, _ in edges(reference)}
    gen_edges = {(s, d) for s, d, _ in edges(design)}
    gen_to_ref = {g: r for r, g in one_to_one.items()}
    # Reference edges whose two ends map one-to-one and are connected in the design.
    found = {(s, d) for s, d in ref_edges
             if s in one_to_one and d in one_to_one and (one_to_one[s], one_to_one[d]) in gen_edges}
    mapped_gen_edges = {(s, d) for s, d in gen_edges if s in gen_to_ref and d in gen_to_ref}
    extra = {(s, d) for s, d in mapped_gen_edges if (gen_to_ref[s], gen_to_ref[d]) not in ref_edges}

    ref_artifacts = set().union(*ref_outputs.values()) if ref_outputs else set()
    gen_artifacts = set().union(*gen_outputs.values()) if gen_outputs else set()
    n_ref, n_gen = len(ref_outputs), len(gen_outputs)
    result = {
        "method": METHOD,
        "reference_tasks": n_ref,
        "design_tasks": n_gen,
        "task_count_delta": n_gen - n_ref,
        "artifact_coverage": _ratio(len(ref_artifacts & gen_artifacts), len(ref_artifacts)),
        "task_recall": _ratio(sum(1 for gs in ref_to_gen.values() if gs), n_ref),
        "task_one_to_one": _ratio(len(one_to_one), n_ref),
        "split_tasks": sorted(r for r, gs in ref_to_gen.items() if len(gs) > 1),
        "merged_tasks": sorted(g for g, rs in matches.items() if len(rs) > 1),
        "unmatched_design_tasks": sorted(g for g, rs in matches.items() if not rs),
        "dependency_recall": _ratio(len(found), len(ref_edges)),
        "dependency_extra": len(extra),
        "missing_dependencies": sorted(f"{s}->{d}" for s, d in ref_edges - found),
    }
    result["design_correct"] = int(
        len(one_to_one) == n_ref == n_gen and len(found) == len(ref_edges) and not extra
        and len(mapped_gen_edges) == len(gen_edges))
    return result


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None
