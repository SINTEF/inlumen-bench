"""What each column means: the dashboard's tables (under the names they show) and the
per-run export (group, unit and definition), linked both ways.

`bench export` writes these next to the data (`<name>.columns.csv`) and the
dashboard offers them as a download and a table. A test checks that every exported
column is described here, so the two cannot drift apart.
"""

from __future__ import annotations

import csv
import io

from .metrics import FAILURE_CATEGORIES, STATIC_TOOLS

# (column, group, unit, description)
_C: list[tuple[str, str, str, str]] = []


def _add(group: str, rows: list[tuple[str, str, str]]) -> None:
    _C.extend((name, group, unit, text) for name, unit, text in rows)


SC = "ServiceCraft"
_add("Run", [
    ("experiment", "", "Experiment (folder name)."),
    ("run", "", "Run identifier: case, method, model, composition source, condition and replicate."),
    ("phase", "", "composition (RQ1) or implementation (RQ2: implementation and execution)."),
    ("case", "", "ServiceCraft case identifier."),
    ("case_number", "", "First three digits of the case identifier."),
    ("method", "", "inlumen (inLUMEN's agents) or direct (one-request baseline)."),
    ("model", "", "Model as endpoint profile:model name."),
    ("replicate", "index", "Replicate number, from 0."),
    ("condition", "", "Experimental condition in words, e.g. 'contracts · repair ×3', 'ungrounded', 'baseline'."),
    ("composition_source", "", "Where the composition came from: 'prompt' (composed in this run), 'reference' "
                               "(ServiceCraft's reference composition) or the composition run it was taken from."),
    ("status", "", "Run status: designed or evaluated (finished), generated (waiting for scoring), "
                   "generation_failed, infra_error, running or pending."),
    ("outcome", "", "scored, failed (counts as a failure), infra (left out of rates) or pending."),
    ("error", "", "Error message of a failed run."),
    ("failure_category", "", "Category of the failure, by ordered rules on the message: "
                             + ", ".join(FAILURE_CATEGORIES) + "."),
])
_add("Condition", [
    ("grounded", "bool", "Composition: inLUMEN's metamodel-grounded design agent (true) or one plain model "
                         "request without tools, catalog or validation (false)."),
    ("artifact_contracts", "bool", "Implementation: components declare their output artifacts (true) or only "
                                   "components and dependencies are given (false)."),
    ("repair", "bool", "Implementation: inLUMEN's repair loop on validation failures is on."),
    ("max_repair_attempts", "count", "Maximum repair attempts."),
    ("sandbox", "", "Where inLUMEN runs generated code while validating (docker)."),
    ("max_cost_usd", "USD", "Cost limit per run."),
    ("max_seconds", "s", "Wall-time limit per run."),
    ("max_requests", "count", "Model-request limit per run. A run that reaches any limit stops and fails "
                              "(category budget)."),
])
_add("Complexity of the reference composition", [
    ("ref_level", "", "low (≤ 4 components), medium (5–6) or high (≥ 7)."),
    ("ref_components", "count", "Components."),
    ("ref_dependencies", "count", "Dependencies between components."),
    ("ref_inputs", "count", "Input files read."),
    ("ref_outputs", "count", "Artifacts written."),
    ("ref_longest_chain", "count", "Components on the longest dependency chain."),
    ("ref_max_fan_out", "count", "Largest number of direct dependants of one component."),
    ("ref_max_fan_in", "count", "Largest number of direct dependencies of one component."),
    ("ref_branching_points", "count", "Components where the pipeline splits or joins."),
    ("ref_topology", "", "ServiceCraft's topology label (linear, fan_in, fan_out, parallel, ...)."),
])
_add("RQ1 composition", [
    ("components", "count", "Nodes in the generated composition graph (Sources and Destinations included)."),
    ("sources", "count", "Source nodes."),
    ("destinations", "count", "Destination nodes."),
    ("dependencies", "count", "Edges in the generated composition graph."),
    ("bindings", "count", "Parameter values set and files attached."),
    ("design_components", "count", "Components of the composition compared with the reference."),
    ("artifact_coverage", "0–1", "Reference artifacts written by some generated component / reference artifacts."),
    ("task_recall", "0–1", "Task coverage: reference components matched by at least one generated component "
                           "(a match writes at least one of the same artifacts) / reference components."),
    ("task_one_to_one", "0–1", "Reference components matched by exactly one generated component that matches "
                               "nothing else / reference components."),
    ("dependency_recall", "0–1", "Reference dependencies whose two ends map one to one and are connected in "
                                 "the composition / reference dependencies."),
    ("binding_recall", "0–1", "Required inputs: reference (component ← input file) bindings found on the mapped "
                              "component / reference bindings."),
    ("design_correct", "0/1", "Complete design correctness: every component maps one to one, no extra "
                              "components, every reference dependency found and no extra dependencies."),
    ("missing_components", "count", "Reference components with no match."),
    ("extra_components", "count", "Generated components that match no reference component."),
    ("split_components", "count", "Reference components matched by several generated components."),
    ("merged_components", "count", "Generated components matching several reference components."),
    ("missing_dependencies", "count", "Reference dependencies not found."),
    ("extra_dependencies", "count", "Dependencies between mapped components that the reference lacks."),
    ("missing_bindings", "count", "Reference bindings not found."),
    ("extra_bindings", "count", "Bindings on mapped components that the reference lacks."),
    ("composition_operations", "count", "Tool calls of inLUMEN's design agent."),
    ("failed_operations", "count", "Tool calls rejected by inLUMEN's checks."),
    ("composition_repaired", "bool", "The design agent's validation repair round ran."),
])
_add("RQ2 validation and repair inside the method", [
    ("attempts", "count", "Generation attempts; inLUMEN runs the pipeline on sample inputs after each "
                          "(validation executions)."),
    ("repairs", "count", "Repair attempts."),
    ("first_passing_attempt", "index", "First attempt that passed inLUMEN's validation."),
    ("initial_valid", "bool", "The first attempt passed inLUMEN's validation."),
    ("final_valid", "bool", "The last attempt passed inLUMEN's validation."),
    ("recovered_by_repair", "bool", "Failed at first, valid at the end."),
    ("final_failure_reason", "", "Last validation failure of a run that stayed invalid."),
    *[(f"feedback_{k}", "count", f"Validation failures of category '{k}' fed back to repair.")
      for k in FAILURE_CATEGORIES],
    ("public_validation", "", "The bench's public check of the final bundle (syntax, Dagster load, step "
                              "failures, missing mandatory artifacts); recorded only, never fed back."),
])
_add(f"RQ2 execution ({SC})", [
    ("dagster_loaded", "bool", "Dagster loaded the definitions (stage 3)."),
    ("dagster_completed", "bool", "The Dagster run completed (stage 4)."),
    ("failed_components", "", "Dagster steps that failed, ';'-separated."),
    ("outputs_produced", "count", "Files the pipeline wrote."),
    ("outputs_checked", "count", "Declared intermediate and final artifacts checked in stage 5."),
    ("outputs_valid", "count", "Artifacts that passed their contract (existence, format, columns, keys)."),
    ("outputs_compared", "count", "Declared outputs compared with private expected outputs (stage 6)."),
    ("outputs_matching_reference", "count", "Outputs matching the expected values within tolerance."),
    ("execution_failure_stage", "", "First stage that did not pass."),
    ("definition_graph_score", "0–1", "Matched asset names and edges / (expected + unexpected ones); "
                                      "needs the reference component names."),
    ("stage_submission", "", "Stage 1, submission files: pass, fail, not_reached, timeout or infra_error."),
    ("stage_static", "", "Stage 2, sources parse and static metrics are collected."),
    ("stage_definition_graph", "", "Stage 3, Dagster loads and the asset graph equals the reference graph "
                                   "(exact names and edges)."),
    ("stage_execution", "", "Stage 4, the job runs to completion and asset checks pass."),
    ("stage_contracts", "", "Stage 5, every declared artifact meets its contract."),
    ("stage_oracle", "", "Stage 6, outputs match the private expected outputs."),
    ("dagster_executable", "0/1", "Execution succeeded with task coverage."),
    ("output_contract_valid", "0/1", "Stage 5 passed."),
    ("behaviorally_correct", "0/1", "Stage 6 passed."),
    ("complete_success", "0/1", "All six stages passed."),
])
_static = [
    ("static_files", "count", "Python files measured (the whole submission)."),
    ("static_physical_lines", "count", "Physical source lines."),
    ("static_syntax_valid", "bool", "Every file parses."),
]
for tool in STATIC_TOOLS:
    _static += [(f"static_{tool}_findings", "count", f"{tool} findings."),
                (f"static_{tool}_flagged_lines", "count", f"Distinct lines with {tool} findings."),
                (f"static_{tool}_line_fraction", "0–1", f"Lines with {tool} findings / physical lines "
                                                         "(lower is better).")]
_static += [
    *[(f"static_pylint_{k}", "count", f"pylint findings of type {k}.")
      for k in ("error", "warning", "refactor", "convention")],
    *[(f"static_bandit_{k}", "count", f"bandit findings of severity {k}.") for k in ("high", "medium", "low")],
    ("static_radon_blocks", "count", "Functions, methods and classes measured by radon."),
    ("static_radon_mean_cc", "", "Mean cyclomatic complexity per block."),
    ("static_radon_max_cc", "", "Highest cyclomatic complexity of a block."),
    ("static_radon_mi", "0–100", "Mean maintainability index of the files (higher is better)."),
    ("static_radon_mi_01", "0–1", "The same divided by 100."),
]
_add(f"RQ2 static diagnostics ({SC} stage 2)", _static)
_add("RQ3 tokens and cost", [
    ("requests", "count", "Model requests that returned (all phases)."),
    ("failed_requests", "count", "Model requests that failed."),
    ("input_tokens", "tokens", "Input tokens, as reported by the provider (failed runs and repairs included)."),
    ("output_tokens", "tokens", "Output tokens, as reported by the provider."),
    ("cached_input_tokens", "tokens", "Part of the input tokens served from the provider's cache, when reported."),
    ("reasoning_tokens", "tokens", "Part of the output tokens spent on reasoning, when reported."),
    ("usage_unavailable_requests", "count", "Requests whose provider reported no token usage."),
    ("cost_usd", "USD", "Sum of provider-reported request costs; local models report none."),
    ("cost_unavailable_requests", "count", "Requests without a reported cost."),
])
_add("RQ3 work performed", [
    ("code_producing_requests", "count", "Requests that produce code (implementation and repair)."),
    ("design_requests", "count", "Requests of the composition step."),
    ("naming_requests", "count", "Requests of inLUMEN's naming pass (artifact names per component)."),
    ("node_generations", "count", "Components whose code package was generated."),
    ("design_repairs", "count", "Composition repair rounds."),
])
_phase = []
for p in ("composition", "implementation", "repair"):
    _phase += [(f"{p}_requests", "count", f"Model requests in the {p} phase (requests are assigned by purpose: "
                                       "inLUMEN's repair requests count as repair)."),
               (f"{p}_input_tokens", "tokens", f"Input tokens in the {p} phase."),
               (f"{p}_output_tokens", "tokens", f"Output tokens in the {p} phase."),
               (f"{p}_model_seconds", "s", f"Summed latency of the {p} phase's model requests.")]
_add("RQ3 by phase", _phase)
_add("RQ3 time", [
    ("model_seconds", "s", "Summed latency of all model requests."),
    ("composition_seconds", "s", "Wall time of the composition phase."),
    ("implementation_seconds", "s", "Wall time of code generation; for inLUMEN this includes its own "
                                    "validation and repair attempts (see the inLUMEN timings below)."),
    ("repair_seconds", "s", "Wall time of a separate repair phase (methods that repair outside code "
                            "generation; empty for inLUMEN and direct)."),
    ("bundle_seconds", "s", "Wall time of assembling the Dagster bundle (no model)."),
    ("validation_seconds", "s", "Wall time of the public validation."),
    ("execution_seconds", "s", "Wall time of ServiceCraft's evaluation, from the run's events."),
    ("generation_seconds", "s", "inLUMEN's own timing: pipeline generation."),
    ("validation_and_repair_seconds", "s", "inLUMEN's own timing: validation and repair."),
    ("compilation_seconds", "s", "inLUMEN's own timing: node compilation."),
    ("construction_seconds", "s", "Construction time: composition + implementation + repair + bundle + "
                                  "validation."),
    ("dagster_step_seconds", "s", "Sum of Dagster step times in ServiceCraft's execution."),
    ("evaluation_seconds", "s", "ServiceCraft's evaluation time for the run."),
    ("wall_seconds", "s", "Wall time of the run in the bench (limits apply to this)."),
])

COLUMNS: dict[str, dict] = {name: {"group": group, "unit": unit, "description": text}
                            for name, group, unit, text in _C}

# ---- the dashboard's tables, under the names they show ------------------------------
# One entry per table: (column as shown, runs-CSV columns it comes from, unit, description).
# Rates count every finished run (scored or failed); infrastructure errors and pending
# runs are left out, as ServiceCraft recommends.

_GROUP = [
    ("method", "method", "", "inlumen (inLUMEN's agents) or direct (one-request baseline)."),
    ("model", "model", "", "Model as endpoint profile:model name."),
    ("compositions", "composition_source", "", "Which compositions were implemented: reference (ServiceCraft's "
                                               "reference composition) or those of a composition experiment."),
    ("condition", "condition", "", "Experimental condition, e.g. 'grounded', 'contracts · repair ×3', 'baseline'."),
    ("finished", "outcome", "runs", "Finished runs (scored or failed) of all scheduled runs. Rates in the row use "
                                    "the finished runs as denominator unless said otherwise."),
    ("failed", "outcome", "runs", "Runs that failed before producing a result (generation failures). They count "
                                  "as failures in every rate."),
    ("infra", "outcome", "runs", "Runs lost to infrastructure errors (provider, Docker, evaluator). Left out of "
                                 "every rate."),
]
_RATE = "share of finished runs"

TABLES: dict[str, list[tuple[str, str, str, str]]] = {
    "Composition (RQ1)": [
        *_GROUP,
        ("valid composition", "status", "%", f"{_RATE.capitalize()} whose composition passed the structural "
         "check: every task writes an output, every mandatory output is written by exactly one task, every input "
         "is a known input file or another task's output, every input file is read, no cycles."),
        ("outputs covered", "artifact_coverage", "%", "Mean, over valid compositions, of the share of the "
         "reference composition's outputs (artifact files) that the composition writes."),
        ("components found", "task_recall", "%", "Mean, over valid compositions, of the share of reference "
         "components matched by at least one component. A component matches a reference component when they "
         "write at least one of the same files (provisional rule)."),
        ("one-to-one", "task_one_to_one", "%", "Mean, over valid compositions, of the share of reference "
         "components matched by exactly one component that matches nothing else (not split, not merged)."),
        ("dependencies found", "dependency_recall", "%", "Mean, over valid compositions, of the share of reference "
         "dependencies whose two ends match one-to-one and are connected in the composition."),
        ("bindings found", "binding_recall", "%", "Mean, over valid compositions, of the share of the reference's "
         "(component, input file) pairs found: a matched component reads the same input file."),
        ("fully correct", "design_correct", "%", "Share of VALID compositions that are fully correct: every "
         "reference component matches one-to-one, same number of components, every reference dependency found "
         "and no extra dependency. Bindings are not part of it. Multiply by 'valid composition' for the share of "
         "finished runs."),
        ("operations / run", "composition_operations", "count", "Mean graph operations (tool calls) of the design "
         "agent per run. Ungrounded: 1 (one request)."),
    ],
    "Implementation (RQ2)": [
        *_GROUP,
        ("valid at first attempt", "initial_valid", "%", "Share of runs with a validation trace whose first "
         "attempt passed the method's own validation (inLUMEN runs the pipeline on sample inputs in its sandbox). "
         "Empty for direct, which has no validation."),
        ("valid at the end", "final_valid", "%", "Share of runs with a validation trace whose last attempt passed "
         "the method's own validation."),
        ("recovered by repair", "recovered_by_repair", "%", "Of the runs whose first attempt failed validation, "
         "the share that passed it at the end."),
        ("attempts / run", "attempts", "count", "Mean attempts per run: the first generation plus each repair."),
    ],
    "Execution (RQ2)": [
        *_GROUP,
        ("Dagster loaded", "dagster_loaded", "%", f"{_RATE.capitalize()} whose Dagster definitions loaded in "
         "ServiceCraft (stage 3: definitions loaded and an execution plan built). Runs that failed before "
         "ServiceCraft count as not loaded."),
        ("executable", "dagster_executable", "%", f"{_RATE.capitalize()} that Dagster ran to completion in "
         "ServiceCraft with every expected task executed (task coverage); a failed asset check also fails it "
         "(stage 4, execution)."),
        ("outputs valid", "output_contract_valid", "%", f"{_RATE.capitalize()} whose intermediate and final "
         "outputs all passed ServiceCraft's output contracts: existence, size, format, required CSV "
         "columns and row counts, finite JSON and required keys (stage 5, contracts)."),
        ("outputs as expected", "behaviorally_correct", "%", f"{_RATE.capitalize()} whose outputs match "
         "ServiceCraft's private expected outputs within atol = rtol = 1e-6, the tolerance of every case "
         "(stage 6, oracle)."),
        ("complete success", "complete_success", "%", f"{_RATE.capitalize()} that passed all six ServiceCraft "
         "stages: submission, static, definition graph, execution, contracts, oracle."),
        ("mean ± std", "complete_success", "%", "Complete success computed per replicate (over the cases), then "
         "mean and standard deviation across replicates."),
        ("pass@1", "complete_success", "%", "Unbiased pass@1 per case over its replicates (Chen et al., 2021), "
         "averaged over cases."),
        ("pass@k", "complete_success", "%", "Same as pass@1 for k = the number of replicates (pass@3 with 3): "
         "the chance that at least one of k runs of a case succeeds."),
    ],
    "Cost and time (RQ3)": [
        *_GROUP,
        ("cost", "cost_usd", "USD", "Total cost of all runs, failed ones included. '+ ?' when some requests "
         "reported no cost (local models report none)."),
        ("cost / run", "cost_usd", "USD", "Total cost divided by the scheduled runs."),
        ("cost / success", "cost_usd", "USD", "Total cost divided by the runs with complete success."),
        ("cost / valid composition", "cost_usd", "USD", "Composition experiments: total cost divided by the runs "
         "with a valid composition."),
        ("tokens / run", "input_tokens, output_tokens", "tokens", "Mean input plus output tokens per run, failed "
         "runs included."),
        ("tokens / success", "input_tokens, output_tokens", "tokens", "All tokens divided by the runs with "
         "complete success."),
        ("tokens / valid composition", "input_tokens, output_tokens", "tokens", "Composition experiments: all "
         "tokens divided by the runs with a valid composition."),
        ("median construction", "construction_seconds", "s", "Median time to build the pipeline: composition, "
         "implementation, repair, bundling and the method's validation. Dagster execution is not included."),
        ("median execution", "evaluation_seconds", "s", "Median time of ServiceCraft's evaluation of a run "
         "(including the Dagster run)."),
    ],
    "By complexity (RQ3)": [
        ("method", "method", "", "As above."),
        ("condition", "condition", "", "As above."),
        ("complexity", "ref_level", "", "Complexity of the reference composition by its number of components: "
                                        "low ≤ 4, medium 5–6, high ≥ 7."),
        ("runs", "run", "runs", "Scheduled runs at this level."),
        ("success", "complete_success", "%", "Implementation experiments: as 'complete success'."),
        ("valid composition", "status", "%", "Composition experiments: as 'valid composition'."),
        ("tokens / run", "input_tokens, output_tokens", "tokens", "As above."),
        ("tokens / success", "input_tokens, output_tokens", "tokens", "As above."),
        ("tokens / valid composition", "input_tokens, output_tokens", "tokens", "As above."),
        ("cost / run", "cost_usd", "USD", "As above."),
        ("median construction", "construction_seconds", "s", "As above."),
    ],
    "Runs": [
        ("run", "run", "", "Run identifier: case, method, model, composition source, condition and replicate."),
        ("status", "status", "", "Run status."),
        ("case", "case_number", "", "Case number."),
        ("method", "method", "", "As above."),
        ("model", "model", "", "As above."),
        ("condition", "condition", "", "As above."),
        ("replicate", "replicate", "index", "Replicate number, from 0."),
        ("components found", "task_recall", "%", "This run's share of reference components found."),
        ("dependencies found", "dependency_recall", "%", "This run's share of reference dependencies found."),
        ("operations", "composition_operations", "count", "This run's graph operations (tool calls)."),
        ("attempts", "attempts", "count", "This run's attempts: first generation plus repairs."),
        ("executable", "dagster_executable", "yes/no", "As 'executable', for this run."),
        ("outputs valid", "output_contract_valid", "yes/no", "As 'outputs valid', for this run."),
        ("outputs as expected", "behaviorally_correct", "yes/no", "As 'outputs as expected', for this run."),
        ("complete success", "complete_success", "yes/no", "As 'complete success', for this run."),
        ("cost", "cost_usd", "USD", "This run's cost."),
        ("time", "wall_seconds", "s", "This run's wall time in the bench (the limits apply to it)."),
        ("error", "error", "", "Error message of a failed run."),
    ],
}

# Runs-CSV column -> the names the dashboard tables show it under.
SHOWN_AS: dict[str, list[str]] = {}
for _rows in TABLES.values():
    for _label, _sources, _unit, _text in _rows:
        for _src in (s.strip() for s in _sources.split(",")):
            if _label != _src and _label not in SHOWN_AS.setdefault(_src, []):
                SHOWN_AS[_src].append(_label)


def table_help(table: str) -> dict[str, str]:
    """Shown column name -> description, for one dashboard table."""
    return {label: text for label, _src, _unit, text in TABLES[table]}


def describe_table(table: str, shown: list[str] | None = None) -> list[dict]:
    rows = TABLES[table]
    return [{"column": label, "description": text, "unit": unit, "in the runs CSV": src}
            for label, src, unit, text in rows if shown is None or label in shown]


def describe(names: list[str]) -> list[dict]:
    """Descriptions of the given runs-CSV columns, in their order."""
    return [{"column": n, **COLUMNS.get(n, {"group": "", "unit": "", "description": ""}),
             "in the dashboard tables": ", ".join(SHOWN_AS.get(n, []))} for n in names]


FIELDS = ["where", "column", "description", "unit", "group", "in the runs CSV", "in the dashboard tables"]


def to_csv(names: list[str]) -> str:
    """Every dashboard table column (under its shown name), then every runs-CSV column."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    for table in TABLES:
        writer.writerows({"where": f"dashboard table: {table}", **row} for row in describe_table(table))
    writer.writerows({"where": "runs CSV", **row} for row in describe(names))
    return buffer.getvalue()
