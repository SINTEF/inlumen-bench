# Evaluation protocol (draft for review)

Status: **draft, 9 Nov 2026.** Written from the code as it runs (inlumen-bench 0.3.4,
inlumen-agents 1.0.0b6.dev3, ServiceCraft evaluator as in the local corpus). Every
definition below says where the number comes from.

Legend for the metric tables: ✅ recorded and in the export (one CSV row per run);
◐ recorded but not split out in the export; ✗ not collected. Every exported column is
described in `<export>.columns.csv` (written by `bench export`, and downloadable from
the dashboard), generated from `src/inlumen_bench/columns.py`.

## 1. Corpus

- **ServiceCraft**, 100 cases. Each case has a natural-language prompt (`prompt.txt`),
  input files, and a *reference composition* (`task_graph.json`). Methods see only the
  prompt and the inputs. The reference graph, the artifact contracts and the expected
  outputs (oracles) are private to the evaluator.
- The prompt names the **mandatory artifacts** (output files) the pipeline must write.
- **Complexity** is measured on the reference composition, never on a generated one:
  components, dependencies, inputs, outputs, longest dependency chain, maximum fan-in
  and fan-out, branching points, ServiceCraft's topology label, and a level:
  low ≤ 4 components, medium 5–6, high ≥ 7.

## 2. Research questions and phases

Pipeline construction has three phases. Each run (an *episode*) is one
case × method × model × condition × replicate.

| RQ | Phase | Question |
|---|---|---|
| RQ1 | Composition | How well do the methods turn requirements into a composition graph, and what does metamodel grounding contribute? |
| RQ2 | Implementation and execution | How often does the composition become code that runs and produces correct outputs, and what do artifact contracts and repair contribute? |
| RQ3 | Cost | What does construction cost in tokens, time and money, and how does it grow with complexity? |

## 3. Methods

**inLUMEN** (`inlumen`, the agents from the `inlumen-agents` package):
- *Composition, grounded* (default): inLUMEN's design agent builds the graph through
  tools that are checked against inLUMEN's metamodel and catalog; the result is
  validated, with one repair round.
- *Composition, ungrounded* (ablation): one plain model request returns the graph as
  JSON, with no tools, catalog or validation.
- In both, a **naming pass** (one extra request) maps each component to the input
  files it reads and the artifacts it writes. Then a structural check runs.
- *Implementation*: inLUMEN's code generator writes one Task package per component,
  with planning, validation on sample inputs (in Docker) and repair. The packages are
  wrapped as Dagster assets.
  - *With artifact contracts* (default): every component declares its output artifacts
    (file names, kinds, ports).
  - *Without artifact contracts* (ablation): components and dependencies only; output
    ports get generic names. The prompt still names the mandatory outputs.
  - *Repair*: up to N repair attempts on validation failures (N = 3 in the current plan).

**Direct baseline** (`direct`): one request for the whole Dagster bundle, given the
prompt, the same composition as inLUMEN (as JSON) and the runtime contract. No repair,
no validation feedback.

Both methods get the same runtime contract, the same composition and the same limits.
Public validation (syntax, Dagster load, step failures, missing mandatory artifacts) is
run on every bundle and **recorded only**; it never feeds back into the direct method.

## 4. Experimental design

### Factors

| Phase | Factor | Levels |
|---|---|---|
| Composition | Grounding | grounded, ungrounded |
| Implementation | Method | inLUMEN, direct |
| Implementation (inLUMEN only) | Artifact contracts | on, off |
| Implementation (inLUMEN only) | Repair | on, at most 3 attempts (fixed) |
| Implementation | Composition given | ServiceCraft's reference composition (main); a generated composition (end to end) |
| Both | Model | Qwen3.8, gpt-oss 20B, Devstral (all local, Ollama, 64k context) |
| Both | Replicate | 3 (the grounded composition run has 5; replicates 0–2 are compared) |

The direct baseline runs once per composition, model and replicate (it has no
contracts or repair to vary). Runs are executed in random order per model; seed 0.

### Limits per run

| Phase | Max cost | Max wall time | Max model requests |
|---|---|---|---|
| Composition | $0.50 | 1,800 s | 40 |
| Implementation | $2.00 | 1,800 s | 60 |

A limit is checked before each request (an in-flight request may overshoot). A run
that hits a limit stops and counts as a failure (category `budget`).

### Experiments

| Experiment | Phase | Design | Runs | Status |
|---|---|---|---|---|
| `composition-20261106-1651` | Composition | 100 cases × 3 models × 5 replicates, grounded only | 1,500 | done |
| `composition-gpt-oss-120b` | Composition | 100 cases × gpt-oss 120B × 5, grounded only | 500 | done |
| `composition-ungrounded` | Composition | 100 cases × 3 models × 3 replicates, **ungrounded**; same limits and options | 900 | running (started 9 Nov 21:16 UTC) |
| `implementation-reference30` | Implementation, **main (RQ2)** | 30 stratified cases from the **reference composition** × 3 models × 3 replicates × (inLUMEN with contracts, without contracts, direct) | 810 | queued after the composition run |
| `implementation-sample10` | Implementation, **end to end** | 10 stratified cases from a **generated** (grounded) composition × 3 models × 3 replicates × the same three setups | 270 | done |
| `implementation-from-composition-20261106-1651` | Implementation | every valid composition, 5 replicates, repair ×5 | 19,740 | stopped, not used |

**Main implementation experiment (RQ2).** Code is generated from ServiceCraft's
reference composition, so implementation is measured apart from composition errors and
all six evaluator stages apply. 30 cases: 10 per complexity level. They include the 10
end-to-end cases; the other 20 are drawn at random (seed 0) per level, cycling through
the level's topologies so each is covered. Script and picks: `create_final.py`,
`sampling.json` in the experiment folder.

**End-to-end experiment.** Code is generated from a composition made in the grounded
composition run, picked where the composition score marks it correct
(`design_correct = 1`). Stratified by complexity: 3 low (fan-out, fan-in, linear), 4
medium (linear, parallel, linear→fan-out, linear→fan-in) and 3 high (fan-in-fan-out
with ≤ 8, 9 and ≥ 10 components); one case at random per stratum, then one of its
correct compositions at random (seed 0). Script and picks: `sample_experiment.py`,
`sampling.json`. Generated components carry their own names, so stage 3 (graph check)
cannot pass and complete success is 0 by construction; this experiment reports stages
4–6 (execution, output contracts, output values).

## 5. Metrics

### 5.1 Correctness (RQ2), from ServiceCraft's evaluator

ServiceCraft runs each bundle in an isolated container (no network, read-only, limited
CPU, memory and time) in six stages; each stage is pass, fail, not reached, timeout or
infrastructure error. An unreached stage never passes.

| Stage | What it checks |
|---|---|
| 1 Submission | paths and Python sources are present and well formed |
| 2 Static | sources parse; static metrics are collected (see 5.8) |
| 3 Definition graph | Dagster loads the definitions; the asset graph is compared to the reference graph (**exact equality of asset names and edges**) |
| 4 Execution | the job runs to completion; asset checks pass |
| 5 Contracts | every declared intermediate and final artifact exists with the right format, required CSV columns and row counts, or required JSON keys |
| 6 Oracle | declared outputs match private expected files (JSON: exact keys and types; CSV: column and row order; numbers within `atol + rtol·|expected|`) |

| Metric | Definition | Status |
|---|---|---|
| `complete_success` | 1 only if all six stages pass | ✅ |
| `dagster_executable` | execution succeeded with task coverage | ✅ |
| `output_contract_valid` | stage 5 passed | ✅ |
| `behaviorally_correct` | stage 6 passed | ✅ |
| `stage_<name>` | pass / fail / not_reached / timeout / infra_error for each of the six stages | ✅ |
| `definition_graph_score` | matched nodes and edges over expected plus unexpected ones | ✅ |
| `outputs_checked`, `outputs_valid` | artifacts checked in stage 5, and how many passed | ✅ |
| `outputs_compared`, `outputs_matching_reference` | outputs compared in stage 6, and how many matched | ✅ |
| Success within budget | success reached without hitting a limit | implicit: a run that hits a limit is a failure, so every success is within budget |
| Intermediate vs final artifact correctness | stage 5 checks intermediate and final artifacts together | ◐ (not split in the export) |

> **Why the main RQ2 experiment uses the reference composition.** In `implementation-sample10`, stage 3 failed for
> **all 149 scored runs** (graph score 0.0), because the code is generated from a
> model-made composition whose components have their own names, while ServiceCraft
> requires the reference task names. `complete_success` is therefore 0 by construction
> in this setup, whatever the code quality. Stages 4–6 still run (ServiceCraft collects
> them after a graph failure). In the sample, inLUMEN with contracts passed 148 of 271
> artifact checks and matched 27 of 265 reference outputs; direct 60 of 106 and 13 of 99.
>
> **Decided:** RQ2 is measured from the reference composition
> (`implementation-reference30`); the end-to-end experiment reports stages 4–6 only.

### 5.2 Composition (RQ1), against the reference composition

Matching rule (`artifact-anchored@1`, provisional): a generated component matches a
reference component when it writes at least one of the same artifacts. Artifact names
are given in the prompt, so no judgement is involved. Labelled provisional until a
reviewed semantic mapping exists; a seeded random sample is also reviewed by hand in
the dashboard.

| Metric | Definition | Status |
|---|---|---|
| `artifact_coverage` | reference artifacts written by some generated component / all reference artifacts | ✅ |
| `task_recall` (task coverage) | reference components matched by at least one generated component / reference components | ✅ |
| `task_one_to_one` | reference components matched by exactly one generated component that matches nothing else / reference components | ✅ |
| `split_components`, `merged_components`, `missing_components`, `extra_components` | counts (lists are in the run folder) | ✅ |
| `dependency_recall` (dependency correctness) | reference dependencies whose two ends map one-to-one and are connected in the design / reference dependencies | ✅ |
| `extra_dependencies` | dependencies between mapped components that the reference does not have | ✅ |
| `binding_recall` (required parameters) | reference (component ← input file) bindings found on the mapped component / reference bindings | ✅ |
| `missing_bindings`, `extra_bindings` | counts | ✅ |
| `design_correct` (complete design correctness) | 1 if every component maps one-to-one, no extra components, every reference dependency is found and there are no extra dependencies | ✅ |
| Composition operations | tool calls of the design agent, failed ones, whether the repair round ran | ✅ |

Note: bindings currently cover input files only, not parameter values; parameter
values are recorded in the composition but not scored.

### 5.3 Validation and repair inside the method (RQ2)

From inLUMEN's own attempt trace (validation on sample inputs during generation).

| Metric | Definition | Status |
|---|---|---|
| `attempts`, `repairs` | generation attempts and repair attempts | ✅ |
| `initial_valid` | the first attempt passed inLUMEN's validation | ✅ |
| `final_valid` | the last attempt passed | ✅ |
| `first_passing_attempt` | index of the first passing attempt | ✅ |
| `recovered_by_repair` | initially failing and finally valid | ✅ |
| `feedback_<category>` | validation failures fed back, by category | ✅ |
| `public_validation` | the bench's public check of the final bundle (recorded only) | ✅ |

### 5.4 Tokens (RQ3)

| Metric | Definition | Status |
|---|---|---|
| Input and output tokens | per run and per phase (composition, implementation, repair), failed requests included | ✅ |
| `cached_input_tokens` | as reported by the provider | ✅ |
| `reasoning_tokens` | as reported by the provider | ✅ (Ollama reports 0) |
| Cache writes | — | ✗ (Ollama does not report it; OpenRouter does for some providers) |

### 5.5 Time (RQ3)

| Metric | Definition | Status |
|---|---|---|
| `construction_seconds` | wall time of composition + implementation + repair + bundle + validation | ✅ |
| `<phase>_seconds` | wall time per phase, from phase events | ✅ |
| `<phase>_model_seconds` | summed latency of model requests per phase | ✅ |
| Tool execution time | time in the design agent's tools and inLUMEN's sandbox runs | ✗ (included in phase wall time, not separated) |
| `evaluation_seconds`, `dagster_step_seconds` | ServiceCraft's evaluation, and the sum of Dagster step times | ✅ |

Wall times depend on the hardware and on the other runs sharing the GPUs: two runs in
parallel, one Ollama server per GPU (2 GPUs), one request at a time per server.

### 5.6 Cost (RQ3)

| Metric | Definition | Status |
|---|---|---|
| `cost_usd` | sum of the cost the provider reports per request (OpenRouter) | ✅ |
| `cost_unavailable_requests` | requests without a reported cost | ✅ |
| Estimated API cost | tokens × a fixed price table | ✗ |
| Cost per successful pipeline | cell cost / successes (all runs' cost, failed ones included) | ✅ (per cell) |

**Decided:** all models run locally, so cost is reported as tokens and time; no
estimated monetary cost.

### 5.7 Work performed (RQ3)

| Metric | Definition | Status |
|---|---|---|
| `requests`, `failed_requests` | model requests per run (and per phase) | ✅ |
| Code-producing requests | implementation + repair requests | ✅ (per phase) |
| `node_generations` | components whose package was generated | ✅ |
| Repairs | repair attempts | ✅ |
| Tool calls | design agent tool calls | ✅ (`composition_operations`) |
| Validation executions | inLUMEN runs the pipeline on sample inputs after every attempt, so this equals `attempts` | ✅ |

### 5.8 Static diagnostics (from ServiceCraft stage 2)

| Metric | Definition | Status |
|---|---|---|
| `static_<tool>_line_fraction` (flake8, pylint, bandit) | distinct source lines with findings / physical lines (lower is better) | ✅ |
| `static_<tool>_findings`, `static_<tool>_flagged_lines` | raw counts | ✅ |
| `static_pylint_<type>`, `static_bandit_<severity>` | findings by pylint type and bandit severity | ✅ |
| `static_radon_mi`, `static_radon_mi_01` | mean file maintainability index (0–100), and divided by 100 (higher is better) | ✅ |
| `static_radon_mean_cc`, `static_radon_max_cc`, `static_radon_blocks` | cyclomatic complexity per block | ✅ |
| `static_files`, `static_physical_lines` | size of what was measured | ✅ |

ServiceCraft measures every Python file of the submission. inLUMEN's bundle contains
the generated packages plus a deterministic Dagster harness written by the bench, while
the direct baseline is usually one file; compare static diagnostics with that in mind
(and with `static_files` and `static_physical_lines`).

### 5.9 Failures

Each failed run gets one category, from its error message, by ordered rules
(first match wins): budget, syntax, format (model reply not readable), provider,
dependency, contract, sandbox, deployment, runtime, other. Infrastructure errors are a
run status of their own. A composition that fails the structural check is a
composition failure.

| Your category | Bench category |
|---|---|
| Design | run status `generation_failed` in the composition phase |
| Generation | `format`, `syntax`, `provider` |
| Packaging | `deployment` |
| Dependency | `dependency` |
| Runtime | `runtime`, `sandbox` |
| Output | `contract` |
| Budget | `budget` |
| Infrastructure | run status `infra_error` |

The rules are regular expressions on messages, so a sample of categorised failures
should be checked by hand before reporting.

## 6. Aggregation

- A **cell** is method × model × composition source × condition.
- Every scheduled run counts. Generation failures count as failures. Pending runs and
  infrastructure errors are reported apart and left out of rates (as ServiceCraft
  recommends).
- Rates are over decided runs (scored + failed). Success is reported as the mean ± std
  of the per-replicate success rate, and as pass@1 and pass@k per case (Chen et al.,
  2021), averaged over cases.
- Cost and tokens include failed runs.

## 7. Scope decisions (deadline 12 Nov 2026)

1. RQ2 from the reference composition on 30 stratified cases, not all 100.
2. Ungrounded composition with 3 replicates, compared with replicates 0–2 of the
   grounded run.
3. Ungrounded compositions do not go through implementation.
4. Repair fixed at 3 attempts, no ablation without repair; first-attempt validity
   (`initial_valid`) is reported instead.
5. No estimated monetary cost (local models).
6. inLUMEN is used as released (1.0.0b6.dev3); unreadable model replies (invalid JSON)
   are reported as failures of category `format`.

## 8. Known limitations

- Component matching for RQ1 is provisional (artifact-anchored) until a reviewed
  semantic mapping exists; a hand-checked sample complements it.
- Failure categories come from rules on error messages; check a sample by hand.
- Wall times depend on shared hardware (two GPUs, two runs in parallel).
- The grounded composition run used bench 0.3.0 with inlumen-agents 1.0.0b6.dev1; the
  ungrounded run uses bench 0.3.3 with 1.0.0b6.dev3. Between them inLUMEN's design agent
  (`inlumen.design`) did not change (only code generation and the resolution of node
  output contracts did), and the bench's composition path changed only in request
  bookkeeping (per-server request slots, one extra event field). Both runs are scored
  with the same code (`bench recompute`).
