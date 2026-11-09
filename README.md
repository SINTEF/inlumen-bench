# inlumen-bench

A driver and dashboard for benchmarking **pipeline construction** on the
[ServiceCraft](https://github.com/aliduabubakari/servicecraft) corpus, across cases,
models, experimental conditions and replicates. Pipeline construction has three phases:
1. **Pipeline composition**: the task's requirements become a *composition graph*
   (components, Sources and Destinations, dependencies and bindings).
2. **Pipeline implementation**: the composition graph becomes *code artifacts*,
   bundled for Dagster.
3. **Pipeline execution**: ServiceCraft's own evaluator runs the bundle in Dagster
   and checks its outputs.

The `inlumen` method runs inLUMEN's own composition and implementation agents from
the [`inlumen-agents`](https://pypi.org/project/inlumen-agents/) package on PyPI (see
[Installing inLUMEN's agents](#installing-inlumens-agents)); `direct` is the baseline.
Every model request is recorded, and everything the research questions need is
stored per run (see [What is recorded](#what-is-recorded)) and can be exported as
one CSV row per run. Composition graphs are exported as inLUMEN Project JSON, which
opens with inLUMEN's Import button.

Terms: a **run** is one case × method × model × condition × replicate (its folder is
still called an *episode*); an **experiment** is a set of runs of one phase.

## Quick start with Docker
Only Docker is needed on the host.
```sh
docker compose up -d          # dashboard on http://localhost:8000 (builds the image the first time)
```
The dashboard is a [Streamlit](https://streamlit.io) app (`src/inlumen_bench/dashboard/app.py`).
In the dashboard:
1. **Corpus**: the dataset in the data folder is found automatically (or fetched with
   one click). The Cases tab lists the 100 cases with their complexity; select one to
   see its requirements, reference composition, components and inputs, or download it
   as inLUMEN Project JSON. The Evaluator tab prepares ServiceCraft's evaluator once
   per machine (about 7 minutes).
2. **Models**: add an endpoint (OpenRouter, Ollama, any OpenAI-compatible URL) and test it.
3. **Launch**, in numbered steps: the phase, the compositions and methods
   (implementation), the support mechanisms to compare (each one on, off, or both),
   models, cases, replicates and limits. The page shows how many runs that makes
   before you start.
4. **Experiments**: select one experiment to open it, or several to stop, export or
   delete them together. An experiment shows live progress and three tabs: results for
   its research question (composition, RQ1, or implementation and execution, RQ2),
   cost and time (RQ3, also by complexity), and its runs. In the runs table, select
   one run to see its composition (with the reference next to it), implementation
   attempts, execution, cost and files, or several runs to delete them (they become
   pending and run again on Resume). "Implement these compositions" on a composition
   experiment starts phase 2 from its valid compositions.

The same can be done from the command line with `scripts/bench-docker`, which takes
the same arguments as `bench` (see below):
```sh
scripts/bench-docker datasets fetch
scripts/bench-docker setup
scripts/bench-docker run pilot -c 016 --model openrouter:<model-id> -m inlumen -m direct -n 3
```
After changing the code, rebuild with `docker compose build` (or `docker build -t inlumen-bench:0.3.2 .`).

### Docker settings
There is one `docker-compose.yml` for a laptop and a server; what differs goes in a
`.env` file next to it (all optional):
```sh
BENCH_DATA_DIR=/Users/you/Documents/Datasets   # default ~/Documents/Datasets
BENCH_RUNS_DIR=/Users/you/bench-runs    # default ./runs; absolute path
BENCH_CONFIG_DIR=/Users/you/.config/inlumen-bench
BENCH_PORT=8000
OPENROUTER_API_KEY=sk-or-...            # passed into the container
```
How it works:
- The container uses the host's Docker engine through `/var/run/docker.sock`,
  because ServiceCraft runs every submission in its own container.
- The data and runs folders are mounted at the **same absolute paths** as on the
  host, since the host engine resolves those paths. Temporary folders go to
  `<runs>/.tmp` for the same reason.
- The dashboard is published on localhost only. It has no login and can start
  paid model runs, so do not expose it on a network. (Streamlit's startup log
  prints "Network" and "External" URLs; they are not reachable.)
- Local model servers on the host are reached as `host.docker.internal`; the
  `ollama` preset uses that automatically inside the container.

### On a shared Linux server
Add to `.env` on the server:
```sh
BENCH_NETWORK=host        # use the host's network
BENCH_UID=1008            # id -u
BENCH_GID=1008            # id -g
BENCH_KEEP_IMAGES=1       # if a Kubernetes kubelet shares the Docker engine
BENCH_PORT=8501
```
- `BENCH_NETWORK=host`: a model server listening only on `127.0.0.1` (such as a
  system Ollama) is reachable at `http://127.0.0.1:11434/v1`, and the dashboard itself
  listens on the server's localhost only.
- The containers run as root, because ServiceCraft's evaluator has to remove files its
  sandbox containers write as uid 65532; after every command and dashboard job, bench
  gives the corpus folders, runs and config back to `BENCH_UID:BENCH_GID`.
- `BENCH_KEEP_IMAGES=1`: a kubelet deletes unused images when the disk is over 85%
  full; bench keeps one idle container per evaluator image (`inlumen-bench-keep-*`)
  so they count as in use, and rebuilds missing ones.

Then `docker compose up -d --build` on the server, and from your own computer:
```sh
ssh -N -L 8501:127.0.0.1:8501 you@server   # dashboard on http://localhost:8501
```
For Ollama, give every model the same fixed context. Without one, a model gets either
Ollama's default (which silently cuts long prompts) or its full trained context (262k
tokens for some), which takes far more GPU memory. A copy with a fixed context reuses
the model's files:
```sh
curl -s localhost:11434/api/create -d '{"model": "qwen3.8-32k", "from": "qwen3.8:latest", "parameters": {"num_ctx": 32768}}'
```
An experiment runs each model's runs in random order, and by default at most one run
per model at a time on each server ("Runs per model at a time", `--runs-per-model`):
Ollama answers one request per model at a time, so more runs of one model would only
wait, and the wait would count as their time.

With several GPUs, start one Ollama server per GPU, so the same model can run on all of
them at once and no GPU waits while another model finishes:
```sh
scripts/ollama-gpus start    # a server per GPU on ports 11435, 11436, ...; also stop, status
```
They run as your user (no sudo), see one GPU each, hold one model at a time and read the
system Ollama's models (read only). They keep running after you log out, not after a
reboot. Then list them as the profile's servers for runs (Models page, or
`bench providers add ollama --preset ollama --endpoint http://127.0.0.1:11435/v1 --endpoint
http://127.0.0.1:11436/v1`), and set "Runs in parallel" (`--workers`) to the number of
servers. Models then run one after another, each on every GPU; when a server moves on to
the next model, or a model has no runs left, the bench unloads it there. Each run records
its server, and the dashboard shows it. Other users of the system Ollama (port 11434) can
still load models onto these GPUs.

For a model too big for one GPU (gpt-oss 120B, for example), the same script starts
one more server, after the per-GPU ones (port 11437 with two GPUs), that sees all GPUs and
splits the model across them. It keeps its models in your own folder, `~/ollama-models`
(`OLLAMA_GPUS_ALL_MODELS`), so you can download big models without sudo, and it starts
only if that folder exists:
```sh
mkdir -p ~/ollama-models && scripts/ollama-gpus start
OLLAMA_HOST=127.0.0.1:11437 ollama pull gpt-oss:120b
```
Give it its own profile (base URL `http://127.0.0.1:11437/v1`, one run in parallel, no
servers for runs). An idle server holds no GPU memory, but don't run the big model while
the per-GPU servers run other models: they share the same GPUs.

Without servers for runs, parallel runs go to different models instead: Ollama puts each
model on its own GPU when it fits, but the last model runs alone on one GPU. Set the
profile's parallel requests (per server) to at least the runs per model, or requests
queue in the bench. With hosted APIs, raise the runs per model.

Progress: the dashboard shows each running run's model, phase and request (with how
long it has been waiting), and the experiment's log is kept in
`runs/<experiment>/run.log`, with a summary line every 30 seconds.

## Native install
Python 3.11+ (or [uv](https://docs.astral.sh/uv/)) and a running Docker engine.
```sh
uv venv --python 3.12 .venv
uv pip install -e '.[test,inlumen]'   # inlumen: inLUMEN's agents, from PyPI (see below)
bench config set data_dir ~/Documents/Datasets
bench datasets fetch && bench datasets check
bench setup
bench serve                   # Streamlit dashboard on http://localhost:8000
```
The data folder is chosen in this order: `--data-dir`, `INLUMEN_BENCH_DATA_DIR`,
`data_dir` in `~/.config/inlumen-bench/config.toml`, then the platform default.
Avoid folders synced by iCloud.

```
<data-dir>/
  servicecraft/              the dataset; never modified after fetch
  servicecraft-qualified/    written by `bench setup`
```
`bench setup` is needed because ServiceCraft's specs pin Docker image IDs from the
authors' machine; it follows ServiceCraft's documented requalification workflow.

### Installing inLUMEN's agents
The `inlumen` method imports [`inlumen-agents`](https://pypi.org/project/inlumen-agents/),
which the `inlumen` extra installs from PyPI at the exact version pinned in
`pyproject.toml` (currently `1.0.0b6.dev2`, a pre-release). The Docker image installs
the same extra. To benchmark another release, change the pin and rebuild.

To try unreleased inLUMEN changes, install a checkout over the pinned release:
```sh
uv pip install -e '../inLUMEN/packages/agents[design,codegen]'
```
Every `inlumen` episode records the package version it ran in `work/inlumen.json`.

## Methods
| Method | What it does |
| --- | --- |
| `direct` | One request for the whole Dagster bundle; no repair. The baseline. |
| `inlumen` | inLUMEN's own agents: the pipeline design agent (step 1) and the code generator (step 2), from `inlumen-agents`. |
| `inlumen-standalone` | The bench's earlier reimplementation of inLUMEN-style construction: design → one model request per asset → public validation → targeted repair. Experiments made before bench 0.2 used it under the name `inlumen`, and still do when resumed. |

Step 1 (design) uses `inlumen`. Step 2 (code) compares `inlumen` and `direct` (and
`inlumen-standalone`) on the same design.

`inlumen` in detail:
1. **Design** (step 1). inLUMEN's pipeline design agent builds an inLUMEN graph from
   the prompt with its own tools, validation and repair round
   (`inlumen.design.run.design`). It runs on the episode's endpoint and model through
   an OpenAI-compatible client (it needs tool calls); each request is recorded as a
   `design_agent` request and counted against the budget (tokens, no cost).
   inLUMEN's graph is saved as `work/inlumen-graph.json` and `work/project.json`.
   inLUMEN chooses file names only when it generates code, so a **naming pass** (one
   bench request, purpose `naming`, one retry) maps each Source to the input files it
   reads and each Task to the artifacts it writes, without changing the steps. The
   result is checked like any design and saved as `work/design.json`. The naming pass
   is the bench's, not inLUMEN's; its requests are counted as `naming_requests`.
2. **Code** (step 2). The design becomes an inLUMEN graph whose tasks declare their
   artifact files, and inLUMEN's code generator (`inlumen.codegen.run.generate`)
   writes one Task package per task, with its own planning, repair rounds and
   validation, in Docker as inLUMEN runs by default. Allowed packages are the
   contract's installed packages, with no network. Requests go through the bench's
   client. The packages are kept in `work/packages/`; the submission runs each one as
   a Dagster asset (`nodes/<asset>_main.py`, its contract in `nodes/<asset>_manifest.py`,
   wiring in `definitions.py` and `bench_inlumen_runtime.py`). Public validation runs
   once and is only recorded: repairs are inLUMEN's own.
   Options (`method_options.inlumen`): `sandbox` (`docker` or `none`),
   `repair_attempts` (default 2), `max_tool_iterations` (design, default 30), and the
   switches `grounded` and `artifact_contracts` (see [Experimental conditions](#experimental-conditions)).

`inlumen-standalone` in detail:
1. **Design** (step 1). The prompt becomes a task graph shaped like ServiceCraft's
   `task_graph.json` (tasks, input and output artifacts); structural problems are
   fed back for up to 2 design repairs. In step 2 the design is given instead. As in inLUMEN's pipeline agent, each input file with its
   own role starts its own branch, independent branches stay independent, and
   branches meet at a joining task with one input per branch. The design is saved as
   `work/design.json` and as inLUMEN Project JSON in `work/project.json`. Ports there
   are typed from the artifacts (`table`, `json`, `model`, …), and a joining task has
   one named input port per branch.
2. **Nodes.** Each task becomes `nodes/<asset>.py` with `run(inputs) -> dict`.
   `definitions.py` (Dagster wiring) and `bench_runtime.py` (artifact writing) are
   written deterministically from the design.
3. **Repair.** The public validator runs the bundle in ServiceCraft's runtime
   image. A failure is traced to an asset, and only that asset is regenerated with
   the error (up to `max_repairs`, default 3). Private checks are never shown.

## Phases
- **Composition** (`--step design`): the requirements only. A composition is *valid*
  when it passes the structural check; it is then compared with the reference.
- **Implementation** (`--step code`), with `--design-from`:
  - `reference`: the method gets ServiceCraft's reviewed composition and the component
    names the evaluator expects. Measures implementation on a correct composition (RQ2).
  - a composition experiment: the method gets that experiment's valid compositions
    (frozen when the implementation experiment is created). Measures the whole chain.
    ServiceCraft's graph check needs the reference component names, which a generated
    composition doesn't have, so `complete_success` is rare here; the other stages
    are still scored.

Experiments made before the two phases load as implementation experiments; their
`natural_language` runs composed and implemented in one go.

## Experimental conditions
The support mechanisms of the `inlumen` method can be switched off for the ablation
study. An experiment holds the full factorial of the chosen levels, and every
condition runs on the same cases, models and replicates (the `direct` baseline runs
once, without conditions):

| Mechanism | Phase | Off means |
| --- | --- | --- |
| Metamodel-grounded composition (`--grounding`) | composition | One plain request for the composition graph as JSON, instead of inLUMEN's design agent (tools checked against inLUMEN's metamodel, validation, one repair round). The naming pass and the structural check are the same. |
| Contract-bounded generation (`--contracts`) | implementation | Components and dependencies without declared output artifacts (generic port names); the requirements still name the mandatory outputs. |
| Iterative repair loop (`--repair`, `--repairs N`) | implementation | The first attempt is final. With repair on, failed validation is fed back for up to N attempts. |

Each takes `on`, `off` or `both`. For example, the implementation ablation with three
models and five replicates on all cases:
```sh
bench run rq2 --step code --design-from reference -m inlumen -m direct \
  --contracts both --repair both --repairs 2 -n 5 \
  --model openrouter:<large> --model openrouter:<medium> --model ollama:<small>
```

## What is recorded
Every run's `episode.json` holds the condition and every measure, and
`bench export` (or "Download runs (CSV)" in the dashboard) writes them as one row per
run. Failed runs are kept and counted everywhere, including costs.

**Conditions and complexity (all RQs)**: method, model, replicate, condition (with
each mechanism's setting and the maximum repair attempts), limits, and the
composition source. Complexity is measured on the *reference* composition, so a
generated composition with missing components can't make a case look simpler:
components, dependencies, inputs, outputs, longest dependency chain, maximum fan-in
and fan-out, branching points, topology, and a level (low ≤ 4 components, medium 5–6,
high ≥ 7).

**RQ1, composition**:
- the composition graph (`work/inlumen-graph.json`, `work/project.json`,
  `work/design.json`) and its summary: components by type, Sources, Destinations,
  dependencies with their ports, and bindings (parameter values, attached files);
- comparison with the reference composition (provisional, components matched by the
  outputs they write): outputs covered, components found and one-to-one, split and
  merged components, dependencies and bindings found, and the lists of missing and
  extra components, dependencies and bindings;
- the composition operation trace (`work/composition-trace.json`): every tool call of
  the design agent with its arguments and result, and whether the validation repair
  round ran;
- manual review: the dashboard draws a seeded random sample of compositions and
  records a verdict per run (`review.json`).

**RQ2, implementation and execution**:
- every implementation request and reply (`work/implementation/`), the contracts
  plan (each component's input and output contracts), the final Task packages with
  their manifests, dependency declarations and validation reports (`work/packages/`),
  and the Dagster bundle (`submission/`);
- the attempt trace (`work/implementation/trace.json`): per attempt its outcome, the
  validation failures, and those fed back to the model; attempts, first passing
  attempt, whether repair recovered the run, and the final failure reason. Failures
  are sorted into categories (contract, syntax, dependency, sandbox, deployment,
  runtime, format, budget, provider);
- bundling and public validation, timed apart from generation and repair;
- execution: whether Dagster loaded and completed the pipeline, which components
  failed, which outputs were produced, how many passed their contract (schema) and
  matched the reference, and ServiceCraft's six stages.

Rates reported: valid at the first attempt and at the end, and the share of initially
failing runs recovered by repair (from the method's own validation, which runs the
pipeline on sample inputs); executable, valid outputs, expected outputs and complete
success (from ServiceCraft). The condition without repair gives ServiceCraft's
first-attempt rates directly.

**RQ3, cost**: input and output tokens and cost of every request (failed ones and
repairs included), split by phase (composition, implementation, repair); wall time
per phase (composition, implementation, repair, bundle, validation) and their sum as
construction time; Dagster execution time apart. Reported per cell: cost and tokens
per run and per success, median construction and execution time, and the same by
complexity level. The baseline's exact input is saved (`work/request-0.json`).

**Per cell** (method × model × compositions × condition): rates over finished runs
(generation failures count as failures; pending runs and infrastructure errors are
left out), mean ± std of the success rate across replicates, pass@1 and pass@k.

## Models
Any OpenAI-compatible endpoint; the API key is optional.
```sh
bench providers add openrouter --preset openrouter --api-key-env OPENROUTER_API_KEY
bench providers add ollama --preset ollama
bench providers add lab --base-url http://gpu-box:8000/v1 --max-parallel 2
bench providers models openrouter --contains claude
bench providers test ollama qwen3-coder
```
Profiles are stored in `~/.config/inlumen-bench/providers.json` (owner-only).
Keys are never written into run folders or returned by the dashboard.

## Experiments from the command line
```sh
# Phase 1: compositions (grounded and ungrounded)
bench run designs-1 --step design -c 016 -c 001 --grounding both \
  --model openrouter:<model-id> --model ollama:qwen3:32b-16k -n 3
# Phase 2: implementation of the reference compositions, or of phase 1's
bench run code-ref --step code --design-from reference -c 016 -c 001 -m inlumen -m direct \
  --model ollama:qwen3-coder -n 3 --max-cost 2 --max-seconds 1800
bench run code-1 --step code --design-from designs-1 --design-model ollama:qwen3:32b-16k \
  -m inlumen -m direct --model ollama:qwen3-coder
bench run code-1 --resume     # continue after a crash or Ctrl-C
bench evaluate code-1         # execute and score anything implemented but not yet scored
bench status code-1
bench list                    # all experiments
bench export -o runs.csv      # one row per run of every experiment (or name some; .jsonl too)
bench recompute               # recompute measures of existing runs (e.g. made by bench < 0.3)
bench delete code-1 code-2    # remove experiment folders (asks first; -y to skip)
```
An interactive terminal shows progress bars; otherwise progress is written as
plain log lines. Experiments started from the CLI also appear in the dashboard.

Each run folder (`episodes/<run>/`) holds the exported workspace (all the method
saw), the runtime contract it received, the submission, the traces and raw
responses in `work/`, `events.jsonl` (progress and request telemetry) and
`episode.json`.

## Other commands
```sh
bench contract 016 --track fixed_graph   # the runtime contract for a case
bench validate path/to/submission 016    # the public validator (no private checks)
```

## Offline testing
`tests/fake_server.py` is an OpenAI-compatible server with canned replies for
case 016, including a deliberately broken asset that the repair loop fixes:
```sh
python tests/fake_server.py 8765
bench providers add fake --base-url http://127.0.0.1:8765/v1   # host.docker.internal from Docker
bench run offline -c 016 -m inlumen-standalone -m direct --model fake:fake-coder
```
The canned replies are for `inlumen-standalone` and `direct`. The `inlumen` method is
tested offline in `tests/test_inlumen_agents.py`, with scripted models.
