# inlumen-bench

A generation driver and dashboard for benchmarking pipeline design and code
generation on the [ServiceCraft](https://github.com/aliduabubakari/servicecraft)
corpus. It runs methods across cases, models, tracks and repetitions, records
telemetry for every model request, and scores submissions with ServiceCraft's
own evaluator. It does not depend on inLUMEN; pipeline designs are exported as
inLUMEN Project JSON so they can be opened with inLUMEN's Import button.

## Quick start with Docker
Only Docker is needed on the host.
```sh
docker compose up -d          # dashboard on http://localhost:8000 (builds the image the first time)
```
The dashboard is a [Streamlit](https://streamlit.io) app (`src/inlumen_bench/dashboard/app.py`).
In the dashboard:
1. **Corpus**: the dataset in the data folder is found automatically (or fetched with
   one click). The Cases tab lists the 100 cases; select one to see its prompt,
   pipeline graph, tasks and input data, or download it as inLUMEN Project JSON. The
   Evaluator tab prepares ServiceCraft's evaluator once per machine (about 7 minutes).
2. **Models**: add an endpoint (OpenRouter, Ollama, any OpenAI-compatible URL) and test it.
3. **Launch**: pick methods, models, tracks, cases and repetitions.
4. **Experiments**: live progress, results per method × model × track, and every
   episode's files, scores and Project JSON.

The same can be done from the command line with `scripts/bench-docker`, which takes
the same arguments as `bench` (see below):
```sh
scripts/bench-docker datasets fetch
scripts/bench-docker setup
scripts/bench-docker run pilot -c 016 --model openrouter:<model-id> -m inlumen -m direct -n 3
```
After changing the code, rebuild with `docker compose build` (or `docker build -t inlumen-bench:0.1.0 .`).

### Docker settings
Put these in a `.env` file next to `compose.yaml` (all optional):
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
  prints "Network" and "External" URLs; they are not reachable, because the port
  is bound to 127.0.0.1.)
- Local model servers on the host are reached as `host.docker.internal`; the
  `ollama` preset uses that automatically inside the container.
- On Linux the container runs as root, so files it writes are owned by root.

### On a shared Linux server
`compose.linux.yaml` makes the containers use the host's network, so a model server
listening only on `127.0.0.1` (such as a system Ollama) is reachable at
`http://127.0.0.1:11434/v1`; the dashboard then listens on the server's localhost only.
The containers run as root, because ServiceCraft's evaluator has to remove files its
sandbox containers write as uid 65532; after every command and dashboard job, bench
gives the corpus folders, runs and config back to your user. In `.env` on the server:
```sh
COMPOSE_FILE=compose.yaml:compose.linux.yaml
BENCH_UID=1008            # id -u
BENCH_GID=1008            # id -g
BENCH_PORT=8501
```
Then `docker compose up -d --build` on the server, and from your own computer:
```sh
ssh -N -L 8501:127.0.0.1:8501 you@server   # dashboard on http://localhost:8501
```
For Ollama, prefer models with a long enough context (the `-16k` variants on a shared
server, or `OLLAMA_CONTEXT_LENGTH` on your own); the default context silently cuts
long prompts. Set the profile's parallel requests to what the server allows.

## Native install
Python 3.11+ (or [uv](https://docs.astral.sh/uv/)) and a running Docker engine.
```sh
uv venv --python 3.12 .venv
uv pip install -e '.[test]'
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

## Methods
| Method | What it does |
| --- | --- |
| `direct` | One request for the whole Dagster bundle; no repair. The baseline. |
| `inlumen` | inLUMEN-style construction, reimplemented standalone: design → one model request per asset in dependency order → public validation → targeted repair of the failing asset. |

`inlumen` in detail:
1. **Design.** Natural-language track: the prompt becomes a task graph shaped like
   ServiceCraft's `task_graph.json` (tasks, input and output artifacts); structural
   problems are fed back for up to 2 design repairs. Fixed-graph track: the
   provided graph is used. As in inLUMEN's pipeline agent, each input file with its
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

## Tracks
- `fixed_graph`: the method also gets the reviewed graph and the asset names the
  evaluator expects. Measures code generation.
- `natural_language`: the prompt only. Measures design and code. ServiceCraft's
  graph check needs exact asset names, which the prompt does not give, so
  `complete_success` is rare here; the other stages are still scored.

## Scores
- **ServiceCraft stages** (per episode): submission, static, definition graph,
  execution, output contracts, behavioural oracle; `complete_success` needs all six.
- **Design scores** (natural-language track, provisional): generated tasks are
  matched to the reference graph by the artifacts they write (the prompt names
  every mandatory artifact). Reported: artifact coverage, task recall, one-to-one
  matches, split/merged tasks, dependency recall, extra dependencies and an
  overall `design_correct`. Labelled provisional until ServiceCraft has a reviewed
  mapping procedure.
- **Per method × model × track**: rates over finished episodes (generation
  failures count as failures; pending and infrastructure errors are left out),
  mean ± std of the success rate across repetitions, pass@1 and pass@k, cost,
  cost per success, tokens and median generation time.

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
bench run pilot -c 016 -c 001 -m inlumen -m direct \
  --model openrouter:<model-id> --model ollama:qwen3-coder \
  -t fixed_graph -t natural_language -n 3 --max-cost 2 --max-seconds 1800
bench run pilot --resume      # continue after a crash or Ctrl-C
bench evaluate pilot          # score anything generated but not yet evaluated
bench status pilot
```
An interactive terminal shows progress bars; otherwise progress is written as
plain log lines. Experiments started from the CLI also appear in the dashboard.

Each episode folder holds the exported workspace (all the method saw), the
runtime contract it received, the submission, raw responses, `events.jsonl`
(progress and request telemetry) and `episode.json`.

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
bench run offline -c 016 -m inlumen -m direct --model fake:fake-coder -t fixed_graph -t natural_language
```
