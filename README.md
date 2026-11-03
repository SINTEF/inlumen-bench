# inlumen-bench

A generation driver for benchmarking pipeline design and code generation on the
[ServiceCraft](https://github.com/aliduabubakari/servicecraft) corpus. It runs
methods (one-shot Direct baseline for now) across cases, models, tracks and
repetitions, records telemetry for every model request, and scores submissions
with ServiceCraft's own evaluator. It does not depend on inLUMEN.

## Requirements
- Python 3.11+ (or [uv](https://docs.astral.sh/uv/))
- A running Docker engine

## Install
```sh
uv venv --python 3.12 .venv
uv pip install -e '.[test]'
```

## Or run it with Docker
Only Docker is needed on the host. `scripts/bench-docker` takes the same
arguments as `bench`:
```sh
docker build -t inlumen-bench:0.1.0 .   # optional: the first run builds it too
scripts/bench-docker datasets fetch
scripts/bench-docker setup
scripts/bench-docker run pilot -c 016 --model openrouter:<model-id> -n 2
```
How it works:
- The container uses the host's Docker engine through `/var/run/docker.sock`,
  because ServiceCraft runs every submission in its own container.
- The script mounts the data folder and the runs folder at the **same absolute
  paths** as on the host, since the host engine resolves those paths. Temporary
  folders go to `<runs>/.tmp` for the same reason.
- Folders: `BENCH_DATA_DIR` (default: same lookup as `bench`), `BENCH_RUNS_DIR`
  (default: `./runs`), `BENCH_CONFIG_DIR` (default: `~/.config/inlumen-bench`,
  shared with a native install).
- API keys: `OPENROUTER_API_KEY` and `OPENAI_API_KEY` are passed through from the
  host environment; use them with `--api-key-env`.
- Local model servers on the host are reached as `host.docker.internal`; the
  `ollama` preset uses that automatically inside the container. A profile created
  natively with `localhost` will not work from the container, and the other way round.
- On Linux the container runs as root, so files it writes in the runs folder are
  owned by root.

## Data folder
Datasets live in one folder, chosen in this order:
1. `--data-dir PATH`
2. `INLUMEN_BENCH_DATA_DIR`
3. `data_dir` in `~/.config/inlumen-bench/config.toml` (`bench config set data_dir ~/datasets`)
4. the platform default (macOS: `~/Library/Application Support/inlumen-bench/datasets`)

Avoid folders synced by iCloud (such as `~/Documents` with Desktop & Documents sync on).

```
<data-dir>/
  servicecraft/              the dataset; never modified after fetch
  servicecraft-qualified/    written by `bench setup`
```

## First run
```sh
bench datasets fetch          # clone ServiceCraft into the data folder
bench datasets check          # layout + prompt hashes against the manifest
bench setup                   # evaluator env, Docker images, local requalification (~15 min)
```
`bench setup` is needed because ServiceCraft's specs pin Docker image IDs from the
authors' machine. It follows ServiceCraft's documented requalification workflow
and writes the result next to the dataset, not into it.

## Models
Any OpenAI-compatible endpoint. The API key is optional.
```sh
bench providers add openrouter --preset openrouter --api-key-env OPENROUTER_API_KEY
bench providers add ollama --preset ollama
bench providers add lab --base-url http://gpu-box:8000/v1 --max-parallel 2
bench providers models openrouter --contains claude
bench providers test ollama qwen3-coder
```
Profiles are stored in `~/.config/inlumen-bench/providers.json` (owner-only).
Keys are never written into run folders.

## Experiments
```sh
bench run pilot --case 016 --case 001 \
  --model openrouter:anthropic/claude-sonnet-5.5 --model ollama:qwen3-coder \
  --track fixed_graph --reps 2 --max-cost 2 --max-seconds 900
bench run pilot --resume      # continue after a crash or Ctrl-C
bench evaluate pilot          # score anything generated but not yet evaluated
bench status pilot
```
An interactive terminal shows progress bars (experiment and one row per running
episode: phase, step, tokens streaming in, budget). Without a terminal the same
progress is written as plain log lines.

Each episode folder holds the exported workspace (all the method saw), the
runtime contract it received, the submission, raw responses, `events.jsonl`
(progress and request telemetry) and `episode.json`.

## Other commands
```sh
bench contract 016 --track fixed_graph   # the runtime contract for a case
bench validate path/to/submission 016    # the public validator (no private checks)
```
