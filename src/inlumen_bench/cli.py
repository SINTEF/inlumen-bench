"""`bench` command line."""

from __future__ import annotations

import atexit
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import bench_paths, config_path, load_config, resolve_data_dir, restore_owner, save_config
from .corpus import Corpus, CorpusError, ensure_images, fetch, setup
from .providers import PRESETS, Profile, load_profiles, remove_profile, upsert_profile

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Benchmark pipeline design and code generation on ServiceCraft.")
datasets_app = typer.Typer(no_args_is_help=True, help="Find, check and fetch datasets.")
providers_app = typer.Typer(no_args_is_help=True, help="Model endpoint profiles.")
config_app = typer.Typer(no_args_is_help=True, help="Show or change settings.")
app.add_typer(datasets_app, name="datasets")
app.add_typer(providers_app, name="providers")
app.add_typer(config_app, name="config")

console = Console()
STATE: dict = {}


@app.callback()
def main(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Overrides INLUMEN_BENCH_DATA_DIR and the config file."),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", envvar="INLUMEN_BENCH_RUNS_DIR", help="Where experiments are written."),
) -> None:
    STATE["data_dir"] = resolve_data_dir(data_dir)
    STATE["data_dir_flag"] = data_dir
    STATE["runs_dir"] = runs_dir
    # tempfile silently falls back to /tmp when TMPDIR is missing, which breaks
    # Docker bind mounts in the container setup; create it up front.
    if os.environ.get("TMPDIR"):
        Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    atexit.register(lambda: restore_owner(*bench_paths(STATE["data_dir"].path, runs_dir.resolve())))


def _corpus() -> Corpus:
    return Corpus(STATE["data_dir"].path)


def _fail(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(2)


@app.command()
def version() -> None:
    console.print(__version__)


# ---- config ---------------------------------------------------------------

@config_app.command("show")
def config_show() -> None:
    data = STATE["data_dir"]
    console.print(f"config file: {config_path()}")
    console.print(f"data folder: {data.path}  [dim](from {data.source})[/dim]")
    for key, value in load_config().items():
        console.print(f"  {key} = {value}")


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    if key not in {"data_dir"}:
        _fail(f"Unknown setting {key!r}; supported: data_dir")
    values = load_config()
    values[key] = value
    console.print(f"saved {save_config(values)}")


# ---- datasets -------------------------------------------------------------

@datasets_app.command("list")
def datasets_list() -> None:
    data = STATE["data_dir"]
    console.print(f"data folder: {data.path}  [dim](from {data.source})[/dim]")
    corpus = _corpus()
    if not corpus.exists():
        console.print("  no datasets (run `bench datasets fetch`)")
        return
    meta = corpus.metadata()
    status = corpus.setup_status()
    qualified = status["qualified"]
    console.print(f"  servicecraft  commit {meta.get('commit', '?')[:12]}{' (modified)' if meta.get('dirty') else ''}  "
                  f"{len(corpus.case_ids())} cases  "
                  + (f"qualified {qualified['qualified_cases']}/{qualified['case_count']}" if qualified else "not set up"))


@datasets_app.command("check")
def datasets_check() -> None:
    problems = _corpus().check()
    if problems:
        for p in problems[:50]:
            console.print(f"  [red]✗[/red] {p}")
        raise typer.Exit(1)
    console.print("[green]✓[/green] servicecraft dataset is complete and prompts match the manifest")


@datasets_app.command("fetch")
def datasets_fetch(ref: Optional[str] = typer.Option(None, help="Commit or tag to check out.")) -> None:
    corpus = fetch(STATE["data_dir"].path, ref=ref)
    console.print(f"fetched servicecraft {corpus.version()} into {corpus.root}")


@app.command("setup")
def setup_cmd(
    workers: int = typer.Option(3, help="Parallel control qualifications."),
    rebuild: bool = typer.Option(False, help="Rebuild the Docker images."),
) -> None:
    """Evaluator environment, Docker images and local requalification of all controls."""
    try:
        summary = setup(_corpus(), workers=workers, rebuild=rebuild, log=lambda m: console.print(f"[dim]{m}[/dim]"))
    except CorpusError as exc:
        _fail(str(exc))
    console.print(f"[green]✓[/green] qualified {summary['qualified_cases']}/{summary['case_count']} cases")


# ---- providers ------------------------------------------------------------

@providers_app.command("list")
def providers_list() -> None:
    profiles = load_profiles()
    if not profiles:
        console.print("no profiles (add one with `bench providers add`)")
        return
    table = Table("name", "base url", "api key", "parallel", "timeout", "stall")
    for p in profiles.values():
        m = p.masked()
        key = f"env:{p.api_key_env}" if p.api_key_env else (m["api_key"] or "—")
        table.add_row(p.name, p.base_url, key, str(p.max_parallel), f"{p.timeout_s:.0f}s", f"{p.stall_s:.0f}s")
    console.print(table)


@providers_app.command("add")
def providers_add(
    name: str,
    preset: str = typer.Option("custom", help=f"One of: {', '.join(PRESETS)}"),
    base_url: Optional[str] = typer.Option(None),
    api_key_env: Optional[str] = typer.Option(None, help="Read the key from this environment variable."),
    ask_key: bool = typer.Option(False, "--ask-key", help="Prompt for the key (stored in the profile file)."),
    max_parallel: Optional[int] = typer.Option(None),
    timeout_s: Optional[float] = typer.Option(None),
    stall_s: Optional[float] = typer.Option(None),
) -> None:
    if preset not in PRESETS:
        _fail(f"Unknown preset {preset!r}")
    values = {**PRESETS[preset]}
    for key, value in (("base_url", base_url), ("max_parallel", max_parallel), ("timeout_s", timeout_s), ("stall_s", stall_s)):
        if value is not None:
            values[key] = value
    if not values["base_url"]:
        _fail("--base-url is required for a custom profile")
    api_key = typer.prompt("API key", hide_input=True) if ask_key else None
    path = upsert_profile(Profile(name=name, api_key=api_key, api_key_env=api_key_env, **values))
    console.print(f"saved profile {name!r} in {path}")


@providers_app.command("remove")
def providers_remove(name: str) -> None:
    console.print("removed" if remove_profile(name) else f"no profile {name!r}")


@providers_app.command("models")
def providers_models(name: str, contains: Optional[str] = typer.Option(None)) -> None:
    from .llm import list_models

    models = list_models(load_profiles()[name])
    for model in models:
        if not contains or contains.lower() in model.lower():
            console.print(model)


@providers_app.command("test")
def providers_test(name: str, model: str) -> None:
    from .llm import ping

    result = ping(load_profiles()[name], model)
    console.print(f"[green]✓[/green] {result.resolved_model or model} answered {result.text.strip()!r} "
                  f"in {result.latency_s:.1f}s · usage {result.usage_provenance} · cost {result.cost_usd}")


# ---- single-case tools ----------------------------------------------------

@app.command()
def contract(case: str, track: str = typer.Option("fixed_graph")) -> None:
    """Print the runtime contract a method receives for a case."""
    from .contract import render

    corpus = _corpus()
    text, sha = render(corpus, corpus.resolve_cases([case])[0], track)
    typer.echo(text)
    typer.echo(f"sha256 {sha}", err=True)


@app.command("validate")
def validate_cmd(submission: Path, case: str) -> None:
    """Run the public validator (the feedback methods may see) on a submission."""
    from .validate import validate

    corpus = _corpus()
    case_id = corpus.resolve_cases([case])[0]
    with tempfile.TemporaryDirectory() as tmp:
        workspace = corpus.export(case_id, Path(tmp) / "ws", "natural_language")
        report = validate(submission, workspace / "inputs", corpus.prompt(case_id))
    console.print_json(json.dumps(report.to_dict()))


# ---- experiments ----------------------------------------------------------

@app.command()
def run(
    name: str,
    step: str = typer.Option("code", "--step", "-s", help="design (step 1) or code (step 2)."),
    design_from: str = typer.Option(
        "reference", "--design-from", "-d",
        help="Step 2: `reference` (ServiceCraft's graphs) or the name of a design experiment."),
    design_model: list[str] = typer.Option(
        [], "--design-model", help="Step 2 from a design experiment: only its designs by these models."),
    case: list[str] = typer.Option([], "--case", "-c", help="Case ID or number; repeat. Default: all."),
    method: list[str] = typer.Option(["inlumen"], "--method", "-m", help="Step 2: inlumen and/or direct."),
    model: list[str] = typer.Option([], "--model", help="profile:model; repeat for several."),
    reps: int = typer.Option(1, "--reps", "-n"),
    seed: int = typer.Option(0),
    max_cost: Optional[float] = typer.Option(None, help="USD per episode."),
    max_seconds: Optional[float] = typer.Option(None, help="Wall time per episode."),
    max_requests: Optional[int] = typer.Option(None, help="Model requests per episode."),
    workers: int = typer.Option(2, help="Runs in parallel (models run one after another)."),
    runs_per_model: int = typer.Option(
        1, help="Runs of one model at a time. Keep 1 for a local Ollama (one request per model at a time); "
                "parallel runs then go to different models, each on its own GPU."),
    grounding: str = typer.Option("on", help="Composition: metamodel grounding on, off or both (ablation)."),
    contracts: str = typer.Option("on", help="Implementation: explicit artifact contracts on, off or both."),
    repair: str = typer.Option("on", help="Implementation: the repair loop on, off or both."),
    repairs: int = typer.Option(2, help="Implementation: maximum repair attempts when repair is on."),
    resume: bool = typer.Option(False, help="Continue an existing experiment."),
    evaluate: bool = typer.Option(True, help="Step 2: score with ServiceCraft when generation ends."),
    plain: Optional[bool] = typer.Option(None, "--plain/--rich", help="Force the plain or the bar view."),
) -> None:
    """Create (or resume) an experiment and run it: `--step design`, then `--step code`."""
    from .evaluate import evaluate_experiment
    from .progress import ProgressDisplay
    from .runner import Experiment, create_experiment, make_conditions, pending_episodes, run_experiment

    corpus = _corpus()
    root = STATE["runs_dir"] / name
    if resume:
        exp = Experiment.load(root)
    else:
        if not model:
            _fail("--model is required")
        limits = {k: v for k, v in {"max_cost_usd": max_cost, "max_seconds": max_seconds, "max_requests": max_requests}.items() if v is not None}
        levels = {"on": (True,), "off": (False,), "both": (True, False)}
        for flag, value in (("--grounding", grounding), ("--contracts", contracts), ("--repair", repair)):
            if value not in levels:
                _fail(f"{flag} takes on, off or both")
        conditions = make_conditions(step, grounding=levels[grounding], contracts=levels[contracts],
                                     repair=levels[repair], repairs=repairs)
        try:
            exp = create_experiment(root, corpus, stage=step, design_from=design_from,
                                    design_models=design_model or None, name=name,
                                    cases=corpus.resolve_cases(case) if case or design_from == "reference" else [],
                                    methods=method, models=model, repetitions=reps, seed=seed, limits=limits,
                                    conditions=conditions, runs_per_model=runs_per_model,
                                    method_options={"inlumen": {"repair_attempts": repairs}} if step == "code" else {})
        except (FileExistsError, ValueError, KeyError, CorpusError) as exc:
            _fail(str(exc))
    todo = pending_episodes(root, exp)
    total = len(exp.episodes())
    console.print(f"{name} ({exp.stage}, {exp.design_source}): {total} episodes, {len(todo)} to run → {root}")
    if todo:
        with ProgressDisplay(len(todo), name, plain=plain):
            run_experiment(root, corpus, workers=workers)
    if evaluate and exp.stage == "code":
        evaluate_experiment(root, corpus, log=lambda m: console.print(f"[dim]{m}[/dim]"))
    status(name)


@app.command("evaluate")
def evaluate_cmd(name: str) -> None:
    """Score generated, not yet evaluated episodes with ServiceCraft."""
    from .evaluate import evaluate_experiment

    evaluate_experiment(STATE["runs_dir"] / name, _corpus(), log=lambda m: console.print(f"[dim]{m}[/dim]"))
    status(name)


@app.command("list")
def list_cmd() -> None:
    """Experiments in the runs folder, newest first."""
    from .runner import Experiment

    runs = STATE["runs_dir"]
    paths = sorted(runs.glob("*/experiment.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    table = Table("name", "step", "designs", "models", "cases", "episodes", "created")
    for path in paths:
        exp = Experiment.load(path.parent)
        table.add_row(exp.name, exp.stage, exp.design_source, ", ".join(exp.models), str(len(exp.cases)),
                      str(len(exp.episodes())), exp.created_at.replace("T", " ")[:16])
    console.print(table)


@app.command()
def delete(names: list[str] = typer.Argument(..., help="One or more experiments."),
           yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation.")) -> None:
    """Delete experiments and everything in their folders."""
    from .runner import delete_experiment, dependents

    roots = [STATE["runs_dir"] / name for name in names]
    for name, root in zip(names, roots):
        if not (root / "experiment.json").is_file():
            _fail(f"No experiment named {name!r} in {STATE['runs_dir']}")
    users = sorted({u for name in names for u in dependents(STATE["runs_dir"], name)} - set(names))
    if users:
        console.print(f"[yellow]Implementation experiments built on their compositions can't be resumed "
                      f"afterwards: {', '.join(users)}[/yellow]")
    if not yes and not typer.confirm(f"Delete {', '.join(names)} with all their runs?"):
        raise typer.Abort()
    for name, root in zip(names, roots):
        delete_experiment(root)
        console.print(f"deleted {name}")


@app.command("recompute")
def recompute_cmd(names: list[str] = typer.Argument(None, help="Experiments; default: all.")) -> None:
    """Recompute every run's measures from its files (e.g. for runs made by an older bench)."""
    from .runner import recompute

    runs = STATE["runs_dir"]
    roots = [runs / n for n in names] if names else sorted(p.parent for p in runs.glob("*/experiment.json"))
    corpus = _corpus()
    for root in roots:
        console.print(f"{root.name}: {recompute(root, corpus)} runs updated")


@app.command()
def export(names: list[str] = typer.Argument(None, help="Experiments; default: all."),
           output: Path = typer.Option(Path("runs.csv"), "--output", "-o", help=".csv or .jsonl"),
           ) -> None:
    """One row per run with conditions, complexity and every RQ1–RQ3 measure."""
    from .results import export_rows, to_csv, to_jsonl

    runs = STATE["runs_dir"]
    roots = ([runs / n for n in names] if names
             else sorted(p.parent for p in runs.glob("*/experiment.json")))
    rows = export_rows(roots)
    output.write_text(to_jsonl(rows) if output.suffix == ".jsonl" else to_csv(rows))
    console.print(f"{len(rows)} runs from {len(roots)} experiments → {output}")


@app.command()
def status(name: str) -> None:
    """Results per method, model and design source."""
    from .results import summarize
    from .runner import Experiment

    root = STATE["runs_dir"] / name
    exp = Experiment.load(root)
    rate = lambda c, k: "—" if c.get(k) is None else f"{c[k] * 100:.0f}%"
    cost = lambda c: f"${c['cost_usd']:.3f}" + (" + ?" if c["cost_incomplete"] else "")
    if exp.stage == "design":
        table = Table("model", "condition", "runs", "pending", "failed", "infra", "valid", "components found",
                      "dependencies found", "bindings found", "correct", "cost")
        for c in summarize(root):
            table.add_row(c["model"], c["condition"], str(c["episodes"]), str(c["pending"]),
                          str(c["generation_failed"]), str(c["infra"]), rate(c, "valid_design"),
                          rate(c, "design_task_recall"), rate(c, "design_dependency_recall"),
                          rate(c, "design_binding_recall"), rate(c, "design_design_correct"), cost(c))
    else:
        table = Table("method", "model", "compositions", "condition", "runs", "pending", "failed", "infra",
                      "first valid", "final valid", "executable", "complete success", "cost")
        for c in summarize(root):
            table.add_row(c["method"], c["model"], str(c["design"]), c["condition"], str(c["episodes"]),
                          str(c["pending"]), str(c["generation_failed"]), str(c["infra"]), rate(c, "initial_valid"),
                          rate(c, "final_valid"), rate(c, "dagster_executable"), rate(c, "complete_success"), cost(c))
    console.print(f"{name}: {'pipeline composition' if exp.stage == 'design' else 'pipeline implementation'}")
    console.print(table)


@app.command()
def serve(
    host: Optional[str] = typer.Option(
        None, help="Default: localhost (inside Docker, the container's network; compose publishes it on "
                   "localhost only). The dashboard has no login: don't expose it."),
    port: int = typer.Option(8000),
) -> None:
    """Start the Streamlit dashboard."""
    import sys

    from inlumen_bench.config import dashboard_host

    host = host or dashboard_host()
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    runs = STATE["runs_dir"].resolve()
    runs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "INLUMEN_BENCH_RUNS_DIR": str(runs)}
    if STATE["data_dir_flag"]:
        env["INLUMEN_BENCH_DATA_DIR"] = str(STATE["data_dir"].path)
    corpus = _corpus()
    if corpus.setup_status()["qualified"]:
        try:
            ensure_images(corpus, log=lambda m: console.print(f"[dim]{m}[/dim]"))
        except Exception as exc:  # The dashboard shows the image state; don't block startup.
            console.print(f"[yellow]evaluator images: {exc}[/yellow]")
    console.print(f"dashboard on http://localhost:{port}")
    args = [sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.address", host, "--server.port", str(port), "--server.headless", "true",
            "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false",
            "--client.toolbarMode", "minimal", "--theme.primaryColor", "#2f6fb3"]
    # Replace this process, so signals (Ctrl-C, docker stop) reach Streamlit directly.
    os.execve(sys.executable, args, env)


if __name__ == "__main__":
    app()
