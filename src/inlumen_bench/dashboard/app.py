"""Streamlit dashboard: launch experiments, follow them live, read results per research question.

Started by `bench serve`. It has no login and can start paid model runs, so keep it
on localhost. State lives in the same files the CLI uses.

Terms (as in the paper): pipeline construction has three phases, pipeline
composition (requirements → composition graph), pipeline implementation
(composition graph → code artifacts) and pipeline execution (ServiceCraft runs
the Dagster bundle and checks its outputs). A run is one case × method × model ×
condition × replicate; an experiment is a set of runs of one phase.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from inlumen_bench import __version__, columns, results
from inlumen_bench.config import resolve_data_dir
from inlumen_bench.corpus import Corpus, CorpusError, fetch, setup
from inlumen_bench.events import read_events
from inlumen_bench.jobs import RUN_LOG, JobBusy, Jobs, kill_run, read_log, run_cli, snapshot
from inlumen_bench.methods import METHODS
from inlumen_bench.metrics import categorize
from inlumen_bench.providers import PRESETS, Profile, load_profiles, remove_profile, upsert_profile
from inlumen_bench.runner import (REFERENCE, Experiment, create_experiment, delete_experiment, delete_runs,
                                  dependents, episode_state, make_conditions, successful_designs)

RUNS = Path(os.environ.get("INLUMEN_BENCH_RUNS_DIR", "runs")).resolve()
NAME_RE = r"^[A-Za-z0-9_.-]{1,60}$"
STATUS_LABEL = {"evaluated": "🟢 executed", "designed": "🟢 composed", "generated": "🔵 awaiting execution",
                "running": "🔵 running", "stopped": "⏹️ stopped", "generation_failed": "🔴 failed",
                "infra_error": "🟠 infrastructure error", "pending": "⚪ pending"}
FINISHED = ("evaluated", "designed", "generation_failed")
PHASE = {"design": "Pipeline composition", "code": "Pipeline implementation"}
PHASE_HELP = {
    "design": "Requirements → composition graph. inLUMEN's design agent composes the pipeline; the result is "
              "checked and compared with ServiceCraft's reference composition. Nothing is executed.",
    "code": "Composition graph → code artifacts. The method writes the code, which is bundled for Dagster; "
            "ServiceCraft then executes the pipeline and checks its outputs (pipeline execution).",
}
METHOD_HELP = {"inlumen": "inLUMEN's own agents (contract-bounded generation and repair loop).",
               "direct": "Baseline: one request for the whole Dagster bundle, no contracts, no repair.",
               "inlumen-standalone": "The bench's earlier reimplementation (kept for old experiments)."}
PERCENT = st.column_config.NumberColumn(format="%.1f%%")


def _help_key(label: str) -> str:
    return "pass@k" if label.startswith("pass@") and label != "pass@1" else label


def table_config(table: str, frame: pd.DataFrame, percent=()) -> dict:
    """Column settings with each column's description as its header tooltip."""
    texts = columns.table_help(table)
    return {c: (st.column_config.NumberColumn(format="%.1f%%", help=texts.get(_help_key(c)))
                if c in percent else st.column_config.Column(help=texts.get(_help_key(c))))
            for c in frame.columns}


def show_table(table: str, frame: pd.DataFrame, percent=(), **kwargs):
    """A results table, its tooltips, and below it what each of its columns means."""
    shown = st.dataframe(frame, hide_index=True, width="stretch",
                         column_config=table_config(table, frame, percent), **kwargs)
    with st.expander("What these columns mean"):
        st.dataframe(pd.DataFrame(columns.describe_table(table, [_help_key(c) for c in frame.columns])),
                     hide_index=True, width="stretch",
                     column_config={"description": st.column_config.TextColumn(width="large")})
    return shown

st.set_page_config(page_title="inlumen-bench", page_icon=":material/science:", layout="wide")


@st.cache_resource
def jobs() -> Jobs:
    """One job registry per server process, shared by every browser session."""
    return Jobs()


def corpus() -> Corpus:
    return Corpus(resolve_data_dir().path)


def experiment_names() -> list[str]:
    if not RUNS.is_dir():
        return []
    paths = sorted(RUNS.glob("*/experiment.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.parent.name for p in paths]


def pct(value) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def usd(value) -> str:
    return "—" if value is None else f"${value:.4f}" if value < 1 else f"${value:.2f}"


def secs(value) -> str:
    if value is None:
        return "—"
    value = float(value)
    return f"{value:.1f}s" if value < 90 else f"{value / 60:.1f} min"


def num(series):
    """Numbers with missing values as NaN, which tables show as empty cells."""
    return pd.to_numeric(series, errors="coerce").astype("float64")


def as_percent(series):
    return num(series) * 100


def design_experiments() -> list[str]:
    return [n for n in experiment_names() if Experiment.load(RUNS / n).stage == "design"]


def is_running(name: str) -> bool:
    return bool(jobs().running_for(f"exp:{name}"))


def bump_list() -> None:
    """A new table key clears its selection, so nothing else gets selected by position."""
    st.session_state["list_version"] = st.session_state.get("list_version", 0) + 1


def design_dot(design: dict) -> str:
    """Graphviz source for a composition graph: input files, components, output-labelled dependencies."""
    from inlumen_bench.design import edges
    lines = ['digraph { rankdir=LR; node [fontname="Helvetica" fontsize=11]; edge [fontname="Helvetica" fontsize=9];']
    inputs = {i["id"] for i in design.get("inputs", [])} | {
        i for t in design["tasks"] for i in t.get("inputs", []) if i.startswith("input:")}
    for i in sorted(inputs):
        lines.append(f'"{i}" [shape=note label="{i.split(":", 1)[1]}" style=filled fillcolor="#eef3f8"];')
    for t in design["tasks"]:
        lines.append(f'"{t["id"]}" [shape=box style="rounded,filled" fillcolor="#dbe7f3" label="{t["id"]}"];')
        for item in t.get("inputs", []):
            if item.startswith("input:"):
                lines.append(f'"{item}" -> "{t["id"]}";')
    lines += [f'"{a}" -> "{b}" [label="{art}"];' for a, b, art in edges(design)]
    return "\n".join(lines + ["}"])


def launch_run(root: Path, workers: int = 2, evaluate: bool = True) -> None:
    """Runs (or resumes) an experiment in its own process; Stop kills it at once."""
    start("experiment", f"exp:{root.name}", run_cli, root, "run", root.name, "--resume", "--plain",
          "--workers", str(workers), "--evaluate" if evaluate else "--no-evaluate")


def start(kind: str, label: str, fn, *args) -> None:
    try:
        jobs().start(kind, label, fn, *args)
    except JobBusy as exc:
        st.warning(str(exc))


def stop(name: str) -> None:
    if not jobs().stop(f"exp:{name}"):
        # Nothing runs here (e.g. after a dashboard restart): clear what was left behind.
        kill_run(None, RUNS / name, log=lambda m: None)


# ---- experiments --------------------------------------------------------------

def experiments_page() -> None:
    names = experiment_names()
    head, button = st.columns([4, 1], vertical_alignment="bottom")
    head.title("Experiments")
    if button.button("New experiment", type="primary", icon=":material/add:", width="stretch"):
        st.switch_page(LAUNCH)
    if not names:
        st.info("No experiments yet.")
        st.markdown("To get started:\n1. **Corpus**: fetch the ServiceCraft dataset and prepare the evaluator.\n"
                    "2. **Models**: add a model endpoint and send a test request.\n"
                    "3. **Launch**: start a pipeline composition experiment, then implement its compositions.")
        return

    rows = []
    for name in names:
        exp = Experiment.load(RUNS / name)
        counts: dict[str, int] = {}
        for ep in exp.episodes():
            status = (episode_state(RUNS / name / "episodes" / ep["id"]) or {}).get("status", "pending")
            counts[status] = counts.get(status, 0) + 1
        total = sum(counts.values())
        finished = sum(counts.get(k, 0) for k in FINISHED)
        # Runs marked running without a dashboard job were started from the CLI.
        state = ("▶ running" if is_running(name) or counts.get("running") else "✓ finished" if total and finished == total
                 else "⏸ paused" if finished or counts.get("stopped") else "○ not started")
        rows.append({"experiment": name, "phase": PHASE[exp.stage], "state": state,
                     "progress": finished / total if total else 0.0, "runs": total,
                     "failed": counts.get("generation_failed", 0),
                     "compositions": "from the prompt" if exp.stage == "design" else exp.design_source,
                     "methods": ", ".join(exp.methods) if exp.stage == "code" else "inlumen",
                     "conditions": ", ".join(c["name"] for c in exp.conditions) or "default",
                     "models": ", ".join(exp.models), "cases × replicates": f"{len(exp.cases)} × {exp.repetitions}",
                     "created": exp.created_at.replace("T", " ")[:16]})
    st.caption("Select one experiment to open it, or several to stop, export or delete them together.")
    picked = st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", on_select="rerun",
                          selection_mode="multi-row", key=f"experiment-list-{st.session_state.get('list_version', 0)}",
                          column_config={"progress": st.column_config.ProgressColumn(
                              "finished", min_value=0, max_value=1, format="percent")})
    selected = [names[i] for i in picked.selection.rows]
    wanted = st.query_params.get("exp")
    if selected:
        st.session_state["had_selection"] = True
        if len(selected) == 1:
            st.query_params["exp"] = selected[0]
            experiment_detail(selected[0])
        else:
            st.query_params.pop("exp", None)
            st.markdown(f"**{len(selected)} experiments selected**: " + ", ".join(selected))
            bulk_actions(st.container(horizontal=True), selected)
        return
    if st.session_state.pop("had_selection", False):
        st.query_params.pop("exp", None)  # the rows were unselected
        return
    if wanted in names:  # opened by link, e.g. after Launch
        experiment_detail(wanted)


def has_running_runs(name: str) -> bool:
    """A dashboard job, or runs marked running (e.g. started from the CLI)."""
    if is_running(name):
        return True
    return any((episode_state(p.parent) or {}).get("status") == "running"
               for p in (RUNS / name / "episodes").glob("*/episode.json"))


def bulk_actions(bar, selected: list[str]) -> None:
    """Download, stop and delete for one or several experiments."""
    running = [n for n in selected if has_running_runs(n)]
    many = len(selected) > 1
    label = f"{len(selected)} experiments" if many else ""
    rows = results.export_rows([RUNS / n for n in selected])
    bar.download_button(f"Download runs {'of ' + label + ' ' if many else ''}(CSV)", results.to_csv(rows),
                        icon=":material/download:", file_name=f"{'experiments' if many else selected[0]}-runs.csv",
                        mime="text/csv", help="One row per run: conditions, complexity and every RQ1–RQ3 measure.")
    if rows:
        bar.download_button("Column descriptions (CSV)", columns.to_csv(list(rows[0])),
                            icon=":material/description:", file_name="runs.columns.csv", mime="text/csv",
                            help="What every column means: the dashboard tables under the names they show, then each "
                                 "column of the runs download, with the links between the two.")
    if many and running and bar.button(f"Stop {len(running)} running", icon=":material/stop_circle:"):
        for name in running:
            stop(name)
        st.toast("Stopping " + ", ".join(running))
        st.rerun()
    with bar.popover(f"Delete {label}".strip(), icon=":material/delete:"):
        st.markdown("Delete " + ", ".join(f"**{n}**" for n in selected) +
                    " with all their runs, code artifacts and scores? This can't be undone.")
        users = sorted({u for n in selected for u in dependents(RUNS, n)} - set(selected))
        if users:
            st.warning("Implementation experiments built on these compositions can't be resumed afterwards: "
                       + ", ".join(users))
        if running:
            st.info("Stop " + ", ".join(running) + " first.")
        if st.button("Delete permanently", type="primary", disabled=bool(running),
                     key="delete-" + "-".join(selected)):
            for name in selected:
                delete_experiment(RUNS / name)
            st.query_params.pop("exp", None)
            bump_list()
            st.toast("Deleted " + ", ".join(selected))
            st.rerun()


def experiment_detail(name: str) -> None:
    root = RUNS / name
    exp = Experiment.load(root)
    running = has_running_runs(name)
    limit_names = {"max_cost_usd": "cost $", "max_seconds": "time (s)", "max_requests": "requests"}
    limits = ", ".join(f"{limit_names.get(k, k)} {v:g}" for k, v in exp.limits.items()) or "none"
    st.header(name, divider="gray")
    st.markdown(f":blue-badge[{PHASE[exp.stage]}] " + (":green-badge[running]" if running else ""))
    facts = {
        "Compositions": "from the requirements" if exp.stage == "design" else exp.design_source,
        "Methods": ", ".join(exp.methods) if exp.stage == "code" else "inlumen",
        "Conditions": ", ".join(c["name"] for c in exp.conditions) or "default (all support on)",
        "Models": ", ".join(exp.models),
        "Cases × replicates": f"{len(exp.cases)} × {exp.repetitions}",
        "Limits per run": limits,
    }
    grid = st.columns(3)
    for i, (k, v) in enumerate(facts.items()):
        grid[i % 3].markdown(f"<small>{k}</small><br>{v}", unsafe_allow_html=True)

    bar = st.container(horizontal=True)
    if bar.button("Resume", icon=":material/play_arrow:", disabled=running,
                  help="Run every run that is not finished (pending, stopped or deleted)."):
        launch_run(root)
        st.rerun()
    if exp.stage == "code":
        if bar.button("Execute pending", icon=":material/fact_check:", disabled=running,
                      help="Pipeline execution: ServiceCraft runs and scores implemented runs not scored yet."):
            start("evaluate", f"exp:{name}", run_cli, root, "evaluate", name)
            st.rerun()
    elif bar.button("Implement these compositions", type="primary", icon=":material/code:",
                    help="Phase 2: launch pipeline implementation from this experiment's valid compositions."):
        st.switch_page(LAUNCH, query_params={"phase": "code", "designs": name})
    if bar.button("Stop", icon=":material/stop_circle:", disabled=not running,
                  help="Kills the run's processes and Docker containers. Unfinished runs rerun on Resume."):
        stop(name)
        st.toast(f"Stopping {name}…")
        st.rerun()
    bulk_actions(bar, [name])

    live_progress(root, name)

    rows = results.episodes(root)
    if exp.stage == "design":
        tabs = st.tabs(["Composition (RQ1)", "Cost and time (RQ3)", f"Runs ({len(rows)})"])
        with tabs[0]:
            composition_results(root)
    else:
        tabs = st.tabs(["Implementation and execution (RQ2)", "Cost and time (RQ3)", f"Runs ({len(rows)})"])
        with tabs[0]:
            implementation_results(root, exp)
    with tabs[1]:
        cost_results(root, exp)
    with tabs[2]:
        runs_tab(root, exp, rows, running)


def _base_columns(df: pd.DataFrame, exp: Experiment) -> dict:
    out = {}
    if exp.stage == "code":
        out["method"] = df["method"]
    out["model"] = df["model"]
    if exp.stage == "code":
        out["compositions"] = [{"reference": "reference", "own": "same run"}.get(v, f"by {v}") for v in df["design"]]
    out["condition"] = df["condition"]
    out["finished"] = df["decided"].astype(str) + "/" + df["episodes"].astype(str)
    out["failed"] = df["generation_failed"]
    out["infra"] = df["infra"]
    return out


def composition_results(root: Path) -> None:
    summary = results.summarize(root)
    if not summary:
        st.info("No runs yet.")
        return
    df = pd.DataFrame(summary)
    exp = Experiment.load(root)
    table = pd.DataFrame({
        **_base_columns(df, exp),
        "valid composition": as_percent(df["valid_design"]),
        "outputs covered": as_percent(df["design_artifact_coverage"]),
        "components found": as_percent(df["design_task_recall"]),
        "one-to-one": as_percent(df["design_task_one_to_one"]),
        "dependencies found": as_percent(df["design_dependency_recall"]),
        "bindings found": as_percent(df["design_binding_recall"]),
        "fully correct": as_percent(df["design_design_correct"]),
        "operations / run": num(df["mean_operations"]),
    })
    show_table("Composition (RQ1)", table, percent=(
        "valid composition", "outputs covered", "components found", "one-to-one", "dependencies found",
        "bindings found", "fully correct"))
    st.caption("A composition is valid when it passes the structural check (every mandatory output written once, "
               "known inputs, no cycles). Comparison with the reference composition matches components by the "
               "outputs they write (provisional): components, dependencies and bindings (which input each "
               "component reads) found, and fully correct compositions. Averages are over valid compositions. "
               "Operations are the design agent's graph operations (tool calls).")
    review_sample(root)


def review_sample(root: Path) -> None:
    """A seeded random sample of compositions for manual review against the reference."""
    with st.expander("Manual review sample", icon=":material/rate_review:"):
        st.caption("Some differences need a person to judge them. Draw a random sample of compositions; each "
                   "run's Composition tab shows the reference next to the generated one and records your verdict.")
        composed = [r for r in results.episodes(root) if r.get("status") == "designed"]
        if not composed:
            st.info("No valid compositions yet.")
            return
        a, b = st.columns(2)
        size = a.number_input("Sample size", 1, len(composed), min(20, len(composed)))
        seed = b.number_input("Seed", value=0, key="review-seed")
        import random
        sample = sorted(random.Random(int(seed)).sample([r["id"] for r in composed], int(size)))
        reviews = {rid: _read_review(root / "episodes" / rid) for rid in sample}
        st.dataframe(pd.DataFrame([{"run": rid, "reviewed": "✓" if rev else "",
                                    "verdict": (rev or {}).get("verdict", "")} for rid, rev in reviews.items()]),
                     hide_index=True, width="stretch")
        st.download_button("Download sample (CSV)", pd.DataFrame({"run": sample}).to_csv(index=False),
                           file_name=f"{root.name}-review-sample.csv", mime="text/csv")


def _read_review(ep_dir: Path) -> dict | None:
    path = ep_dir / "review.json"
    return json.loads(path.read_text()) if path.is_file() else None


def implementation_results(root: Path, exp: Experiment) -> None:
    summary = results.summarize(root)
    if not summary:
        st.info("No runs yet.")
        return
    df = pd.DataFrame(summary)
    k = next((c for c in df.columns if c.startswith("pass@") and c != "pass@1"), None)
    st.markdown("**Implementation**: does the method's own validation pass, and does repair recover failures?")
    show_table("Implementation (RQ2)", pd.DataFrame({
        **_base_columns(df, exp),
        "valid at first attempt": as_percent(df["initial_valid"]),
        "valid at the end": as_percent(df["final_valid"]),
        "recovered by repair": as_percent(df["recovered_by_repair"]),
        "attempts / run": num(df["mean_attempts"]),
    }), percent=("valid at first attempt", "valid at the end", "recovered by repair"))
    st.caption("From the method's validation trace (inLUMEN validates by running the pipeline on sample inputs "
               "in its sandbox). Recovered by repair: share of runs failing their first attempt that pass in the "
               "end. The baseline has no validation, so these are empty for it; compare it on execution below. "
               "To measure ServiceCraft's first-attempt rates, add the condition without repair.")
    st.markdown("**Execution** (ServiceCraft runs the Dagster bundle)")
    show_table("Execution (RQ2)", pd.DataFrame({
        **_base_columns(df, exp),
        "Dagster loaded": as_percent(df["dagster_loaded"]),
        "executable": as_percent(df["dagster_executable"]),
        "outputs valid": as_percent(df["output_contract_valid"]),
        "outputs as expected": as_percent(df["behaviorally_correct"]),
        "complete success": as_percent(df["complete_success"]),
        "mean ± std": [f"{pct(m)} ± {pct(s)}" if m is not None else "—"
                       for m, s in zip(df["success_mean"], df["success_std"])],
        "pass@1": as_percent(df["pass@1"]),
        **({k: as_percent(df[k])} if k else {}),
    }), percent=tuple(c for c in ("Dagster loaded", "executable", "outputs valid", "outputs as expected",
                                  "complete success", "pass@1", k) if c))
    note = ("Rates use every finished run (generation failures count as failures); pending runs and "
            "infrastructure errors are left out. Executable: Dagster completed the run; outputs valid: every "
            "output passed its contract (schema); as expected: outputs match the reference behaviour.")
    if exp.designs:
        note += (" ServiceCraft's graph check expects the reference component names, so code from generated "
                 "compositions usually fails it; compare execution, outputs and behaviour.")
    st.caption(note)

    runs = pd.DataFrame(results.export_rows([root]))
    feedback = [c for c in runs.columns if c.startswith("feedback_")]
    if feedback:
        runs[feedback] = runs[feedback].apply(pd.to_numeric, errors="coerce").fillna(0)
    if feedback and runs[feedback].to_numpy().sum():
        st.markdown("**Failures fed back to repair**, by category")
        chart = (runs.groupby("condition")[feedback].sum().rename(columns=lambda c: c.removeprefix("feedback_"))
                 .reset_index().melt("condition", var_name="category", value_name="failures"))
        st.bar_chart(chart, x="category", y="failures", color="condition", stack=False, height=260)
    final = runs[runs["outcome"] == "failed"]
    if not final.empty:
        st.markdown("**Why failed runs stopped**")
        st.dataframe(final.groupby(["method", "condition", "failure_category"]).size().rename("runs").reset_index(),
                     hide_index=True)


def cost_results(root: Path, exp: Experiment) -> None:
    summary = results.summarize(root)
    if not summary:
        st.info("No runs yet.")
        return
    df = pd.DataFrame(summary)
    success = "valid composition" if exp.stage == "design" else "success"
    show_table("Cost and time (RQ3)", pd.DataFrame({
        **_base_columns(df, exp),
        "cost": [usd(v) + (" + ?" if inc else "") for v, inc in zip(df["cost_usd"], df["cost_incomplete"])],
        "cost / run": [usd(v) for v in df["cost_per_run"]],
        f"cost / {success}": [usd(v) for v in df["cost_per_success"]],
        "tokens / run": num(df["tokens_per_run"]),
        f"tokens / {success}": num(df["tokens_per_success"]),
        "median construction": [secs(v) for v in df["median_construction_s"]],
        **({"median execution": [secs(v) for v in df["median_execution_s"]]} if exp.stage == "code" else {}),
    }))
    st.caption("Failed runs are included in every cost. Construction time covers composition, implementation, "
               "repair, bundling and validation; Dagster execution is reported apart. \"+ ?\" marks cells where "
               "the provider reported no cost for some requests (local models report none).")

    runs = pd.DataFrame(results.export_rows([root]))
    token_cols = [f"{p}_{t}" for p in ("composition", "implementation", "repair") for t in ("input_tokens",
                                                                                         "output_tokens")]
    if not runs.empty:
        runs[token_cols] = runs[token_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    if not runs.empty and runs[token_cols].to_numpy().sum():
        st.markdown("**Mean tokens per run, by phase**")
        by_phase = runs.assign(group=runs["method"] + " · " + runs["condition"])
        long = []
        for phase in ("composition", "implementation", "repair"):
            tokens = by_phase[f"{phase}_input_tokens"].fillna(0) + by_phase[f"{phase}_output_tokens"].fillna(0)
            long.append(pd.DataFrame({"group": by_phase["group"], "phase": phase, "tokens": tokens}))
        chart = pd.concat(long).groupby(["group", "phase"], as_index=False)["tokens"].mean()
        st.bar_chart(chart, x="group", y="tokens", color="phase", horizontal=True, height=220)

    st.markdown("**By complexity of the reference composition**")
    by_level = results.summarize(root, group_by=("method", "condition", "complexity_level"))
    if by_level:
        lv = pd.DataFrame(by_level)
        order = {"low": 0, "medium": 1, "high": 2}
        lv = lv.assign(_o=lv["complexity_level"].map(order)).sort_values(["method", "condition", "_o"])
        rate = "valid_design" if exp.stage == "design" else "complete_success"
        show_table("By complexity (RQ3)", pd.DataFrame({
            "method": lv["method"], "condition": lv["condition"], "complexity": lv["complexity_level"],
            "runs": lv["episodes"], success: as_percent(lv[rate]),
            "tokens / run": num(lv["tokens_per_run"]), f"tokens / {success}": num(lv["tokens_per_success"]),
            "cost / run": [usd(v) for v in lv["cost_per_run"]],
            "median construction": [secs(v) for v in lv["median_construction_s"]],
        }), percent=(success,))
        st.caption("Levels by the number of components in the reference composition: low ≤ 4, medium 5–6, "
                   "high ≥ 7. The exported CSV has every complexity measure (dependencies, longest chain, "
                   "fan-in, fan-out, branching points).")


def runs_tab(root: Path, exp: Experiment, rows: list[dict], running: bool) -> None:
    with st.expander("What the columns of this list mean"):
        st.dataframe(pd.DataFrame(columns.describe_table("Runs")), hide_index=True, width="stretch",
                     column_config={"description": st.column_config.TextColumn(width="large")})
    with st.expander("What the columns of the runs download (CSV) mean"):
        st.dataframe(pd.DataFrame(columns.describe(list(columns.COLUMNS))), hide_index=True, width="stretch",
                     column_config={"description": st.column_config.TextColumn(width="large")})
    f1, f2, f3 = st.columns([2, 1, 1])
    query = f1.text_input("Search", placeholder="case, method, model or error", key=f"q-{root.name}").lower()
    statuses = f2.multiselect("Status", sorted({r.get("status", "pending") for r in rows}),
                              format_func=lambda s: STATUS_LABEL.get(s, s), key=f"s-{root.name}")
    conditions = f3.multiselect("Condition", sorted({r["condition"] for r in rows}), key=f"c-{root.name}")
    shown = [r for r in rows
             if (not query or query in " ".join(str(r.get(k, "")) for k in ("id", "method", "model", "error")).lower())
             and (not statuses or r.get("status", "pending") in statuses)
             and (not conditions or r["condition"] in conditions)]
    is_design = exp.stage == "design"
    table = pd.DataFrame([{
        "run": r["id"], "status": STATUS_LABEL.get(r.get("status"), r.get("status")),
        "case": r["case_id"][:3], **({} if is_design else {"method": r["method"]}), "model": r["model"],
        "condition": r["condition"], "replicate": r["repetition"],
        **({"components found": pct((r.get("design_score") or {}).get("task_recall")),
            "dependencies found": pct((r.get("design_score") or {}).get("dependency_recall")),
            "operations": (r.get("composition_trace") or {}).get("operations")} if is_design else
           {"attempts": (r.get("implementation") or {}).get("attempts"),
            **{label: _score(r, key) for key, label in (("dagster_executable", "executable"),
                                                        ("output_contract_valid", "outputs valid"),
                                                        ("behaviorally_correct", "outputs as expected"),
                                                        ("complete_success", "complete success"))}}),
        "cost": usd((r.get("usage") or {}).get("cost_usd")),
        "time": secs((r.get("budget") or {}).get("elapsed_s")),
        "error": r.get("error") or "",
    } for r in shown])
    if table.empty:
        st.info("No runs match the filters.")
        return
    st.caption(f"{len(shown)} of {len(rows)} runs. Select one to see its details, or several to delete them.")
    picked = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun", selection_mode="multi-row",
                          key=f"runs-{root.name}-{st.session_state.get('list_version', 0)}",
                          column_config=table_config("Runs", table))
    selected = [table.iloc[i]["run"] for i in picked.selection.rows]
    if not selected:
        return
    with st.popover(f"Delete {len(selected)} selected run{'s' if len(selected) > 1 else ''}",
                    icon=":material/delete:"):
        st.markdown("Their files and scores are removed and they become **pending**: Resume runs them again.")
        if running:
            st.info("Stop the experiment first.")
        if exp.stage == "design" and dependents(RUNS, root.name):
            st.warning("Implementation experiments using these compositions can't be resumed afterwards: "
                       + ", ".join(dependents(RUNS, root.name)))
        if st.button("Delete runs", type="primary", disabled=running, key=f"del-runs-{root.name}"):
            count = delete_runs(root, selected)
            bump_list()
            st.toast(f"Deleted {count} runs from {root.name}")
            st.rerun()
    if len(selected) == 1:
        run_detail(root, exp, selected[0])


def _score(row: dict, key: str) -> str:
    value = (row.get("scores") or {}).get(key)
    return "—" if value is None else "✅" if value == 1 else "❌"


@st.fragment(run_every=2)
def live_progress(root: Path, name: str) -> None:
    snap = snapshot(root)
    counts, total = snap["counts"], snap["total"]
    finished = sum(counts.get(k, 0) for k in FINISHED)
    st.progress(finished / total if total else 0.0, text=f"{finished} of {total} runs finished")
    figures = {"Finished": finished, "Running": counts.get("running", 0),
               "Failed": counts.get("generation_failed", 0), "Infrastructure errors": counts.get("infra_error", 0)}
    if Experiment.load(root).stage == "code":
        figures["Awaiting execution"] = counts.get("generated", 0)
    figures["Pending"] = counts.get("pending", 0) + counts.get("stopped", 0)
    for column, (label, value) in zip(st.columns(len(figures)), figures.items()):
        column.metric(label, value)
    if snap["active"]:
        st.dataframe(pd.DataFrame(snap["active"]).rename(columns={"episode": "run", "step": "activity"}),
                     column_order=["run", "model", "server", "phase", "activity", "request", "budget", "elapsed_s"],
                     hide_index=True, width="stretch",
                     column_config={"elapsed_s": st.column_config.NumberColumn("time (s)", format="%.0f")})
    job = jobs().running_for(f"exp:{name}") or jobs().last_for(f"exp:{name}")
    lines = read_log(root, 60) or (job["log"][-60:] if job else [])
    if lines:
        status = job["status"] if job else "from the run log"
        with st.expander(f"Log · {status}", expanded=bool(job) and job["status"] == "failed"):
            st.caption(f"Newest last. The full log is `{root / RUN_LOG}`; a summary line every 30 s "
                       "lists what each running run is doing.")
            st.code("\n".join(lines), language=None)
    # Refresh results and the run table when a run changes state.
    key = f"counts-{name}"
    current = json.dumps(counts, sort_keys=True)
    previous = st.session_state.get(key)
    st.session_state[key] = current
    if previous is not None and previous != current:
        st.rerun()


# ---- one run --------------------------------------------------------------------

def run_detail(root: Path, exp: Experiment, run_id: str) -> None:
    ep_dir = root / "episodes" / run_id
    state = episode_state(ep_dir) or {"status": "pending"}
    st.subheader(run_id, divider="gray")
    if state.get("error"):
        st.error(state["error"])
    c = state.get("conditions") or {}
    st.caption(f"{STATUS_LABEL.get(state.get('status'), state.get('status'))} · condition: "
               f"{results.condition_label({**state, 'stage': exp.stage}) if c else state.get('condition', '—')}"
               + (f" · complexity {state['complexity']['level']} ({state['complexity']['components']} components)"
                  if state.get("complexity") else ""))
    if state.get("status") == "pending":
        st.info("This run hasn't started yet.")
        return
    names = ["Composition"] + ([] if exp.stage == "design" else ["Implementation", "Execution"]) + [
        "Cost and time", "Files"]
    tabs = dict(zip(names, st.tabs(names)))
    with tabs["Composition"]:
        run_composition(root, ep_dir, state, exp)
    if exp.stage == "code":
        with tabs["Implementation"]:
            run_implementation(ep_dir, state)
        with tabs["Execution"]:
            run_execution(state)
    with tabs["Cost and time"]:
        run_cost(state)
    with tabs["Files"]:
        run_files(ep_dir, exp)


def run_composition(root: Path, ep_dir: Path, state: dict, exp: Experiment) -> None:
    design_file = ep_dir / "work" / "design.json"
    score = state.get("design_score")
    comp = state.get("composition")
    if comp:
        m = st.columns(5)
        m[0].metric("Components", comp["components"])
        m[1].metric("Sources", len(comp["sources"]))
        m[2].metric("Destinations", len(comp["destinations"]))
        m[3].metric("Dependencies", comp["dependencies"])
        m[4].metric("Bindings", comp["bindings"])
    if score:
        st.markdown("**Compared with the reference composition** (provisional)")
        m = st.columns(5)
        m[0].metric("Components", f"{score.get('design_tasks')} vs {score.get('reference_tasks')}")
        m[1].metric("Components found", pct(score.get("task_recall")))
        m[2].metric("Dependencies found", pct(score.get("dependency_recall")))
        m[3].metric("Bindings found", pct(score.get("binding_recall")))
        m[4].metric("Fully correct", "yes" if score.get("design_correct") else "no")
        problems = {"Missing components": score.get("missing_components"),
                    "Extra components": score.get("unmatched_design_tasks"),
                    "Split components": score.get("split_tasks"), "Merged components": score.get("merged_tasks"),
                    "Missing dependencies": score.get("missing_dependencies"),
                    "Extra dependencies": score.get("extra_dependencies"),
                    "Missing bindings": score.get("missing_bindings"), "Extra bindings": score.get("extra_bindings"),
                    "Missing outputs": score.get("missing_artifacts")}
        listed = {k: v for k, v in problems.items() if v}
        if listed:
            st.markdown("\n".join(f"- **{k}**: " + ", ".join(f"`{x}`" for x in v) for k, v in listed.items()))
        else:
            st.success("Matches the reference composition.")
    left, right = st.columns(2) if score and design_file.is_file() else (st.container(), None)
    if design_file.is_file():
        source = f" (from `{state['design_episode']}`)" if state.get("design_episode") else ""
        left.markdown(f"**Composition graph{source}**")
        left.graphviz_chart(design_dot(json.loads(design_file.read_text())), width="stretch")
    if right is not None:
        from inlumen_bench.design import from_task_graph
        reference = corpus().root / "corpus/pipelines" / state["case_id"] / "task_graph.json"
        if reference.is_file():
            right.markdown("**Reference composition**")
            right.graphviz_chart(design_dot(from_task_graph(json.loads(reference.read_text()))), width="stretch")
    project = ep_dir / "work" / "project.json"
    if project.is_file():
        st.download_button("Download Project JSON for inLUMEN", project.read_bytes(), icon=":material/download:",
                           file_name=f"{ep_dir.name}.project.json", mime="application/json")
    trace_file = ep_dir / "work" / "composition-trace.json"
    trace = state.get("composition_trace") or {}
    if trace_file.is_file():
        operations = json.loads(trace_file.read_text())
        with st.expander(f"Operation trace · {len(operations)} operations"
                         + (f", {trace.get('failed_operations')} failed" if trace.get("failed_operations") else "")
                         + (" · repaired after validation" if trace.get("repaired") else "")):
            if operations:
                st.dataframe(pd.DataFrame(operations), hide_index=True, width="stretch")
            if trace.get("validation_errors"):
                st.markdown("Validation errors: " + "; ".join(trace["validation_errors"]))
    if exp.stage == "design" and state.get("status") == "designed":
        review_form(ep_dir)


def review_form(ep_dir: Path) -> None:
    review = _read_review(ep_dir) or {}
    with st.expander("Manual review", icon=":material/rate_review:", expanded=False):
        with st.form(f"review-{ep_dir.name}"):
            options = ["correct", "minor differences", "incorrect"]
            verdict = st.radio("Verdict", options, horizontal=True,
                               index=options.index(review["verdict"]) if review.get("verdict") in options else 0)
            a, b, c = st.columns(3)
            comps = a.checkbox("Components correct", value=review.get("components_correct", False))
            deps = b.checkbox("Dependencies correct", value=review.get("dependencies_correct", False))
            binds = c.checkbox("Bindings correct", value=review.get("bindings_correct", False))
            notes = st.text_area("Notes", value=review.get("notes", ""))
            if st.form_submit_button("Save review", type="primary"):
                (ep_dir / "review.json").write_text(json.dumps({
                    "verdict": verdict, "components_correct": comps, "dependencies_correct": deps,
                    "bindings_correct": binds, "notes": notes,
                    "reviewed_at": datetime.now().isoformat(timespec="seconds")}, indent=2))
                st.toast("Review saved")


def run_implementation(ep_dir: Path, state: dict) -> None:
    impl = state.get("implementation") or {}
    c = state.get("conditions") or {}
    st.markdown(" · ".join([
        f"artifact contracts **{'on' if c.get('artifact_contracts', True) else 'off'}**"
        if not c.get("baseline") else "**baseline** (no contracts, no repair)",
        f"repair **{'on, up to ' + str(c.get('max_repair_attempts')) if c.get('repair') else 'off'}**",
        f"validation: {c.get('validation', '—')}"]))
    if not impl:
        st.info("No implementation trace (the run failed before code generation, or it predates traces).")
        return
    m = st.columns(4)
    m[0].metric("Attempts", impl.get("attempts"))
    m[1].metric("First passing attempt", impl.get("first_passing_attempt") or "—")
    m[2].metric("Valid at the end", "—" if impl.get("final_valid") is None else
                "yes" if impl.get("final_valid") else "no")
    m[3].metric("Recovered by repair", "yes" if impl.get("recovered_by_repair") else "no")
    if impl.get("final_failure_reason"):
        st.warning(f"Stopped on a **{impl.get('final_failure_category')}** failure: {impl['final_failure_reason']}")
    trace_file = ep_dir / "work" / "implementation" / "trace.json"
    if trace_file.is_file():
        trace = json.loads(trace_file.read_text())
        st.markdown("**Attempts**")
        st.dataframe(pd.DataFrame([{
            "attempt": a["attempt"], "kind": a["kind"], "outcome": a.get("outcome"),
            "failures": len(a.get("errors") or []),
            "failure categories": ", ".join(f"{k} {v}" for k, v in categorize(a.get("errors") or []).items()),
            "fed back": len(a.get("fed_back") or []), "tokens": (a.get("input_tokens") or 0) + (a.get("output_tokens") or 0),
            "time": secs(a.get("seconds")),
        } for a in trace["attempts"]]), hide_index=True, width="stretch")
        for a in trace["attempts"]:
            if a.get("errors"):
                with st.expander(f"Attempt {a['attempt']}: {len(a['errors'])} validation failures"):
                    st.markdown("\n".join(f"- {e}" for e in a["errors"]))
        if impl.get("final_checks"):
            st.caption("Checks of the final validation: " + ", ".join(impl["final_checks"]))
    stages = impl.get("stage_seconds") or {}
    if stages or impl.get("bundle_seconds") is not None:
        st.caption("Time: " + " · ".join(f"{k.replace('_', ' ')} {secs(v)}" for k, v in stages.items())
                   + (f" · Dagster bundle {secs(impl['bundle_seconds'])}" if impl.get("bundle_seconds") is not None else "")
                   + (f" · public validation {secs(impl['public_validation_seconds'])} ({impl.get('public_validation')})"
                      if impl.get("public_validation_seconds") is not None else ""))


def run_execution(state: dict) -> None:
    stages = state.get("stages") or {}
    ex = state.get("execution") or {}
    if not stages:
        st.info("Not executed yet. Use **Execute pending** on the experiment.")
        return
    mark = {"pass": "✅", "not_reached": "⚪"}
    st.markdown("  ".join(f"{mark.get(v, '❌')} {k.replace('_', ' ')}" for k, v in stages.items())
                + ("  \n⚪ not reached" if "not_reached" in stages.values() else ""))
    if not ex:
        return
    m = st.columns(4)
    m[0].metric("Dagster loaded", "yes" if ex.get("loaded") else "no")
    m[1].metric("Completed", "yes" if ex.get("completed") else "no")
    m[2].metric("Outputs valid", f"{ex.get('outputs_valid', 0)}/{ex.get('outputs_checked', 0)}")
    m[3].metric("Outputs as expected", f"{ex.get('outputs_matching_reference', 0)}/{ex.get('outputs_compared', 0)}")
    if ex.get("static_errors"):
        st.error("Static check: " + "; ".join(ex["static_errors"]))
    if ex.get("failed_components"):
        st.error("Failed components: " + ", ".join(ex["failed_components"]))
    if ex.get("missing_components") or ex.get("unexpected_components"):
        st.warning(f"Missing components: {', '.join(ex.get('missing_components') or []) or '—'} · "
                   f"unexpected: {', '.join(ex.get('unexpected_components') or []) or '—'}")
    if ex.get("outputs_invalid"):
        st.markdown("**Outputs failing their checks**: " + ", ".join(f"`{o}`" for o in ex["outputs_invalid"])
                    + ("  \n" + "; ".join(ex.get("output_errors") or [])))
    st.caption(f"Outputs produced: {', '.join(ex.get('outputs_produced') or []) or 'none'} · Dagster steps "
               f"{secs(ex.get('dagster_step_seconds'))} · evaluation {secs(ex.get('evaluation_seconds'))}")


def run_cost(state: dict) -> None:
    usage = state.get("usage") or {}
    m = st.columns(4)
    m[0].metric("Requests", f"{usage.get('requests', 0)}" + (f" (+{usage['failed_requests']} failed)"
                                                            if usage.get("failed_requests") else ""))
    m[1].metric("Input tokens", f"{usage.get('input_tokens', 0):,}")
    m[2].metric("Output tokens", f"{usage.get('output_tokens', 0):,}")
    m[3].metric("Cost", usd(usage.get("cost_usd")) + (" + ?" if usage.get("cost_unavailable") else ""))
    phases = state.get("phases") or {}
    if phases.get("by_phase"):
        st.markdown("**By phase**")
        st.dataframe(pd.DataFrame([{"phase": p, **v} for p, v in phases["by_phase"].items()]),
                     hide_index=True, width="stretch")
    if phases.get("seconds"):
        st.caption("Wall time: " + " · ".join(f"{k} {secs(v)}" for k, v in phases["seconds"].items()))


def run_files(ep_dir: Path, exp: Experiment) -> None:
    files = sorted(str(p.relative_to(ep_dir)) for p in ep_dir.rglob("*")
                   if p.is_file() and "inputs" not in p.relative_to(ep_dir).parts)
    if not files:
        st.info("No files yet.")
        return
    wanted = "work/design.json" if exp.stage == "design" else "submission/definitions.py"
    default = next((i for i, f in enumerate(files) if f == wanted), 0)
    path = st.selectbox("File", files, index=default, key=f"file-{ep_dir.name}")
    target = ep_dir / path
    st.download_button("Download", target.read_bytes(), file_name=target.name, icon=":material/download:")
    if target.stat().st_size > 2_000_000:
        st.warning("This file is too large to show.")
    elif path.endswith(".jsonl"):
        st.dataframe(pd.DataFrame(read_events(target)[-300:]), hide_index=True, width="stretch")
    else:
        language = {".py": "python", ".json": "json", ".md": "markdown"}.get(target.suffix)
        st.code(target.read_text(errors="replace"), language=language, line_numbers=target.suffix == ".py")


# ---- launch -------------------------------------------------------------------

def launch_page() -> None:
    st.title("Launch an experiment")
    c = corpus()
    profiles = load_profiles()
    if not c.exists():
        st.warning("Fetch the dataset on the Corpus page first.")
        return
    if not profiles:
        st.warning("Add a model endpoint on the Models page first.")

    st.subheader("1 · Phase")
    wanted = st.query_params.get("phase", "design")
    phase = st.segmented_control("Phase", ["design", "code"], format_func=lambda p: PHASE[p], required=True,
                                 default=wanted if wanted in PHASE else "design", label_visibility="collapsed")
    st.caption(PHASE_HELP[phase])

    source, designs, methods = REFERENCE, [], ["inlumen"]
    if phase == "code":
        st.subheader("2 · Compositions and methods")
        left, right = st.columns(2)
        sources = [REFERENCE, *design_experiments()]
        wanted_source = st.query_params.get("designs")
        source = left.selectbox(
            "Compositions to implement", sources, index=sources.index(wanted_source) if wanted_source in sources else 0,
            format_func=lambda n: "Reference compositions (ServiceCraft's reviewed graphs)" if n == REFERENCE
            else f"Valid compositions from {n}")
        methods = right.multiselect("Methods", list(METHODS), default=["inlumen", "direct"],
                                    format_func=lambda m: m + (" (baseline)" if m == "direct" else ""),
                                    help="  \n".join(f"**{m}**: {METHOD_HELP.get(m, '')}" for m in METHODS))
        if source != REFERENCE:
            available = successful_designs(RUNS / source)
            if not available:
                st.warning(f"{source} has no valid compositions yet.")
            design_models = sorted({d["model"] for d in available})
            picked = left.multiselect("Composed by", design_models, default=design_models,
                                      help="Use only the compositions these models made.")
            designs = [d for d in available if d["model"] in picked]

    st.subheader(f"{2 if phase == 'design' else 3} · Support mechanisms (ablation)")
    both = {"on": (True,), "off": (False,), "both": (True, False)}
    level = lambda label, key, help_: st.segmented_control(  # noqa: E731
        label, ["on", "off", "both"], default="on", required=True, key=key, help=help_,
        format_func={"on": "On", "off": "Off", "both": "Both (compare)"}.get)
    repairs = 2
    if phase == "design":
        grounding = level("Metamodel-grounded composition", "grounding",
                          "On: inLUMEN's design agent builds the graph through tools checked against inLUMEN's "
                          "metamodel, validates it and repairs it once. Off: one plain request for the graph as "
                          "JSON. The naming pass and the structural check are the same.")
        conditions = make_conditions("design", grounding=both[grounding])
        repair_level = contracts = "on"
    else:
        a, b, r = st.columns([2, 2, 1])
        with a:
            contracts = level("Contract-bounded generation", "contracts",
                              "On: every component declares its output artifacts (names, kinds) for the code "
                              "generator. Off: components and dependencies only; the requirements still name the "
                              "mandatory outputs.")
        with b:
            repair_level = level("Iterative repair loop", "repair",
                                 "On: failed validation is fed back to the model for up to the maximum attempts. "
                                 "Off: the first attempt is final.")
        repairs = r.number_input("Max repair attempts", 1, 6, 2, disabled=repair_level == "off")
        conditions = make_conditions("code", contracts=both[contracts], repair=both[repair_level], repairs=int(repairs))
        if "inlumen" not in methods and (contracts != "on" or repair_level != "on"):
            st.warning("The support mechanisms belong to the inlumen method; add it to compare them.")
    names = [cond["name"] for cond in conditions] or ["default (all support mechanisms on)"]
    st.markdown(f"{len(names)} condition{'s' if len(names) > 1 else ''}: "
                + " ".join(f":blue-badge[{n}]" for n in names)
                + (" — the direct baseline runs once, without conditions." if phase == "code" and "direct" in methods
                   else ""))

    st.subheader(f"{3 if phase == 'design' else 4} · Models")
    known = st.session_state.setdefault("model_options", [])
    pick, load = st.columns([3, 1], vertical_alignment="bottom")
    profile = pick.selectbox("Endpoint profile", list(profiles) or ["—"])
    if load.button("Load its models", width="stretch", disabled=not profiles):
        from inlumen_bench.llm import list_models
        try:
            loaded = [f"{profile}:{m}" for m in list_models(profiles[profile])]
            st.session_state["model_options"] = sorted(set(known) | set(loaded))
            known = st.session_state["model_options"]
        except Exception as exc:
            st.error(f"Could not list models: {exc}")
    models = st.multiselect("Models (profile:model)", known, accept_new_options=True,
                            placeholder="Pick from the loaded list or type profile:model",
                            help="The paper's design uses 3 models that vary in size and family.")

    st.subheader(f"{4 if phase == 'design' else 5} · Cases")
    cases = [{"id": cid, "number": cid[:3], "topology": t, "category": "A" if int(cid[:3]) <= 13 else "B"}
             for cid, t in ((e["case_id"], e.get("topology")) for e in c.registry()["cases"])]
    if source != REFERENCE:
        with_design = sorted({d["case_id"][:3] for d in designs})
        typed = st.text_input("Case numbers (empty: every case with a composition)", placeholder="001, 016")
        chosen = [x for x in typed.replace(",", " ").split() if x]
        numbers = [n for n in with_design if not chosen or n in chosen]
        designs = [d for d in designs if d["case_id"][:3] in numbers]
        st.caption(f"{len(designs)} compositions over {len(numbers)} cases: {', '.join(numbers) or '—'}")
    else:
        topologies = sorted({x["topology"] for x in cases if x["topology"]})
        presets = [f"All {len(cases)} cases", "Pilot: 10 cases across categories and topologies",
                   "Category A (rich domains, 001–013)", "Category B (sports and transportation)",
                   *[f"Topology: {t.replace('_', ' ')}" for t in topologies], "Custom list"]
        preset = st.selectbox("Selection", presets)
        numbers = _pick_cases(preset, cases)
        if preset == "Custom list":
            typed = st.text_input("Case numbers", placeholder="001, 016, 055")
            numbers = [x for x in typed.replace(",", " ").split() if x]
        else:
            st.caption(f"{len(numbers)} cases" + (f": {', '.join(numbers)}" if len(numbers) <= 20 else ""))

    st.subheader(f"{5 if phase == 'design' else 6} · Replicates and limits")
    a, b, pm, w = st.columns(4)
    reps = a.number_input("Replicates", 1, 50, 5, help="Repetitions of each case × method × model × condition.")
    lanes = sum(len(profiles[n].run_endpoints) for n in {m.partition(":")[0] for m in models}
                if n in profiles)
    workers = b.number_input("Runs in parallel", 1, 32, max(2, lanes),
                             help="Runs at the same time. With a profile that has one Ollama server per GPU "
                                  "(endpoints), set this to the number of servers: every GPU runs a model, and "
                                  "the same model can run on all of them. Otherwise parallel runs go to "
                                  "different models.")
    per_model = pm.number_input("Runs per model at a time", 1, 32, 1,
                                help="Per server: 1 for Ollama, which answers one request per model at a time, "
                                     "so more runs only wait, and the wait counts as their time. "
                                     "Hosted APIs (OpenRouter) can take more.")
    seed = w.number_input("Seed for the run order", value=0)
    if lanes:
        st.caption(f"{lanes} Ollama servers side by side (one per GPU): models run one after another, "
                   f"each on all of them.")
        if int(workers) != lanes * int(per_model):
            st.caption(f":orange[Set runs in parallel to {lanes * int(per_model)} to keep every server busy "
                       f"without queueing.]")
    elif models and int(workers) > len(models) * int(per_model):
        st.caption(f":orange[Only {len(models) * int(per_model)} of the {int(workers)} parallel runs can be busy: "
                   f"{len(models)} model{'s' if len(models) != 1 else ''} × {int(per_model)} run(s) per model.]")
    with st.expander("Limits per run", icon=":material/tune:"):
        l1, l2, l3, l4 = st.columns(4)
        max_cost = l1.number_input("Max cost (USD)", 0.0, value=2.0 if phase == "code" else 0.5, step=0.5)
        max_seconds = l2.number_input("Max time (s)", 60, value=1800 if phase == "code" else 900, step=60)
        max_requests = l3.number_input("Max model requests", 1, value=60 if phase == "code" else 40)
        tool_iterations = l4.number_input("Max agent operations per turn", 5, 100, 30, disabled=phase != "design",
                                          help="Composition: tool calls the design agent may make in one turn.")
        evaluate = phase == "code" and st.checkbox("Execute with ServiceCraft when implementation ends", value=True)

    # ---- summary and start
    units = len(designs) if source != REFERENCE else len(numbers)
    n_cond = max(1, len(conditions))
    per_unit = (n_cond if phase == "design" else
                sum(n_cond if m == "inlumen" else 1 for m in methods))
    total = units * per_unit * len(models) * int(reps)
    st.divider()
    default_name = f"{'composition' if phase == 'design' else 'implementation'}-{datetime.now():%Y%m%d-%H%M}"
    n1, n2 = st.columns([2, 3], vertical_alignment="bottom")
    name = n1.text_input("Experiment name", value=default_name)
    what = f"{units} {'compositions' if source != REFERENCE else 'cases'}"
    n2.info(f"{what} × {per_unit} method-condition{'s' if per_unit != 1 else ''} × {len(models)} model"
            f"{'s' if len(models) != 1 else ''} × {int(reps)} replicates = **{total} runs**")
    problems = [p for p, bad in (
        ("Use letters, digits, dots, dashes or underscores in the name.", not re.match(NAME_RE, name or "")),
        (f"An experiment named {name} exists.", (RUNS / (name or "-") / "experiment.json").exists()),
        ("Pick at least one method.", phase == "code" and not methods),
        ("Add at least one model.", not models),
        ("Pick at least one case.", not units),
        ("Use profile:model with a known profile: " + ", ".join(
            m for m in models if ":" not in m or m.split(":", 1)[0] not in profiles),
         any(":" not in m or m.split(":", 1)[0] not in profiles for m in models)),
    ) if bad]
    for p in problems:
        st.caption(f":orange[● {p}]")
    if st.button(f"Start {total} runs", type="primary", icon=":material/rocket_launch:", disabled=bool(problems)):
        root = RUNS / name
        limits = {"max_cost_usd": float(max_cost), "max_seconds": float(max_seconds),
                  "max_requests": int(max_requests)}
        try:
            if phase == "design":
                create_experiment(root, c, stage="design", name=name, cases=c.resolve_cases(numbers),
                                  models=models, repetitions=int(reps), seed=int(seed), limits=limits,
                                  conditions=conditions,
                                  method_options={"inlumen": {"max_tool_iterations": int(tool_iterations)}},
                                  runs_per_model=int(per_model))
            else:
                # inlumen-standalone reads max_repairs; inLUMEN's code generator, repair_attempts (0-6).
                options = {m: {"max_repairs": int(repairs), "repair_attempts": int(repairs)}
                           for m in methods if m != "direct"}
                create_experiment(root, c, stage="code", design_from=source,
                                  design_models=sorted({d["model"] for d in designs}) or None, name=name,
                                  cases=c.resolve_cases(numbers), methods=methods, models=models,
                                  repetitions=int(reps), seed=int(seed), limits=limits, method_options=options,
                                  conditions=conditions, runs_per_model=int(per_model))
        except (FileExistsError, ValueError, KeyError, CorpusError) as exc:
            st.error(str(exc).strip("'\""))
            return
        launch_run(root, int(workers), evaluate)
        st.switch_page(EXPERIMENTS, query_params={"exp": name})


def _pick_cases(preset: str, cases: list[dict]) -> list[str]:
    if preset.startswith("All"):
        return [c["number"] for c in cases]
    if preset.startswith("Category A") or preset.startswith("Category B"):
        cat = preset.split()[1]
        return [c["number"] for c in cases if c["category"] == cat]
    if preset.startswith("Topology: "):
        topo = preset.removeprefix("Topology: ").replace(" ", "_")
        return [c["number"] for c in cases if c["topology"] == topo]
    if preset.startswith("Pilot"):
        # One case per topology, alternating categories B and A, then fill up to 10.
        a = [c for c in cases if c["category"] == "A"]
        b = [c for c in cases if c["category"] == "B"]
        mixed = [x for pair in zip(b, a) for x in pair] + b[len(a):] + a[len(b):]
        out, seen = [], set()
        for c in mixed:
            if c["topology"] not in seen and len(out) < 10:
                seen.add(c["topology"])
                out.append(c["number"])
        out += [c["number"] for c in mixed if c["number"] not in out][: 10 - len(out)]
        return sorted(out)
    return []


# ---- models -------------------------------------------------------------------

def models_page() -> None:
    st.title("Models")
    profiles = load_profiles()
    if profiles:
        st.dataframe(pd.DataFrame([{
            "name": p.name, "base url": p.base_url, "servers for runs": ", ".join(p.run_endpoints) or "base url",
            "api key": f"env:{p.api_key_env}" if p.api_key_env else (p.masked()["api_key"] or "none"),
            "parallel": p.max_parallel, "timeout (s)": p.timeout_s, "stall (s)": p.stall_s,
        } for p in profiles.values()]), hide_index=True, width="stretch")
    else:
        st.info("No profiles yet. Add OpenRouter, a local Ollama, or any OpenAI-compatible URL below.")

    st.subheader("Test a profile")
    if profiles:
        t1, t2 = st.columns([1, 3], vertical_alignment="bottom")
        name = t1.selectbox("Profile", list(profiles), key="test_profile")
        listed = st.session_state.get(f"models-{name}", [])
        if t2.button("Load model list"):
            from inlumen_bench.llm import list_models
            try:
                listed = list_models(profiles[name])
                st.session_state[f"models-{name}"] = listed
            except Exception as exc:
                st.error(f"Could not list models: {exc}")
        model = (st.selectbox("Model", listed, index=None, accept_new_options=True,
                              placeholder=f"{len(listed)} models, type to search")
                 if listed else st.text_input("Model ID", placeholder="anthropic/claude-sonnet-5.5"))
        if st.button("Send test request"):
            if not model:
                st.error("Enter a model ID first.")
            else:
                from inlumen_bench.llm import ping
                with st.spinner("Waiting for the reply…"):
                    try:
                        r = ping(profiles[name], model)
                        st.success(f"Reply: {r.text.strip()[:200]!r} · {r.latency_s:.1f}s · usage {r.usage_provenance} · "
                                   f"cost {r.cost_usd if r.cost_usd is not None else 'not reported'}. "
                                   f"Use it as `{name}:{model}`.")
                    except Exception as exc:
                        st.error(str(exc))

    st.subheader("Add or edit a profile")
    choice = st.selectbox("Profile to edit", ["New profile", *profiles])
    existing = profiles.get(choice)
    preset = st.selectbox("Preset", list(PRESETS), disabled=existing is not None,
                          help="Fills in sensible defaults for that kind of endpoint.")
    base = asdict(existing) if existing else {**PRESETS[preset], "name": "" if preset == "custom" else preset}
    with st.form("profile"):
        f1, f2 = st.columns(2)
        name = f1.text_input("Name", value=base.get("name", ""), disabled=existing is not None)
        url = f2.text_input("Base URL", value=base.get("base_url", ""))
        key = f1.text_input("API key (optional, stored only in your config folder)", type="password",
                            placeholder="unchanged" if existing and existing.api_key else "leave empty for local servers")
        key_env = f2.text_input("Or read the key from this environment variable",
                                value=base.get("api_key_env") or "", placeholder="OPENROUTER_API_KEY")
        servers = st.text_input(
            "Servers for runs (optional, one per GPU)", value=", ".join(base.get("endpoints") or []),
            placeholder="http://127.0.0.1:11435/v1, http://127.0.0.1:11436/v1",
            help="Several servers with the same models, used side by side: runs go to these instead of "
                 "the base URL. Start one Ollama per GPU with scripts/ollama-gpus.")
        g1, g2, g3 = st.columns(3)
        parallel = g1.number_input("Parallel requests per server", min_value=1, value=int(base.get("max_parallel", 4)))
        timeout = g2.number_input("Timeout (s)", min_value=1, value=int(base.get("timeout_s", 600)))
        stall = g3.number_input("Stall warning after (s without data)", min_value=1, value=int(base.get("stall_s", 120)))
        if st.form_submit_button("Save profile", type="primary"):
            if not re.match(r"^[A-Za-z0-9_.-]{1,40}$", name):
                st.error("Enter a name using letters, digits, dots, dashes or underscores.")
            elif not url.strip():
                st.error("Enter the base URL, for example http://localhost:11434/v1.")
            else:
                upsert_profile(Profile(
                    name=name, base_url=url.strip(),
                    api_key=key or (existing.api_key if existing else None),
                    api_key_env=key_env.strip() or None,
                    headers=existing.headers if existing else {},
                    max_parallel=int(parallel), timeout_s=float(timeout), stall_s=float(stall),
                    endpoints=[u.strip() for u in servers.replace(",", " ").split() if u.strip()]))
                st.toast(f"Saved {name}")
                st.rerun()
    if existing and st.button(f"Remove {existing.name}", type="secondary"):
        remove_profile(existing.name)
        st.rerun()


# ---- corpus -------------------------------------------------------------------

def corpus_page() -> None:
    st.title("Corpus")
    data = resolve_data_dir()
    c = Corpus(data.path)
    st.caption(f"Data folder (from {data.source}): `{data.path}`  ·  runs: `{RUNS}`  ·  inlumen-bench {__version__}")
    if not c.exists():
        st.info("The ServiceCraft dataset isn't in the data folder yet.")
        if st.button("Fetch dataset", type="primary"):
            start("fetch", "fetch", lambda log, stop: (log("cloning servicecraft"), fetch(data.path), log("done")))
        corpus_jobs()
        return
    q = c.setup_status()["qualified"]
    meta = c.metadata()
    st.markdown(
        f"ServiceCraft · {len(c.case_ids())} cases · commit `{(meta.get('commit') or '?')[:12]}`"
        + (" (modified)" if meta.get("dirty") else "") + "  ·  evaluator: "
        + (f"ready ({q['qualified_cases']}/{q['case_count']} cases)" if q else "**not prepared**, see the Evaluator tab"))
    cases_tab, evaluator_tab = st.tabs(["Cases", "Evaluator"])
    with cases_tab:
        case_browser(c)
    with evaluator_tab:
        evaluator_setup(c, q)


@st.cache_data(show_spinner=False)
def case_table(root: str, commit: str) -> pd.DataFrame:
    """One row per case, from each case's task_graph.json. Cached per dataset commit."""
    from inlumen_bench.metrics import complexity
    rows = []
    for case_dir in sorted((Path(root) / "corpus/pipelines").iterdir()):
        graph_file = case_dir / "task_graph.json"
        if not graph_file.is_file():
            continue
        g = json.loads(graph_file.read_text())
        cx = complexity(g)
        rows.append({"case": case_dir.name[:3], "title": g.get("title", ""), "domain": g.get("domain", ""),
                     "topology": g.get("topology", ""), "complexity": cx["level"], "components": cx["components"],
                     "dependencies": cx["dependencies"], "longest chain": cx["longest_chain"],
                     "inputs": cx["inputs"], "category": "A" if int(case_dir.name[:3]) <= 13 else "B",
                     "id": case_dir.name})
    return pd.DataFrame(rows)


def case_browser(c: Corpus) -> None:
    table = case_table(str(c.root), c.version())
    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    text = f1.text_input("Search", placeholder="title, domain or case number")
    domains = f2.multiselect("Domain", sorted(table["domain"].unique()))
    topologies = f3.multiselect("Topology", sorted(table["topology"].unique()))
    levels = f4.multiselect("Complexity", ["low", "medium", "high"])
    shown = table
    if text:
        needle = text.lower()
        shown = shown[shown.apply(lambda r: needle in f"{r['case']} {r['title']} {r['domain']}".lower(), axis=1)]
    for column, chosen in (("domain", domains), ("topology", topologies), ("complexity", levels)):
        if chosen:
            shown = shown[shown[column].isin(chosen)]
    st.caption(f"{len(shown)} of {len(table)} cases. Select a row to see its requirements and reference composition.")
    picked = st.dataframe(shown.drop(columns="id"), hide_index=True, width="stretch", height=min(320, 35 * (len(shown) + 1) + 3),
                          on_select="rerun", selection_mode="single-row", key="case_table")
    rows = picked.selection.rows
    if rows:
        case_detail(c, shown.iloc[rows[0]]["id"])


def case_detail(c: Corpus, case_id: str) -> None:
    from inlumen_bench.design import from_task_graph, to_project_json
    case_dir = c.root / "corpus/pipelines" / case_id
    graph = json.loads((case_dir / "task_graph.json").read_text())
    st.subheader(f"{case_id[:3]} · {graph.get('title', '')}")
    st.caption(f"`{case_id}` · {graph.get('domain', '')} · {graph.get('topology', '')} · "
               f"{len(graph['tasks'])} components · review status: {graph.get('review_status', '?')}")
    design = from_task_graph(graph)
    project = to_project_json(design, f"{case_id[:3]} {graph.get('title', '')}".strip(),
                              "ServiceCraft reference pipeline")
    st.download_button("Download the reference composition as inLUMEN Project JSON",
                       json.dumps(project, indent=2), file_name=f"{case_id}.inlumen.json",
                       mime="application/json", icon=":material/download:")
    req_tab, graph_tab, comp_tab, inputs_tab, desc_tab = st.tabs(
        ["Requirements", "Reference composition", "Components", "Inputs", "Description"])
    with req_tab:
        st.caption("The task's requirements: what every method receives, in both phases.")
        st.code(c.prompt(case_id), language=None, wrap_lines=True)
    with graph_tab:
        st.caption("Methods see it only when implementing reference compositions.")
        st.graphviz_chart(design_dot(design), width="stretch")
    with comp_tab:
        st.dataframe(pd.DataFrame([{
            "component": t["id"], "name": t.get("name", ""), "operation": t.get("operation", ""),
            "inputs": ", ".join(t.get("inputs", [])), "outputs": ", ".join(t.get("outputs", [])),
            "requirements": " ".join(t.get("requirements", [])),
        } for t in graph["tasks"]]), hide_index=True, width="stretch")
    with inputs_tab:
        for i in graph["inputs"]:
            path = case_dir / i["path"]
            st.markdown(f"**{i['path']}** · required columns: {', '.join(i.get('required_columns', [])) or '—'}")
            if path.suffix == ".csv" and path.is_file():
                frame = pd.read_csv(path)
                st.caption(f"{len(frame)} rows × {len(frame.columns)} columns; first 50 rows")
                st.dataframe(frame.head(50), hide_index=True, width="stretch")
            elif path.is_file():
                st.code(path.read_text(errors="replace")[:5000], language=None)
    with desc_tab:
        st.caption("The authors' description of the case. Methods do not receive it.")
        description = case_dir / "description.md"
        st.markdown(description.read_text() if description.is_file() else "No description.")


def evaluator_setup(c: Corpus, q: dict | None) -> None:
    st.markdown(
        "Pipeline execution runs each bundle in ServiceCraft's evaluator, which needs two Docker images and a "
        "**qualified corpus**: proof that each case's reference solution passes all checks on this machine.\n\n"
        "ServiceCraft ships its own qualification, but it is tied to Docker image IDs from the authors' computer, "
        "so it can't be used elsewhere. Preparing the evaluator builds the images and reruns the qualification "
        "here, following ServiceCraft's documented procedure (about 7 minutes). The result goes to "
        f"`{c.qualified_root}`; the dataset itself is not changed.\n\n"
        "You can browse cases and run composition experiments without it, but implementations can't be executed "
        "and scored until it is done. It's needed once per machine, and again after fetching a new dataset "
        "version or rebuilding the images.")
    status = c.setup_status()
    m1, m2 = st.columns(2)
    m1.markdown("**Docker images**\n\n" + "\n".join(
        f"- `{img}` {'✅ built' if image_id else '❌ missing'}" for img, image_id in status["images"].items()))
    m2.markdown("**Qualified corpus**\n\n" + (
        f"✅ {q['qualified_cases']}/{q['case_count']} cases, finished {q['finished_at'].replace('T', ' ')[:16]} UTC"
        if q else "❌ not prepared"))
    problems = c.check()
    if problems:
        st.error("Dataset check failed:\n\n" + "\n".join(f"- {p}" for p in problems[:20]))
    else:
        st.caption("Dataset check passed: all prompts match the manifest.")
    if st.button("Prepare the evaluator again" if q else "Prepare the evaluator", type="secondary" if q else "primary",
                 disabled=bool(problems)):
        def work(log, stop):
            summary = setup(corpus(), log=log)
            log(f"qualified {summary['qualified_cases']}/{summary['case_count']} cases")
        start("setup", "setup", work)
    corpus_jobs()


@st.fragment(run_every=2)
def corpus_jobs() -> None:
    job = jobs().running_for("setup") or jobs().running_for("fetch") or jobs().last_for("setup") or jobs().last_for("fetch")
    if job:
        st.code(f"{job['label']}: {job['status']}\n\n" + "\n".join(job["log"]), language=None)


EXPERIMENTS = st.Page(experiments_page, title="Experiments", icon=":material/science:", default=True)
LAUNCH = st.Page(launch_page, url_path="launch", title="Launch", icon=":material/rocket_launch:")
MODELS = st.Page(models_page, url_path="models", title="Models", icon=":material/dns:")
CORPUS = st.Page(corpus_page, url_path="corpus", title="Corpus", icon=":material/dataset:")
st.navigation([EXPERIMENTS, LAUNCH, MODELS, CORPUS]).run()
