"""Streamlit dashboard: corpus, models, launching experiments, live progress and results.

Started by `bench serve`. It has no login and can start paid model runs, so keep it
on localhost. State lives in the same files the CLI uses.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from inlumen_bench import __version__, results
from inlumen_bench.config import resolve_data_dir
from inlumen_bench.corpus import Corpus, CorpusError, fetch, setup
from inlumen_bench.events import read_events
from inlumen_bench.jobs import JobBusy, Jobs, snapshot
from inlumen_bench.methods import METHODS
from inlumen_bench.providers import PRESETS, Profile, load_profiles, remove_profile, upsert_profile
from inlumen_bench.runner import Experiment, create_experiment, episode_state, run_experiment

RUNS = Path(os.environ.get("INLUMEN_BENCH_RUNS_DIR", "runs")).resolve()
NAME_RE = r"^[A-Za-z0-9_.-]{1,60}$"
STATUS_ICON = {"evaluated": "🟢", "generated": "🔵", "running": "🔵", "generation_failed": "🔴",
               "infra_error": "🟠", "pending": "⚪"}

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


def run_job(log, stop, root: Path, workers: int, evaluate: bool) -> None:
    from inlumen_bench.evaluate import evaluate_experiment

    c = corpus()
    log("generation started")
    run_experiment(root, c, workers=workers, stop=stop)
    if stop.is_set():
        log("stopped; finished episodes are kept, resume to continue")
        return
    log("generation finished")
    if evaluate:
        evaluate_experiment(root, c, log=log)
    log("done")


def start(kind: str, label: str, fn, *args) -> None:
    try:
        jobs().start(kind, label, fn, *args)
    except JobBusy as exc:
        st.warning(str(exc))


# ---- experiments --------------------------------------------------------------

def experiments_page() -> None:
    names = experiment_names()
    head, button = st.columns([4, 1], vertical_alignment="bottom")
    head.title("Experiments")
    if button.button("New experiment", type="primary", width="stretch"):
        st.switch_page(LAUNCH)
    if not names:
        st.info("No experiments yet. Start one on the Launch page, or with `bench run`.")
        return

    rows = []
    for name in names:
        exp = Experiment.load(RUNS / name)
        counts: dict[str, int] = {}
        for ep in exp.episodes():
            status = (episode_state(RUNS / name / "episodes" / ep["id"]) or {}).get("status", "pending")
            counts[status] = counts.get(status, 0) + 1
        total = sum(counts.values())
        finished = counts.get("evaluated", 0) + counts.get("generation_failed", 0)
        rows.append({"name": name, "running": bool(jobs().running_for(f"exp:{name}")),
                     "progress": finished / total if total else 0.0, "episodes": total,
                     "methods": ", ".join(exp.methods), "models": ", ".join(exp.models),
                     "tracks": ", ".join(exp.tracks), "cases × reps": f"{len(exp.cases)} × {exp.repetitions}",
                     "created": exp.created_at.replace("T", " ")[:16]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "progress": st.column_config.ProgressColumn("finished", min_value=0, max_value=1, format="percent"),
        "running": st.column_config.CheckboxColumn("running")})

    wanted = st.query_params.get("exp")
    name = st.selectbox("Experiment", names, index=names.index(wanted) if wanted in names else 0)
    st.query_params["exp"] = name
    experiment_detail(name)


def experiment_detail(name: str) -> None:
    root = RUNS / name
    exp = Experiment.load(root)
    label = f"exp:{name}"
    limits = ", ".join(f"{k.removeprefix('max_')} {v}" for k, v in exp.limits.items()) or "none"
    st.caption(f"{len(exp.cases)} cases · {', '.join(exp.methods)} · {', '.join(exp.models)} · "
               f"{', '.join(exp.tracks)} · {exp.repetitions} repetitions · limits {limits} · "
               f"dataset {(exp.dataset.get('commit') or '')[:12]}")

    a, b, c, _ = st.columns([1, 1, 1, 3])
    workers = 2
    if a.button("Resume", width="stretch", help="Run the episodes that are not finished yet."):
        start("experiment", label, run_job, root, workers, True)
    if b.button("Evaluate pending", width="stretch", help="Score generated episodes with ServiceCraft."):
        from inlumen_bench.evaluate import evaluate_experiment
        start("evaluate", label, lambda log, stop: evaluate_experiment(root, corpus(), log=log))
    if c.button("Stop", width="stretch", help="Stop after the episodes that are running now."):
        st.toast("Stopping after the running episodes." if jobs().stop(label)
                 else "Not running in this dashboard.")

    live_progress(root, name)

    st.subheader("Results")
    summary = results.summarize(root)
    if summary:
        df = pd.DataFrame(summary)
        # Missing rates become NaN, which the table shows as empty instead of "None".
        rate_cols = [c for c in df.columns if c in results.SCORE_KEYS or c.startswith(("pass@", "design_"))]
        df[rate_cols] = df[rate_cols].apply(pd.to_numeric, errors="coerce")
        k = next((c for c in df.columns if c.startswith("pass@") and c != "pass@1"), None)
        table = pd.DataFrame({
            "method": df["method"], "model": df["model"], "track": df["track"].str.replace("_", " "),
            "finished": df["decided"].astype(str) + "/" + df["episodes"].astype(str),
            "gen. failed": df["generation_failed"], "infra": df["infra"],
            "executable": df["dagster_executable"] * 100, "outputs valid": df["output_contract_valid"] * 100,
            "behaviour": df["behaviorally_correct"] * 100, "complete success": df["complete_success"] * 100,
            "mean ± std": [f"{pct(m)} ± {pct(s)}" if m is not None else "—"
                           for m, s in zip(df["success_mean"], df["success_std"])],
            "pass@1": df["pass@1"] * 100,
            **({k: df[k] * 100} if k else {}),
            "design correct": df["design_design_correct"] * 100,
            "dependency recall": df["design_dependency_recall"] * 100,
            "cost": [usd(v) + (" + ?" if inc else "") for v, inc in zip(df["cost_usd"], df["cost_incomplete"])],
            "cost / success": [usd(v) for v in df["cost_per_success"]],
            "median time (s)": df["median_generation_s"],
        })
        percent = st.column_config.NumberColumn(format="%.1f%%")
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            c: percent for c in ("executable", "outputs valid", "behaviour", "complete success", "pass@1",
                                 k, "design correct", "dependency recall") if c})
        st.caption("Rates use every finished episode (generation failures count as failures); pending and "
                   "infrastructure errors are left out. Design scores are provisional: tasks are matched "
                   "to the reference graph by the artifacts they write.")

    st.subheader("Episodes")
    rows = results.episodes(root)
    query = st.text_input("Filter", placeholder="case, method, model, track or status").lower()
    shown = [r for r in rows if not query or query in " ".join(
        str(r.get(k, "")) for k in ("id", "method", "model", "track", "status")).lower()]
    table = pd.DataFrame([{
        "episode": r["id"], "status": f"{STATUS_ICON.get(r.get('status'), '')} {str(r.get('status', '')).replace('_', ' ')}",
        **{label_: _score(r, key) for key, label_ in (("dagster_executable", "executable"),
                                                       ("output_contract_valid", "outputs"),
                                                       ("behaviorally_correct", "behaviour"),
                                                       ("complete_success", "success"))},
        "design": pct((r.get("design_score") or {}).get("dependency_recall")) if r.get("design_score") else "—",
        "repairs": (r.get("counters") or {}).get("repairs"),
        "cost": usd((r.get("usage") or {}).get("cost_usd")),
        "time (s)": (r.get("budget") or {}).get("elapsed_s"),
        "error": r.get("error") or "",
    } for r in shown])
    if table.empty:
        st.info("No episodes match the filter.")
        return
    picked = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                          selection_mode="single-row", key=f"episodes-{name}")
    selected = picked.selection.rows
    if selected:
        episode_detail(root, table.iloc[selected[0]]["episode"])
    else:
        st.caption("Select a row to see the episode's stages, usage and files.")


def _score(row: dict, key: str) -> str:
    value = (row.get("scores") or {}).get(key)
    return "—" if value is None else "✅" if value == 1 else "❌"


@st.fragment(run_every=2)
def live_progress(root: Path, name: str) -> None:
    snap = snapshot(root)
    counts, total = snap["counts"], snap["total"]
    finished = counts.get("evaluated", 0) + counts.get("generation_failed", 0)
    st.progress(finished / total if total else 0.0,
                text=f"{finished}/{total} finished · " + " · ".join(
                    f"{k.replace('_', ' ')} {v}" for k, v in sorted(counts.items())))
    if snap["active"]:
        st.dataframe(pd.DataFrame(snap["active"]), hide_index=True, width="stretch", column_config={
            "elapsed_s": st.column_config.NumberColumn("time (s)", format="%.0f")})
    job = jobs().running_for(f"exp:{name}") or jobs().last_for(f"exp:{name}")
    if job:
        st.code(f"{job['status']}\n" + "\n".join(job["log"][-12:]), language=None)
    # Refresh results and the episode table when an episode changes state.
    key = f"counts-{name}"
    current = json.dumps(counts, sort_keys=True)
    previous = st.session_state.get(key)
    st.session_state[key] = current
    if previous is not None and previous != current:
        st.rerun()


def episode_detail(root: Path, episode_id: str) -> None:
    ep_dir = root / "episodes" / episode_id
    state = episode_state(ep_dir) or {}
    st.markdown(f"#### `{episode_id}`")
    if state.get("error"):
        st.error(state["error"])
    cols = st.columns(4)
    cols[0].markdown("**Status**\n\n" + f"{STATUS_ICON.get(state.get('status'), '')} {state.get('status', '—')}")
    stages = state.get("stages") or {}
    cols[1].markdown("**ServiceCraft stages**\n\n" + ("\n".join(
        f"- {k}: {'✅' if v == 'pass' else '❌'} {v}" for k, v in stages.items()) or "—"))
    usage = state.get("usage") or {}
    cols[2].markdown("**Usage**\n\n" + (f"- requests {usage.get('requests')} ({usage.get('failed_requests')} failed)\n"
                                        f"- input {usage.get('input_tokens')} · output {usage.get('output_tokens')}\n"
                                        f"- cost {usd(usage.get('cost_usd'))}" + (" + ?" if usage.get("cost_unavailable") else "")
                                        if usage else "—"))
    counters = state.get("counters") or {}
    cols[3].markdown("**Counters**\n\n" + ("\n".join(f"- {k} {v}" for k, v in counters.items()) or "—"))
    if state.get("design_score"):
        with st.expander("Design compared with the reference (provisional)", expanded=True):
            st.json(state["design_score"])

    project = ep_dir / "work" / "project.json"
    if project.is_file():
        st.download_button("Download Project JSON for inLUMEN", project.read_bytes(),
                           file_name=f"{episode_id}.project.json", mime="application/json", type="primary")

    files = sorted(str(p.relative_to(ep_dir)) for p in ep_dir.rglob("*")
                   if p.is_file() and "inputs" not in p.relative_to(ep_dir).parts)
    default = next((i for i, f in enumerate(files) if f == "submission/definitions.py"), 0)
    path = st.selectbox("File", files, index=default, key=f"file-{episode_id}")
    target = ep_dir / path
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
    if not profiles:
        st.warning("Add a model endpoint on the Models page first.")
    if not c.exists():
        st.warning("Fetch the dataset on the Corpus page first.")
        return
    registry = {e["case_id"]: e for e in c.registry()["cases"]}
    cases = [{"id": cid, "number": cid[:3], "topology": registry.get(cid, {}).get("topology"),
              "category": "A" if int(cid[:3]) <= 13 else "B"} for cid in c.case_ids()]

    left, right = st.columns(2)
    name = left.text_input("Name", placeholder="pilot-1")
    reps = right.number_input("Repetitions per model", min_value=1, max_value=50, value=3)
    methods = left.multiselect("Methods", list(METHODS), default=["inlumen"],
                               help="  \n".join(f"**{m.name}**: {_summary(m)}" for m in METHODS.values()))
    tracks = right.multiselect("Tracks", ["fixed_graph", "natural_language"], default=["fixed_graph"],
                               format_func=lambda t: {"fixed_graph": "fixed graph (code generation)",
                                                      "natural_language": "natural language (design + code)"}[t])

    st.subheader("Models")
    known = st.session_state.setdefault("model_options", [])
    pick, load = st.columns([3, 1], vertical_alignment="bottom")
    profile = pick.selectbox("Endpoint profile", list(profiles) or ["—"])
    if load.button("Load its model list", width="stretch", disabled=not profiles):
        from inlumen_bench.llm import list_models
        try:
            loaded = [f"{profile}:{m}" for m in list_models(profiles[profile])]
            st.session_state["model_options"] = sorted(set(known) | set(loaded))
            known = st.session_state["model_options"]
        except Exception as exc:
            st.error(f"Could not list models: {exc}")
    models = st.multiselect("Models (profile:model)", known, accept_new_options=True,
                            placeholder="Pick from the loaded list or type profile:model")

    st.subheader("Cases")
    topologies = sorted({x["topology"] for x in cases if x["topology"]})
    presets = ["Pilot: 10 cases across categories and topologies", f"All {len(cases)} cases",
               "Category A (rich domains, 001–013)", "Category B (sports and transportation)",
               *[f"Topology: {t.replace('_', ' ')}" for t in topologies], "Custom list"]
    preset = st.selectbox("Selection", presets)
    numbers = _pick_cases(preset, cases, topologies)
    if preset == "Custom list":
        typed = st.text_input("Case numbers", placeholder="001, 016, 055")
        numbers = [x for x in typed.replace(",", " ").split() if x]
    else:
        st.caption(", ".join(numbers))

    st.subheader("Limits per episode")
    l1, l2, l3, l4, l5 = st.columns(5)
    max_cost = l1.number_input("Max cost (USD)", min_value=0.0, value=2.0, step=0.5)
    max_seconds = l2.number_input("Max time (s)", min_value=60, value=1800, step=60)
    max_requests = l3.number_input("Max model requests", min_value=1, value=60)
    max_repairs = l4.number_input("Max repairs (inlumen)", min_value=0, max_value=10, value=3)
    workers = l5.number_input("Episodes in parallel", min_value=1, max_value=32, value=2)
    seed = st.number_input("Random seed for the run order", value=0)
    evaluate = st.checkbox("Score with ServiceCraft when generation ends", value=True)

    episodes = len(numbers) * len(methods) * len(models) * len(tracks) * int(reps)
    st.info(f"{len(numbers)} cases × {len(methods)} methods × {len(models)} models × {len(tracks)} tracks × "
            f"{int(reps)} repetitions = **{episodes} episodes**")
    if st.button("Start experiment", type="primary"):
        import re
        error = ("Enter a name using letters, digits, dots, dashes or underscores." if not re.match(NAME_RE, name)
                 else "Pick at least one method." if not methods else "Pick at least one track." if not tracks
                 else "Add at least one model." if not models else "Pick at least one case." if not numbers else "")
        bad = [m for m in models if ":" not in m or m.split(":", 1)[0] not in profiles]
        if not error and bad:
            error = f"Use profile:model with a known profile: {', '.join(bad)}"
        if error:
            st.error(error)
            return
        root = RUNS / name
        options = {m: {"max_repairs": int(max_repairs)} for m in methods if m != "direct"}
        try:
            create_experiment(root, c, name=name, cases=c.resolve_cases(numbers), methods=methods, models=models,
                              tracks=tracks, repetitions=int(reps), seed=int(seed),
                              limits={"max_cost_usd": float(max_cost), "max_seconds": float(max_seconds),
                                      "max_requests": int(max_requests)}, method_options=options)
        except (FileExistsError, ValueError, KeyError, CorpusError) as exc:
            st.error(str(exc).strip("'\""))
            return
        start("experiment", f"exp:{name}", run_job, root, int(workers), evaluate)
        st.query_params["exp"] = name
        st.switch_page(EXPERIMENTS, query_params={"exp": name})


def _summary(method) -> str:
    """First line of the method's module docstring."""
    import sys
    doc = sys.modules[method.__module__].__doc__ or method.__doc__ or ""
    return doc.strip().splitlines()[0] if doc.strip() else ""


def _pick_cases(preset: str, cases: list[dict], topologies: list[str]) -> list[str]:
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
            "name": p.name, "base url": p.base_url,
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
        g1, g2, g3 = st.columns(3)
        parallel = g1.number_input("Parallel requests", min_value=1, value=int(base.get("max_parallel", 4)))
        timeout = g2.number_input("Timeout (s)", min_value=1, value=int(base.get("timeout_s", 600)))
        stall = g3.number_input("Stall warning after (s without data)", min_value=1, value=int(base.get("stall_s", 120)))
        if st.form_submit_button("Save profile", type="primary"):
            import re
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
                    max_parallel=int(parallel), timeout_s=float(timeout), stall_s=float(stall)))
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
    rows = []
    for case_dir in sorted((Path(root) / "corpus/pipelines").iterdir()):
        graph_file = case_dir / "task_graph.json"
        if not graph_file.is_file():
            continue
        g = json.loads(graph_file.read_text())
        rows.append({"case": case_dir.name[:3], "title": g.get("title", ""), "domain": g.get("domain", ""),
                     "topology": g.get("topology", ""), "tasks": len(g.get("tasks", [])),
                     "inputs": len(g.get("inputs", [])), "category": "A" if int(case_dir.name[:3]) <= 13 else "B",
                     "id": case_dir.name})
    return pd.DataFrame(rows)


def case_browser(c: Corpus) -> None:
    table = case_table(str(c.root), c.version())
    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    text = f1.text_input("Search", placeholder="title, domain or case number")
    domains = f2.multiselect("Domain", sorted(table["domain"].unique()))
    topologies = f3.multiselect("Topology", sorted(table["topology"].unique()))
    categories = f4.multiselect("Category", ["A", "B"])
    shown = table
    if text:
        needle = text.lower()
        shown = shown[shown.apply(lambda r: needle in f"{r['case']} {r['title']} {r['domain']}".lower(), axis=1)]
    for column, chosen in (("domain", domains), ("topology", topologies), ("category", categories)):
        if chosen:
            shown = shown[shown[column].isin(chosen)]
    st.caption(f"{len(shown)} of {len(table)} cases. Select a row to see its prompt and pipeline.")
    picked = st.dataframe(shown.drop(columns="id"), hide_index=True, width="stretch", height=min(320, 35 * (len(shown) + 1) + 3),
                          on_select="rerun", selection_mode="single-row", key="case_table")
    rows = picked.selection.rows
    if rows:
        case_detail(c, shown.iloc[rows[0]]["id"])


def case_detail(c: Corpus, case_id: str) -> None:
    from inlumen_bench.design import edges, from_task_graph, to_project_json
    case_dir = c.root / "corpus/pipelines" / case_id
    graph = json.loads((case_dir / "task_graph.json").read_text())
    st.subheader(f"{case_id[:3]} · {graph.get('title', '')}")
    st.caption(f"`{case_id}` · {graph.get('domain', '')} · {graph.get('topology', '')} · "
               f"{len(graph['tasks'])} tasks · review status: {graph.get('review_status', '?')}")
    design = from_task_graph(graph)
    project = to_project_json(design, f"{case_id[:3]} {graph.get('title', '')}".strip(),
                              "ServiceCraft reference pipeline")
    st.download_button("Download reference pipeline as inLUMEN Project JSON",
                       json.dumps(project, indent=2), file_name=f"{case_id}.inlumen.json",
                       mime="application/json", icon=":material/download:")
    prompt_tab, graph_tab, tasks_tab, inputs_tab, desc_tab = st.tabs(
        ["Prompt", "Pipeline", "Tasks", "Input data", "Description"])
    with prompt_tab:
        st.caption("What every method receives, on both tracks.")
        st.code(c.prompt(case_id), language=None, wrap_lines=True)
    with graph_tab:
        st.caption("The reference task graph. Methods see it only on the fixed-graph track.")
        lines = ['digraph { rankdir=LR; node [fontname="Helvetica" fontsize=11]; edge [fontname="Helvetica" fontsize=9];']
        for i in graph["inputs"]:
            lines.append(f'"{i["id"]}" [shape=note label="{i["path"].split("/")[-1]}" style=filled fillcolor="#eef3f8"];')
        for t in graph["tasks"]:
            lines.append(f'"{t["id"]}" [shape=box style="rounded,filled" fillcolor="#dbe7f3" label="{t["id"]}"];')
            for item in t.get("inputs", []):
                if item.startswith("input:"):
                    lines.append(f'"{item}" -> "{t["id"]}";')
        lines += [f'"{a}" -> "{b}" [label="{art}"];' for a, b, art in edges(design)]
        st.graphviz_chart("\n".join(lines + ["}"]), width="stretch")
    with tasks_tab:
        st.dataframe(pd.DataFrame([{
            "task": t["id"], "name": t.get("name", ""), "operation": t.get("operation", ""),
            "reads": ", ".join(t.get("inputs", [])), "writes": ", ".join(t.get("outputs", [])),
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
        "Scoring a submission runs it in ServiceCraft's evaluator, which needs two Docker images and a "
        "**qualified corpus**: proof that each case's reference solution passes all checks on this machine.\n\n"
        "ServiceCraft ships its own qualification, but it is tied to Docker image IDs from the authors' computer, "
        "so it can't be used elsewhere. Preparing the evaluator builds the images and reruns the qualification "
        "here, following ServiceCraft's documented procedure (about 7 minutes). The result goes to "
        f"`{c.qualified_root}`; the dataset itself is not changed.\n\n"
        "You can browse cases and generate submissions without it, but experiments can't be scored until it is done. "
        "It's needed once per machine, and again after fetching a new dataset version or rebuilding the images.")
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
