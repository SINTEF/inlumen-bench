"""What the bench records for the research questions: conditions, traces, complexity, costs."""

import json

import pytest

from inlumen_bench import design_score, metrics
from inlumen_bench.results import condition_label, export_rows, run_row, to_csv
from inlumen_bench.runner import Experiment, make_conditions

REFERENCE = {
    "topology": "fan_in",
    "inputs": [{"id": "input:a.csv"}, {"id": "input:b.csv"}],
    "tasks": [
        {"id": "load_a", "inputs": ["input:a.csv"], "outputs": ["a_clean.csv"]},
        {"id": "load_b", "inputs": ["input:b.csv"], "outputs": ["b_clean.csv"]},
        {"id": "join", "inputs": ["a_clean.csv", "b_clean.csv"], "outputs": ["joined.csv"]},
        {"id": "report", "inputs": ["joined.csv"], "outputs": ["report.json"]},
    ],
}


def test_complexity_is_measured_on_the_reference_composition():
    c = metrics.complexity(REFERENCE)
    assert (c["components"], c["dependencies"], c["inputs"], c["longest_chain"]) == (4, 3, 2, 3)
    assert (c["max_fan_in"], c["max_fan_out"], c["branching_points"], c["level"]) == (2, 1, 1, "low")


def test_bindings_and_missing_parts_are_listed():
    design = {"tasks": [
        {"id": "a", "inputs": ["input:a.csv", "input:b.csv"], "outputs": ["a_clean.csv"]},  # extra binding
        {"id": "j", "inputs": ["a_clean.csv"], "outputs": ["joined.csv"]},
        {"id": "r", "inputs": ["joined.csv"], "outputs": ["report.json"]},
    ]}
    s = design_score.score(design, REFERENCE)
    assert s["missing_components"] == ["load_b"]
    assert s["missing_bindings"] == ["load_b<-b.csv"] and s["extra_bindings"] == ["load_a<-b.csv"]
    assert s["binding_recall"] == 0.5 and s["missing_dependencies"] == ["load_b->join"]


def test_composition_summary_counts_components_ports_and_bindings():
    graph = {"nodes": [
        {"id": "s", "data": {"type": "source", "label": "A", "files": ["a.csv"]}},
        {"id": "t", "data": {"type": "task", "label": "Clean", "param": {"threshold": 3}}},
        {"id": "d", "data": {"type": "destination", "label": "Out"}},
    ], "edges": [{"source": "s", "target": "t", "sourceHandle": "data", "targetHandle": "in"},
                 {"source": "t", "target": "d"}]}
    c = metrics.composition_summary(graph)
    assert (c["components"], c["sources"], c["destinations"], c["dependencies"], c["bindings"]) == (
        3, ["s"], ["d"], 2, 2)
    assert c["dependency_list"][0] == {"source": "s", "target": "t", "source_port": "data", "target_port": "in"}


@pytest.mark.parametrize("message, category", [
    ("main.py has invalid syntax: unexpected indent", "syntax"),
    ("Requirement is not in the allowed package list: torch", "dependency"),
    ("ModuleNotFoundError: No module named 'polars'", "dependency"),
    ("Missing planned output contract: summary", "contract"),
    ("Output file does not exist for joined.", "contract"),
    ("Container sample run timed out after 900s.", "sandbox"),
    ("pipeline.py imports banned module: subprocess", "sandbox"),
    ("Pipeline compilation failed: bad graph", "deployment"),
    ("ZeroDivisionError: division by zero", "runtime"),
    ("something odd", "other"),
])
def test_failures_are_sorted_into_categories(message, category):
    assert metrics.classify_failure(message) == category


def test_tokens_and_time_are_split_by_phase():
    events = [
        {"type": "phase_started", "phase": "design", "mono": 0.0},
        {"type": "request_finished", "purpose": "design_agent", "latency_s": 2, "usage": {"input_tokens": 100,
                                                                                         "output_tokens": 10}},
        {"type": "request_finished", "purpose": "naming", "latency_s": 1, "usage": {"input_tokens": 50,
                                                                                    "output_tokens": 5}},
        {"type": "phase_started", "phase": "codegen", "mono": 10.0},
        {"type": "request_finished", "purpose": "generate_canonical_pipeline", "latency_s": 3,
         "usage": {"input_tokens": 300, "output_tokens": 30}, "cost_usd": 0.01},
        {"type": "request_failed", "purpose": "repair_canonical_pipeline", "latency_s": 4},
        {"type": "phase_started", "phase": "bundle", "mono": 30.0},
        {"type": "phase_started", "phase": "validate", "mono": 31.0},
        {"type": "episode_finished", "mono": 35.0},
    ]
    u = metrics.phase_usage(events)
    assert u["by_phase"]["composition"]["input_tokens"] == 150 and u["by_phase"]["composition"]["requests"] == 2
    assert u["by_phase"]["implementation"]["cost_usd"] == 0.01
    assert u["by_phase"]["repair"] == {"requests": 1, "failed_requests": 1, "input_tokens": 0,
                                       "output_tokens": 0, "cost_usd": 0.0, "model_seconds": 4}
    assert u["seconds"] == {"composition": 10.0, "implementation": 20.0, "bundle": 1.0, "validation": 4.0}


def test_execution_summary_reads_servicecraft(tmp_path):
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/joined.csv").write_text("x\n")
    result = {"elapsed_seconds": 9.4, "stages": {
        "definition_graph": {"status": "pass", "structure_score_01": 1.0, "actual": {"status": "pass"}},
        "execution": {"status": "fail", "run_success": False, "completed_steps": ["load"], "events": [
            {"type": "STEP_SUCCESS", "step_key": "load", "message": 'Finished execution of step "load" in 13ms.'},
            {"type": "STEP_FAILURE", "step_key": "join", "message": "boom"}]},
        "contracts": {"status": "fail", "checks": [{"artifact": "joined.csv", "status": "fail",
                                                    "error": "Required CSV columns missing"}]},
        "oracle": {"status": "fail", "checks": [{"id": "joined.csv", "status": "fail"}]},
    }}
    e = metrics.execution_summary(result, tmp_path)
    assert e["loaded"] and not e["completed"] and e["failed_components"] == ["join"]
    assert e["outputs_produced"] == ["joined.csv"] and e["outputs_invalid"] == ["joined.csv"]
    assert e["failure_stage"] == "execution" and e["dagster_step_seconds"] == 0.013


def test_conditions_form_a_factorial_for_inlumen_only(tmp_path):
    assert make_conditions("code") == [] and make_conditions("design") == []
    conditions = make_conditions("code", contracts=(True, False), repair=(True, False), repairs=3)
    assert [c["name"] for c in conditions] == ["contracts · repair ×3", "contracts · no repair",
                                               "no contracts · repair ×3", "no contracts · no repair"]
    exp = Experiment(name="abl", cases=["016_x"], methods=["inlumen", "direct"], models=["p:m"],
                     tracks=["fixed_graph"], repetitions=2, conditions=conditions)
    runs = exp.episodes()
    assert len(runs) == 2 * 4 + 2  # direct runs once per replicate
    ids = {r["id"] for r in runs}
    assert "016-inlumen-p-m-fg-ncnr-r1" in ids and "016-direct-p-m-fg-r0" in ids
    no_repair = next(r for r in runs if r["id"] == "016-inlumen-p-m-fg-ncnr-r1")
    assert no_repair["condition_options"] == {"artifact_contracts": False, "repair_attempts": 0}
    design = make_conditions("design", grounding=(True, False))
    assert [c["options"] for c in design] == [{"grounded": True}, {"grounded": False}]


def test_condition_labels():
    assert condition_label({"stage": "design", "conditions": {"grounded": False}}) == "ungrounded"
    assert condition_label({"method": "direct", "conditions": {}}) == "baseline"
    assert condition_label({"conditions": {"phase": "implementation", "artifact_contracts": False,
                                           "repair": True, "max_repair_attempts": 2}}) == "no contracts · repair ×2"


def test_export_has_one_row_per_run_with_every_measure(tmp_path):
    root = tmp_path / "exp"
    exp = Experiment(name="exp", cases=["016_x"], methods=["inlumen"], models=["p:m"], tracks=["fixed_graph"],
                     repetitions=1, limits={"max_cost_usd": 2})
    exp.save(root)
    run_id = exp.episodes()[0]["id"]
    (root / "episodes" / run_id).mkdir(parents=True)
    state = {**exp.episodes()[0], "status": "evaluated",
             "complexity": metrics.complexity(REFERENCE),
             "conditions": {"phase": "implementation", "artifact_contracts": True, "repair": True,
                            "max_repair_attempts": 2, "limits": {"max_cost_usd": 2}},
             "implementation": {"attempts": 2, "initial_valid": False, "final_valid": True,
                                "recovered_by_repair": True, "feedback_categories": {"contract": 1}},
             "execution": {"loaded": True, "completed": True, "outputs_produced": ["a"], "evaluation_seconds": 9},
             "scores": {"dagster_executable": 1, "complete_success": 1},
             "usage": {"input_tokens": 100, "output_tokens": 10, "cost_usd": 0.02},
             "phases": {"by_phase": {"repair": {"input_tokens": 40}}, "seconds": {"implementation": 5, "repair": 2}},
             "budget": {"elapsed_s": 8}}
    (root / "episodes" / run_id / "episode.json").write_text(json.dumps(state))
    rows = export_rows([root])
    assert len(rows) == 1
    row = rows[0]
    assert row["condition"] == "contracts · repair ×2" and row["ref_level"] == "low"
    assert row["recovered_by_repair"] is True and row["feedback_contract"] == 1 and row["feedback_syntax"] == 0
    assert row["repair_input_tokens"] == 40 and row["construction_seconds"] == 7
    assert row["max_cost_usd"] == 2 and row["dagster_completed"] is True
    assert to_csv(rows).splitlines()[0].startswith("experiment,run,phase,case")
    assert run_row("x", {"id": "pending", "status": "pending"})["outcome"] == "pending"


def test_deleted_runs_become_pending(tmp_path):
    from inlumen_bench.results import episodes
    from inlumen_bench.runner import delete_runs, pending_episodes

    root = tmp_path / "exp"
    exp = Experiment(name="exp", cases=["016_x"], methods=["inlumen"], models=["p:m"], tracks=["fixed_graph"],
                     repetitions=2)
    exp.save(root)
    ids = [e["id"] for e in exp.episodes()]
    for run_id in ids:
        (root / "episodes" / run_id).mkdir(parents=True)
        (root / "episodes" / run_id / "episode.json").write_text(json.dumps({"status": "evaluated"}))
    (root / "episodes" / f"{ids[0]}.interrupted-20260101T000000").mkdir()
    assert pending_episodes(root, exp) == []
    assert delete_runs(root, [ids[0]]) == 1
    assert [e["id"] for e in pending_episodes(root, exp)] == [ids[0]]
    assert not list((root / "episodes").glob("*.interrupted-*"))
    assert {r["status"] for r in episodes(root)} == {"pending", "evaluated"}
    with pytest.raises(ValueError):
        delete_runs(root, ["not-a-run"])


def test_runs_go_one_model_at_a_time_in_random_order_within_each():
    exp = Experiment(name="m", cases=["016_x", "001_y", "002_z"], methods=["inlumen", "direct"],
                     models=["p:b", "p:a"], tracks=["fixed_graph"], repetitions=2, seed=3)
    runs = exp.episodes()
    models = [r["model"] for r in runs]
    assert models == ["p:b"] * 12 + ["p:a"] * 12  # in the order given, never interleaved
    first = [r["id"] for r in runs[:12]]
    assert first != sorted(first) and first == [r["id"] for r in exp.episodes()[:12]]  # shuffled, reproducibly
    design = Experiment(name="d", stage="design", cases=["016_x", "001_y"], methods=["inlumen"],
                        models=["p:a", "p:b"], tracks=["natural_language"], repetitions=3)
    assert [r["model"] for r in design.episodes()] == ["p:a"] * 6 + ["p:b"] * 6


def test_model_queue_keeps_each_model_with_one_worker():
    from inlumen_bench.runner import ModelQueue

    runs = [{"id": f"{m}{i}", "model": f"p:{m}"} for m in ("a", "b", "c") for i in range(2)]
    q = ModelQueue(runs, per_model=1)
    first, second = q.take()[0], q.take()[0]
    assert (first["model"], second["model"]) == ("p:a", "p:b")  # two workers, two models
    assert q.take() is not None and q.take() is None  # "c" goes to a third worker; nothing else is free
    assert q.release(first) is None and q.take(prefer="p:a")[0]["id"] == "a1"  # a worker keeps its model
    assert q.release({"model": "p:a"}) == [None]  # a's last run: unload it from the base URL
    two = ModelQueue(runs, per_model=2)
    assert [two.take()[0]["model"] for _ in range(3)] == ["p:a", "p:a", "p:b"]


def test_model_queue_runs_each_model_on_every_gpu_server():
    from inlumen_bench.runner import ModelQueue

    runs = [{"id": f"{m}{i}", "model": f"ollama:{m}"} for m in ("a", "b") for i in range(3)]
    gpu0, gpu1 = "http://127.0.0.1:11435/v1", "http://127.0.0.1:11436/v1"
    q = ModelQueue(runs, per_model=1, endpoints={"ollama": [gpu0, gpu1]})
    a0, a1 = q.take(), q.take()
    assert (a0[0]["model"], a0[1], a1[0]["model"], a1[1]) == ("ollama:a", gpu0, "ollama:a", gpu1)  # both GPUs
    assert q.take() is None  # each server runs one request at a time
    assert q.release(a0[0], gpu0) is None
    a2 = q.take(prefer="ollama:a", lane=gpu0)
    assert a2[0]["id"] == "a2" and a2[1] == gpu0 and a2[2] is None  # same model, same GPU: nothing to unload
    assert q.release(a1[0], gpu1) is None
    b0 = q.take(prefer="ollama:a", lane=gpu1)  # a has no runs left: the free GPU moves on to b
    assert (b0[0]["model"], b0[1], b0[2]) == ("ollama:b", gpu1, "ollama:a")  # after unloading a there
    assert q.release(a2[0], gpu0) == [gpu0]  # a's last run: unload it where it is still loaded
    b1 = q.take(prefer="ollama:a", lane=gpu0)
    assert (b1[0]["model"], b1[1], b1[2]) == ("ollama:b", gpu0, None)


def test_run_experiment_runs_models_side_by_side(tmp_path, monkeypatch):
    import threading
    import time

    from inlumen_bench import runner

    exp = Experiment(name="lanes", cases=["016_x", "001_y"], methods=["inlumen"], models=["p:a", "p:b", "p:c"],
                     tracks=["fixed_graph"], repetitions=2)
    exp.save(tmp_path)
    live: dict[str, int] = {}
    peak = {"models": 0, "same": 0}
    lock = threading.Lock()
    unloaded = []

    def fake_run(root, corpus, exp, ep, profiles, limits, endpoint=None):
        with lock:
            live[ep["model"]] = live.get(ep["model"], 0) + 1
            peak["models"] = max(peak["models"], sum(1 for n in live.values() if n))
            peak["same"] = max(peak["same"], max(live.values()))
        time.sleep(0.02)
        with lock:
            live[ep["model"]] -= 1

    monkeypatch.setattr(runner, "run_episode", fake_run)
    monkeypatch.setattr(runner, "load_profiles", dict)
    monkeypatch.setattr(runner, "unload_model", lambda profiles, model, log, url=None, **kw: unloaded.append(model))
    runner.run_experiment(tmp_path, corpus=None, workers=2)
    assert peak == {"models": 2, "same": 1}
    assert sorted(unloaded) == ["p:a", "p:b", "p:c"]


def test_run_experiment_keeps_every_gpu_server_busy(tmp_path, monkeypatch):
    import threading
    import time

    from inlumen_bench import runner
    from inlumen_bench.providers import Profile

    servers = ["http://127.0.0.1:11435/v1", "http://127.0.0.1:11436/v1"]
    profiles = {"ollama": Profile(name="ollama", base_url="http://127.0.0.1:11434/v1", endpoints=servers)}
    exp = Experiment(name="gpus", cases=["016_x", "001_y"], methods=["inlumen"],
                     models=["ollama:big", "ollama:small"], tracks=["fixed_graph"], repetitions=3)
    exp.save(tmp_path)
    busy: dict[str, str] = {}
    seen = []
    lock = threading.Lock()
    unloaded = []

    def fake_run(root, corpus, exp, ep, profiles, limits, endpoint=None):
        with lock:
            assert endpoint in servers and endpoint not in busy  # one run per server at a time
            busy[endpoint] = ep["model"]
            seen.append(tuple(sorted(busy.values())))
        time.sleep(0.01 if ep["model"] == "ollama:small" else 0.03)
        with lock:
            del busy[endpoint]

    monkeypatch.setattr(runner, "run_episode", fake_run)
    monkeypatch.setattr(runner, "load_profiles", lambda: profiles)
    monkeypatch.setattr(runner, "unload_model",
                        lambda profiles, model, log, url=None, **kw: unloaded.append((model, url)))
    runner.run_experiment(tmp_path, corpus=None, workers=2)
    assert ("ollama:big", "ollama:big") in seen  # the same model on both GPUs at once
    assert len(seen) == 12 and {m for m, _ in unloaded} == {"ollama:big", "ollama:small"}
    assert all(url in servers for _, url in unloaded)


def test_progress_shows_waiting_time_model_and_missing_cost():
    from inlumen_bench.events import Bus
    from inlumen_bench.progress import ProgressTracker, _budget_text, request_text

    t = ProgressTracker(1, "x", bus=Bus())
    t.on_event({"type": "episode_started", "episode_id": "e", "group": "g", "model": "ollama:m", "mono": 0})
    t.on_event({"type": "request_started", "episode_id": "e", "streamed": False, "mono": 100.0})
    view = t.active["e"]
    assert view.model == "ollama:m" and request_text(view, now=142.4) == "waiting for reply · 42s"
    t.on_event({"type": "request_finished", "episode_id": "e", "usage": {"output_tokens": 1200}})
    assert request_text(view) == "done 1,200 tok"
    t.on_event({"type": "log", "episode_id": None, "message": "ollama:m: all runs done; unloaded from Ollama"})
    assert t.changes[-1].endswith("unloaded from Ollama") and "(ollama:m)" in t.changes[0]
    assert _budget_text({"cost_incomplete": True, "requests": 3, "max_requests": 40}) == "no cost data · 3/40 req"


def test_run_log_is_kept_in_the_experiment_folder(tmp_path):
    from inlumen_bench.jobs import log_line, read_log

    for i in range(5):
        log_line(tmp_path, f"line {i}")
    assert [l.split(" ", 2)[2] for l in read_log(tmp_path, 3)] == ["line 2", "line 3", "line 4"]
    assert len(read_log(tmp_path)) == 5


def test_empty_reply_after_a_tool_call_is_a_model_failure():
    from inlumen_bench.runner import is_model_failure

    error = "RuntimeError: RuntimeError: Reflect on tool use produced no valid text response."
    assert is_model_failure(error)
    assert metrics.classify_failure(error) == "format"
    assert not is_model_failure("FileNotFoundError: design.json is missing")
