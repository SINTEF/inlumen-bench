import json
from pathlib import Path

import pytest

from inlumen_bench import design_score
from inlumen_bench.design import asset_name, check, edges, parse_design, to_project_json, topo_order
from inlumen_bench.harness import RUNTIME, default_names, definitions
from inlumen_bench.methods.inlumen import Inlumen
from inlumen_bench.results import pass_at_k
from inlumen_bench.validate import PublicReport

FIXTURE = Path(__file__).parent / "fixtures" / "case016_design.json"


@pytest.fixture
def design():
    d = json.loads(FIXTURE.read_text())
    d["inputs"] = [{"id": "input:data.csv", "path": "inputs/data.csv"}]
    return d


def test_fixture_design_is_valid_and_ordered(design):
    mandatory = [o for t in design["tasks"] for o in t["outputs"]]
    assert check(design, mandatory, ["data.csv"]) == []
    order = [t["id"] for t in topo_order(design)]
    assert order.index("validate") < order.index("integrate") < order.index("report")
    assert ("cluster", "teams", "cluster_assignments.csv") in edges(design)


def test_check_reports_structural_problems(design):
    design["tasks"][1]["inputs"] = ["nobody_writes_this.csv"]
    design["tasks"][2]["outputs"] = ["players_clean.csv"]
    problems = check(design, ["missing.csv"], ["data.csv"])
    text = "\n".join(problems)
    assert "no task writes" in text and "several tasks" in text and "missing.csv" in text


def test_cycle_is_detected():
    cyclic = {"tasks": [{"id": "a", "inputs": ["b.csv"], "outputs": ["a.csv"]},
                        {"id": "b", "inputs": ["a.csv"], "outputs": ["b.csv"]}]}
    assert any("cycle" in p for p in check(cyclic, [], []))


def test_asset_names_avoid_dagster_reserved_words():
    assert asset_name("context") == "context_task"
    assert asset_name("2-Clean Data") == "t_2_clean_data"


def test_parse_design_from_fenced_reply():
    d = parse_design('Here:\n```json\n{"tasks": [{"id": "x", "outputs": ["x.csv"]}]}\n```')
    assert d["tasks"][0]["inputs"] == [] and d["inputs"] == []


def test_project_json_matches_inlumen_import_shape(design):
    doc = to_project_json(design, "Case 016")
    assert doc["schema_version"] == "inlumen.project@2"
    nodes = {n["id"]: n for n in doc["pipeline"]["nodes"]}
    kinds = [n["kind"] for n in nodes.values()]
    assert kinds.count("source") == 1 and kinds.count("task") == 9 and kinds.count("destination") == 1
    for c in doc["pipeline"]["connections"]:
        src, dst = nodes[c["from"]["node"]], nodes[c["to"]["node"]]
        assert c["from"]["port"] in {p["id"] for p in src["outputs"]}
        assert c["to"]["port"] in {p["id"] for p in dst["inputs"]}


def test_generated_definitions_compile(design):
    source = definitions(design, default_names(design))
    compile(source, "definitions.py", "exec")
    compile(RUNTIME, "bench_runtime.py", "exec")
    assert "def integrate(performance, context_features):" in source


def test_design_score_perfect_and_merged(design):
    reference = json.loads(FIXTURE.read_text())
    assert design_score.score(design, reference)["design_correct"] == 1
    merged = json.loads(FIXTURE.read_text())
    merged["tasks"] = [t for t in merged["tasks"] if t["id"] not in ("profiles", "teams")]
    merged["tasks"].append({"id": "both", "inputs": ["cluster_assignments.csv"],
                            "outputs": ["cluster_profiles.csv", "team_cluster_mix.csv"]})
    s = design_score.score(merged, reference)
    assert s["design_correct"] == 0 and s["merged_tasks"] == ["both"] and s["artifact_coverage"] == 1.0


def test_repair_attribution_uses_tracebacks_then_steps(design):
    order = topo_order(design)
    names = default_names(design)
    by_asset = {names[t["id"]]: t for t in order}
    report = PublicReport("fail", "execution", errors=['step teams: Execution of step "teams" failed.'],
                          node_errors={"teams": "Traceback ..."})
    assert Inlumen._attribute(report, by_asset, order, names)["id"] == "teams"
    report = PublicReport("fail", "definition", errors=['File "/candidate/nodes/cluster.py", line 3'])
    assert Inlumen._attribute(report, by_asset, order, names)["id"] == "cluster"


def test_pass_at_k():
    assert pass_at_k(3, 0, 1) == 0.0
    assert pass_at_k(3, 3, 3) == 1.0
    assert pass_at_k(4, 1, 1) == pytest.approx(0.25)
    assert pass_at_k(2, 1, 3) is None


def test_validator_infra_error_is_not_a_model_failure(design, tmp_path, monkeypatch):
    from inlumen_bench.events import EventLog
    from inlumen_bench.methods import inlumen
    from inlumen_bench.methods.base import EpisodeContext, InfraError

    workspace = tmp_path / "workspace"
    (workspace / "inputs").mkdir(parents=True)
    (workspace / "prompt.txt").write_text("Build the pipeline.")
    (workspace / "provided_graph.json").write_text(json.dumps(design))
    ctx = EpisodeContext("016", "fixed_graph", workspace, "contract", tmp_path / "submission",
                         tmp_path / "work", None, EventLog(None), None)
    ctx.submission_dir.mkdir()
    ctx.work_dir.mkdir()
    monkeypatch.setattr(Inlumen, "_node", lambda self, *a, **k: "def run(inputs):\n    return {}\n")
    monkeypatch.setattr(inlumen, "validate", lambda *a, **k: PublicReport(
        "infra_error", "execution", errors=["Unable to find image 'servicecraft-runtime:1.13.12'"]))
    with pytest.raises(InfraError, match="could not run"):
        Inlumen().generate(ctx)


def test_project_json_uses_typed_ports_and_named_joins():
    two_inputs = {
        "inputs": [{"id": "input:customers.csv"}, {"id": "input:orders.csv"}],
        "tasks": [
            {"id": "clean_customers", "inputs": ["input:customers.csv"], "outputs": ["customers_clean.csv"]},
            {"id": "clean_orders", "inputs": ["input:orders.csv"], "outputs": ["orders_clean.csv"]},
            {"id": "merge", "inputs": ["customers_clean.csv", "orders_clean.csv"],
             "outputs": ["customer_orders.csv", "summary.json"]},
        ],
    }
    doc = to_project_json(two_inputs, "join")
    nodes = {n["id"]: n for n in doc["pipeline"]["nodes"]}
    assert [p["id"] for p in nodes["src_customers_csv"]["outputs"]] == ["data"]
    assert nodes["src_customers_csv"]["outputs"][0]["type"] == "table"
    merge = nodes["merge"]
    assert [p["id"] for p in merge["inputs"]] == ["customers_clean", "orders_clean"]
    assert [(p["id"], p["type"]) for p in merge["outputs"]] == [("customer_orders", "table"), ("summary", "json")]
    assert [a["kind"] for a in merge["implementation"]["output_artifacts"]] == ["table", "json"]
    targets = [(c["to"]["node"], c["to"]["port"]) for c in doc["pipeline"]["connections"]]
    assert len(targets) == len(set(targets))  # one producer per input port
    assert ("merge", "orders_clean") in targets
    assert check(two_inputs, [], ["customers.csv", "orders.csv", "extra.csv"]) == [
        "input files no task reads (each input starts its own branch): ['input:extra.csv']"]


def test_evaluator_entry_ids_are_safe():
    import re

    from inlumen_bench.evaluate import entry_id
    eid = entry_id("001-inlumen-ollama-llama3.2-3b-fg-r0")
    assert re.fullmatch(r"[A-Za-z0-9_-]+", eid) and eid == "001-inlumen-ollama-llama3_2-3b-fg-r0"
