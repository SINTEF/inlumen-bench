"""The `inlumen` method: inLUMEN's own agents through the inlumen-agents package."""

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("inlumen.codegen.run", reason="needs inlumen-bench[inlumen]")
pytest.importorskip("autogen_ext", reason="needs inlumen-bench[inlumen]")

from inlumen.codegen.generator import deterministic_pipeline_payload  # noqa: E402

from inlumen_bench.budget import Budget  # noqa: E402
from inlumen_bench.events import EventLog  # noqa: E402
from inlumen_bench.methods import get_method  # noqa: E402
from inlumen_bench.methods import inlumen as adapter  # noqa: E402
from inlumen_bench.methods.base import EpisodeContext  # noqa: E402
from inlumen_bench.package_harness import canvas_graph, submission_files  # noqa: E402
from inlumen_bench.runner import Experiment  # noqa: E402
from inlumen_bench.validate import PublicReport  # noqa: E402

DESIGN = {
    "title": "orders",
    "inputs": [{"id": "input:orders.csv", "path": "inputs/orders.csv"},
               {"id": "input:rates.csv", "path": "inputs/rates.csv"}],
    "tasks": [
        {"id": "clean", "name": "Clean orders", "operation": "cleaning", "inputs": ["input:orders.csv"],
         "outputs": ["clean.csv"], "requirements": ["drop empty rows"], "parameters": {}},
        {"id": "join", "name": "Join rates", "operation": "join", "inputs": ["clean.csv", "input:rates.csv"],
         "outputs": ["joined.csv", "summary.json"], "requirements": [], "parameters": {}},
    ],
}
PROMPT = ("Price the orders.\n\n5. Mandatory output artifacts\n- `clean.csv`\n- `joined.csv`\n"
          "- `summary.json`\n")
CONTRACT = "## Installed packages\n```\npandas==2.2.2\nnumpy==2.0.1\n```\n"


class FakeLLM:
    """The bench client: answers codegen with inLUMEN's deterministic draft, naming as scripted."""

    def __init__(self, naming=None):
        self.purposes, self.naming = [], list(naming or [])
        self.model, self.ref = "fake-coder", "fake:fake-coder"
        self.profile = SimpleNamespace(headers={}, is_openrouter=False, base_url="http://127.0.0.1:1/v1",
                                       resolved_key=lambda: None, timeout_s=10)

    def chat(self, messages, purpose="", max_tokens=None):
        self.purposes.append(purpose)
        if purpose.startswith("naming"):
            text = "```json\n" + json.dumps(self.naming.pop(0)) + "\n```"
        else:
            plan = json.loads(messages[-1]["content"])["pipeline_plan"]
            text = json.dumps(deterministic_pipeline_payload(plan))
        return SimpleNamespace(text=text, usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                               cost_usd=None, finish_reason="stop")


def _ctx(tmp_path: Path, llm, design=None, options=None) -> EpisodeContext:
    workspace = tmp_path / "workspace"
    (workspace / "inputs").mkdir(parents=True)
    (workspace / "inputs/orders.csv").write_text("id,amount\n1,9.5\n2,3\n")
    (workspace / "inputs/rates.csv").write_text("currency,rate\nEUR,1.1\n")
    (workspace / "prompt.txt").write_text(PROMPT)
    ctx = EpisodeContext("001_x", "natural_language", workspace, CONTRACT, tmp_path / "submission",
                         tmp_path / "work", llm, EventLog(None), Budget(), design=design,
                         options={"sandbox": "none", **(options or {})})
    ctx.work_dir.mkdir()
    return ctx


def test_installed_packages_come_from_the_contract():
    assert adapter.installed_packages(CONTRACT) == ["pandas==2.2.2", "numpy==2.0.1"]


def test_canvas_graph_declares_each_artifact_with_its_file_name():
    names = {"clean": "clean", "join": "join"}
    graph = canvas_graph(DESIGN, names)
    by_id = {n["id"]: n["data"] for n in graph["nodes"]}
    assert by_id["src_orders_csv"]["files"] == ["orders.csv"]
    declared = by_id["join"]["implementation"]["output_artifacts"]
    assert [(a["filename"], a["port"], a["kind"]) for a in declared] == [
        ("joined.csv", "joined", "table"), ("summary.json", "summary", "json")]
    assert {(e["source"], e["sourceHandle"], e["target"], e["targetHandle"]) for e in graph["edges"]} == {
        ("src_orders_csv", "data", "clean", "input"), ("clean", "output", "join", "clean"),
        ("src_rates_csv", "data", "join", "rates")}


def test_code_step_runs_inlumen_codegen_and_the_packages_run(tmp_path, monkeypatch):
    llm = FakeLLM()
    ctx = _ctx(tmp_path, llm, design=DESIGN)
    monkeypatch.setattr(adapter, "validate", lambda *a, **k: PublicReport("pass", "done"))
    result = get_method("inlumen").generate(ctx)

    assert result.status == "generated", result.error
    assert llm.purposes == ["generate_canonical_pipeline"]
    assert result.counters["code_producing_requests"] == 1 and result.counters["node_generations"] == 2
    files = sorted(str(p.relative_to(ctx.submission_dir)) for p in ctx.submission_dir.rglob("*.py"))
    assert files == ["bench_inlumen_runtime.py", "definitions.py", "nodes/__init__.py", "nodes/clean_main.py",
                     "nodes/clean_manifest.py", "nodes/join_main.py", "nodes/join_manifest.py"]
    assert json.loads((ctx.work_dir / "inlumen.json").read_text())["package"] == "inlumen-agents"

    # Run the submission's packages in dependency order, as the Dagster assets do.
    out = tmp_path / "outputs"
    monkeypatch.setenv("SERVICECRAFT_INPUT_DIR", str(ctx.workspace / "inputs"))
    monkeypatch.setenv("SERVICECRAFT_OUTPUT_DIR", str(out))
    monkeypatch.syspath_prepend(str(ctx.submission_dir))
    for module in [m for m in sys.modules if m == "bench_inlumen_runtime" or m.startswith("nodes")]:
        del sys.modules[module]
    import bench_inlumen_runtime as runtime
    from nodes import clean_manifest, join_manifest

    nodes = ctx.submission_dir / "nodes"
    cwd = os.getcwd()
    clean = runtime.run_package("clean", nodes / "clean_main.py", clean_manifest.MANIFEST, {})
    joined = runtime.run_package("join", nodes / "join_main.py", join_manifest.MANIFEST, {"clean": clean})
    assert os.getcwd() == cwd and "PIPELINE_OUTPUT_DIR" not in os.environ
    assert sorted(clean) == ["clean.csv"] and sorted(joined) == ["joined.csv", "summary.json"]
    assert sorted(p.name for p in out.iterdir()) == ["clean.csv", "joined.csv", "summary.json"]


def test_submission_wires_assets_by_dependency():
    packages = {name: {"main.py": "print('x')\n", "node-manifest.json": "{}"} for name in ("clean", "join")}
    files = submission_files(DESIGN, {"clean": "clean", "join": "join"}, packages)
    definitions = files["definitions.py"]
    assert "def clean():" in definitions and "def join(clean):" in definitions
    compile(definitions, "definitions.py", "exec")
    with pytest.raises(ValueError):
        submission_files(DESIGN, {"clean": "clean", "join": "join"}, {"clean": packages["clean"]})


def _design_model():
    from autogen_core import FunctionCall
    from autogen_core.models import CreateResult, RequestUsage
    from autogen_ext.models.replay import ReplayChatCompletionClient

    def call(name, **arguments):
        return CreateResult(finish_reason="function_calls", cached=False,
                            usage=RequestUsage(prompt_tokens=100, completion_tokens=10),
                            content=[FunctionCall(id=f"c-{arguments.get('label')}", name=name,
                                                  arguments=json.dumps({"params": json.dumps(arguments)}))])

    return ReplayChatCompletionClient([
        call("create_step", type="source", label="Orders", description="Order table"),
        call("create_step", type="task", label="Clean orders", description="Drop empty rows"),
        call("create_step", type="destination", label="Clean table", description="CSV"),
        CreateResult(finish_reason="stop", content="Done.", cached=False,
                     usage=RequestUsage(prompt_tokens=120, completion_tokens=20)),
    ], model_info={"vision": False, "function_calling": True, "json_output": True, "family": "unknown",
                   "structured_output": False})


def test_design_step_runs_inlumen_design_and_names_artifacts(tmp_path, monkeypatch):
    prompt = "Clean the orders.\n\n5. Mandatory output artifacts\n- `clean.csv`\n"
    llm = FakeLLM(naming=[
        {"sources": {"1": ["orders.csv"]}, "tasks": {"2": ["cleaned.csv"]}},  # wrong name: repaired
        {"sources": {"1": ["orders.csv", "rates.csv"]}, "tasks": {"2": ["clean.csv"]}},
    ])
    ctx = _ctx(tmp_path, llm)
    (ctx.workspace / "prompt.txt").write_text(prompt)
    events = []
    ctx.events = SimpleNamespace(emit=lambda kind, **data: events.append((kind, data)))
    monkeypatch.setattr(adapter.Inlumen, "design_model", lambda self, ctx: _design_model())
    result = get_method("inlumen").design(ctx)

    assert result.status == "designed", result.error
    assert llm.purposes == ["naming", "naming_repair"]
    assert result.counters["design_requests"] == 4 and result.counters["naming_requests"] == 2
    design = json.loads((ctx.work_dir / "design.json").read_text())
    assert [(t["id"], t["inputs"], t["outputs"]) for t in design["tasks"]] == [
        ("clean_orders", ["input:orders.csv", "input:rates.csv"], ["clean.csv"])]
    assert json.loads((ctx.work_dir / "project.json").read_text())["schema_version"] == "inlumen.project@2"
    finished = [d for kind, d in events if kind == "request_finished"]
    assert len(finished) == 4 and finished[0]["purpose"] == "design_agent"
    assert finished[0]["usage"]["input_tokens"] == 100 and ctx.budget.requests == 4


def test_experiments_from_before_keep_the_standalone_method(tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    old = {"name": "old", "cases": ["001_x"], "methods": ["inlumen"], "models": ["p:m"],
           "tracks": ["fixed_graph"], "repetitions": 1}
    (root / "experiment.json").write_text(json.dumps(old))
    assert Experiment.load(root).method_for("inlumen") == "inlumen-standalone"
    new = Experiment(**{**old, "name": "new"})
    new.save(tmp_path / "new")
    assert Experiment.load(tmp_path / "new").method_for("inlumen") == "inlumen"


def test_asyncio_is_not_left_running():
    asyncio.run(asyncio.sleep(0))


def test_a_design_budget_stop_is_reported_as_a_budget_failure(tmp_path, monkeypatch):
    from inlumen_bench.budget import BudgetExceeded, Limits

    ctx = _ctx(tmp_path, FakeLLM())
    ctx.budget = Budget(Limits(max_requests=1))
    monkeypatch.setattr(adapter.Inlumen, "design_model", lambda self, ctx: _design_model())
    with pytest.raises(BudgetExceeded):
        get_method("inlumen").design(ctx)


def test_results_name_the_method_that_ran(tmp_path):
    from inlumen_bench.results import episodes

    root = tmp_path / "old"
    root.mkdir()
    (root / "experiment.json").write_text(json.dumps(
        {"name": "old", "cases": ["001_x"], "methods": ["inlumen"], "models": ["p:m"],
         "tracks": ["fixed_graph"], "repetitions": 1}))
    assert {row["method"] for row in episodes(root)} == {"inlumen-standalone"}


def test_the_sandbox_works_under_the_bench_temporary_folder(tmp_path, monkeypatch):
    import tempfile

    seen = {}
    import inlumen.codegen.run as run_module
    real = run_module.generate

    async def capture(graph, **kwargs):
        seen["sandbox"] = kwargs["sandbox"]
        kwargs["sandbox"] = run_module.NoSandbox()
        return await real(graph, **kwargs)

    monkeypatch.setattr(run_module, "generate", capture)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "bench-tmp"))
    monkeypatch.setattr(adapter, "validate", lambda *a, **k: PublicReport("pass", "done"))
    ctx = _ctx(tmp_path, FakeLLM(), design=DESIGN, options={"sandbox": "docker"})
    assert get_method("inlumen").generate(ctx).status == "generated"
    assert seen["sandbox"].workdir == str(tmp_path / "bench-tmp" / "inlumen-codegen")


def test_an_unusable_model_reply_is_a_generation_failure(tmp_path):
    class Garbled(FakeLLM):
        def chat(self, messages, purpose="", max_tokens=None):
            reply = super().chat(messages, purpose, max_tokens)
            return SimpleNamespace(**{**vars(reply), "text": reply.text + "\n{}"})

    result = get_method("inlumen").generate(_ctx(tmp_path, Garbled(), design=DESIGN))
    assert result.status == "generation_failed" and "invalid JSON" in result.error
