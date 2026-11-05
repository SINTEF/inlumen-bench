"""Two-step experiments: design (step 1), code from a design (step 2), deletion."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inlumen_bench.events import EventLog
from inlumen_bench.methods import inlumen_standalone as inlumen
from inlumen_bench.methods.base import EpisodeContext
from inlumen_bench.methods.inlumen_standalone import InlumenStandalone as Inlumen
from inlumen_bench.runner import Experiment, delete_experiment, dependents, successful_designs
from inlumen_bench.validate import PublicReport

DESIGN = {
    "title": "t",
    "tasks": [
        {"id": "clean", "inputs": ["input:data.csv"], "outputs": ["clean.csv"]},
        {"id": "report", "inputs": ["clean.csv"], "outputs": ["report.json"]},
    ],
}
PROMPT = "Build it.\n\n5. Mandatory output artifacts\n- `clean.csv`\n- `report.json`\n"


def _ctx(tmp_path: Path, llm=None, design=None) -> EpisodeContext:
    workspace = tmp_path / "workspace"
    (workspace / "inputs").mkdir(parents=True)
    (workspace / "inputs/data.csv").write_text("a\n1\n")
    (workspace / "prompt.txt").write_text(PROMPT)
    ctx = EpisodeContext("001_x", "natural_language", workspace, "contract", tmp_path / "submission",
                         tmp_path / "work", llm, EventLog(None), None, design=design)
    ctx.work_dir.mkdir()
    return ctx


class FakeLLM:
    def __init__(self, reply: str):
        self.reply, self.purposes = reply, []

    def chat(self, messages, purpose=""):
        self.purposes.append(purpose)
        return SimpleNamespace(text=self.reply)


def test_design_step_saves_a_checked_design(tmp_path):
    llm = FakeLLM("```json\n" + json.dumps(DESIGN) + "\n```")
    result = Inlumen().design(_ctx(tmp_path, llm))
    assert result.status == "designed" and llm.purposes == ["design"]
    saved = json.loads((tmp_path / "work/design.json").read_text())
    assert [t["id"] for t in saved["tasks"]] == ["clean", "report"]
    assert (tmp_path / "work/project.json").is_file()
    assert not (tmp_path / "submission").exists()


def test_code_step_uses_the_given_design_without_designing(tmp_path, monkeypatch):
    llm = FakeLLM("")
    ctx = _ctx(tmp_path, llm, design={**DESIGN, "inputs": [{"id": "input:data.csv", "path": "inputs/data.csv"}]})
    ctx.submission_dir.mkdir()
    monkeypatch.setattr(Inlumen, "_node", lambda self, *a, **k: "def run(inputs):\n    return {}\n")
    monkeypatch.setattr(inlumen, "validate", lambda *a, **k: PublicReport("pass", "done"))
    result = Inlumen().generate(ctx)
    assert result.status == "generated"
    assert result.counters["design_requests"] == 0 and llm.purposes == []
    assert (ctx.submission_dir / "nodes/clean.py").is_file()


def _experiment(root: Path, **extra) -> Experiment:
    exp = Experiment(name=root.name, cases=["001_x", "002_y"], methods=["inlumen"], models=["p:m1", "p:m2"],
                     tracks=["natural_language"], repetitions=2, **extra)
    exp.save(root)
    return exp


def test_design_experiment_episodes_and_successful_designs(tmp_path):
    root = tmp_path / "d1"
    exp = _experiment(root, stage="design")
    eps = exp.episodes()
    assert len(eps) == 2 * 2 * 2 and all(e["method"] == "inlumen" for e in eps)
    ok = next(e for e in eps if e["id"] == "001-design-p-m1-r0")
    (root / "episodes" / ok["id"] / "work").mkdir(parents=True)
    (root / "episodes" / ok["id"] / "work/design.json").write_text(json.dumps(DESIGN))
    (root / "episodes" / ok["id"] / "episode.json").write_text(json.dumps({"status": "designed"}))
    assert successful_designs(root) == [{"case_id": "001_x", "model": "p:m1", "repetition": 0,
                                         "episode": "001-design-p-m1-r0"}]


def test_code_experiment_from_designs_pairs_each_design(tmp_path):
    designs = [{"case_id": "001_x", "model": "p:m1", "repetition": 0, "episode": "001-design-p-m1-r0"}]
    exp = _experiment(tmp_path / "c1", stage="code", design_from="d1", designs=designs)
    eps = exp.episodes()
    assert len(eps) == 1 * 1 * 2 * 2  # designs × methods × models × repetitions
    assert {e["design_episode"] for e in eps} == {"001-design-p-m1-r0"}
    assert len({e["id"] for e in eps}) == len(eps)
    assert exp.design_source == "designs from d1"


def test_delete_experiment_and_dependents(tmp_path):
    _experiment(tmp_path / "d1", stage="design")
    _experiment(tmp_path / "c1", stage="code", design_from="d1", designs=[])
    assert dependents(tmp_path, "d1") == ["c1"]
    delete_experiment(tmp_path / "d1")
    assert not (tmp_path / "d1").exists()
    with pytest.raises(FileNotFoundError):
        delete_experiment(tmp_path / "missing")


def test_old_experiments_load_as_code_from_reference(tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    (root / "experiment.json").write_text(json.dumps({
        "name": "old", "cases": ["001_x"], "methods": ["direct"], "models": ["p:m"],
        "tracks": ["fixed_graph"], "repetitions": 1}))
    exp = Experiment.load(root)
    assert exp.stage == "code" and exp.design_source == "reference designs"
    assert exp.episodes()[0]["id"] == "001-direct-p-m-fg-r0"


def test_stop_kills_the_run_at_once(tmp_path, monkeypatch):
    import threading
    import time

    from inlumen_bench import jobs

    exp = tmp_path / "e1"
    ep = exp / "episodes" / "001-x"
    ep.mkdir(parents=True)
    (ep / "episode.json").write_text(json.dumps({"status": "running"}))
    stop, lines = threading.Event(), []
    # A stand-in for `bench run` that would run for a minute.
    jobs_cli = jobs.sys.executable
    worker = threading.Thread(target=lambda: jobs.run_cli(lines.append, stop, exp, "--help"))
    original = jobs.subprocess.Popen

    def slow_popen(cmd, **kw):
        return original([jobs_cli, "-c", "import time; print('working', flush=True); time.sleep(60)"], **kw)

    monkeypatch.setattr(jobs, "kill_containers", lambda experiment: 0)  # no Docker daemon in tests
    jobs.subprocess.Popen = slow_popen
    try:
        worker.start()
        time.sleep(1)
        started = time.monotonic()
        stop.set()
        worker.join(timeout=10)
    finally:
        jobs.subprocess.Popen = original
    assert not worker.is_alive() and time.monotonic() - started < 5
    assert json.loads((ep / "episode.json").read_text())["status"] == "stopped"
    assert not (exp / ".tmp").exists()
    assert any(line.startswith("stopped:") for line in lines)
