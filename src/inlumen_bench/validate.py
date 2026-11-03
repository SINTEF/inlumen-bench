"""Public validation: the feedback a method may see while generating.

It runs the submission in ServiceCraft's runtime image with the same isolation
flags as the evaluator, using the image's own runtime worker, but without the
expected graph, private artifact contracts or oracles. It reports only syntax
errors, Dagster load errors, step failures and which mandatory artifacts (listed
in the public prompt) were not written.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .corpus import RUNTIME_IMAGE
from .events import EventLog

ERROR_TAIL = 4000


@dataclass
class PublicReport:
    status: str  # pass | fail | timeout | infra_error
    stage: str  # syntax | definition | execution | artifacts | done
    errors: list[str] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)
    edges: list[list[str]] = field(default_factory=list)
    written: list[str] = field(default_factory=list)
    missing_artifacts: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def feedback(self) -> str:
        """Short text suitable for a repair prompt."""
        if self.status == "pass":
            return "Public validation passed."
        parts = [f"Public validation failed at stage `{self.stage}`."]
        parts += [e[-ERROR_TAIL:] for e in self.errors]
        if self.missing_artifacts:
            parts.append("Mandatory artifacts not written: " + ", ".join(self.missing_artifacts))
        return "\n\n".join(parts)


def mandatory_artifacts(prompt: str) -> list[str]:
    """File names listed under '5. Mandatory output artifacts' in a ServiceCraft prompt."""
    match = re.search(r"5\. Mandatory output artifacts\s*\n(.*?)(?:\n\d\. |\Z)", prompt, re.S)
    if not match:
        return []
    return re.findall(r"^- `([^`]+)`", match.group(1), re.M)


def check_syntax(submission: Path) -> list[str]:
    errors = []
    for path in sorted(submission.rglob("*.py")):
        try:
            compile(path.read_text(), str(path.relative_to(submission)), "exec")
        except SyntaxError as exc:
            errors.append(f"{exc.filename}:{exc.lineno}: {exc.msg}")
    return errors


def validate(
    submission: Path,
    inputs: Path,
    prompt: str,
    *,
    events: EventLog | None = None,
    timeout_s: float = 900,
    memory: str = "2g",
    cpus: float = 1.0,
    image: str = RUNTIME_IMAGE,
    keep_dir: Path | None = None,
) -> PublicReport:
    start = time.monotonic()
    emit = events.emit if events else (lambda *a, **k: None)

    emit("validation_progress", message="syntax")
    if not (submission / "definitions.py").is_file():
        return PublicReport("fail", "syntax", ["definitions.py is missing"], elapsed_s=0.0)
    syntax = check_syntax(submission)
    if syntax:
        return PublicReport("fail", "syntax", syntax, elapsed_s=time.monotonic() - start)

    work = Path(keep_dir or tempfile.mkdtemp(prefix="bench-validate-"))
    try:
        return _run(submission, inputs, prompt, work, emit, start, timeout_s, memory, cpus, image)
    finally:
        if keep_dir is None:
            shutil.rmtree(work, ignore_errors=True)


def _run(submission, inputs, prompt, work, emit, start, timeout_s, memory, cpus, image) -> PublicReport:
    protocol, outputs, evidence = work / "protocol", work / "outputs", work / "evidence"
    for d in (protocol, outputs, evidence):
        d.mkdir(parents=True, exist_ok=True)
        d.chmod(0o777)
    (protocol / "runtime.json").write_text(
        json.dumps(
            {
                "module": "definitions",
                "attribute": "defs",
                "project_dir": ".",
                "python_paths": [".", "src"],
                "mode": "assets",
                "job_name": None,
                "run_config": {},
                "environment_aliases": {},
            }
        )
    )
    name = f"bench-validate-{uuid.uuid4().hex[:10]}"
    mounts = [
        (submission.resolve(), "/candidate", True),
        (inputs.resolve(), "/inputs", True),
        (protocol, "/protocol", True),
        (outputs, "/outputs", False),
        (evidence, "/evidence", False),
    ]
    args = [
        "docker", "run", "--rm", f"--name={name}",
        "--network=none", "--read-only", "--user=65532:65532", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--pids-limit=128",
        f"--memory={memory}", f"--cpus={cpus}", "--tmpfs=/tmp:rw,nosuid,size=512m", "--workdir=/tmp",
    ]
    for host, target, ro in mounts:
        args += ["--mount", f"type=bind,source={host},target={target}" + (",readonly" if ro else "")]
    args += [image, "python", "/opt/evaluator/runtime_worker.py"]

    emit("validation_progress", message="load + run in Docker")
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        return PublicReport("timeout", "execution", [f"run exceeded {timeout_s:.0f}s"],
                            elapsed_s=time.monotonic() - start)
    elapsed = time.monotonic() - start
    evidence_file = evidence / "runtime.json"
    if not evidence_file.is_file():
        if proc.returncode == 137:
            return PublicReport("fail", "execution", ["process killed (likely out of memory)"], elapsed_s=elapsed)
        return PublicReport("infra_error", "execution", [(proc.stderr or proc.stdout)[-ERROR_TAIL:]], elapsed_s=elapsed)

    data = json.loads(evidence_file.read_text())
    definition = data.get("definition", {})
    if definition.get("status") != "pass":
        return PublicReport("fail", "definition", [definition.get("error", "definition failed")[-ERROR_TAIL:]],
                            elapsed_s=elapsed)
    report = PublicReport("fail", "execution", nodes=definition.get("nodes", []),
                          edges=definition.get("edges", []), elapsed_s=elapsed)
    execution = data.get("execution", {})
    if proc.returncode == 137 and execution.get("status") == "not_reached":
        report.errors.append("process killed during execution (likely out of memory)")
        return report
    if execution.get("error"):
        report.errors.append(execution["error"][-ERROR_TAIL:])
    for event in execution.get("events", []):
        if event.get("type") == "STEP_FAILURE":
            report.errors.append(f"step {event.get('step_key')}: {event.get('message', '')}"[-ERROR_TAIL:])
    for check in execution.get("asset_checks", []):
        if not check.get("passed"):
            report.errors.append(f"asset check {check['asset']}.{check['name']} failed")

    report.written = sorted(p.name for p in outputs.iterdir() if p.is_file())
    report.missing_artifacts = [a for a in mandatory_artifacts(prompt) if a not in report.written]
    if execution.get("status") != "pass" or report.errors:
        return report
    if report.missing_artifacts:
        report.stage = "artifacts"
        return report
    report.status, report.stage = "pass", "done"
    return report
