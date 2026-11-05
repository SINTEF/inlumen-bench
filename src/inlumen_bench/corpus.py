"""The ServiceCraft corpus: locating, checking, exporting and local qualification.

Layout inside the data folder:

  <data-dir>/servicecraft/              the dataset, never written to after fetch
  <data-dir>/servicecraft/dataset.json  name, source, commit, fetch date
  <data-dir>/servicecraft-qualified/    produced by `bench setup` (controls requalified
                                        against the locally built Docker images)

ServiceCraft's evaluator is run as a command in its own small virtualenv (in the
user cache folder) with PYTHONPATH pointing at the dataset, so nothing is
installed into or written inside the dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import cache_dir

NAME = "servicecraft"
SOURCE_URL = "https://github.com/aliduabubakari/servicecraft"
RUNTIME_IMAGE = "servicecraft-runtime:1.13.12"
ORACLE_IMAGE = "servicecraft-oracle:corpus-v1"
TOOL_PYTHON_ENV = "INLUMEN_BENCH_TOOL_PYTHON"
# Keep in sync with the `evaluator` extra in pyproject.toml.
TOOL_REQUIREMENTS = ["pydantic>=2,<3", "jsonschema==4.24.0"]
TRACKS = ("natural_language", "fixed_graph")
REQUIRED_PATHS = (
    "corpus/pipelines",
    "corpus/prompt-manifest.json",
    "corpus/tools/export_inputs.py",
    "evaluation/app/service_craft.py",
    "evaluation/Dockerfile",
    "evaluation/Dockerfile.oracles",
    "evaluation/corpus/registry.json",
)
CASE_FILES = ("prompt.txt", "description.md", "task_graph.json", "inputs")


class CorpusError(Exception):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Corpus:
    data_dir: Path

    @property
    def root(self) -> Path:
        return self.data_dir / NAME

    @property
    def qualified_root(self) -> Path:
        return self.data_dir / f"{NAME}-qualified"

    @property
    def qualified_corpus(self) -> Path:
        return self.qualified_root / "qualified-corpus"

    @property
    def tool_python(self) -> Path:
        # The Docker image points this at its own interpreter, which has the evaluator deps.
        if os.environ.get(TOOL_PYTHON_ENV):
            return Path(os.environ[TOOL_PYTHON_ENV])
        return cache_dir() / "servicecraft-tools" / "bin" / "python"

    # ---- metadata -------------------------------------------------------

    def exists(self) -> bool:
        return self.root.is_dir()

    def metadata(self) -> dict:
        meta_path = self.root / "dataset.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {"name": NAME}
        if (self.root / ".git").exists():
            try:
                meta["commit"] = _git(self.root, "rev-parse", "HEAD")
                meta["dirty"] = bool(_git(self.root, "status", "--porcelain", "--untracked-files=no"))
            except subprocess.CalledProcessError:
                pass
        return meta

    def version(self) -> str:
        meta = self.metadata()
        return meta.get("commit", meta.get("version", "unknown"))[:12]

    def registry(self, qualified: bool = False) -> dict:
        path = (
            self.qualified_corpus / "registry.json"
            if qualified
            else self.root / "evaluation/corpus/registry.json"
        )
        if not path.is_file():
            raise CorpusError(f"Missing registry: {path}" + (" (run `bench setup`)" if qualified else ""))
        return json.loads(path.read_text())

    def case_ids(self) -> list[str]:
        return sorted(p.name for p in (self.root / "corpus/pipelines").iterdir() if p.is_dir())

    def resolve_cases(self, selectors: list[str] | None) -> list[str]:
        """Accepts full IDs or numeric prefixes such as `016`; empty means all cases."""
        ids = self.case_ids()
        if not selectors:
            return ids
        out = []
        for sel in selectors:
            matches = [c for c in ids if c == sel or c.split("_", 1)[0] == sel.zfill(3)]
            if len(matches) != 1:
                raise CorpusError(f"Unknown case {sel!r}")
            out.append(matches[0])
        return out

    def case_info(self, case_id: str) -> dict:
        entry = next(c for c in self.registry()["cases"] if c["case_id"] == case_id)
        return {
            "case_id": case_id,
            "task_count": entry.get("task_count"),
            "topology": entry.get("topology"),
            "category": "A" if int(case_id[:3]) <= 13 else "B",
            "logical_to_asset": entry.get("logical_to_asset", {}),
        }

    def prompt(self, case_id: str) -> str:
        return (self.root / "corpus/pipelines" / case_id / "prompt.txt").read_text()

    # ---- checks ---------------------------------------------------------

    def check(self) -> list[str]:
        """Returns problems; an empty list means the dataset is usable."""
        if not self.exists():
            return [f"{self.root} does not exist (run `bench datasets fetch`)"]
        problems = [f"missing {rel}" for rel in REQUIRED_PATHS if not (self.root / rel).exists()]
        if problems:
            return problems
        manifest = json.loads((self.root / "corpus/prompt-manifest.json").read_text())
        expected = {c["case_id"]: c["prompt_sha256"] for c in manifest["cases"]}
        ids = self.case_ids()
        if set(ids) != set(expected):
            problems.append(f"cases on disk ({len(ids)}) differ from prompt manifest ({len(expected)})")
        for case_id in ids:
            case = self.root / "corpus/pipelines" / case_id
            problems += [f"{case_id}: missing {f}" for f in CASE_FILES if not (case / f).exists()]
            prompt = case / "prompt.txt"
            if prompt.is_file() and expected.get(case_id) and sha256_file(prompt) != expected[case_id]:
                problems.append(f"{case_id}: prompt.txt hash differs from prompt manifest")
            if not (self.root / "evaluation/corpus/cases" / case_id / "evaluation.json").is_file():
                problems.append(f"{case_id}: missing evaluation spec")
        return problems

    def setup_status(self) -> dict:
        status = {
            "tool_env": self.tool_python.is_file(),
            "images": {img: _image_id(img) for img in (RUNTIME_IMAGE, ORACLE_IMAGE)},
            "qualified": None,
        }
        setup_file = self.qualified_root / "setup.json"
        registry_file = self.qualified_corpus / "registry.json"
        if setup_file.is_file():
            status["qualified"] = json.loads(setup_file.read_text())
        elif registry_file.is_file():
            # A qualified corpus placed in the data folder by hand or by an older run.
            registry = json.loads(registry_file.read_text())
            qualified = registry.get("qualified_cases")
            status["qualified"] = {
                "qualified_cases": len(qualified) if isinstance(qualified, list) else qualified,
                "case_count": registry.get("case_count"),
                "finished_at": datetime.fromtimestamp(registry_file.stat().st_mtime, UTC).isoformat(timespec="seconds"),
            }
        return status

    # ---- generation workspace ------------------------------------------

    def export(self, case_id: str, destination: Path, track: str) -> Path:
        """ServiceCraft's own allowlist export: prompt, description, inputs (+ graph)."""
        if track not in TRACKS:
            raise CorpusError(f"Unknown track {track!r}")
        self._run_tool(
            [str(self.root / "corpus/tools/export_inputs.py"), case_id, str(destination), "--track", track],
            cwd=self.root / "corpus/tools",
        )
        return destination

    # ---- evaluator ------------------------------------------------------

    def evaluator_command(self, *args: str) -> tuple[list[str], dict, Path]:
        cwd = self.root / "evaluation"
        env = {**os.environ, "PYTHONPATH": str(cwd), "PYTHONDONTWRITEBYTECODE": "1"}
        return [str(self.tool_python), "-m", "app.service_craft", *args], env, cwd

    def _run_tool(self, args: list[str], cwd: Path, log: Path | None = None) -> str:
        python = str(self.tool_python) if self.tool_python.is_file() else sys.executable
        env = {**os.environ, "PYTHONPATH": str(self.root / "evaluation"), "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run([python, *args], cwd=cwd, env=env, capture_output=True, text=True)
        if log:
            log.write_text(result.stdout + result.stderr)
        if result.returncode:
            raise CorpusError(f"{Path(args[0]).name} failed:\n{(result.stderr or result.stdout)[-3000:]}")
        return result.stdout


def fetch(data_dir: Path, url: str = SOURCE_URL, ref: str | None = None) -> Corpus:
    corpus = Corpus(data_dir)
    if corpus.exists():
        raise CorpusError(f"{corpus.root} already exists")
    data_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", url, str(corpus.root)], check=True)
    if ref:
        subprocess.run(["git", "-C", str(corpus.root), "checkout", "--quiet", ref], check=True)
    meta = {
        "name": NAME,
        "source": url,
        "commit": _git(corpus.root, "rev-parse", "HEAD"),
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (corpus.root / "dataset.json").write_text(json.dumps(meta, indent=2) + "\n")
    return corpus


def setup(corpus: Corpus, *, workers: int = 3, rebuild: bool = False, log=print) -> dict:
    """Tool env, Docker images and local requalification of all controls.

    ServiceCraft's specs pin the authors' local image IDs, which do not exist on a
    new machine. Its documented fix is to requalify the corpus against freshly
    built images; the result goes to `servicecraft-qualified/`, outside the dataset.
    """
    problems = corpus.check()
    if problems:
        raise CorpusError("Dataset check failed:\n" + "\n".join(problems[:20]))

    venv = corpus.tool_python.parent.parent
    if not corpus.tool_python.is_file():
        log(f"creating evaluator environment in {venv}")
        if shutil.which("uv"):
            subprocess.run(["uv", "venv", "--quiet", "--python", "3.12", str(venv)], check=True)
            subprocess.run(["uv", "pip", "install", "--quiet", "--python", str(corpus.tool_python), *TOOL_REQUIREMENTS], check=True)
        else:
            subprocess.run([_python311(), "-m", "venv", str(venv)], check=True)
            subprocess.run([str(corpus.tool_python), "-m", "pip", "install", "--quiet", *TOOL_REQUIREMENTS], check=True)

    evaluation = corpus.root / "evaluation"
    _build_images(corpus, rebuild, log)

    out = corpus.qualified_root
    if out.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        out.rename(out.with_name(f"{out.name}.old-{stamp}"))
    out.mkdir(parents=True)
    tools = evaluation / "tools"
    bench_root = str(corpus.root / "corpus")
    steps = [
        ("prepare", ["prepare_corpus.py", "--benchmark-root", bench_root, "--output", str(out / "prepared-controls")]),
        ("qualify", ["qualify_corpus.py", "--benchmark-root", bench_root, "--prepared", str(out / "prepared-controls"),
                     "--output", str(out / "qualification-runs/corpus"), "--workers", str(workers)]),
        ("assemble", ["assemble_corpus.py", "--benchmark-root", bench_root, "--prepared", str(out / "prepared-controls"),
                      "--qualifications", str(out / "qualification-runs/corpus"), "--output", str(corpus.qualified_corpus)]),
    ]
    for name, args in steps:
        log(f"{name}: running {args[0]}")
        corpus._run_tool([str(tools / args[0]), *args[1:]], cwd=evaluation, log=out / f"{name}.log")

    registry = corpus.registry(qualified=True)
    qualified = registry.get("qualified_cases")
    summary = {
        "dataset": corpus.metadata(),
        "images": {img: _image_id(img) for img in (RUNTIME_IMAGE, ORACLE_IMAGE)},
        "qualified_cases": len(qualified) if isinstance(qualified, list) else qualified,
        "case_count": registry.get("case_count"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / "setup.json").write_text(json.dumps(summary, indent=2) + "\n")
    _keep_images(log)
    return summary


IMAGES = ((RUNTIME_IMAGE, "Dockerfile"), (ORACLE_IMAGE, "Dockerfile.oracles"))
KEEP_ENV = "INLUMEN_BENCH_KEEP_IMAGES"


def _build_images(corpus: Corpus, rebuild: bool, log) -> None:
    evaluation = corpus.root / "evaluation"
    for image, dockerfile in IMAGES:
        if rebuild or not _image_id(image):
            log(f"building {image}")
            subprocess.run(
                ["docker", "build", "--quiet", "-t", image, "-f", str(evaluation / dockerfile), str(evaluation)],
                check=True,
                stdout=subprocess.DEVNULL,
            )


def ensure_images(corpus: Corpus, log=print) -> None:
    """Rebuilds evaluator images that disappeared and checks they are the qualified ones.

    Some servers delete unused images on their own (a Kubernetes kubelet does when the
    disk is over 85% full). A rebuild from Docker's build cache gives the same image ID;
    a different ID means the qualified corpus no longer matches and setup must run again.
    """
    status = corpus.setup_status()
    _build_images(corpus, False, log)
    pinned = (status["qualified"] or {}).get("images") or {}
    changed = [img for img, image_id in pinned.items() if image_id and _image_id(img) != image_id]
    if changed:
        raise CorpusError(f"{', '.join(changed)} changed since setup; run `bench setup` again")
    _keep_images(log)


def _keep_images(log) -> None:
    """With INLUMEN_BENCH_KEEP_IMAGES set, keeps one idle container per evaluator image,
    so automatic image cleanup sees the images as in use."""
    if not os.environ.get(KEEP_ENV):
        return
    for image, _ in IMAGES:
        name = "inlumen-bench-keep-" + image.split(":")[0]
        image_id = _image_id(image)
        current = subprocess.run(["docker", "container", "inspect", "--format", "{{.Image}}", name],
                                 capture_output=True, text=True).stdout.strip()
        if current == image_id:
            continue
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        log(f"keeping {image} in use with container {name}")
        subprocess.run(["docker", "run", "-d", "--name", name, "--restart", "unless-stopped", "--network", "none",
                        "--memory", "32m", "--entrypoint", "sleep", image, "infinity"],
                       check=True, capture_output=True)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _image_id(image: str) -> str | None:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image], capture_output=True, text=True
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def _python311() -> str:
    for name in ("python3.13", "python3.12", "python3.11"):
        if shutil.which(name):
            return name
    if sys.version_info >= (3, 11):
        return sys.executable
    raise CorpusError("Python 3.11+ or uv is required for the evaluator environment")
