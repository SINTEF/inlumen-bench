"""Renders the shared runtime/submission contract for one case and track.

ServiceCraft expects the operator to give every method the same frozen contract
(entry point, paths, packages, limits). The template is versioned; each episode
records the SHA-256 of the exact text it received.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .config import cache_dir
from .corpus import RUNTIME_IMAGE, Corpus, _image_id

CONTRACT_VERSION = "v1"
TEMPLATE = Path(__file__).parent / "contracts" / f"runtime-contract-{CONTRACT_VERSION}.md"
PYTHON_VERSION = "3.11"
DAGSTER_VERSION = "1.13.12"


def installed_packages(image: str = RUNTIME_IMAGE) -> str:
    """The image's full `pip freeze`, cached per image ID."""
    image_id = _image_id(image)
    if not image_id:
        raise RuntimeError(f"Docker image {image} not found (run `bench setup`)")
    cached = cache_dir() / f"packages-{image_id.split(':')[-1][:16]}.txt"
    if not cached.is_file():
        text = subprocess.check_output(
            ["docker", "run", "--rm", "--network=none", image, "cat", "/opt/evaluator/packages.txt"],
            text=True,
        )
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(text)
    return cached.read_text().strip()


def render(corpus: Corpus, case_id: str, track: str) -> tuple[str, str]:
    spec = json.loads((corpus.root / "evaluation/corpus/cases" / case_id / "evaluation.json").read_text())
    if track == "fixed_graph":
        graph = json.loads((corpus.root / "corpus/pipelines" / case_id / "task_graph.json").read_text())
        mapping = corpus.case_info(case_id)["logical_to_asset"]
        rows = "\n".join(
            f"  - task `{t['id']}` → asset `{mapping.get(t['id'], t['id'])}`" for t in graph["tasks"]
        )
        asset_names = (
            "- Use these asset names for the tasks in `provided_graph.json`:\n" + rows
        )
    else:
        asset_names = "- Choose short snake_case asset names, one per logical task."
    text = TEMPLATE.read_text().format(
        asset_names=asset_names,
        python_version=PYTHON_VERSION,
        dagster_version=DAGSTER_VERSION,
        timeout_seconds=spec.get("timeout_seconds", 900),
        memory=spec.get("memory", "2g"),
        cpus=spec.get("cpus", 1.0),
        packages=installed_packages(),
    )
    return text, hashlib.sha256(text.encode()).hexdigest()
