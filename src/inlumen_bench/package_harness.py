"""inLUMEN Task packages as a ServiceCraft submission, and designs as inLUMEN graphs.

inLUMEN's code generator writes one Task package per step: a `main.py` that reads
an input manifest and writes its declared artifacts to an output folder, plus a
`node-manifest.json` with its data contract. ServiceCraft wants Dagster assets
that write named artifacts. This module bridges the two, with no generated code:

- `canvas_graph(design, names)`: a bench design (ServiceCraft task graph shape) as
  an inLUMEN canvas graph. Each input file is a Source; each task a Task whose
  output artifacts are declared with their exact file names, so inLUMEN's
  generator writes ServiceCraft's names.
- `submission_files(design, names, packages)`: the submission. Each package's
  `main.py` becomes `nodes/<asset>_main.py`, its manifest `nodes/<asset>_manifest.py`
  (only Python files are allowed), and `definitions.py` runs them in dependency
  order through `bench_inlumen_runtime.py`, which stages inputs in /tmp, runs the
  package in-process and copies its artifacts to `SERVICECRAFT_OUTPUT_DIR`.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath

from .design import asset_name, port_ids, port_kind, producers, topo_order

RUNTIME = '''"""Runs inLUMEN Task packages as Dagster assets (written by inlumen-bench, not generated)."""

import json
import os
import runpy
import shutil
import tempfile
import traceback
from pathlib import Path

INPUT_DIR = Path(os.environ.get("SERVICECRAFT_INPUT_DIR", "inputs"))
OUTPUT_DIR = Path(os.environ.get("SERVICECRAFT_OUTPUT_DIR", "outputs"))
_RUN = {}


def _run_dir():
    if "dir" not in _RUN:
        _RUN["dir"] = Path(tempfile.mkdtemp(prefix="inlumen-packages-"))
    return _RUN["dir"]


def _stage(asset, contract, inputs_dir, upstream):
    """Copies each declared input into the package's input folder; returns manifest entries."""
    entries = []
    for declared in contract.get("inputs", []):
        node = declared.get("source_node", "")
        name = Path(declared["filename"]).name
        if node in upstream:
            if name not in upstream[node]:
                raise FileNotFoundError(f"{asset}: {node} produced no artifact {name}")
            origin = Path(upstream[node][name])
        else:  # A Source step: one of the case's input files.
            origin = INPUT_DIR / name
        target = inputs_dir / declared["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if origin.is_dir():
            shutil.copytree(origin, target)
        else:
            shutil.copy2(origin, target)
        entries.append({**declared, "path": str(target)})
    return entries


def run_package(asset, main_path, manifest, upstream):
    """Runs one package; `upstream` maps upstream assets to {file name: path}.

    Returns this asset's artifacts as {file name: path}, after copying each one
    to SERVICECRAFT_OUTPUT_DIR under its declared name.
    """
    contract = manifest.get("data_contract") or {}
    root = _run_dir() / asset
    if root.exists():
        shutil.rmtree(root)
    inputs_dir, outputs_dir, work_dir = root / "inputs", root / "outputs", root / "work"
    for folder in (inputs_dir, outputs_dir, work_dir):
        folder.mkdir(parents=True)
    env = {
        "PIPELINE_INPUT_DIR": str(inputs_dir),
        "PIPELINE_OUTPUT_DIR": str(outputs_dir),
        "PIPELINE_WORK_DIR": str(work_dir),
        "INLUMEN_INPUT_MANIFEST": str(root / "input_manifest.json"),
        "INLUMEN_OUTPUT_MANIFEST": str(root / "outputs.json"),
        "INLUMEN_CONTEXT_PATH": str(root / "node-manifest.json"),
    }
    saved = {key: os.environ.get(key) for key in env}
    try:
        entries = _stage(asset, contract, inputs_dir, upstream)
        Path(env["INLUMEN_INPUT_MANIFEST"]).write_text(json.dumps(
            {"schema_version": "inlumen.input-manifest@1", "inputs": entries, "files": entries}))
        Path(env["INLUMEN_CONTEXT_PATH"]).write_text(json.dumps(manifest))
        os.environ.update(env)
        try:
            runpy.run_path(str(main_path), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise RuntimeError(f"{asset} exited with status {exc.code}") from exc
        produced = {}
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for declared in contract.get("outputs", []):
            name = declared["filename"]
            path = outputs_dir / name
            if not path.exists():
                raise FileNotFoundError(f"{asset} did not write its artifact {name}")
            if path.is_dir():
                shutil.copytree(path, OUTPUT_DIR / name, dirs_exist_ok=True)
            else:
                shutil.copy2(path, OUTPUT_DIR / name)
            produced[name] = str(path)
        return produced
    except Exception:
        diagnostics = os.environ.get("BENCH_DIAGNOSTICS_DIR")
        if diagnostics:
            Path(diagnostics).mkdir(parents=True, exist_ok=True)
            (Path(diagnostics) / f"{asset}.txt").write_text(traceback.format_exc()[-6000:])
        raise
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
'''


def source_ids(design: dict, reserved: set[str]) -> dict[str, str]:
    """Input item (`input:<file>`) → Source step id, distinct from the asset names."""
    out: dict[str, str] = {}
    items = sorted({i for t in design["tasks"] for i in t.get("inputs", []) if i.startswith("input:")})
    used = set(reserved)
    for item in items:
        base = f"src_{asset_name(item.split(':', 1)[1])}"
        candidate, n = base, 2
        while candidate in used:
            candidate, n = f"{base}_{n}", n + 1
        used.add(candidate)
        out[item] = candidate
    return out


def output_artifact(filename: str, port: str) -> dict:
    from inlumen.contracts.artifact_contract import classify_artifact

    return {"name": PurePosixPath(filename).stem, "filename": filename, "port": port,
            **classify_artifact(filename)}


def canvas_graph(design: dict, names: dict[str, str], title: str = "") -> dict:
    """A bench design as an inLUMEN canvas graph; Task step ids are the asset names."""
    order = topo_order(design)
    sources = source_ids(design, set(names.values()))
    made_by = producers(design)
    out_port: dict[str, tuple[str, str]] = {item: (sid, "data") for item, sid in sources.items()}
    for task in order:
        for pid, artifact in zip(port_ids(task["outputs"], "output"), task["outputs"]):
            out_port[artifact] = (names[task["id"]], pid)

    nodes: list[dict] = []
    for item, sid in sources.items():
        filename = item.split(":", 1)[1]
        nodes.append({"id": sid, "data": {
            "type": "source", "label": filename, "description": f"Input file {filename}.",
            "files": [filename],
            "ports": {"inputs": [], "outputs": [
                {"id": "data", "name": filename, "type": port_kind(filename), "required": True}]},
        }})
    edges: list[dict] = []
    for task in order:
        step = names[task["id"]]
        in_ids = port_ids(task["inputs"], "input")
        out_ids = port_ids(task["outputs"], "output")
        requirements = " ".join(task.get("requirements") or [])
        nodes.append({"id": step, "data": {
            "type": "task",
            "label": task.get("name") or task["id"],
            "description": f"{task.get('operation', '')}: {requirements}".strip(": "),
            "param": dict(task.get("parameters") or {}),
            "ports": {
                "inputs": [{"id": pid, "name": item.split(":", 1)[-1], "type": port_kind(item), "required": True}
                           for pid, item in zip(in_ids, task["inputs"])],
                "outputs": [{"id": pid, "name": artifact, "type": port_kind(artifact)}
                            for pid, artifact in zip(out_ids, task["outputs"])],
            },
            "implementation": {"output_artifacts": [
                output_artifact(artifact, pid) for pid, artifact in zip(out_ids, task["outputs"])]},
        }})
        for pid, item in zip(in_ids, task["inputs"]):
            if item not in out_port or made_by.get(item) == task["id"]:
                continue
            src, src_port = out_port[item]
            edges.append({"id": f"{src}:{src_port}->{step}:{pid}", "source": src, "target": step,
                          "sourceHandle": src_port, "targetHandle": pid})
    return {"nodes": nodes, "edges": edges, "pipeline": {"name": title or design.get("title") or ""}}


def submission_files(design: dict, names: dict[str, str], packages: dict[str, dict[str, str]]) -> dict[str, str]:
    """The Dagster submission for inLUMEN's packages (keyed by asset name)."""
    made_by = producers(design)
    order = topo_order(design)
    files = {"bench_inlumen_runtime.py": RUNTIME, "nodes/__init__.py": ""}
    lines = [
        '"""Dagster wiring around inLUMEN Task packages (written by inlumen-bench, not generated)."""',
        "",
        "from pathlib import Path",
        "",
        "import dagster as dg",
        "",
        "from bench_inlumen_runtime import run_package",
    ]
    for task in order:
        name = names[task["id"]]
        package = packages.get(name) or {}
        if "main.py" not in package or "node-manifest.json" not in package:
            raise ValueError(f"inLUMEN generated no package for {name}")
        files[f"nodes/{name}_main.py"] = package["main.py"]
        manifest = json.loads(package["node-manifest.json"])
        files[f"nodes/{name}_manifest.py"] = (
            '"""Data contract of the inLUMEN package for this asset (node-manifest.json)."""\n\n'
            "import json\n\n"
            f"MANIFEST = json.loads({json.dumps(manifest, indent=1, sort_keys=True)!r})\n")
        lines.append(f"from nodes import {name}_manifest as _{name}")
    lines += ["", '_NODES = Path(__file__).resolve().parent / "nodes"', ""]
    for task in order:
        name = names[task["id"]]
        ups = []
        for item in task["inputs"]:
            src = made_by.get(item)
            if src and src != task["id"] and names[src] not in ups:
                ups.append(names[src])
        lines += [
            "",
            f"@dg.asset(name={name!r})",
            f"def {name}({', '.join(ups)}):",
            f"    return run_package({name!r}, _NODES / {name + '_main.py'!r}, _{name}.MANIFEST,",
            f"                       {{{', '.join(f'{u!r}: {u}' for u in ups)}}})",
        ]
    lines += ["", "", f"defs = dg.Definitions(assets=[{', '.join(names[t['id']] for t in order)}])", ""]
    files["definitions.py"] = "\n".join(lines)
    return files
