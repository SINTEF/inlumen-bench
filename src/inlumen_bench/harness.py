"""Deterministic Dagster wiring around generated node modules.

The generated part of a submission is one `nodes/<asset>.py` per task with
`run(inputs) -> dict`. This module writes the rest: `bench_runtime.py` (input
paths, artifact writing, error capture) and `definitions.py` (one asset per task,
dependencies as asset inputs). Both are plain Python source in the submission.
"""

from __future__ import annotations

from .design import asset_name, producers, topo_order

RUNTIME = '''"""Runtime helpers for the generated nodes (written by inlumen-bench, not generated)."""

import json
import math
import os
import pickle
import traceback
from pathlib import Path

import pandas as pd

INPUT_DIR = Path(os.environ.get("SERVICECRAFT_INPUT_DIR", "inputs"))
OUTPUT_DIR = Path(os.environ.get("SERVICECRAFT_OUTPUT_DIR", "outputs"))


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON artifacts must contain only finite numbers")
    if isinstance(value, dict):
        for v in value.values():
            _finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _finite(v)


def write_artifact(name, value):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    suffix = path.suffix.lower()
    if suffix == ".csv":
        if not isinstance(value, pd.DataFrame):
            value = pd.DataFrame(value)
        value.to_csv(path, index=False)
    elif suffix == ".json":
        _finite(value)
        path.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default))
    elif suffix == ".pkl":
        path.write_bytes(pickle.dumps(value))
    elif isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(str(value))


def _json_default(value):
    for attr in ("item", "tolist"):  # numpy scalars and arrays
        if hasattr(value, attr):
            return getattr(value, attr)()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def run_node(task_id, run, inputs, upstream, outputs):
    """Calls a node with its inputs, checks and writes its artifacts."""
    available = {}
    for produced in upstream.values():
        available.update(produced)
    values = {}
    for item in inputs:
        if item.startswith("input:"):
            values[item] = INPUT_DIR / item.split(":", 1)[1]
        else:
            values[item] = available[item]
    try:
        produced = run(values)
        if not isinstance(produced, dict):
            raise TypeError(f"run() must return a dict of artifacts, got {type(produced).__name__}")
        missing = [name for name in outputs if name not in produced]
        if missing:
            raise KeyError(f"run() did not return artifacts {missing}")
        for name in outputs:
            write_artifact(name, produced[name])
    except Exception:
        diagnostics = os.environ.get("BENCH_DIAGNOSTICS_DIR")
        if diagnostics:
            Path(diagnostics).mkdir(parents=True, exist_ok=True)
            (Path(diagnostics) / f"{task_id}.txt").write_text(traceback.format_exc()[-6000:])
        raise
    return {name: produced[name] for name in outputs}
'''


def definitions(design: dict, names: dict[str, str]) -> str:
    """`definitions.py` for a design; `names` maps task IDs to asset names."""
    made_by = producers(design)
    order = topo_order(design)
    lines = [
        '"""Dagster wiring (written by inlumen-bench from the design, not generated)."""',
        "",
        "import dagster as dg",
        "",
        "from bench_runtime import run_node",
    ]
    for task in order:
        lines.append(f"from nodes import {names[task['id']]} as _{names[task['id']]}")
    lines.append("")
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
            f"    return run_node({name!r}, _{name}.run, {task['inputs']!r},",
            f"                    {{{', '.join(f'{u!r}: {u}' for u in ups)}}}, {task['outputs']!r})",
        ]
    lines += ["", "", f"defs = dg.Definitions(assets=[{', '.join(names[t['id']] for t in order)}])", ""]
    return "\n".join(lines)


def default_names(design: dict) -> dict[str, str]:
    return {t["id"]: asset_name(t["id"]) for t in design["tasks"]}
