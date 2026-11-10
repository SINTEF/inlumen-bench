"""Pipeline designs: a task graph in the shape of ServiceCraft's `task_graph.json`.

A design has input files and tasks. Each task names the artifacts it reads
(input file IDs such as `input:data.csv`, or artifacts of other tasks) and the
artifacts it writes. Dependencies follow from artifacts, as in ServiceCraft.
The same structure is exported as inLUMEN Project JSON for the Import button.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import PurePosixPath

PROJECT_SCHEMA = "inlumen.project@2"
# Dagster passes the execution context as `context`, so an asset can't use that name.
RESERVED_ASSET_NAMES = {"context"}
_IDENT = re.compile(r"[^a-z0-9_]+")
# Port kinds by file extension, as in inLUMEN's shared/port_types.json (ADR 0003):
# a port shows the kind of the artifact that flows through it.
PORT_KINDS = {
    **dict.fromkeys(("csv", "tsv", "parquet", "xlsx", "xls", "feather"), "table"),
    **dict.fromkeys(("json", "jsonl", "ndjson", "geojson"), "json"),
    **dict.fromkeys(("txt", "md", "xml", "html"), "text"),
    **dict.fromkeys(("png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"), "image"),
    **dict.fromkeys(("pdf", "docx", "doc", "pptx"), "document"),
    **dict.fromkeys(("pkl", "pickle", "joblib", "onnx", "pt", "safetensors"), "model"),
}


def asset_name(task_id: str) -> str:
    name = _IDENT.sub("_", task_id.lower()).strip("_") or "task"
    if name[0].isdigit():
        name = f"t_{name}"
    return f"{name}_task" if name in RESERVED_ASSET_NAMES else name


def producers(design: dict) -> dict[str, str]:
    """Artifact name → ID of the task that writes it."""
    return {out: t["id"] for t in design["tasks"] for out in t.get("outputs", [])}


def edges(design: dict) -> list[tuple[str, str, str]]:
    """(upstream task, downstream task, artifact) for every artifact a task reads."""
    made_by = producers(design)
    out = []
    for task in design["tasks"]:
        for item in task.get("inputs", []):
            if item in made_by and made_by[item] != task["id"]:
                out.append((made_by[item], task["id"], item))
    return out


def topo_order(design: dict) -> list[dict]:
    tasks = {t["id"]: t for t in design["tasks"]}
    upstream = {tid: set() for tid in tasks}
    for src, dst, _ in edges(design):
        upstream[dst].add(src)
    order, done = [], set()
    while len(order) < len(tasks):
        ready = [tid for tid in tasks if tid not in done and upstream[tid] <= done]
        if not ready:
            raise ValueError("the task graph has a cycle")
        for tid in ready:  # Keep the author's order among ready tasks.
            order.append(tasks[tid])
            done.add(tid)
    return order


def check(design: dict, mandatory: list[str], input_files: list[str]) -> list[str]:
    """Structural problems a method may be told about (all from public information)."""
    problems = []
    tasks = design.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return ["the design has no tasks"]
    ids = [t.get("id") for t in tasks]
    if any(not isinstance(i, str) or not i for i in ids):
        problems.append("every task needs a non-empty string `id`")
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        problems.append(f"duplicate task ids: {sorted(dupes)}")
    names = [asset_name(i) for i in ids if isinstance(i, str)]
    if len(set(names)) != len(names):
        problems.append("task ids must stay distinct after conversion to snake_case asset names")
    written: dict[str, list[str]] = {}
    for t in tasks:
        for out in t.get("outputs", []):
            written.setdefault(out, []).append(t.get("id"))
    for art, writers in written.items():
        if len(writers) > 1:
            problems.append(f"artifact {art} is written by several tasks: {writers}")
    known_inputs = {f"input:{name}" for name in input_files}
    for t in tasks:
        if not t.get("outputs"):
            problems.append(f"task {t.get('id')} writes no artifact")
        for item in t.get("inputs", []):
            if item.startswith("input:"):
                if item not in known_inputs:
                    problems.append(f"task {t.get('id')} reads unknown input file {item}")
            elif item not in written:
                problems.append(f"task {t.get('id')} reads {item}, which no task writes")
    read = {i for t in tasks for i in t.get("inputs", [])}
    unread = sorted(known_inputs - read)
    if unread:
        problems.append(f"input files no task reads (each input starts its own branch): {unread}")
    missing = [a for a in mandatory if a not in written]
    if missing:
        problems.append(f"mandatory artifacts not written by any task: {missing}")
    if not problems:
        try:
            topo_order(design)
        except ValueError as exc:
            problems.append(str(exc))
    return problems


def from_task_graph(graph: dict) -> dict:
    """ServiceCraft's (provided) task graph → design; drops evaluator-only fields.

    Keeps each artifact's required columns: they are part of the artifact contract
    the provided graph gives every method (the direct baseline reads them in the graph).
    """
    design = {
        "title": graph.get("title", ""),
        "inputs": [{"id": i["id"], "path": i["path"]} for i in graph["inputs"]],
        "tasks": [
            {k: t.get(k, default) for k, default in
             (("id", ""), ("name", ""), ("operation", ""), ("inputs", []), ("outputs", []),
              ("requirements", []), ("parameters", {}))}
            for t in graph["tasks"]
        ],
    }
    columns = {a["id"]: list(a.get("required_columns") or []) for a in graph.get("artifacts") or [] if a.get("id")}
    if any(columns.values()):
        design["artifact_columns"] = {k: v for k, v in columns.items() if v}
    return design


def parse_design(text: str) -> dict:
    """The first JSON object in a reply (fenced or bare)."""
    fenced = re.search(r"```(?:json)?\s*\n(\{.*?\})\s*```", text, re.S)
    raw = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    if not raw:
        raise ValueError("no JSON object found in the reply")
    design = json.loads(raw)
    if not isinstance(design, dict):
        raise ValueError("the design must be a JSON object")
    design.setdefault("inputs", [])
    for task in design.get("tasks", []):
        for key, default in (("name", ""), ("operation", ""), ("inputs", []), ("outputs", []),
                             ("requirements", []), ("parameters", {})):
            task.setdefault(key, default)
    return design


def port_kind(artifact: str) -> str:
    """inLUMEN port type of one file artifact (`any` when the extension is unknown)."""
    return PORT_KINDS.get(PurePosixPath(artifact).suffix.lstrip(".").lower(), "any")


def port_ids(artifacts: list[str], single: str) -> list[str]:
    """Port ids for a node side: inLUMEN's default id when there is one port, else one
    named port per artifact (a joining Task declares one named input per branch)."""
    if len(artifacts) == 1:
        return [single]
    used: set[str] = set()
    ids = []
    for artifact in artifacts:
        base = _IDENT.sub("_", PurePosixPath(artifact.split(":", 1)[-1]).stem.lower()).strip("_") or "port"
        base = base if base[0].isalpha() else f"f_{base}"
        base = base[:28]
        candidate, n = base, 2
        while candidate in used:
            candidate, n = f"{base}_{n}", n + 1
        used.add(candidate)
        ids.append(candidate)
    return ids


def _port(port_id: str, artifact: str, description: str = "") -> dict:
    return {"id": port_id, "name": artifact.split(":", 1)[-1], "type": port_kind(artifact),
            "required": True, "description": description}


def to_project_json(design: dict, name: str, description: str = "") -> dict:
    """inLUMEN Project JSON (`inlumen.project@2`), importable from the canvas."""
    order = topo_order(design)
    made_by = producers(design)
    depth: dict[str, int] = {}
    for task in order:
        ups = [depth[made_by[i]] for i in task["inputs"] if i in made_by and made_by[i] != task["id"]]
        depth[task["id"]] = 1 + (max(ups) if ups else 0)

    nodes, connections, rows = [], [], {}

    def place(col: int) -> dict:
        row = rows.get(col, 0)
        rows[col] = row + 1
        return {"x": 80 + col * 320, "y": 80 + row * 180}

    # Each input file has its own role, so it is its own Source and starts its own
    # branch; branches meet at a Task with one named input per upstream artifact.
    input_ids = {i["id"] for i in design.get("inputs", [])} | {
        i for t in design["tasks"] for i in t["inputs"] if i.startswith("input:")}
    for input_id in sorted(input_ids):
        file_name = input_id.split(":", 1)[-1]
        nodes.append({
            "id": f"src_{asset_name(file_name)}", "kind": "source",
            "template": {"id": "source.file", "name": "File"},
            "label": file_name, "description": f"Input file {file_name}.",
            "position": place(0), "inputs": [], "outputs": [_port("data", file_name, "Input file.")],
            "parameters": {"filename": file_name},
        })
    out_port: dict[str, tuple[str, str]] = {}  # artifact -> (node, port)
    for input_id in input_ids:
        out_port[input_id] = (f"src_{asset_name(input_id.split(':', 1)[1])}", "data")
    for task in order:
        for port_id, out in zip(port_ids(task["outputs"], "output"), task["outputs"]):
            out_port[out] = (asset_name(task["id"]), port_id)
    for task in order:
        tid = asset_name(task["id"])
        requirements = " ".join(task.get("requirements", []))
        in_ids = port_ids(task["inputs"], "input")
        joins = len(task["inputs"]) > 1
        nodes.append({
            "id": tid, "kind": "task",
            "template": {"id": "task.blank", "name": "Blank Task"},
            "label": task.get("name") or task["id"],
            "description": (f"{task.get('operation', '')}: {requirements}").strip(": "),
            "position": place(depth[task["id"]]),
            "inputs": [_port(pid, item, f"{item.split(':', 1)[-1]} branch joined by this step." if joins else "")
                       for pid, item in zip(in_ids, task["inputs"])],
            "outputs": [_port(pid, o) for pid, o in zip(port_ids(task["outputs"], "output"), task["outputs"])],
            "parameters": task.get("parameters", {}),
            "implementation": {"kind": "python",
                               # The artifact contract inLUMEN codegen generates against.
                               "output_artifacts": [
                                   {"name": PurePosixPath(o).stem, "filename": o, "port": pid,
                                    "kind": "binary" if port_kind(o) == "any" else port_kind(o)}
                                   for pid, o in zip(port_ids(task["outputs"], "output"), task["outputs"])]},
        })
        for pid, item in zip(in_ids, task["inputs"]):
            if item not in out_port or (item in made_by and made_by[item] == task["id"]):
                continue
            src, src_port = out_port[item]
            connections.append({"id": f"{src}:{src_port}->{tid}:{pid}",
                                "from": {"node": src, "port": src_port}, "to": {"node": tid, "port": pid}})
    consumed = {i for t in design["tasks"] for i in t["inputs"]}
    last_col = max(depth.values(), default=0) + 1
    for task in order:
        for out in task["outputs"]:
            if out in consumed:
                continue
            dest = f"dst_{asset_name(out)}"
            src, src_port = out_port[out]
            nodes.append({
                "id": dest, "kind": "destination",
                "template": {"id": "destination.file", "name": "File"},
                "label": out, "description": f"Writes {out}.", "position": place(last_col),
                "inputs": [_port("data", out)], "outputs": [], "parameters": {"filename": out},
            })
            connections.append({"id": f"{src}:{src_port}->{dest}:data",
                                "from": {"node": src, "port": src_port}, "to": {"node": dest, "port": "data"}})
    return {
        "schema_version": PROJECT_SCHEMA,
        "project": {"name": name, "description": description,
                    "exported_at": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")},
        "pipeline": {"nodes": nodes, "connections": connections},
    }
