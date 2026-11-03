"""Shared pieces for methods: episode context, results and code bundle handling."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..budget import Budget
from ..events import EventLog
from ..llm import LLMClient

BUNDLE_INSTRUCTIONS = """\
Return every source file as a fenced code block whose first line is a comment
naming the file, for example:

```python
# file: definitions.py
import dagster as dg
...
```

Return complete files only. `definitions.py` is required."""

_BLOCK = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\n(.*?)```", re.S)
_FILE_LINE = re.compile(r"^\s*#\s*(?:file|path|filename)\s*:\s*(\S+)\s*$", re.I)
_SAFE_PATH = re.compile(r"^(?:src/)?[A-Za-z0-9_][A-Za-z0-9_./-]*\.py$")


@dataclass
class EpisodeContext:
    case_id: str
    track: str
    workspace: Path  # ServiceCraft export: prompt.txt, inputs/, provided_graph.json
    contract: str
    submission_dir: Path
    work_dir: Path  # raw responses and method scratch files
    llm: LLMClient
    events: EventLog
    budget: Budget

    @property
    def prompt(self) -> str:
        return (self.workspace / "prompt.txt").read_text()

    @property
    def provided_graph(self) -> str | None:
        path = self.workspace / "provided_graph.json"
        return path.read_text() if path.is_file() else None


@dataclass
class MethodResult:
    status: str  # "generated" or "generation_failed"
    error: str | None = None
    checkpoints: list[dict] = field(default_factory=list)
    counters: dict = field(default_factory=dict)


class Method:
    name = "base"
    version = "0"

    def generate(self, ctx: EpisodeContext) -> MethodResult:
        raise NotImplementedError


def parse_bundle(text: str) -> dict[str, str]:
    """Files from fenced blocks; a lone unnamed block is taken as definitions.py."""
    files: dict[str, str] = {}
    unnamed: list[str] = []
    for block in _BLOCK.findall(text):
        lines = block.split("\n")
        first = _FILE_LINE.match(lines[0]) if lines else None
        if first:
            path = first.group(1).removeprefix("./")
            if _SAFE_PATH.match(path) and ".." not in path:
                files[path] = "\n".join(lines[1:]).rstrip() + "\n"
        elif block.strip():
            unnamed.append(block)
    if not files and len(unnamed) == 1:
        files["definitions.py"] = unnamed[0].rstrip() + "\n"
    return files


def write_bundle(files: dict[str, str], submission_dir: Path) -> str:
    """Writes a fresh submission and returns its checkpoint hash."""
    submission_dir.mkdir(parents=True, exist_ok=True)
    for old in submission_dir.rglob("*.py"):
        old.unlink()
    for rel, content in files.items():
        target = submission_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return bundle_hash(submission_dir)


def bundle_hash(submission_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(submission_dir.rglob("*.py")):
        digest.update(str(path.relative_to(submission_dir)).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()
