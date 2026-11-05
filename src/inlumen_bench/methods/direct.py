"""Direct one-shot baseline: one request for the whole bundle, no repair."""

from __future__ import annotations

import json

from .base import BUNDLE_INSTRUCTIONS, EpisodeContext, Method, MethodResult, parse_bundle, write_bundle

SYSTEM = """\
You are an expert data engineer. Write executable local Dagster code that implements
the task exactly as specified. Follow the runtime and submission contract precisely."""


class Direct(Method):
    name = "direct"
    version = "1.1"

    def generate(self, ctx: EpisodeContext) -> MethodResult:
        user = [f"# Task\n\n{ctx.prompt}"]
        if ctx.design is not None:
            design = json.dumps(ctx.design, indent=1)
            (ctx.work_dir / "design.json").write_text(design)
            user.append("# Pipeline design\n\nImplement this task graph: one Dagster asset per task, named "
                        f"after the task id, reading and writing exactly these artifacts.\n\n```json\n{design}\n```")
        elif ctx.provided_graph:
            user.append(f"# Provided task graph (`provided_graph.json`)\n\n```json\n{ctx.provided_graph}```")
        user.append(f"# Runtime and submission contract\n\n{ctx.contract}")
        user.append(f"# Output format\n\n{BUNDLE_INSTRUCTIONS}")
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n\n".join(user)},
        ]
        # Exactly what the baseline receives, for the matched comparison (RQ3).
        (ctx.work_dir / "request-0.json").write_text(json.dumps(messages, indent=1))
        details = ctx.details
        details["conditions"] = {"phase": "implementation", "baseline": True, "repair": False,
                                 "max_repair_attempts": 0, "retries_allowed": 0,
                                 "receives": "requirements, composition graph and runtime contract"}
        ctx.events.emit("phase_started", phase="codegen")
        ctx.events.emit("stage_started", stage="one-shot generation")
        result = ctx.llm.chat(messages, purpose="generate_bundle")
        (ctx.work_dir / "response-0.md").write_text(result.text)

        files = parse_bundle(result.text)
        usage = result.usage or {}
        attempt = {"attempts": 1, "repairs": 0, "first_passing_attempt": None, "initial_valid": None,
                   "final_valid": None, "recovered_by_repair": False,
                   "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}
        if "definitions.py" not in files:
            details["implementation"] = {**attempt, "final_failure_reason": "response contained no definitions.py",
                                         "final_failure_category": "format"}
            return MethodResult(
                "generation_failed",
                error="response contained no definitions.py",
                counters={"model_requests": 1, "code_producing_requests": 1},
                details=details,
            )
        details["implementation"] = attempt
        checkpoint = write_bundle(files, ctx.submission_dir)
        ctx.events.emit("checkpoint", kind="final", sha256=checkpoint, files=sorted(files))
        return MethodResult(
            "generated",
            checkpoints=[{"kind": "final", "sha256": checkpoint}],
            counters={"model_requests": 1, "code_producing_requests": 1, "repairs": 0},
            details=details,
        )
