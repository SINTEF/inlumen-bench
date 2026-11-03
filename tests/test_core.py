import json
import os
import stat

import httpx
import pytest

from inlumen_bench import config, providers
from inlumen_bench.budget import Budget, BudgetExceeded, Limits
from inlumen_bench.events import Bus, EventLog, read_events
from inlumen_bench.llm import LLMClient, LLMError
from inlumen_bench.methods.base import parse_bundle
from inlumen_bench.providers import ModelRef, Profile
from inlumen_bench.runner import usage_totals
from inlumen_bench.validate import mandatory_artifacts


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv(config.CONFIG_DIR_ENV, str(tmp_path / "cfg"))
    monkeypatch.delenv(config.DATA_DIR_ENV, raising=False)


def test_data_dir_order(tmp_path, monkeypatch):
    assert config.resolve_data_dir().source == "default"
    config.save_config({"data_dir": str(tmp_path / "from-config")})
    assert config.resolve_data_dir().source == "config"
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "from-env"))
    assert config.resolve_data_dir().path == (tmp_path / "from-env").resolve()
    assert config.resolve_data_dir(tmp_path / "flag").source == "flag"


def test_profiles_are_owner_only_and_masked():
    path = providers.upsert_profile(Profile("or", "https://openrouter.ai/api/v1", api_key="sk-or-v1-abcdef123456"))
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    loaded = providers.load_profiles()["or"]
    assert loaded.is_openrouter
    assert "abcdef" not in json.dumps(loaded.masked())


def test_model_ref():
    ref = ModelRef.parse("openrouter:anthropic/claude-sonnet-5.5")
    assert (ref.profile, ref.model) == ("openrouter", "anthropic/claude-sonnet-5.5")
    assert "/" not in ref.slug
    with pytest.raises(ValueError):
        ModelRef.parse("no-profile")


def test_parse_bundle_named_and_unnamed():
    text = "Here:\n```python\n# file: definitions.py\nimport dagster\n```\n```python\n# file: src/helpers.py\nX = 1\n```"
    assert set(parse_bundle(text)) == {"definitions.py", "src/helpers.py"}
    assert set(parse_bundle("```python\nimport dagster\n```")) == {"definitions.py"}
    assert parse_bundle("```python\n# file: ../evil.py\nx\n```") == {}


def test_mandatory_artifacts():
    prompt = "5. Mandatory output artifacts\n\n- `a.csv`\n- `b.json`\n\n6. Execution\n\n- `not_me.csv`\n"
    assert mandatory_artifacts(prompt) == ["a.csv", "b.json"]


def _sse(chunks: list[dict]) -> bytes:
    return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"


def test_streaming_records_usage_cost_and_progress(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, content=_sse([
            {"model": "m-resolved", "choices": [{"delta": {"content": "hel"}}]},
            {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
                                       "cost": 0.0012, "prompt_tokens_details": {"cached_tokens": 4}}},
        ]))

    events = EventLog(tmp_path / "events.jsonl", "ep", bus=Bus())
    client = LLMClient(Profile("local", "http://x/v1"), "m", events, transport=httpx.MockTransport(handler))
    result = client.chat([{"role": "user", "content": "hi"}], purpose="test")
    assert result.text == "hello"
    assert seen["auth"] is None  # no key configured → no Authorization header
    assert seen["body"]["stream_options"] == {"include_usage": True}
    assert result.usage["cached_input_tokens"] == 4 and result.cost_usd == 0.0012
    totals = usage_totals(tmp_path / "events.jsonl")
    assert totals["requests"] == 1 and totals["input_tokens"] == 10 and totals["cost_usd"] == 0.0012


def test_missing_usage_is_unavailable_not_zero(tmp_path):
    handler = lambda r: httpx.Response(200, content=_sse([{"choices": [{"delta": {"content": "x"}}]}]))
    events = EventLog(tmp_path / "e.jsonl", "ep", bus=Bus())
    budget = Budget()
    client = LLMClient(Profile("l", "http://x/v1"), "m", events, budget, transport=httpx.MockTransport(handler))
    result = client.chat([], purpose="t")
    assert result.usage is None and result.usage_provenance == "unavailable"
    assert budget.cost_incomplete
    assert usage_totals(tmp_path / "e.jsonl")["usage_unavailable"] == 1


def test_http_error_is_recorded_and_not_retried_for_400(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": "bad model"})

    events = EventLog(tmp_path / "e.jsonl", "ep", bus=Bus())
    client = LLMClient(Profile("l", "http://x/v1"), "m", events, transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError):
        client.chat([], purpose="t")
    assert len(calls) == 1
    assert [e["type"] for e in read_events(tmp_path / "e.jsonl")] == ["request_started", "request_failed"]


def test_budget_blocks_next_request():
    budget = Budget(Limits(max_requests=1))
    budget.charge(0.1)
    with pytest.raises(BudgetExceeded):
        budget.check()
