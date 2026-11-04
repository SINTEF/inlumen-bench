"""One client for every OpenAI-compatible endpoint (OpenRouter, Ollama, vLLM, ...).

Every provider request, including failed and retried ones, is recorded as events
with latency, raw usage and a provenance flag. Missing usage is recorded as
unavailable, never as zero. Streaming lets the progress view show tokens arriving
and detect stalled requests.
"""

from __future__ import annotations

import json
import random
import threading
import time
import uuid
from dataclasses import dataclass, field

import httpx

from .budget import Budget
from .events import EventLog
from .providers import ModelRef, Profile

APP_TITLE = "inlumen-bench"
PROGRESS_INTERVAL_S = 1.0
INFRA_RETRIES = 3
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

_semaphores: dict[str, threading.BoundedSemaphore] = {}
_semaphores_lock = threading.Lock()


def _profile_slot(profile: Profile) -> threading.BoundedSemaphore:
    """Caps concurrent requests per endpoint profile across all episodes."""
    with _semaphores_lock:
        if profile.name not in _semaphores:
            _semaphores[profile.name] = threading.BoundedSemaphore(max(1, profile.max_parallel))
        return _semaphores[profile.name]


class LLMError(Exception):
    pass


class StallError(LLMError):
    pass


@dataclass
class ChatResult:
    text: str
    request_id: str
    model: str
    resolved_model: str | None
    finish_reason: str | None
    latency_s: float
    usage: dict | None
    usage_provenance: str  # "provider_reported" or "unavailable"
    cost_usd: float | None
    raw_usage: dict | None = field(default=None, repr=False)


class LLMClient:
    def __init__(
        self,
        profile: Profile,
        model: str,
        events: EventLog,
        budget: Budget | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.profile = profile
        self.model = model
        self.events = events
        self.budget = budget or Budget()
        self._transport = transport

    @property
    def ref(self) -> ModelRef:
        return ModelRef(self.profile.name, self.model)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.profile.headers}
        key = self.profile.resolved_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if self.profile.is_openrouter:
            headers.setdefault("X-Title", APP_TITLE)
        return headers

    def _client(self) -> httpx.Client:
        timeout = httpx.Timeout(
            connect=30, read=self.profile.stall_s, write=60, pool=self.profile.timeout_s
        )
        return httpx.Client(timeout=timeout, transport=self._transport)

    def chat(
        self,
        messages: list[dict],
        *,
        purpose: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        parent_id: str | None = None,
    ) -> ChatResult:
        body: dict = {"model": self.model, "messages": messages, "stream": True}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if self.profile.stream_usage:
            body["stream_options"] = {"include_usage": True}
        if self.profile.is_openrouter:
            body["usage"] = {"include": True}

        last_error: Exception | None = None
        for attempt in range(INFRA_RETRIES + 1):
            self.budget.check()
            request_id = uuid.uuid4().hex[:12]
            try:
                with _profile_slot(self.profile):
                    return self._stream(body, request_id, purpose, attempt, parent_id, max_tokens)
            except _Retryable as exc:
                last_error = exc.__cause__ or exc
                if attempt == INFRA_RETRIES:
                    break
                delay = min(60.0, 2**attempt + random.random())
                self.events.emit("log", level="warn", message=f"retrying in {delay:.0f}s: {last_error}")
                time.sleep(delay)
        raise LLMError(f"request failed after {INFRA_RETRIES + 1} tries: {last_error}")

    def _stream(self, body, request_id, purpose, attempt, parent_id, max_tokens) -> ChatResult:
        url = self.profile.base_url.rstrip("/") + "/chat/completions"
        self.events.emit(
            "request_started",
            request_id=request_id,
            parent_id=parent_id,
            purpose=purpose,
            model=str(self.ref),
            attempt=attempt,
            max_tokens=max_tokens,
        )
        start = time.monotonic()
        chunks: list[str] = []
        received_chars = 0
        usage = None
        resolved_model = None
        finish_reason = None
        last_progress = start

        def fail(kind: str, message: str, retryable: bool, status: int | None = None):
            latency = time.monotonic() - start
            self.budget.charge(None)
            self.events.emit(
                "request_failed",
                request_id=request_id,
                purpose=purpose,
                kind=kind,
                status=status,
                message=message[:2000],
                latency_s=round(latency, 3),
                received_chars=received_chars,
                usage_provenance="unavailable",
                budget=self.budget.snapshot(),
            )
            err = (StallError if kind == "stall" else LLMError)(message)
            if retryable:
                raise _Retryable() from err
            raise err

        try:
            with self._client() as client, client.stream(
                "POST", url, headers=self._headers(), json=body
            ) as response:
                if response.status_code >= 400:
                    text = response.read().decode(errors="replace")
                    fail("http", f"HTTP {response.status_code}: {text}", response.status_code in RETRY_STATUS, response.status_code)
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if data.get("error"):
                        fail("provider", json.dumps(data["error"]), True)
                    resolved_model = data.get("model") or resolved_model
                    if data.get("usage"):
                        usage = data["usage"]
                    for choice in data.get("choices") or []:
                        message = choice.get("delta") or {}
                        delta = message.get("content") or ""
                        if delta:
                            chunks.append(delta)
                            received_chars += len(delta)
                        # Thinking models stream reasoning first; it is progress, not reply text.
                        received_chars += len(message.get("reasoning") or message.get("reasoning_content") or "")
                        finish_reason = choice.get("finish_reason") or finish_reason
                    now = time.monotonic()
                    if now - last_progress >= PROGRESS_INTERVAL_S:
                        last_progress = now
                        self.events.emit(
                            "request_progress",
                            request_id=request_id,
                            purpose=purpose,
                            received_chars=received_chars,
                            # Rough live estimate only; the final usage comes from the provider.
                            approx_tokens=received_chars // 4,
                            elapsed_s=round(now - start, 1),
                        )
        except httpx.ReadTimeout:
            self.events.emit("stall_warning", request_id=request_id, seconds=self.profile.stall_s)
            fail("stall", f"no data for {self.profile.stall_s:.0f}s", True)
        except httpx.TransportError as exc:
            fail("transport", f"{type(exc).__name__}: {exc}", True)

        latency = time.monotonic() - start
        cost = None
        if usage and isinstance(usage.get("cost"), (int, float)):
            cost = float(usage["cost"])
        self.budget.charge(cost)
        normalized = _normalize_usage(usage)
        self.events.emit(
            "request_finished",
            request_id=request_id,
            purpose=purpose,
            model=str(self.ref),
            resolved_model=resolved_model,
            finish_reason=finish_reason,
            latency_s=round(latency, 3),
            received_chars=received_chars,
            usage=normalized,
            raw_usage=usage,
            usage_provenance="provider_reported" if usage else "unavailable",
            cost_usd=cost,
            cost_provenance="provider_reported" if cost is not None else "unavailable",
            budget=self.budget.snapshot(),
        )
        self.events.emit("budget_update", **self.budget.snapshot())
        return ChatResult(
            text="".join(chunks),
            request_id=request_id,
            model=self.model,
            resolved_model=resolved_model,
            finish_reason=finish_reason,
            latency_s=latency,
            usage=normalized,
            usage_provenance="provider_reported" if usage else "unavailable",
            cost_usd=cost,
            raw_usage=usage,
        )


class _Retryable(Exception):
    pass


def _normalize_usage(usage: dict | None) -> dict | None:
    """Input/output totals plus subsets (cached input, reasoning) when reported."""
    if not usage:
        return None
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "cached_input_tokens": prompt_details.get("cached_tokens"),
        "reasoning_tokens": completion_details.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def list_models(profile: Profile) -> list[str]:
    client = LLMClient(profile, "", EventLog(None))
    with client._client() as http:
        response = http.get(profile.base_url.rstrip("/") + "/models", headers=client._headers())
        response.raise_for_status()
        data = response.json()
    return sorted(item.get("id", "") for item in data.get("data", []) if item.get("id"))


def ping(profile: Profile, model: str) -> ChatResult:
    client = LLMClient(profile, model, EventLog(None))
    return client.chat(
        [{"role": "user", "content": "Reply with the single word: ok"}],
        purpose="ping",
        # Room for thinking models, which reason before they answer.
        max_tokens=2048,
    )
