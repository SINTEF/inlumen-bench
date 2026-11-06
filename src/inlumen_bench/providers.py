"""Endpoint profiles for OpenAI-compatible model servers.

Profiles live in `<config dir>/providers.json`, readable only by the owner. An API
key is optional: local servers such as Ollama need none. A key can also be read
from an environment variable named in `api_key_env`, so it never touches disk.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import config_dir, restore_owner

OPENROUTER_URL = "https://openrouter.ai/api/v1"
# Inside the Docker image, "localhost" is the container, so local servers are reached via the host,
# unless the container shares the host's network (BENCH_NETWORK=host).
LOCAL_HOST = os.environ.get("INLUMEN_BENCH_LOCAL_HOST") or (
    "127.0.0.1" if os.environ.get("INLUMEN_BENCH_NETWORK") == "host"
    else "host.docker.internal" if os.environ.get("INLUMEN_BENCH_IN_DOCKER") else "localhost")

PRESETS: dict[str, dict] = {
    "openrouter": {
        "base_url": OPENROUTER_URL,
        "max_parallel": 8,
        "timeout_s": 300,
        "stall_s": 90,
    },
    "ollama": {
        "base_url": f"http://{LOCAL_HOST}:11434/v1",
        "max_parallel": 1,
        "timeout_s": 1800,
        "stall_s": 300,
    },
    "custom": {"base_url": "", "max_parallel": 4, "timeout_s": 600, "stall_s": 120},
}


@dataclass
class Profile:
    name: str
    base_url: str
    api_key: str | None = None
    api_key_env: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    max_parallel: int = 4
    timeout_s: float = 600
    # Seconds without any streamed bytes before a request is treated as stalled.
    stall_s: float = 120
    # Ask the server for a final usage chunk while streaming (OpenAI `stream_options`).
    stream_usage: bool = True
    # Several servers with the same models, used side by side for runs (e.g. one Ollama per
    # GPU, see scripts/ollama-gpus). When set, runs go to these instead of `base_url`; each
    # runs one model at a time. `base_url` is still used to list and test models.
    endpoints: list[str] = field(default_factory=list)

    @property
    def run_endpoints(self) -> list[str]:
        return [u.strip() for u in self.endpoints if u.strip()]

    @property
    def is_openrouter(self) -> bool:
        return self.base_url.rstrip("/").startswith("https://openrouter.ai")

    def resolved_key(self) -> str | None:
        if self.api_key_env:
            return os.environ.get(self.api_key_env) or None
        return self.api_key or None

    def masked(self) -> dict:
        data = asdict(self)
        key = self.api_key
        data["api_key"] = f"{key[:6]}…{key[-4:]}" if key and len(key) > 12 else ("set" if key else None)
        return data


def providers_path() -> Path:
    return config_dir() / "providers.json"


def load_profiles() -> dict[str, Profile]:
    path = providers_path()
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    return {name: Profile(**{**value, "name": name}) for name, value in raw.items()}


def save_profiles(profiles: dict[str, Profile]) -> Path:
    path = providers_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: {k: v for k, v in asdict(p).items() if k != "name"} for name, p in profiles.items()}
    # Create with owner-only permissions before writing any key material.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.chmod(path, 0o600)
    restore_owner(path.parent)
    return path


def upsert_profile(profile: Profile) -> Path:
    profiles = load_profiles()
    profiles[profile.name] = profile
    return save_profiles(profiles)


def remove_profile(name: str) -> bool:
    profiles = load_profiles()
    if name not in profiles:
        return False
    del profiles[name]
    save_profiles(profiles)
    return True


@dataclass(frozen=True)
class ModelRef:
    """`profile:model`, e.g. `openrouter:anthropic/claude-sonnet-5.5` or `ollama:qwen3-coder`."""

    profile: str
    model: str

    @classmethod
    def parse(cls, value: str) -> "ModelRef":
        profile, sep, model = value.partition(":")
        if not sep or not profile or not model:
            raise ValueError(f"Expected 'profile:model', got {value!r}")
        return cls(profile, model)

    def __str__(self) -> str:
        return f"{self.profile}:{self.model}"

    @property
    def slug(self) -> str:
        return "".join(c if c.isalnum() or c in "-." else "-" for c in f"{self.profile}-{self.model}")
