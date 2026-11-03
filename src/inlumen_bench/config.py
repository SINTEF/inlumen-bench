"""Locations of the config file, the data folder and the tool cache.

The data folder is resolved in a fixed order: CLI flag, environment variable,
config file, then the platform default from platformdirs.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

import platformdirs

APP = "inlumen-bench"
DATA_DIR_ENV = "INLUMEN_BENCH_DATA_DIR"
CONFIG_DIR_ENV = "INLUMEN_BENCH_CONFIG_DIR"


def config_dir() -> Path:
    """`~/.config/inlumen-bench` on every Unix, so the path is the same on macOS and Linux."""
    if os.environ.get(CONFIG_DIR_ENV):
        return Path(os.environ[CONFIG_DIR_ENV]).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base).expanduser() / APP


def config_path() -> Path:
    return config_dir() / "config.toml"


def cache_dir() -> Path:
    return Path(platformdirs.user_cache_dir(APP))


def load_config() -> dict:
    path = config_path()
    if not path.is_file():
        return {}
    return tomllib.loads(path.read_text())


def save_config(values: dict) -> Path:
    """Writes a flat table of strings, numbers and booleans; enough for this config."""
    lines = []
    for key, value in sorted(values.items()):
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (int, float)):
            rendered = str(value)
        else:
            escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
            rendered = f'"{escaped}"'
        lines.append(f"{key} = {rendered}")
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


@dataclass(frozen=True)
class DataDir:
    path: Path
    source: str  # "flag", "env", "config" or "default"


def resolve_data_dir(flag: str | Path | None = None) -> DataDir:
    if flag:
        return DataDir(Path(flag).expanduser().resolve(), "flag")
    if os.environ.get(DATA_DIR_ENV):
        return DataDir(Path(os.environ[DATA_DIR_ENV]).expanduser().resolve(), "env")
    configured = load_config().get("data_dir")
    if configured:
        return DataDir(Path(configured).expanduser().resolve(), "config")
    return DataDir(Path(platformdirs.user_data_dir(APP)) / "datasets", "default")
