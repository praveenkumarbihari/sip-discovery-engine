"""Configuration loading. All tunables live in config.yaml; secrets live in .env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.yaml"
DEFAULT_DB = ROOT / "data" / "sipscout.db"


class ConfigError(RuntimeError):
    """Raised when required configuration or secrets are missing."""


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load config.yaml (and .env into the process environment)."""
    load_dotenv(ROOT / ".env")
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def require_env(name: str) -> str:
    """Return an environment variable or raise a ConfigError naming it."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"Missing {name}. Set it in .env (see .env.example).")
    return value


def author_salt(cfg: dict[str, Any]) -> str:
    """Salt used to hash usernames."""
    return require_env(cfg["privacy"]["author_salt_env"])
