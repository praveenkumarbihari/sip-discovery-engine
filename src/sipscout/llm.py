"""LLM access through OpenRouter (OpenAI-compatible API)."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from .config import require_env


def get_client(cfg: dict[str, Any]) -> OpenAI:
    """OpenAI SDK client pointed at OpenRouter."""
    llm = cfg["llm"]
    return OpenAI(base_url=llm["base_url"], api_key=require_env(llm["api_key_env"]))
