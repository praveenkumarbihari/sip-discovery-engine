"""Pydantic models shared across stages."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

SOURCES = ("playstore", "appstore", "reddit_post", "reddit_comment", "youtube_comment", "tradingqna")


def make_id(source: str, native_id: str) -> str:
    """Stable record id: sha1(source:native_id)."""
    return hashlib.sha1(f"{source}:{native_id}".encode("utf-8")).hexdigest()


def utcnow() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(timezone.utc)


class Record(BaseModel):
    """One piece of public user text."""

    id: str
    source: str
    native_id: str
    app: str | None = None
    container: str | None = None
    container_title: str | None = None
    parent_id: str | None = None
    url: str
    author_hash: str
    title: str | None = None
    text: str
    rating: int | None = None
    score: int | None = None
    created_at: datetime
    app_version: str | None = None
    lang: str = "unknown"
    query: str = ""
    matched_keywords: dict[str, list[str]] = Field(default_factory=dict)
    is_relevant: bool = False
    collected_at: datetime = Field(default_factory=utcnow)
    raw: dict[str, Any] = Field(default_factory=dict)
