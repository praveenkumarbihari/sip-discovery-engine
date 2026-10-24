"""Shared source machinery: record building, polite HTTP with retries."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, TypeVar

import httpx
from langdetect import DetectorFactory, LangDetectException, detect
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from ..filters import evaluate
from ..models import Record, make_id
from ..privacy import clean_text, hash_author

DetectorFactory.seed = 0
T = TypeVar("T")


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


def retrying(fn: Callable[..., T], *, any_exception: bool = False) -> Callable[..., T]:
    """Wrap fn in 5 attempts with exponential backoff and jitter."""
    kwargs: dict[str, Any] = dict(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=30),
        reraise=True,
    )
    if not any_exception:
        kwargs["retry"] = retry_if_exception(_retryable)
    return retry(**kwargs)(fn)


def detect_lang(text: str) -> str:
    """langdetect language code, or 'unknown'."""
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def to_utc(dt: datetime) -> datetime:
    """Force a datetime to timezone-aware UTC (naive values are treated as UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class Source(ABC):
    """Base class: subclasses yield cleaned, filtered Records."""

    name: str = ""

    def __init__(self, cfg: dict[str, Any], salt: str, conn: Any = None) -> None:
        self.cfg = cfg
        self.salt = salt
        self.conn = conn
        self.stats: Counter[str] = Counter()
        self.delay = float(cfg.get("http", {}).get("delay_seconds", 1.0))
        self.user_agent = cfg.get("http", {}).get("user_agent", "sipscout/0.1")

    @abstractmethod
    def collect(self, *, limit: int | None = None, since: datetime | None = None,
                yes: bool = False, include_competitors: bool = False) -> Iterator[Record]:
        """Yield records. `limit` caps records per container (app/lang, query)."""

    def sleep(self) -> None:
        """Polite delay between unauthenticated calls."""
        time.sleep(self.delay)

    def client(self) -> httpx.Client:
        """HTTP client with the project User-Agent."""
        return httpx.Client(headers={"User-Agent": self.user_agent}, timeout=30, follow_redirects=True)

    def get_json(self, client: httpx.Client, url: str, **params: Any) -> Any:
        """GET JSON with retries and a polite delay."""
        def _go() -> Any:
            resp = client.get(url, params=params or None)
            resp.raise_for_status()
            return resp.json()
        data = retrying(_go)()
        self.sleep()
        return data

    def build(self, *, source: str, native_id: str, url: str, username: str | None, text: str | None,
              created_at: datetime, title: str | None = None, query: str = "", sip_query: bool = False,
              strip_html: bool = False, raw: dict[str, Any] | None = None, **fields: Any) -> Record | None:
        """Clean, scrub, filter and wrap a payload. Returns None (and counts it) if too short."""
        self.stats["fetched"] += 1
        body = clean_text(text, strip_html=strip_html)
        clean_title = clean_text(title, strip_html=strip_html) or None
        combined = f"{clean_title or ''} {body}".strip()
        if len(combined) < int(self.cfg["filters"]["min_text_chars"]):
            self.stats["dropped_short"] += 1
            return None
        matched, relevant = evaluate(combined, self.cfg, sip_query=sip_query)
        return Record(
            id=make_id(source, native_id),
            source=source,
            native_id=native_id,
            url=url,
            author_hash=hash_author(username, self.salt),
            title=clean_title,
            text=body,
            created_at=to_utc(created_at),
            lang=detect_lang(combined),
            query=query,
            matched_keywords=matched,
            is_relevant=relevant,
            raw=raw or {},
            **fields,
        )
