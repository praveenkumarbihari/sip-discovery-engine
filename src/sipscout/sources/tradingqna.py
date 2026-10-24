"""TradingQ&A (Discourse) via its public JSON API."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Iterator

import httpx
from rich.console import Console

from ..models import Record
from .base import Source

console = Console()


_QUOTE = re.compile(r"<aside\b[^>]*class=\"[^\"]*\bquote\b[^\"]*\"[^>]*>(?:(?!<aside\b).)*?</aside>", re.DOTALL | re.IGNORECASE)


def strip_quotes(cooked: str) -> str:
    """Remove Discourse quote blocks (they carry another user's name and duplicate their text)."""
    prev = None
    while prev != cooked:
        prev, cooked = cooked, _QUOTE.sub(" ", cooked)
    return cooked


def parse_post(src: "TradingQnASource", base: str, topic: dict[str, Any], post: dict[str, Any], query: str) -> Record | None:
    """Convert a Discourse post to a Record (HTML in `cooked` is stripped)."""
    slug, tid = topic.get("slug", "t"), topic["id"]
    number = post.get("post_number", 1)
    return src.build(
        source="tradingqna",
        native_id=str(post["id"]),
        url=f"{base}/t/{slug}/{tid}/{number}",
        username=post.get("username"),
        text=strip_quotes(post.get("cooked") or ""),
        strip_html=True,
        created_at=datetime.fromisoformat(post["created_at"].replace("Z", "+00:00")),
        query=query,
        sip_query="sip" in query.lower(),
        container=str(tid),
        container_title=topic.get("title"),
        parent_id=str(post["reply_to_post_number"]) if post.get("reply_to_post_number") else None,
        score=int(post.get("like_count") or post.get("score") or 0),
        raw={"post_number": number},
    )


class TradingQnASource(Source):
    """Topics matching SIP queries, with all their posts."""

    name = "tradingqna"

    def collect(self, *, limit: int | None = None, since: datetime | None = None,
                yes: bool = False, include_competitors: bool = False) -> Iterator[Record]:
        tcfg = self.cfg["tradingqna"]
        base = tcfg["base_url"].rstrip("/")
        cap = limit or int(tcfg["max_topics_per_query"])
        seen: set[int] = set()
        with self.client() as client:
            for query in tcfg["queries"]:
                try:
                    data = self.get_json(client, f"{base}/search.json", q=query)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code in (401, 403):
                        console.print("[yellow]TradingQ&A blocked (login wall or 403); skipping source.[/yellow]")
                        self.stats["blocked"] += 1
                        return
                    raise
                for topic in (data.get("topics") or [])[:cap]:
                    if topic["id"] in seen:
                        continue
                    seen.add(topic["id"])
                    detail = self.get_json(client, f"{base}/t/{topic['id']}.json")
                    for post in (detail.get("post_stream") or {}).get("posts", []):
                        rec = parse_post(self, base, detail, post, query)
                        if rec and (not since or rec.created_at >= since):
                            yield rec
