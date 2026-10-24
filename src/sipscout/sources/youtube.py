"""YouTube Data API v3: search videos, then pull comments. Quota tracked in SQLite."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterator

from googleapiclient.discovery import build as gbuild
from googleapiclient.errors import HttpError
from rich.console import Console

from ..config import require_env
from ..db import add_quota, quota_used
from ..models import Record
from .base import Source, retrying

console = Console()
COST_SEARCH, COST_COMMENTS, COST_VIDEOS = 100, 1, 1


class QuotaExhausted(Exception):
    """Raised when the daily budget would be exceeded."""


def parse_comment(src: "YouTubeSource", video_id: str, video_title: str, query: str,
                  snippet: dict[str, Any], comment_id: str, parent_id: str | None) -> Record | None:
    """Convert a comment snippet (top-level or reply) to a Record."""
    return src.build(
        source="youtube_comment",
        native_id=comment_id,
        url=f"https://www.youtube.com/watch?v={video_id}&lc={comment_id}",
        username=snippet.get("authorDisplayName"),
        text=snippet.get("textDisplay") or snippet.get("textOriginal"),
        created_at=datetime.fromisoformat(snippet["publishedAt"].replace("Z", "+00:00")),
        query=query,
        sip_query=True,
        container=video_id,
        container_title=video_title,
        parent_id=parent_id,
        score=snippet.get("likeCount"),
    )


class YouTubeSource(Source):
    """Comments on SIP-related videos."""

    name = "youtube"

    def _spend(self, units: int) -> None:
        budget = int(self.cfg["youtube"]["daily_quota_budget"])
        if quota_used(self.conn, "youtube") + units > budget:
            raise QuotaExhausted(f"YouTube daily budget {budget} reached")
        add_quota(self.conn, "youtube", units)

    def collect(self, *, limit: int | None = None, since: datetime | None = None,
                yes: bool = False, include_competitors: bool = False) -> Iterator[Record]:
        ycfg = self.cfg["youtube"]
        yt = gbuild("youtube", "v3", developerKey=require_env("YOUTUBE_API_KEY"), cache_discovery=False)
        max_comments = limit or int(ycfg["max_comments_per_video"])
        seen: set[str] = set()
        try:
            for query in ycfg["queries"]:
                for rel_lang in ycfg.get("relevance_languages", ["en"]):
                    self._spend(COST_SEARCH)
                    res = retrying(yt.search().list(
                        q=query, part="snippet", type="video", regionCode=ycfg["region_code"],
                        relevanceLanguage=rel_lang, maxResults=int(ycfg["max_videos_per_query"]),
                    ).execute, any_exception=True)()
                    for item in res.get("items", []):
                        vid = item["id"]["videoId"]
                        if vid in seen:
                            continue
                        seen.add(vid)
                        title = item["snippet"]["title"]
                        yield from self._comments(yt, vid, title, query, max_comments, since)
        except QuotaExhausted as exc:
            console.print(f"[yellow]{exc}; stopping cleanly.[/yellow]")
            self.stats["quota_stopped"] += 1

    def _comments(self, yt: Any, vid: str, title: str, query: str, cap: int,
                  since: datetime | None) -> Iterator[Record]:
        token = None
        got = 0
        while got < cap:
            self._spend(COST_COMMENTS)
            try:
                res = retrying(yt.commentThreads().list(
                    part="snippet,replies", videoId=vid, maxResults=100, pageToken=token,
                    textFormat="plainText", order="time",
                ).execute, any_exception=True)()
            except HttpError as exc:
                if exc.resp.status in (403, 404):  # comments disabled / video gone
                    self.stats["videos_skipped"] += 1
                    return
                raise
            for thread in res.get("items", []):
                top = thread["snippet"]["topLevelComment"]
                got += 1
                rec = parse_comment(self, vid, title, query, top["snippet"], top["id"], None)
                if rec and (not since or rec.created_at >= since):
                    yield rec
                for reply in (thread.get("replies") or {}).get("comments", []):
                    r = parse_comment(self, vid, title, query, reply["snippet"], reply["id"], top["id"])
                    if r and (not since or r.created_at >= since):
                        yield r
            token = res.get("nextPageToken")
            if not token:
                break
