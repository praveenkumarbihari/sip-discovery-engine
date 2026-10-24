"""Reddit via praw (read-only)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator

import praw

from ..config import require_env
from ..models import Record
from .base import Source, retrying

_GONE = {"[deleted]", "[removed]"}


def _author(obj: Any) -> str | None:
    author = getattr(obj, "author", None)
    return getattr(author, "name", None) if author else None


class RedditSource(Source):
    """Posts (title + selftext) and their comments for SIP queries across subreddits."""

    name = "reddit"

    def _client(self) -> praw.Reddit:
        return praw.Reddit(
            client_id=require_env("REDDIT_CLIENT_ID"),
            client_secret=require_env("REDDIT_CLIENT_SECRET"),
            user_agent=require_env("REDDIT_USER_AGENT"),
            check_for_async=False,
        )

    def post_record(self, sub: Any, query: str) -> Record | None:
        """Convert a praw submission to a Record (None for deleted/removed/short)."""
        body = getattr(sub, "selftext", "") or ""
        if body.strip() in _GONE or (sub.title or "").strip() in _GONE:
            return None
        return self.build(
            source="reddit_post",
            native_id=sub.id,
            url=f"https://www.reddit.com{sub.permalink}",
            username=_author(sub),
            text=body,
            title=sub.title,
            created_at=datetime.fromtimestamp(sub.created_utc, tz=timezone.utc),
            query=query,
            sip_query="sip" in query.lower(),
            container=str(sub.subreddit),
            container_title=sub.title,
            score=sub.score,
            raw={"num_comments": sub.num_comments, "upvote_ratio": getattr(sub, "upvote_ratio", None),
                 "flair": getattr(sub, "link_flair_text", None)},
        )

    def comment_record(self, c: Any, sub: Any, query: str) -> Record | None:
        """Convert a praw comment to a Record."""
        body = getattr(c, "body", "") or ""
        if body.strip() in _GONE:
            return None
        return self.build(
            source="reddit_comment",
            native_id=c.id,
            url=f"https://www.reddit.com{c.permalink}",
            username=_author(c),
            text=body,
            created_at=datetime.fromtimestamp(c.created_utc, tz=timezone.utc),
            query=query,
            sip_query="sip" in (sub.title or "").lower() or "sip" in query.lower(),
            container=str(sub.subreddit),
            container_title=sub.title,
            parent_id=c.parent_id,
            score=c.score,
        )

    def collect(self, *, limit: int | None = None, since: datetime | None = None,
                yes: bool = False, include_competitors: bool = False) -> Iterator[Record]:
        rcfg = self.cfg["reddit"]
        reddit = self._client()
        per_query = limit or int(rcfg["limit_per_query"])
        seen: set[str] = set()
        for sub_name in rcfg["subreddits"]:
            sr = reddit.subreddit(sub_name)
            for query in rcfg["queries"]:
                posts = retrying(lambda: list(sr.search(query, sort="new", time_filter=rcfg["time_filter"], limit=per_query)),
                                 any_exception=True)()
                for sub in posts:
                    if sub.id in seen:
                        continue
                    seen.add(sub.id)
                    created = datetime.fromtimestamp(sub.created_utc, tz=timezone.utc)
                    if since and created < since:
                        continue
                    rec = self.post_record(sub, query)
                    if rec:
                        yield rec
                    if rcfg["fetch_comments"]:
                        sub.comments.replace_more(limit=0)
                        for n, c in enumerate(sub.comments.list()):
                            if n >= int(rcfg["max_comments_per_post"]):
                                break
                            crec = self.comment_record(c, sub, query)
                            if crec and (not since or crec.created_at >= since):
                                yield crec
                    self.sleep()
