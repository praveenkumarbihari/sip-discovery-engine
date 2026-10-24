"""Google Play Store reviews via google-play-scraper."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterator

from google_play_scraper import Sort, reviews, search
from rich.console import Console
from rich.prompt import Confirm

from ..models import Record
from .base import Source, retrying

console = Console()
_DROP_RAW = ("userName", "userImage")


def parse_review(src: "PlayStoreSource", app_key: str, package: str, lang: str, item: dict[str, Any]) -> Record | None:
    """Convert one google-play-scraper review dict into a Record."""
    raw = {k: v for k, v in item.items() if k not in _DROP_RAW}
    return src.build(
        source="playstore",
        native_id=item["reviewId"],
        url=f"https://play.google.com/store/apps/details?id={package}&reviewId={item['reviewId']}",
        username=item.get("userName"),
        text=item.get("content"),
        created_at=item["at"],
        query=f"playstore:{lang}",
        raw=raw,
        app=app_key,
        container=package,
        rating=item.get("score"),
        score=item.get("thumbsUpCount"),
        app_version=item.get("reviewCreatedVersion") or item.get("appVersion"),
    )


class PlayStoreSource(Source):
    """Newest-first Play Store reviews per app and language."""

    name = "playstore"

    def _apps(self, include_competitors: bool) -> dict[str, dict[str, Any]]:
        apps = self.cfg["apps"]
        return apps if include_competitors else {k: v for k, v in apps.items() if k == "groww"}

    def _resolve(self, key: str, app: dict[str, Any], yes: bool) -> str | None:
        if app.get("play_package"):
            return app["play_package"]
        hits = retrying(search, any_exception=True)(app["name"], lang="en", country=self.cfg["playstore"]["country"], n_hits=3)
        if not hits:
            console.print(f"[yellow]No Play Store match for {app['name']}[/yellow]")
            return None
        top = hits[0]
        console.print(f"Play Store match for {app['name']}: {top['title']} ({top['appId']})")
        try:
            if yes or Confirm.ask("Use this app?", default=True):
                return top["appId"]
        except EOFError:
            console.print("[yellow]No terminal to confirm; rerun with --yes to accept.[/yellow]")
        return None

    def collect(self, *, limit: int | None = None, since: datetime | None = None,
                yes: bool = False, include_competitors: bool = False) -> Iterator[Record]:
        pcfg = self.cfg["playstore"]
        cap = limit or int(pcfg["max_reviews_per_app_lang"])
        for key, app in self._apps(include_competitors).items():
            package = self._resolve(key, app, yes)
            if not package:
                continue
            for lang in pcfg["langs"]:
                token = None
                got = 0
                done = False
                while not done and got < cap:
                    batch, token = retrying(reviews, any_exception=True)(
                        package, lang=lang, country=pcfg["country"], sort=Sort.NEWEST,
                        count=min(200, cap - got), continuation_token=token,
                    )
                    if not batch:
                        break
                    for item in batch:
                        got += 1
                        if since and item["at"].replace(tzinfo=since.tzinfo) < since:
                            done = True
                            break
                        rec = parse_review(self, key, package, lang, item)
                        if rec:
                            yield rec
                    if token is None:
                        break
                    self.sleep()
