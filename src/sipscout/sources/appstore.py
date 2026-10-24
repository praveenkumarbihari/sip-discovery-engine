"""Apple App Store reviews via the public iTunes RSS feed (about 500 most recent per app)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterator

from rich.console import Console
from rich.prompt import Confirm

from ..models import Record
from .base import Source

console = Console()


def parse_entry(src: "AppStoreSource", app_key: str, app_id: str, entry: dict[str, Any]) -> Record | None:
    """Convert one RSS entry to a Record. Entries without a rating (the app itself) are skipped."""
    if "im:rating" not in entry:
        return None
    review_id = entry["id"]["label"]
    raw = {k: v for k, v in entry.items() if k != "author"}
    link = (entry.get("link") or {}).get("attributes", {}).get("href") or f"https://apps.apple.com/in/app/id{app_id}"
    return src.build(
        source="appstore",
        native_id=review_id,
        url=link,
        username=((entry.get("author") or {}).get("name") or {}).get("label"),
        text=(entry.get("content") or {}).get("label"),
        title=(entry.get("title") or {}).get("label"),
        created_at=datetime.fromisoformat(entry["updated"]["label"]),
        query="appstore:rss",
        raw=raw,
        app=app_key,
        container=app_id,
        rating=int(entry["im:rating"]["label"]),
        app_version=(entry.get("im:version") or {}).get("label"),
    )


class AppStoreSource(Source):
    """App Store reviews, newest first."""

    name = "appstore"

    def _resolve(self, app: dict[str, Any], yes: bool) -> str | None:
        if app.get("appstore_id"):
            return str(app["appstore_id"])
        with self.client() as client:
            data = self.get_json(client, "https://itunes.apple.com/search", term=app["name"],
                                 country=self.cfg["appstore"]["country"], entity="software", limit=3)
        results = data.get("results", [])
        if not results:
            console.print(f"[yellow]No App Store match for {app['name']}[/yellow]")
            return None
        top = results[0]
        console.print(f"App Store match for {app['name']}: {top['trackName']} ({top['trackId']})")
        try:
            if yes or Confirm.ask("Use this app?", default=True):
                return str(top["trackId"])
        except EOFError:
            console.print("[yellow]No terminal to confirm; rerun with --yes to accept.[/yellow]")
        return None

    def collect(self, *, limit: int | None = None, since: datetime | None = None,
                yes: bool = False, include_competitors: bool = False) -> Iterator[Record]:
        acfg = self.cfg["appstore"]
        apps = self.cfg["apps"]
        if not include_competitors:
            apps = {k: v for k, v in apps.items() if k == "groww"}
        for key, app in apps.items():
            app_id = self._resolve(app, yes)
            if not app_id:
                continue
            got = 0
            with self.client() as client:
                for page in range(1, int(acfg["pages"]) + 1):
                    url = (f"https://itunes.apple.com/{acfg['country']}/rss/customerreviews/"
                           f"page={page}/id={app_id}/sortby=mostrecent/json")
                    data = self.get_json(client, url)
                    entries = (data.get("feed") or {}).get("entry") or []
                    if isinstance(entries, dict):
                        entries = [entries]
                    if not entries:
                        break
                    stop = False
                    for entry in entries:
                        rec = parse_entry(self, key, app_id, entry)
                        if rec is None:
                            continue
                        if since and rec.created_at < since:
                            stop = True
                            break
                        got += 1
                        yield rec
                        if limit and got >= limit:
                            stop = True
                            break
                    if stop:
                        break
