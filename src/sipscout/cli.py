"""sipscout command line."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table

from . import db
from .config import ROOT, ConfigError, author_salt, load_config
from .filters import evaluate
from .sources import SOURCE_CLASSES

app = typer.Typer(help="SIP Discovery Engine", no_args_is_help=True, add_completion=False)
console = Console()
BATCH = 200


def _since(cfg: dict, override: Optional[str]) -> datetime | None:
    value = override or cfg["date_window"]["since"]
    if not value:
        return None
    return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc)


@app.command()
def doctor() -> None:
    """Check keys and reachability for every source."""
    cfg = load_config()
    table = Table("Check", "Status", "Detail")
    problems = 0

    def row(name: str, ok: bool, detail: str = "") -> None:
        nonlocal problems
        problems += 0 if ok else 1
        table.add_row(name, "[green]OK[/green]" if ok else "[red]MISSING[/red]", detail)

    row("AUTHOR_SALT", bool(os.environ.get(cfg["privacy"]["author_salt_env"])), "needed to hash usernames")
    row(cfg["llm"]["api_key_env"], bool(os.environ.get(cfg["llm"]["api_key_env"])), "Stage 2+ and Ask page (OpenRouter)")
    for var in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"):
        row(var, bool(os.environ.get(var)), "reddit")
    row("YOUTUBE_API_KEY", bool(os.environ.get("YOUTUBE_API_KEY")), "youtube")

    probes = {
        "Play Store": "https://play.google.com/store/apps/details?id=" + str(cfg["apps"]["groww"]["play_package"]),
        "App Store": "https://itunes.apple.com/search?term=groww&country=in&entity=software&limit=1",
        "TradingQ&A": cfg["tradingqna"]["base_url"] + "/search.json?q=SIP",
    }
    for name, url in probes.items():
        try:
            r = httpx.get(url, timeout=15, follow_redirects=True, headers={"User-Agent": cfg["http"]["user_agent"]})
            row(f"{name} reachable", r.status_code < 400, f"HTTP {r.status_code}")
        except Exception as exc:  # noqa: BLE001
            row(f"{name} reachable", False, type(exc).__name__)
    console.print(table)
    console.print("[green]All checks passed.[/green]" if not problems else f"[yellow]{problems} item(s) need attention.[/yellow]")


@app.command()
def collect(
    source: str = typer.Option("all", help="playstore|appstore|reddit|youtube|tradingqna|all"),
    include_competitors: bool = typer.Option(False, "--include-competitors"),
    since: Optional[str] = typer.Option(None, help="YYYY-MM-DD, overrides config"),
    limit: Optional[int] = typer.Option(None, help="Cap per app/language, query or video"),
    yes: bool = typer.Option(False, "--yes", help="Accept app-match prompts"),
) -> None:
    """Stage 1: collect public text into SQLite."""
    cfg = load_config()
    try:
        salt = author_salt(cfg)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    names = list(SOURCE_CLASSES) if source == "all" else [source]
    unknown = [n for n in names if n not in SOURCE_CLASSES]
    if unknown:
        console.print(f"[red]Unknown source: {unknown}[/red]")
        raise typer.Exit(2)
    conn = db.connect()
    since_dt = _since(cfg, since)
    summary = Table("Source", "Fetched", "Kept", "Dropped short", "Inserted", "Updated", "Errors", "Notes")

    for name in names:
        src = SOURCE_CLASSES[name](cfg, salt, conn)
        run_id = db.start_run(conn, "collect", name)
        kept = inserted = updated = errors = 0
        notes = ""
        batch = []
        console.print(f"[bold]Collecting {name}...[/bold]")
        try:
            for rec in src.collect(limit=limit, since=since_dt, yes=yes, include_competitors=include_competitors):
                batch.append(rec)
                kept += 1
                if len(batch) >= BATCH:
                    i, u = db.upsert_records(conn, batch)
                    inserted, updated, batch = inserted + i, updated + u, []
                    console.print(f"  {kept} kept so far")
        except Exception as exc:  # noqa: BLE001 - one failing source must not stop the rest
            errors += 1
            notes = f"{type(exc).__name__}: {exc}"[:300]
            console.print(f"[red]{name} failed: {notes}[/red]")
        if batch:
            i, u = db.upsert_records(conn, batch)
            inserted, updated = inserted + i, updated + u
        if src.stats.get("blocked"):
            notes = (notes + " blocked/login wall").strip()
        if src.stats.get("quota_stopped"):
            notes = (notes + " stopped at quota budget").strip()
        db.finish_run(conn, run_id, src.stats["fetched"], inserted, updated, errors, notes)
        summary.add_row(name, str(src.stats["fetched"]), str(kept), str(src.stats["dropped_short"]),
                        str(inserted), str(updated), str(errors), notes)
    console.print(summary)
    console.print(f"Total records in database: {db.count_records(conn)}")


@app.command("filter")
def filter_cmd() -> None:
    """Re-run the keyword pre-filter over stored records (after editing config keywords)."""
    cfg = load_config()
    conn = db.connect()
    changed = relevant = 0
    for r in conn.execute("SELECT id, title, text, query, is_relevant FROM records").fetchall():
        combined = f"{r['title'] or ''} {r['text']}".strip()
        matched, rel = evaluate(combined, cfg, sip_query="sip" in (r["query"] or "").lower())
        relevant += int(rel)
        if int(rel) != r["is_relevant"]:
            changed += 1
        conn.execute("UPDATE records SET matched_keywords=?, is_relevant=? WHERE id=?",
                     (json.dumps(matched, ensure_ascii=False), int(rel), r["id"]))
    conn.commit()
    console.print(f"{relevant} keyword-relevant records ({changed} changed).")


@app.command()
def publish() -> None:
    """Stage 4: write the scrubbed snapshot to app/data/ for the Streamlit app."""
    from .publish import APP_DATA, publish as do_publish

    meta = do_publish(load_config(), db.connect())
    console.print(f"Wrote snapshot to {APP_DATA}")
    console.print(meta["counts"])
    if not meta["has_extraction"]:
        console.print("[yellow]No extractions yet: the app shows keyword-relevant records only.[/yellow]")


@app.command("set-key")
def set_key(name: str = typer.Argument(..., help="e.g. OPENROUTER_API_KEY, REDDIT_CLIENT_ID")) -> None:
    """Prompt (hidden) for a secret and write it to .env; the value is never echoed."""
    from getpass import getpass

    allowed = {"OPENROUTER_API_KEY", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "YOUTUBE_API_KEY"}
    if name not in allowed:
        console.print(f"[red]Name must be one of {sorted(allowed)}[/red]")
        raise typer.Exit(2)
    value = getpass(f"{name} (input hidden): ").strip().strip("\"'")
    if not value:
        console.print("[red]Nothing entered; .env unchanged.[/red]")
        raise typer.Exit(1)
    path = ROOT / ".env"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    out, done = [], False
    for ln in lines:
        if ln.startswith(f"{name}="):
            if done:  # drop duplicate lines; the last one would otherwise win
                continue
            ln, done = f"{name}={value}", True
        out.append(ln)
    if not done:
        out.append(f"{name}={value}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    console.print(f"Saved {name} to .env ({len(value)} characters).")


@app.command()
def extract(
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate tokens and cost; call nothing"),
    max_records: Optional[int] = typer.Option(None, "--max-records", help="Cap records this run"),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Prompt version, e.g. v2"),
    only_golden: bool = typer.Option(False, "--only-golden"),
) -> None:
    """Stage 2: extract structured fields from keyword-relevant records."""
    from . import extract as ex

    cfg = load_config()
    pv = prompt or cfg["extraction"]["prompt_version"]
    conn = db.connect()
    try:
        only = ex.golden_ids() if only_golden else None
        if dry_run:
            d = ex.dry_run(cfg, conn, pv, only, max_records)
            t = Table("Item", "Value", show_header=False)
            t.add_row("Model / version", f"{d['model']}  ({d['version']})")
            t.add_row("Records to extract", f"{d['records']:,}  (already done: {d['already_done']:,})")
            t.add_row("API calls", f"{d['calls']:,}")
            t.add_row("Est. input tokens", f"{d['input_tokens']:,}")
            t.add_row("Est. output tokens", f"{d['output_tokens']:,}")
            if d["cost_usd"] is None:
                t.add_row("Est. cost", "unknown (set price_per_mtok_* in config.yaml)")
            else:
                t.add_row("Est. cost", f"~${d['cost_usd']:.2f}  (${d['price_in']:g} in / ${d['price_out']:g} out per M; {d['price_source']})")
            console.print(t)
            console.print("Estimates are rough (about 3 chars/token); escalation adds a little for low-confidence items.")
            return
        cfg_key = cfg["llm"]["api_key_env"]
        if not os.environ.get(cfg_key):
            console.print(f"[red]{cfg_key} is not set in .env.[/red]")
            raise typer.Exit(1)
        stats = ex.run_extract(cfg, conn, pv, only, max_records)
        console.print(stats)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except ex.LLMAuthError as exc:
        console.print(f"[red]LLM call rejected: {exc}\nCheck OPENROUTER_API_KEY and the model slugs in config.yaml.[/red]")
        raise typer.Exit(1)
    except ex.LLMCreditError as exc:
        console.print(f"[red]{exc}\nRecords already extracted are kept; only pending ones are retried.[/red]")
        raise typer.Exit(1)


@app.command()
def export(
    format: str = typer.Option("csv", "--format", help="csv|jsonl|parquet"),
    relevant_only: bool = typer.Option(False, "--relevant-only"),
    out: Optional[Path] = typer.Option(None, help="Output path"),
) -> None:
    """Export records. CSV is UTF-8 with BOM so Hindi opens correctly in Excel."""
    import pandas as pd

    conn = db.connect()
    cols = ("id, source, app, container, container_title, parent_id, url, author_hash, title, text, rating, "
            "score, created_at, app_version, lang, query, matched_keywords, is_relevant")
    where = " WHERE is_relevant=1" if relevant_only else ""
    df = pd.read_sql_query(f"SELECT {cols} FROM records{where} ORDER BY created_at DESC", conn)
    path = out or ROOT / "data" / f"export.{format}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    elif format == "jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    elif format == "parquet":
        df.to_parquet(path, index=False)
    else:
        console.print("[red]format must be csv, jsonl or parquet[/red]")
        raise typer.Exit(2)
    console.print(f"Wrote {len(df)} rows to {path}")


if __name__ == "__main__":
    app()
