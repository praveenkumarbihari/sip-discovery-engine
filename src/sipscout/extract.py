"""Stage 2: structured extraction of keyword-relevant records through an LLM (OpenRouter)."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
import openai
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from .config import ROOT
from .llm import get_client
from .models import utcnow

console = Console()
MAX_CHARS = 1500  # per-record text sent to the model

Stage = Literal["first_time", "experienced", "unknown"]
Month = Literal["M0-1", "M1-3", "M3-6", "M6-12", "M12+", "unknown"]
Action = Literal["started", "continuing", "paused", "reduced", "stopped", "restarted", "redeemed",
                 "considering_stop", "none", "unknown"]
Voluntary = Literal["voluntary", "involuntary", "mixed", "unknown"]
Outside = Literal["asked_friend_family", "youtube", "reddit_forum", "customer_support", "switched_app",
                  "none", "unknown"]
Segment = Literal["salaried", "first_job", "irregular_income", "student", "family_money",
                  "influencer_started", "friend_started", "multiple_sips", "unknown"]


class Extraction(BaseModel):
    """Per-record structured fields (stored as JSON in records.extraction)."""

    is_relevant: bool
    relevance_reason: str
    investor_stage: Stage
    lifecycle_month: Month
    action: Action
    voluntary: Voluntary
    primary_theme: str
    secondary_themes: list[str] = Field(default_factory=list)
    other_theme_description: str | None = None
    trigger: str | None = None
    what_they_checked: str | None = None
    confusion: str | None = None
    outside_app_behaviour: list[Outside] = Field(default_factory=list)
    segment_signals: list[Segment] = Field(default_factory=list)
    unmet_need: str | None = None
    severity: int = Field(ge=1, le=5)
    metric_proximity: int = Field(ge=1, le=5)
    sentiment: Literal["negative", "neutral", "positive"]
    evidence_quote: str = Field(max_length=200)
    confidence: float = Field(ge=0, le=1)


def norm(s: str) -> str:
    """Case- and whitespace-insensitive form for quote checks."""
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_ok(quote: str, text: str) -> bool:
    """True when the quote is a real substring of the text."""
    return bool(quote.strip()) and norm(quote) in norm(text)


def validate_item(raw: dict[str, Any], text: str, theme_ids: set[str]) -> tuple[dict | None, str | None]:
    """Schema-validate one model output, then verify theme ids and the evidence quote."""
    body = {k: v for k, v in raw.items() if k != "id"}
    try:
        ex = Extraction(**body)
    except ValidationError as exc:
        return None, "schema: " + "; ".join(f"{e['loc'][0]} {e['msg']}" for e in exc.errors())[:300]
    if ex.primary_theme not in theme_ids:
        return None, f"unknown primary_theme {ex.primary_theme!r}"
    ex.secondary_themes = [t for t in ex.secondary_themes if t in theme_ids and t != ex.primary_theme]
    if not quote_ok(ex.evidence_quote, text):
        return None, "evidence_quote is not a substring of the text"
    return ex.model_dump(), None


def tool_schema(theme_ids: list[str]) -> dict[str, Any]:
    """Function-calling schema: {extractions: [Extraction + local id]}."""
    item = Extraction.model_json_schema()
    item.pop("title", None)
    item["properties"]["primary_theme"] = {"type": "string", "enum": theme_ids}
    item["properties"]["secondary_themes"] = {"type": "array", "items": {"type": "string", "enum": theme_ids}}
    item["properties"] = {"id": {"type": "string"}, **item["properties"]}
    item["required"] = ["id", *[r for r in item.get("required", []) if r != "id"]]
    return {
        "type": "function",
        "function": {
            "name": "record_extractions",
            "description": "Return one extraction object per input record.",
            "parameters": {
                "type": "object",
                "properties": {"extractions": {"type": "array", "items": item}},
                "required": ["extractions"],
            },
        },
    }


def load_prompt(cfg: dict[str, Any], version: str) -> str:
    """Read prompts/extract_<version>.md and insert the taxonomy."""
    path = ROOT / "prompts" / f"extract_{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    themes = "\n".join(f"- `{t['id']}`: {t['label']}" for t in cfg["taxonomy"]["themes"])
    return path.read_text(encoding="utf-8").replace("{{themes}}", themes)


def record_text(row: sqlite3.Row) -> str:
    """Text sent to the model (and checked for quotes): title + body, truncated."""
    return (f"{row['title']}\n{row['text']}" if row["title"] else row["text"])[:MAX_CHARS]


_FATAL = (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)


class LLMAuthError(RuntimeError):
    """Key rejected, no access, or unknown model: retrying cannot help, so abort the run."""


class LLMCreditError(RuntimeError):
    """OpenRouter balance too low (HTTP 402): every further call would fail too, so stop the run."""


class LLMQuotaError(LLMCreditError):
    """Daily free-model request cap reached (HTTP 429 per-day): stop now and resume tomorrow."""


def _is_credit_error(exc: BaseException) -> bool:
    return getattr(exc, "status_code", None) == 402


def _is_daily_cap(exc: BaseException) -> bool:
    return getattr(exc, "status_code", None) == 429 and bool(re.search(r"per[- ]day", str(exc), re.I))


def _retry_transient(fn: Any) -> Any:
    """5 attempts with backoff for transient errors only (never auth, model, credit or daily-cap errors)."""
    return retry(retry=retry_if_exception(lambda e: not isinstance(e, _FATAL) and not _is_credit_error(e)
                                          and not _is_daily_cap(e)),
                 stop=stop_after_attempt(5), wait=wait_exponential_jitter(initial=1, max=30), reraise=True)(fn)


@dataclass
class BatchResult:
    """Outcome of one model call."""

    ok: dict[str, dict] = field(default_factory=dict)       # id -> extraction
    errors: dict[str, str] = field(default_factory=dict)    # id -> reason
    input_tokens: int = 0
    output_tokens: int = 0


def call_batch(client: Any, model: str, system: str, schema: dict, rows: list[sqlite3.Row],
               theme_ids: set[str], max_tokens: int = 8000, reasoning: bool = True) -> BatchResult:
    """One API call for a batch of records; validates every returned item."""
    res = BatchResult()
    payload = [{"id": f"r{i}", "text": record_text(r)} for i, r in enumerate(rows)]
    by_key = {p["id"]: (rows[i]["id"], p["text"]) for i, p in enumerate(payload)}
    user = "Extract these records:\n" + json.dumps(payload, ensure_ascii=False)

    def _go() -> Any:
        return client.chat.completions.create(
            model=model, temperature=0, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            tools=[schema], tool_choice={"type": "function", "function": {"name": "record_extractions"}},
            **({} if reasoning else {"extra_body": {"reasoning": {"enabled": False}}}),
        )

    try:
        resp = _retry_transient(_go)()
    except _FATAL as exc:
        raise LLMAuthError(f"{type(exc).__name__}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        if _is_credit_error(exc):
            afford = re.search(r"can only afford (\d+)", str(exc))
            detail = f" (the balance covers about {afford.group(1)} tokens, a call reserves up to {max_tokens})" if afford else ""
            raise LLMCreditError(f"OpenRouter credit exhausted{detail}. Add credit at "
                                 "https://openrouter.ai/settings/credits or switch extraction.model in config.yaml "
                                 "to a ':free' model, then run extract again.") from exc
        if _is_daily_cap(exc):
            raise LLMQuotaError("Daily limit for free OpenRouter models reached. Run extract again tomorrow; "
                                "it resumes where it stopped.") from exc
        for rid, _ in by_key.values():
            res.errors[rid] = f"api: {type(exc).__name__}: {exc}"[:300]
        return res
    if not getattr(resp, "choices", None):      # some free providers return HTTP 200 with only an error body
        err = getattr(resp, "error", None) or (getattr(resp, "model_extra", None) or {}).get("error") or {}
        text = json.dumps(err, default=str)[:300] if err else "empty response (no choices)"
        if re.search(r"per[- ]day", text, re.I):
            raise LLMQuotaError("Daily limit for free OpenRouter models reached. Run extract again tomorrow; "
                                "it resumes where it stopped.")
        for rid, _ in by_key.values():
            res.errors[rid] = f"api: {text}"
        return res
    if resp.usage:
        res.input_tokens, res.output_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
    msg = resp.choices[0].message
    try:
        if msg.tool_calls:
            args = json.loads(msg.tool_calls[0].function.arguments)
        else:  # some providers ignore tool_choice; accept raw JSON content
            args = json.loads(re.sub(r"^```(?:json)?|```$", "", (msg.content or "").strip(), flags=re.M).strip())
        items = args["extractions"] if isinstance(args, dict) else args
    except Exception as exc:  # noqa: BLE001
        for rid, _ in by_key.values():
            res.errors[rid] = f"unparseable response: {type(exc).__name__}"
        return res
    seen: set[str] = set()
    for item in items or []:
        key = item.get("id") if isinstance(item, dict) else None
        if key not in by_key or key in seen:
            continue
        seen.add(key)
        rid, text = by_key[key]
        out, err = validate_item(item, text, theme_ids)
        if out:
            res.ok[rid] = out
        else:
            res.errors[rid] = err or "invalid"
    for key, (rid, _) in by_key.items():
        if rid not in res.ok and rid not in res.errors:
            res.errors[rid] = "missing from response"
    return res


def fetch_prices(models: list[str]) -> dict[str, tuple[float, float]]:
    """USD per million tokens (input, output) from OpenRouter's public model list."""
    try:
        data = httpx.get("https://openrouter.ai/api/v1/models", timeout=20).json()["data"]
    except Exception:  # noqa: BLE001
        return {}
    return {m["id"]: (float(m["pricing"]["prompt"]) * 1e6, float(m["pricing"]["completion"]) * 1e6)
            for m in data if m["id"] in models}


def pending_rows(conn: sqlite3.Connection, version_tag: str, only_ids: set[str] | None,
                 max_records: int | None) -> tuple[list[sqlite3.Row], int]:
    """Keyword-relevant records not yet extracted at this version; also returns the already-done count."""
    done = conn.execute("SELECT COUNT(*) FROM records WHERE is_relevant=1 AND extraction_version=?",
                        (version_tag,)).fetchone()[0]
    rows = conn.execute(
        "SELECT id, title, text FROM records WHERE is_relevant=1 "
        "AND (extraction_version IS NULL OR extraction_version != ?) "
        "ORDER BY extraction_version IS NOT NULL, id", (version_tag,)).fetchall()  # never-extracted first
    if only_ids is not None:
        rows = [r for r in rows if r["id"] in only_ids]
    if max_records:
        rows = rows[:max_records]
    return rows, done


def version_tag(cfg: dict[str, Any], prompt_version: str) -> str:
    """extraction_version = prompt_version + model."""
    return f"{prompt_version}|{cfg['extraction']['model']}"


def dry_run(cfg: dict[str, Any], conn: sqlite3.Connection, prompt_version: str, only_ids: set[str] | None,
            max_records: int | None) -> dict[str, Any]:
    """Estimate calls and tokens without calling the API."""
    ecfg = cfg["extraction"]
    system = load_prompt(cfg, prompt_version)
    schema = tool_schema([t["id"] for t in cfg["taxonomy"]["themes"]])
    rows, done = pending_rows(conn, version_tag(cfg, prompt_version), only_ids, max_records)
    fixed = (len(system) + len(json.dumps(schema))) / 3.5
    calls = -(-len(rows) // int(ecfg["batch_size"])) if rows else 0
    text_tokens = sum(len(record_text(r)) for r in rows) / 3.0  # Hindi/Hinglish tokenise heavily
    in_tok = int(calls * fixed + text_tokens + len(rows) * 12)
    out_tok = int(len(rows) * 230)
    price_in, price_out = ecfg.get("price_per_mtok_input"), ecfg.get("price_per_mtok_output")
    source = "config"
    if price_in is None or price_out is None:
        found = fetch_prices([ecfg["model"]]).get(ecfg["model"])
        if found:
            (price_in, price_out), source = found, "OpenRouter list price"
    cost = (in_tok * price_in + out_tok * price_out) / 1e6 if price_in is not None else None
    return {"records": len(rows), "already_done": done, "calls": calls, "input_tokens": in_tok,
            "output_tokens": out_tok, "price_in": price_in, "price_out": price_out,
            "price_source": source if cost is not None else None, "cost_usd": cost,
            "model": ecfg["model"], "version": version_tag(cfg, prompt_version)}


def _save(conn: sqlite3.Connection, rid: str, payload: dict, version: str | None) -> None:
    conn.execute("UPDATE records SET extraction=?, extraction_version=?, extracted_at=? WHERE id=?",
                 (json.dumps(payload, ensure_ascii=False), version, utcnow().isoformat(), rid))


def _log_call(conn: sqlite3.Connection, model: str, pv: str, n: int, res: BatchResult) -> None:
    conn.execute("INSERT INTO llm_calls (at, stage, model, prompt_version, records, input_tokens, output_tokens) "
                 "VALUES (?,?,?,?,?,?,?)", (utcnow().isoformat(), "extract", model, pv, n,
                                            res.input_tokens, res.output_tokens))


def run_extract(cfg: dict[str, Any], conn: sqlite3.Connection, prompt_version: str,
                only_ids: set[str] | None = None, max_records: int | None = None) -> dict[str, int]:
    """Extract pending records: batch call, single retry of failures, escalation of low confidence."""
    ecfg = cfg["extraction"]
    theme_ids = [t["id"] for t in cfg["taxonomy"]["themes"]]
    tset = set(theme_ids)
    system, schema = load_prompt(cfg, prompt_version), tool_schema(theme_ids)
    tag = version_tag(cfg, prompt_version)
    rows, _ = pending_rows(conn, tag, only_ids, max_records)
    if not rows:
        return {"records": 0, "extracted": 0, "errors": 0, "escalated": 0}
    client = get_client(cfg)
    bs = int(ecfg["batch_size"])
    mt = int(ecfg.get("max_output_tokens", 8000))
    think = bool(ecfg.get("reasoning", True))
    batches = [rows[i:i + bs] for i in range(0, len(rows), bs)]
    by_id = {r["id"]: r for r in rows}
    stats = {"records": len(rows), "extracted": 0, "errors": 0, "escalated": 0}

    def work(batch: list[sqlite3.Row]) -> tuple[list[sqlite3.Row], BatchResult, BatchResult | None]:
        first = call_batch(client, ecfg["model"], system, schema, batch, tset, mt, think)
        retry_res = None
        if first.errors:                        # retry the failed records together, once (one call, not one per record)
            retry_res = call_batch(client, ecfg["model"], system, schema,
                                   [by_id[rid] for rid in first.errors], tset, mt, think)
        return batch, first, retry_res

    with ThreadPoolExecutor(max_workers=int(ecfg["max_concurrency"])) as pool:
        futures = [pool.submit(work, b) for b in batches]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                batch, first, retry_res = fut.result()
            except LLMCreditError:
                for f in futures:
                    f.cancel()
                conn.commit()
                console.print(f"  stopped: {stats['extracted']} records extracted and saved this run; "
                              f"{len(rows) - stats['extracted']} still pending.")
                raise
            _log_call(conn, ecfg["model"], prompt_version, len(batch), first)
            final = dict(first.ok)
            errs = dict(first.errors)
            if retry_res is not None:
                _log_call(conn, ecfg["model"], prompt_version, len(first.errors), retry_res)
                final.update(retry_res.ok)
                errs = {rid: retry_res.errors.get(rid, err) for rid, err in errs.items() if rid not in retry_res.ok}
            for rid, ex in final.items():
                ex["_runs"] = [{"model": ecfg["model"], "confidence": ex["confidence"]}]
                _save(conn, rid, ex, tag)
                stats["extracted"] += 1
            for rid, err in errs.items():
                # Keep an earlier model's valid extraction rather than replacing it with an error.
                conn.execute("UPDATE records SET extraction=?, extracted_at=? WHERE id=? AND extraction_version IS NULL",
                             (json.dumps({"extraction_error": err}), utcnow().isoformat(), rid))
                stats["errors"] += 1
            conn.commit()
            console.print(f"  batch {n}/{len(batches)} done ({stats['extracted']} ok, {stats['errors']} errors)")
            for reason, count in Counter(errs.values()).most_common(3):
                console.print(f"    {count} failed: {reason[:200]}")

    # Escalate low-confidence results to the stronger model; keep both results.
    low = [rid for rid in by_id
           if ecfg.get("escalation_model")
           and (cur := conn.execute("SELECT extraction, extraction_version FROM records WHERE id=?", (rid,)).fetchone())
           and cur["extraction_version"] == tag
           and json.loads(cur["extraction"])["confidence"] < float(ecfg["escalate_below_confidence"])]
    for i in range(0, len(low), bs):
        chunk = [by_id[r] for r in low[i:i + bs]]
        res = call_batch(client, ecfg["escalation_model"], system, schema, chunk, tset, mt, think)
        _log_call(conn, ecfg["escalation_model"], prompt_version, len(chunk), res)
        for rid, ex in res.ok.items():
            old = json.loads(conn.execute("SELECT extraction FROM records WHERE id=?", (rid,)).fetchone()[0])
            ex["_runs"] = old["_runs"] + [{"model": ecfg["escalation_model"], "confidence": ex["confidence"],
                                           "first_pass": {k: v for k, v in old.items() if k != "_runs"}}]
            _save(conn, rid, ex, tag)
            stats["escalated"] += 1
        conn.commit()
    return stats


def golden_ids(path: Path | None = None) -> set[str]:
    """Record ids in the golden-set CSV (Stage 3)."""
    import csv
    path = path or ROOT / "data" / "golden_set.csv"
    if not path.exists():
        raise FileNotFoundError("data/golden_set.csv not found. Run `sipscout golden sample` first (Stage 3).")
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return {row["id"] for row in csv.DictReader(fh)}
