"""Stage 4: write the scrubbed snapshot the public app reads (app/data/)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ROOT
from .extract import Extraction
from .models import utcnow
from .privacy import scrub_pii
from .score import theme_table

APP_DATA = ROOT / "app" / "data"
EXTRACT_FIELDS = [f for f in Extraction.model_fields]
LIST_FIELDS = ["secondary_themes", "outside_app_behaviour", "segment_signals"]
BASE_COLS = ["id", "source", "url", "text", "created_at", "lang", "rating", "app_version"]
LIMITATIONS = [
    "App Store RSS exposes only the most recent ~500 reviews per app.",
    "YouTube collection is limited by the daily API quota.",
    "Public posts over-represent people with complaints.",
    "Groww's internal lapse data is not available, so no actual lapse rates are reported.",
    "Very short texts are dropped before analysis.",
]


def _load(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT id, source, url, text, created_at, lang, rating, app_version, is_relevant, extraction, extraction_version "
        "FROM records WHERE is_relevant=1").fetchall()
    out = []
    for r in rows:
        d = {c: r[c] for c in BASE_COLS}
        ext = json.loads(r["extraction"]) if r["extraction"] and r["extraction_version"] else None
        d["extracted"] = ext is not None
        for f in EXTRACT_FIELDS:
            v = ext.get(f) if ext else None
            d[f] = (v or []) if f in LIST_FIELDS else v
        out.append(d)
    cols = BASE_COLS + ["extracted"] + EXTRACT_FIELDS
    return pd.DataFrame(out, columns=cols)


def _corpus(conn: sqlite3.Connection) -> pd.DataFrame:
    """Every stored record (already PII-scrubbed at collection), for browsing and release/rating trends."""
    df = pd.read_sql_query(
        "SELECT id, source, url, title, text, created_at, lang, rating, app_version, matched_keywords, is_relevant "
        "FROM records", conn)
    df["text"] = [scrub_pii(f"{t} {x}".strip() if isinstance(t, str) and t.strip() else x)
                  for t, x in zip(df.pop("title"), df["text"])]
    df["matched_groups"] = [sorted(json.loads(m or "{}") or {}) for m in df.pop("matched_keywords")]
    df["is_relevant"] = df["is_relevant"].astype(bool)
    df["created_at"] = df["created_at"].astype(str)
    return df


def publish(cfg: dict[str, Any], conn: sqlite3.Connection, out_dir: Path | None = None) -> dict[str, Any]:
    """Write records/themes/clusters/sources/golden parquet files and meta.json."""
    out = Path(out_dir or APP_DATA)
    out.mkdir(parents=True, exist_ok=True)
    df = _load(conn)
    extracted = df[df["extracted"]]
    ai_rel = extracted[extracted["is_relevant"] == True]  # noqa: E712
    has_ext = len(extracted) > 0
    records = ai_rel if has_ext else df  # before extraction, publish keyword-relevant records only
    records = records.copy()
    records["created_at"] = records["created_at"].astype(str)
    records.to_parquet(out / "records.parquet", index=False)
    _corpus(conn).to_parquet(out / "corpus.parquet", index=False)

    themes = theme_table(ai_rel, cfg["taxonomy"]["themes"], cfg.get("scoring", {}).get("current_handling") or {})
    themes.to_parquet(out / "themes.parquet", index=False)

    per_source = conn.execute(
        "SELECT source, COUNT(*) stored, SUM(is_relevant) kw, MIN(created_at) first, MAX(created_at) last, "
        "COUNT(DISTINCT container) containers FROM records GROUP BY source").fetchall()
    src_rows = []
    for r in per_source:
        sub = df[df["source"] == r["source"]]
        src_rows.append({"source": r["source"], "stored": r["stored"], "keyword_relevant": int(r["kw"] or 0),
                         "ai_relevant": int((ai_rel["source"] == r["source"]).sum()),
                         "extracted": int(sub["extracted"].sum()), "containers": r["containers"],
                         "first": str(r["first"])[:10], "last": str(r["last"])[:10]})
    pd.DataFrame(src_rows, columns=["source", "stored", "keyword_relevant", "ai_relevant", "extracted",
                                    "containers", "first", "last"]).to_parquet(out / "sources.parquet", index=False)

    pd.DataFrame(columns=["cluster_id", "name", "description", "size", "overlap", "emergent", "quotes"]
                 ).astype({"cluster_id": "int64", "size": "int64"}).to_parquet(out / "clusters.parquet", index=False)
    pd.DataFrame({"field": pd.Series(dtype="str"), "n": pd.Series(dtype="int64"),
                  "agreement": pd.Series(dtype="float64"), "kappa": pd.Series(dtype="float64")}
                 ).to_parquet(out / "golden.parquet", index=False)

    stored = int(conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])
    versions = [r[0] for r in conn.execute(
        "SELECT DISTINCT extraction_version FROM records WHERE extraction_version IS NOT NULL")]
    tokens = conn.execute("SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0) FROM llm_calls").fetchone()
    meta = {
        "snapshot_date": utcnow().strftime("%Y-%m-%d"),
        "research_question": ("Why do first-time SIP investors on Groww continue, pause, reduce or stop their SIPs "
                              "in the first 12 months, and which of those reasons is the biggest, most under-served "
                              "opportunity?"),
        "has_extraction": has_ext,
        "counts": {"stored": stored, "keyword_relevant": int(len(df)), "extracted": int(len(extracted)),
                   "ai_relevant": int(len(ai_rel))},
        "date_window": cfg["date_window"],
        "prompt_version": cfg["extraction"]["prompt_version"],
        "extraction_versions": versions,
        "models": {"extraction": cfg["extraction"]["model"], "escalation": cfg["extraction"]["escalation_model"]},
        "tokens": {"input": int(tokens[0]), "output": int(tokens[1])},
        "limitations": LIMITATIONS,
        "scoring_formula": "opportunity = frequency x severity x metric_proximity x gap  (gap = 6 - current_handling)",
        "current_handling_set": bool(cfg.get("scoring", {}).get("current_handling")),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
