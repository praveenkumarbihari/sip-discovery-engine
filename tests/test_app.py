import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from sipscout import db
from sipscout.models import Record, make_id
from sipscout.publish import publish
from sipscout.score import theme_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import guardrail  # noqa: E402


@pytest.mark.parametrize("text", [
    "You should switch to XYZ Bluechip Fund now",
    "I recommend continuing your SIP.",
    "Many users stopped. Act now before it's too late!",
    "HDFC Flexicap did better for them.",
    "It is a good idea to stop your SIP.",
    "Reliance shares are a safe pick.",
    "Don't miss out, invest in this scheme immediately.",
    "Pause your SIP until the market recovers.",
    "Users were confused. Stop the SIP and redeem the units.",
    "- Continue your SIPs through the dip.",
])
def test_guardrail_rejects_advice(text):
    assert not guardrail.is_allowed(text), text


@pytest.mark.parametrize("text", [
    "12 of 40 records say the autopay mandate failed [1]. Users said support was slow [2].",
    "Most users who paused mentioned a market fall; some said they could not read the statement [1].",
    "Users said they had to pause the SIP after a job loss [1]. Others chose to stop SIPs when cash was tight [2].",
    "Of 285 records, 19 said the app failed to stop the SIP they cancelled [3].",
    "Some wanted to continue investing but the mandate broke [4].",
])
def test_guardrail_allows_descriptions(text):
    assert guardrail.is_allowed(text), guardrail.violations(text)


@pytest.mark.parametrize("q", ["Should I stop my SIP?", "Which fund is best?", "What should I buy now?"])
def test_advice_questions_detected(q):
    assert guardrail.is_advice_question(q)


def test_descriptive_questions_pass():
    assert not guardrail.is_advice_question("Why do people pause their SIP?")


def _seed(conn, with_extraction):
    ext = {"is_relevant": True, "relevance_reason": "r", "investor_stage": "unknown", "lifecycle_month": "M3-6",
           "action": "stopped", "voluntary": "involuntary", "primary_theme": "involuntary_payment",
           "secondary_themes": [], "other_theme_description": None, "trigger": "t", "what_they_checked": None,
           "confusion": None, "outside_app_behaviour": ["none"], "segment_signals": ["salaried"],
           "unmet_need": "alerts", "severity": 4, "metric_proximity": 5, "sentiment": "negative",
           "evidence_quote": "mandate expired", "confidence": 0.9}
    rec = Record(id=make_id("playstore", "1"), source="playstore", native_id="1", url="http://x/1", author_hash="abc",
                 text="my sip mandate expired", created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
                 is_relevant=True, container="pkg", raw={"secret": "x"})
    db.upsert_records(conn, [rec])
    if with_extraction:
        conn.execute("UPDATE records SET extraction=?, extraction_version='v1|m'", (json.dumps(ext),))
        conn.commit()


def test_publish_without_extraction(conn, cfg, tmp_path):
    _seed(conn, False)
    meta = publish(cfg, conn, tmp_path)
    assert meta["has_extraction"] is False and meta["counts"]["keyword_relevant"] == 1
    recs = pd.read_parquet(tmp_path / "records.parquet")
    assert len(recs) == 1 and "author_hash" not in recs.columns and "raw" not in recs.columns


def test_publish_with_extraction_and_score(conn, cfg, tmp_path):
    _seed(conn, True)
    meta = publish(cfg, conn, tmp_path)
    assert meta["has_extraction"] and meta["counts"]["ai_relevant"] == 1
    th = pd.read_parquet(tmp_path / "themes.parquet")
    row = th[th["theme"] == "involuntary_payment"].iloc[0]
    assert row["count"] == 1 and row["frequency"] == 1.0 and pd.isna(row["opportunity"])  # no handling score yet


def test_opportunity_formula():
    df = pd.DataFrame({"primary_theme": ["a", "a", "b", "b"], "severity": [4, 2, 5, 5],
                       "metric_proximity": [5, 3, 4, 4], "voluntary": ["involuntary"] * 4})
    t = theme_table(df, [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}], {"a": 2, "b": 5}).set_index("theme")
    # a: freq 0.5, sev 3, prox 4, gap 4 -> 24 ; b: freq 0.5, sev 5, prox 4, gap 1 -> 10
    assert t.loc["a", "opportunity"] == pytest.approx(24.0)
    assert t.loc["b", "opportunity"] == pytest.approx(10.0)


# ---- no-PII scan over the committed snapshot and any exports
EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<!\d)(?:\+91[\s\-]?)?[6-9]\d{9}(?!\d)")
PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
MENTION = re.compile(r"(?<![\w@])@[A-Za-z0-9_]{3,}")


def _scan_texts():
    for folder in (ROOT / "app" / "data", ROOT / "data"):
        for p in folder.glob("*"):
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
                assert "author_hash" not in df.columns and "raw" not in df.columns, p
                for col in df.columns:
                    if col in ("id", "url") or df[col].dtype.kind not in "OUT":  # ids/urls are system-made, not free text
                        continue
                    for v in df[col].dropna():
                        if isinstance(v, str):
                            yield p.name, v
            elif p.suffix in (".csv", ".jsonl", ".json", ".md"):
                yield p.name, p.read_text(encoding="utf-8-sig", errors="ignore")


def test_no_pii_in_published_or_exported_data():
    for name, text in _scan_texts():
        for label, rx in (("email", EMAIL), ("phone", PHONE), ("pan", PAN), ("mention", MENTION)):
            assert not rx.search(text), f"{label} found in {name}: {rx.search(text).group(0)!r}"
