import json
import sqlite3
from types import SimpleNamespace

from sipscout import extract as ex

THEMES = {"involuntary_payment", "other", "positive_continuation"}


def good(**kw):
    base = dict(is_relevant=True, relevance_reason="r", investor_stage="unknown", lifecycle_month="unknown",
                action="stopped", voluntary="involuntary", primary_theme="involuntary_payment",
                secondary_themes=[], trigger="t", severity=4, metric_proximity=5, sentiment="negative",
                evidence_quote="mandate expired", confidence=0.9)
    base.update(kw)
    return base


def test_quote_validation_is_case_and_whitespace_insensitive():
    assert ex.quote_ok("Mandate   EXPIRED", "my  mandate expired\nyesterday")
    assert not ex.quote_ok("mandate cancelled", "my mandate expired")
    assert not ex.quote_ok("", "anything")


def test_validate_item_accepts_and_rejects():
    text = "My SIP mandate expired and nobody told me"
    out, err = ex.validate_item(good(id="r0"), text, THEMES)
    assert out and not err and out["severity"] == 4
    assert ex.validate_item(good(evidence_quote="made up quote"), text, THEMES)[0] is None
    assert ex.validate_item(good(primary_theme="nope"), text, THEMES)[0] is None
    assert ex.validate_item(good(severity=9), text, THEMES)[0] is None
    assert ex.validate_item(good(action="exploded"), text, THEMES)[0] is None


def test_unknown_secondary_themes_dropped():
    out, _ = ex.validate_item(good(secondary_themes=["other", "bogus", "involuntary_payment"]),
                              "mandate expired", THEMES)
    assert out["secondary_themes"] == ["other"]


def test_tool_schema_enforces_theme_enum():
    s = ex.tool_schema(["a", "b"])["function"]["parameters"]["properties"]["extractions"]["items"]
    assert s["properties"]["primary_theme"]["enum"] == ["a", "b"] and "id" in s["required"]


class FakeClient:
    """Returns canned tool-call responses; records what it was asked."""

    def __init__(self, items):
        self.items = items
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        call = SimpleNamespace(function=SimpleNamespace(arguments=json.dumps({"extractions": self.items})))
        msg = SimpleNamespace(tool_calls=[call], content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))


def _rows(texts):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("create table records (id text, title text, text text)")
    for i, t in enumerate(texts):
        conn.execute("insert into records values (?,?,?)", (f"id{i}", None, t))
    return conn.execute("select * from records").fetchall()


def test_call_batch_maps_ids_and_flags_bad_quote():
    rows = _rows(["my mandate expired", "I love chai"])
    client = FakeClient([good(id="r0"), good(id="r1", evidence_quote="not in text")])
    res = ex.call_batch(client, "m", "sys", {}, rows, THEMES)
    assert set(res.ok) == {"id0"} and "substring" in res.errors["id1"]
    assert (res.input_tokens, res.output_tokens) == (100, 50)


def test_call_batch_missing_item_reported():
    rows = _rows(["my mandate expired", "mandate expired again"])
    res = ex.call_batch(FakeClient([good(id="r0")]), "m", "sys", {}, rows, THEMES)
    assert res.errors["id1"] == "missing from response"
