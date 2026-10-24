import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sipscout import db
from sipscout.filters import evaluate, match_keywords
from sipscout.models import Record, make_id
from sipscout.privacy import clean_text, hash_author, scrub_pii
from sipscout.sources.appstore import AppStoreSource, parse_entry
from sipscout.sources.playstore import PlayStoreSource, parse_review
from sipscout.sources.tradingqna import TradingQnASource, parse_post

FIX = Path(__file__).parent / "fixtures"


def test_scrub_pii():
    out = scrub_pii("mail me a.b@gmail.com or call +91 98765 43210, PAN ABCDE1234F, acct 123456789012, pay raj@okhdfcbank")
    assert "[EMAIL]" in out and "[PHONE]" in out and "[PAN]" in out and "[NUM]" in out and "[UPI]" in out
    assert "gmail" not in out and "98765" not in out and "ABCDE" not in out and "raj@" not in out


def test_scrub_at_mentions():
    out = scrub_pii("Hi @aakashnand, we will add support. cc @Arul.P")
    assert "aakashnand" not in out and "Arul" not in out and out.count("[USER]") == 2


def test_clean_text_keeps_emoji_and_unescapes():
    assert clean_text("SIP &amp; MF   \n 😡") == "SIP & MF 😡"
    assert clean_text("<p>hello <b>SIP</b></p>", strip_html=True) == "hello SIP"


def test_hash_author_deterministic_and_salted():
    assert hash_author("bob", "s1") == hash_author("bob", "s1")
    assert hash_author("bob", "s1") != hash_author("bob", "s2")
    assert len(hash_author("bob", "s1")) == 12
    assert "bob" not in hash_author("bob", "s1")


def test_make_id():
    assert make_id("playstore", "abc") == make_id("playstore", "abc")
    assert make_id("playstore", "abc") != make_id("appstore", "abc")


def test_keywords_word_boundary_and_hinglish(cfg):
    groups = cfg["filters"]["keyword_groups"]
    assert "sip_core" not in match_keywords("I was sipping chai", groups)
    m = match_keywords("Mera SIP band karna hai, paise kat gaye", groups)
    assert "sip_core" in m and "hinglish" in m
    rel_hit = evaluate("My SIP autopay mandate failed", cfg)[1]
    rel_miss = evaluate("Great app for trading stocks", cfg)[1]
    assert rel_hit and not rel_miss
    assert evaluate("Please help with this", cfg, sip_query=False)[1] is False
    assert evaluate("I want to stop this", cfg, sip_query=True)[1] is True


def test_payment_terms_count_as_sip_context_except_ipo(cfg):
    assert evaluate("Unable to change Autopay mandate from eNACH to UPI", cfg)[1] is True
    assert evaluate("IPO autopay mandate failed again", cfg)[1] is False
    assert evaluate("Groww helped me start my first sip", cfg)[1] is True


def _rec(native="1", text="my sip stopped"):
    return Record(id=make_id("playstore", native), source="playstore", native_id=native, url="u",
                  author_hash="h", text=text, created_at=datetime(2025, 5, 1, tzinfo=timezone.utc))


def test_upsert_idempotent(conn):
    assert db.upsert_records(conn, [_rec("1"), _rec("2")]) == (2, 0)
    assert db.upsert_records(conn, [_rec("1"), _rec("2")]) == (0, 2)
    assert db.count_records(conn) == 2


def test_upsert_preserves_extraction(conn):
    db.upsert_records(conn, [_rec("1")])
    conn.execute("UPDATE records SET extraction='{\"x\":1}', extraction_version='v1'")
    db.upsert_records(conn, [_rec("1", text="my sip stopped again")])
    row = conn.execute("SELECT extraction, text FROM records").fetchone()
    assert row["extraction"] == '{"x":1}' and row["text"] == "my sip stopped again"


def test_quota(conn):
    db.add_quota(conn, "youtube", 100)
    db.add_quota(conn, "youtube", 5)
    assert db.quota_used(conn, "youtube") == 105


def _src(cls, cfg):
    return cls(cfg, "salt")


def test_playstore_parser(cfg):
    item = json.loads((FIX / "playstore_review.json").read_text(encoding="utf-8"))
    item["at"] = datetime.fromisoformat(item["at"])
    rec = parse_review(_src(PlayStoreSource, cfg), "groww", "com.nextbillion.groww", "en", item)
    assert rec.source == "playstore" and rec.rating == 1 and rec.is_relevant
    assert "[PHONE]" in rec.text and "9876543210" not in rec.text
    assert "userName" not in rec.raw and "Test User" not in json.dumps(rec.raw, default=str)
    assert rec.raw["replyContent"]


def test_short_text_dropped_and_counted(cfg):
    src = _src(PlayStoreSource, cfg)
    item = {"reviewId": "x", "userName": "u", "content": "bad", "score": 1, "thumbsUpCount": 0,
            "at": datetime(2025, 6, 1)}
    assert parse_review(src, "groww", "pkg", "en", item) is None
    assert src.stats["dropped_short"] == 1


def test_appstore_parser_skips_app_entry(cfg):
    data = json.loads((FIX / "appstore_feed.json").read_text(encoding="utf-8"))
    src = _src(AppStoreSource, cfg)
    recs = [parse_entry(src, "groww", "123", e) for e in data["feed"]["entry"]]
    assert recs[0] is None
    r = recs[1]
    assert r.rating == 2 and r.title and r.app_version == "9.1"
    assert "author" not in r.raw


def test_tradingqna_quote_blocks_removed(cfg):
    from sipscout.sources.tradingqna import strip_quotes
    cooked = ('<aside class="quote no-group" data-username="Vinodh"><div class="title">Vinodh:</div>'
              '<blockquote><p>old text</p></blockquote></aside><p>My SIP stopped</p>')
    out = strip_quotes(cooked)
    assert "Vinodh" not in out and "old text" not in out and "My SIP stopped" in out


def test_tradingqna_parser_strips_html(cfg):
    data = json.loads((FIX / "tradingqna_topic.json").read_text(encoding="utf-8"))
    src = _src(TradingQnASource, cfg)
    post = data["post_stream"]["posts"][0]
    rec = parse_post(src, "https://tradingqna.com", data, post, "SIP mandate")
    assert "<" not in rec.text and "SIP mandate" in rec.text
    assert rec.is_relevant and rec.container_title == data["title"]
