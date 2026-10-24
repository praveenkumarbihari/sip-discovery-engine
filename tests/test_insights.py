import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import insights  # noqa: E402

LABELS = {"involuntary_payment": "Payment failed", "market_fall_anxiety": "Market fall",
          "app_ux_friction": "App friction", "positive_continuation": "Kept going"}


def _records(texts, ratings=None):
    n = len(texts)
    return pd.DataFrame({"id": [str(i) for i in range(n)], "source": ["playstore"] * n, "url": ["u"] * n,
                         "text": texts, "rating": ratings or [None] * n,
                         "created_at": pd.date_range("2025-05-01", periods=n, freq="30D", tz="UTC").astype(str)})


def test_keyword_tagging():
    t = insights.tag(_records([
        "My SIP autopay mandate failed again so I stopped the SIP",
        "Market crash scared me, I paused my sip",
        "Patience and long term SIP investing works, I will continue",
        "Great app for stocks",
    ]), has_extraction=False).set_index("id")
    assert "involuntary_payment" in t.at["0", "themes"] and t.at["0", "action"] == "stopped" and t.at["0", "churn"]
    assert "market_fall_anxiety" in t.at["1", "themes"] and t.at["1", "action"] == "paused"
    assert "positive_continuation" in t.at["2", "themes"] and not t.at["2", "churn"]
    assert t.at["3", "themes"] == []  # no SIP context


def test_rating_drives_severity():
    t = insights.tag(_records(["sip mandate failed", "sip mandate failed"], [1, 5]), False)
    assert list(t["severity"]) == [5.0, 1.0]


def test_opportunity_ranking_and_scenario():
    t = insights.tag(_records([
        "SIP mandate failed, stopped my SIP",
        "SIP autopay debit failed, cancelled the sip",
        "SIP autopay mandate not working",
        "market crash, worried about my sip",
    ], [1, 1, 2, None]), False)
    opp = insights.opportunities(t, LABELS)
    assert opp.iloc[0]["theme"] == "involuntary_payment" and opp.iloc[0]["score"] == 100
    assert "positive_continuation" not in set(opp["theme"])
    sc = insights.scenario(opp, starters_per_month=1000, lapse_rate=0.5, fix_effect=0.2)
    lapses = 1000 * 12 * 0.5
    assert sc["lapse_share"].sum() == pytest.approx(1.0)
    assert sc["sips_saved"].sum() <= lapses * 0.2 + 1


@pytest.mark.parametrize("text,stage", [
    ("I just started my first sip", "M0-1"),
    ("SIP chal raha hai 2 mahine se, loss dikha raha", "M1-3"),
    ("been doing this sip for 8 months", "M6-12"),
    ("investing for 3 years now", "M12+"),
    ("my sip failed", "unknown"),
])
def test_lifecycle_stage(text, stage):
    assert insights.lifecycle_stage(text) == stage


def test_release_impact_orders_versions_and_rating_drag():
    corpus = pd.DataFrame({"id": [str(i) for i in range(6)], "source": ["playstore"] * 6,
                           "app_version": ["18.9.1", "18.9.1", "18.10.0", "18.10.0", "18.10.0", None],
                           "rating": [5, 4, 1, 1, 2, 3], "is_relevant": [False, True, True, False, False, False],
                           "created_at": pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")})
    rel = insights.release_impact(corpus, min_reviews=2)
    assert rel["app_version"].tolist() == ["18.9.1", "18.10.0"]  # numeric, not string, order
    assert rel.iloc[1]["one_star"] == pytest.approx(2 / 3)
    t = insights.tag(_records(["sip support worst, no response", "sip mandate failed", "sip going well"], [1, 2, 5]), False)
    drag = insights.rating_drag(t, LABELS | {"trust_or_support": "Support"}).set_index("theme")
    assert drag.at["trust_or_support", "drag"] > 0


def test_extraction_mode_uses_fields():
    r = _records(["x", "y"]).assign(primary_theme=["involuntary_payment", "app_ux_friction"], action=["stopped", "none"],
                                    voluntary=["involuntary", "voluntary"], severity=[4, 2], metric_proximity=[5, 1],
                                    investor_stage=["first_time", "unknown"], evidence_quote=["q1", "q2"])
    t = insights.tag(r, has_extraction=True)
    assert t["themes"].tolist() == [["involuntary_payment"], ["app_ux_friction"]]
    assert t["churn"].tolist() == [True, False] and t["newcomer"].tolist() == [True, False]
    opp = insights.opportunities(t, LABELS).set_index("theme")
    assert opp.at["involuntary_payment", "metric_link"] == pytest.approx(1.0)
