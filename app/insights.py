"""Opportunity analysis behind the "Opportunity summary" page.

Uses the LLM extraction when the snapshot has it. Before extraction it falls back to
transparent keyword rules over keyword-relevant text, and every number is labelled
provisional. Pure pandas, no Streamlit, so it is testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


def to_dt(s: pd.Series) -> pd.Series:
    """Parse mixed ISO timestamps (with and without fractional seconds) to UTC."""
    return pd.to_datetime(s, errors="coerce", utc=True, format="ISO8601")


def _rx(*parts: str) -> re.Pattern[str]:
    return re.compile("|".join(parts), re.I)


SIP_CONTEXT = _rx(r"\bsips?\b", r"mutual\s*funds?", r"\bmf\b", r"systematic", r"mandate", r"auto\s*-?\s*pay",
                  r"auto\s*-?\s*debit", r"instal+ments?", r"\bnach\b", r"monthly\s+invest")

THEME_RULES: dict[str, re.Pattern[str]] = {
    "involuntary_payment": _rx(r"auto\s*-?\s*pay", r"auto\s*-?\s*debit", r"mandate", r"\bnach\b",
                               r"(payment|debit|transaction|instal+ment|sip)\s+(has\s+)?(failed|failure|bounced?)",
                               r"insufficient\s+(balance|funds)", r"not\s+(been\s+)?deducted", r"deducted\s+twice",
                               r"pais[ae]\s+kat", r"kat\s+gay[ae]", r"upi\s+autopay"),
    "payday_mismatch": _rx(r"sip\s+date", r"debit\s+date", r"change\s+(the\s+)?date", r"date\s+change",
                           r"salary\s+(date|day|credit|comes)", r"before\s+salary", r"month\s+end", r"payday"),
    "bank_or_job_change": _rx(r"(change|changed|new|another|close[d]?)\s+(my\s+)?bank", r"bank\s+(account\s+)?change",
                              r"job\s+(loss|change|switch)", r"lost\s+(my\s+)?job", r"laid\s+off", r"layoff",
                              r"unemploy", r"naukri\s+(chali|gayi|chhut)"),
    "pause_never_resumed": _rx(r"\bpaus", r"\bskip", r"\bresum", r"\brestart", r"ruk\s+gay", r"band\s+kar\s+ke\s+phir"),
    "expectation_mismatch": _rx(r"negative\s+return", r"\blos(s|ses|ing)\b", r"in\s+(the\s+)?red", r"not\s+growing",
                                r"(low|poor|less|no|bad)\s+returns?", r"returns?\s+(are|is)\s+(low|negative|bad|poor)",
                                r"\bminus\b", r"nuksa+n", r"ghata", r"loss\s+(show|dikha)"),
    "cant_read_performance": _rx(r"(don'?t|do\s+not|didn'?t|can'?t|cannot)\s+understand", r"confus", r"\bxirr\b",
                                 r"absolute\s+return", r"how\s+(is|are)\s+(the\s+)?returns?\s+calculated",
                                 r"\bnav\b", r"statement", r"samajh\s+(nahi|nhi|na)", r"clueless", r"\bdoubts?\b"),
    "market_fall_anxiety": _rx(r"market\s+(is\s+)?(fall|falling|crash|down|correction|dip)", r"\bcrash", r"\bpanic",
                               r"\bscared", r"\bfear", r"\bworried", r"\btension", r"\bdar\s+lag", r"correction"),
    "cash_need_or_life_event": _rx(r"emergency", r"medical", r"hospital", r"wedding", r"marriage", r"(college|school)\s+fees",
                                   r"need(ed)?\s+(the\s+)?(money|cash|funds)", r"\bemi\b", r"\bloan\b",
                                   r"pais[ae]\s+ki\s+(zarurat|jarurat)", r"urgent(ly)?\s+need"),
    "overcommitment": _rx(r"too\s+many\s+(sips|funds)", r"(multiple|many)\s+sips", r"(can'?t|cannot|unable\s+to)\s+afford",
                          r"(reduce|decrease|lower)\s+(the\s+|my\s+)?(sip|amount|instal+ment)", r"step\s*-?\s*up",
                          r"high\s+amount", r"\bafford"),
    "app_ux_friction": _rx(r"\bbug", r"glitch", r"not\s+working", r"\berror", r"\bhang", r"\bstuck", r"no\s+option",
                           r"option\s+(is\s+)?not", r"(can'?t|cannot|unable\s+to)\s+(find|cancel|modify|redeem|withdraw|login|stop)",
                           r"not\s+user\s+friendly", r"\blogin", r"\botp\b", r"\bkyc\b", r"\bslow\b", r"\bupdate"),
    "trust_or_support": _rx(r"customer\s+(care|support|service)", r"\bsupport\b", r"no\s+response", r"fraud", r"scam",
                            r"complain", r"cheat", r"penalty", r"(hidden|extra)\s+charges?", r"\btrust", r"refund",
                            r"\bworst\b", r"pathetic"),
    "positive_continuation": _rx(r"patien", r"long\s*-?\s*term", r"keep\s+investing", r"stay(ed)?\s+invested",
                                 r"compound", r"disciplin", r"consisten", r"never\s+stop", r"don'?t\s+stop",
                                 r"(still|will)\s+continu"),
}

ACTION_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("stopped", _rx(r"\bstop", r"\bcancel", r"discontinu", r"band\s+kar", r"close\s+(my\s+|the\s+)?sip",
                    r"\bredeem", r"redemption", r"\bwithdraw", r"\bexit", r"nikal")),
    ("paused", _rx(r"\bpaus", r"\bskip")),
    ("reduced", _rx(r"(reduce|decrease|lower)\w*\s+(the\s+|my\s+)?(sip|amount|instal+ment)")),
    ("continuing", _rx(r"\bcontinu", r"keep\s+investing", r"stay(ed)?\s+invested", r"\bresum", r"\brestart",
                       r"step\s*-?\s*up", r"increase\w*\s+(my\s+|the\s+)?sip")),
]
CHURN_ACTIONS = {"stopped", "paused", "reduced", "redeemed", "considering_stop"}
NEWCOMER = _rx(r"\bnew\s+(to|in|investor)", r"beginner", r"first\s+(sip|time|investment)", r"just\s+started",
               r"recently\s+started", r"started\s+(a\s+|my\s+)?(sip|investing)", r"\bnaya\b", r"\bshuru",
               r"(last|this)\s+(month|year)\s+(i\s+)?start", r"\bnewbie")
NEGATIVE = _rx(r"\bloss", r"\bworst", r"fraud", r"scam", r"upset", r"angry", r"frustrat", r"pathetic", r"\bpoor\b",
               r"useless", r"waste", r"\bbad\b", r"disappoint", r"\bfailed", r"\bnot\s+working", r"nuksa+n", r"ghata")


@dataclass(frozen=True)
class Play:
    addressability: int  # 1-5: how much of this is inside the product's control
    lever: str
    metric: str
    test: str


PLAYBOOK: dict[str, Play] = {
    "involuntary_payment": Play(5, "Pre-debit balance reminders, mandate-health checks and one-tap retry after a failed instalment",
                                "Failed-instalment recovery rate; involuntary lapses per 1,000 SIPs",
                                "A/B a T-2 balance reminder plus same-day retry on failure"),
    "payday_mismatch": Play(5, "Suggest an SIP date after salary credit; make date changes free and instant",
                            "First-instalment success rate; failures in the first 3 months",
                            "Default the date picker to payday + 2 days for salaried users"),
    "bank_or_job_change": Play(4, "Guided mandate migration on bank change; offer pause instead of cancel on income loss",
                               "SIPs retained 60 days after a bank-account change",
                               "Detect mandate revocation and trigger a re-mandate flow"),
    "pause_never_resumed": Play(5, "Default pause end date, auto-resume reminder and one-tap resume",
                                "Pause-to-resume rate within 3 months",
                                "Compare auto-resume opt-out vs current manual resume"),
    "expectation_mismatch": Play(4, "Set expectations at setup: show a range of past outcomes for the chosen horizon",
                                 "Stop rate in months 0-6 after a negative-return month",
                                 "Onboarding variant with horizon-based outcome ranges"),
    "cant_read_performance": Play(4, "Plain-language returns view (invested vs current value, XIRR explained) and a first-statement walkthrough",
                                  "Returns-related support contacts; stops after first red month",
                                  "Usability test the returns screen with first-time investors"),
    "market_fall_anxiety": Play(3, "Descriptive context during market falls (how past falls played out), shown in-app, not advice",
                                "Stop and pause rate in the 2 weeks after a >5% index fall",
                                "Holdout test of contextual drawdown messaging"),
    "cash_need_or_life_event": Play(3, "Offer pause, reduce or partial withdrawal before full cancellation",
                                    "Cancel-flow save rate (pause/reduce chosen instead of stop)",
                                    "Add alternatives step to the cancel flow"),
    "overcommitment": Play(4, "Affordability check at setup; reduce or consolidate in one tap instead of stopping",
                           "Reduce-vs-stop ratio; SIP count per new investor",
                           "Nudge when total SIPs exceed a share of declared income"),
    "app_ux_friction": Play(5, "Fix the top SIP flows (modify, redeem, cancel, KYC, login) found in reviews",
                            "Task completion rate; SIP-related 1-2 star reviews",
                            "Funnel analytics on modify and redeem flows"),
    "trust_or_support": Play(4, "Faster SIP-specific support and clear communication of charges and penalties",
                             "Time to resolution; trust-related 1 star reviews",
                             "Dedicated SIP support queue pilot"),
    "positive_continuation": Play(0, "Reinforce what keeps people going (habit, progress, goals)", "Streak and goal engagement", "-"),
    "other": Play(1, "Review manually", "-", "-"),
}


# ---------------------------------------------------------------- tagging
def _rule_action(text: str) -> str:
    for name, rx in ACTION_RULES:
        if rx.search(text):
            return name
    return "none"


def _rule_severity(text: str, rating: float | None) -> float:
    if rating is not None and not pd.isna(rating):
        return float(min(5, max(1, 6 - rating)))
    return float(min(4, 2 + len(NEGATIVE.findall(text))))  # 5 is reserved for 1-star reviews


def tag(records: pd.DataFrame, has_extraction: bool) -> pd.DataFrame:
    """One row per record with themes (list), action, churn, involuntary, severity, proximity, newcomer, date."""
    df = records.copy()
    text = df["text"].fillna("").astype(str)
    out = pd.DataFrame({"id": df["id"], "source": df["source"], "text": text, "url": df.get("url"),
                        "created_at": to_dt(df["created_at"])})
    out["newcomer"] = text.apply(lambda s: bool(NEWCOMER.search(s)))
    if has_extraction:
        out["themes"] = df["primary_theme"].apply(lambda t: [t] if t else [])
        out["action"] = df["action"].fillna("unknown")
        out["involuntary"] = df["voluntary"].eq("involuntary")
        out["severity"] = pd.to_numeric(df["severity"], errors="coerce")
        out["proximity"] = pd.to_numeric(df["metric_proximity"], errors="coerce")
        out["newcomer"] = df["investor_stage"].eq("first_time") | out["newcomer"]
        out["quote"] = df["evidence_quote"].fillna(text.str[:220])
    else:
        ctx = text.str.contains(SIP_CONTEXT)
        out["themes"] = [[t for t, rx in THEME_RULES.items() if rx.search(s)] if c else [] for s, c in zip(text, ctx)]
        out["action"] = [_rule_action(s) if c else "none" for s, c in zip(text, ctx)]
        out["involuntary"] = [("involuntary_payment" in th) for th in out["themes"]]
        rating = df["rating"] if "rating" in df else pd.Series([None] * len(df), index=df.index)
        out["severity"] = [_rule_severity(s, r) for s, r in zip(text, rating)]
        out["proximity"] = float("nan")
        out["quote"] = text.str[:220]
    out["churn"] = out["action"].isin(CHURN_ACTIONS)
    out["rating"] = pd.to_numeric(df["rating"], errors="coerce") if "rating" in df else float("nan")
    if has_extraction and "lifecycle_month" in df:
        out["stage"] = df["lifecycle_month"].fillna("unknown")
    else:
        out["stage"] = [lifecycle_stage(s, d) for s, d in zip(text, out["created_at"])]
    return out


# ---------------------------------------------------------------- lifecycle proxy
STAGES = ["M0-1", "M1-3", "M3-6", "M6-12", "M12+", "unknown"]
_DURATION = re.compile(r"(\d{1,2}|one|two|three|four|five|six|ek|do|teen|char|paanch|chhe)\s*"
                       r"(months?|mahin[ea]|mahine|years?|yrs?|saal|sal)\b", re.I)
_WORDNUM = {"one": 1, "ek": 1, "two": 2, "do": 2, "three": 3, "teen": 3, "four": 4, "char": 4, "five": 5,
            "paanch": 5, "six": 6, "chhe": 6}
_SINCE = re.compile(r"since\s+(20[12]\d)", re.I)
_JUST = re.compile(r"just\s+started|first\s+sip|new\s+to\s+(sip|mutual)|started\s+(last|this)\s+(month|week)|"
                   r"first\s+(instal+ment|debit)|abhi\s+(shuru|start)", re.I)


def _bucket(months: float) -> str:
    if months <= 1:
        return "M0-1"
    if months <= 3:
        return "M1-3"
    if months <= 6:
        return "M3-6"
    if months <= 12:
        return "M6-12"
    return "M12+"


def lifecycle_stage(text: str, created: pd.Timestamp | None = None) -> str:
    """Rough months-into-SIP from phrases like '3 months', '2 saal', 'since 2021', 'just started'."""
    if _JUST.search(text):
        return "M0-1"
    m = _DURATION.search(text)
    if m:
        n = m.group(1).lower()
        n = int(n) if n.isdigit() else _WORDNUM.get(n, 0)
        unit = m.group(2).lower()
        months = n * 12 if unit.startswith(("y", "s")) else n
        if 0 < months <= 600:
            return _bucket(months)
    m = _SINCE.search(text)
    if m and created is not None and not pd.isna(created):
        return _bucket(max(0, (created.year - int(m.group(1))) * 12))
    return "unknown"


# ---------------------------------------------------------------- deeper signals
_STOP = set("""a an the and or but if to of in on for with at by from is are was were be been am i me my we our you your
it its this that these those they them their he she his her as so do does did not no yes can could will would should
have has had just very also than then there here what which who when where why how all any some more most much many
sir pls please plz hai hain ka ki ke ko se me mein ho kya aur bhi to nahi nhi na ye yeh wo woh h k""".split())
_TOKEN = re.compile(r"[a-z][a-z']+")


def _bigrams(text: str) -> set[str]:
    toks = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 2]
    return {f"{a} {b}" for a, b in zip(toks, toks[1:]) if a != b}


def emerging_terms(tagged: pd.DataFrame, recent_days: int = 90, min_recent: int = 4, top: int = 15) -> pd.DataFrame:
    """Two-word phrases whose share of records rose most in the last `recent_days` vs before (smoothed lift)."""
    df = tagged.dropna(subset=["created_at"])
    if df.empty:
        return pd.DataFrame(columns=["phrase", "recent", "earlier", "lift"])
    cutoff = df["created_at"].max() - pd.Timedelta(days=recent_days)
    recent, earlier = df[df["created_at"] >= cutoff], df[df["created_at"] < cutoff]
    if recent.empty or earlier.empty:
        return pd.DataFrame(columns=["phrase", "recent", "earlier", "lift"])

    def counts(part: pd.DataFrame) -> pd.Series:
        return pd.Series([g for t in part["text"] for g in _bigrams(t)]).value_counts()

    rc, ec = counts(recent), counts(earlier)
    out = pd.DataFrame({"recent": rc}).join(pd.DataFrame({"earlier": ec}), how="left").fillna(0)
    out = out[out["recent"] >= min_recent]
    out["recent_rate"] = out["recent"] / len(recent)
    out["earlier_rate"] = out["earlier"] / len(earlier)
    out["lift"] = (out["recent"] + 1) / len(recent) / ((out["earlier"] + 1) / len(earlier))
    out = out[out["lift"] > 1].sort_values(["lift", "recent"], ascending=False).head(top)
    return out.reset_index(names="phrase")[["phrase", "recent", "earlier", "recent_rate", "earlier_rate", "lift"]]


def cooccurrence(tagged: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """How often two problem themes appear in the same record (compounding problems)."""
    themes = [t for t in labels if t not in ("positive_continuation", "other")]
    rows = tagged["themes"].apply(lambda th: [t for t in th if t in themes])
    rows = rows[rows.str.len() > 1]
    mat = pd.DataFrame(0, index=themes, columns=themes)
    for th in rows:
        for a in th:
            for b in th:
                if a != b:
                    mat.at[a, b] += 1
    keep = mat.sum(axis=1) > 0
    mat = mat.loc[keep, keep]
    mat.index = [labels[t] for t in mat.index]
    mat.columns = [labels[t] for t in mat.columns]
    return mat


def _version_key(v: str) -> tuple:
    return tuple(int(p) if p.isdigit() else 0 for p in str(v).split("."))


def release_impact(corpus: pd.DataFrame, min_reviews: int = 40, last: int = 15) -> pd.DataFrame:
    """Per Play Store app version: rating, 1-star share and SIP-issue share, oldest to newest."""
    allps = corpus[corpus["source"] == "playstore"]
    ps = allps[allps["app_version"].notna()].copy()
    if ps.empty:
        return pd.DataFrame()
    base = allps["rating"].mean()
    ps["one_star"] = ps["rating"] == 1
    g = ps.groupby("app_version").agg(reviews=("id", "size"), avg_rating=("rating", "mean"),
                                      one_star=("one_star", "mean"), sip_related=("is_relevant", "mean"),
                                      sip_reviews=("is_relevant", "sum"),
                                      first_seen=("created_at", "min"))
    g = g[g["reviews"] >= min_reviews].reset_index()
    g = g.iloc[sorted(range(len(g)), key=lambda i: _version_key(g.at[i, "app_version"]))].tail(last)
    g["rating_vs_avg"] = g["avg_rating"] - base
    return g.reset_index(drop=True)


def rating_drag(tagged: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """Stars each theme costs the store rating: (avg of other store reviews - theme avg) x theme share of store reviews."""
    store = tagged[tagged["rating"].notna()]
    if store.empty:
        return pd.DataFrame(columns=["theme", "label", "store_reviews", "avg_rating", "drag"])
    rows = []
    for theme, label in labels.items():
        has = store["themes"].apply(lambda th, t=theme: t in th)
        n = int(has.sum())
        if not n or theme == "other":
            continue
        avg = float(store.loc[has, "rating"].mean())
        rest = float(store.loc[~has, "rating"].mean()) if (~has).any() else avg
        rows.append({"theme": theme, "label": label, "store_reviews": n, "avg_rating": avg,
                     "drag": (rest - avg) * n / len(store)})
    return pd.DataFrame(rows).sort_values("drag", ascending=False)


def stage_profile(tagged: pd.DataFrame) -> pd.DataFrame:
    """Where in the SIP lifecycle stop/pause/reduce mentions happen (records with a known stage)."""
    known = tagged[(tagged["stage"] != "unknown") & (tagged["themes"].str.len() > 0)]
    if known.empty:
        return pd.DataFrame(columns=["stage", "records", "churn_share"])
    g = known.groupby("stage").agg(records=("id", "size"), churn_share=("churn", "mean")).reindex(STAGES[:-1]).dropna()
    return g.reset_index()


# ---------------------------------------------------------------- scoring
def opportunities(tagged: pd.DataFrame, labels: dict[str, str], handling: dict[str, float] | None = None,
                  addressability: dict[str, int] | None = None, recent_days: int = 90) -> pd.DataFrame:
    """Compare problem themes on reach, link to the continuation metric, severity and addressability.

    score = reach x metric_link x (severity / 5) x (addressability / 5) x gap, scaled so the top theme = 100.
    metric_link is the share of a theme's records that describe stopping, pausing or reducing (direct
    continuation loss) plus half credit for involuntary payment failures that did not mention an action.
    """
    handling = handling or {}
    addressability = addressability or {}
    classified = tagged[tagged["themes"].str.len() > 0]
    total = max(len(classified), 1)
    churn_total = max(int(classified["churn"].sum()), 1)
    cutoff = tagged["created_at"].max() - pd.Timedelta(days=recent_days) if tagged["created_at"].notna().any() else None
    recent = classified[classified["created_at"] >= cutoff] if cutoff is not None else classified.iloc[0:0]
    earlier = classified[classified["created_at"] < cutoff] if cutoff is not None else classified
    rows = []
    for theme, label in labels.items():
        if theme in ("positive_continuation", "other"):
            continue
        has = classified["themes"].apply(lambda th, t=theme: t in th)
        sub = classified[has]
        n = len(sub)
        play = PLAYBOOK.get(theme, PLAYBOOK["other"])
        addr = addressability.get(theme, play.addressability)
        churn_share = float(sub["churn"].mean()) if n else 0.0
        invol = float(sub["involuntary"].mean()) if n else 0.0
        link = churn_share + 0.5 * float((sub["involuntary"] & ~sub["churn"]).mean()) if n else 0.0
        if sub["proximity"].notna().any():
            link = max(link, (float(sub["proximity"].mean()) - 1) / 4)
        sev = float(sub["severity"].mean()) if n else float("nan")
        h = handling.get(theme)
        gap = (6 - float(h)) / 5 if h is not None else 1.0
        r_share = recent["themes"].apply(lambda th, t=theme: t in th).mean() if len(recent) else float("nan")
        e_share = earlier["themes"].apply(lambda th, t=theme: t in th).mean() if len(earlier) else float("nan")
        rows.append({
            "theme": theme, "label": label, "records": n, "reach": n / total,
            "churn_records": int(sub["churn"].sum()), "churn_voice": sub["churn"].sum() / churn_total,
            "churn_share": churn_share, "involuntary_share": invol, "metric_link": min(link, 1.0),
            "severity": sev, "newcomer_share": float(sub["newcomer"].mean()) if n else 0.0,
            "momentum_pp": (r_share - e_share) * 100 if pd.notna(r_share) and pd.notna(e_share) else float("nan"),
            "addressability": addr, "gap": gap,
            "raw": (n / total) * min(link, 1.0) * ((sev if n else 0) / 5) * (addr / 5) * gap,
            "lever": play.lever, "metric": play.metric, "test": play.test,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    top = out["raw"].max()
    out["score"] = (out["raw"] / top * 100).round(0) if top > 0 else 0.0
    out = out.sort_values(["score", "records"], ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    out["tier"] = pd.cut(out["score"], [-1, 33, 66, 101], labels=["Monitor", "Investigate", "Prioritise"]).astype(str)
    return out


def scenario(opp: pd.DataFrame, starters_per_month: int, lapse_rate: float, fix_effect: float) -> pd.DataFrame:
    """Illustrative sizing: SIPs kept per year if each theme's share of lapses were reduced by `fix_effect`.

    Lapses are allocated to themes in proportion to their share of churn mentions (normalised,
    because one record can carry several themes in keyword mode).
    """
    if opp.empty:
        return opp.assign(lapse_share=[], sips_saved=[])
    alloc = opp["churn_voice"] / opp["churn_voice"].sum() if opp["churn_voice"].sum() else opp["reach"] / opp["reach"].sum()
    lapses = starters_per_month * 12 * lapse_rate
    eff = fix_effect * opp["addressability"] / 5
    return opp.assign(lapse_share=alloc, sips_saved=(lapses * alloc * eff).round(0))


def evidence(tagged: pd.DataFrame, theme: str, n: int = 2, exclude: set[str] | None = None) -> pd.DataFrame:
    """Most severe, readable quotes for a theme (churn mentions first), skipping ids in `exclude`."""
    sub = tagged[tagged["themes"].apply(lambda th: theme in th) & ~tagged["id"].isin(exclude or set())]
    sub = sub[sub["quote"].str.len().between(40, 400)]
    store = sub["source"].isin(["playstore", "appstore"])
    return sub.assign(_store=store).sort_values(["churn", "_store", "severity"], ascending=False).head(n).drop(columns="_store")


def by_source(tagged: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """Theme share within each source (store reviews vs community posts often differ)."""
    ex = tagged[tagged["themes"].str.len() > 0].explode("themes")
    ex = ex[~ex["themes"].isin(["positive_continuation", "other"])]
    if ex.empty:
        return pd.DataFrame()
    tab = pd.crosstab(ex["themes"], ex["source"], normalize="columns")
    tab.index = [labels.get(i, i) for i in tab.index]
    return tab


def retention_drivers(tagged: pd.DataFrame) -> dict[str, float]:
    """How often people describe reasons they kept going, and how often they say they continue."""
    classified = tagged[tagged["themes"].str.len() > 0]
    if classified.empty:
        return {"positive_share": 0.0, "continuing_share": 0.0, "churn_share": 0.0}
    return {"positive_share": float(classified["themes"].apply(lambda t: "positive_continuation" in t).mean()),
            "continuing_share": float(classified["action"].isin({"continuing", "restarted"}).mean()),
            "churn_share": float(classified["churn"].mean())}
