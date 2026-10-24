"""SIP Continuation Discovery Engine: public, read-only explorer.

Reads only the published snapshot in app/data/. It never scrapes, never opens the
pipeline database and never sees Reddit, YouTube or Google credentials.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import guardrail
import insights

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PROMPTS = HERE.parent / "prompts"
FOOTER = "Research prototype using public posts and reviews. Not investment advice. Not affiliated with Groww."
MONTH_ORDER = ["M0-1", "M1-3", "M3-6", "M6-12", "M12+", "unknown"]
EXAMPLES = [
    "Why do people pause their SIP?",
    "What happens when an autopay mandate fails?",
    "How do people react when the market falls?",
    "What confuses people about their SIP returns?",
]
MAX_QUESTIONS = 10
PRIMARY = "#4F46E5"
TIER_COLORS = {"Prioritise": "#E5484D", "Investigate": "#F5A524", "Monitor": "#8B8D98"}
SOURCE_NAMES = {"playstore": "Play Store", "appstore": "App Store", "tradingqna": "TradingQ&A",
                "youtube_comment": "YouTube", "reddit": "Reddit", "reddit_post": "Reddit", "reddit_comment": "Reddit"}
log = logging.getLogger("ask")

st.set_page_config(page_title="SIP Continuation Discovery Engine", page_icon="📈", layout="wide")

CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.hero {background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #DB2777 100%); color: #fff;
       padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem; box-shadow: 0 8px 24px rgba(79,70,229,.18);}
.hero h1 {color: #fff; font-size: 1.9rem; margin: 0 0 .35rem 0; padding: 0;}
.hero p {color: rgba(255,255,255,.92); margin: 0; font-size: 1rem; line-height: 1.5;}
.hero .eyebrow {text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; opacity: .85; margin-bottom: .4rem;}
.pill {display: inline-block; padding: .18rem .65rem; border-radius: 999px; font-size: .75rem; font-weight: 600;
       background: rgba(255,255,255,.18); color: #fff; margin-top: .8rem; margin-right: .4rem;}
.kpi {background: #fff; border: 1px solid #E5E7EB; border-radius: 14px; padding: 1rem 1.1rem; height: 100%;
      box-shadow: 0 1px 2px rgba(16,24,40,.04);}
.kpi .label {font-size: .78rem; color: #6B7280; text-transform: uppercase; letter-spacing: .06em; font-weight: 600;}
.kpi .value {font-size: 1.7rem; font-weight: 700; color: #111827; margin-top: .2rem; line-height: 1.15;}
.kpi .value.text {font-size: 1.15rem; line-height: 1.3;}
.kpi .hint {font-size: .8rem; color: #6B7280; margin-top: .25rem;}
.section {margin: 1.8rem 0 .6rem 0;}
.section h3 {margin: 0; font-size: 1.25rem;}
.section p {margin: .2rem 0 0 0; color: #6B7280; font-size: .92rem;}
.insight {background: #F5F6FA; border-left: 4px solid #4F46E5; border-radius: 10px; padding: .75rem 1rem; margin-bottom: .55rem;
          font-size: .96rem; line-height: 1.5;}
.opp {border: 1px solid #E5E7EB; border-radius: 16px; padding: 1.1rem 1.2rem; height: 100%; background: #fff;
      box-shadow: 0 1px 3px rgba(16,24,40,.06);}
.opp .rank {font-size: .75rem; color: #6B7280; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;}
.opp h4 {margin: .25rem 0 .5rem 0; font-size: 1.08rem;}
.opp .stats {display: flex; gap: .9rem; flex-wrap: wrap; margin: .4rem 0 .7rem 0;}
.opp .stat b {display: block; font-size: 1.15rem; color: #111827;}
.opp .stat span {font-size: .72rem; color: #6B7280; text-transform: uppercase; letter-spacing: .05em;}
.opp .row {font-size: .88rem; margin: .35rem 0; color: #374151;}
.opp .row b {color: #111827;}
.opp blockquote {border-left: 3px solid #D1D5DB; margin: .5rem 0 0 0; padding: .1rem .7rem; color: #4B5563; font-size: .85rem;}
.badge {display: inline-block; padding: .12rem .55rem; border-radius: 999px; font-size: .72rem; font-weight: 700; color: #fff;}
.note {font-size: .82rem; color: #6B7280;}
[data-testid="stSidebar"] {background: #F8F9FC;}
[data-testid="stMetric"] {background: #fff; border: 1px solid #E5E7EB; border-radius: 14px; padding: .8rem 1rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load() -> dict | None:
    """Load the published snapshot (cached)."""
    if not (DATA / "meta.json").exists():
        return None
    out = {n: pd.read_parquet(DATA / f"{n}.parquet") for n in ("records", "themes", "clusters", "sources", "golden")}
    out["meta"] = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    out["records"]["created_at"] = insights.to_dt(out["records"]["created_at"])
    corpus = DATA / "corpus.parquet"
    out["corpus"] = pd.read_parquet(corpus) if corpus.exists() else out["records"].assign(is_relevant=True, matched_groups=None)
    out["corpus"]["created_at"] = insights.to_dt(out["corpus"]["created_at"])
    return out


@st.cache_data(show_spinner=False)
def tagged() -> pd.DataFrame:
    d = load()
    return insights.tag(d["records"], d["meta"]["has_extraction"])


def labels_of(d: dict) -> dict[str, str]:
    return dict(zip(d["themes"]["theme"], d["themes"]["label"]))


def handling_of(d: dict) -> dict[str, float]:
    th = d["themes"]
    return {t: 6 - g for t, g in zip(th["theme"], th["gap"]) if pd.notna(g)}


# ---------------------------------------------------------------- ui helpers
def esc(s: object) -> str:
    return html.escape(str(s))


def hero(title: str, subtitle: str, eyebrow: str = "", pills: list[str] | None = None) -> None:
    pill_html = "".join(f'<span class="pill">{esc(p)}</span>' for p in (pills or []))
    st.markdown(f'<div class="hero"><div class="eyebrow">{esc(eyebrow)}</div><h1>{esc(title)}</h1>'
                f'<p>{esc(subtitle)}</p>{pill_html}</div>', unsafe_allow_html=True)


def kpis(items: list[tuple[str, str, str]]) -> None:
    for col, (label, value, hint) in zip(st.columns(len(items)), items):
        cls = "value text" if len(value) > 12 else "value"
        col.markdown(f'<div class="kpi"><div class="label">{esc(label)}</div><div class="{cls}">{esc(value)}</div>'
                     f'<div class="hint">{esc(hint)}</div></div>', unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    cap = f"<p>{esc(caption)}</p>" if caption else ""
    st.markdown(f'<div class="section"><h3>{esc(title)}</h3>{cap}</div>', unsafe_allow_html=True)


def styled(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family="Inter, Segoe UI, sans-serif", size=13),
                      title_font=dict(size=15), legend=dict(orientation="h", y=-0.15))
    return fig


def chart(fig: go.Figure, height: int = 380) -> None:
    st.plotly_chart(styled(fig, height), width="stretch")


def pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.0%}"


def need_extraction(meta: dict) -> bool:
    """Show a notice and return True when no extractions exist yet."""
    if meta["has_extraction"]:
        return False
    st.info("Structured extraction has not been run on this snapshot yet, so LLM theme analysis is unavailable. "
            "Run `sipscout extract`, then `sipscout publish`. The **Opportunity summary** page already works "
            "from keyword rules in the meantime.", icon="ℹ️")
    return True


# ---------------------------------------------------------------- opportunity summary
def page_summary(d: dict) -> None:
    meta = d["meta"]
    labels = labels_of(d)
    tg = tagged()
    mode = "Based on LLM extraction" if meta["has_extraction"] else "Provisional: keyword rules (extraction not run yet)"
    hero("Opportunity summary",
         "Which reasons for stopping, pausing or reducing SIPs matter most for 12-month SIP continuation, "
         "how big each one is, and what Groww could do about it.",
         eyebrow="Business metric: first-time investors' SIP continuation in months 0-12",
         pills=[mode, f"Snapshot {meta['snapshot_date']}", f"Since {meta['date_window']['since']}"])

    with st.expander("Adjust scoring assumptions", icon="⚙️"):
        st.caption("Addressability (1-5) is a judgement of how much of the problem sits inside the product's control. "
                   "Change it to see how the ranking moves.")
        cols = st.columns(3)
        addr = {}
        for i, (theme, label) in enumerate((t, l) for t, l in labels.items() if t not in ("positive_continuation", "other")):
            addr[theme] = cols[i % 3].slider(label, 1, 5, insights.PLAYBOOK[theme].addressability, key=f"addr_{theme}")

    opp = insights.opportunities(tg, labels, handling_of(d), addr)
    if opp.empty or opp["records"].sum() == 0:
        st.warning("No records could be mapped to a theme yet.")
        return
    classified = int((tg["themes"].str.len() > 0).sum())
    drivers = insights.retention_drivers(tg)
    top = opp.iloc[0]

    kpis([
        ("Records analysed", f"{classified:,}", f"of {len(tg):,} keyword-relevant mapped to a theme"),
        ("Describe a stop, pause or reduce", pct(drivers["churn_share"]), "direct link to the continuation metric"),
        ("Payment or mandate issues", pct(opp.loc[opp['theme'] == 'involuntary_payment', 'reach'].sum()),
         "share of records about autopay, mandates or failed debits"),
        ("#1 opportunity", top["label"], f"score {top['score']:.0f}/100 · {top['records']} records"),
    ])

    section("Headline findings", "Generated from the numbers below; each line is traceable to the table.")
    for line in headlines(opp, drivers):
        st.markdown(f'<div class="insight">{line}</div>', unsafe_allow_html=True)

    section("Compare opportunity areas",
            "Up and to the right means a theme is both common and closely linked to SIPs being stopped. "
            "Bubble size is severity.")
    left, right = st.columns([1, 1])
    with left:
        fig = px.scatter(opp.assign(n=opp["rank"].astype(str)), x="reach", y="metric_link", size="severity",
                         color="tier", text="n", color_discrete_map=TIER_COLORS, size_max=34,
                         hover_name="label",
                         hover_data={"records": True, "score": True, "reach": ":.0%", "metric_link": ":.0%",
                                     "severity": ":.1f", "tier": False, "n": False},
                         labels={"reach": "Reach (share of analysed records)",
                                 "metric_link": "Link to continuation", "tier": ""})
        fig.update_traces(textposition="middle center", textfont=dict(color="white", size=12), marker_opacity=0.85)
        fig.add_hline(y=opp["metric_link"].median(), line_dash="dot", line_color="#CBD5E1")
        fig.add_vline(x=opp["reach"].median(), line_dash="dot", line_color="#CBD5E1")
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(title="Opportunity matrix (numbers = rank)")
        chart(fig, 470)
    with right:
        bar = opp.sort_values("rank", ascending=False)
        fig = go.Figure(go.Bar(x=bar["score"], y=[f"#{r}  {l}" for r, l in zip(bar["rank"], bar["label"])],
                               orientation="h", text=bar["score"].map("{:.0f}".format), cliponaxis=False,
                               marker_color=[TIER_COLORS[t] for t in bar["tier"]], textposition="outside"))
        fig.update_layout(title="Opportunity score (top = 100)", xaxis=dict(range=[0, 112], showgrid=False))
        chart(fig, 470)

    drag = insights.rating_drag(tg, labels)
    if not drag.empty:
        opp = opp.merge(drag[["theme", "drag"]], on="theme", how="left")
    else:
        opp = opp.assign(drag=float("nan"))
    table = opp[["rank", "label", "tier", "score", "records", "reach", "churn_share", "involuntary_share",
                 "severity", "drag", "newcomer_share", "momentum_pp", "addressability"]]
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "label": st.column_config.TextColumn("Opportunity area", width="large"),
            "tier": st.column_config.TextColumn("Tier"),
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
            "records": st.column_config.NumberColumn("Records"),
            "reach": st.column_config.NumberColumn("Reach", format="percent", help="Share of analysed records mentioning it"),
            "churn_share": st.column_config.NumberColumn("Stop/pause/reduce", format="percent",
                                                         help="Share of this theme's records that describe a stop, pause or reduce"),
            "involuntary_share": st.column_config.NumberColumn("Involuntary", format="percent"),
            "severity": st.column_config.NumberColumn("Severity", format="%.1f", help="1-5; from star rating or negative language"),
            "drag": st.column_config.NumberColumn("Rating drag", format="%+.2f★",
                                                  help="Stars this problem costs SIP-related store reviews (see Signals & trends)"),
            "newcomer_share": st.column_config.NumberColumn("New investors", format="percent"),
            "momentum_pp": st.column_config.NumberColumn("Trend (pp)", format="%+.1f",
                                                         help="Change in reach, last 90 days vs earlier, in percentage points"),
            "addressability": st.column_config.NumberColumn("Addressable", format="%d/5"),
        })
    st.markdown('<div class="note">Score = reach × link to continuation × severity/5 × addressability/5 × gap, '
                'scaled so the top theme is 100. Gap uses how well Groww handles the theme today when '
                '<code>scoring.current_handling</code> is set, otherwise 1. Tiers: Prioritise ≥ 67, Investigate 34-66, Monitor ≤ 33. '
                'In keyword mode one record can belong to several themes.</div>', unsafe_allow_html=True)

    section("Top opportunities and what to do about them",
            "Product ideas to test, the metric each should move, and representative user quotes.")
    used: set[str] = set()
    for col, (_, row) in zip(st.columns(3), opp.head(3).iterrows()):
        quotes = insights.evidence(tg, row["theme"], 2, used)
        used.update(quotes["id"])
        qhtml = "".join(f"<blockquote>“{esc(q[:230])}{'…' if len(q) > 230 else ''}”<br>"
                        f"<small>{esc(SOURCE_NAMES.get(s, s))}</small></blockquote>"
                        for q, s in zip(quotes["quote"], quotes["source"]))
        col.markdown(f"""<div class="opp">
<div class="rank">#{row['rank']} · <span class="badge" style="background:{TIER_COLORS[row['tier']]}">{esc(row['tier'])}</span></div>
<h4>{esc(row['label'])}</h4>
<div class="stats">
  <div class="stat"><b>{row['score']:.0f}</b><span>score</span></div>
  <div class="stat"><b>{row['records']}</b><span>records</span></div>
  <div class="stat"><b>{pct(row['churn_share'])}</b><span>stop/pause</span></div>
  <div class="stat"><b>{row['severity']:.1f}</b><span>severity</span></div>
</div>
<div class="row"><b>Lever:</b> {esc(row['lever'])}</div>
<div class="row"><b>Metric to move:</b> {esc(row['metric'])}</div>
<div class="row"><b>How to validate:</b> {esc(row['test'])}</div>
{qhtml}
</div>""", unsafe_allow_html=True)

    section("What-if sizing",
            "An illustrative estimate of SIPs kept per year if each problem were partly fixed. "
            "The inputs are assumptions, not Groww data; change them to match internal numbers.")
    a, b, c = st.columns(3)
    starters = a.number_input("New SIPs started per month", 1_000, 50_000_000, 100_000, step=10_000)
    lapse = b.slider("Share that lapse within 12 months", 0.05, 0.9, 0.40, 0.05, format="%.2f")
    effect = c.slider("Reduction achieved by a fix (at 5/5 addressability)", 0.05, 0.8, 0.20, 0.05, format="%.2f")
    sc = insights.scenario(opp, int(starters), lapse, effect).sort_values("sips_saved")
    total = sc["sips_saved"].sum()
    lapses = starters * 12 * lapse
    kpis([("Lapses per year (assumed)", f"{lapses:,.0f}", "starters × 12 × lapse rate"),
          ("SIPs kept per year, all fixes", f"{total:,.0f}", f"{total / lapses:.1%} of lapses avoided"),
          ("Biggest single contributor", f"{sc.iloc[-1]['sips_saved']:,.0f}", sc.iloc[-1]["label"])])
    fig = px.bar(sc, x="sips_saved", y="label", orientation="h", color="tier", color_discrete_map=TIER_COLORS,
                 text=sc["sips_saved"].map("{:,.0f}".format),
                 hover_data={"lapse_share": ":.0%", "addressability": True, "tier": False},
                 labels={"sips_saved": "SIPs kept per year", "label": "", "lapse_share": "Share of lapses", "tier": ""})
    fig.update_layout(title="SIPs kept per year by opportunity (illustrative)")
    chart(fig, 430)
    st.markdown('<div class="note">Lapses are allocated to themes in proportion to their share of stop, pause and reduce '
                'mentions, and fix effect is scaled by addressability. The sizing can rank themes differently from the '
                'opportunity score, which also weighs reach and severity. Public posts over-represent complaints, so treat '
                'this as a way to rank and size hypotheses, not a forecast.</div>', unsafe_allow_html=True)

    src = insights.by_source(tg, labels)
    if not src.empty:
        section("Where each problem shows up",
                "Share of each source's theme mentions. App-store reviews and community posts surface different problems.")
        src.columns = [SOURCE_NAMES.get(c, c) for c in src.columns]
        fig = px.imshow(src, text_auto=".0%", aspect="auto", color_continuous_scale="Purples",
                        labels={"x": "", "y": "", "color": "Share"})
        fig.update_coloraxes(showscale=False)
        chart(fig, 420)

    section("What keeps people going")
    kpis([("Mention a reason to continue", pct(drivers["positive_share"]), "patience, long term, discipline, compounding"),
          ("Say they continue or restarted", pct(drivers["continuing_share"]), "of analysed records"),
          ("Say they stopped, paused or reduced", pct(drivers["churn_share"]), "of analysed records")])
    if not meta["has_extraction"]:
        st.info("These numbers come from transparent keyword rules (see `app/insights.py`). After `sipscout extract` "
                "and `sipscout publish`, this page switches to the LLM-extracted theme, action, severity and "
                "metric proximity for each record.", icon="🧪")


def headlines(opp: pd.DataFrame, drivers: dict) -> list[str]:
    lines = []
    t1 = opp.iloc[0]
    lines.append(f"<b>{esc(t1['label'])}</b> is the largest opportunity: {t1['records']} records ({pct(t1['reach'])} of analysed), "
                 f"{pct(t1['churn_share'])} of which describe stopping, pausing or reducing, at average severity {t1['severity']:.1f}/5.")
    if len(opp) > 2:
        t2, t3 = opp.iloc[1], opp.iloc[2]
        lines.append(f"Next come <b>{esc(t2['label'])}</b> (score {t2['score']:.0f}) and <b>{esc(t3['label'])}</b> "
                     f"(score {t3['score']:.0f}). Together the top three account for "
                     f"{pct(opp.head(3)['churn_voice'].sum() / max(opp['churn_voice'].sum(), 1e-9))} of stop, pause and reduce mentions.")
    controllable = opp[opp["addressability"] >= 4]
    share = controllable["churn_voice"].sum() / max(opp["churn_voice"].sum(), 1e-9)
    lines.append(f"<b>{pct(share)}</b> of stop, pause and reduce mentions fall in themes the product can largely control "
                 f"(addressability 4-5), such as payments, pause and resume, and app flows, rather than market moves.")
    mom = opp.dropna(subset=["momentum_pp"])
    mom = mom[mom["records"] >= 10]
    if len(mom):
        up = mom.sort_values("momentum_pp").iloc[-1]
        if up["momentum_pp"] > 0:
            lines.append(f"<b>{esc(up['label'])}</b> is growing fastest: its reach rose {up['momentum_pp']:+.1f} percentage "
                         f"points in the last 90 days compared with earlier.")
    new = opp[opp["records"] >= 10].sort_values("newcomer_share")
    if len(new) and new.iloc[-1]["newcomer_share"] > 0:
        lines.append(f"New investors appear most often in <b>{esc(new.iloc[-1]['label'])}</b> "
                     f"({pct(new.iloc[-1]['newcomer_share'])} of its records mention being new or just started).")
    lines.append(f"Retention side: {pct(drivers['positive_share'])} of analysed records give a reason to keep going "
                 f"versus {pct(drivers['churn_share'])} describing a stop, pause or reduce.")
    return lines


# ---------------------------------------------------------------- overview
def page_overview(d: dict) -> None:
    meta, c = d["meta"], d["meta"]["counts"]
    hero("SIP Continuation Discovery Engine", meta["research_question"], eyebrow="Overview",
         pills=[f"Snapshot {meta['snapshot_date']}", f"Since {meta['date_window']['since']}"])
    kpis([("Records stored", f"{c['stored']:,}", "public reviews, posts and comments"),
          ("Keyword-relevant", f"{c['keyword_relevant']:,}", pct(c["keyword_relevant"] / max(c["stored"], 1)) + " of stored"),
          ("Extracted", f"{c['extracted']:,}", "structured by the LLM"),
          ("AI-relevant", f"{c['ai_relevant']:,}", "confirmed on-topic by the LLM")])
    left, right = st.columns([3, 2])
    with left:
        section("From collection to analysis")
        fig = go.Figure(go.Funnel(y=["Stored", "Keyword-relevant", "Extracted", "AI-relevant"],
                                  x=[c["stored"], c["keyword_relevant"], c["extracted"], c["ai_relevant"]],
                                  textinfo="value+percent initial", marker_color=[PRIMARY, "#7C3AED", "#A855F7", "#DB2777"]))
        chart(fig, 340)
    with right:
        section("Records by source")
        s = d["sources"].assign(name=lambda x: x["source"].map(lambda v: SOURCE_NAMES.get(v, v)))
        fig = px.bar(s.melt(id_vars="name", value_vars=["stored", "keyword_relevant"]), x="value", y="name",
                     color="variable", orientation="h", barmode="group",
                     color_discrete_map={"stored": "#C7D2FE", "keyword_relevant": PRIMARY},
                     labels={"value": "Records", "name": "", "variable": ""})
        chart(fig, 340)
    section("Sources")
    s = d["sources"].assign(source=lambda x: x["source"].map(lambda v: SOURCE_NAMES.get(v, v)))
    st.dataframe(s.rename(columns={"stored": "Stored", "keyword_relevant": "Keyword-relevant", "ai_relevant": "AI-relevant",
                                   "extracted": "Extracted", "containers": "Apps / containers", "first": "First",
                                   "last": "Last", "source": "Source"}),
                 hide_index=True, width="stretch")
    st.markdown('<div class="note"><b>Stored</b> is everything collected; <b>Keyword-relevant</b> is what gets analysed '
                '(SIP, mutual fund or mandate terms plus a stop, pause, payment or experience word). Most Play Store '
                'reviews are about stocks or the app in general. Switch the Evidence explorer to <b>All collected</b> to '
                'read the rest and see why each was left out.</div>', unsafe_allow_html=True)
    rec = d["records"].dropna(subset=["created_at"])
    if len(rec):
        section("Keyword-relevant volume over time")
        m = rec.assign(month=rec["created_at"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp(),
                       src=rec["source"].map(lambda v: SOURCE_NAMES.get(v, v)))
        m = m.groupby(["month", "src"]).size().reset_index(name="n")
        chart(px.bar(m, x="month", y="n", color="src", labels={"month": "", "n": "Records", "src": ""}), 320)
    need_extraction(meta)


# ---------------------------------------------------------------- themes
def page_themes(d: dict) -> None:
    meta, th, rec = d["meta"], d["themes"], d["records"]
    hero("Themes", "LLM-extracted reasons behind continuing, pausing, reducing or stopping SIPs.", eyebrow="Explore")
    if need_extraction(meta):
        return
    st.markdown("**Opportunity score** = frequency × severity × metric proximity × gap, where "
                "gap = 6 − how well Groww handles the theme today (1–5).")
    if not meta["current_handling_set"]:
        st.warning("`scoring.current_handling` is not filled in, so the gap and opportunity score are blank. "
                   "Frequency, severity and proximity are real.")
    table = th[th["count"] > 0].sort_values(["opportunity", "count"], ascending=False)
    st.dataframe(table.rename(columns={"label": "Theme", "count": "Records", "frequency": "Share", "severity": "Severity",
                                       "metric_proximity": "Proximity", "share_involuntary": "Involuntary", "gap": "Gap",
                                       "opportunity": "Opportunity"}).drop(columns=["theme"]),
                 hide_index=True, width="stretch",
                 column_config={"Share": st.column_config.NumberColumn(format="percent"),
                                "Involuntary": st.column_config.NumberColumn(format="percent"),
                                "Severity": st.column_config.NumberColumn(format="%.1f"),
                                "Proximity": st.column_config.NumberColumn(format="%.1f"),
                                "Opportunity": st.column_config.NumberColumn(format="%.2f")})
    left, right = st.columns(2)
    with left:
        fig = px.bar(table.sort_values("frequency"), x="frequency", y="label", orientation="h",
                     title="Share of AI-relevant records by theme", labels={"frequency": "Share", "label": ""},
                     color_discrete_sequence=[PRIMARY])
        fig.update_xaxes(tickformat=".0%")
        chart(fig)
    with right:
        vol = rec.merge(th[["theme", "label"]], left_on="primary_theme", right_on="theme")
        vol = vol.groupby(["label", "voluntary"]).size().reset_index(name="n")
        chart(px.bar(vol, x="n", y="label", color="voluntary", orientation="h", barmode="stack",
                     title="Voluntary vs involuntary, by theme", labels={"n": "Records", "label": ""}))
    heat = rec.merge(th[["theme", "label"]], left_on="primary_theme", right_on="theme")
    heat = heat.pivot_table(index="label", columns="lifecycle_month", values="id", aggfunc="count", fill_value=0)
    heat = heat.reindex(columns=[m for m in MONTH_ORDER if m in heat.columns])
    chart(px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale="Purples",
                    title="Theme by lifecycle month", labels={"x": "Lifecycle month", "y": "", "color": "Records"}), 420)
    seg = rec.explode("segment_signals")
    seg = seg[seg["segment_signals"].notna() & (seg["segment_signals"] != "unknown")]
    if len(seg):
        seg = seg.merge(th[["theme", "label"]], left_on="primary_theme", right_on="theme")
        seg = seg.groupby(["segment_signals", "label"]).size().reset_index(name="n")
        chart(px.bar(seg, x="segment_signals", y="n", color="label", title="Segment signals by theme",
                     labels={"segment_signals": "Segment signal", "n": "Records", "label": "Theme"}))
    else:
        st.caption("No segment signals extracted yet.")


# ---------------------------------------------------------------- signals
def page_signals(d: dict) -> None:
    labels = labels_of(d)
    tg = tagged()
    hero("Signals & trends",
         "Early-warning views: how releases move ratings, what is newly being talked about, which problems compound, "
         "and where in the first year people drop off.", eyebrow="Insights",
         pills=["Release watch", "Emerging phrases", "Rating drag", "Co-occurrence", "Lifecycle"])

    rel = insights.release_impact(d["corpus"])
    if not rel.empty:
        section("Release watch (Play Store)",
                "Average star rating and 1★ share for each recent app version with enough reviews. A drop after a "
                "release is worth checking against what shipped.")
        latest, worst = rel.iloc[-1], rel.sort_values("avg_rating").iloc[0]
        overall = d["corpus"].loc[d["corpus"]["source"] == "playstore", "rating"].mean()
        first = rel.iloc[0]
        kpis([("Latest version", f"v{latest['app_version']}", f"{latest['avg_rating']:.2f}★ · {latest['reviews']} reviews"),
              ("Change since v" + str(first["app_version"]), f"{latest['avg_rating'] - first['avg_rating']:+.2f}★",
               f"1★ share {first['one_star']:.0%} → {latest['one_star']:.0%}"),
              ("Lowest-rated version", f"v{worst['app_version']}", f"{worst['avg_rating']:.2f}★"),
              ("All Play Store reviews", f"{overall:.2f}★", f"{int((d['corpus']['source'] == 'playstore').sum()):,} reviews")])
        fig = go.Figure()
        fig.add_bar(x=rel["app_version"], y=rel["avg_rating"], name="Avg rating",
                    marker_color=["#E5484D" if v < 0 else "#30A46C" for v in rel["rating_vs_avg"]],
                    text=rel["avg_rating"].map("{:.2f}".format), textposition="outside", cliponaxis=False)
        fig.add_scatter(x=rel["app_version"], y=rel["one_star"], name="1★ share", yaxis="y2", mode="lines+markers",
                        line=dict(color=PRIMARY, width=3))
        fig.add_hline(y=overall, line_dash="dot", line_color="#94A3B8", annotation_text=f"overall {overall:.2f}★")
        fig.update_layout(yaxis=dict(title="Avg rating", range=[1, 5.2]),
                          yaxis2=dict(title="1★ share", overlaying="y", side="right", tickformat=".0%", showgrid=False),
                          xaxis=dict(type="category", title="App version"))
        chart(fig, 420)
        st.markdown('<div class="note">Green: above the overall Play Store average; red: below. Reviews that mention SIPs '
                    'are a small share of all reviews, so this reflects overall app health that first-time SIP investors '
                    'also experience.</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        section("Emerging phrases", "Two-word phrases growing fastest in analysed records, last 90 days vs before.")
        em = insights.emerging_terms(tg)
        if em.empty:
            st.caption("Not enough recent data.")
        else:
            em = em.sort_values("lift")
            fig = px.bar(em, x="lift", y="phrase", orientation="h", color_discrete_sequence=["#7C3AED"],
                         text=em["recent"].map(lambda v: f"{int(v)} recent"),
                         hover_data={"recent": True, "earlier": True, "lift": ":.1f"},
                         labels={"lift": "Growth (× more frequent)", "phrase": ""})
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(range=[0, em["lift"].max() * 1.45])
            chart(fig, 460)
            st.markdown('<div class="note">A burst can come from the comments on a single viral video; open the phrase in '
                        'the Evidence explorer to check.</div>', unsafe_allow_html=True)
    with right:
        section("Rating drag by problem", "Stars each problem costs SIP-related store reviews, relative to the rest.")
        drag = insights.rating_drag(tg, labels)
        drag = drag[drag["store_reviews"] >= 3]
        if drag.empty:
            st.caption("Not enough rated store reviews yet.")
        else:
            drag = drag.sort_values("drag")
            fig = px.bar(drag, x="drag", y="label", orientation="h",
                         color=drag["drag"] > 0, color_discrete_map={True: "#E5484D", False: "#30A46C"},
                         text=drag["drag"].map("{:+.2f}★".format),
                         hover_data={"store_reviews": True, "avg_rating": ":.2f"},
                         labels={"drag": "Stars lost", "label": ""})
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False)
            fig.update_xaxes(range=[min(0, drag["drag"].min() * 1.6), drag["drag"].max() * 1.45])
            chart(fig, 460)
            st.markdown('<div class="note">drag = (avg rating of other SIP reviews − avg rating with this problem) × its '
                        'share of SIP store reviews.</div>', unsafe_allow_html=True)

    section("Problem trend by month", "Monthly mentions of the five largest problem areas.")
    ex = tg[tg["themes"].str.len() > 0].explode("themes")
    ex = ex[~ex["themes"].isin(["positive_continuation", "other"])].dropna(subset=["created_at"])
    if len(ex):
        top5 = ex["themes"].value_counts().head(5).index
        ex = ex[ex["themes"].isin(top5)]
        ex = ex.assign(month=ex["created_at"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp(),
                       label=ex["themes"].map(labels))
        m = ex.groupby(["month", "label"]).size().reset_index(name="n")
        chart(px.line(m, x="month", y="n", color="label", markers=True, labels={"month": "", "n": "Records", "label": ""}), 360)

    left, right = st.container(), st.container()
    with left:
        section("Problems that travel together", "Records mentioning both problems. Compounding issues need joint fixes.")
        co = insights.cooccurrence(tg, labels)
        if co.empty:
            st.caption("No co-occurring problems yet.")
        else:
            fig = px.imshow(co, text_auto=True, aspect="auto", color_continuous_scale="Purples", labels={"color": "Records"})
            fig.update_coloraxes(showscale=False)
            fig.update_xaxes(tickangle=-35)
            chart(fig, 560)
    with right:
        section("When in the SIP journey", "Records that say how long they had been investing, bucketed by months.")
        known = tg[tg["stage"] != "unknown"]
        if known.empty:
            st.caption("No duration phrases found.")
        else:
            prof = known.groupby("stage").agg(records=("id", "size"), churn_share=("churn", "mean"))
            prof = prof.reindex([s for s in insights.STAGES if s in prof.index]).reset_index()
            fig = go.Figure()
            fig.add_bar(x=prof["stage"], y=prof["records"], name="Records", marker_color="#C7D2FE",
                        text=prof["records"], textposition="outside", cliponaxis=False)
            fig.add_scatter(x=prof["stage"], y=prof["churn_share"], name="Stop/pause/reduce share", yaxis="y2",
                            mode="lines+markers", line=dict(color="#E5484D", width=3))
            fig.update_layout(yaxis=dict(title="Records"), yaxis2=dict(overlaying="y", side="right", tickformat=".0%",
                                                                      range=[0, 1], showgrid=False))
            chart(fig, 380)
            st.markdown(f'<div class="note">{len(known):,} of {len(tg):,} analysed records state a duration '
                        f'("3 months", "2 saal", "since 2021", "just started"). Mentions of long horizons such as '
                        f'"for 10 years" can also land in M12+.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- explorer
def why_excluded(groups) -> str:
    g = set(groups if groups is not None else [])
    if not g & {"sip_core", "payment"}:
        return "Not analysed: no SIP, mutual fund or mandate terms"
    if "non_sip" in g and "sip_core" not in g:
        return "Not analysed: payment terms in an IPO or trading context"
    return "Not analysed: SIP terms but no stop, pause, payment or experience words"


def page_explorer(d: dict) -> None:
    meta, rec, corpus = d["meta"], d["records"], d["corpus"]
    labels = labels_of(d)
    hero("Evidence explorer", "Read the posts and reviews behind every number, including the ones the filter left out.",
         eyebrow="Evidence")
    tg = tagged().set_index("id")
    n_rel = len(rec)
    scope = st.segmented_control("Scope", [f"Analysed ({n_rel:,})", f"All collected ({len(corpus):,})"],
                                 default=f"Analysed ({n_rel:,})", label_visibility="collapsed")
    show_all = bool(scope and scope.startswith("All"))
    base = corpus if show_all else rec
    if show_all:
        st.caption("All collected records. Only keyword-relevant ones (SIP terms plus a stop, pause, payment or experience "
                   "word) are analysed; each excluded card says why.")
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1, 1, 1, 2])
        sources = f1.multiselect("Source", sorted(base["source"].unique()), format_func=lambda v: SOURCE_NAMES.get(v, v))
        stars = f2.multiselect("Star rating", [1, 2, 3, 4, 5], format_func=lambda v: "★" * v)
        langs = f3.multiselect("Language", sorted(base["lang"].dropna().unique()))
        query = f4.text_input("Search text", placeholder="e.g. mandate, pause, returns")
        g1, g2, g3, g4 = st.columns(4)
        themes = actions = vol = months = kw_themes = kw_actions = []
        if not show_all and meta["has_extraction"]:
            themes = g1.multiselect("Theme", list(labels), format_func=labels.get)
            actions = g2.multiselect("Action", sorted(rec["action"].dropna().unique()))
            vol = g3.multiselect("Voluntary", sorted(rec["voluntary"].dropna().unique()))
            months = g4.multiselect("Lifecycle month", [m for m in MONTH_ORDER if m in set(rec["lifecycle_month"].dropna())])
        elif not show_all:
            kw_themes = g1.multiselect("Opportunity area (keyword)", list(labels), format_func=labels.get)
            kw_actions = g2.multiselect("Action (keyword)", ["stopped", "paused", "reduced", "continuing", "none"])
        dates = base["created_at"].dropna()
        lo, hi = (dates.min().date(), dates.max().date()) if len(dates) else (None, None)
        span = g3.date_input("Date range", (lo, hi), min_value=lo, max_value=hi) if lo else None
        order = g4.selectbox("Sort by", ["Newest", "Oldest", "Lowest rating", "Highest rating", "Longest"])
    df = base
    for col, vals in (("source", sources), ("rating", stars), ("lang", langs), ("primary_theme", themes),
                      ("action", actions), ("voluntary", vol), ("lifecycle_month", months)):
        if vals:
            df = df[df[col].isin(vals)]
    if kw_themes:
        df = df[df["id"].map(lambda i: i in tg.index and bool(set(tg.at[i, "themes"]) & set(kw_themes)))]
    if kw_actions:
        df = df[df["id"].map(tg["action"]).isin(kw_actions)]
    if query:
        df = df[df["text"].str.contains(query, case=False, regex=False, na=False)]
    if span and len(span) == 2 and (span[0] > lo or span[1] < hi):
        day = df["created_at"].dt.date
        df = df[(day >= span[0]) & (day <= span[1])]
    sort = {"Newest": ("created_at", False), "Oldest": ("created_at", True), "Lowest rating": ("rating", True),
            "Highest rating": ("rating", False)}
    if order == "Longest":
        df = df.assign(_len=df["text"].str.len()).sort_values("_len", ascending=False).drop(columns="_len")
    else:
        col, asc = sort[order]
        df = df.sort_values(col, ascending=asc, na_position="last")

    per_page = 20
    pages = max(1, -(-len(df) // per_page))
    a, b, c = st.columns([3, 1, 1])
    msg = f"**{len(df):,}** records match"
    if show_all and len(df):
        msg += f" · {int(df['is_relevant'].sum()):,} analysed, {int((~df['is_relevant']).sum()):,} not analysed"
    a.markdown(msg)
    page = b.number_input("Page", 1, pages, 1, label_visibility="collapsed") if pages > 1 else 1
    c.download_button("Download CSV", df.drop(columns=[x for x in ("matched_groups", "secondary_themes", "outside_app_behaviour",
                                                                  "segment_signals") if x in df.columns])
                      .to_csv(index=False).encode("utf-8-sig"), "sip_evidence.csv", "text/csv", width="stretch")
    for _, r in df.iloc[(page - 1) * per_page: page * per_page].iterrows():
        with st.container(border=True):
            bits = [SOURCE_NAMES.get(r["source"], r["source"]), str(r["created_at"])[:10], r["lang"]]
            if pd.notna(r.get("rating")):
                bits.append("★" * int(r["rating"]))
            if pd.notna(r.get("app_version")) and r.get("app_version"):
                bits.append(f"v{r['app_version']}")
            if meta["has_extraction"] and not show_all and r["primary_theme"]:
                bits += [labels.get(r["primary_theme"], r["primary_theme"]), f"action: {r['action']}"]
            elif r["id"] in tg.index:
                kt = tg.at[r["id"], "themes"]
                if len(kt):
                    bits.append(", ".join(labels.get(t, t) for t in kt))
                if tg.at[r["id"], "action"] != "none":
                    bits.append(f"action: {tg.at[r['id'], 'action']}")
            st.caption(" · ".join(str(x) for x in bits if x))
            if show_all and not r["is_relevant"]:
                st.markdown(f'<span class="badge" style="background:#9CA3AF">{esc(why_excluded(r["matched_groups"]))}</span>',
                            unsafe_allow_html=True)
            use_quote = meta["has_extraction"] and not show_all and r.get("evidence_quote")
            st.markdown(f"> {r['evidence_quote'] if use_quote else r['text'][:280]}")
            if meta["has_extraction"] and not show_all and r.get("unmet_need"):
                st.markdown(f"**Unmet need:** {r['unmet_need']}")
            c1, c2 = st.columns([5, 1])
            with c1.expander("Full text"):
                st.write(r["text"])
            c2.link_button("Original ↗", r["url"], width="stretch")


# ---------------------------------------------------------------- ask
def _secret(name: str, default: str | None = None) -> str | None:
    """Streamlit secrets first (cloud), then environment / local .env (development)."""
    try:
        value = st.secrets.get(name)
    except Exception:  # noqa: BLE001  no secrets file
        value = None
    if not value:
        try:
            from dotenv import load_dotenv
            load_dotenv(HERE.parent / ".env")
        except ImportError:
            pass
        value = os.environ.get(name)
    value = (value or "").strip().strip("\"'")
    return value or default


def _section(name: str) -> str:
    text = (PROMPTS / "answer_v1.md").read_text(encoding="utf-8")
    return text.split(f"## {name}", 1)[1].split("\n## ", 1)[0].strip()


def _chat(client, model: str, system: str, user: str, max_tokens: int) -> str:
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=max_tokens,
                                       messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return (r.choices[0].message.content or "").strip()


def _facts(df: pd.DataFrame, total: int, labels: dict) -> dict:
    top = lambda s, n=4: {k: int(v) for k, v in s.value_counts().head(n).items()}  # noqa: E731
    seg = df.explode("segment_signals")["segment_signals"].dropna()
    return {"matching_records": len(df), "all_ai_relevant_records": total,
            "top_themes": {labels.get(k, k): v for k, v in top(df["primary_theme"]).items()},
            "voluntary_split": top(df["voluntary"]), "top_lifecycle_months": top(df["lifecycle_month"][df["lifecycle_month"] != "unknown"]),
            "top_segment_signals": top(seg[seg != "unknown"]), "top_sources": top(df["source"])}


def page_ask(d: dict) -> None:
    meta, rec, th = d["meta"], d["records"], d["themes"]
    hero("Ask the data", "Plain-language questions answered from the extracted records, with sources. "
         "Descriptions only, never investment advice.", eyebrow="Explore")
    if need_extraction(meta):
        return
    key = _secret("OPENROUTER_API_KEY")
    if not key:
        st.warning("The Ask page needs an `OPENROUTER_API_KEY` in this app's secrets. Every other page works without it.")
        return
    st.session_state.setdefault("asked", 0)
    st.session_state.setdefault("rejections", [])
    st.caption(f"I describe what users said in public posts and reviews. I don't give investment advice. "
               f"Questions used: {st.session_state.asked}/{MAX_QUESTIONS}.")

    def pick(q: str) -> None:
        st.session_state.q = q

    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        cols[i % 2].button(q, on_click=pick, args=(q,), key=f"ex{i}", width="stretch")
    question = st.text_input("Your question", key="q", placeholder="Why do people pause their SIP?")
    if not st.button("Ask", type="primary") or not question.strip():
        return
    if st.session_state.asked >= MAX_QUESTIONS:
        st.error("Question limit reached for this session.")
        return
    st.session_state.asked += 1
    if guardrail.is_advice_question(question):
        log.warning("rejected advice-seeking question")
        st.session_state.rejections.append(("question", question))
        st.info(guardrail.REFUSAL)
        return

    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    model = _secret("ANSWER_MODEL", "anthropic/claude-haiku-5.5")
    labels = dict(zip(th["theme"], th["label"]))
    options = ("Themes:\n" + "\n".join(f"- {k}: {v}" for k, v in labels.items())
               + "\nsource: " + ", ".join(sorted(rec["source"].unique()))
               + "\naction: " + ", ".join(sorted(rec["action"].dropna().unique()))
               + "\nvoluntary: " + ", ".join(sorted(rec["voluntary"].dropna().unique()))
               + "\nlifecycle_month: " + ", ".join(MONTH_ORDER))
    try:
        with st.spinner("Finding relevant records..."):
            raw = _chat(client, model, _section("MAP").replace("{{options}}", options), question, 300)
            m = re.search(r"\{.*\}", raw, re.S)
            plan = json.loads(m.group(0)) if m else {}
            if plan.get("intent") == "advice":
                st.session_state.rejections.append(("question", question))
                st.info(guardrail.REFUSAL)
                return
            if plan.get("intent") == "off_topic":
                st.info("That question isn't about why people continue, pause, reduce or stop SIPs.")
                return
            df = rec
            if plan.get("themes"):
                df = df[df["primary_theme"].isin([t for t in plan["themes"] if t in labels])]
            for col, f in (("source", "source"), ("action", "action"), ("voluntary", "voluntary"),
                           ("lifecycle_month", "lifecycle_month")):
                vals = (plan.get("filters") or {}).get(f)
                if vals:
                    df = df[df[col].isin(vals)]
            if df.empty:
                st.info("No records matched that question.")
                return
            quotes = df.sort_values(["confidence", "severity"], ascending=False).head(6)
            facts = _facts(df, len(rec), labels)
            qlist = [{"n": i, "source": r.source, "text": r.evidence_quote or r.text[:200]}
                     for i, r in enumerate(quotes.itertuples(), 1)]
        with st.spinner("Writing the answer..."):
            answer = _chat(client, model, _section("ANSWER"),
                           json.dumps({"question": question, "counts": facts, "quotes": qlist}, ensure_ascii=False), 400)
    except Exception as exc:  # noqa: BLE001
        st.error(f"The model call failed ({type(exc).__name__}). Check the OpenRouter key and credit.")
        return
    problems = guardrail.violations(answer)
    if problems:
        log.warning("rejected answer: %s", problems)
        st.session_state.rejections.append(("answer", ", ".join(problems)))
        st.info(guardrail.REFUSAL)
        return
    with st.container(border=True):
        st.markdown(answer)
        st.caption(f"Based on {facts['matching_records']:,} of {facts['all_ai_relevant_records']:,} AI-relevant records.")
    section("Sources")
    for i, r in enumerate(quotes.itertuples(), 1):
        st.markdown(f"[{i}] “{r.evidence_quote or r.text[:160]}” · {r.source} · [original]({r.url})")


# ---------------------------------------------------------------- methodology
def page_method(d: dict) -> None:
    meta = d["meta"]
    hero("Methodology", "How the data is collected, structured, scored and served.", eyebrow="About")
    st.graphviz_chart("""digraph { rankdir=LR; bgcolor="transparent";
        node [shape=box, style="rounded,filled", fillcolor="#EEF2FF", color="#C7D2FE", fontname="Helvetica"];
        Collect -> Extract -> Analyse -> Serve;
        Collect [label="1. Collect\\nPlay Store, App Store,\\nReddit, YouTube, TradingQ&A"];
        Extract [label="2. Extract\\nLLM fills a fixed schema;\\nquotes verified in code"];
        Analyse [label="3. Analyse\\nthemes, opportunity scores,\\nwhat-if sizing"];
        Serve [label="4. Serve\\nthis app"]; }""")
    tab1, tab2, tab3, tab4 = st.tabs(["Opportunity scoring", "Extraction schema", "Quality & versions", "Limitations"])
    with tab1:
        st.markdown("""
- **Reach**: share of analysed records that mention the theme.
- **Link to continuation**: share of the theme's records that describe stopping, pausing or reducing a SIP, plus half
  credit for involuntary payment failures. With extraction, the LLM's 1-5 *metric proximity* is used when higher.
- **Severity**: 6 − star rating for store reviews, otherwise negative-language intensity (keyword mode) or the LLM's 1-5 severity.
- **Addressability**: 1-5 judgement of how much is inside the product's control (adjustable on the summary page).
- **Gap**: 6 − `scoring.current_handling` when set; otherwise 1.
- **Score** = reach × link × severity/5 × addressability/5 × gap, scaled so the top theme = 100.
- **What-if sizing** allocates assumed lapses to themes by their share of stop/pause/reduce mentions.
""")
        st.caption("Keyword mode (before extraction) uses the rules in `app/insights.py`; one record can match several themes.")
    with tab2:
        st.dataframe(pd.DataFrame([
            ("is_relevant", "Person describes their own (or a close person's) SIP experience"),
            ("investor_stage / lifecycle_month", "first_time, experienced; M0-1 to M12+"),
            ("action / voluntary", "started, paused, reduced, stopped, ...; voluntary, involuntary, mixed"),
            ("primary_theme / secondary_themes", "From the fixed taxonomy"),
            ("trigger / what_they_checked / confusion", "What happened right before, what they looked at, what confused them"),
            ("outside_app_behaviour / segment_signals", "Friends, YouTube, support; salaried, student, ..."),
            ("unmet_need", "What would have kept them going, in their terms"),
            ("severity / metric_proximity", "1-5 each"),
            ("evidence_quote / confidence", "Exact substring of the text; 0-1"),
        ], columns=["Field", "Meaning"]), hide_index=True, width="stretch")
    with tab3:
        g = d["golden"]
        st.markdown("**Golden-set agreement**")
        if len(g):
            st.dataframe(g, hide_index=True, width="stretch")
        else:
            st.caption("Not yet available: the golden set has not been labelled and scored.")
        st.markdown("**Versions**")
        st.json({"prompt": meta["prompt_version"], "extraction_versions": meta["extraction_versions"],
                 "models": meta["models"], "tokens": meta["tokens"], "snapshot": meta["snapshot_date"]})
    with tab4:
        for item in meta["limitations"]:
            st.markdown(f"- {item}")


# ---------------------------------------------------------------- main
data = load()
if data is None:
    st.error("No published snapshot found. Run `sipscout publish` first.")
    st.caption(FOOTER)
    st.stop()

with st.sidebar:
    st.markdown("### 📈 SIP Discovery Engine")
    st.caption("Why first-time investors continue, pause, reduce or stop SIPs.")

def _page(fn, title: str, icon: str, path: str, default: bool = False) -> st.Page:
    def run() -> None:
        fn(data)
    run.__name__ = path
    return st.Page(run, title=title, icon=icon, url_path=path, default=default)


nav = st.navigation({
    "Evidence": [_page(page_explorer, "Evidence explorer", "🔎", "evidence", default=True)],
    "Insights": [_page(page_summary, "Opportunity summary", "🎯", "summary"),
                 _page(page_signals, "Signals & trends", "📡", "signals"),
                 _page(page_overview, "Overview", "📊", "overview")],
    "Explore": [_page(page_themes, "Themes", "🧭", "themes"),
                _page(page_ask, "Ask the data", "💬", "ask")],
    "About": [_page(page_method, "Methodology", "📐", "methodology")],
})
with st.sidebar:
    c = data["meta"]["counts"]
    st.divider()
    st.caption(f"Snapshot **{data['meta']['snapshot_date']}** · {c['stored']:,} stored · {c['keyword_relevant']:,} relevant")
    st.caption(FOOTER)
nav.run()
st.divider()
st.caption(FOOTER)
