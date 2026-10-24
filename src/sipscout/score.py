"""Opportunity scoring: frequency x severity x metric_proximity x gap."""

from __future__ import annotations

from typing import Any

import pandas as pd

FORMULA = "opportunity = frequency x severity x metric_proximity x gap, where gap = 6 - current_handling (1-5)"
COLUMNS = ["theme", "label", "count", "frequency", "severity", "metric_proximity", "share_involuntary",
           "gap", "opportunity"]


def theme_table(df: pd.DataFrame, themes: list[dict[str, str]], handling: dict[str, float] | None) -> pd.DataFrame:
    """Per-theme stats from AI-relevant extracted records.

    `df` needs primary_theme, severity, metric_proximity, voluntary. A theme with no
    current_handling value gets a NaN gap and opportunity (the user must supply it).
    """
    handling = handling or {}
    total = len(df)
    rows = []
    for t in themes:
        sub = df[df["primary_theme"] == t["id"]] if total else df
        n = len(sub)
        h = handling.get(t["id"])
        gap = 6 - float(h) if h is not None else float("nan")
        freq = n / total if total else 0.0
        sev = float(sub["severity"].mean()) if n else float("nan")
        prox = float(sub["metric_proximity"].mean()) if n else float("nan")
        invol = float((sub["voluntary"] == "involuntary").mean()) if n else float("nan")
        rows.append({"theme": t["id"], "label": t["label"], "count": n, "frequency": freq, "severity": sev,
                     "metric_proximity": prox, "share_involuntary": invol, "gap": gap,
                     "opportunity": freq * sev * prox * gap if n else float("nan")})
    return pd.DataFrame(rows, columns=COLUMNS)
