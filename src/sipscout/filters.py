"""Keyword pre-filter. Flags records for Stage 2; never drops them."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=256)
def _pattern(keyword: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(keyword.lower()).replace(r"\ ", r"\s+") + r"(?!\w)")


def match_keywords(text: str, groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return {group: [matched keywords]} using case-insensitive word-boundary matching."""
    lowered = text.lower()
    found: dict[str, list[str]] = {}
    for group, words in groups.items():
        hits = [w for w in words if _pattern(w).search(lowered)]
        if hits:
            found[group] = hits
    return found


def is_relevant(matched: dict[str, list[str]], sip_query: bool = False) -> bool:
    """SIP context AND at least one of payment/action/experience/hinglish.

    SIP context is sip_core, an SIP-specific query, or payment terms (autopay, mandate, NACH)
    without IPO/trading context, since those reviews are about SIP debits even when "SIP" is unsaid.
    """
    payment_core = "payment" in matched and "non_sip" not in matched
    has_core = "sip_core" in matched or sip_query or payment_core
    has_signal = any(g in matched for g in ("payment", "action", "experience", "hinglish"))
    return has_core and has_signal


def evaluate(text: str, cfg: dict[str, Any], sip_query: bool = False) -> tuple[dict[str, list[str]], bool]:
    """Match keywords from config and decide relevance."""
    matched = match_keywords(text, cfg["filters"]["keyword_groups"])
    return matched, is_relevant(matched, sip_query)
