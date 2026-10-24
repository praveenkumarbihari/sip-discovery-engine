"""Username hashing and PII scrubbing."""

from __future__ import annotations

import hashlib
import html
import re

# UPI handles end in a known PSP suffix; match before generic emails.
_UPI = re.compile(
    r"(?<![\w.])[\w.\-]{2,}@(?:ok(?:hdfcbank|icici|sbi|axis)|ybl|ibl|axl|paytm|apl|upi|sbi|icici|hdfcbank|axisbank|oksbi|barodampay|pnb|idfcbank|kotak|airtel|jio|fbl|ezetap|postbank|cnrb|boi|aubank|rbl)\b",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
_MENTION = re.compile(r"(?<![\w@])@[A-Za-z0-9_][A-Za-z0-9_.\-]{1,}")
_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_PHONE = re.compile(r"(?<![\d])(?:\+91[\s\-]?|91[\s\-]?|0)?[6-9]\d{4}[\s\-]?\d{5}(?![\d])")
_LONG_NUM = re.compile(r"\d{9,}")
_WS = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")


def hash_author(username: str | None, salt: str) -> str:
    """sha256(salt + username)[:12]. Raw usernames are never stored."""
    return hashlib.sha256((salt + (username or "")).encode("utf-8")).hexdigest()[:12]


def scrub_pii(text: str) -> str:
    """Replace emails, UPI IDs, @mentions, PAN, phone numbers and long digit runs with placeholders."""
    text = _UPI.sub("[UPI]", text)
    text = _EMAIL.sub("[EMAIL]", text)
    text = _MENTION.sub("[USER]", text)
    text = _PAN.sub("[PAN]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _LONG_NUM.sub("[NUM]", text)
    return text


def clean_text(text: str | None, strip_html: bool = False) -> str:
    """Unescape HTML, normalise whitespace, then scrub PII. Emojis are kept."""
    if not text:
        return ""
    text = html.unescape(text)
    if strip_html:
        text = _TAG.sub(" ", text)
        text = html.unescape(text)
    text = _WS.sub(" ", text).strip()
    return scrub_pii(text)
