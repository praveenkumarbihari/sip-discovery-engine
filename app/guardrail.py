"""Advice guardrail for the "Ask the data" page.

The app may only describe what users said. It must never name a fund, AMC or stock,
recommend buying, selling, continuing or stopping an investment, or use urgency.
"""

from __future__ import annotations

import re

REFUSAL = "I can only describe what users said, not give investment advice."

_AMC = (r"sbi|hdfc|icici|prudential|nippon|axis|kotak|parag\s+parikh|ppfas|uti|mirae|motilal|quant|"
        r"aditya\s+birla|sundaram|tata|franklin|dsp|invesco|canara|edelweiss|bandhan|hsbc|whiteoak|navi|"
        r"groww\s+(?:nifty|liquid|value|multicap)|zerodha|bluechip|blue\s+chip|flexicap|flexi\s+cap|smallcap|"
        r"small\s+cap|midcap|mid\s+cap|elss|nifty\s*(?:50|next|bees)|sensex|index\s+fund|etf")
_STOCKS = (r"reliance|tcs|infosys|wipro|adani|tata\s+motors|hdfc\s*bank|icici\s*bank|bajaj|zomato|paytm|"
           r"itc|ongc|sbin|lic|yes\s+bank|suzlon|nvidia|tesla|apple|bitcoin|crypto")

_FUND_NAMED = re.compile(rf"\b(?:{_AMC})\b", re.I)
_FUND_SUFFIX = re.compile(r"\b[A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*)*\s+(?:Fund|Funds|ETF|Mutual\s+Fund)\b")
_STOCK_NAMED = re.compile(rf"\b(?:{_STOCKS})\b", re.I)

_ACTIONS = r"(?:buy|buying|sell|selling|switch|switching|invest|investing|continue|continuing|stop|stopping|pause|pausing|redeem|redeeming|increase|reduce|start|starting|hold|holding|exit|exiting)"
_ADVICE = [
    rf"\byou\s+(?:should|must|ought\s+to|need\s+to|have\s+to|had\s+better|could|can\s+consider)\b",
    r"\bi\s+(?:recommend|suggest|advise|would\s+advise|would\s+suggest)\b",
    r"\bwe\s+(?:recommend|suggest|advise)\b",
    rf"\b(?:consider|try|better\s+to|best\s+to|advisable\s+to|wise\s+to|worth)\s+{_ACTIONS}\b",
    rf"\b(?:buy|sell|switch\s+to|invest\s+in|continue|stop|pause|redeem|increase|reduce|hold|exit)\s+(?:your|the|this|that|a|an|more)?\s*(?:sip|sips|fund|funds|units|investment|investments|stocks?|shares)\b",
    r"\b(?:is|it\s+is|it's)\s+(?:a\s+)?(?:good|bad|great|smart|wise)\s+(?:idea|time|move|choice)\s+to\b",
    r"\bthe\s+best\s+(?:fund|sip|option|choice|stock)\b",
    r"\bguaranteed\s+returns?\b",
]
_URGENCY = [
    r"\b(?:act|buy|invest|hurry|decide|start)\s+(?:now|today|immediately|fast|quickly)\b",
    r"\bright\s+now\b", r"\bimmediately\b", r"\burgent(?:ly)?\b", r"\bhurry\b", r"\blast\s+chance\b",
    r"\bdon'?t\s+(?:miss|wait|delay)\b", r"\bbefore\s+it'?s\s+too\s+late\b", r"\blimited\s+time\b",
    r"\bfomo\b", r"\bfear\s+of\s+missing\b", r"\brun\s+out\s+of\s+time\b", r"\bwhile\s+(?:you\s+still\s+can|stocks\s+last)\b",
]
_ADVICE_RE = [re.compile(p, re.I) for p in _ADVICE]
_URGENCY_RE = [re.compile(p, re.I) for p in _URGENCY]

# Questions asking for advice (not for a description of what users said).
_ADVICE_QUESTION = [re.compile(p, re.I) for p in [
    r"\bshould\s+i\b", r"\bwhich\s+(?:fund|sip|stock|mutual\s+fund|scheme)\b", r"\bbest\s+(?:fund|sip|stock|scheme)\b",
    r"\bwhat\s+(?:should|to)\s+(?:i\s+)?(?:buy|sell|invest|do)\b", r"\bwhere\s+(?:should|to)\s+i\s+invest\b",
    r"\brecommend\b", r"\bis\s+it\s+(?:good|safe|wise)\s+to\b", r"\bshould\s+we\b", r"\bpredict\b|\bforecast\b",
]]


def violations(answer: str) -> list[str]:
    """Reasons the answer breaks the rules; empty list means it is allowed."""
    found: list[str] = []
    if _FUND_NAMED.search(answer) or _FUND_SUFFIX.search(answer):
        found.append("names a fund or AMC")
    if _STOCK_NAMED.search(answer):
        found.append("names a stock")
    if any(r.search(answer) for r in _ADVICE_RE):
        found.append("gives advice")
    if any(r.search(answer) for r in _URGENCY_RE):
        found.append("uses urgency language")
    return found


def is_allowed(answer: str) -> bool:
    """True when the answer passes every check."""
    return not violations(answer)


def is_advice_question(question: str) -> bool:
    """True when the user is asking for a recommendation rather than a description."""
    return any(r.search(question) for r in _ADVICE_QUESTION)
