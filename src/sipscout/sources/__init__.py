"""Source collectors."""

from .appstore import AppStoreSource
from .base import Source
from .playstore import PlayStoreSource
from .reddit import RedditSource
from .tradingqna import TradingQnASource
from .youtube import YouTubeSource

SOURCE_CLASSES: dict[str, type[Source]] = {
    "playstore": PlayStoreSource,
    "appstore": AppStoreSource,
    "reddit": RedditSource,
    "youtube": YouTubeSource,
    "tradingqna": TradingQnASource,
}

__all__ = ["SOURCE_CLASSES", "Source"]
