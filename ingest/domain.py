"""Platform-agnostic shapes produced by extractors and consumed by the labeler
and the PawonWarga client. Field names mirror PawonWarga-BE's IngestPostRequest
/ IngestCommentRequest (see PawonWarga-BE/internal/handler/ingest.go) so
to_payload() below is a near-literal transcription.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class NormalizedComment:
    platform_comment_id: str
    content: str
    published_at: datetime
    author_handle: Optional[str] = None
    like_count: int = 0
    raw_payload: Any = None

    # Filled in by the labeler, not the extractor.
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    model_version: Optional[str] = None

    def to_payload(self) -> dict:
        return {
            "platform_comment_id": self.platform_comment_id,
            "author_handle": self.author_handle,
            "content": self.content,
            "like_count": self.like_count,
            "published_at": self.published_at.isoformat(),
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "model_version": self.model_version,
            "raw_payload": self.raw_payload,
        }


@dataclass
class NormalizedPost:
    platform: str  # "instagram" | "x" | "tiktok" — must match model.Platform on the Go side
    platform_post_id: str
    content: str
    published_at: datetime
    crawled_at: datetime
    author_handle: Optional[str] = None
    author_name: Optional[str] = None
    url: Optional[str] = None
    like_count: int = 0
    comment_count: int = 0
    share_count: int = 0
    view_count: int = 0
    raw_payload: Any = None
    comments: list[NormalizedComment] = field(default_factory=list)

    # Filled in by the labeler, not the extractor.
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    model_version: Optional[str] = None

    def to_payload(self) -> dict:
        return {
            "platform": self.platform,
            "platform_post_id": self.platform_post_id,
            "author_handle": self.author_handle,
            "author_name": self.author_name,
            "content": self.content,
            "url": self.url,
            "like_count": self.like_count,
            "comment_count": self.comment_count,
            "share_count": self.share_count,
            "view_count": self.view_count,
            "published_at": self.published_at.isoformat(),
            "crawled_at": self.crawled_at.isoformat(),
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "model_version": self.model_version,
            "raw_payload": self.raw_payload,
            "comments": [c.to_payload() for c in self.comments],
        }
