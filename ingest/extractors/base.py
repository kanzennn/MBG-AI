"""Extractor contract: turn one Argus posts_with_comments envelope document
into a platform-agnostic NormalizedPost. Extractors only map fields — they
never call the model or an HTTP client.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone

from ingest.domain import NormalizedPost


class Extractor(ABC):
    platform: str

    @abstractmethod
    def extract(self, doc: dict) -> NormalizedPost:
        """doc is one raw Argus document: {batch_id, extracted_at, raw_payload: {post, comments}, ...}."""
        raise NotImplementedError


def parse_extracted_at(doc: dict) -> datetime:
    # Argus stores extracted_at as an ISO-8601 string (see Argus/CLAUDE.md).
    return datetime.fromisoformat(doc["extracted_at"].replace("Z", "+00:00"))


def from_unix(ts: int) -> datetime:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc)
