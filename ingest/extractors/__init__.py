from ingest.extractors.base import Extractor
from ingest.extractors.instagram import InstagramExtractor
from ingest.extractors.tiktok import TikTokExtractor
from ingest.extractors.x import XExtractor

EXTRACTORS: dict[str, Extractor] = {
    "instagram": InstagramExtractor(),
    "tiktok": TikTokExtractor(),
    "x": XExtractor(),
}
