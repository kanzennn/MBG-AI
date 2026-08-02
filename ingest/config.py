"""Environment-driven settings for the ingest worker. See .env.example."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

PLATFORMS = ("instagram", "x", "tiktok")


@dataclass(frozen=True)
class Settings:
    argus_base_url: str
    pawonwarga_base_url: str
    pawonwarga_api_key: str
    model_dir: str
    model_version: str
    batch_page_size: int
    log_file: str
    platforms: tuple[str, ...] = field(default_factory=lambda: PLATFORMS)


def load_settings() -> Settings:
    return Settings(
        argus_base_url=os.environ["ARGUS_BASE_URL"].rstrip("/"),
        pawonwarga_base_url=os.environ["PAWONWARGA_BASE_URL"].rstrip("/"),
        pawonwarga_api_key=os.environ["PAWONWARGA_API_KEY"],
        model_dir=os.environ.get("INGEST_MODEL_DIR", "model_output_tweet/best"),
        model_version=os.environ.get("INGEST_MODEL_VERSION", "indobertweet-v1"),
        batch_page_size=int(os.environ.get("INGEST_BATCH_PAGE_SIZE", "50")),
        log_file=os.environ.get("INGEST_LOG_FILE", "logs/ingest.log"),
    )
