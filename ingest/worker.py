"""Orchestrates the labeling pipeline: for each platform, for each pending
posts_with_comments batch in Argus, extract -> label -> ingest each document
into PawonWarga-BE, then mark the batch processed (or leave it pending on
partial failure so the next run retries it).

No business logic lives here — extraction is in ingest/extractors, labeling
in ingest/labeler.py, I/O in ingest/clients.
"""

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ingest.clients.argus_client import TARGET_COLLECTION, ArgusClient
from ingest.clients.pawonwarga_client import PawonWargaClient
from ingest.config import Settings, load_settings
from ingest.extractors import EXTRACTORS
from ingest.labeler import Labeler

logger = logging.getLogger(__name__)


def _configure_logging(log_file: str) -> None:
    # Two sinks: stdout (captured by `docker exec`/cron's own redirect, and
    # useful for a live `docker exec ... tail`) and a rotating file the VPS
    # host can read directly via docker-compose.yml's ./logs volume mount —
    # no need to exec into the container just to check on a run.
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    file_handler = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])


def run(settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    _configure_logging(settings.log_file)

    argus = ArgusClient(settings.argus_base_url)
    pawonwarga = PawonWargaClient(settings.pawonwarga_base_url, settings.pawonwarga_api_key)

    logger.info("loading model from %s ...", settings.model_dir)
    labeler = Labeler(settings.model_dir, settings.model_version)
    logger.info("model loaded (version=%s)", settings.model_version)

    started = time.monotonic()
    for platform in settings.platforms:
        _run_platform(platform, argus, pawonwarga, labeler, settings.batch_page_size)
    logger.info("=== ingest run finished in %.1fs ===", time.monotonic() - started)


def _run_platform(
    platform: str,
    argus: ArgusClient,
    pawonwarga: PawonWargaClient,
    labeler: Labeler,
    page_size: int,
) -> None:
    extractor = EXTRACTORS[platform]
    all_batch_ids = argus.get_pending_batch_ids(platform)

    # This worker only consumes posts_with_comments for now — see
    # ingest/clients/argus_client.py.
    target_batches = [
        batch_id for batch_id in all_batch_ids
        if argus.get_batch_collection(platform, batch_id) == TARGET_COLLECTION
    ]

    logger.info(
        "[%s] %d/%d pending batch(es) are %s — %d skipped",
        platform, len(target_batches), len(all_batch_ids), TARGET_COLLECTION,
        len(all_batch_ids) - len(target_batches),
    )

    for i, batch_id in enumerate(target_batches, start=1):
        logger.info(
            "[%s] batch %d/%d (%.0f%%) — %s",
            platform, i, len(target_batches), i / len(target_batches) * 100, batch_id,
        )
        _process_batch(platform, batch_id, extractor, argus, pawonwarga, labeler, page_size)


def _process_batch(
    platform: str,
    batch_id: str,
    extractor,
    argus: ArgusClient,
    pawonwarga: PawonWargaClient,
    labeler: Labeler,
    page_size: int,
) -> None:
    total = 0
    failures = 0
    grand_total = 0
    log_every = 1  # recomputed once grand_total is known, so small batches log every doc

    for doc, grand_total in argus.iter_batch_documents(platform, batch_id, page_size):
        total += 1
        if total == 1:
            log_every = max(1, grand_total // 10)  # ~10 progress lines for large batches

        try:
            post = extractor.extract(doc)
            labeler.label(post)
            pawonwarga.ingest_post(post.to_payload())
        except Exception:
            failures += 1
            logger.exception(
                "[%s] batch=%s document %d/%d failed to ingest", platform, batch_id, total, grand_total
            )

        if total % log_every == 0 or total == grand_total:
            logger.info(
                "[%s] batch=%s progress: %d/%d (%.0f%%)",
                platform, batch_id, total, grand_total, total / grand_total * 100,
            )

    if failures == 0:
        # Only a fully successful batch is marked "processed". Argus's
        # /batches/pending only returns status="pending" documents, so
        # marking a partially-failed batch "failed" would make it silently
        # invisible to future runs. Leaving it "pending" instead means the
        # next `ingest` run retries it automatically — safe because
        # PawonWarga-BE's upsert is idempotent, so docs that already
        # succeeded are just refreshed, never duplicated.
        argus.update_batch_status(platform, batch_id, "processed")

    logger.info(
        "[%s] batch=%s finished: %d document(s), %d failure(s) -> %s",
        platform, batch_id, total, failures,
        "processed" if failures == 0 else "left pending for retry",
    )
