"""MongoDB client for Argus's sentimen output.

Argus v2 exposed a batch-polling HTTP API (`/api/v1/sentimen/{platform}/batches/pending`,
`/batch/{id}/data`, `PATCH /batch/{id}/status`) that this client used to call. Argus v3
has no such API — its sentimen surface is job-triggered (POST a keyword, poll a jobId)
and writes documents straight to MongoDB, with no endpoint to list "whatever is
pending" across runs. Since the document shape it writes is the same one Argus v2's
API used to hand back (see ARGUS_V3/src/sentimen/document.ts, copied field-for-field
from MesinTempur), this client now reads that MongoDB collection directly instead of
going through Argus at all. No code below this module needed to change — extractors
already read `doc["raw_payload"]`/`doc["extracted_at"]`, which are unchanged.
"""

from __future__ import annotations

from typing import Iterator

from pymongo import MongoClient
from pymongo.collection import Collection

# posts_with_comments is the only collection this worker consumes (one doc
# per post, nesting the post + its comments) — see ARGUS_V3's
# src/sentimen/document.ts (SENTIMEN_COLLECTION).
TARGET_COLLECTION = "posts_with_comments"

# Kira's own platform key -> Argus v3's per-platform database name, copied
# verbatim from ARGUS_V3/src/sentimen/document.ts's DATABASE map. Argus v3's
# internal Platform enum spells X as "twitter"; Kira's own PLATFORMS tuple
# (ingest/config.py) already spells it "x" — this map is the translation.
_DATABASE_BY_PLATFORM = {
    "x": "sentimen_x",
    "instagram": "sentimen_instagram",
    "tiktok": "sentimen_tiktok",
}


class ArgusClient:
    """Same four-method shape the old HTTP client had — get_pending_batch_ids,
    get_batch_collection, iter_batch_documents, update_batch_status — so
    worker.py's orchestration (which batch, in what order, when to mark it
    processed) did not need to change, only how each method gets its answer.
    """

    def __init__(self, mongodb_url: str):
        self._client: MongoClient = MongoClient(mongodb_url)

    def _collection(self, platform: str) -> Collection:
        return self._client[_DATABASE_BY_PLATFORM[platform]][TARGET_COLLECTION]

    def get_pending_batch_ids(self, platform: str) -> list[str]:
        return self._collection(platform).distinct("batch_id", {"status": "pending"})

    def get_batch_collection(self, platform: str, batch_id: str) -> str | None:
        """Returns which collection a batch lives in, or None if not found.

        Argus v3's sentimen surface only ever writes to posts_with_comments
        (there is no other search_mode on this surface — see document.ts),
        so this just confirms the batch actually has documents.
        """
        exists = self._collection(platform).find_one({"batch_id": batch_id}, {"_id": 1})
        return TARGET_COLLECTION if exists is not None else None

    def iter_batch_documents(
        self, platform: str, batch_id: str, page_size: int
    ) -> Iterator[tuple[dict, int]]:
        """Yields (document, total_document_count) for every raw document in a
        batch. total_document_count is exposed so callers can report progress
        without a separate counting request — mirrors the old HTTP client's
        pagination contract, backed here by one query instead of many pages.
        """
        collection = self._collection(platform)
        total = collection.count_documents({"batch_id": batch_id})
        for doc in collection.find({"batch_id": batch_id}, batch_size=page_size):
            yield doc, total

    def update_batch_status(self, platform: str, batch_id: str, status: str) -> None:
        self._collection(platform).update_many(
            {"batch_id": batch_id}, {"$set": {"status": status}}
        )

    def close(self) -> None:
        self._client.close()
