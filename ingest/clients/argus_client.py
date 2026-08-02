"""HTTP client for Argus's ETL-facing batch API (see Argus/README.md
"MongoDB Document Schema" and "Data pipeline"). No direct MongoDB access —
Argus is the only thing that talks to its own database.
"""

import requests

# posts_with_comments is the only collection this worker consumes (one doc
# per post, nesting the post + its comments) — see Argus/CLAUDE.md.
TARGET_COLLECTION = "posts_with_comments"


class ArgusClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout

    def _url(self, platform: str, path: str) -> str:
        return f"{self.base_url}/api/v1/sentimen/{platform}{path}"

    def get_pending_batch_ids(self, platform: str) -> list[str]:
        resp = requests.get(self._url(platform, "/batches/pending"), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["data"]["pending_batch_ids"]

    def get_batch_collection(self, platform: str, batch_id: str) -> str | None:
        """Returns which collection a batch lives in, or None if not found."""
        resp = requests.get(self._url(platform, f"/batch/{batch_id}"), timeout=self.timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()["data"]["collection"]

    def iter_batch_documents(self, platform: str, batch_id: str, page_size: int):
        """Yields (document, total_document_count) for every raw document in a
        batch, transparently paginating. total_document_count is exposed so
        callers can report progress without a separate counting request."""
        page = 1
        while True:
            resp = requests.get(
                self._url(platform, f"/batch/{batch_id}/data"),
                params={"page": page, "limit": page_size},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            pagination = body["pagination"]
            for doc in body["data"]:
                yield doc, pagination["total"]

            if page >= pagination["totalPages"]:
                return
            page += 1

    def update_batch_status(self, platform: str, batch_id: str, status: str) -> None:
        resp = requests.patch(
            self._url(platform, f"/batch/{batch_id}/status"),
            json={"status": status},
            timeout=self.timeout,
        )
        resp.raise_for_status()
