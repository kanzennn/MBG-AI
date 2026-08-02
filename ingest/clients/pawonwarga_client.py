"""HTTP client for PawonWarga-BE's internal ingest endpoint
(see PawonWarga-BE/internal/handler/ingest.go)."""

import requests


class PawonWargaClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 30):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

    def ingest_post(self, payload: dict) -> None:
        resp = requests.post(
            f"{self.base_url}/api/v1/ingest/posts",
            json=payload,
            headers={"X-API-Key": self.api_key},
            timeout=self.timeout,
        )
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # requests' default HTTPError message drops the response body,
            # which is exactly where PawonWarga-BE puts the useful part
            # (e.g. per-field validation errors) — attach it.
            raise requests.exceptions.HTTPError(f"{e} | body={resp.text}", response=resp) from e
