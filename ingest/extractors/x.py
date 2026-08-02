from datetime import datetime

from ingest.domain import NormalizedComment, NormalizedPost
from ingest.extractors.base import Extractor, parse_extracted_at

# X's created_at is a fixed-format string, e.g. "Sat Jul 25 18:26:52 +0000 2026"
# — not unix epoch or ISO-8601 like the other two platforms.
_X_CREATED_AT_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def _parse_x_created_at(value: str) -> datetime:
    return datetime.strptime(value, _X_CREATED_AT_FORMAT)


def _tweet_text(node: dict) -> str:
    # Tweets over 280 chars are truncated in legacy.full_text; the full body
    # lives in note_tweet.note_tweet_results.result.text (see Argus's
    # src/scripts/x-cleaner.ts, which hits the same gotcha). Retweets have
    # their own truncation inside legacy.retweeted_status_result, which is
    # not unwrapped here — legacy.full_text's "RT @user: ..." preview is used
    # as-is for retweets.
    note = (
        (node.get("note_tweet") or {})
        .get("note_tweet_results", {})
        .get("result", {})
        .get("text")
    )
    if note:
        return note
    return node.get("legacy", {}).get("full_text", "")


def _tweet_author(node: dict) -> tuple[str | None, str | None]:
    # X moved screen_name/name from user.legacy to user.core; read core
    # first, fall back to legacy (same schema-drift handling as x-cleaner.ts).
    try:
        user = node["core"]["user_results"]["result"]
    except (KeyError, TypeError):
        return None, None

    core = user.get("core") or {}
    legacy = user.get("legacy") or {}
    return (
        core.get("screen_name") or legacy.get("screen_name"),
        core.get("name") or legacy.get("name"),
    )


def _tweet_view_count(node: dict) -> int:
    try:
        return int(node.get("views", {}).get("count") or 0)
    except (TypeError, ValueError):
        return 0


class XExtractor(Extractor):
    platform = "x"

    def extract(self, doc: dict) -> NormalizedPost:
        payload = doc["raw_payload"]
        post = payload["post"]
        legacy = post.get("legacy", {})
        crawled_at = parse_extracted_at(doc)
        handle, name = _tweet_author(post)

        return NormalizedPost(
            platform=self.platform,
            platform_post_id=legacy.get("id_str") or str(post["rest_id"]),
            content=_tweet_text(post),
            published_at=_parse_x_created_at(legacy["created_at"]),
            crawled_at=crawled_at,
            author_handle=handle,
            author_name=name,
            url=f"https://x.com/{handle}/status/{legacy.get('id_str')}" if handle else None,
            like_count=legacy.get("favorite_count") or 0,
            comment_count=legacy.get("reply_count") or 0,
            # No single "shares" concept on X — approximate with retweets + quotes.
            share_count=(legacy.get("retweet_count") or 0) + (legacy.get("quote_count") or 0),
            view_count=_tweet_view_count(post),
            raw_payload=post,
            comments=[self._extract_comment(c) for c in payload.get("comments", [])],
        )

    @staticmethod
    def _extract_comment(reply: dict) -> NormalizedComment:
        # X replies inside posts_with_comments are full tweet nodes, structurally
        # identical to `post` (same legacy/core shape) — reuse the same helpers.
        legacy = reply.get("legacy", {})
        handle, _ = _tweet_author(reply)
        return NormalizedComment(
            platform_comment_id=legacy.get("id_str") or str(reply["rest_id"]),
            content=_tweet_text(reply),
            published_at=_parse_x_created_at(legacy["created_at"]),
            author_handle=handle,
            like_count=legacy.get("favorite_count") or 0,
            raw_payload=reply,
        )
