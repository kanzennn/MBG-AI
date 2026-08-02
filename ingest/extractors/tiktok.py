from ingest.domain import NormalizedComment, NormalizedPost
from ingest.extractors.base import Extractor, from_unix, parse_extracted_at


class TikTokExtractor(Extractor):
    platform = "tiktok"

    def extract(self, doc: dict) -> NormalizedPost:
        payload = doc["raw_payload"]
        post = payload["post"]
        crawled_at = parse_extracted_at(doc)

        author = post.get("author") or {}
        stats = post.get("stats") or {}
        unique_id = author.get("uniqueId")

        return NormalizedPost(
            platform=self.platform,
            platform_post_id=str(post["id"]),
            content=post.get("desc") or "",
            published_at=from_unix(post["createTime"]),
            crawled_at=crawled_at,
            author_handle=unique_id,
            author_name=author.get("nickname"),
            url=f"https://www.tiktok.com/@{unique_id}/video/{post['id']}" if unique_id else None,
            like_count=stats.get("diggCount") or 0,
            comment_count=stats.get("commentCount") or 0,
            share_count=stats.get("shareCount") or 0,
            view_count=stats.get("playCount") or 0,
            raw_payload=post,
            comments=[self._extract_comment(c) for c in payload.get("comments", [])],
        )

    @staticmethod
    def _extract_comment(comment: dict) -> NormalizedComment:
        user = comment.get("user") or {}
        # comment.user.uniqueId is frequently null on TikTok (unlike the post's
        # author) — fall back to nickname so author_handle isn't empty.
        handle = user.get("uniqueId") or user.get("nickname")
        return NormalizedComment(
            platform_comment_id=str(comment["cid"]),
            content=comment.get("text") or "",
            published_at=from_unix(comment["create_time"]),
            author_handle=handle,
            like_count=comment.get("digg_count") or 0,
            raw_payload=comment,
        )
