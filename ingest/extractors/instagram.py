from ingest.domain import NormalizedComment, NormalizedPost
from ingest.extractors.base import Extractor, from_unix, parse_extracted_at


class InstagramExtractor(Extractor):
    platform = "instagram"

    def extract(self, doc: dict) -> NormalizedPost:
        payload = doc["raw_payload"]
        post = payload["post"]
        crawled_at = parse_extracted_at(doc)

        owner = post.get("owner") or post.get("user") or {}
        caption = post.get("caption") or {}

        return NormalizedPost(
            platform=self.platform,
            platform_post_id=str(post.get("pk") or post["id"]),
            content=caption.get("text") or post.get("accessibility_caption") or "",
            published_at=from_unix(post["taken_at"]),
            crawled_at=crawled_at,
            author_handle=owner.get("username"),
            author_name=owner.get("full_name"),
            url=f"https://www.instagram.com/p/{post['code']}/" if post.get("code") else None,
            like_count=post.get("like_count") or 0,
            comment_count=post.get("comment_count") or 0,
            raw_payload=post,
            comments=[self._extract_comment(c) for c in payload.get("comments", [])],
        )

    @staticmethod
    def _extract_comment(comment: dict) -> NormalizedComment:
        user = comment.get("user") or {}
        return NormalizedComment(
            platform_comment_id=str(comment["pk"]),
            content=comment.get("text") or "",
            published_at=from_unix(comment.get("created_at_utc") or comment["created_at"]),
            author_handle=user.get("username"),
            like_count=comment.get("comment_like_count") or 0,
            raw_payload=comment,
        )
