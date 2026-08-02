"""Wraps the trained IndoBERT model (main.py's load_model/predict_with_model)
to attach sentiment to a NormalizedPost and its comments. The model is loaded
once per Labeler instance and reused for every post the worker processes.
"""

from main import load_model, predict_with_model

from ingest.domain import NormalizedPost

# main.py's labels are "Positive"/"Neutral"/"Negative"; PawonWarga-BE's
# model.Sentiment enum is lowercase (see PawonWarga-BE/internal/model/post.go).
_LABEL_TO_SENTIMENT = {"Positive": "positive", "Neutral": "neutral", "Negative": "negative"}


class Labeler:
    def __init__(self, model_dir: str, model_version: str, max_length: int = 256):
        self.tokenizer, self.model = load_model(model_dir)
        self.model_version = model_version
        self.max_length = max_length

    def label(self, post: NormalizedPost) -> NormalizedPost:
        # Batch the post's content with all its comments in one forward pass
        # instead of one predict call per text.
        texts = [post.content] + [c.content for c in post.comments]
        results = predict_with_model(self.tokenizer, self.model, texts, self.max_length)

        post.sentiment = _LABEL_TO_SENTIMENT[results[0]["label"]]
        post.sentiment_score = results[0]["confidence"]
        post.model_version = self.model_version

        for comment, result in zip(post.comments, results[1:]):
            comment.sentiment = _LABEL_TO_SENTIMENT[result["label"]]
            comment.sentiment_score = result["confidence"]
            comment.model_version = self.model_version

        return post
