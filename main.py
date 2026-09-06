"""
Sentiment analysis fine-tuning with IndoBERT.
Labels: Positive, Negative, Neutral
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ── Constants ────────────────────────────────────────────────────────────────

LABELS = ["Positive", "Neutral", "Negative"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

MODEL_NAME = "indolem/indobertweet-base-uncased"
DATA_DIR = Path("data")
OUTPUT_DIR = Path("model_output_tweet")
REPORTS_DIR = Path("reports")


# ── Dataset ──────────────────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }



# ── Data loading ─────────────────────────────────────────────────────────────

# Ordered by preference: cleaned text first when a file provides both
TEXT_COL_PRIORITY = ["text_clean", "text", "tweet", "sentence", "content", "comment", "full_text"]
LABEL_COLS = {"sentiment", "label", "target", "class", "sentiment_label"}

# Numeric label map for INA_TweetsPPKM dataset (0=Positive,1=Neutral,2=Negative)
NUMERIC_LABEL_MAP = {0: "Positive", 2: "Negative", 1: "Neutral",
                     "0": "Positive", "2": "Negative", "1": "Neutral"}

# Files whose numeric labels don't follow the default convention above
FILE_NUMERIC_LABEL_MAPS = {
    # labeled with 0=Neutral, 1=Positive, 2=Negative
    "sentimen_twitter_labeled_4.csv": {"0": "Neutral", "1": "Positive", "2": "Negative"},
}


def _read_csv(path: Path) -> pd.DataFrame:
    """Try reading with comma then tab separator, utf-8 then latin-1.
    Encodings are tried strict first so a real decode failure actually raises
    and falls through to the next encoding — lossy replacement is only used
    as a last resort if no strict combination works, otherwise a genuinely
    latin-1 file would silently "succeed" as UTF-8 with mangled characters."""
    for errors in ["strict", "replace"]:
        for sep in [",", "\t"]:
            for enc in ["utf-8", "latin-1"]:
                try:
                    df = pd.read_csv(path, sep=sep, encoding=enc, encoding_errors=errors)
                    # A single-column result means the separator was wrong
                    if len(df.columns) > 1:
                        return df
                except Exception:
                    continue
    raise ValueError(f"Cannot read {path.name} with any known separator/encoding.")


def _clean_text(s: pd.Series) -> pd.Series:
    """Strip URLs and @mentions — noise absent from inference-time input."""
    return (
        s.astype(str)
        .str.replace(r"https?://\S+", "", regex=True)
        .str.replace(r"@\w+", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _extract_frame(path: Path) -> pd.DataFrame | None:
    """Read one CSV and return a normalised (text, sentiment) DataFrame.
    Returns None if the file has no usable label column (skip silently)."""
    df = _read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    text_col  = next((c for c in TEXT_COL_PRIORITY if c in df.columns), None)
    label_col = next((c for c in df.columns if c in LABEL_COLS), None)

    if text_col is None or label_col is None:
        return None  # no labels — skip (e.g. raw/unlabeled files)

    has_source_col = "source_file" in df.columns
    cols = [text_col, label_col] + (["source_file"] if has_source_col else [])
    df = df[cols].rename(columns={text_col: "text", label_col: "sentiment"})
    df = df.dropna(subset=["text", "sentiment"])
    df["sentiment"] = df["sentiment"].astype(str).str.strip()

    # Numeric labels → named labels. The convention can vary per original batch
    # file, so prefer the per-row `source_file` provenance column (merged CSVs
    # combine multiple batches) and fall back to this file's own name.
    def _map_label(s: str, origin: str) -> str:
        m = FILE_NUMERIC_LABEL_MAPS.get(origin, NUMERIC_LABEL_MAP)
        return m.get(s, m.get(int(s) if s.lstrip("-").isdigit() else s, s))

    origins = df["source_file"] if has_source_col else path.name
    if has_source_col:
        df["sentiment"] = [_map_label(s, o) for s, o in zip(df["sentiment"], origins)]
    else:
        df["sentiment"] = df["sentiment"].apply(lambda s: _map_label(s, origins))

    df["sentiment"] = df["sentiment"].str.capitalize()
    df["text"] = _clean_text(df["text"])

    return df[df["sentiment"].isin(LABELS)][["text", "sentiment"]].reset_index(drop=True)


def load_dataset() -> pd.DataFrame:
    """Load all CSVs from data/, normalise labels, and concatenate."""
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in data/. Run: python download_data.py"
        )

    frames = []
    for f in sorted(csv_files):
        frame = _extract_frame(f)
        if frame is None or len(frame) == 0:
            print(f"  {f.name}: skipped (no label column or no usable rows)")
            continue
        print(f"  {f.name}: {len(frame):,} usable rows")
        frames.append(frame)

    if not frames:
        raise ValueError("No usable data found in any CSV file.")

    df = pd.concat(frames, ignore_index=True)

    # Drop texts too short to carry sentiment signal
    n_before = len(df)
    df = df[df["text"].str.split().str.len() >= 4]
    print(f"\nDropped {n_before - len(df):,} texts shorter than 4 words.")

    # Same text labeled differently across files → annotation noise, drop all
    dedup_key = df["text"].str.lower()
    conflict = dedup_key.map(df.groupby(dedup_key)["sentiment"].nunique()) > 1
    print(f"Dropped {conflict.sum():,} rows with conflicting labels across files.")
    df = df[~conflict]

    df = df.loc[~dedup_key[~conflict].duplicated()].reset_index(drop=True)
    print(f"Total loaded: {len(df):,} samples (after dedup).")
    print(df["sentiment"].value_counts().to_string())

    df["dup_group"] = _near_dup_groups(df["text"])
    sizes = df["dup_group"].value_counts()
    n_clustered = len(df) - (sizes == 1).sum()
    print(
        f"Near-duplicate clusters: {len(sizes):,} groups "
        f"({n_clustered:,} rows in multi-text clusters, largest {sizes.max()})."
    )
    return df


def _near_dup_groups(texts: pd.Series) -> np.ndarray:
    """Cluster near-duplicate texts (TF-IDF cosine >= 0.8) into connected
    components. Template/campaign tweets that differ by a link or a couple of
    words end up in one group so the split can keep them on the same side."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import connected_components

    norm = (
        texts.str.lower()
        .str.replace(r"[^a-z0-9# ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    tfidf = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(norm)

    adjacency = (
        NearestNeighbors(metric="cosine", radius=0.2)
        .fit(tfidf)
        .radius_neighbors_graph(tfidf, mode="connectivity")
    )
    _, labels = connected_components(adjacency, directed=False)
    return labels


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/val split that never puts two texts from the same
    near-duplicate cluster on different sides (no template leakage).

    Assigns whole dup_group clusters to val with a greedy heuristic that fills
    each label's val share up to test_size. Unlike StratifiedGroupKFold(
    n_splits=round(1/test_size)), this doesn't require >= round(1/test_size)
    distinct groups per class (a real risk once heavily-templated minority
    classes collapse into a handful of clusters) and it targets test_size
    directly instead of only reproducing it exactly for values where
    1/test_size happens to be a whole number.
    """
    rng = np.random.default_rng(random_state)

    label_ids = sorted(df["label_id"].unique())
    group_label_counts = (
        df.groupby(["dup_group", "label_id"]).size()
        .unstack(fill_value=0)
        .reindex(columns=label_ids, fill_value=0)
    )
    targets = df["label_id"].value_counts().reindex(label_ids, fill_value=0) * test_size

    order = group_label_counts.index.to_numpy().copy()
    rng.shuffle(order)

    val_running = pd.Series(0.0, index=label_ids)
    val_groups = []
    for g in order:
        counts = group_label_counts.loc[g]
        if ((val_running + counts) <= targets).all():
            val_running += counts
            val_groups.append(g)

    val_mask = df["dup_group"].isin(val_groups)
    train_df, val_df = df[~val_mask], df[val_mask]

    achieved = len(val_df) / len(df)
    print(
        f"Split: {len(train_df):,} train / {len(val_df):,} val "
        f"(target val {test_size:.0%}, achieved {achieved:.1%})"
    )
    empty = [ID2LABEL[i] for i in label_ids if val_running.get(i, 0) == 0]
    if empty:
        print(f"WARNING: labels with zero val rows after grouped split: {empty}")

    return train_df, val_df


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    from sklearn.metrics import f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": float(acc), "macro_f1": float(macro_f1)}


# ── Report ───────────────────────────────────────────────────────────────────

def save_training_report(
    *,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    warmup_ratio: float,
    train_size: int,
    val_size: int,
    label_dist: dict,
    log_history: list,
    cls_report: str,
    conf_matrix,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"report_{ts}.txt"

    lines = [
        "=" * 64,
        "TRAINING REPORT",
        f"Date           : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model          : {model_name}",
        "─" * 64,
        "Hyperparameters",
        f"  Epochs        : {epochs}",
        f"  Batch size    : {batch_size}  (effective: {batch_size * 2} with grad accum ×2)",
        f"  Learning rate : {lr}",
        f"  LR scheduler  : cosine  (warmup {warmup_ratio:.0%})",
        f"  Max length    : {max_length}",
        "  Loss          : Weighted CrossEntropy + label smooth 0.05",
        "─" * 64,
        "Dataset",
        f"  Train samples : {train_size:,}",
        f"  Val samples   : {val_size:,}",
        "  Label distribution (train):",
    ]
    for label, count in label_dist.items():
        lines.append(f"    {label:<10}: {count:,}")

    lines += ["─" * 64, "Per-epoch metrics"]
    epoch_metrics = [e for e in log_history if "eval_loss" in e]
    if epoch_metrics:
        lines.append(f"  {'Epoch':>6}  {'Eval Loss':>10}  {'Accuracy':>10}  {'Macro F1':>10}")
        for m in epoch_metrics:
            lines.append(
                f"  {m.get('epoch', 0):>6.0f}  "
                f"{m.get('eval_loss', float('nan')):>10.4f}  "
                f"{m.get('eval_accuracy', float('nan')):>10.4f}  "
                f"{m.get('eval_macro_f1', float('nan')):>10.4f}"
            )
    else:
        lines.append("  (no epoch metrics recorded)")

    lines += [
        "─" * 64,
        "Classification Report",
        cls_report,
        "─" * 64,
        "Confusion Matrix  (rows=actual, cols=predicted)",
        f"  Labels: {LABELS}",
        str(conf_matrix),
        "=" * 64,
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ── Training ─────────────────────────────────────────────────────────────────

def train(
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 3e-5,
    max_length: int = 256,
    model_name: str = MODEL_NAME,
    output_dir: str | Path = OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    df = load_dataset()

    df["label_id"] = df["sentiment"].map(LABEL2ID)
    train_df, val_df = split_data(df)

    print(f"\nTrain: {len(train_df):,}  |  Val: {len(val_df):,}")
    print(train_df["label_id"].map(ID2LABEL).value_counts().to_string())

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_dataset = SentimentDataset(train_df["text"], train_df["label_id"], tokenizer, max_length)
    val_dataset = SentimentDataset(val_df["text"], val_df["label_id"], tokenizer, max_length)

    # Weighted cross-entropy to handle class imbalance
    counts = train_df["label_id"].value_counts().sort_index()
    weights = torch.tensor(
        [1.0 / counts.get(i, 1) for i in range(len(LABELS))], dtype=torch.float
    )
    weights = weights / weights.sum() * len(LABELS)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            device = outputs.logits.device
            loss = nn.CrossEntropyLoss(
                weight=weights.to(device), label_smoothing=0.05
            )(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_ratio = 0.06
    args = TrainingArguments(
        output_dir=str(output_dir),
        seed=42,
        data_seed=42,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # Save final model + tokenizer
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))
    print(f"\nModel saved to {output_dir / 'best'}")

    # Evaluation report
    preds_output = trainer.predict(val_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    all_ids = list(range(len(LABELS)))
    cls_report = classification_report(labels, preds, labels=all_ids, target_names=LABELS, zero_division=0)
    conf_mat = confusion_matrix(labels, preds, labels=all_ids)

    print("\n── Classification Report ──")
    print(cls_report)
    print("── Confusion Matrix ──")
    print(conf_mat)

    label_dist = train_df["label_id"].map(ID2LABEL).value_counts().to_dict()
    report_path = save_training_report(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        warmup_ratio=warmup_ratio,
        train_size=len(train_df),
        val_size=len(val_df),
        label_dist=label_dist,
        log_history=trainer.state.log_history,
        cls_report=cls_report,
        conf_matrix=conf_mat,
    )
    print(f"\nReport saved to {report_path}")


# ── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate(model_dir: str = str(OUTPUT_DIR / "best"), max_length: int = 256):
    """Run classification report on the 20% validation split without retraining."""
    df = load_dataset()
    df["label_id"] = df["sentiment"].map(LABEL2ID)
    _, val_df = split_data(df)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    val_dataset = SentimentDataset(val_df["text"], val_df["label_id"], tokenizer, max_length)
    inputs = {k: v for k, v in val_dataset.encodings.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = np.argmax(logits.numpy(), axis=-1)
    labels = val_dataset.labels.numpy()

    all_ids = list(range(len(LABELS)))
    print("\n── Classification Report ──")
    print(classification_report(labels, preds, labels=all_ids, target_names=LABELS, zero_division=0))
    print("── Confusion Matrix ──")
    print(confusion_matrix(labels, preds, labels=all_ids))


# ── Baseline ─────────────────────────────────────────────────────────────────

def baseline():
    """TF-IDF + Logistic Regression on the same split — sanity check for label
    quality and a floor the transformer has to beat. Runs in seconds, no GPU."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    df = load_dataset()
    df["label_id"] = df["sentiment"].map(LABEL2ID)
    train_df, val_df = split_data(df)
    print(f"\nTrain: {len(train_df):,}  |  Val: {len(val_df):,}")

    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    clf.fit(train_df["text"], train_df["label_id"])
    preds = clf.predict(val_df["text"])

    all_ids = list(range(len(LABELS)))
    print("\n── Baseline: TF-IDF + Logistic Regression ──")
    print(classification_report(val_df["label_id"], preds, labels=all_ids, target_names=LABELS, zero_division=0))
    print("── Confusion Matrix ──")
    print(confusion_matrix(val_df["label_id"], preds, labels=all_ids))


# ── Inference ────────────────────────────────────────────────────────────────

def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir: str = str(OUTPUT_DIR / "best")):
    """Load tokenizer + model once. Reuse across many predict_with_model() calls
    instead of calling predict() per text/batch, which reloads the weights
    every time (see ingest/labeler.py for the worker's usage). Moves the model
    to the GPU when one is available — predict_with_model() reads the device
    back off the model, so no other call site needs to change."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(_select_device())
    model.eval()
    return tokenizer, model


def predict_with_model(tokenizer, model, texts: list[str], max_length: int = 256) -> list[dict]:
    """Return label + confidence for each text using an already-loaded model."""
    device = next(model.parameters()).device
    cleaned = _clean_text(pd.Series(texts)).tolist()
    inputs = tokenizer(
        cleaned, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for prob_row in probs:
        idx = int(np.argmax(prob_row))
        results.append({"label": ID2LABEL[idx], "confidence": float(prob_row[idx])})
    return results


def predict(texts: list[str], model_dir: str = str(OUTPUT_DIR / "best"), max_length: int = 256) -> list[dict]:
    """Return label + confidence for each text. Loads the model fresh — fine for
    a one-off CLI call, but see load_model()/predict_with_model() for repeated use."""
    tokenizer, model = load_model(model_dir)
    return predict_with_model(tokenizer, model, texts, max_length)


# ── Batch labeling ───────────────────────────────────────────────────────────

def label_csv(
    input_path: str,
    output_path: str,
    text_col: str = "text",
    batch_size: int = 64,
    model_dir: str = str(OUTPUT_DIR / "best"),
    max_length: int = 256,
):
    """Label every row in a CSV and write results to a new CSV."""
    df = _read_csv(Path(input_path))
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available: {list(df.columns)}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    texts = _clean_text(df[text_col].fillna("")).tolist()
    labels, confidences = [], []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).numpy()
        for prob_row in probs:
            idx = int(np.argmax(prob_row))
            labels.append(ID2LABEL[idx])
            confidences.append(round(float(prob_row[idx]), 4))

        done = min(i + batch_size, len(texts))
        print(f"  {done:,} / {len(texts):,}", end="\r")

    df["sentiment"] = labels
    df["confidence"] = confidences
    df.to_csv(output_path, index=False)
    print(f"\nLabeled {len(df):,} rows → {output_path}")


# ── Label audit ──────────────────────────────────────────────────────────────

def _previous_audit_files() -> list[Path]:
    """Every past audit round's output, so a new round automatically skips text
    that was already reviewed — including candidates the new round itself
    produces, once that file lands here for the *next* run."""
    return sorted(REPORTS_DIR.glob("label_audit_*.csv"))


def find_label_disagreements(
    confidence_threshold: float = 0.90,
    top_n: int | None = None,
    exclude: list[str] | None = None,
    model_dir: str | None = None,
    max_length: int = 256,
    batch_size: int = 64,
    output: str | None = None,
):
    """Re-run the trained model over data/merged_labeled.csv and surface rows
    where the model disagrees with the stored label at high confidence — the
    same "model is probably right, label is probably wrong" signal used for
    reports/label_audit_candidates.csv (see reports/CATATAN_SESI_2026-07-06.md).
    Rows whose text already appears in a previous reports/label_audit_*.csv are
    skipped, so re-running this after a review round only surfaces new ground.
    """
    model_dir = model_dir or _default_model_dir()
    exclude_paths = [Path(p) for p in exclude] if exclude else _previous_audit_files()

    df = _read_csv(DATA_DIR / "merged_labeled.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.dropna(subset=["text", "label"])
    df["text"] = _clean_text(df["text"])
    df["label"] = df["label"].astype(str).str.strip().str.capitalize()
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)

    already_audited: set[str] = set()
    for path in exclude_paths:
        if not path.exists():
            print(f"  (skip, not found: {path})")
            continue
        prev = _read_csv(path)
        prev.columns = [c.lower().strip() for c in prev.columns]
        if "text" in prev.columns:
            already_audited.update(_clean_text(prev["text"].dropna()).tolist())
        print(f"  excluding already-reviewed text from {path}")

    before = len(df)
    df = df[~df["text"].isin(already_audited)].reset_index(drop=True)
    print(f"Excluded {before - len(df):,} rows already covered by previous audits.")

    tokenizer, model = load_model(model_dir)
    print(f"Running {len(df):,} remaining rows through {model_dir} ...")

    preds, confs = [], []
    for i in range(0, len(df), batch_size):
        batch = df["text"].iloc[i : i + batch_size].tolist()
        results = predict_with_model(tokenizer, model, batch, max_length)
        preds.extend(r["label"] for r in results)
        confs.extend(r["confidence"] for r in results)
        done = min(i + batch_size, len(df))
        print(f"  {done:,} / {len(df):,}", end="\r")
    print()

    df["pred"] = preds
    df["conf"] = confs

    candidates = df[(df["pred"] != df["label"]) & (df["conf"] >= confidence_threshold)]
    candidates = candidates.sort_values("conf", ascending=False)
    if top_n:
        candidates = candidates.head(top_n)

    cols = ["text", "label", "pred", "conf"]
    if "source_file" in candidates.columns:
        cols = ["source_file"] + cols
    candidates = candidates[cols].rename(columns={"label": "sentiment", "source_file": "source"})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output) if output else REPORTS_DIR / f"label_audit_candidates_{ts}.csv"
    candidates.to_csv(out_path, index=False)
    print(
        f"\n{len(candidates):,} candidates (model disagrees, confidence >= "
        f"{confidence_threshold:.0%}) -> {out_path}"
    )
    return out_path


def _default_model_dir() -> str:
    """Most recently modified model_output_tweet*/best directory, or
    OUTPUT_DIR if none has been trained yet. Keeps evaluate/predict/label's
    default model in sync with whatever `train --output-dir` last wrote.
    Scoped to model_output_tweet* only — the indobertweet-tuned line is the
    one this project is focused on; the legacy indobert-base model_output/
    directory (if still present on disk) is intentionally never
    auto-selected, only reachable via an explicit --model-dir."""
    candidates = [p for p in Path(".").glob("model_output_tweet*/best") if p.is_dir()]
    if not candidates:
        return str(OUTPUT_DIR / "best")
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    train_p = subparsers.add_parser("train")
    train_p.add_argument("--epochs", type=int, default=15)
    train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--lr", type=float, default=3e-5)
    train_p.add_argument("--max-length", type=int, default=256)
    train_p.add_argument("--model", default=MODEL_NAME)
    train_p.add_argument("--output-dir", default=str(OUTPUT_DIR))

    subparsers.add_parser("baseline")

    eval_p = subparsers.add_parser("evaluate")
    eval_p.add_argument("--model-dir", default=_default_model_dir())
    eval_p.add_argument("--max-length", type=int, default=256)

    predict_p = subparsers.add_parser("predict")
    predict_p.add_argument("--text", nargs="+", required=True)
    predict_p.add_argument("--model-dir", default=_default_model_dir())
    predict_p.add_argument("--max-length", type=int, default=256)

    label_p = subparsers.add_parser("label")
    label_p.add_argument("--input", required=True, help="Input CSV path")
    label_p.add_argument("--output", required=True, help="Output CSV path")
    label_p.add_argument("--text-col", default="text", help="Name of the text column")
    label_p.add_argument("--batch-size", type=int, default=64)
    label_p.add_argument("--model-dir", default=_default_model_dir())
    label_p.add_argument("--max-length", type=int, default=256)

    subparsers.add_parser(
        "ingest",
        help="Pull pending posts_with_comments batches from Argus, label them, "
             "and push to PawonWarga-BE. Config via env — see .env.example.",
    )

    audit_p = subparsers.add_parser(
        "audit",
        help="Find rows in data/merged_labeled.csv where the model disagrees "
             "with the stored label at high confidence, skipping text already "
             "covered by a previous reports/label_audit_*.csv round. Writes a "
             "new reports/label_audit_candidates_<timestamp>.csv for review.",
    )
    audit_p.add_argument("--confidence-threshold", type=float, default=0.90)
    audit_p.add_argument("--top-n", type=int, default=None, help="Keep only the N highest-confidence candidates")
    audit_p.add_argument("--exclude", nargs="*", default=None, help="Override which past audit CSVs to skip (default: all reports/label_audit_*.csv)")
    audit_p.add_argument("--model-dir", default=_default_model_dir())
    audit_p.add_argument("--max-length", type=int, default=256)
    audit_p.add_argument("--batch-size", type=int, default=64)
    audit_p.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.command == "train":
        train(args.epochs, args.batch_size, args.lr, args.max_length, args.model, args.output_dir)
    elif args.command == "baseline":
        baseline()
    elif args.command == "evaluate":
        evaluate(args.model_dir, args.max_length)
    elif args.command == "predict":
        results = predict(args.text, args.model_dir, args.max_length)
        for text, res in zip(args.text, results):
            print(f"{text!r:60s} → {res['label']} ({res['confidence']:.2%})")
    elif args.command == "label":
        label_csv(args.input, args.output, args.text_col, args.batch_size, args.model_dir, args.max_length)
    elif args.command == "ingest":
        # Imported lazily: ingest.labeler imports load_model/predict_with_model
        # back from this module, so importing it before main.py has finished
        # defining them (i.e. at module load time) would be a circular import.
        from ingest.worker import run
        run()
    elif args.command == "audit":
        find_label_disagreements(
            args.confidence_threshold,
            args.top_n,
            args.exclude,
            args.model_dir,
            args.max_length,
            args.batch_size,
            args.output,
        )
    else:
        parser.print_help()
