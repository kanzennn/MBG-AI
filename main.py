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
from sklearn.model_selection import train_test_split
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
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

MODEL_NAME = "indolem/indobert-base-uncased"
DATA_DIR = Path("data")
OUTPUT_DIR = Path("model_output")
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

TEXT_COLS  = {"text", "tweet", "sentence", "content", "comment", "full_text"}
LABEL_COLS = {"sentiment", "label", "target", "class"}

# Numeric label map for INA_TweetsPPKM dataset (0=Positive,1=Neutral,2=Negative)
NUMERIC_LABEL_MAP = {0: "Positive", 2: "Negative", 1: "Neutral",
                     "0": "Positive", "2": "Negative", "1": "Neutral"}


def _read_csv(path: Path) -> pd.DataFrame:
    """Try reading with comma then tab separator, utf-8 then latin-1."""
    for sep in [",", "\t"]:
        for enc in ["utf-8", "latin-1"]:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, encoding_errors="replace")
                # A single-column result means the separator was wrong
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue
    raise ValueError(f"Cannot read {path.name} with any known separator/encoding.")


def _extract_frame(path: Path) -> pd.DataFrame | None:
    """Read one CSV and return a normalised (text, sentiment) DataFrame.
    Returns None if the file has no usable label column (skip silently)."""
    df = _read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    text_col  = next((c for c in df.columns if c in TEXT_COLS), None)
    label_col = next((c for c in df.columns if c in LABEL_COLS), None)

    if text_col is None or label_col is None:
        return None  # no labels — skip (e.g. raw/unlabeled files)

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "sentiment"})
    df = df.dropna()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()

    # Numeric labels → named labels
    df["sentiment"] = df["sentiment"].apply(
        lambda s: NUMERIC_LABEL_MAP.get(s, NUMERIC_LABEL_MAP.get(
            int(s) if s.lstrip("-").isdigit() else s, s
        ))
    )

    df["sentiment"] = df["sentiment"].str.capitalize()

    return df[df["sentiment"].isin(LABELS)].reset_index(drop=True)


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
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"\nTotal loaded: {len(df):,} samples (after dedup).")
    print(df["sentiment"].value_counts().to_string())
    return df


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
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
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
        f"Model          : {MODEL_NAME}",
        "─" * 64,
        "Hyperparameters",
        f"  Epochs        : {epochs}",
        f"  Batch size    : {batch_size}  (effective: {batch_size * 2} with grad accum ×2)",
        f"  Learning rate : {lr}",
        f"  LR scheduler  : cosine  (warmup 10%)",
        f"  Max length    : {max_length}",
        f"  Loss          : Weighted CrossEntropy + label smooth 0.05",
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

def train(epochs: int = 15, batch_size: int = 8, lr: float = 3e-5, max_length: int = 128):
    df = load_dataset()

    df["label_id"] = df["sentiment"].map(LABEL2ID)

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label_id"]
    )

    print(f"\nTrain: {len(train_df):,}  |  Val: {len(val_df):,}")
    print(train_df["label_id"].map(ID2LABEL).value_counts().to_string())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        max_grad_norm=0.5,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()

    # Save final model + tokenizer
    trainer.save_model(str(OUTPUT_DIR / "best"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "best"))
    print(f"\nModel saved to {OUTPUT_DIR / 'best'}")

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
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        train_size=len(train_df),
        val_size=len(val_df),
        label_dist=label_dist,
        log_history=trainer.state.log_history,
        cls_report=cls_report,
        conf_matrix=conf_mat,
    )
    print(f"\nReport saved to {report_path}")


# ── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate(model_dir: str = str(OUTPUT_DIR / "best"), max_length: int = 128):
    """Run classification report on the 20% validation split without retraining."""
    df = load_dataset()
    df["label_id"] = df["sentiment"].map(LABEL2ID)
    _, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_id"])

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


# ── Inference ────────────────────────────────────────────────────────────────

def predict(texts: list[str], model_dir: str = str(OUTPUT_DIR / "best")) -> list[dict]:
    """Return label + confidence for each text."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).numpy()

    results = []
    for prob_row in probs:
        idx = int(np.argmax(prob_row))
        results.append({"label": ID2LABEL[idx], "confidence": float(prob_row[idx])})
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    train_p = subparsers.add_parser("train")
    train_p.add_argument("--epochs", type=int, default=15)
    train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--lr", type=float, default=3e-5)
    train_p.add_argument("--max-length", type=int, default=128)

    eval_p = subparsers.add_parser("evaluate")
    eval_p.add_argument("--model-dir", default=str(OUTPUT_DIR / "best"))
    eval_p.add_argument("--max-length", type=int, default=128)

    predict_p = subparsers.add_parser("predict")
    predict_p.add_argument("--text", nargs="+", required=True)
    predict_p.add_argument("--model-dir", default=str(OUTPUT_DIR / "best"))

    args = parser.parse_args()

    if args.command == "train":
        train(args.epochs, args.batch_size, args.lr, args.max_length)
    elif args.command == "evaluate":
        evaluate(args.model_dir, args.max_length)
    elif args.command == "predict":
        results = predict(args.text, args.model_dir)
        for text, res in zip(args.text, results):
            print(f"{text!r:60s} → {res['label']} ({res['confidence']:.2%})")
    else:
        parser.print_help()
