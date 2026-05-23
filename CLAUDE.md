# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sentiment analysis fine-tuning project using Indonesian BERT (`indolem/indobert-base-uncased`). Classifies text into **Positive, Negative, Neutral**. Data: `data/twitter_mbg_labeled.csv` (custom-labeled Indonesian Twitter dataset).

## Environment

Python virtual environment at `.venv/`. Activate in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install all dependencies:

```powershell
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only
# or: pip install torch  # auto-selects CUDA if available
```

## Common Commands

```powershell
# Train  (80/20 split, defaults: 10 epochs, batch 8, lr 3e-5, cosine schedule)
python main.py train

# Train with custom hyperparameters
python main.py train --epochs 15 --batch-size 8 --lr 3e-5

# Re-evaluate without retraining
python main.py evaluate

# Run inference on trained model
python main.py predict --text "I love this product!" "This is terrible."
```

## Architecture

| File | Role |
|---|---|
| `main.py` | Entry point — `train` and `predict` subcommands |
| `data/` | Raw CSV files (gitignored) |
| `model_output/best/` | Saved model + tokenizer after training |

### Training flow (`main.py`)

1. `load_dataset()` — reads all CSVs from `data/`, auto-detects text/label columns, filters to the three target labels
2. `train_test_split` — 80 % train / 20 % validation, stratified
3. `SentimentDataset` — wraps tokenised tensors for the Trainer API
4. `Trainer` — HuggingFace Trainer with early stopping (patience=5), cosine LR schedule, warmup 10%, best checkpoint saved at `model_output/best/`
5. After training: prints full `classification_report` and confusion matrix on the validation set

### Label mapping

```
0 → Positive
1 → Negative
2 → Neutral
```
