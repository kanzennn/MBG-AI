# Indonesian Sentiment Analysis (IndoBERT)

Fine-tunes [`indolem/indobert-base-uncased`](https://huggingface.co/indolem/indobert-base-uncased) to classify Indonesian text into **Positive**, **Negative**, or **Neutral** sentiment. Trained on a custom-labeled Indonesian Twitter dataset.

## Requirements

- Python 3.10+
- PyTorch (CPU or CUDA)

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# CPU-only PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or with CUDA (auto-detects GPU):
pip install torch
```

## Data

Place your labeled CSV file at `data/twitter_mbg_labeled.csv`. The file must contain a text column (`text`, `tweet`, `sentence`, `content`, `comment`, or `full_text`) and a label column (`sentiment`, `label`, `target`, or `class`) with values `Positive`, `Negative`, or `Neutral`.

## Usage

### Train

```powershell
# Default hyperparameters (10 epochs, batch size 8, lr 3e-5, cosine schedule)
python main.py train

# Custom hyperparameters
python main.py train --epochs 15 --batch-size 8 --lr 3e-5 --max-length 128
```

The best checkpoint is saved to `model_output/best/`. Training uses early stopping (patience=5), cosine LR schedule with 10% warmup, and weighted cross-entropy loss to handle class imbalance. A timestamped report is written to `reports/report_YYYYMMDD_HHMMSS.txt` after each run, containing the model name, hyperparameters, per-epoch metrics, classification report, and confusion matrix.

### Evaluate

Re-run the classification report on the held-out 20% validation split without retraining:

```powershell
python main.py evaluate
python main.py evaluate --model-dir model_output/best
```

### Predict

```powershell
python main.py predict --text "Produk ini sangat bagus!" "Sangat mengecewakan."
```

Output example:

```
'Produk ini sangat bagus!'        → Positive (94.31%)
'Sangat mengecewakan.'            → Negative (88.72%)
```

## Label Mapping

| ID | Label    |
|----|----------|
| 0  | Positive |
| 1  | Negative |
| 2  | Neutral  |

## Project Structure

```
.
├── main.py                     # Training, evaluation, and inference
├── requirements.txt
├── data/
│   └── twitter_mbg_labeled.csv # Labeled dataset (gitignored)
├── model_output/
│   └── best/                   # Saved model + tokenizer after training
└── reports/
    └── report_YYYYMMDD_HHMMSS.txt  # Auto-generated after each training run
```
