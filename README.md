# Indonesian Sentiment Analysis (IndoBERT)

Fine-tunes [`indolem/indobertweet-base-uncased`](https://huggingface.co/indolem/indobertweet-base-uncased) — IndoBERT pretrained on Indonesian tweets — to classify Indonesian text into **Positive**, **Negative**, or **Neutral** sentiment. Trained on a custom-labeled Indonesian Twitter dataset. This is the model the [ingest worker](#ingest-worker) uses; the base `indolem/indobert-base-uncased` variant is no longer the focus (still usable via `--model`/`--output-dir` if you want to compare).

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

Place your labeled CSV file at `data/merged_labeled.csv`. The file must contain a text column (`text_clean`, `text`, `tweet`, `sentence`, `content`, `comment`, or `full_text` — `text_clean` is preferred when present) and a label column (`sentiment`, `label`, `target`, `class`, or `sentiment_label`) with values `Positive`, `Negative`, or `Neutral`. An optional `source_file` column tracks which original batch each row came from, used to apply per-batch numeric-label conventions where they differ.

## Usage

### Train

```powershell
# Default: indolem/indobertweet-base-uncased, 15 epochs, batch size 8, lr 3e-5, cosine schedule
python main.py train

# Custom hyperparameters
python main.py train --epochs 15 --batch-size 8 --lr 3e-5 --max-length 256

# Compare against the base (non-tweet) model
python main.py train --model indolem/indobert-base-uncased --output-dir model_output
```

The best checkpoint is saved to `<output-dir>/best/` (`model_output_tweet/best/` by default). Training uses early stopping (patience=3), cosine LR schedule with 6% warmup, and weighted cross-entropy loss to handle class imbalance. A timestamped report is written to `reports/report_YYYYMMDD_HHMMSS.txt` after each run, containing the model name, hyperparameters, per-epoch metrics, classification report, and confusion matrix.

### Baseline

Fast TF-IDF + Logistic Regression sanity check (no GPU, seconds) — a floor the transformer has to beat and a quick signal on label quality:

```powershell
python main.py baseline
```

### Evaluate

Re-run the classification report on the held-out ~20% validation split without retraining:

```powershell
python main.py evaluate
python main.py evaluate --model-dir model_output/best   # e.g. to check the base-model run instead
```

If `--model-dir` is omitted, it defaults to the most recently trained `model_output_tweet*/best` directory found on disk.

### Predict

```powershell
python main.py predict --text "Produk ini sangat bagus!" "Sangat mengecewakan."
```

Output example:

```
'Produk ini sangat bagus!'        → Positive (94.31%)
'Sangat mengecewakan.'            → Negative (88.72%)
```

### Label

Batch-label a new CSV with a trained model:

```powershell
python main.py label --input data/raw_scrape.csv --output data/labeled.csv
```

## Label Mapping

| ID | Label    |
|----|----------|
| 0  | Positive |
| 1  | Neutral  |
| 2  | Negative |

## Ingest Worker

`python main.py ingest` pulls pending `posts_with_comments` batches from [Argus](../Argus), labels them with a trained model, and pushes the results to PawonWarga-BE's internal ingest endpoint. It is a **one-shot batch job**, not a server — run it manually, or on a schedule via cron (see [CI/CD](#cicd-deploy-to-vps) below).

```powershell
cp .env.example .env
# fill in ARGUS_BASE_URL, PAWONWARGA_BASE_URL, PAWONWARGA_API_KEY

python main.py ingest
```

Config is entirely environment-driven — see [ingest/config.py](ingest/config.py) and `.env.example`. `INGEST_MODEL_DIR` (default `model_output_tweet/best`) must point at a trained model directory.

## Docker

```bash
docker build -t sentiment-analysis .
docker compose up -d        # uses docker-compose.yml
```

The image only contains `main.py`, `ingest/`, and installed dependencies — it does **not** include the trained model (`model_output*/`, gitignored/dockerignored, large binary artifacts) or `data/`. The container itself just idles (`CMD ["sleep", "infinity"]`); `python main.py ingest` is run inside it periodically via cron (`docker exec sentiment-app python main.py ingest`), the same pattern [Argus](../Argus) uses for its nightly crawl.

**Important notes:**

- `docker-compose.yml` mounts `./model_output_tweet` read-only into the container — train locally (or elsewhere), then copy the resulting `best/` directory to the server once. There is no in-container training. (The base `model_output/` isn't mounted — the deployed ingest worker only ever runs the tweet model.)
- `ARGUS_BASE_URL` / `PAWONWARGA_BASE_URL` should point at `http://host.docker.internal:<port>` to reach Argus/PawonWarga-BE's ports published on the same VPS host (the compose file sets up `extra_hosts` for this).

## CI/CD (Deploy to VPS)

- **CI** — [.github/workflows/ci.yml](.github/workflows/ci.yml) runs `ruff check .` on every push and pull request to `main`. Static analysis only (no heavy ML deps installed).
- **CD** — every push to `main` also triggers [.github/workflows/deploy.yml](.github/workflows/deploy.yml):
  1. **Build & Push** — builds the Docker image, pushes to `ghcr.io/<owner>/<repo>:latest` and `:<git-sha>`.
  2. **Deploy** — SSHes into the VPS, pulls the new image, and restarts only the `app` service via `docker compose up -d --no-deps app` in `/opt/pawonwarga/kira`.

**Required GitHub Actions secrets** (repo-level — add via the GitHub UI or `gh secret set`; these do not carry over from Argus/PawonWarga-BE's repos even on the same VPS):

| Secret | Description |
|--------|-------------|
| `VPS_HOST` | VPS IP / hostname |
| `VPS_USER` | SSH user |
| `VPS_SSH_KEY` | SSH private key (PEM) |
| `VPS_PORT` | SSH port (optional, defaults to 22) |
| `CR_PAT` | GitHub PAT with `read:packages` scope — used by the VPS to pull from ghcr.io |

**One-time server setup:**

```bash
mkdir -p /opt/pawonwarga/kira
# copy docker-compose.yml and a filled-in .env to /opt/pawonwarga/kira/
# copy the trained model directory (model_output_tweet/best) there too
```

Then schedule the ingest job in cron:

```cron
*/30 * * * * cd /opt/pawonwarga/kira && docker exec sentiment-app python main.py ingest >> /var/log/sentiment-ingest.log 2>&1
```

After the one-time setup, every push to `main` redeploys the image automatically; the cron schedule and the container's model volume are untouched by a deploy.

## Project Structure

```
.
├── main.py                     # Training, baseline, evaluation, inference, batch labeling, and the `ingest` entry point
├── requirements.txt
├── ingest/                     # Ingest worker: Argus → label → PawonWarga-BE (see Ingest Worker above)
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/
│   ├── ci.yml                  # Lint on push/PR
│   └── deploy.yml              # Build, push, and deploy to the VPS on push to main
├── data/
│   └── merged_labeled.csv      # Labeled dataset (gitignored)
├── model_output_tweet/
│   └── best/                   # Deployed model: indobertweet fine-tune (gitignored, default --output-dir)
├── model_output/                # Legacy base-model (indobert) runs — not deployed, kept for comparison only
│   └── best/
└── reports/
    └── report_YYYYMMDD_HHMMSS.txt  # Auto-generated after each training run
```
