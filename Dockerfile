FROM python:3.11-slim

WORKDIR /app

# System deps for scientific-Python wheels (scikit-learn/numpy/pandas build
# tooling on slim images) — kept minimal, removed after apt cache cleanup.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY main.py .
COPY ingest ./ingest

# No persistent server here — `ingest` (see main.py's `ingest` subcommand) is
# a one-shot batch job invoked periodically via cron (`docker exec ... python
# main.py ingest`), not a long-running process. The container just idles so
# cron always has a warm, dependency-ready environment to exec into.
CMD ["sleep", "infinity"]
