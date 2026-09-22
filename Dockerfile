# glmbot — production-ready container image (works on amd64/arm64 incl. phones via Termux-adjacent hosts)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps kept minimal (TLS + tzdata). No compilers needed: pure wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY bot.py pyproject.toml README.md config.example.yml ./
COPY glmbot ./glmbot

# Data (SQLite + logs + cache) lives here; mount a volume to persist.
RUN mkdir -p data/logs data/cache
VOLUME ["/app/data"]

# Drop privileges: run as non-root.
RUN useradd -m -u 10001 trader && chown -R trader:trader /app
USER trader

# Default config path; override with -c or GLMBOT_CONFIG env.
ENV GLMBOT_CONFIG=/app/data/config.yml

ENTRYPOINT ["python", "bot.py"]
CMD ["--help"]
