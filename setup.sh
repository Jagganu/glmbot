#!/usr/bin/env bash
# glmbot universal setup — Windows (Git Bash) / Linux / macOS.
# Usage: bash setup.sh [--light]   (--light skips rich; Termux users: bash setup-termux.sh)
set -e
LIGHT=0
if [ "${1:-}" = "--light" ]; then LIGHT=1; fi

echo "=== glmbot setup ==="
PYBIN="python3"
command -v python3 >/dev/null || PYBIN="py"
command -v "$PYBIN" >/dev/null || { echo "ERROR: no python found"; exit 1; }

echo "[1/4] python: $($PYBIN --version)"
echo "[2/4] installing dependencies ..."
$PYBIN -m pip install --upgrade pip
if [ "$LIGHT" = "1" ]; then
  $PYBIN -m pip install requests PyYAML
else
  $PYBIN -m pip install -r requirements.txt
fi

echo "[3/4] config ..."
if [ ! -f config.yml ]; then
  cp config.example.yml config.yml
  echo "  -> created config.yml (edit api keys: nano config.yml)"
  chmod 600 config.yml || true
else
  echo "  -> config.yml exists, keeping it"
fi
mkdir -p data/logs data/cache

echo "[4/4] verifying ..."
$PYBIN -c "import requests, yaml; print('  deps OK')"
$PYBIN tests.py 2>&1 | tail -4 || true

echo ""
echo "=== done ==="
echo "  1. edit config.yml (keys, trade_type, watchlist, risk)"
echo "  2. $PYBIN bot.py doctor"
echo "  3. $PYBIN bot.py backtest -d 7"
echo "  4. $PYBIN bot.py run"
