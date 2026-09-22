#!/data/data/com.termux/files/usr/bin/bash
# glmbot Termux (Android) setup — run INSIDE Termux, in the glmbot folder.
#
#   pkg install git -y
#   git clone <your-repo> glmbot && cd glmbot
#   bash setup-termux.sh [--full]
#   python bot.py doctor && python bot.py run
#
# Default = light (~2 MB: requests + PyYAML, ASCII tables).
# --full adds rich (colored tables) + dev tools.
set -e

FULL=0
if [ "${1:-}" = "--full" ]; then FULL=1; fi

echo "=== glmbot Termux setup ==="
echo "[1/4] installing python via pkg ..."
pkg update -y || true
pkg install -y python openssl

echo "[2/4] installing python deps ..."
python -m pip install --upgrade pip
if [ "$FULL" = "1" ]; then
  python -m pip install -r requirements.txt
else
  python -m pip install requests PyYAML
  echo "  (light install — add colors later with: pip install rich)"
fi

echo "[3/4] preparing config + data dirs ..."
if [ ! -f config.yml ]; then
  cp config.example.yml config.yml
  chmod 600 config.yml || true
  echo "  -> created config.yml (edit keys: nano config.yml)"
else
  echo "  -> config.yml exists, keeping it"
fi
mkdir -p data/logs data/cache

echo "[4/4] verifying ..."
python -c "import requests, yaml; print('  deps OK')"
python tests.py 2>&1 | tail -4 || true
python bot.py validate || echo "  (validate failed — edit config.yml)"

echo ""
echo "=== done ==="
echo "  1. nano config.yml            # paste API keys"
echo "  2. python bot.py doctor       # full diagnostics"
echo "  3. python bot.py backtest -d 7"
echo "  4. python bot.py run          # start trading"
echo ""
echo "keep alive on Android:"
echo "  termux-wake-lock              # stop battery-killer"
echo "  tmux new -s bot               # survives app switch (pkg install tmux)"
echo "  # reattach: tmux attach -t bot"
