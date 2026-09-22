.PHONY: help install install-light dev test lint format typecheck run backtest doctor clean docker-build docker-run

PY ?= py
UNIT ?= tests.py

help:
	@echo "glmbot targets:"
	@echo "  make install        full install (requests + yaml + rich)"
	@echo "  make install-light  minimal install (Termux friendly, no rich)"
	@echo "  make dev            install + dev tools (pytest, ruff, mypy)"
	@echo "  make test           run unit tests"
	@echo "  make lint           ruff check"
	@echo "  make format         ruff format check"
	@echo "  make typecheck      mypy glmbot"
	@echo "  make run            start trading loop (paper or live per config)"
	@echo "  make backtest       backtest watchlist (7d default)"
	@echo "  make doctor         environment + connectivity diagnostics"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

install-light:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install requests PyYAML

dev:
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

test:
	$(PY) $(UNIT)

lint:
	ruff check glmbot bot.py tests.py || $(PY) -m ruff check glmbot bot.py tests.py

format:
	ruff format --check glmbot bot.py tests.py || $(PY) -m ruff format --check glmbot bot.py tests.py

typecheck:
	mypy glmbot || $(PY) -m mypy glmbot

run:
	$(PY) bot.py run

backtest:
	$(PY) bot.py backtest -d 7

doctor:
	$(PY) bot.py doctor

clean:
	rm -rf __pycache__ glmbot/__pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -name "*.pyc" -delete

docker-build:
	docker build -t glmbot:latest .

docker-run:
	docker run --rm -it --env-file .env -v ./data:/app/data glmbot:latest run
