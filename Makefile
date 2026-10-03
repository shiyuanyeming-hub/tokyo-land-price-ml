# Tokyo Land Price ML - task runner
#
# Every target is a thin wrapper around the CLI entry points so that the
# README, CI and a human all run exactly the same commands.

PYTHON ?= python3
CONFIG ?= configs/config.yaml

# LightGBM's native library needs an OpenMP runtime. On macOS with Homebrew's
# libomp installed this is automatic; set LIGHTGBM_LIBOMP to override.
MPLCONFIGDIR ?= /tmp/mplcache
export MPLCONFIGDIR

.PHONY: help all data models errors figures report test test-fast lint clean notebook

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

all: ## Run the complete pipeline (data -> models -> leakage -> errors -> figures -> report)
	$(PYTHON) -m src.pipeline --config $(CONFIG) --stage all

data: ## Load, clean and feature-engineer the raw CSVs, then print the data profile
	$(PYTHON) -m src.pipeline --config $(CONFIG) --stage all

test: ## Run the full pytest suite
	$(PYTHON) -m pytest

test-fast: ## Run the test suite without the slow model-fitting tests
	$(PYTHON) -m pytest -m "not slow"

lint: ## Check formatting and imports (requires ruff)
	ruff check src tests

notebook: ## Re-execute both notebooks in place
	$(PYTHON) scripts/run_notebooks.py

clean: ## Remove generated artefacts (raw data is never touched)
	rm -rf data/processed/*.csv data/processed/*.json reports/figures/*.png \
		reports/metrics.json reports/model_comparison.md reports/improvements.md \
		reports/error_analysis.md
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
