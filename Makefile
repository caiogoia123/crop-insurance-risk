# Crop insurance risk - pipeline entry points (requires uv)
UV ?= uv
PY := $(UV) run python

.PHONY: setup data features train evaluate inmet scr test lint format api all

setup:            ## install pinned dependencies (training extras + dev tools)
	$(UV) sync --extra train

data:             ## stream PSR (PII dropped on the fly), municipalities, NASA POWER, ONI
	$(PY) -m croprisk.data all
	$(PY) -m croprisk.dataset

features:         ## climate, history and contract features
	$(PY) -m croprisk.features.build

train: features   ## tune once, then walk-forward backtest of every model + final model
	$(PY) -m croprisk.backtest
	$(PY) -m croprisk.train

evaluate:         ## metrics tables and figures from the backtest predictions
	$(PY) -m croprisk.evaluate
	$(PY) -m croprisk.explain

inmet:            ## INMET vs NASA POWER data-quality study
	$(PY) -m croprisk.quality.inmet

scr:              ## optional: PSR claim waves vs rural-credit default (BCB SCR.data)
	$(PY) -m croprisk.analysis.scr_credit

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

api:              ## run the API locally on :8001
	$(UV) run uvicorn croprisk.serving.app:app --port 8001

all: data train evaluate
