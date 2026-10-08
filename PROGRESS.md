# Progress

Working log to resume the work if context is lost. Most recent first.

## Status (2026-10-08)
- [x] Scaffold: uv, pinned deps, git init (local)
- [x] PSR ingestion with PII dropped while streaming (`make data`)
- [x] Municipality coordinates + POWER grid cells (1,304 cells for all of Brazil)
- [ ] NASA POWER download (running in background, `data/logs/power.log`)
- [x] Cleaning/target/censoring (`croprisk.dataset`): 1,532,459 policies, 18 safras
- [ ] Climate features (climatology 1981-2005, in-season observed, ENSO ONI)
- [ ] History features (as-of loss rates)
- [ ] Walk-forward backtest, baselines, LR, LightGBM
- [ ] Metrics, plots, SHAP, MLflow
- [ ] INMET data-quality module
- [ ] API (FastAPI) + Docker + VM deploy + retrain timer
- [ ] READMEs, cards, CI, GitHub repo
- [ ] Final review, memory, CLAUDE.md row, report

## Notes
- 2016-2024 file truncated at Excel limit (see DECISIONS D2).
- Claims cutoff ~2025-03-16; 2024/25 excluded (D5).
- Coverage dates are contractual -> crop calendar windows (D3).
