# Progress

Working log to resume the work if context is lost. Most recent first.

## Status (2026-10-08, about 21:50 local)
- [x] Scaffold, pinned deps, git, public repo caiogoia123/crop-insurance-risk, CI green
- [x] PSR ingestion (PII dropped while streaming), municipalities, POWER (1,304 cells), ONI
- [x] Cleaning/target/censoring: 1,532,459 policies, safras 2006/07-2023/24
- [x] Features: crop-calendar windows, climatology 1981-2005, in-season anomalies, as-of history,
      relative money/yield (D12, fixed after the nominal drift was found)
- [x] INMET vs POWER quality module (reports/inmet_quality.json + 3 figures)
- [x] API (FastAPI) + bundle + retrain code + deploy files + 58 tests
- [x] VM: docker.io 29.1.3 installed, repo cloned at /opt/crop-risk/app, image built (old commit)
- [ ] Backtest rerun with relative features (running: data/logs/backtest.log)
- [ ] evaluate + explain -> figures, results.md, metrics_summary.json
- [ ] train bundle -> scp to VM -> git pull + rebuild image -> run-api.sh -> nginx -> timer
- [ ] Trigger one retrain on the VM to validate it end to end
- [ ] READMEs (EN+PT), MODEL_CARD, DATA_CARD, ROLLBACK, final review, memory, CLAUDE.md row, report

## Commands
- `make data` / `make features` / `uv run python -m croprisk.backtest` / `make evaluate` / `uv run python -m croprisk.train`
- VM: `ssh -i ~/.ssh/caio-vm ubuntu@<vm>`; base dir /opt/crop-risk (app, data, models, logs, reports)

## VM state before changes (2026-10-09 00:01 UTC)
11 GiB RAM (8.8 free), disk 41 GB free, nginx active, no docker, ports 22/80/443 (+111 rpcbind),
iptables INPUT allows 22/80/443 only, site https://caiogoia.duckdns.org/ = 200.
