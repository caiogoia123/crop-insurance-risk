# Progress

Working log to resume the work if context is lost. Most recent first.

## Status (2026-10-08, about 22:10 local)
- [x] Scaffold, pinned deps, git, public repo caiogoia123/crop-insurance-risk, CI green
- [x] PSR ingestion (PII dropped while streaming), municipalities, POWER (1,304 cells), ONI
- [x] Cleaning/target/censoring: 1,532,459 policies, safras 2006/07-2023/24
- [x] Features: crop-calendar windows, climatology 1981-2005, in-season anomalies, as-of history,
      money/yield relative to previous safra (D12), cumulative counts removed (D11)
- [x] INMET vs POWER quality module; SCR.data credit study (no relation found)
- [x] API + bundle + retrain + deploy files + 59 tests
- [x] VM: docker.io 29.1.3, repo at /opt/crop-risk/app, image built, API container up
      (old bundle 20261009-6de7ef3d), nginx /crop-risk/ installed (backup caiogoia.bak-croprisk),
      rate limit tested (429 after burst), timer enabled (next 2026-11-03 04:30 UTC), POWER cache copied
- [ ] Backtest rerun (running: data/logs/backtest.log) -> then train.py writes final bundle
- [ ] evaluate + explain -> figures, results.md, metrics_summary.json, shap_summary.json
- [ ] Redeploy: git pull + rebuild image, copy final bundle, CURRENT, run-api.sh, check
- [ ] Trigger one retrain on the VM (systemctl start crop-risk-retrain) and check retrain.jsonl
- [ ] READMEs (EN+PT: docs/readme_method_*.md drafts), MODEL_CARD numbers, final review,
      memory file, CLAUDE.md row, report

## Findings so far
- Pre-season ranking is unstable by year (AUC < 0.5 in 2011/12 validation); insurer rate beat
  the first pre-season model on the 2023/24 holdout (0.61 vs 0.59).
- In-season (mid-window weather) validation AUC ~0.74 vs ~0.67 pre-season.
- INMET: median station 23% days without valid rain; IDW LOO r 0.77; POWER monthly r 0.91.
- SCR: claim waves do not precede custeio default (Spearman -0.12, p 0.23); 2024-25 default
  spike is unrelated to PSR claims.

## VM state before changes (2026-10-09 00:01 UTC)
11 GiB RAM (8.8 free), disk 41 GB free, nginx active, no docker, ports 22/80/443 (+111 rpcbind),
iptables INPUT allows 22/80/443 only, site https://caiogoia.duckdns.org/ = 200.
