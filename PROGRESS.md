# Progress

Working log to resume the work if context is lost. Most recent first.

## Status (2026-10-09, about 00:00 local): done
- [x] Repo public: github.com/caiogoia123/crop-insurance-risk, CI green (ruff + 61 tests)
- [x] Data: PSR streamed with PII dropped, NASA POWER 1,304 cells, ONI, municipalities
- [x] 1,532,459 policies, safras 2006/07-2023/24 (2024/25 censored)
- [x] Backtest: 16 experiments x 11 test safras, MLflow local, reports/results.md
- [x] evaluate + explain: figures, metrics_summary.json, shap_summary.json
- [x] INMET quality study; SCR.data credit study (optional, done)
- [x] Served model: logistic pre-season (D20); bundle 20261009-33c5352d trained on PC
- [x] VM: docker.io, image crop-risk:latest, container crop-risk-api (127.0.0.1:8001),
      Nginx /crop-risk/ with rate limit, timer crop-risk-retrain (monthly, day 3 04:30 UTC)
- [x] First retrain on the VM ran end to end (21 min): challenger 20261009-b8820ffa promoted
      (AUC 0.6295 vs 0.6293, Brier equal), API restarted, health ok, site 200
- [x] READMEs EN/PT, MODEL_CARD, DATA_CARD, DECISIONS (D1-D22), deploy/ROLLBACK.md
- [x] Memory file and CLAUDE.md row (base folder)

## Key results
- Pre-season AUC: served logistic 0.646 vs insurer rate 0.625 (7/11 safras; 0.635 with 1-safra gap)
- Climatology/ENSO add ~0 (+0.002 logistic, -0.005 LightGBM); in-season weather +0.06 (LightGBM 0.695)
- Double lift served model vs rate: loss ratio 0.49 -> 1.45, claim rate 9.2% -> 28.4%
- Models rank, they do not forecast the season's claim level (7%-32%)
- INMET: median station 23% days without valid rain; POWER monthly r 0.91
- SCR: claim waves do not precede custeio default (Spearman -0.12, p 0.23)

## Pending / ideas
- Recalibration of the season level (weight recent safras, or a season-level model) not attempted
- XGBoost not run (LightGBM covers gradient boosting)
- Docker legacy builder warning on the VM (buildx not installed; harmless)

## VM state before changes (2026-10-09 00:01 UTC)
11 GiB RAM (8.8 free), disk 41 GB free, nginx active, no docker, ports 22/80/443 (+111 rpcbind),
iptables INPUT allows 22/80/443 only, site https://caiogoia.duckdns.org/ = 200.
After: disk 5.2 GB used (39 GB free), API container ~210 MB RAM, retrain peak ~2.2 GB.
