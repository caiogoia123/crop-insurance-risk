# Model card: pre-season crop insurance claim model

## Summary
LightGBM classifier that estimates, at contract time, the probability that a
subsidized crop insurance policy (PSR, Brazil) will be indemnified. Served at
`https://caiogoia.duckdns.org/crop-risk/` (`POST /predict`). A second model, the
in-season early-warning model, is evaluated in the backtest but not served.

| | |
|---|---|
| Type | binary classification, gradient-boosted trees (LightGBM 4.7) |
| Version | see `/model-info` (`version`, `created_at`) |
| Training data | PSR/SISSER policies, safras 2006/07 to 2022/23 |
| Holdout | safra 2023/24 (never seen by the served model) |
| Features | contract fields, as-of portfolio history, 1981-2005 climatology of the crop's critical window, ENSO at contract (see [README](README.md#method)) |
| Not used | indemnity, cause of loss, premium, rate, subsidy, policyholder identity, farm coordinates |
| Owner | Caio Goia (portfolio project, not a production underwriting system) |

## Intended use
- Research and portfolio demonstration of risk scoring with public data.
- Ranking policies by relative claim risk at contract time, and comparing that
  ranking with the price charged.

## Out of scope
- Pricing or accepting real policies. The data has selection bias (only
  subsidized policies), no farm-level information and a claims cutoff.
- Any use on individual farmers: the model scores contracts from public aggregates
  and contract fields; it must not be used to profile people.

## Performance
RESULTS_PLACEHOLDER

## Calibration
CALIBRATION_PLACEHOLDER

## Ethical and privacy notes
- Personal data (name, CPF/CNPJ) is dropped while the source CSV is streamed;
  property coordinates are dropped too. The repository has no policy-level data.
- The served bundle contains the trees and municipality-level aggregates built from
  at least 10 policies each.
- The insurer is a feature (underwriting and claims practices differ between
  insurers). It is a business attribute, not a protected one.

## Monitoring and retraining
- Monthly retrain on the VM (systemd timer, day 3, 04:30 UTC, low priority). The
  challenger replaces the champion only if it is not worse on the latest matured
  safra (AUC -0.002 / Brier +0.001 tolerance) and the API passes a health check
  after the swap; otherwise the champion stays.
- Each run appends metrics, score PSI and the top-10 feature PSI to
  `/opt/crop-risk/logs/retrain.jsonl`.
- Expected drift: ENSO and history features move every season by construction
  (PSI > 1 for ONI is normal: one ENSO state per season). Contract-value features
  are relative to the previous safra to avoid nominal-money drift.
