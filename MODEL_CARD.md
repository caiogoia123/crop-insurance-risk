# Model card: pre-season crop insurance claim model

## Summary
Logistic regression that estimates, at contract time, the probability that a
subsidized crop insurance policy (PSR, Brazil) will be indemnified. Served at
`https://caiogoia.duckdns.org/crop-risk/` (`POST /predict`). A second model, the
in-season early-warning model, is evaluated in the backtest but not served.

| | |
|---|---|
| Type | binary classification, logistic regression (scikit-learn 1.9) on quantile-normalized numeric features and one-hot categories |
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
Walk-forward backtest, 11 test safras (2013/14 to 2023/24), each scored by a model
trained only on earlier safras:

| | Served model | Insurer's rate | Best in-season model (not served) |
|---|---|---|---|
| AUC, mean (sd) | 0.646 (0.058) | 0.625 (0.080) | 0.695 (0.069) |
| AUC, worst safra | 0.562 (2015/16) | 0.510 (2021/22) | 0.558 (2015/16) |
| KS | 0.248 | 0.245 | 0.307 |
| PR-AUC (base rate 0.168) | 0.245 | 0.228 | 0.299 |
| Brier | 0.1398 | 0.1388 | 0.1344 |
| Safras better than the rate | 7/11 | - | 9/11 |
| With a 1-safra label gap | AUC 0.635 | 0.625 | 0.670 |

Holdout 2023/24 (the served model was trained on 2006/07-2022/23): AUC 0.629 vs
0.610 for the rate, KS 0.196, Brier 0.1411 vs 0.1406, score PSI train vs holdout 0.08.

Why this model and not LightGBM: among contract-time candidates it had the best mean
AUC and by far the most stable ranking. The pre-season LightGBM falls to AUC 0.39 in
2021/22 and 0.48 in 2015/16, when droughts hit crops and regions that are usually
safe. Differences between the candidates are within noise (95% CI of LightGBM minus
logistic: -0.064 to +0.038); stability decided.

Where it is weak: drought claims (AUC 0.61 for drought vs 0.72-0.77 for hail and
frost) and seasons with an unusual mix of losses. Inside a crop group the ranking
is modest (mean AUC 0.59).

## Calibration
The model does not predict the season's claim level. Observed claim rates moved
between 7% and 32% across test safras; the mean prediction moved between 10% and 23%,
partly following the previous season: after the bad 2015/16 and 2021/22 seasons it
predicted 20% for 2016/17 (observed 7%) and 23% for 2022/23 (observed 10%). Brier
skill against the training base rate is about zero. Treat the probability
as a relative risk score; do not add it up to forecast a season's claims.

## Ethical and privacy notes
- Personal data (name, CPF/CNPJ) is dropped while the source CSV is streamed;
  property coordinates are dropped too. The repository has no policy-level data.
- The served bundle contains the fitted model and municipality-level aggregates built from
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
