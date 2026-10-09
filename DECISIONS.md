# Decisions

Each entry: what was decided, why, and what was rejected. Numbers come from the
pipeline (`reports/cleaning_report.json`, `reports/metrics_summary.json`).

## Data

### D1. Personal data is dropped while streaming, before anything is written
The PSR CSVs contain the policyholder's name and CPF/CNPJ. `croprisk.data.psr`
streams each CSV over HTTP, parses it in chunks of 100k rows and drops the
personal columns from every chunk before writing parquet. The raw CSV never
touches the disk. The parser fails closed: an unexpected new column stops the
ingestion instead of being written, because it could be personal data.
- Also dropped: insurer-side identifiers (`NR_PROPOSTA`, `NR_APOLICE`), which the
  insurer can link back to a person, and the exact property coordinates
  (data minimization: climate is joined by municipality, so they are not needed).
- Kept: `ID_PROPOSTA` (MAPA system id) only for de-duplication, in `data/`, which
  is git-ignored.

### D2. The 2016-2024 file is truncated; the affected insurer is removed from 2020 on
The 2016-2024 CSV and XLSX both have exactly 1,048,565 data rows (the XLSX
sheet dimension is `A1:AL1048566`), 10 rows below Excel's 1,048,576 row limit.
The file is sorted by insurer name and, inside the last insurer's block (Too
Seguros), by year; the file ends in the middle of 2020 with increasing proposal
ids. Too Seguros has 6,773 policies in 2020 but only 55, 125 and 93 in
2021-2023. Conclusion: the file was cut at the Excel limit and the missing rows
are Too Seguros 2020-2023. Too Seguros policies from 2020 on (9,106 rows, 0.6%)
are excluded; earlier years are kept. Other insurers are complete: their blocks
come before the cut. This is reported as a limitation.

### D3. Coverage dates are contractual, so climate windows follow a crop calendar
In 2016+, `DT_INICIO_VIGENCIA` equals the contract date for over 75% of policies,
and 28% of soy policies start in April-July, months before planting. Coverage
often runs for 365 days (median for soy in MT: 365 days). Neither date marks the
crop's phenology. In 2006-2015 both dates are a placeholder (2016-07-22/23).
- Decision: each crop group gets a fixed critical window per macro-region
  (South = PR/SC/RS, Other), taken from CONAB crop calendars (`croprisk/calendar.py`):
  e.g. soy South Dec 1 + 105 days, second-crop corn Mar 15 + 107 days, wheat South
  Jul 1 + 122 days. Vegetables, planted all year, use contract + 30 days, 120 days.
- A policy gets the first window of which at least half falls after the contract
  date. Check on 2016+ policies (which have real coverage dates): the window falls
  inside the coverage period for 99% of soy, second-crop corn and rice policies,
  96% of temperate fruit and wheat, 85% of coffee.
- Coverage start `t0` = `DT_INICIO_VIGENCIA` in 2016+, `DT_PROPOSTA` in 2006-2015
  (median gap between the two in 2016+: 0 days).

### D4. Safra = Aug-Jul agricultural year of the window start
A calendar year would put summer soy (planted Sep-Dec, losses Jan-Mar) in a
different fold from the second-crop corn that suffers the same drought a few
weeks later. With the Aug-Jul year of the window start, summer crops, the
following second-crop corn and the winter wheat of the same season share a fold.

### D5. Target and censoring
- Target: `VALOR_INDENIZAÇÃO > 0`. 111k policies have an `EVENTO_PREPONDERANTE`
  but zero indemnity (notified, not paid): they are negatives.
- The files have no claim date. The 2025 file has no indemnity at all, and
  policies starting from Dec 2024 have almost none (0.09%): claims stop being
  registered in early 2025. The claims cutoff is estimated from the data as the
  99.5th percentile of the window end among paid policies (2025-03-16).
- A policy is *matured* if its window ended 90+ days before the cutoff. A safra is
  kept only if 98%+ of its policies are matured. Result: 2006/07 to 2023/24 are
  kept; 2024/25 (0.3% matured) is excluded. A partially observed safra would look
  like an unusually good year and bias the model and the evaluation.

### D6. Scope and odd records
- Out of scope: livestock, forest, pasture, revenue and parametric products
  (different triggers, not crop yield).
- Removed: duplicated proposal id, missing municipality code or start date,
  non-positive sum insured/premium/rate, rate >= 1, coverage shorter than 30 days
  or longer than 2 years.
- Kept as missing (not removed): zero area (15k, mostly fruit insured by plant),
  zero coverage level (8.5k, a missing-value code), zero yields. Coverage stored
  as a percent (55, 100) is rescaled; impossible values (1.43, 900) become missing.
- There is no cancellation flag in the public data. The SISSER only holds policies
  that received the federal subsidy, so cancelled proposals are not expected to be
  there; the money-value checks above catch the odd ones.
- Rare crops are grouped (`crop_group`, 14 groups) for calendars and features.

## Features

### D7. Climate source: NASA POWER per grid cell, not stations
NASA POWER (MERRA-2 based) is gap-free and covers every municipality; INMET
stations have long gaps (median station: 23% of days without a valid rain total in
2021-2022, see the INMET section of the README). Each municipality seat is snapped
to the nearest POWER node (0.5 x 0.625 degree); 1,304 cells cover Brazil, so there
is one request per cell, never one per policy. Series are cached in `data/raw/power/`.

### D8. Pre-season climate = 1981-2005 climatology of the critical window
The reference period ends before the first policy (2006), so the "normal" climate
of a location never contains the years being predicted (a unit test checks that
changing post-2005 weather does not change the climatology). Features: mean, CV and
10th percentile of window rain, drought frequency (rain < 70% of normal), typical
longest dry spell, max 5-day rain, hot days (Tmax >= 34 C), frost-risk days
(Tmin <= 3 C: a 2 m grid-cell air temperature rarely reaches 0 C even when there is
frost on the ground), temperature and root-zone soil moisture.

### D9. ENSO (ONI) at contract time is part of the pre-season block
The El Nino / La Nina state is public and known when the contract is signed, and La
Nina is the classic driver of droughts in southern Brazil. The value used is the
latest 3-month ONI published (season end + 10 days) before the contract date.
Caveat: NOAA revises ONI slightly when base periods are updated; this small
look-ahead is accepted and documented.

### D10. In-season = weather observed up to the middle of the critical window
The early-warning model sees the weather from the window start to its midpoint
(plus the 30 days before the window), as anomalies against the 1981-2005 normal of
exactly the same calendar window. A second variant uses the whole window
("end of window") as an upper reference; it is close to ex-post information and
is not presented as an early warning.

### D11. Portfolio history is computed as-of
Loss rates by municipality x crop group, UF x crop group and crop group use only
safras before the policy's safra (`s - 1 - gap`), with shrinkage toward the next
level (50 pseudo-policies). Municipality cells with fewer than 10 policies fall
back to the UF rate: this is both a variance rule and a privacy rule (no published
aggregate is based on fewer than 10 policies). Before the first safra there is no
history and the features are missing. A unit test caught an earlier version that
filled them with an all-years mean (a small leak into 2006/07 training rows); fixed
before any reported number.

### D12. Money and yield are relative to recent market medians
Soy sum insured per hectare grew from R$695 (2006) to R$5,476 (2023) in nominal
terms. Raw R$ values put every recent policy outside the training range (score
PSI 0.45 on the 2023/24 holdout in a first version). Sum insured per hectare and
expected yield are divided by the median of the same crop and UF over the previous
3 safras. These medians use contract fields only (no claims), so censored safras
can also serve as a reference. Area stays in hectares (no inflation).

### D13. The insurer's rate is not a feature of the main models
`PE_TAXA` (and the premium and subsidy derived from it) is the benchmark, so the
main models exclude it to keep the comparison fair. A separate variant adds it as
a feature to measure whether the model's information complements the price.

## Validation

### D14. Walk-forward by safra, expanding window
Test safras 2013/14 to 2023/24 (11 folds). For each, models are trained on all
earlier safras (expanding window) and scored on the test safra. Expanding is the
default because there are only 18 safras and weather-loss relations need as many
drought years as possible; a rolling 8-safra window is reported as sensitivity.
Random splits are never used: one drought hits thousands of policies in the same
season.

### D15. Label-maturity gap as sensitivity
At the start of safra s, some claims of safra s-1 (e.g. winter wheat) are still
open. The main backtest follows the usual "train on the past, test on the next
safra"; a variant trains on safras <= s-2 and uses history features with the same
gap, to show how much this matters.

### D16. Hyperparameters chosen once, on safras 2009-2012, by mean AUC
Selection happens on validation safras that precede every test safra. Single-year
early stopping was tried first and rejected: the pre-season ranking inverts in
drought years (validation AUC < 0.5 in 2011/12), and logloss is dominated by the
year's base rate, which no contract-time feature can know, so early stopping
stopped after 1-3 trees. The final protocol evaluates a small grid at fixed tree
counts (50-800) and picks the best mean AUC over 2009-2012, separately per
feature set.

### D17. Comparing with the insurer's rate
`PE_TAXA` is premium / sum insured: it prices frequency x severity + loadings,
while the target is frequency. Ranking metrics (AUC, KS, Gini) use the raw rate,
which is fair for "who is riskier". For Brier and calibration the rate is mapped to
a probability by a logistic fit of log(rate) on the training folds. The business
view is a double-lift table: policies sorted by model risk / rate-implied risk; if
the model knows something the price does not, the loss ratio rises across those
deciles.

## Production

### D18. The served model never sees the most recent matured safra
Holdout H = latest matured safra. The served pre-season model is trained on
safras <= H-1 and its holdout metrics are stored in the bundle. The monthly
retrain trains a challenger the same way and scores champion and challenger on the
same unseen H; the challenger replaces the champion only if AUC is not lower by
more than 0.002 and Brier is not higher by more than 0.001. Cost: the served model
lacks one safra (about 6% of the data). Benefit: the gate never compares on seen data.

### D19. Small infrastructure footprint on the free VM
Docker from the Ubuntu archive (`docker.io`), one image for API and retrain,
built on the VM (arm64). API container: 127.0.0.1 only, 1 CPU, 1.5 GB, rotated
logs, restart unless-stopped. Retrain container: low priority (nice 19, idle IO),
1.5 CPU, 6 GB, monthly on day 3 at 04:30 UTC. Nginx adds `/crop-risk/` with a
5 requests/s per-IP limit. The model bundle only contains aggregated tables
(municipality level, n >= 10) and the LightGBM trees, never policy rows.
