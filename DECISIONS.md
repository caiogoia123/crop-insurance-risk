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
