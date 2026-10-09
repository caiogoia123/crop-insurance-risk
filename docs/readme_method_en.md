## Method

### Data and target
- **Policies:** PSR/SISSER (MAPA), one row per subsidized crop policy, 2006-2025.
  Name and CPF/CNPJ are dropped from each chunk while the CSV is streamed, before
  anything is written. 1,532,459 policies remain after cleaning ([DATA_CARD](DATA_CARD.md)).
- **Target:** policy indemnified (`VALOR_INDENIZAÇÃO > 0`).
- **Censoring:** the public files stop registering claims in early 2025 (policies
  starting in December 2024 have 0.09% claims; the 2025 file has none). A safra is
  used only if 98%+ of its policies ended their risk window 90+ days before the
  estimated claims cutoff. 2024/25 is out; the last test safra is 2023/24.
- **One source problem found and handled:** the 2016-2024 file is cut at the Excel
  row limit (1,048,565 rows); the insurer at the end of the file loses its 2020-2023
  rows, so that insurer is excluded from 2020 on ([DECISIONS D2](DECISIONS.md)).

### When does the weather matter? A crop calendar
Coverage dates in the PSR are contractual: soy cover often starts months before
planting and lasts 365 days. So each crop group gets a **critical window** from CONAB
crop calendars (e.g. soy in the South: Dec 1 + 105 days; second-crop corn: Mar 15 +
107 days; wheat in the South: Jul 1 + 122 days), and each policy gets the first
window that starts after its contract. On 2016+ policies the window falls inside
the real coverage period for 99% of soy and second-crop corn.

The safra (fold) is the Aug-Jul year of the window start, so summer soy, the
following second-crop corn and winter wheat of the same season share a fold.

### Features
| Block | What | Known when |
|---|---|---|
| Contract | crop, UF, insurer, product type, area, coverage level, sum insured/ha and expected yield relative to the previous safra's median, contract month, days to the critical window | contract |
| Portfolio history | past loss rate of the municipality x crop group, UF x crop group, crop group (only earlier safras, shrunk toward the next level, n >= 10) | contract |
| Climatology | 1981-2005 statistics of the policy's own critical window at its NASA POWER cell: normal rain, variability, drought frequency, dry spells, max 5-day rain, hot days, frost-risk days, soil moisture | contract |
| ENSO | latest published Oceanic Niño Index and its 3-month change | contract |
| Season weather | rain, dry spells, heat, frost and soil-moisture **anomalies** from the window start to its midpoint, plus the 30 days before | mid-window |

### Validation
- **Walk-forward by safra:** for each test safra from 2013/14 to 2023/24, train on
  all earlier safras and score the test safra (11 folds). Never a random split.
- **Hyperparameters** chosen once, on safras 2009-2012 (before every test safra),
  by mean AUC over those years.
- **Leakage audit:** indemnity, cause of loss, premium and rate are never features
  of the main models (unit-tested); history is computed as-of; the climatology
  period ends before the first policy (unit-tested); in-season features stop at the
  middle of the critical window.
- **Sensitivity:** a one-safra gap between training and test (claims of the last
  safra may still be open when the next one is priced) and a rolling 8-safra window.
