# Data card

## Sources

| Data | Provider | Access | License | Used for |
|---|---|---|---|---|
| PSR policies (SISSER) 2006-2025 | MAPA, Coordenação-Geral de Seguro Rural | CKAN `sisser3`, 3 CSVs (files last modified 2025-09-03) | CC-BY | target, contract fields |
| Daily climate 1981-2026 | NASA POWER (MERRA-2 based), community AG | `api/temporal/daily/point`, one call per 0.5 x 0.625 degree cell | NASA open data | climatology and season weather |
| Oceanic Niño Index | NOAA CPC | `oni.ascii.txt` | public domain | ENSO state at contract |
| Municipality seat coordinates | kelvins/municipios-brasileiros (IBGE codes) | GitHub raw CSV | MIT | municipality -> climate cell |
| Hourly stations 2021-2022 (PR, SC, RS) | INMET | yearly zip of historical data | public | data-quality study only |

## Privacy (LGPD)

- Name (`NM_SEGURADO`) and CPF/CNPJ (`NR_DOCUMENTO_SEGURADO`) are dropped from each
  100k-row chunk while the CSV is streamed; the raw file with personal data is
  never written to disk. Insurer-side identifiers (`NR_PROPOSTA`, `NR_APOLICE`) and
  exact property coordinates are dropped too (data minimization).
- Policy-level data lives in `data/` (git-ignored). The repository holds code,
  aggregated reports and figures only; figures and tables are aggregated by safra,
  crop group, UF or decile.
- The API bundle holds aggregated tables at municipality level with at least 10
  policies behind each rate; smaller cells use the state rate.
- Tests run on a small synthetic fixture (`tests/fixtures/`, marked as synthetic).

## From raw rows to the modeling table

| Step | Rows removed | Rows left |
|---|---:|---:|
| Raw rows (3 files) | | 1,712,385 |
| Duplicated proposal id | 1 | 1,712,384 |
| Out-of-scope crop (livestock, forest, pasture) | 29,607 | 1,682,777 |
| Out-of-scope product (livestock, forest, revenue, parametric) | 411 | 1,682,366 |
| Missing municipality code or start date | 1,311 | 1,681,055 |
| Invalid money values (sum insured, premium or rate <= 0, rate >= 1) | 164 | 1,680,891 |
| Coverage shorter than 30 days or longer than 2 years (2016+) | 166 | 1,680,725 |
| Too Seguros 2020+ (file truncated, see below) | 9,106 | 1,671,619 |
| Safra not fully matured (2024/25 and later: claims not registered yet) | 139,160 | **1,532,459** |

Zero area (15k rows, mostly fruit), zero coverage level (8.5k) and zero yields are
missing-value codes: they become missing, the rows are kept. Full log:
`reports/cleaning_report.json`.

## Known problems in the source

1. **Truncated file.** The 2016-2024 CSV and XLSX have 1,048,565 data rows, 10 rows
   below Excel's limit. The file is sorted by insurer, and the last insurer's block
   (Too Seguros) stops in the middle of 2020. Too Seguros 2020+ is excluded; the
   other insurers come before the cut.
2. **Placeholder dates in 2006-2015.** Coverage start and end are 2016-07-22/23 for
   every row of the 2006-2015 file. The contract date (`DT_PROPOSTA`) is used
   instead (in 2016+ the median gap between the two is 0 days).
3. **Coverage dates are contractual.** Soy coverage often starts months before
   planting and lasts 365 days, so it does not mark the crop cycle. Climate windows
   follow a crop calendar instead (`croprisk/calendar.py`, DECISIONS D3).
4. **No claim date, claims cutoff.** Indemnities stop in early 2025: policies that
   start in December 2024 have 0.09% claims and the 2025 file has none. The cutoff
   is estimated at 2025-03-16 and a safra is used only when 98%+ of its policies
   ended their critical window 90+ days before it.
5. **No cancellation flag.** The SISSER holds policies that received the federal
   subsidy; there is no status column.
6. **Nominal money values.** Sum insured per hectare of soy went from R$695 (2006)
   to R$5,476 (2023). Values are used relative to recent medians (DECISIONS D12).

## Target

`y = 1` when `VALOR_INDENIZAÇÃO > 0`. Policies with a recorded event but zero
indemnity (111k: notified, not paid) are `y = 0`. `EVENTO_PREPONDERANTE` and the
indemnity are never features.

| Safra | Policies | Claim rate | Loss ratio (indemnity / premium) |
|---|---:|---:|---:|
| 2006/07 | 23,160 | 1.8% | 0.17 |
| 2007/08 | 33,596 | 6.2% | 0.38 |
| 2008/09 | 66,947 | 16.9% | 0.80 |
| 2009/10 | 65,977 | 6.5% | 0.28 |
| 2010/11 | 50,860 | 9.4% | 0.50 |
| 2011/12 | 61,861 | 22.4% | 0.93 |
| 2012/13 | 72,150 | 10.4% | 0.40 |
| 2013/14 | 114,454 | 12.0% | 0.55 |
| 2014/15 | 98,972 | 11.8% | 0.64 |
| 2015/16 | 42,060 | 20.5% | 0.95 |
| 2016/17 | 72,570 | 6.8% | 0.30 |
| 2017/18 | 69,475 | 15.0% | 0.50 |
| 2018/19 | 72,721 | 22.6% | 0.90 |
| 2019/20 | 108,194 | 17.6% | 0.71 |
| 2020/21 | 185,388 | 19.7% | 0.93 |
| 2021/22 | 199,286 | 31.8% | 1.79 |
| 2022/23 | 101,773 | 9.9% | 0.31 |
| 2023/24 | 93,015 | 17.4% | 0.39 |

Main causes among paid claims: drought 53%, hail 25%, frost 13%, excess rain 5%.

| Crop group | Policies | Claim rate |
|---|---:|---:|
| soy | 659,784 | 14.5% |
| second-crop corn | 220,232 | 24.2% |
| temperate fruit (grape, apple, peach...) | 195,566 | 24.5% |
| winter cereals (wheat, barley, oats...) | 141,798 | 20.0% |
| first-crop corn | 85,870 | 12.3% |
| coffee | 64,165 | 6.3% |
| rice | 55,476 | 7.0% |
| vegetables | 52,851 | 14.6% |
| sugarcane | 31,002 | 2.8% |
| other groups (5) | 25,715 | 7-29% |

## Biases and limits

- **Selection:** only subsidized policies (PSR). Unsubsidized insurance, uninsured
  farmers and the PROAGRO program are not in the data. Subsidy budgets changed a lot
  between years (42k policies in 2015/16, 199k in 2021/22), so the mix of crops,
  regions and insurers changes with policy, not only with risk.
- **Registration delay:** the last kept safra (2023/24) may still miss a few late
  claims; the gap sensitivity in the backtest bounds the effect.
- **Location:** climate is taken at the municipality seat's grid cell (about 55 x 70 km),
  not at the farm.
