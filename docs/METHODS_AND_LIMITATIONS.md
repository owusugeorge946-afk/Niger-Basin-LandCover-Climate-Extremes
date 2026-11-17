# Methods, provenance and interpretation notes

## Inputs, grids and periods

- 1992–2020 land-cover transitions, fractional mapping to approximately 0.25° climate cells; main analytic mask: >=90% basin coverage.
- 2,661 main climate cells, rather than the older modal-classification count of 2,836.
- Climate evaluation: 1995–2005; 1981–1994 independent percentile-reference sensitivity and 1981–2005 overlapping-reference sensitivity.
- CHIRPS precipitation independently evaluates mean precipitation for a historical climate comparison, not causal effects.

## Quantities checked independently

| Index | ERA5 | CMCC-ESM2 | CMCC minus ERA5 | Units |
|---|---:|---:|---:|---|
| TXx | 42.1011 | 43.5882 | +1.4872 | °C |
| TNx | 29.9412 | 29.3411 | -0.6001 | °C |
| R10mm | 15.8225 | 28.9142 | +13.0917 | days/year |
| PRCPTOT | 493.1748 | 669.5048 | +176.3300 | mm/year |
| CDD | 127.9037 | 120.7919 | -7.1117 | days |

Weighted estimates above summarize the full 1995–2005 basin evaluation, from the archived validated report. They are **checks**, not data encoded into the figure-generation pipeline.

## Geographic confounding

Per ten-percentage-point change in land-cover fraction, the CMCC minus ERA5 rainfall-bias association fell from about +27.417 mm/year (unadjusted) to +8.030 mm/year (geography + CHIRPS adjustment); the adjusted 95% bootstrap CI crosses zero. Interpret spatial associations as confounded, not causal land–atmosphere feedbacks.

## Percentile baseline sensitivity

| Index | 1981–1994 independent | 1981–2005 overlapping | Units |
|---|---:|---:|---|
| TX90p | -2.7523 | -1.0094 | percentage points |
| TN90p | -6.4828 | -2.8383 | percentage points |
| WSDI | -2.6449 | +1.3223 | days/year |
| CSDI | +4.9934 | +2.4891 | days/year |

WSDI bias changes sign with reference period. The independent baseline is too short for conventional 30-year climatology and the overlapping reference was not corrected using ETCCDI in-base bootstrap.

## Source unit checks

- ERA5 daily TX/TN: Kelvin minus 273.15 = °C; precipitation: daily m × 1000 = mm/day.
- NASA NEX-GDDP-CMIP6 `tasmax`/`tasmin`: Kelvin minus 273.15; `pr` in kg m-2 s-1 × 86400 = mm/day.
- ERA5-Land accumulated net solar/thermal radiation divided by month seconds -> W/m², positive downward; sensible and latent components sign-flipped for upward-positive representation; monthly climatologies averaged with equal month weights.

## Scope and caveats

Reanalysis climatologies and statistically downscaled climate data do not identify a controlled response to land-cover change. Land-cover transitions from 1992–2020 extend beyond the 1995–2005 climate evaluation. No atmospheric moisture budget, circulation, or land-use perturbation simulation was executed. Do not use causal language. No event-based causal inference is provided.
