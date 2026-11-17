# Archived analytical source from the authors' NIGER Colab notebook.
# Contains original Drive paths; run in original study environment.
# Do not run previous steps if the verified inputs already exist.

# ===============================================================
# NIGER RIVER BASIN — REVIEWER 1 REVISION
# STEP 7: FULL-RECORD CLIMATE CONSISTENCY AUDIT, 1981–2005
# ===============================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import calendar
import json
import numpy as np
import pandas as pd
import rasterio

ROOT = Path("/content/drive/MyDrive")
BASE = ROOT / "Niger_Basin_Reviewer1_Revision"

OUT = BASE / "07_Annual_Climate_Validation"
OUT.mkdir(parents=True, exist_ok=True)

GRID_CSV = (
    BASE / "05_Grid_Coverage_QC" /
    "Main_Analysis_Grid_Coverage90.csv"
)

grid = pd.read_csv(GRID_CSV).sort_values("grid_cell_id")
rows = grid["grid_row"].to_numpy(dtype=int)
cols = grid["grid_col"].to_numpy(dtype=int)
weights = grid["basin_area_km2"].to_numpy(dtype=float)
ids = grid["grid_cell_id"].to_numpy(dtype=int)

assert len(ids) == len(set(ids))
assert np.all(weights > 0)

def source_path(product, variable, year):
    period = "1981_1994" if year <= 1994 else "1995_2005"
    folder = (
        f"Niger_ERA5_Daily_{period}"
        if product == "ERA5"
        else f"Niger_CMCC_ESM2_Daily_{period}"
    )
    prefix = "ERA5" if product == "ERA5" else "CMCC"
    return ROOT / folder / f"{prefix}_{variable}_{year}.tif"

def read_daily(product, variable, year):
    path = source_path(product, variable, year)

    with rasterio.open(path) as ds:
        assert ds.crs.to_epsg() == 4326
        assert (ds.height, ds.width) == (79, 111)

        expected_days = 366 if calendar.isleap(year) else 365
        assert ds.count == expected_days

        # Read only selected basin climate cells.
        # Dataset is small enough for a full annual stack.
        data = ds.read(masked=True).astype("float64")
        selected = data[:, rows, cols]

        values = np.asarray(
            selected.filled(np.nan),
            dtype=np.float64
        )

    return values

def weighted_mean(values, areas):
    valid = np.isfinite(values) & np.isfinite(areas)
    if not np.any(valid):
        return np.nan
    return float(np.average(
        values[valid], weights=areas[valid]
    ))

def safe_total(values, condition):
    # Sum only if all calendar days are valid.
    return np.where(
        condition,
        np.sum(
            np.where(np.isfinite(values), values, 0),
            axis=0
        ),
        np.nan
    )

all_cell_records = []
all_annual_records = []
quality_records = []

print("=" * 78)
print("STEP 7A — COMPLETE DAILY CLIMATE VALIDATION")
print("=" * 78)

for product in ["ERA5", "CMCC"]:

    for year in range(1981, 2006):

        tx = read_daily(product, "TX", year)
        tn = read_daily(product, "TN", year)
        pr = read_daily(product, "PR", year)

        assert tx.shape == tn.shape == pr.shape

        n_days, n_cells = tx.shape

        valid_tx = np.isfinite(tx)
        valid_tn = np.isfinite(tn)
        valid_pr = np.isfinite(pr)

        valid_temperature = valid_tx & valid_tn

        negative_pr = valid_pr & (pr < -1e-6)
        inverted_temp = (
            valid_temperature & (tx < tn)
        )

        tx_complete = valid_tx.all(axis=0)
        tn_complete = valid_tn.all(axis=0)
        pr_complete = valid_pr.all(axis=0)

        # Do not silently clip negative precipitation.
        pr_physically_valid = (
            pr_complete &
            (~negative_pr.any(axis=0))
        )

        tx_mean = np.where(
            tx_complete,
            np.nanmean(tx, axis=0),
            np.nan
        )

        tn_mean = np.where(
            tn_complete,
            np.nanmean(tn, axis=0),
            np.nan
        )

        pr_total = safe_total(
            pr, pr_physically_valid
        )

        # Wet-day threshold is 1 mm/day,
        # conditional on the provisional raw-mm interpretation.
        wet = (pr >= 1.0) & valid_pr

        wet_days = np.where(
            pr_physically_valid,
            wet.sum(axis=0),
            np.nan
        )

        wet_amount = np.sum(
            np.where(wet, pr, 0),
            axis=0
        )

        wet_intensity = np.divide(
            wet_amount,
            wet_days,
            out=np.full(n_cells, np.nan),
            where=wet_days > 0
        )

        annual = pd.DataFrame({
            "product": product,
            "year": year,
            "grid_cell_id": ids,
            "grid_row": rows,
            "grid_col": cols,
            "longitude": grid["longitude"].to_numpy(),
            "latitude": grid["latitude"].to_numpy(),
            "basin_area_km2": weights,
            "tx_mean_c": tx_mean,
            "tn_mean_c": tn_mean,
            "prcptot_all_days_mm": pr_total,
            "wet_days_ge1mm": wet_days,
            "wet_day_intensity_mm": wet_intensity,
            "tx_complete": tx_complete,
            "tn_complete": tn_complete,
            "pr_complete": pr_complete,
            "pr_nonnegative": pr_physically_valid
        })

        all_cell_records.append(annual)

        all_annual_records.append({
            "product": product,
            "year": year,
            "tx_mean_c": weighted_mean(
                tx_mean, weights
            ),
            "tn_mean_c": weighted_mean(
                tn_mean, weights
            ),
            "precipitation_mm": weighted_mean(
                pr_total, weights
            ),
            "wet_days_ge1mm": weighted_mean(
                wet_days, weights
            ),
            "wet_day_intensity_mm": weighted_mean(
                wet_intensity, weights
            )
        })

        quality_records.append({
            "product": product,
            "year": year,
            "days": n_days,
            "cells": n_cells,
            "missing_tx_values": int(
                (~valid_tx).sum()
            ),
            "missing_tn_values": int(
                (~valid_tn).sum()
            ),
            "missing_pr_values": int(
                (~valid_pr).sum()
            ),
            "negative_pr_values": int(
                negative_pr.sum()
            ),
            "tx_below_tn_values": int(
                inverted_temp.sum()
            ),
            "fully_complete_cells": int(
                (
                    tx_complete &
                    tn_complete &
                    pr_complete
                ).sum()
            )
        })

        print(
            f"{product:4s} {year}: "
            f"rainfall={all_annual_records[-1]['precipitation_mm']:.1f} mm, "
            f"TX={all_annual_records[-1]['tx_mean_c']:.2f} °C, "
            f"TN={all_annual_records[-1]['tn_mean_c']:.2f} °C"
        )

# ===============================================================
# STEP 7B — SAVE FULL GRID-CELL AND ANNUAL RECORDS
# ===============================================================

cells = pd.concat(
    all_cell_records, ignore_index=True
)

annual = pd.DataFrame(all_annual_records)
quality = pd.DataFrame(quality_records)

cells.to_csv(
    OUT / "Annual_Climate_Grid_Cell_1981_2005.csv",
    index=False
)

annual.to_csv(
    OUT / "Annual_Basin_Climate_1981_2005.csv",
    index=False
)

quality.to_csv(
    OUT / "Annual_Data_Quality_1981_2005.csv",
    index=False
)

print("\n" + "=" * 78)
print("STEP 7B — DATA-QUALITY CHECK")
print("=" * 78)

print(quality.groupby("product")[
    [
        "missing_tx_values",
        "missing_tn_values",
        "missing_pr_values",
        "negative_pr_values",
        "tx_below_tn_values"
    ]
].sum().to_string())

# ===============================================================
# STEP 7C — COMPARE BIAS AND EVALUATION PERIODS
# ===============================================================

print("\n" + "=" * 78)
print("STEP 7C — ERA5 / CMCC ANNUAL COMPARISON")
print("=" * 78)

e = annual[annual["product"] == "ERA5"].copy()
c = annual[annual["product"] == "CMCC"].copy()

comparison = e.merge(
    c,
    on="year",
    suffixes=("_ERA5", "_CMCC"),
    validate="one_to_one"
)

for variable in [
    "tx_mean_c",
    "tn_mean_c",
    "precipitation_mm",
    "wet_days_ge1mm",
    "wet_day_intensity_mm"
]:
    comparison[f"{variable}_difference"] = (
        comparison[f"{variable}_CMCC"] -
        comparison[f"{variable}_ERA5"]
    )

comparison["precipitation_relative_bias_pct"] = (
    100 *
    (
        comparison["precipitation_mm_CMCC"] -
        comparison["precipitation_mm_ERA5"]
    ) /
    comparison["precipitation_mm_ERA5"]
)

comparison.to_csv(
    OUT / "Annual_ERA5_CMCC_Comparison.csv",
    index=False
)

periods = {
    "Early historical 1981-1994": (1981, 1994),
    "Original evaluation 1995-2005": (1995, 2005),
    "Full archive 1981-2005": (1981, 2005)
}

period_results = []

for period, (start, end) in periods.items():

    subset = comparison[
        comparison["year"].between(start, end)
    ]

    period_results.append({
        "period": period,
        "years": len(subset),
        "ERA5_annual_rainfall_mm": subset[
            "precipitation_mm_ERA5"
        ].mean(),
        "CMCC_annual_rainfall_mm": subset[
            "precipitation_mm_CMCC"
        ].mean(),
        "rainfall_mean_bias_mm": subset[
            "precipitation_mm_difference"
        ].mean(),
        "rainfall_relative_bias_pct": (
            100 *
            (
                subset["precipitation_mm_CMCC"].mean() -
                subset["precipitation_mm_ERA5"].mean()
            ) /
            subset["precipitation_mm_ERA5"].mean()
        ),
        "tx_mean_difference_c": subset[
            "tx_mean_c_difference"
        ].mean(),
        "tn_mean_difference_c": subset[
            "tn_mean_c_difference"
        ].mean(),
        "wet_days_difference": subset[
            "wet_days_ge1mm_difference"
        ].mean()
    })

period_df = pd.DataFrame(period_results)

period_df.to_csv(
    OUT / "Historical_Period_Comparison.csv",
    index=False
)

print(period_df.to_string(index=False))

# ===============================================================
# STEP 7D — SPATIAL CLIMATOLOGICAL BIAS
# ===============================================================

print("\n" + "=" * 78)
print("STEP 7D — SPATIAL CLIMATOLOGICAL BIASES")
print("=" * 78)

eval_cells = cells[
    cells["year"].between(1995, 2005)
]

spatial = eval_cells.groupby(
    ["product", "grid_cell_id"],
    as_index=False
)[
    [
        "tx_mean_c",
        "tn_mean_c",
        "prcptot_all_days_mm",
        "wet_days_ge1mm"
    ]
].mean()

sp_e = spatial[
    spatial["product"] == "ERA5"
].drop(columns="product")

sp_c = spatial[
    spatial["product"] == "CMCC"
].drop(columns="product")

spatial_comparison = sp_e.merge(
    sp_c,
    on="grid_cell_id",
    suffixes=("_ERA5", "_CMCC"),
    validate="one_to_one"
)

for variable in [
    "tx_mean_c",
    "tn_mean_c",
    "prcptot_all_days_mm",
    "wet_days_ge1mm"
]:
    spatial_comparison[f"{variable}_bias"] = (
        spatial_comparison[f"{variable}_CMCC"] -
        spatial_comparison[f"{variable}_ERA5"]
    )

spatial_comparison = spatial_comparison.merge(
    grid[
        [
            "grid_cell_id",
            "longitude",
            "latitude",
            "basin_area_km2",
            "changed_fraction"
        ]
    ],
    on="grid_cell_id",
    validate="one_to_one"
)

spatial_comparison.to_csv(
    OUT / "Spatial_Climate_Bias_1995_2005.csv",
    index=False
)

print(
    spatial_comparison[
        [
            "tx_mean_c_bias",
            "tn_mean_c_bias",
            "prcptot_all_days_mm_bias",
            "wet_days_ge1mm_bias"
        ]
    ].describe().to_string()
)

# ===============================================================
# STEP 7E — FINAL STATUS
# ===============================================================

status = {
    "years": [1981, 2005],
    "analysis_mask": ">=90% Niger Basin coverage",
    "grid_cells": len(grid),
    "products": ["ERA5", "CMCC-ESM2"],
    "variables": ["TX", "TN", "PR"],
    "precipitation_assumption": (
        "Raw GeoTIFF precipitation values provisionally "
        "interpreted as mm/day; original metadata absent."
    ),
    "temperature_assumption": (
        "Raw TX and TN values provisionally interpreted "
        "as degrees Celsius."
    ),
    "analysis_status": "exploratory unit-consistency audit",
    "caution": (
        "Do not treat daily weather correlation as "
        "a climate-model skill score."
    )
}

with open(OUT / "Step7_Metadata.json", "w") as f:
    json.dump(status, f, indent=2)

print("\nSTEP 7 COMPLETE")
print("Saved to:", OUT)
print("No source files modified.")
