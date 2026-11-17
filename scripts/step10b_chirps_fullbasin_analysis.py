# Archived analytical source from the authors' NIGER Colab notebook.
# Contains original Drive paths; run in original study environment.
# Do not run previous steps if the verified inputs already exist.

# ================================================================
# NIGER BASIN — REVIEWER 1 REVISION
# STEP 10B: FULL-BASIN CHIRPS COMPARISON
#
# Run after the Step 10A Earth Engine export is complete.
# ================================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/content/drive/MyDrive")
BASE = ROOT / "Niger_Basin_Reviewer1_Revision"
OUT = BASE / "10_Full_Basin_CHIRPS_Validation"
OUT.mkdir(parents=True, exist_ok=True)

CHIRPS_NAME = "Niger_CHIRPS_FullGrid_1981_2005.csv"

# Earth Engine sometimes exports into the named Drive folder.
candidates = [
    ROOT / "Niger_Basin_Reviewer1_Revision" / CHIRPS_NAME,
    ROOT / CHIRPS_NAME
]

matches = [p for p in candidates if p.exists()]

if not matches:
    raise FileNotFoundError(
        "CHIRPS export CSV not yet found. "
        "Check that the Earth Engine task completed successfully."
    )

CHIRPS_CSV = matches[0]

ANNUAL_CSV = (
    BASE / "07_Annual_Climate_Validation" /
    "Annual_Climate_Grid_Cell_1981_2005.csv"
)

GRID_CSV = (
    BASE / "05_Grid_Coverage_QC" /
    "Main_Analysis_Grid_Coverage90.csv"
)

chirps_wide = pd.read_csv(CHIRPS_CSV)
annual = pd.read_csv(ANNUAL_CSV)
grid = pd.read_csv(GRID_CSV)

chirps_wide["grid_cell_id"] = pd.to_numeric(
    chirps_wide["grid_cell_id"],
    errors="raise"
).astype(int)

years = list(range(1981, 2006))
year_columns = [f"p_{year}" for year in years]

assert set(year_columns).issubset(chirps_wide.columns)
assert chirps_wide["grid_cell_id"].is_unique
assert len(chirps_wide) == len(grid)
assert set(chirps_wide.grid_cell_id) == set(grid.grid_cell_id)

chirps_long = chirps_wide.melt(
    id_vars="grid_cell_id",
    value_vars=year_columns,
    var_name="year",
    value_name="CHIRPS_annual_mm"
)

chirps_long["year"] = (
    chirps_long["year"]
    .str.replace("p_", "", regex=False)
    .astype(int)
)

chirps_long["CHIRPS_annual_mm"] = pd.to_numeric(
    chirps_long["CHIRPS_annual_mm"],
    errors="coerce"
)

if chirps_long["CHIRPS_annual_mm"].isna().any():
    raise RuntimeError("Missing CHIRPS values found.")

if (chirps_long["CHIRPS_annual_mm"] < 0).any():
    raise RuntimeError("Negative CHIRPS precipitation found.")

print("=" * 76)
print("STEP 10B — CHIRPS DATA INTEGRITY")
print("=" * 76)

print("Grid cells:", chirps_wide.grid_cell_id.nunique())
print("Years:", chirps_long.year.nunique())
print("Observations:", len(chirps_long))

# ---------------------------------------------------------------
# Join ERA5, CMCC and CHIRPS by year and grid cell
# ---------------------------------------------------------------

rain = annual[
    ["product", "year", "grid_cell_id",
     "prcptot_all_days_mm"]
].copy()

rain = rain.pivot(
    index=["grid_cell_id", "year"],
    columns="product",
    values="prcptot_all_days_mm"
).reset_index()

rain.columns.name = None

rain = rain.rename(
    columns={
        "ERA5": "ERA5_annual_mm",
        "CMCC": "CMCC_annual_mm"
    }
)

combined = rain.merge(
    chirps_long,
    on=["grid_cell_id", "year"],
    validate="one_to_one"
)

combined = combined.merge(
    grid[
        [
            "grid_cell_id", "longitude",
            "latitude", "basin_area_km2",
            "changed_fraction"
        ]
    ],
    on="grid_cell_id",
    validate="many_to_one"
)

expected = len(grid) * 25
assert len(combined) == expected

combined.to_csv(
    OUT / "CHIRPS_ERA5_CMCC_All_Grid_Years.csv",
    index=False
)

# ---------------------------------------------------------------
# Compare climatologies rather than synchronized daily weather
# ---------------------------------------------------------------

periods = {
    "1981–1994": (1981, 1994),
    "1995–2005": (1995, 2005),
    "1981–2005": (1981, 2005)
}

def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return np.average(values[ok], weights=weights[ok])

records = []
spatial_tables = []

for period_name, (start, end) in periods.items():

    part = combined[
        combined.year.between(start, end)
    ].copy()

    # Equal weighting across years within each cell.
    climate = part.groupby(
        "grid_cell_id",
        as_index=False
    )[
        [
            "CHIRPS_annual_mm",
            "ERA5_annual_mm",
            "CMCC_annual_mm"
        ]
    ].mean()

    climate = climate.merge(
        grid[
            [
                "grid_cell_id", "longitude",
                "latitude", "basin_area_km2",
                "changed_fraction"
            ]
        ],
        on="grid_cell_id",
        validate="one_to_one"
    )

    climate["ERA5_bias_mm"] = (
        climate["ERA5_annual_mm"] -
        climate["CHIRPS_annual_mm"]
    )

    climate["CMCC_bias_mm"] = (
        climate["CMCC_annual_mm"] -
        climate["CHIRPS_annual_mm"]
    )

    climate["period"] = period_name
    spatial_tables.append(climate)

    weights = climate.basin_area_km2

    for product in ["ERA5", "CMCC"]:

        diff = climate[f"{product}_bias_mm"]

        records.append({
            "period": period_name,
            "product": product,
            "grid_cells": len(climate),
            "CHIRPS_annual_mm": weighted_mean(
                climate.CHIRPS_annual_mm,
                weights
            ),
            "product_annual_mm": weighted_mean(
                climate[f"{product}_annual_mm"],
                weights
            ),
            "mean_bias_mm": weighted_mean(
                diff, weights
            ),
            "mean_absolute_error_mm": weighted_mean(
                np.abs(diff), weights
            ),
            "root_mean_square_error_mm": np.sqrt(
                weighted_mean(diff**2, weights)
            )
        })

spatial_df = pd.concat(spatial_tables, ignore_index=True)
metrics = pd.DataFrame(records)

metrics.to_csv(
    OUT / "CHIRPS_Full_Basin_Period_Metrics.csv",
    index=False
)

spatial_df.to_csv(
    OUT / "CHIRPS_Full_Basin_Spatial_Climatology.csv",
    index=False
)

print("\n" + "=" * 76)
print("STEP 10C — FULL-BASIN CHIRPS COMPARISON")
print("=" * 76)
print(metrics.to_string(index=False))

# ---------------------------------------------------------------
# Regional check: geographic dependence of errors
# ---------------------------------------------------------------

evaluation = spatial_df[
    spatial_df.period == "1995–2005"
].copy()

evaluation["latitude_zone"] = pd.cut(
    evaluation.latitude,
    bins=[-90, 10, 15, 20, 90],
    labels=[
        "South: below 10°N",
        "10–15°N",
        "15–20°N",
        "North: above 20°N"
    ]
)

regional_records = []

for zone, subset in evaluation.groupby(
    "latitude_zone",
    observed=True
):
    w = subset.basin_area_km2

    regional_records.append({
        "latitude_zone": str(zone),
        "grid_cells": len(subset),
        "CHIRPS_mm": weighted_mean(
            subset.CHIRPS_annual_mm, w
        ),
        "ERA5_bias_mm": weighted_mean(
            subset.ERA5_bias_mm, w
        ),
        "CMCC_bias_mm": weighted_mean(
            subset.CMCC_bias_mm, w
        ),
        "ERA5_MAE_mm": weighted_mean(
            subset.ERA5_bias_mm.abs(), w
        ),
        "CMCC_MAE_mm": weighted_mean(
            subset.CMCC_bias_mm.abs(), w
        )
    })

regional = pd.DataFrame(regional_records)

regional.to_csv(
    OUT / "CHIRPS_Latitudinal_Zone_Comparison.csv",
    index=False
)

print("\n" + "=" * 76)
print("STEP 10D — NORTH–SOUTH ERROR STRUCTURE")
print("=" * 76)
print(regional.to_string(index=False))

# ---------------------------------------------------------------
# Extreme-location consistency check
# ---------------------------------------------------------------

print("\n" + "=" * 76)
print("STEP 10E — CRITICAL CELLS RECHECK")
print("=" * 76)

critical = evaluation[
    evaluation.grid_cell_id.isin(
        [7639, 7750, 2189, 2190]
    )
]

print(
    critical[
        [
            "grid_cell_id", "latitude", "longitude",
            "ERA5_annual_mm", "CMCC_annual_mm",
            "CHIRPS_annual_mm",
            "ERA5_bias_mm", "CMCC_bias_mm"
        ]
    ].to_string(index=False)
)

print("\nSTEP 10 COMPLETE")
print("Saved results:", OUT)
print("Original climate datasets unchanged.")
