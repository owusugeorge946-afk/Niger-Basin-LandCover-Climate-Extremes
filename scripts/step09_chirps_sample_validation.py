# Archived analytical source from the authors' NIGER Colab notebook.
# Contains original Drive paths; run in original study environment.
# Do not run previous steps if the verified inputs already exist.

# ================================================================
# NIGER RIVER BASIN — REVIEWER 1 REVISION
# STEP 9 — CORRECTED CHIRPS PRECIPITATION BENCHMARK
#
# Fix: ONE multiband reduceRegions operation instead of
#      25 concurrent annual regional aggregations.
#
# Earth Engine project: nana213
# Period: 1981–2005
# Evaluation: 1995–2005
# Selected sample: 15 most positive + 15 most negative
#                  CMCC-ESM2 minus ERA5 precipitation cells
#
# All original data remain unchanged.
# ================================================================

from google.colab import drive
drive.mount("/content/drive")

import ee
import json
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from pathlib import Path

# ================================================================
# 1. INITIALIZE EARTH ENGINE
# ================================================================

PROJECT_ID = "nana213"

try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate(auth_mode="notebook")
    ee.Initialize(project=PROJECT_ID)

print("=" * 78)
print("STEP 9A — EARTH ENGINE INITIALIZED")
print("=" * 78)
print("Project:", PROJECT_ID)

# ================================================================
# 2. DEFINE SOURCE FILES
# ================================================================

ROOT = Path("/content/drive/MyDrive")
BASE = ROOT / "Niger_Basin_Reviewer1_Revision"

STEP8 = BASE / "08_Precipitation_Bias_Diagnostics"
STEP7 = BASE / "07_Annual_Climate_Validation"
STEP5 = BASE / "05_Grid_Coverage_QC"

OUT = BASE / "09_CHIRPS_Independent_Check"
OUT.mkdir(parents=True, exist_ok=True)

SPATIAL = STEP8 / "Spatial_Rainfall_Bias_Diagnostics.csv"
ANNUAL = STEP7 / "Annual_Climate_Grid_Cell_1981_2005.csv"
GRID = STEP5 / "Niger_Grid_Coverage_QC.tif"

for p in [SPATIAL, ANNUAL, GRID]:
    if not p.exists():
        raise FileNotFoundError(str(p))

spatial = pd.read_csv(SPATIAL)
annual = pd.read_csv(ANNUAL)

print("\nSource files loaded.")
print("Spatial records:", len(spatial))
print("Annual records:", len(annual))

# ================================================================
# 3. SELECT 30 EXTREME-BIAS GRID CELLS
# ================================================================

negative = spatial.nsmallest(15, "rainfall_bias_mm")
positive = spatial.nlargest(15, "rainfall_bias_mm")

selected = (
    pd.concat([negative, positive], ignore_index=True)
    .drop_duplicates("grid_cell_id")
    .copy()
)

selected["grid_cell_id"] = selected[
    "grid_cell_id"
].astype(int)

selected["bias_group"] = np.where(
    selected["rainfall_bias_mm"] < 0,
    "Negative CMCC-ERA5 bias",
    "Positive CMCC-ERA5 bias"
)

assert len(selected) == 30, (
    f"Expected 30 distinct cells; found {len(selected)}"
)

selected.to_csv(
    OUT / "Selected_Extreme_Bias_Cells.csv",
    index=False
)

print("\n" + "=" * 78)
print("STEP 9B — SELECTED CLIMATE CELLS")
print("=" * 78)
print("Number of cells:", len(selected))

# ================================================================
# 4. BUILD ORIGINAL 0.25-DEGREE CLIMATE-CELL GEOMETRIES
# ================================================================

with rasterio.open(GRID) as src:
    gt = src.transform
    raster_height = src.height
    raster_width = src.width

features = []

for rec in selected.itertuples(index=False):

    r = int(rec.grid_row)
    c = int(rec.grid_col)

    assert 0 <= r < raster_height
    assert 0 <= c < raster_width

    left = gt.c + c * gt.a
    right = left + gt.a

    top = gt.f + r * gt.e
    bottom = top + gt.e

    geometry = ee.Geometry.Rectangle(
        [left, bottom, right, top],
        proj="EPSG:4326",
        geodesic=False
    )

    features.append(
        ee.Feature(
            geometry,
            {"grid_cell_id": int(rec.grid_cell_id)}
        )
    )

regions = ee.FeatureCollection(features)

print("\nConstructed 30 grid-cell geometries.")

# ================================================================
# 5. PREPARE ANNUAL CHIRPS IMAGES
# ================================================================

print("\n" + "=" * 78)
print("STEP 9C — BUILD MULTIBAND CHIRPS IMAGE")
print("=" * 78)

CHIRPS_ID = "UCSB-CHG/CHIRPS/DAILY"

chirps = ee.ImageCollection(
    CHIRPS_ID
).select("precipitation")

years = list(range(1981, 2006))
annual_images = []

for year in years:

    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"

    annual_total = (
        chirps
        .filterDate(start, end)
        .sum()
        .rename(f"p_{year}")
        .toFloat()
    )

    annual_images.append(annual_total)

# Combine all 25 annual precipitation images
# into one image with 25 distinct bands.
#
# This does NOT perform any regional aggregation.
# The aggregation is performed ONCE in Step 6.

multiband = ee.Image.cat(*annual_images)

expected_bands = [f"p_{year}" for year in years]

print("Expected annual bands:", len(expected_bands))
print("Period:", years[0], "to", years[-1])

# ================================================================
# 6. EXECUTE ONE REGION AGGREGATION
# ================================================================

print("\n" + "=" * 78)
print("STEP 9D — SINGLE CHIRPS REGIONAL EXTRACTION")
print("=" * 78)

print("Computing 25 annual fields for 30 grid cells.")
print("Using ONE reduceRegions operation.")

# One spatial reduction, instead of 25 reductions.
region_results = multiband.reduceRegions(
    collection=regions,
    reducer=ee.Reducer.mean(),
    scale=5566,
    crs="EPSG:4326",
    tileScale=4
)

# Only one server request for regional aggregation.
try:
    response = region_results.getInfo()

except ee.EEException as exc:

    message = str(exc)

    print("\nEarth Engine response:")
    print(message[:1500])

    if (
        "Too many concurrent aggregations" in message
        or "429" in message
    ):
        raise RuntimeError(
            "Earth Engine is still limiting aggregations. "
            "The corrected script now uses only one regional "
            "reduction; check whether other Earth Engine jobs "
            "under the same project are running, then rerun "
            "this cell after those jobs finish."
        ) from exc

    raise

features_out = response["features"]

print("Returned grid-cell features:", len(features_out))

assert len(features_out) == len(selected), (
    "Mismatch in returned climate-cell count."
)

# ================================================================
# 7. CONVERT MULTIBAND RESULTS TO LONG-FORM CSV
# ================================================================

print("\n" + "=" * 78)
print("STEP 9E — ORGANIZE ANNUAL CHIRPS RESULTS")
print("=" * 78)

records = []

for feature in features_out:

    props = feature["properties"]

    cell_id = int(props["grid_cell_id"])

    for year in years:

        band = f"p_{year}"
        value = props.get(band)

        records.append({
            "grid_cell_id": cell_id,
            "year": year,
            "CHIRPS_annual_mm": (
                float(value)
                if value is not None
                else np.nan
            )
        })

chirps_df = pd.DataFrame(records)

assert len(chirps_df) == 30 * 25

if chirps_df.duplicated(
    ["grid_cell_id", "year"]
).any():
    raise RuntimeError("Duplicate CHIRPS records.")

if chirps_df["CHIRPS_annual_mm"].isna().any():

    missing = chirps_df[
        chirps_df["CHIRPS_annual_mm"].isna()
    ]

    print(missing.to_string(index=False))

    raise RuntimeError(
        "Missing CHIRPS annual values."
    )

if (chirps_df["CHIRPS_annual_mm"] < 0).any():
    raise RuntimeError(
        "Negative CHIRPS annual totals identified."
    )

chirps_df = chirps_df.sort_values(
    ["grid_cell_id", "year"]
).reset_index(drop=True)

chirps_df.to_csv(
    OUT / "CHIRPS_Annual_1981_2005.csv",
    index=False
)

print("Valid annual CHIRPS records:", len(chirps_df))
print("Unique cells:", chirps_df.grid_cell_id.nunique())
print("Unique years:", chirps_df.year.nunique())

print("\nCHIRPS annual-total distribution (mm/year):")
print(chirps_df["CHIRPS_annual_mm"].describe())

# ================================================================
# 8. JOIN CHIRPS TO ORIGINAL ERA5 AND CMCC RESULTS
# ================================================================

print("\n" + "=" * 78)
print("STEP 9F — THREE-DATASET CLIMATE COMPARISON")
print("=" * 78)

rainfall = annual.loc[
    annual["grid_cell_id"].isin(
        selected["grid_cell_id"]
    ),
    [
        "product",
        "grid_cell_id",
        "year",
        "prcptot_all_days_mm"
    ]
].copy()

wide = rainfall.pivot(
    index=["grid_cell_id", "year"],
    columns="product",
    values="prcptot_all_days_mm"
).reset_index()

wide.columns.name = None

wide = wide.rename(
    columns={
        "ERA5": "ERA5_annual_mm",
        "CMCC": "CMCC_annual_mm"
    }
)

required = {
    "ERA5_annual_mm",
    "CMCC_annual_mm"
}

assert required.issubset(wide.columns)

merged = wide.merge(
    chirps_df,
    on=["grid_cell_id", "year"],
    how="inner",
    validate="one_to_one"
)

assert len(merged) == 750

merged = merged.merge(
    selected[
        [
            "grid_cell_id",
            "longitude",
            "latitude",
            "rainfall_bias_mm",
            "bias_group"
        ]
    ],
    on="grid_cell_id",
    validate="many_to_one"
)

merged["ERA5_minus_CHIRPS_mm"] = (
    merged["ERA5_annual_mm"] -
    merged["CHIRPS_annual_mm"]
)

merged["CMCC_minus_CHIRPS_mm"] = (
    merged["CMCC_annual_mm"] -
    merged["CHIRPS_annual_mm"]
)

merged.to_csv(
    OUT / "ERA5_CMCC_CHIRPS_Annual_Comparison.csv",
    index=False
)

print("Matched records:", len(merged))

# ================================================================
# 9. HISTORICAL CLIMATOLOGY COMPARISONS
# ================================================================

print("\n" + "=" * 78)
print("STEP 9G — HISTORICAL CLIMATOLOGY BENCHMARK")
print("=" * 78)

periods = {
    "1981-1994": (1981, 1994),
    "1995-2005": (1995, 2005),
    "1981-2005": (1981, 2005)
}

results = []

for label, (start, end) in periods.items():

    period_data = merged.loc[
        merged["year"].between(start, end)
    ]

    climatology = period_data.groupby(
        "grid_cell_id",
        as_index=False
    )[
        [
            "ERA5_annual_mm",
            "CMCC_annual_mm",
            "CHIRPS_annual_mm"
        ]
    ].mean()

    climatology["ERA5_bias_vs_CHIRPS_mm"] = (
        climatology["ERA5_annual_mm"] -
        climatology["CHIRPS_annual_mm"]
    )

    climatology["CMCC_bias_vs_CHIRPS_mm"] = (
        climatology["CMCC_annual_mm"] -
        climatology["CHIRPS_annual_mm"]
    )

    ea = climatology[
        "ERA5_bias_vs_CHIRPS_mm"
    ].abs()

    ca = climatology[
        "CMCC_bias_vs_CHIRPS_mm"
    ].abs()

    climatology["closer_product"] = np.where(
        ea < ca,
        "ERA5",
        np.where(
            ca < ea,
            "CMCC",
            "Tie"
        )
    )

    climatology["period"] = label
    results.append(climatology)

benchmark = pd.concat(
    results,
    ignore_index=True
)

benchmark = benchmark.merge(
    selected[
        [
            "grid_cell_id",
            "longitude",
            "latitude",
            "bias_group"
        ]
    ],
    on="grid_cell_id",
    validate="many_to_one"
)

benchmark.to_csv(
    OUT / "CHIRPS_Climatological_Benchmark.csv",
    index=False
)

evaluation = benchmark.loc[
    benchmark["period"] == "1995-2005"
].copy()

print("\n1995–2005 benchmark:")
print(
    evaluation[
        [
            "grid_cell_id",
            "longitude",
            "latitude",
            "ERA5_annual_mm",
            "CMCC_annual_mm",
            "CHIRPS_annual_mm",
            "closer_product"
        ]
    ].sort_values("grid_cell_id").to_string(index=False)
)

print("\nClosest dataset to CHIRPS:")
print(
    evaluation["closer_product"]
    .value_counts()
    .to_string()
)

# ================================================================
# 10. INSPECT THE FOUR CRITICAL LOCATIONS
# ================================================================

print("\n" + "=" * 78)
print("STEP 9H — CRITICAL PRECIPITATION CELLS")
print("=" * 78)

critical_ids = [7639, 2189, 2190, 7750]

for cell_id in critical_ids:

    result = evaluation.loc[
        evaluation["grid_cell_id"] == cell_id
    ]

    if result.empty:
        print(f"Cell {cell_id} unavailable.")
        continue

    print("\nGRID CELL:", cell_id)

    print(
        result[
            [
                "longitude",
                "latitude",
                "ERA5_annual_mm",
                "CMCC_annual_mm",
                "CHIRPS_annual_mm",
                "ERA5_bias_vs_CHIRPS_mm",
                "CMCC_bias_vs_CHIRPS_mm",
                "closer_product"
            ]
        ].to_string(index=False)
    )

# ================================================================
# 11. CREATE HIGH-RESOLUTION BENCHMARK FIGURE
# ================================================================

print("\n" + "=" * 78)
print("STEP 9I — BENCHMARK FIGURE")
print("=" * 78)

plot_df = evaluation.sort_values(
    "CHIRPS_annual_mm"
).copy()

x = np.arange(len(plot_df))
bar_width = 0.27

fig, ax = plt.subplots(
    figsize=(16, 7)
)

ax.bar(
    x - bar_width,
    plot_df["ERA5_annual_mm"],
    width=bar_width,
    label="ERA5"
)

ax.bar(
    x,
    plot_df["CMCC_annual_mm"],
    width=bar_width,
    label="CMCC-ESM2"
)

ax.bar(
    x + bar_width,
    plot_df["CHIRPS_annual_mm"],
    width=bar_width,
    label="CHIRPS"
)

ax.set_xticks(x)

ax.set_xticklabels(
    plot_df["grid_cell_id"].astype(str),
    rotation=90
)

ax.set_xlabel("Climate-grid cell ID")

ax.set_ylabel(
    "Mean annual precipitation (mm/year)"
)

ax.set_title(
    "Independent precipitation benchmark: "
    "extreme-bias climate cells (1995–2005)"
)

ax.legend(frameon=False)

ax.grid(
    axis="y",
    linestyle=":",
    alpha=0.3
)

ax.set_axisbelow(True)

fig.tight_layout()

fig.savefig(
    OUT / "Figure_CHIRPS_Benchmark_1995_2005.png",
    dpi=900,
    bbox_inches="tight"
)

fig.savefig(
    OUT / "Figure_CHIRPS_Benchmark_1995_2005.pdf",
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# ================================================================
# 12. SAVE REPRODUCIBILITY METADATA
# ================================================================

metadata = {
    "study": "Niger Basin Reviewer 1 Revision",
    "step": 9,
    "project": PROJECT_ID,
    "dataset": CHIRPS_ID,
    "period": "1981-2005",
    "evaluation": "1995-2005",
    "selected_cells": len(selected),
    "annual_records": len(chirps_df),
    "aggregation_count": 1,
    "spatial_method": (
        "CHIRPS mean annual precipitation within each "
        "full 0.25-degree climate-grid rectangle, "
        "sampled at approximately 5.6 km."
    ),
    "limitations": [
        "Selected cells are extremes of the ERA5-CMCC bias",
        "The full climate-cell rectangle is sampled",
        "CHIRPS is a diagnostic comparator, not ground truth",
        "Independent gauge validation remains desirable"
    ]
}

with open(OUT / "Step9_Metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 78)
print("STEP 9 COMPLETED")
print("=" * 78)

print("Project:", PROJECT_ID)
print("Saved outputs:", OUT)
print("Original data unchanged.")
