# Archived analytical source from the authors' NIGER Colab notebook.
# Contains original Drive paths; run in original study environment.
# Do not run previous steps if the verified inputs already exist.

# ================================================================
# NIGER BASIN — REVIEWER 1 REVISION
# STEP 8: PRECIPITATION-BIAS DIAGNOSTICS, 1981–2005
#
# Uses saved Step 7 results.
# No source rasters are modified.
# ================================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

ROOT = Path("/content/drive/MyDrive")
BASE = ROOT / "Niger_Basin_Reviewer1_Revision"

STEP7 = BASE / "07_Annual_Climate_Validation"
STEP5 = BASE / "05_Grid_Coverage_QC"

OUT = BASE / "08_Precipitation_Bias_Diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

CELL_FILE = STEP7 / "Annual_Climate_Grid_Cell_1981_2005.csv"
GRID_FILE = STEP5 / "Main_Analysis_Grid_Coverage90.csv"
GRID_RASTER = STEP5 / "Niger_Grid_Coverage_QC.tif"

for p in [CELL_FILE, GRID_FILE, GRID_RASTER]:
    if not p.exists():
        raise FileNotFoundError(str(p))

cells = pd.read_csv(CELL_FILE)
grid = pd.read_csv(GRID_FILE)

required = {
    "product", "year", "grid_cell_id",
    "prcptot_all_days_mm", "wet_days_ge1mm",
    "wet_day_intensity_mm", "tx_mean_c",
    "tn_mean_c"
}

assert required.issubset(cells.columns)
assert cells["grid_cell_id"].notna().all()

def heading(name):
    print("\n" + "=" * 78)
    print(name)
    print("=" * 78)

def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = (
        np.isfinite(values) &
        np.isfinite(weights) &
        (weights > 0)
    )
    if not ok.any():
        return np.nan
    return float(np.average(values[ok], weights=weights[ok]))

# ---------------------------------------------------------------
# 1. Rebuild paired cell-year records
# ---------------------------------------------------------------

heading("STEP 8A — VERIFY PAIRED CELL-YEAR RECORDS")

e = cells.loc[cells["product"] == "ERA5"].copy()
c = cells.loc[cells["product"] == "CMCC"].copy()

keys = ["grid_cell_id", "year"]

assert not e.duplicated(keys).any()
assert not c.duplicated(keys).any()

paired = e.merge(
    c,
    on=keys,
    suffixes=("_ERA5", "_CMCC"),
    validate="one_to_one"
)

expected_rows = len(grid) * 25

print("Expected cell-year pairs:", expected_rows)
print("Found cell-year pairs:", len(paired))

assert len(paired) == expected_rows

variables = [
    "prcptot_all_days_mm",
    "wet_days_ge1mm",
    "wet_day_intensity_mm",
    "tx_mean_c",
    "tn_mean_c"
]

for variable in variables:
    paired[f"{variable}_bias"] = (
        paired[f"{variable}_CMCC"] -
        paired[f"{variable}_ERA5"]
    )

# Check whether annual rainfall values are plausible
# before examining differences.
for product in ["ERA5", "CMCC"]:
    col = f"prcptot_all_days_mm_{product}"

    print(
        product,
        "annual rainfall range:",
        round(paired[col].min(), 2),
        "to",
        round(paired[col].max(), 2),
        "mm"
    )

    assert (paired[col] >= 0).all()

# ---------------------------------------------------------------
# 2. Identify the extreme spatial rainfall-bias cells
# ---------------------------------------------------------------

heading("STEP 8B — LOCATE EXTREME PRECIPITATION BIASES")

evaluation = paired[
    paired["year"].between(1995, 2005)
].copy()

spatial = evaluation.groupby(
    "grid_cell_id", as_index=False
).agg(
    ERA5_rainfall_mm=(
        "prcptot_all_days_mm_ERA5", "mean"
    ),
    CMCC_rainfall_mm=(
        "prcptot_all_days_mm_CMCC", "mean"
    ),
    rainfall_bias_mm=(
        "prcptot_all_days_mm_bias", "mean"
    ),
    ERA5_wet_days=(
        "wet_days_ge1mm_ERA5", "mean"
    ),
    CMCC_wet_days=(
        "wet_days_ge1mm_CMCC", "mean"
    ),
    ERA5_wet_intensity=(
        "wet_day_intensity_mm_ERA5", "mean"
    ),
    CMCC_wet_intensity=(
        "wet_day_intensity_mm_CMCC", "mean"
    )
)

spatial = spatial.merge(
    grid[
        [
            "grid_cell_id", "grid_row", "grid_col",
            "longitude", "latitude",
            "basin_area_km2", "changed_fraction",
            "basin_coverage_fraction"
        ]
    ],
    on="grid_cell_id",
    validate="one_to_one"
)

spatial["relative_rainfall_bias_pct"] = np.where(
    spatial["ERA5_rainfall_mm"] > 0,
    100 * spatial["rainfall_bias_mm"] /
    spatial["ERA5_rainfall_mm"],
    np.nan
)

spatial["wet_day_bias"] = (
    spatial["CMCC_wet_days"] -
    spatial["ERA5_wet_days"]
)

spatial["wet_intensity_bias"] = (
    spatial["CMCC_wet_intensity"] -
    spatial["ERA5_wet_intensity"]
)

spatial.to_csv(
    OUT / "Spatial_Rainfall_Bias_Diagnostics.csv",
    index=False
)

columns = [
    "grid_cell_id", "longitude", "latitude",
    "basin_coverage_fraction",
    "ERA5_rainfall_mm", "CMCC_rainfall_mm",
    "rainfall_bias_mm", "relative_rainfall_bias_pct",
    "changed_fraction"
]

worst_negative = spatial.nsmallest(
    15, "rainfall_bias_mm"
)

worst_positive = spatial.nlargest(
    15, "rainfall_bias_mm"
)

print("\n15 MOST NEGATIVE RAINFALL BIASES")
print(worst_negative[columns].to_string(index=False))

print("\n15 MOST POSITIVE RAINFALL BIASES")
print(worst_positive[columns].to_string(index=False))

worst_negative.to_csv(
    OUT / "Most_Negative_Rainfall_Bias_Cells.csv",
    index=False
)

worst_positive.to_csv(
    OUT / "Most_Positive_Rainfall_Bias_Cells.csv",
    index=False
)

# ---------------------------------------------------------------
# 3. Inspect annual histories for problematic cells
# ---------------------------------------------------------------

heading("STEP 8C — INSPECT EXTREME CELL TIME SERIES")

extreme_ids = set(
    worst_negative.head(5)["grid_cell_id"]
) | set(
    worst_positive.head(5)["grid_cell_id"]
)

histories = paired[
    paired["grid_cell_id"].isin(extreme_ids)
].sort_values(
    ["grid_cell_id", "year"]
).copy()

histories.to_csv(
    OUT / "Extreme_Cell_Annual_Histories.csv",
    index=False
)

for cell_id in list(extreme_ids)[:5]:
    subset = histories[
        histories["grid_cell_id"] == cell_id
    ]

    loc = grid[
        grid["grid_cell_id"] == cell_id
    ].iloc[0]

    print(
        f"\nCell {cell_id}: "
        f"lon={loc.longitude:.3f}, "
        f"lat={loc.latitude:.3f}"
    )

    print(subset[
        [
            "year",
            "prcptot_all_days_mm_ERA5",
            "prcptot_all_days_mm_CMCC",
            "prcptot_all_days_mm_bias"
        ]
    ].to_string(index=False))

# ---------------------------------------------------------------
# 4. Examine rainfall frequency and intensity
# ---------------------------------------------------------------

heading("STEP 8D — RAINFALL FREQUENCY AND INTENSITY")

print("\n1995–2005 cell-level bias distributions:")
print(
    spatial[
        [
            "rainfall_bias_mm",
            "wet_day_bias",
            "wet_intensity_bias"
        ]
    ].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    ).to_string()
)

area = spatial["basin_area_km2"].to_numpy()

for name in [
    "rainfall_bias_mm",
    "wet_day_bias",
    "wet_intensity_bias"
]:
    print(
        name,
        "area-weighted bias:",
        round(
            weighted_mean(spatial[name], area),
            4
        )
    )

# ---------------------------------------------------------------
# 5. Compare period-dependent model biases
# ---------------------------------------------------------------

heading("STEP 8E — HISTORICAL PERIOD SENSITIVITY")

periods = [
    ("1981-1994", 1981, 1994),
    ("1995-2005", 1995, 2005),
    ("1981-2005", 1981, 2005)
]

period_records = []

for label, start, end in periods:

    subset = paired[
        paired["year"].between(start, end)
    ].copy()

    # First average by cell across years, then
    # area-weight the cells. Each year receives equal weight.
    per_cell = subset.groupby(
        "grid_cell_id", as_index=False
    )[
        [
            "prcptot_all_days_mm_ERA5",
            "prcptot_all_days_mm_CMCC",
            "wet_days_ge1mm_ERA5",
            "wet_days_ge1mm_CMCC",
            "wet_day_intensity_mm_ERA5",
            "wet_day_intensity_mm_CMCC"
        ]
    ].mean()

    per_cell = per_cell.merge(
        grid[["grid_cell_id", "basin_area_km2"]],
        on="grid_cell_id",
        validate="one_to_one"
    )

    w = per_cell["basin_area_km2"]

    era_r = weighted_mean(
        per_cell["prcptot_all_days_mm_ERA5"], w
    )

    cmcc_r = weighted_mean(
        per_cell["prcptot_all_days_mm_CMCC"], w
    )

    period_records.append({
        "period": label,
        "number_of_years": end - start + 1,
        "ERA5_rainfall_mm": era_r,
        "CMCC_rainfall_mm": cmcc_r,
        "rainfall_bias_mm": cmcc_r - era_r,
        "rainfall_relative_bias_pct": (
            100 * (cmcc_r - era_r) / era_r
        ),
        "ERA5_wet_days": weighted_mean(
            per_cell["wet_days_ge1mm_ERA5"], w
        ),
        "CMCC_wet_days": weighted_mean(
            per_cell["wet_days_ge1mm_CMCC"], w
        ),
        "ERA5_wet_intensity": weighted_mean(
            per_cell["wet_day_intensity_mm_ERA5"], w
        ),
        "CMCC_wet_intensity": weighted_mean(
            per_cell["wet_day_intensity_mm_CMCC"], w
        )
    })

period_df = pd.DataFrame(period_records)

period_df.to_csv(
    OUT / "Period_Dependent_Precipitation_Bias.csv",
    index=False
)

print(period_df.to_string(index=False))

# ---------------------------------------------------------------
# 6. Create a publication-resolution diagnostic figure
# ---------------------------------------------------------------

heading("STEP 8F — PRECIPITATION-BIAS MAP")

with rasterio.open(GRID_RASTER) as src:
    height, width = src.height, src.width
    bounds = src.bounds

bias_map = np.full(
    (height, width),
    np.nan,
    dtype=np.float32
)

r = spatial["grid_row"].astype(int).to_numpy()
c = spatial["grid_col"].astype(int).to_numpy()

bias_map[r, c] = spatial[
    "rainfall_bias_mm"
].to_numpy(dtype=np.float32)

extent = [
    bounds.left, bounds.right,
    bounds.bottom, bounds.top
]

# Symmetric color limits around zero, excluding
# the most extreme 1% solely for color scaling.
limit = np.nanpercentile(
    np.abs(bias_map), 99
)

fig, ax = plt.subplots(figsize=(11, 7))

im = ax.imshow(
    bias_map,
    extent=extent,
    origin="upper",
    cmap="RdBu_r",
    vmin=-limit,
    vmax=limit,
    interpolation="nearest"
)

ax.set_title(
    "CMCC-ESM2 minus ERA5 annual precipitation, 1995–2005"
)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.grid(
    linestyle=":",
    linewidth=0.5,
    alpha=0.3
)

cb = fig.colorbar(
    im, ax=ax, shrink=0.78
)
cb.set_label("Mean annual precipitation bias (mm/year)")

# Mark the most negative cell.
worst = worst_negative.iloc[0]

ax.scatter(
    worst["longitude"],
    worst["latitude"],
    s=100,
    facecolors="none",
    edgecolors="black",
    linewidths=1.5,
    zorder=5
)

fig.tight_layout()

fig.savefig(
    OUT / "Figure_Spatial_Rainfall_Bias_1995_2005.png",
    dpi=900,
    bbox_inches="tight"
)

fig.savefig(
    OUT / "Figure_Spatial_Rainfall_Bias_1995_2005.pdf",
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# ---------------------------------------------------------------
# 7. Final interpretation guardrails
# ---------------------------------------------------------------

metadata = {
    "study": "Niger Basin Reviewer 1 Revision",
    "step": 8,
    "evaluation_period": "1995-2005",
    "historical_archive": "1981-2005",
    "analysis_mask": "90% minimum basin coverage",
    "precipitation_units": (
        "Provisionally mm/day; source metadata absent"
    ),
    "bias_sign": "CMCC minus ERA5",
    "interpretation_limitations": [
        "Daily precipitation is not expected to be phase-aligned",
        "Historical rainfall biases can vary by period",
        "Climatological bias is not causal land-cover feedback",
        "Extreme negative cell biases require location review",
        "The 1992-2020 land-cover interval extends beyond "
        "the 1995-2005 climate evaluation period"
    ]
}

with open(
    OUT / "Step8_Metadata.json", "w"
) as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 78)
print("STEP 8 COMPLETE")
print("=" * 78)
print("Output directory:", OUT)
print("No original files were overwritten.")
