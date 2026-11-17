# Archived analytical source from the authors' NIGER Colab notebook.
# Contains original Drive paths; run in original study environment.
# Do not run previous steps if the verified inputs already exist.

# ======================================================================
# NIGER RIVER BASIN — REVIEWER 1 REVISION
# STEP 12: INDEPENDENT RECONSTRUCTION OF FIVE CLIMATE-EXTREME INDICES
# ======================================================================
#
# Study: Niger River Basin
# Evaluation period: 1995–2005
# Climate products:
#   1. ERA5
#   2. NEX-GDDP-CMIP6 / CMCC-ESM2
#
# Indices:
#   TXx     - Annual maximum of daily Tmax (°C)
#   TNx     - Annual maximum of daily Tmin (°C)
#   R10mm   - Annual number of days with PR >= 10 mm
#   PRCPTOT - Annual total precipitation on wet days (PR >= 1 mm)
#   CDD     - Maximum consecutive days with PR < 1 mm
#
# Bias definition:
#   Bias = CMCC-ESM2 - ERA5
#
# Statistical region:
#   Climate grid cells with >= 90% Niger Basin coverage
#
# Deliverables:
#   - 22 annual extreme-index files (CSV per product/year optional)
#   - One complete annual-index CSV
#   - One climatological-index comparison CSV
#   - Basin-wide area-weighted index summary
#   - Original-versus-recomputed index audit
#   - Latitude-zone bias table
#   - Five multiband spatial-bias GeoTIFF bands
#   - High-resolution spatial map and bias-comparison figures
#
# CAUTIONS:
#   - Precipitation is provisionally treated as mm/day.
#   - TX/TN are provisionally treated as degrees Celsius.
#   - The 1992–2020 land-cover transitions are NOT contemporaneous
#     with the full 1995–2005 climate-evaluation period.
#   - Climate-model discrepancies are NOT causal land-cover effects.
# ======================================================================

# ----------------------------------------------------------------------
# 0. IMPORTS
# ----------------------------------------------------------------------

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import json
import calendar
import warnings

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm

# ----------------------------------------------------------------------
# 1. DIRECTORIES AND CONFIGURATION
# ----------------------------------------------------------------------

ROOT = Path("/content/drive/MyDrive")

BASE = ROOT / "Niger_Basin_Reviewer1_Revision"

OUT = BASE / "12_Recomputed_Absolute_Extremes"
OUT.mkdir(parents=True, exist_ok=True)

GRID_CSV = (
    BASE /
    "05_Grid_Coverage_QC" /
    "Main_Analysis_Grid_Coverage90.csv"
)

ERA5_FOLDER = ROOT / "Niger_ERA5_Daily_1995_2005"
CMCC_FOLDER = ROOT / "Niger_CMCC_ESM2_Daily_1995_2005"

ERA5_ARCHIVED = (
    ROOT /
    "Niger_ERA5_Extremes_1995_2005" /
    "ERA5_Extremes_Climatology_1995_2005.tif"
)

CMCC_ARCHIVED = (
    CMCC_FOLDER /
    "CMCC_ESM2_Extremes_Climatology_1995_2005.tif"
)

START_YEAR = 1995
END_YEAR = 2005

PRODUCTS = ["ERA5", "CMCC"]

INDICES = [
    "TXx",
    "TNx",
    "R10mm",
    "PRCPTOT",
    "CDD"
]

INDEX_UNITS = {
    "TXx": "°C",
    "TNx": "°C",
    "R10mm": "days",
    "PRCPTOT": "mm/year",
    "CDD": "days"
}

BIAS_SIGN = "CMCC minus ERA5"

# No conversion factors are applied here.
# Step 6 found that the raw ranges are consistent with
# precipitation in mm/day and temperature in degrees Celsius.
# Original source-processing provenance should still be checked.

PRECIPITATION_FACTOR = 1.0
TEMPERATURE_OFFSET = 0.0

def heading(title):
    print("\n" + "=" * 82)
    print(title)
    print("=" * 82)

def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = (
        np.isfinite(values) &
        np.isfinite(weights) &
        (weights > 0)
    )

    if not valid.any():
        return np.nan

    return float(
        np.average(
            values[valid],
            weights=weights[valid]
        )
    )

# ----------------------------------------------------------------------
# 2. LOAD AND VERIFY THE ANALYSIS GRID
# ----------------------------------------------------------------------

heading("STEP 12A — ANALYSIS GRID VALIDATION")

if not GRID_CSV.exists():
    raise FileNotFoundError(
        f"Analysis grid not found: {GRID_CSV}"
    )

grid = pd.read_csv(GRID_CSV)

grid = grid.sort_values(
    "grid_cell_id"
).reset_index(drop=True)

required_grid_columns = {
    "grid_cell_id",
    "grid_row",
    "grid_col",
    "longitude",
    "latitude",
    "basin_area_km2",
    "changed_fraction"
}

missing_columns = (
    required_grid_columns - set(grid.columns)
)

if missing_columns:
    raise RuntimeError(
        "Missing grid columns: "
        + ", ".join(sorted(missing_columns))
    )

if grid["grid_cell_id"].duplicated().any():
    raise RuntimeError("Duplicate climate-grid cell IDs.")

grid_rows = grid["grid_row"].to_numpy(dtype=int)
grid_cols = grid["grid_col"].to_numpy(dtype=int)
grid_ids = grid["grid_cell_id"].to_numpy(dtype=int)
grid_weights = grid[
    "basin_area_km2"
].to_numpy(dtype=float)

if np.any(grid_weights <= 0):
    raise RuntimeError("Invalid grid-cell area weights.")

N_CELLS = len(grid)

print("Number of retained climate cells:", N_CELLS)
print("Evaluation period:", START_YEAR, "to", END_YEAR)
print("Indices:", ", ".join(INDICES))
print("Bias sign:", BIAS_SIGN)

# ----------------------------------------------------------------------
# 3. DAILY FILE PATHS
# ----------------------------------------------------------------------

def daily_path(product, variable, year):

    if product == "ERA5":
        return (
            ERA5_FOLDER /
            f"ERA5_{variable}_{year}.tif"
        )

    if product == "CMCC":
        return (
            CMCC_FOLDER /
            f"CMCC_{variable}_{year}.tif"
        )

    raise ValueError(f"Unknown product: {product}")

def read_daily(product, variable, year):

    path = daily_path(product, variable, year)

    if not path.exists():
        raise FileNotFoundError(str(path))

    with rasterio.open(path) as ds:

        expected_days = (
            366 if calendar.isleap(year) else 365
        )

        if ds.count != expected_days:
            raise RuntimeError(
                f"{path.name}: expected {expected_days} "
                f"daily bands, found {ds.count}"
            )

        if ds.crs.to_epsg() != 4326:
            raise RuntimeError(
                f"Unexpected CRS in {path.name}: {ds.crs}"
            )

        if np.max(grid_rows) >= ds.height:
            raise RuntimeError(
                "Grid row exceeds raster height."
            )

        if np.max(grid_cols) >= ds.width:
            raise RuntimeError(
                "Grid column exceeds raster width."
            )

        # Read each annual daily stack.
        raster = ds.read(masked=True)

        # Select only the validated basin grid cells.
        selected = raster[:, grid_rows, grid_cols]

        values = np.asarray(
            selected.filled(np.nan),
            dtype=np.float64
        )

        if values.shape != (
            expected_days,
            N_CELLS
        ):
            raise RuntimeError(
                f"Unexpected data shape: {values.shape}"
            )

        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Missing or invalid daily values: {path.name}"
            )

        return values

# ----------------------------------------------------------------------
# 4. CONSECUTIVE DRY-DAY CALCULATION
# ----------------------------------------------------------------------

def longest_dry_spell(precipitation):

    """
    Compute annual CDD for every climate-grid cell.

    A dry day is defined as precipitation < 1 mm/day.

    Runs are reset at the beginning of each calendar year.
    """

    dry = precipitation < 1.0

    n_days, n_cells = dry.shape

    current = np.zeros(
        n_cells,
        dtype=np.int32
    )

    maximum = np.zeros(
        n_cells,
        dtype=np.int32
    )

    for day in range(n_days):

        current = np.where(
            dry[day],
            current + 1,
            0
        )

        maximum = np.maximum(
            maximum,
            current
        )

    return maximum.astype(float)

# ----------------------------------------------------------------------
# 5. CALCULATE ANNUAL EXTREME INDICES
# ----------------------------------------------------------------------

heading("STEP 12B — RECOMPUTE ANNUAL EXTREME INDICES")

annual_frames = []

annual_quality = []

for product in PRODUCTS:

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        tx = read_daily(product, "TX", year)
        tn = read_daily(product, "TN", year)
        pr = read_daily(product, "PR", year)

        tx = tx + TEMPERATURE_OFFSET
        tn = tn + TEMPERATURE_OFFSET
        pr = pr * PRECIPITATION_FACTOR

        if np.any(pr < 0):
            raise RuntimeError(
                f"Negative precipitation: {product} {year}"
            )

        if np.any(tx < tn):
            raise RuntimeError(
                f"Tmax < Tmin: {product} {year}"
            )

        # ----------------------------------------------------------
        # TXx: Annual maximum daily Tmax
        # ----------------------------------------------------------

        txx = np.max(tx, axis=0)

        # ----------------------------------------------------------
        # TNx: Annual maximum daily Tmin
        # ----------------------------------------------------------

        tnx = np.max(tn, axis=0)

        # ----------------------------------------------------------
        # R10mm: Number of days with precipitation >=10 mm
        # ----------------------------------------------------------

        r10mm = np.sum(
            pr >= 10.0,
            axis=0
        ).astype(float)

        # ----------------------------------------------------------
        # PRCPTOT: Annual precipitation on wet days >=1 mm
        # ----------------------------------------------------------

        prcptot = np.sum(
            np.where(
                pr >= 1.0,
                pr,
                0.0
            ),
            axis=0,
            dtype=np.float64
        )

        # ----------------------------------------------------------
        # CDD: Maximum consecutive days with PR <1 mm
        # ----------------------------------------------------------

        cdd = longest_dry_spell(pr)

        # ----------------------------------------------------------
        # Quality checks
        # ----------------------------------------------------------

        all_day_total = pr.sum(
            axis=0,
            dtype=np.float64
        )

        if np.any(
            prcptot > all_day_total + 1e-5
        ):
            raise RuntimeError(
                "PRCPTOT exceeds total precipitation."
            )

        if np.any(
            cdd > pr.shape[0]
        ):
            raise RuntimeError(
                "CDD exceeds calendar-year length."
            )

        if np.any(
            r10mm > pr.shape[0]
        ):
            raise RuntimeError(
                "R10mm exceeds calendar-year length."
            )

        frame = pd.DataFrame({
            "product": product,
            "year": year,
            "grid_cell_id": grid_ids,
            "TXx": txx,
            "TNx": tnx,
            "R10mm": r10mm,
            "PRCPTOT": prcptot,
            "CDD": cdd
        })

        annual_frames.append(frame)

        annual_quality.append({
            "product": product,
            "year": year,
            "number_of_cells": N_CELLS,
            "number_of_days": pr.shape[0],
            "negative_precipitation_values": int(
                np.sum(pr < 0)
            ),
            "temperature_order_violations": int(
                np.sum(tx < tn)
            ),
            "missing_values": int(
                np.sum(~np.isfinite(tx)) +
                np.sum(~np.isfinite(tn)) +
                np.sum(~np.isfinite(pr))
            )
        })

        print(
            f"{product:4s} {year} | "
            f"TXx={weighted_mean(txx,grid_weights):.2f} °C | "
            f"TNx={weighted_mean(tnx,grid_weights):.2f} °C | "
            f"R10mm={weighted_mean(r10mm,grid_weights):.2f} d | "
            f"PRCPTOT={weighted_mean(prcptot,grid_weights):.2f} mm | "
            f"CDD={weighted_mean(cdd,grid_weights):.2f} d"
        )

annual = pd.concat(
    annual_frames,
    ignore_index=True
)

quality = pd.DataFrame(
    annual_quality
)

EXPECTED_ANNUAL_ROWS = (
    len(PRODUCTS) *
    (END_YEAR - START_YEAR + 1) *
    N_CELLS
)

assert len(annual) == EXPECTED_ANNUAL_ROWS

assert not annual.duplicated(
    ["product", "year", "grid_cell_id"]
).any()

annual.to_csv(
    OUT / "Recomputed_Annual_Extremes_1995_2005.csv",
    index=False
)

quality.to_csv(
    OUT / "Extreme_Index_Quality_Audit.csv",
    index=False
)

print("\nAnnual records:", len(annual))
print("Expected records:", EXPECTED_ANNUAL_ROWS)

# ----------------------------------------------------------------------
# 6. CLIMATOLOGICAL AVERAGE OF ANNUAL INDICES
# ----------------------------------------------------------------------

heading("STEP 12C — CLIMATOLOGICAL EXTREME INDICES")

climatology = annual.groupby(
    ["product", "grid_cell_id"],
    as_index=False
)[INDICES].mean()

era5 = climatology.loc[
    climatology["product"] == "ERA5"
].drop(columns="product")

cmcc = climatology.loc[
    climatology["product"] == "CMCC"
].drop(columns="product")

paired = era5.merge(
    cmcc,
    on="grid_cell_id",
    suffixes=("_ERA5", "_CMCC"),
    validate="one_to_one"
)

for index in INDICES:

    paired[f"{index}_bias"] = (
        paired[f"{index}_CMCC"] -
        paired[f"{index}_ERA5"]
    )

paired = paired.merge(
    grid[
        [
            "grid_cell_id",
            "grid_row",
            "grid_col",
            "longitude",
            "latitude",
            "basin_area_km2",
            "changed_fraction"
        ]
    ],
    on="grid_cell_id",
    validate="one_to_one"
)

paired.to_csv(
    OUT / "Recomputed_Climatological_Extreme_Biases.csv",
    index=False
)

print("Climatological climate cells:", len(paired))
print("Index variables:", INDICES)

# ----------------------------------------------------------------------
# 7. BASIN-WIDE AREA-WEIGHTED INDEX COMPARISON
# ----------------------------------------------------------------------

heading("STEP 12D — BASIN-WIDE EXTREME-INDEX BIASES")

summary_rows = []

for index in INDICES:

    w = paired["basin_area_km2"]

    era_mean = weighted_mean(
        paired[f"{index}_ERA5"],
        w
    )

    cmcc_mean = weighted_mean(
        paired[f"{index}_CMCC"],
        w
    )

    bias = weighted_mean(
        paired[f"{index}_bias"],
        w
    )

    summary_rows.append({
        "index": index,
        "unit": INDEX_UNITS[index],
        "ERA5_mean": era_mean,
        "CMCC_mean": cmcc_mean,
        "CMCC_minus_ERA5": bias,
        "direction": (
            "Positive" if bias > 0
            else "Negative" if bias < 0
            else "Zero"
        )
    })

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUT / "Five_Extreme_Index_Bias_Summary.csv",
    index=False
)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ----------------------------------------------------------------------
# 8. COMPARE RECOMPUTED INDICES WITH ARCHIVED INDICES
# ----------------------------------------------------------------------

heading("STEP 12E — ORIGINAL VERSUS RECOMPUTED INDICES")

audit_rows = []

archived_paths = {
    "ERA5": ERA5_ARCHIVED,
    "CMCC": CMCC_ARCHIVED
}

# Convert absent nodata tags carefully.
def read_archived_band(ds, index):

    descriptions = list(ds.descriptions)

    if index not in descriptions:
        return None

    band_number = descriptions.index(index) + 1

    arr = ds.read(
        band_number,
        masked=True
    )

    vals = arr[
        grid_rows,
        grid_cols
    ]

    return np.asarray(
        np.ma.filled(vals, np.nan),
        dtype=float
    )

for product, archived_path in archived_paths.items():

    if not archived_path.exists():

        print(
            f"Archive not found for {product}: "
            f"{archived_path}"
        )

        continue

    with rasterio.open(archived_path) as ds:

        print(
            f"\n{product} archived raster:"
        )
        print("Path:", archived_path)
        print("Bands:", ds.count)
        print("Descriptions:", ds.descriptions)

        if (
            np.max(grid_rows) >= ds.height or
            np.max(grid_cols) >= ds.width
        ):
            raise RuntimeError(
                "Original index raster does not match "
                "the target climate grid."
            )

        for index in INDICES:

            original = read_archived_band(
                ds,
                index
            )

            if original is None:

                print(
                    f"{product} {index}: "
                    "archived band unavailable"
                )

                continue

            revised = paired[
                f"{index}_{product}"
            ].to_numpy(dtype=float)

            valid = (
                np.isfinite(original) &
                np.isfinite(revised)
            )

            if not valid.any():
                print(
                    f"{product} {index}: "
                    "no valid paired cells"
                )
                continue

            difference = (
                revised[valid] -
                original[valid]
            )

            audit_rows.append({
                "product": product,
                "index": index,
                "n_cells": int(valid.sum()),
                "mean_difference": float(
                    np.mean(difference)
                ),
                "mean_absolute_difference": float(
                    np.mean(np.abs(difference))
                ),
                "max_absolute_difference": float(
                    np.max(np.abs(difference))
                ),
                "fraction_nearly_equal": float(
                    np.mean(
                        np.isclose(
                            revised[valid],
                            original[valid],
                            rtol=1e-5,
                            atol=1e-4
                        )
                    )
                )
            })

audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    OUT / "Original_vs_Recomputed_Extremes.csv",
    index=False
)

if not audit_df.empty:
    print(audit_df.to_string(index=False))
else:
    print("No archived rasters available for comparison.")

# ----------------------------------------------------------------------
# 9. LATITUDINAL CLIMATE-EXTREME BIAS ANALYSIS
# ----------------------------------------------------------------------

heading("STEP 12F — NORTH–SOUTH BIAS STRUCTURE")

paired["latitude_zone"] = pd.cut(
    paired["latitude"],
    bins=[-90, 10, 15, 20, 90],
    labels=[
        "Below 10°N",
        "10–15°N",
        "15–20°N",
        "Above 20°N"
    ]
)

zone_records = []

for zone, subset in paired.groupby(
    "latitude_zone",
    observed=True
):

    record = {
        "latitude_zone": str(zone),
        "n_cells": len(subset)
    }

    for index in INDICES:

        record[f"{index}_bias"] = weighted_mean(
            subset[f"{index}_bias"],
            subset["basin_area_km2"]
        )

    zone_records.append(record)

zone_df = pd.DataFrame(
    zone_records
)

zone_df.to_csv(
    OUT / "Latitude_Zone_Extreme_Biases.csv",
    index=False
)

print(
    zone_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

# ----------------------------------------------------------------------
# 10. CREATE SPATIAL CLIMATE-EXTREME BIAS GEOTIFF
# ----------------------------------------------------------------------

heading("STEP 12G — EXPORT FIVE-BAND BIAS GEOTIFF")

with rasterio.open(
    daily_path("ERA5", "TX", START_YEAR)
) as reference_grid:

    shape = (
        reference_grid.height,
        reference_grid.width
    )

    bounds = reference_grid.bounds

    profile = reference_grid.profile.copy()

    bias_stack = np.full(
        (len(INDICES), *shape),
        -9999.0,
        dtype=np.float32
    )

    for i, index in enumerate(INDICES):

        values = paired[
            f"{index}_bias"
        ].to_numpy(dtype=np.float32)

        bias_stack[
            i,
            grid_rows,
            grid_cols
        ] = values

    profile.update(
        driver="GTiff",
        count=len(INDICES),
        dtype="float32",
        nodata=-9999.0,
        compress="deflate",
        predictor=3
    )

    bias_raster_path = (
        OUT /
        "Niger_Five_Climate_Extreme_Biases_1995_2005.tif"
    )

    with rasterio.open(
        bias_raster_path,
        "w",
        **profile
    ) as dst:

        dst.write(bias_stack)

        for i, index in enumerate(
            INDICES,
            start=1
        ):
            dst.set_band_description(
                i,
                f"{index}_CMCC_minus_ERA5"
            )

print("Saved:", bias_raster_path)

# ----------------------------------------------------------------------
# 11. FIVE-PANEL PUBLICATION FIGURE
# ----------------------------------------------------------------------

heading("STEP 12H — PUBLICATION-QUALITY SPATIAL FIGURE")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "savefig.dpi": 900
})

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 10),
    constrained_layout=True
)

axes = axes.ravel()

extent = [
    bounds.left,
    bounds.right,
    bounds.bottom,
    bounds.top
]

for i, index in enumerate(INDICES):

    ax = axes[i]

    data = bias_stack[i].astype(float)

    data[data == -9999] = np.nan

    valid = np.isfinite(data)

    if not valid.any():
        ax.set_visible(False)
        continue

    limit = np.nanpercentile(
        np.abs(data),
        98
    )

    if not np.isfinite(limit) or limit == 0:
        limit = 1.0

    im = ax.imshow(
        np.ma.masked_invalid(data),
        extent=extent,
        origin="upper",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(
            vmin=-limit,
            vcenter=0,
            vmax=limit
        ),
        interpolation="nearest"
    )

    ax.set_title(
        f"({chr(97+i)}) {index}"
    )

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

    ax.grid(
        linestyle=":",
        linewidth=0.5,
        alpha=0.25
    )

    cb = fig.colorbar(
        im,
        ax=ax,
        shrink=0.75,
        pad=0.02
    )

    cb.set_label(
        f"Bias ({INDEX_UNITS[index]})"
    )

axes[-1].axis("off")

fig.suptitle(
    "CMCC-ESM2 minus ERA5 climate-extreme "
    "indices, Niger River Basin (1995–2005)",
    fontsize=15
)

fig.savefig(
    OUT /
    "Figure_Five_Extreme_Index_Spatial_Biases.png",
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    OUT /
    "Figure_Five_Extreme_Index_Spatial_Biases.pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)

# ----------------------------------------------------------------------
# 12. PUBLICATION-QUALITY BASIN-SUMMARY FIGURE
# ----------------------------------------------------------------------

heading("STEP 12I — EXTREME-INDEX SUMMARY FIGURE")

fig, axes = plt.subplots(
    1,
    len(INDICES),
    figsize=(16, 4.5),
    constrained_layout=True
)

for i, index in enumerate(INDICES):

    ax = axes[i]

    vals = [
        summary_df.loc[
            summary_df["index"] == index,
            "ERA5_mean"
        ].iloc[0],
        summary_df.loc[
            summary_df["index"] == index,
            "CMCC_mean"
        ].iloc[0]
    ]

    ax.bar(
        ["ERA5", "CMCC"],
        vals,
        color=["#326DA8", "#DD7F3C"],
        width=0.62
    )

    ax.set_title(index)

    ax.set_ylabel(
        INDEX_UNITS[index]
    )

    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.3
    )

    ax.set_axisbelow(True)

fig.suptitle(
    "Area-weighted annual climate extremes "
    "(1995–2005)",
    fontsize=14
)

fig.savefig(
    OUT /
    "Figure_Five_Extreme_Index_Basin_Comparison.png",
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    OUT /
    "Figure_Five_Extreme_Index_Basin_Comparison.pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)

# ----------------------------------------------------------------------
# 13. METADATA AND FINAL SUMMARY
# ----------------------------------------------------------------------

heading("STEP 12J — FINAL SCIENTIFIC AUDIT")

metadata = {
    "study": "Niger River Basin Reviewer 1 Revision",
    "step": 12,
    "evaluation_start": START_YEAR,
    "evaluation_end": END_YEAR,
    "grid_cells": int(N_CELLS),
    "analysis_grid": (
        "0.25-degree climate grid; "
        "minimum 90% basin coverage"
    ),
    "datasets": [
        "ERA5",
        "NEX-GDDP-CMIP6 CMCC-ESM2"
    ],
    "bias_definition": BIAS_SIGN,
    "indices": {
        "TXx": "Annual maximum daily Tmax (°C)",
        "TNx": "Annual maximum daily Tmin (°C)",
        "R10mm": "Annual days with rainfall >=10 mm",
        "PRCPTOT": (
            "Annual total rainfall on wet days >=1 mm"
        ),
        "CDD": (
            "Annual maximum consecutive days "
            "with rainfall <1 mm"
        )
    },
    "temperature_offset_applied": TEMPERATURE_OFFSET,
    "precipitation_factor_applied": PRECIPITATION_FACTOR,
    "units_status": (
        "Provisionally inferred from raster value ranges; "
        "source-processing provenance not yet verified"
    ),
    "cdd_year_boundary": (
        "Consecutive dry-day spell resets at January 1"
    ),
    "statistical_interpretation": (
        "Descriptive model-minus-reanalysis discrepancy. "
        "No causal land-cover inference."
    ),
    "archive_comparison_file": (
        "Original_vs_Recomputed_Extremes.csv"
    )
}

with open(
    OUT / "Step12_Metadata.json",
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=2
    )

print("\n" + "=" * 82)
print("STEP 12 COMPLETED SUCCESSFULLY")
print("=" * 82)

print("Climate-grid cells:", N_CELLS)
print("Annual records:", len(annual))
print("Indices reconstructed:", ", ".join(INDICES))
print("Period:", START_YEAR, "to", END_YEAR)
print("Output directory:", OUT)

print("\nMAIN OUTPUTS:")
print("1. Recomputed_Annual_Extremes_1995_2005.csv")
print("2. Recomputed_Climatological_Extreme_Biases.csv")
print("3. Five_Extreme_Index_Bias_Summary.csv")
print("4. Original_vs_Recomputed_Extremes.csv")
print("5. Latitude_Zone_Extreme_Biases.csv")
print("6. Niger_Five_Climate_Extreme_Biases_1995_2005.tif")
print("7. Figure_Five_Extreme_Index_Spatial_Biases.png")
print("8. Figure_Five_Extreme_Index_Basin_Comparison.png")

print("\nOriginal source rasters were not modified.")
