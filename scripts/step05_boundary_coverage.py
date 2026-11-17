# Archived analytical source from the authors' NIGER Colab notebook.
# Contains original Drive paths; run in original study environment.
# Do not run previous steps if the verified inputs already exist.

# ================================================================
# NIGER RIVER BASIN — REVIEWER 1 REVISION
# STEP 5: CLIMATE-GRID COVERAGE AND BOUNDARY SENSITIVITY
# ================================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

ROOT = Path("/content/drive/MyDrive")

SOURCE = (
    ROOT / "Niger_Basin_Reviewer1_Revision" /
    "04_Basin_Adjusted_Fractions"
)

OUT = (
    ROOT / "Niger_Basin_Reviewer1_Revision" /
    "05_Grid_Coverage_QC"
)

OUT.mkdir(parents=True, exist_ok=True)

CSV = SOURCE / "Basin_Adjusted_Grid_Fractions.csv"
TRANSITIONS = SOURCE / "Basin_Adjusted_Transition_Fractions.csv"
RASTER = SOURCE / "Niger_Basin_Fractional_Change_1992_2020.tif"

for p in [CSV, TRANSITIONS, RASTER]:
    if not p.exists():
        raise FileNotFoundError(p)

df = pd.read_csv(CSV)
transitions = pd.read_csv(TRANSITIONS.csv if False else TRANSITIONS)

# ---------------------------------------------------------------
# 1. Calculate the area of each complete climate grid cell
# ---------------------------------------------------------------

with rasterio.open(RASTER) as src:

    transform = src.transform
    height = src.height
    width = src.width
    crs = src.crs
    profile = src.profile.copy()

    assert crs.to_epsg() == 4326

    EARTH_RADIUS = 6371008.8

    rows = df["grid_row"].to_numpy(dtype=np.int32)

    lat_top = transform.f + rows * transform.e
    lat_bottom = lat_top + transform.e

    full_cell_area = (
        EARTH_RADIUS**2 *
        np.deg2rad(abs(transform.a)) *
        np.abs(
            np.sin(np.deg2rad(lat_top)) -
            np.sin(np.deg2rad(lat_bottom))
        ) / 1e6
    )

    df["full_grid_area_km2"] = full_cell_area

    df["basin_coverage_fraction"] = (
        df["basin_area_km2"] /
        df["full_grid_area_km2"]
    )

    df["basin_coverage_fraction"] = (
        df["basin_coverage_fraction"].clip(0, 1)
    )

    # Recalculate without changing the original source CSV.
    df["changed_fraction_checked"] = (
        df["changed_area_km2"] /
        df["basin_area_km2"]
    )

    assert np.allclose(
        df["changed_fraction"],
        df["changed_fraction_checked"],
        rtol=1e-5,
        atol=1e-7
    ), "Saved fractional values do not reconcile."

    assert (
        df["changed_fraction_checked"].between(
            -1e-7, 1 + 1e-7
        )
    ).all()

    # -----------------------------------------------------------
    # 2. Coverage-threshold sensitivity
    # -----------------------------------------------------------

    thresholds = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95]

    records = []

    for threshold in thresholds:

        subset = df.loc[
            df["basin_coverage_fraction"] >= threshold
        ].copy()

        total_area = subset["basin_area_km2"].sum()
        changed_area = subset["changed_area_km2"].sum()

        records.append({
            "minimum_coverage": threshold,
            "number_of_cells": len(subset),
            "retained_basin_area_km2": total_area,
            "retained_changed_area_km2": changed_area,
            "changed_area_pct": (
                100 * changed_area / total_area
                if total_area > 0 else np.nan
            ),
            "cells_with_any_change": int(
                (subset["changed_area_km2"] > 0).sum()
            ),
            "cells_without_detected_change": int(
                (subset["changed_area_km2"] == 0).sum()
            )
        })

    sensitivity = pd.DataFrame(records)

    sensitivity.to_csv(
        OUT / "Boundary_Coverage_Sensitivity.csv",
        index=False
    )

    print("=" * 75)
    print("STEP 5A — BOUNDARY-COVERAGE SENSITIVITY")
    print("=" * 75)
    print(sensitivity.to_string(index=False))

    # -----------------------------------------------------------
    # 3. Main analysis mask
    # -----------------------------------------------------------

    MIN_COVERAGE = 0.90

    # This is a prespecified conservative screening choice,
    # not a statistical optimum.
    main = df.loc[
        df["basin_coverage_fraction"] >= MIN_COVERAGE
    ].copy()

    main.to_csv(
        OUT / "Main_Analysis_Grid_Coverage90.csv",
        index=False
    )

    print("\nMain analysis cells:", len(main))
    print(
        "Retained basin area:",
        round(main["basin_area_km2"].sum(), 2),
        "km²"
    )

    print(
        "Retained changed area:",
        round(main["changed_area_km2"].sum(), 2),
        "km²"
    )

    print(
        "Basin-wide changed share in retained cells:",
        round(
            100 * main["changed_area_km2"].sum() /
            main["basin_area_km2"].sum(),
            4
        ),
        "%"
    )

    # -----------------------------------------------------------
    # 4. Screen transition-level table using selected IDs
    # -----------------------------------------------------------

    selected_ids = set(
        main["grid_cell_id"].astype(int)
    )

    trans_main = transitions.loc[
        transitions["grid_cell_id"].isin(selected_ids)
    ].copy()

    trans_main.to_csv(
        OUT / "Main_Analysis_Transition_Fractions90.csv",
        index=False
    )

    comparison_area = trans_main[
        "transition_area_km2"
    ].sum()

    grid_area = main["changed_area_km2"].sum()

    print("\nTransition-area reconciliation:")
    print("Transition table:", comparison_area)
    print("Grid summary:", grid_area)
    print("Difference:", comparison_area - grid_area)

    assert np.isclose(
        comparison_area,
        grid_area,
        rtol=1e-8,
        atol=0.01
    )

    # -----------------------------------------------------------
    # 5. Produce grid-level quality-control rasters
    # -----------------------------------------------------------

    coverage_map = np.full(
        (height, width),
        -9999.0,
        dtype=np.float32
    )

    change_map = np.full_like(coverage_map, -9999.0)

    r = df["grid_row"].astype(int).to_numpy()
    c = df["grid_col"].astype(int).to_numpy()

    coverage_map[r, c] = df[
        "basin_coverage_fraction"
    ].to_numpy(dtype=np.float32)

    change_map[r, c] = df[
        "changed_fraction_checked"
    ].to_numpy(dtype=np.float32)

    qa_raster = OUT / "Niger_Grid_Coverage_QC.tif"

    profile.update(
        driver="GTiff",
        count=2,
        dtype="float32",
        nodata=-9999,
        compress="deflate"
    )

    with rasterio.open(qa_raster, "w", **profile) as dst:
        dst.write(coverage_map, 1)
        dst.write(change_map, 2)
        dst.set_band_description(
            1, "BasinCoverageFraction"
        )
        dst.set_band_description(
            2, "ChangedFractionWithinBasin"
        )

    # -----------------------------------------------------------
    # 6. Create scientific quality-control figure
    # -----------------------------------------------------------

    masked_coverage = np.ma.masked_equal(
        coverage_map, -9999
    )

    masked_change = np.ma.masked_equal(
        change_map, -9999
    )

    # Visualization only; source raster unchanged.
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.5),
        constrained_layout=True
    )

    extent = [
        src.bounds.left,
        src.bounds.right,
        src.bounds.bottom,
        src.bounds.top
    ]

    a = axes[0].imshow(
        masked_coverage,
        extent=extent,
        origin="upper",
        vmin=0, vmax=1,
        cmap="viridis"
    )

    axes[0].set_title(
        "(a) Niger Basin coverage per climate cell"
    )

    b = axes[1].imshow(
        masked_change,
        extent=extent,
        origin="upper",
        vmin=0, vmax=1,
        cmap="magma"
    )

    axes[1].set_title(
        "(b) Changed land-cover fraction, 1992–2020"
    )

    for ax in axes:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.grid(
            alpha=0.2, linestyle=":", linewidth=0.5
        )

    fig.colorbar(
        a, ax=axes[0],
        label="Basin coverage fraction",
        shrink=0.75
    )

    fig.colorbar(
        b, ax=axes[1],
        label="Changed fraction of basin-covered cell",
        shrink=0.75
    )

    fig.savefig(
        OUT / "Figure_Grid_Coverage_Audit.png",
        dpi=900, bbox_inches="tight"
    )

    fig.savefig(
        OUT / "Figure_Grid_Coverage_Audit.pdf",
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

# ---------------------------------------------------------------
# 7. Store reproducibility metadata
# ---------------------------------------------------------------

metadata = {
    "step": 5,
    "main_coverage_threshold": 0.90,
    "thresholds_tested": thresholds,
    "denominator": (
        "Basin-covered land area approximated using "
        "300-m source-pixel centre inclusion"
    ),
    "full_cell_area": (
        "Spherical latitude-dependent area "
        "for 0.25-degree climate cells"
    ),
    "time_period_of_land_cover": "1992-2020",
    "climate_bias_period": "1995-2005",
    "interpretation": (
        "Fractional land-cover changes are descriptive "
        "exposure indicators, not causal effects."
    )
}

with open(OUT / "Step5_Metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 75)
print("STEP 5 COMPLETE")
print("=" * 75)
print("Output directory:", OUT)
print("Original source data unchanged.")
