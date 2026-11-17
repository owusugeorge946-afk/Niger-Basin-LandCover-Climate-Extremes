# Archived analytical source from the authors' NIGER Colab notebook.
# Contains original Drive paths; run in original study environment.
# Do not run previous steps if the verified inputs already exist.

# ================================================================
# NIGER RIVER BASIN — REVIEWER 1 REVISION
# STEP 4: BASIN-ADJUSTED FRACTIONAL LAND-COVER TRANSITIONS
# ================================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import pandas as pd
import rasterio
import fiona

from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.features import geometry_mask
from rasterio.enums import Resampling

ROOT = Path("/content/drive/MyDrive")

LCC = ROOT / "Niger_LCC_1992_2020"
CLIMATE = ROOT / "Niger_ERA5_Daily_1995_2005"

OUT = (
    ROOT / "Niger_Basin_Reviewer1_Revision" /
    "04_Basin_Adjusted_Fractions"
)
OUT.mkdir(parents=True, exist_ok=True)

LOSS = LCC / "Niger_LCC_Loss_1992_2020.tif"
GAIN = LCC / "Niger_LCC_Gain_1992_2020.tif"
BOUNDARY = LCC / "Niger_River_Basin_Boundary.shp"
GRID = CLIMATE / "ERA5_TX_1995.tif"

for p in [LOSS, GAIN, BOUNDARY, GRID]:
    if not p.exists():
        raise FileNotFoundError(p)

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

# ---------------------------------------------------------------
# 1. Read basin boundary
# ---------------------------------------------------------------

section("STEP 4A — BASIN BOUNDARY VALIDATION")

with fiona.open(BOUNDARY) as src:
    geometries = [
        feature["geometry"]
        for feature in src
        if feature["geometry"] is not None
    ]
    boundary_crs = src.crs_wkt or src.crs

print("Boundary geometries:", len(geometries))
print("Boundary CRS:", boundary_crs)

if not geometries:
    raise RuntimeError("No valid basin geometries found")

# ---------------------------------------------------------------
# 2. Open land-cover and climate grids
# ---------------------------------------------------------------

with rasterio.open(LOSS) as loss_ds, \
     rasterio.open(GAIN) as gain_ds, \
     rasterio.open(GRID) as grid_ds:

    assert loss_ds.crs == gain_ds.crs
    assert loss_ds.transform == gain_ds.transform
    assert loss_ds.shape == gain_ds.shape

    assert loss_ds.crs == grid_ds.crs
    assert rasterio.crs.CRS.from_user_input(
        boundary_crs
    ) == loss_ds.crs, (
        "Boundary CRS differs from source raster CRS."
    )

    assert loss_ds.crs.to_epsg() == 4326

    source_transform = loss_ds.transform
    grid_transform = grid_ds.transform

    gh, gw = grid_ds.height, grid_ds.width
    n_cells = gh * gw

    R = 6371008.8  # Mean spherical Earth radius, metres

    # Approximate latitude-dependent pixel areas.
    def row_areas_km2(transform, rows):
        top = transform.f + rows * transform.e
        bottom = top + transform.e

        return (
            R**2 *
            np.deg2rad(abs(transform.a)) *
            np.abs(
                np.sin(np.deg2rad(top)) -
                np.sin(np.deg2rad(bottom))
            ) / 1e6
        )

    basin_area = np.zeros(n_cells, dtype=np.float64)
    changed_area = np.zeros(n_cells, dtype=np.float64)

    transitions = defaultdict(float)

    total_basin_pixels = 0
    total_changed_pixels = 0
    unpaired_pixels = 0
    equal_positive_pixels = 0

    section("STEP 4B — PIXEL-LEVEL BASIN MASKING")

    CHUNK = 512

    for row0 in range(0, loss_ds.height, CHUNK):

        h = min(CHUNK, loss_ds.height - row0)

        for col0 in range(0, loss_ds.width, CHUNK):

            w = min(CHUNK, loss_ds.width - col0)

            window = Window(col0, row0, w, h)
            wt = window_transform(window, source_transform)

            # Basin pixels are selected by pixel centre.
            inside_basin = geometry_mask(
                geometries,
                out_shape=(h, w),
                transform=wt,
                invert=True,
                all_touched=False
            )

            if not inside_basin.any():
                continue

            rr, cc = np.nonzero(inside_basin)

            global_rows = row0 + rr
            global_cols = col0 + cc

            lon = (
                source_transform.c +
                (global_cols + 0.5) * source_transform.a
            )

            lat = (
                source_transform.f +
                (global_rows + 0.5) * source_transform.e
            )

            gc = np.floor(
                (lon - grid_transform.c) /
                grid_transform.a
            ).astype(np.int32)

            gr = np.floor(
                (lat - grid_transform.f) /
                grid_transform.e
            ).astype(np.int32)

            valid_grid = (
                (gr >= 0) & (gr < gh) &
                (gc >= 0) & (gc < gw)
            )

            if not valid_grid.any():
                continue

            rr = rr[valid_grid]
            cc = cc[valid_grid]
            gr = gr[valid_grid]
            gc = gc[valid_grid]

            global_rows = global_rows[valid_grid]

            flat = gr * gw + gc

            weights = row_areas_km2(
                source_transform, global_rows
            )

            # Denominator: basin-covered pixel area.
            basin_area += np.bincount(
                flat,
                weights=weights,
                minlength=n_cells
            )

            total_basin_pixels += len(flat)

            a = loss_ds.read(1, window=window)[rr, cc]
            b = gain_ds.read(1, window=window)[rr, cc]

            va = (a >= 1) & (a <= 9)
            vb = (b >= 1) & (b <= 9)

            unpaired_pixels += int(np.sum(va ^ vb))

            equal_positive_pixels += int(
                np.sum(va & vb & (a == b))
            )

            changed = va & vb & (a != b)

            if not changed.any():
                continue

            total_changed_pixels += int(changed.sum())

            changed_flat = flat[changed]
            changed_weights = weights[changed]

            changed_area += np.bincount(
                changed_flat,
                weights=changed_weights,
                minlength=n_cells
            )

            origin = a[changed].astype(np.int16)
            destination = b[changed].astype(np.int16)

            codes = origin * 100 + destination

            combined = (
                changed_flat.astype(np.int64) * 1000 +
                codes
            )

            keys, inverse = np.unique(
                combined, return_inverse=True
            )

            areas = np.bincount(
                inverse,
                weights=changed_weights,
                minlength=len(keys)
            )

            for key, area in zip(keys, areas):
                cell = int(key // 1000)
                transition = int(key % 1000)
                transitions[(cell, transition)] += float(area)

        print(
            f"Processed source rows "
            f"{min(row0+h, loss_ds.height):,}"
            f"/{loss_ds.height:,}"
        )

    # -----------------------------------------------------------
    # 3. Recalculate basin-adjusted change fractions
    # -----------------------------------------------------------

    section("STEP 4C — BASIN-ADJUSTED FRACTIONS")

    fraction = np.full(n_cells, np.nan, dtype=np.float64)

    valid = basin_area > 0

    fraction[valid] = (
        changed_area[valid] / basin_area[valid]
    )

    if np.any(fraction[valid] > 1.001):
        raise RuntimeError(
            "Changed area exceeds basin-covered cell area"
        )

    valid_cells = np.flatnonzero(valid)

    grid_rows = valid_cells // gw
    grid_cols = valid_cells % gw

    longitudes = (
        grid_transform.c +
        (grid_cols + 0.5) * grid_transform.a
    )

    latitudes = (
        grid_transform.f +
        (grid_rows + 0.5) * grid_transform.e
    )

    grid_df = pd.DataFrame({
        "grid_cell_id": valid_cells,
        "grid_row": grid_rows,
        "grid_col": grid_cols,
        "longitude": longitudes,
        "latitude": latitudes,
        "basin_area_km2": basin_area[valid],
        "changed_area_km2": changed_area[valid],
        "changed_fraction": fraction[valid]
    })

    grid_df["unchanged_fraction"] = (
        1 - grid_df["changed_fraction"]
    )

    # Classification for descriptive sensitivity analysis.
    grid_df["change_intensity"] = pd.cut(
        grid_df["changed_fraction"],
        bins=[-1e-10, 0, 0.05, 0.20, 0.40, 1.000001],
        labels=[
            "No detected change",
            ">0–5%",
            ">5–20%",
            ">20–40%",
            ">40%"
        ],
        include_lowest=True
    )

    grid_df.to_csv(
        OUT / "Basin_Adjusted_Grid_Fractions.csv",
        index=False
    )

    print("Basin-intersecting climate cells:", len(grid_df))
    print("Basin area:", round(basin_area.sum(), 3), "km²")
    print("Changed area:", round(changed_area.sum(), 3), "km²")

    print(
        "Basin changed proportion:",
        round(
            100 * changed_area.sum() / basin_area.sum(),
            4
        ),
        "%"
    )

    print("\nFraction distribution:")
    print(grid_df["changed_fraction"].describe())

    print("\nChange-intensity groups:")
    print(
        grid_df["change_intensity"]
        .value_counts(dropna=False)
        .to_string()
    )

    # -----------------------------------------------------------
    # 4. Reconstruct each transition fraction
    # -----------------------------------------------------------

    section("STEP 4D — FRACTIONAL TRANSITION COMPOSITION")

    records = []

    for (cell, code), area in transitions.items():

        if basin_area[cell] <= 0:
            continue

        records.append({
            "grid_cell_id": cell,
            "grid_row": cell // gw,
            "grid_col": cell % gw,
            "transition_code": code,
            "from_id": code // 100,
            "to_id": code % 100,
            "transition_area_km2": area,
            "basin_area_km2": basin_area[cell],
            "transition_fraction": (
                area / basin_area[cell]
            )
        })

    transition_df = pd.DataFrame(records)

    transition_df.to_csv(
        OUT / "Basin_Adjusted_Transition_Fractions.csv",
        index=False
    )

    print("Nonzero cell-transition combinations:",
          len(transition_df))

    if not transition_df.empty:

        summary = transition_df.groupby(
            "transition_code",
            as_index=False
        )["transition_area_km2"].sum()

        summary = summary.sort_values(
            "transition_area_km2",
            ascending=False
        )

        summary.to_csv(
            OUT / "Transition_Area_Summary.csv",
            index=False
        )

        print("\nLargest transitions:")
        print(summary.head(12).to_string(index=False))

    # -----------------------------------------------------------
    # 5. Write raster products
    # -----------------------------------------------------------

    section("STEP 4E — EXPORT FRACTIONAL MAPS")

    raster = np.full(
        (4, gh, gw),
        -9999.0,
        dtype=np.float32
    )

    raster[0] = np.where(
        valid, fraction, -9999
    ).reshape(gh, gw)

    raster[1] = np.where(
        valid, basin_area, -9999
    ).reshape(gh, gw)

    raster[2] = np.where(
        valid, changed_area, -9999
    ).reshape(gh, gw)

    raster[3] = np.where(
        valid, 1 - fraction, -9999
    ).reshape(gh, gw)

    profile = grid_ds.profile.copy()

    profile.update(
        driver="GTiff",
        height=gh,
        width=gw,
        count=4,
        dtype="float32",
        nodata=-9999,
        compress="deflate"
    )

    raster_path = OUT / (
        "Niger_Basin_Fractional_Change_1992_2020.tif"
    )

    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(raster)

        for i, label in enumerate([
            "ChangedFraction",
            "BasinCoveredArea_km2",
            "ChangedArea_km2",
            "UnchangedFraction"
        ], start=1):
            dst.set_band_description(i, label)

    print("Saved:", raster_path)

    # -----------------------------------------------------------
    # 6. Scientific consistency audit
    # -----------------------------------------------------------

    section("STEP 4F — SCIENTIFIC CONSISTENCY AUDIT")

    reconstructed = sum(
        r["transition_area_km2"]
        for r in records
    )

    difference = reconstructed - changed_area.sum()

    print("Basin pixel count:", total_basin_pixels)
    print("Changed pixel count:", total_changed_pixels)
    print("Unpaired class pixels:", unpaired_pixels)
    print("Same positive class pixels:", equal_positive_pixels)

    print(
        "Total changed-area reconciliation:",
        difference,
        "km²"
    )

    if abs(difference) > 0.01:
        raise RuntimeError(
            "Fractional transition areas do not reconcile"
        )

    if unpaired_pixels:
        print(
            "WARNING: Some pixels contain only one "
            "valid transition class."
        )

    if equal_positive_pixels:
        print(
            "WARNING: Some positive class pairs "
            "are identical."
        )

    # -----------------------------------------------------------
    # 7. Save metadata and limitations
    # -----------------------------------------------------------

    metadata = {
        "study": "Niger Basin biogeophysical revision",
        "land_cover_period": "1992-2020",
        "climate_grid": "ERA5_TX_1995.tif",
        "fraction_denominator": (
            "Approximate basin-covered climate-cell area"
        ),
        "spatial_method": (
            "300-m pixel-centre allocation to 0.25-degree grid"
        ),
        "area_method": (
            "Spherical latitude-dependent pixel area"
        ),
        "basin_grid_cells": int(valid.sum()),
        "changed_grid_cells": int(
            np.sum(valid & (changed_area > 0))
        ),
        "limitations": [
            "Fine-pixel centre approximation at grid boundaries",
            "Basin polygon rasterized at source resolution",
            "No independent stable land-cover class fractions",
            "No quality-controlled valid-data land mask",
            "No causal identification of climate-model bias"
        ]
    }

    with open(OUT / "Step4_Metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSTEP 4 COMPLETED")
    print("Output directory:", OUT)
    print("Original datasets were not modified.")
