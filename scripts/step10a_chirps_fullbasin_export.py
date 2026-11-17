# Archived analytical source from the authors' NIGER Colab notebook.
# Contains original Drive paths; run in original study environment.
# Do not run previous steps if the verified inputs already exist.

# ================================================================
# NIGER BASIN — REVIEWER 1 REVISION
# STEP 10A: FULL-BASIN CHIRPS VALIDATION EXPORT
# Project: nana213
# Years: 1981–2005
# Spatial unit: Original ERA5/CMCC 0.25-degree cells
# Main mask: >=90% Niger Basin coverage
#
# One multiband image + one reduceRegions + one batch export.
# ================================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import ee
import pandas as pd
import rasterio

PROJECT_ID = "nana213"

try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate(auth_mode="notebook")
    ee.Initialize(project=PROJECT_ID)

ROOT = Path("/content/drive/MyDrive")
BASE = ROOT / "Niger_Basin_Reviewer1_Revision"

GRID_CSV = (
    BASE / "05_Grid_Coverage_QC" /
    "Main_Analysis_Grid_Coverage90.csv"
)

GRID_RASTER = (
    BASE / "05_Grid_Coverage_QC" /
    "Niger_Grid_Coverage_QC.tif"
)

grid = pd.read_csv(GRID_CSV)

assert grid["grid_cell_id"].is_unique

with rasterio.open(GRID_RASTER) as ds:
    transform = ds.transform
    assert ds.crs.to_epsg() == 4326

features = []

for row in grid.itertuples(index=False):

    r = int(row.grid_row)
    c = int(row.grid_col)

    left = transform.c + c * transform.a
    right = left + transform.a
    top = transform.f + r * transform.e
    bottom = top + transform.e

    geom = ee.Geometry.Rectangle(
        [left, bottom, right, top],
        proj="EPSG:4326",
        geodesic=False
    )

    features.append(
        ee.Feature(
            geom,
            {"grid_cell_id": int(row.grid_cell_id)}
        )
    )

regions = ee.FeatureCollection(features)

years = list(range(1981, 2006))

chirps = (
    ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    .select("precipitation")
)

annual_images = []

for year in years:

    annual_image = (
        chirps
        .filterDate(
            f"{year}-01-01",
            f"{year + 1}-01-01"
        )
        .sum()
        .rename(f"p_{year}")
        .toFloat()
    )

    annual_images.append(annual_image)

multiband = ee.Image.cat(*annual_images)

# Single spatial aggregation.
results = multiband.reduceRegions(
    collection=regions,
    reducer=ee.Reducer.mean(),
    scale=5566,
    crs="EPSG:4326",
    tileScale=8
)

EXPORT_NAME = "Niger_CHIRPS_FullGrid_1981_2005"

selectors = (
    ["grid_cell_id"] +
    [f"p_{year}" for year in years]
)

task = ee.batch.Export.table.toDrive(
    collection=results,
    description=EXPORT_NAME,
    folder="Niger_Basin_Reviewer1_Revision",
    fileNamePrefix=EXPORT_NAME,
    fileFormat="CSV",
    selectors=selectors
)

task.start()

print("=" * 72)
print("STEP 10A — FULL-BASIN CHIRPS EXPORT STARTED")
print("=" * 72)
print("Earth Engine project:", PROJECT_ID)
print("Number of climate cells:", len(grid))
print("Annual precipitation bands:", len(years))
print("Expected cell-year values:", len(grid) * len(years))
print("Task ID:", task.id)
print("Task state:", task.status().get("state"))
print("Export filename:", EXPORT_NAME + ".csv")
print("Original files unchanged.")
