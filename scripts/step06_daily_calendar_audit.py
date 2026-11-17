# Archived analytical source from the authors' NIGER Colab notebook.
# Contains original Drive paths; run in original study environment.
# Do not run previous steps if the verified inputs already exist.

# ================================================================
# NIGER RIVER BASIN — REVIEWER 1 REVISION
# STEP 6: DAILY CALENDAR, GRID, AND UNIT AUDIT
# ================================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
from datetime import datetime
import calendar
import json
import re

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

ROOT = Path("/content/drive/MyDrive")
OUT = ROOT / "Niger_Basin_Reviewer1_Revision" / "06_Climate_Unit_Audit"
OUT.mkdir(parents=True, exist_ok=True)

def source_path(product, variable, year):
    period = "1981_1994" if year <= 1994 else "1995_2005"
    folder = (
        f"Niger_{product}_Daily_{period}"
        if product == "ERA5"
        else f"Niger_CMCC_ESM2_Daily_{period}"
    )
    prefix = "ERA5" if product == "ERA5" else "CMCC"
    return ROOT / folder / f"{prefix}_{variable}_{year}.tif"

def extract_dates(descriptions):
    dates = []
    for description in descriptions:
        text = description or ""
        match = re.search(r"(?<!\d)(\d{8})(?!\d)", text)
        dates.append(match.group(1) if match else None)
    return dates

print("=" * 76)
print("STEP 6A — AUDIT ALL DAILY DATASETS, 1981–2005")
print("=" * 76)

records = []
grid_signature = {}

for product in ["ERA5", "CMCC"]:
    for variable in ["TX", "TN", "PR"]:
        for year in range(1981, 2006):
            path = source_path(product, variable, year)

            if not path.exists():
                records.append({
                    "product": product,
                    "variable": variable,
                    "year": year,
                    "status": "MISSING",
                    "path": str(path)
                })
                continue

            with rasterio.open(path) as ds:
                expected = [
                    f"{year}{month:02d}{day:02d}"
                    for month in range(1, 13)
                    for day in range(
                        1, calendar.monthrange(year, month)[1] + 1
                    )
                ]

                observed = extract_dates(ds.descriptions)
                calendar_valid = observed == expected

                if not calendar_valid:
                    missing_dates = sorted(
                        set(expected) - set(observed)
                    )
                    unexpected_dates = sorted(
                        set(observed) - set(expected),
                        key=lambda v: str(v)
                    )
                else:
                    missing_dates = []
                    unexpected_dates = []

                signature = (
                    str(ds.crs),
                    ds.width,
                    ds.height,
                    tuple(round(v, 9) for v in ds.transform)
                )

                grid_signature[
                    (product, variable, year)
                ] = signature

                records.append({
                    "product": product,
                    "variable": variable,
                    "year": year,
                    "status": "OK" if calendar_valid else "CHECK",
                    "bands": ds.count,
                    "expected_days": len(expected),
                    "calendar_valid": calendar_valid,
                    "missing_dates": ",".join(missing_dates[:10]),
                    "unexpected_dates": ",".join(
                        str(v) for v in unexpected_dates[:10]
                    ),
                    "crs": str(ds.crs),
                    "width": ds.width,
                    "height": ds.height,
                    "nodata": str(ds.nodata),
                    "dataset_tags": json.dumps(ds.tags()),
                    "band1_tags": json.dumps(ds.tags(1)),
                    "path": str(path)
                })

audit = pd.DataFrame(records)
audit.to_csv(
    OUT / "All_150_Daily_File_Calendar_Audit.csv",
    index=False
)

print(audit.groupby(
    ["product", "variable", "status"]
).size().to_string())

invalid = audit[audit["status"] != "OK"]
print("\nFiles requiring investigation:", len(invalid))

if len(invalid):
    print(invalid[
        ["product", "variable", "year", "status",
         "missing_dates", "unexpected_dates"]
    ].to_string(index=False))

# ---------------------------------------------------------------
# 2. Check spatial grid compatibility
# ---------------------------------------------------------------

print("\n" + "=" * 76)
print("STEP 6B — GRID COMPATIBILITY")
print("=" * 76)

unique_grids = set(grid_signature.values())
print("Distinct grid geometries:", len(unique_grids))

for i, signature in enumerate(unique_grids, start=1):
    print(f"Grid {i}:", signature)

if len(unique_grids) != 1:
    print(
        "WARNING: Climate rasters do not all share the "
        "same grid. Alignment is needed before comparisons."
    )
else:
    print("All available daily rasters share one grid.")

# ---------------------------------------------------------------
# 3. Sample complete daily records at reduced spatial resolution
# ---------------------------------------------------------------

print("\n" + "=" * 76)
print("STEP 6C — TEMPERATURE AND PRECIPITATION DISTRIBUTIONS")
print("=" * 76)

SAMPLE_YEARS = [1981, 1995, 2000, 2004, 2005]
stats = []
daily_samples = {}

for year in SAMPLE_YEARS:
    for product in ["ERA5", "CMCC"]:
        for variable in ["TX", "TN", "PR"]:

            path = source_path(product, variable, year)

            with rasterio.open(path) as ds:
                # Spatial sampling only; all daily bands preserved.
                sample = ds.read(
                    out_shape=(ds.count, 20, 28),
                    masked=True,
                    resampling=Resampling.nearest
                ).astype(np.float64)

                arr = np.asarray(
                    sample.filled(np.nan),
                    dtype=np.float64
                )

                arr[~np.isfinite(arr)] = np.nan
                valid = arr[np.isfinite(arr)]

                daily_samples[(product, variable, year)] = arr

                if not len(valid):
                    print("NO VALID VALUES:", path.name)
                    continue

                nonzero = valid[valid > 0]

                row = {
                    "product": product,
                    "variable": variable,
                    "year": year,
                    "n_values": len(valid),
                    "min": float(np.min(valid)),
                    "p01": float(np.percentile(valid, 1)),
                    "p50": float(np.median(valid)),
                    "p95": float(np.percentile(valid, 95)),
                    "p99": float(np.percentile(valid, 99)),
                    "max": float(np.max(valid)),
                    "positive_fraction": float(
                        np.mean(valid > 0)
                    )
                }

                if len(nonzero):
                    row["positive_p50"] = float(
                        np.median(nonzero)
                    )
                    row["positive_p95"] = float(
                        np.percentile(nonzero, 95)
                    )

                stats.append(row)

                print(
                    f"{product:5s} {variable:2s} {year} "
                    f"P50={row['p50']:.6g} "
                    f"P95={row['p95']:.6g} "
                    f"MAX={row['max']:.6g}"
                )

stats_df = pd.DataFrame(stats)
stats_df.to_csv(
    OUT / "Sampled_Daily_Climate_Distributions.csv",
    index=False
)

# ---------------------------------------------------------------
# 4. Precipitation unit candidates
# ---------------------------------------------------------------

print("\n" + "=" * 76)
print("STEP 6D — PRECIPITATION UNIT CANDIDATES")
print("=" * 76)

# These are candidate interpretations, NOT accepted conversions:
# raw        : already mm/day
# raw*1000   : metres/day -> mm/day
# raw*86400  : kg m-2 s-1 -> mm/day
#
# ERA5 and CMCC must be assessed independently.

candidates = []

for product in ["ERA5", "CMCC"]:
    for year in SAMPLE_YEARS:

        arr = daily_samples.get((product, "PR", year))
        if arr is None:
            continue

        valid = arr[np.isfinite(arr) & (arr >= 0)]

        for label, factor in [
            ("raw_as_mm_day", 1.0),
            ("m_per_day_to_mm_day", 1000.0),
            ("kg_m2_s_to_mm_day", 86400.0)
        ]:

            vals = valid * factor

            candidates.append({
                "product": product,
                "year": year,
                "candidate": label,
                "factor": factor,
                "mean_daily_mm": float(np.mean(vals)),
                "p95_daily_mm": float(
                    np.percentile(vals, 95)
                ),
                "p99_daily_mm": float(
                    np.percentile(vals, 99)
                ),
                "maximum_daily_mm": float(np.max(vals)),
                "annualized_mean_mm": float(
                    np.mean(vals) *
                    (366 if calendar.isleap(year) else 365)
                )
            })

candidate_df = pd.DataFrame(candidates)

candidate_df.to_csv(
    OUT / "Precipitation_Conversion_Candidates.csv",
    index=False
)

print(candidate_df[
    ["product", "year", "candidate",
     "annualized_mean_mm", "p99_daily_mm",
     "maximum_daily_mm"]
].to_string(index=False))

print("\nIMPORTANT:")
print(
    "These values are spatial samples and hypothetical unit "
    "interpretations, not verified basin precipitation totals."
)

# ---------------------------------------------------------------
# 5. Cross-product daily-value comparisons
# ---------------------------------------------------------------

print("\n" + "=" * 76)
print("STEP 6E — SAME-DATE ERA5 / CMCC COMPARISONS")
print("=" * 76)

comparisons = []

for variable in ["TX", "TN", "PR"]:
    for year in SAMPLE_YEARS:

        e = daily_samples.get(("ERA5", variable, year))
        c = daily_samples.get(("CMCC", variable, year))

        if e is None or c is None:
            continue

        if e.shape != c.shape:
            print("SHAPE MISMATCH:", variable, year)
            continue

        valid = np.isfinite(e) & np.isfinite(c)

        if variable == "PR":
            valid &= (e > 0) & (c > 0)

        if np.count_nonzero(valid) < 100:
            continue

        ev = e[valid]
        cv = c[valid]

        ratios = ev / cv if variable == "PR" else None

        comparisons.append({
            "variable": variable,
            "year": year,
            "n_paired": len(ev),
            "ERA5_median_raw": float(np.median(ev)),
            "CMCC_median_raw": float(np.median(cv)),
            "median_difference_raw": float(
                np.median(cv - ev)
            ),
            "median_ERA5_to_CMCC_ratio": (
                float(np.median(ratios))
                if ratios is not None else np.nan
            ),
            "pearson_raw": float(
                np.corrcoef(ev, cv)[0, 1]
            )
        })

comp_df = pd.DataFrame(comparisons)

comp_df.to_csv(
    OUT / "Cross_Product_Sampled_Comparisons.csv",
    index=False
)

print(comp_df.to_string(index=False))

# ---------------------------------------------------------------
# 6. Print GeoTIFF metadata for original precipitation fields
# ---------------------------------------------------------------

print("\n" + "=" * 76)
print("STEP 6F — SOURCE METADATA INSPECTION")
print("=" * 76)

for product in ["ERA5", "CMCC"]:

    path = source_path(product, "PR", 1995)

    with rasterio.open(path) as ds:
        print(f"\n{product} precipitation: {path.name}")
        print("Dataset tags:", ds.tags())
        print("Band 1 tags:", ds.tags(1))
        print("Units, if exposed:", ds.units[:3])
        print("Scale:", ds.scales[:3])
        print("Offset:", ds.offsets[:3])

# ---------------------------------------------------------------
# 7. Save the audit decision status
# ---------------------------------------------------------------

summary = {
    "annual_file_count": len(audit),
    "invalid_or_missing_files": len(invalid),
    "distinct_grids": len(unique_grids),
    "sample_years": SAMPLE_YEARS,
    "precipitation_units_verified": False,
    "temperature_units_verified": False,
    "conversion_applied": False,
    "notes": [
        "Do not use conversion candidates as final values.",
        "Check source product documentation and processing scripts.",
        "Confirm precipitation flux/accumulation units.",
        "Confirm temperature Celsius/Kelvin units.",
        "Reconstruct ETCCDI indices only after unit verification."
    ]
}

with open(OUT / "Step6_Audit_Status.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 76)
print("STEP 6 COMPLETE")
print("=" * 76)
print("Files saved to:", OUT)
print("No original raster was modified.")
