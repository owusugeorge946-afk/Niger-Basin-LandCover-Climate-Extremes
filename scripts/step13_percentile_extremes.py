# Archived analytical source from the authors' NIGER Colab notebook.
# Contains original Drive paths; run in original study environment.
# Do not run previous steps if the verified inputs already exist.

# ================================================================
# NIGER BASIN — REVIEWER 1 REVISION
# STEP 13: FOUR PERCENTILE-BASED TEMPERATURE INDICES
#
# Indices: TX90p, TN90p, WSDI, CSDI
# Evaluation: 1995–2005
# Primary reference: 1981–1994 (independent, 14 years)
# Sensitivity reference: 1981–2005 (overlapping, 25 years)
#
# The full-period sensitivity calculation is NOT an ETCCDI
# in-base bootstrap and must not be labelled as one.
#
# Dataset values assumed to be degrees Celsius.
# Original rasters are not modified.
# ================================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import calendar
import json
import re
import numpy as np
import pandas as pd
import rasterio

ROOT = Path("/content/drive/MyDrive")
BASE = ROOT / "Niger_Basin_Reviewer1_Revision"
OUT = BASE / "13_Percentile_Temperature_Extremes"
OUT.mkdir(parents=True, exist_ok=True)

GRID_FILE = (
    BASE / "05_Grid_Coverage_QC" /
    "Main_Analysis_Grid_Coverage90.csv"
)

grid = pd.read_csv(GRID_FILE).sort_values("grid_cell_id")

ids = grid.grid_cell_id.to_numpy(dtype=int)
rr = grid.grid_row.to_numpy(dtype=int)
cc = grid.grid_col.to_numpy(dtype=int)
areas = grid.basin_area_km2.to_numpy(dtype=float)

YEARS = list(range(1981, 2006))
EVAL = list(range(1995, 2006))
BASELINES = {
    "independent_1981_1994": list(range(1981, 1995)),
    "overlapping_1981_2005": YEARS
}

N = len(ids)

assert N == 2661
assert grid.grid_cell_id.is_unique

def heading(value):
    print("\n" + "=" * 76)
    print(value)
    print("=" * 76)

def path_for(product, variable, year):
    period = "1981_1994" if year <= 1994 else "1995_2005"
    folder = (
        f"Niger_ERA5_Daily_{period}"
        if product == "ERA5"
        else f"Niger_CMCC_ESM2_Daily_{period}"
    )
    prefix = "ERA5" if product == "ERA5" else "CMCC"
    return ROOT / folder / f"{prefix}_{variable}_{year}.tif"

def read_year(product, variable, year):
    path = path_for(product, variable, year)
    with rasterio.open(path) as ds:
        expected = 366 if calendar.isleap(year) else 365
        assert ds.count == expected, path.name

        dates = [
            re.search(r"(?<!\d)(\d{8})(?!\d)", d or "").group(1)
            for d in ds.descriptions
        ]

        expected_dates = [
            f"{year}{month:02d}{day:02d}"
            for month in range(1, 13)
            for day in range(
                1, calendar.monthrange(year, month)[1] + 1
            )
        ]

        assert dates == expected_dates, (
            "Calendar mismatch: " + path.name
        )

        a = ds.read(masked=True)[:, rr, cc]

        values = np.asarray(
            np.ma.filled(a, np.nan),
            dtype=np.float32
        )

        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Missing daily temperatures: {path.name}"
            )

        return values

def remove_feb29(array, year):
    if calendar.isleap(year):
        assert array.shape[0] == 366
        return np.concatenate(
            [array[:59], array[60:]],
            axis=0
        )
    assert array.shape[0] == 365
    return array

def percentile_thresholds(stack, selected_indices, percentile):
    """
    stack: [year, 365 no-leap calendar days, cell]
    5-day circular calendar window (-2,-1,0,+1,+2).
    """
    sub = stack[selected_indices]
    thresholds = np.empty((365, N), dtype=np.float32)

    for day in range(365):
        window_days = [
            (day + offset) % 365
            for offset in (-2, -1, 0, 1, 2)
        ]

        sample = sub[:, window_days, :]
        sample = sample.reshape(-1, N)

        thresholds[day] = np.percentile(
            sample,
            percentile,
            axis=0,
            method="linear"
        ).astype(np.float32)

    return thresholds

def expand_threshold_to_year(threshold, year):
    """
    Leap day uses interpolation between February 28 and
    March 1 thresholds. It is excluded from threshold fitting.
    """
    if not calendar.isleap(year):
        return threshold

    feb29 = (
        0.5 * (threshold[58] + threshold[59])
    )[None, :]

    return np.concatenate(
        [threshold[:59], feb29, threshold[59:]],
        axis=0
    )

def spell_days(condition, min_length=6):
    """
    Count all days belonging to within-calendar-year spells
    of at least min_length consecutive True values.
    """
    length = np.zeros(condition.shape[1], dtype=np.int32)
    output = np.zeros(condition.shape[1], dtype=np.int32)

    for day in range(condition.shape[0]):
        ended = ~condition[day]
        output += np.where(
            ended & (length >= min_length),
            length,
            0
        )
        length = np.where(condition[day], length + 1, 0)

    output += np.where(length >= min_length, length, 0)
    return output

def area_mean(values):
    return float(np.average(values, weights=areas))

# ---------------------------------------------------------------
# Process each product independently.
# ---------------------------------------------------------------

heading("STEP 13A — TEMPERATURE THRESHOLD RECONSTRUCTION")

annual_records = []
threshold_metadata = []

for product in ["ERA5", "CMCC"]:

    print("\nPRODUCT:", product)

    # Store one variable at a time to reduce peak memory.
    stacks = {}

    for variable in ["TX", "TN"]:

        daily = [
            read_year(product, variable, year)
            for year in YEARS
        ]

        stacks[variable] = np.stack(
            [
                remove_feb29(a, year)
                for a, year in zip(daily, YEARS)
            ],
            axis=0
        )

        del daily

        print(
            variable,
            "stack shape:",
            stacks[variable].shape
        )

    for baseline_name, baseline_years in BASELINES.items():

        print("\nBASELINE:", baseline_name)

        positions = np.array(
            [YEARS.index(y) for y in baseline_years],
            dtype=int
        )

        tx90 = percentile_thresholds(
            stacks["TX"], positions, 90
        )
        tn90 = percentile_thresholds(
            stacks["TN"], positions, 90
        )
        tn10 = percentile_thresholds(
            stacks["TN"], positions, 10
        )

        threshold_metadata.append({
            "product": product,
            "baseline": baseline_name,
            "baseline_years": len(baseline_years),
            "evaluation_years": len(EVAL),
            "window_days": 5,
            "quantile_method": "linear",
            "bootstrap_applied": False
        })

        # Save percentile threshold rasters as compressed arrays.
        # Dimensions: calendar_day x selected climate_cell.
        np.savez_compressed(
            OUT / f"{product}_{baseline_name}_Thresholds.npz",
            TX90=tx90,
            TN90=tn90,
            TN10=tn10,
            grid_cell_id=ids
        )

        for year in EVAL:

            idx = YEARS.index(year)

            # Daily values for evaluation year are reopened
            # to preserve leap-day observations.
            tx = read_year(product, "TX", year)
            tn = read_year(product, "TN", year)

            hot_tx = tx > expand_threshold_to_year(
                tx90, year
            )

            hot_tn = tn > expand_threshold_to_year(
                tn90, year
            )

            cold_tn = tn < expand_threshold_to_year(
                tn10, year
            )

            tx90p = 100.0 * hot_tx.mean(axis=0)
            tn90p = 100.0 * hot_tn.mean(axis=0)

            wsdi = spell_days(hot_tx, min_length=6)
            csdi = spell_days(cold_tn, min_length=6)

            annual_records.append(
                pd.DataFrame({
                    "product": product,
                    "baseline": baseline_name,
                    "year": year,
                    "grid_cell_id": ids,
                    "TX90p": tx90p,
                    "TN90p": tn90p,
                    "WSDI": wsdi,
                    "CSDI": csdi
                })
            )

            print(
                year,
                f"TX90p={area_mean(tx90p):.2f}%",
                f"TN90p={area_mean(tn90p):.2f}%",
                f"WSDI={area_mean(wsdi):.2f}d",
                f"CSDI={area_mean(csdi):.2f}d"
            )

    del stacks

# ---------------------------------------------------------------
# Calculate climate-index biases.
# ---------------------------------------------------------------

heading("STEP 13B — CLIMATOLOGICAL INDEX BIASES")

annual = pd.concat(annual_records, ignore_index=True)

assert not annual.duplicated(
    ["product", "baseline", "year", "grid_cell_id"]
).any()

annual.to_csv(
    OUT / "Annual_Percentile_Indices_1995_2005.csv",
    index=False
)

INDEX_NAMES = ["TX90p", "TN90p", "WSDI", "CSDI"]

climate = annual.groupby(
    ["baseline", "product", "grid_cell_id"],
    as_index=False
)[INDEX_NAMES].mean()

pairs = []

for baseline in BASELINES:

    sub = climate[climate.baseline == baseline]

    era = sub[
        sub["product"] == "ERA5"
    ].drop(columns=["product", "baseline"])

    cmcc = sub[
        sub["product"] == "CMCC"
    ].drop(columns=["product", "baseline"])

    comparison = era.merge(
        cmcc,
        on="grid_cell_id",
        suffixes=("_ERA5", "_CMCC"),
        validate="one_to_one"
    )

    for index in INDEX_NAMES:
        comparison[f"{index}_bias"] = (
            comparison[f"{index}_CMCC"] -
            comparison[f"{index}_ERA5"]
        )

    comparison["baseline"] = baseline
    pairs.append(comparison)

paired = pd.concat(pairs, ignore_index=True)

paired = paired.merge(
    grid[
        ["grid_cell_id", "latitude", "longitude",
         "basin_area_km2", "changed_fraction"]
    ],
    on="grid_cell_id",
    validate="many_to_one"
)

paired.to_csv(
    OUT / "Recomputed_Percentile_Index_Biases.csv",
    index=False
)

summary = []

for baseline in BASELINES:
    sub = paired[paired.baseline == baseline]
    weights = sub["basin_area_km2"]

    for index in INDEX_NAMES:
        summary.append({
            "baseline": baseline,
            "index": index,
            "ERA5": float(np.average(
                sub[f"{index}_ERA5"], weights=weights
            )),
            "CMCC": float(np.average(
                sub[f"{index}_CMCC"], weights=weights
            )),
            "CMCC_minus_ERA5": float(np.average(
                sub[f"{index}_bias"], weights=weights
            ))
        })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    OUT / "Four_Percentile_Index_Bias_Summary.csv",
    index=False
)

print("\nAREA-WEIGHTED SUMMARY")
print(summary_df.to_string(index=False))

# ---------------------------------------------------------------
# Baseline sensitivity
# ---------------------------------------------------------------

heading("STEP 13C — REFERENCE-PERIOD SENSITIVITY")

pivot = summary_df.pivot(
    index="index",
    columns="baseline",
    values="CMCC_minus_ERA5"
)

pivot["difference_25yr_minus_14yr"] = (
    pivot["overlapping_1981_2005"] -
    pivot["independent_1981_1994"]
)

pivot.to_csv(
    OUT / "Percentile_Baseline_Sensitivity.csv"
)

print(pivot.to_string())

# ---------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------

heading("STEP 13D — QUALITY CONTROL")

for index in ["TX90p", "TN90p"]:
    assert annual[index].between(0, 100).all()

for index in ["WSDI", "CSDI"]:
    assert (annual[index] >= 0).all()
    assert (annual[index] <= 366).all()

for index, pct in [("WSDI", "TX90p"), ("CSDI", None)]:
    # Spell-day counts can never exceed total days
    # satisfying the corresponding threshold.
    if index == "WSDI":
        max_days = (
            annual["TX90p"] / 100.0 *
            annual["year"].map(
                lambda y: 366 if calendar.isleap(y) else 365
            )
        )
        assert np.all(annual["WSDI"] <= max_days + 1e-5)

expected_rows = (
    2 * len(BASELINES) * len(EVAL) * N
)

assert len(annual) == expected_rows

print("Annual index records:", len(annual))
print("Expected records:", expected_rows)
print("Quality checks passed.")

# ---------------------------------------------------------------
# Save provenance and limitations.
# ---------------------------------------------------------------

metadata = {
    "study": "Niger Basin Reviewer 1 Revision",
    "step": 13,
    "evaluation_period": "1995-2005",
    "reference_periods": list(BASELINES.keys()),
    "calendar_window": 5,
    "temperature_percentiles": {
        "TX90p": "TX > 90th percentile",
        "TN90p": "TN > 90th percentile",
        "WSDI": "TX > 90th percentile for >=6 consecutive days",
        "CSDI": "TN < 10th percentile for >=6 consecutive days"
    },
    "spell_year_boundary": "Spells reset at calendar-year boundary",
    "leap_day": (
        "Excluded from reference quantile estimation; "
        "evaluation threshold interpolated from adjacent days"
    ),
    "reference_year_count": {
        "independent": 14,
        "overlapping": 25
    },
    "in_base_bootstrap": False,
    "limitations": [
        "Neither reference period is 30 years",
        "The 1981-2005 sensitivity includes evaluation years",
        "Overlapping-baseline results are not bootstrap corrected",
        "Model-specific percentile thresholds represent "
        "relative extremes within each climate product",
        "Neither analysis identifies causal land-cover feedbacks"
    ]
}

with open(OUT / "Step13_Metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSTEP 13 COMPLETE")
print("Output directory:", OUT)
print("Original climate rasters unchanged.")
