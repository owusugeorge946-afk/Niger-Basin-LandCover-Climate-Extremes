# Archived analytical source from the authors' NIGER Colab notebook.
# Contains original Drive paths; run in original study environment.
# Do not run previous steps if the verified inputs already exist.

# ================================================================
# NIGER RIVER BASIN — REVIEWER 1 REVISION
# STEP 11: SPATIALLY ADJUSTED CLIMATE-BIAS ASSOCIATIONS
#
# Outcomes:
#   CMCC-ESM2 minus CHIRPS mean annual rainfall, 1995–2005
#   ERA5 minus CHIRPS mean annual rainfall, 1995–2005
#   CMCC-ESM2 minus ERA5 mean annual rainfall, 1995–2005
#
# Main predictor:
#   Fraction of basin-covered climate cell mapped as
#   changed between 1992 and 2020.
#
# Controls:
#   Latitude, longitude, their quadratic spatial terms,
#   and logarithm of CHIRPS mean annual rainfall.
#
# Uncertainty:
#   2-degree geographic block bootstrap.
#
# IMPORTANT:
# This is a descriptive cross-period association.
# It does NOT estimate causal land-cover feedbacks.
# ================================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/content/drive/MyDrive")
BASE = ROOT / "Niger_Basin_Reviewer1_Revision"

SOURCE = (
    BASE / "10_Full_Basin_CHIRPS_Validation" /
    "CHIRPS_Full_Basin_Spatial_Climatology.csv"
)

OUT = BASE / "11_Spatial_Confounding_Audit"
OUT.mkdir(parents=True, exist_ok=True)

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

df = pd.read_csv(SOURCE)

df = df.loc[
    df["period"].astype(str) == "1995–2005"
].copy()

if df.empty:
    raise RuntimeError(
        "No 1995–2005 records found. "
        "Check period labels in the Step 10 output."
    )

assert df["grid_cell_id"].is_unique

required = [
    "longitude",
    "latitude",
    "basin_area_km2",
    "changed_fraction",
    "CHIRPS_annual_mm",
    "ERA5_annual_mm",
    "CMCC_annual_mm"
]

for col in required:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=required)

assert (df["basin_area_km2"] > 0).all()
assert df["changed_fraction"].between(0, 1.00001).all()
assert (df["CHIRPS_annual_mm"] >= 0).all()

print("=" * 78)
print("STEP 11A — VALIDATED ANALYSIS TABLE")
print("=" * 78)
print("Climate cells:", len(df))
print("Mapped changed fraction:")
print(df["changed_fraction"].describe().to_string())

# ---------------------------------------------------------------
# 1. Prepare bias variables
# ---------------------------------------------------------------

df["CMCC_minus_CHIRPS"] = (
    df["CMCC_annual_mm"] -
    df["CHIRPS_annual_mm"]
)

df["ERA5_minus_CHIRPS"] = (
    df["ERA5_annual_mm"] -
    df["CHIRPS_annual_mm"]
)

df["CMCC_minus_ERA5"] = (
    df["CMCC_annual_mm"] -
    df["ERA5_annual_mm"]
)

# Fraction measured in 10-percentage-point units.
# A regression coefficient refers to an increase
# of 0.10 in the land-cover changed fraction.
df["change_10pp"] = df["changed_fraction"] / 0.10

# ---------------------------------------------------------------
# 2. Construct geographically adjusted design matrix
# ---------------------------------------------------------------

lat = df["latitude"].to_numpy(dtype=float)
lon = df["longitude"].to_numpy(dtype=float)

lat_c = (lat - np.mean(lat)) / 5.0
lon_c = (lon - np.mean(lon)) / 5.0

climate = np.log1p(
    df["CHIRPS_annual_mm"].to_numpy(dtype=float)
)

climate_c = climate - climate.mean()

change = df["change_10pp"].to_numpy(dtype=float)
n = len(df)

X_simple = np.column_stack([
    np.ones(n),
    change
])

X_adjusted = np.column_stack([
    np.ones(n),
    change,
    lat_c,
    lon_c,
    lat_c**2,
    lon_c**2,
    lat_c * lon_c,
    climate_c
])

names_simple = ["Intercept", "Change per 10 pp"]

names_adjusted = [
    "Intercept",
    "Change per 10 pp",
    "Latitude",
    "Longitude",
    "Latitude squared",
    "Longitude squared",
    "Latitude × longitude",
    "Log CHIRPS rainfall"
]

weights = df["basin_area_km2"].to_numpy(dtype=float)
weights = weights / weights.mean()

# ---------------------------------------------------------------
# 3. Weighted linear regression
# ---------------------------------------------------------------

def fit_weighted(X, y, w):
    valid = (
        np.isfinite(y) &
        np.isfinite(w) &
        (w > 0) &
        np.all(np.isfinite(X), axis=1)
    )

    Xv = X[valid]
    yv = y[valid]
    wv = w[valid]

    sqrt_w = np.sqrt(wv)

    xw = Xv * sqrt_w[:, None]
    yw = yv * sqrt_w

    coef, _, rank, _ = np.linalg.lstsq(
        xw, yw, rcond=None
    )

    if rank < X.shape[1]:
        raise RuntimeError("Regression design is rank deficient.")

    fitted = Xv @ coef
    residual = yv - fitted

    mean_y = np.average(yv, weights=wv)

    tss = np.sum(wv * (yv - mean_y)**2)
    rss = np.sum(wv * residual**2)

    r2 = 1 - rss / tss if tss > 0 else np.nan

    return coef, r2

# ---------------------------------------------------------------
# 4. Geographic block bootstrap
# ---------------------------------------------------------------

# Fixed 2-degree latitude-longitude blocks.
# Neighboring cells are resampled together.
BLOCK_DEGREES = 2.0
N_BOOTSTRAPS = 500
SEED = 20261116

lat_block = np.floor(
    (lat - np.floor(lat.min())) / BLOCK_DEGREES
).astype(int)

lon_block = np.floor(
    (lon - np.floor(lon.min())) / BLOCK_DEGREES
).astype(int)

block_keys = [
    f"{a}_{b}"
    for a, b in zip(lat_block, lon_block)
]

df["spatial_block_id"] = block_keys

unique_blocks = np.unique(block_keys)
block_to_indices = {
    block: np.flatnonzero(
        np.asarray(block_keys) == block
    )
    for block in unique_blocks
}

print("\n" + "=" * 78)
print("STEP 11B — SPATIAL BLOCK DESIGN")
print("=" * 78)
print("Block width:", BLOCK_DEGREES, "degrees")
print("Number of occupied blocks:", len(unique_blocks))
print("Bootstrap replicates:", N_BOOTSTRAPS)

rng = np.random.default_rng(SEED)

# Use identical spatial resamples for the three
# climate-bias outcomes to facilitate comparisons.
bootstrap_samples = []

for _ in range(N_BOOTSTRAPS):

    sampled_blocks = rng.choice(
        unique_blocks,
        size=len(unique_blocks),
        replace=True
    )

    indices = np.concatenate([
        block_to_indices[b] for b in sampled_blocks
    ])

    bootstrap_samples.append(indices)

# ---------------------------------------------------------------
# 5. Fit models and calculate confidence intervals
# ---------------------------------------------------------------

print("\n" + "=" * 78)
print("STEP 11C — SPATIALLY ADJUSTED ASSOCIATIONS")
print("=" * 78)

outcomes = {
    "CMCC minus CHIRPS": "CMCC_minus_CHIRPS",
    "ERA5 minus CHIRPS": "ERA5_minus_CHIRPS",
    "CMCC minus ERA5": "CMCC_minus_ERA5"
}

results = []

for outcome_name, column in outcomes.items():

    y = df[column].to_numpy(dtype=float)

    for model_name, X in [
        ("Unadjusted", X_simple),
        ("Geography + CHIRPS adjusted", X_adjusted)
    ]:

        coef, r2 = fit_weighted(X, y, weights)

        boot_coef = []

        for indices in bootstrap_samples:

            try:
                b, _ = fit_weighted(
                    X[indices],
                    y[indices],
                    weights[indices]
                )
                boot_coef.append(b[1])

            except (RuntimeError, np.linalg.LinAlgError):
                continue

        boot_coef = np.asarray(boot_coef)

        if len(boot_coef) < 0.90 * N_BOOTSTRAPS:
            raise RuntimeError(
                "Too many invalid bootstrap replicates "
                f"for {outcome_name}, {model_name}."
            )

        lower, upper = np.percentile(
            boot_coef, [2.5, 97.5]
        )

        results.append({
            "outcome": outcome_name,
            "model": model_name,
            "n_cells": len(y),
            "n_spatial_blocks": len(unique_blocks),
            "change_coefficient_mm_per_10pp": coef[1],
            "ci95_lower": lower,
            "ci95_upper": upper,
            "weighted_R2": r2,
            "bootstrap_replicates_valid": len(boot_coef)
        })

        print(f"\n{outcome_name} — {model_name}")
        print(
            "Change coefficient:",
            f"{coef[1]:.3f} mm/year per 10 pp"
        )
        print(
            "Spatial-block bootstrap 95% CI:",
            f"[{lower:.3f}, {upper:.3f}]"
        )
        print("Weighted R²:", round(r2, 4))

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUT / "Spatially_Adjusted_Associations.csv",
    index=False
)

# ---------------------------------------------------------------
# 6. Sensitivity to strong precipitation-bias outliers
# ---------------------------------------------------------------

print("\n" + "=" * 78)
print("STEP 11D — OUTLIER SENSITIVITY")
print("=" * 78)

sensitivity = []

for outcome_name, column in outcomes.items():

    y = df[column].to_numpy(dtype=float)

    abs_cutoff = np.quantile(np.abs(y), 0.99)

    masks = {
        "All cells": np.ones(len(y), dtype=bool),
        "Exclude upper 1% absolute bias": (
            np.abs(y) <= abs_cutoff
        )
    }

    for label, mask in masks.items():

        coef, r2 = fit_weighted(
            X_adjusted[mask],
            y[mask],
            weights[mask]
        )

        sensitivity.append({
            "outcome": outcome_name,
            "sample": label,
            "n_cells": int(mask.sum()),
            "change_coefficient_mm_per_10pp": coef[1],
            "weighted_R2": r2
        })

sensitivity_df = pd.DataFrame(sensitivity)

sensitivity_df.to_csv(
    OUT / "Outlier_Sensitivity.csv",
    index=False
)

print(sensitivity_df.to_string(index=False))

# ---------------------------------------------------------------
# 7. Produce coefficient comparison figure
# ---------------------------------------------------------------

print("\n" + "=" * 78)
print("STEP 11E — REGRESSION COEFFICIENT FIGURE")
print("=" * 78)

plot_df = results_df.copy()

labels = [
    f"{r.outcome}\n{r.model}"
    for r in plot_df.itertuples()
]

ypos = np.arange(len(plot_df))

est = plot_df[
    "change_coefficient_mm_per_10pp"
].to_numpy()

low = plot_df["ci95_lower"].to_numpy()
high = plot_df["ci95_upper"].to_numpy()

fig, ax = plt.subplots(figsize=(11, 7))

ax.errorbar(
    est,
    ypos,
    xerr=[
        est - low,
        high - est
    ],
    fmt="o",
    capsize=4,
    linewidth=1.5
)

ax.axvline(
    0,
    color="black",
    linestyle="--",
    linewidth=1
)

ax.set_yticks(ypos)
ax.set_yticklabels(labels)
ax.invert_yaxis()

ax.set_xlabel(
    "Change in annual precipitation bias (mm/year)\n"
    "per 10-percentage-point increase in mapped "
    "land-cover transition fraction"
)

ax.set_title(
    "Descriptive land-cover association before "
    "and after geographic adjustment"
)

ax.grid(
    axis="x",
    linestyle=":",
    alpha=0.3
)

fig.tight_layout()

fig.savefig(
    OUT / "Figure_Spatial_Adjustment_Comparison.png",
    dpi=900,
    bbox_inches="tight"
)

fig.savefig(
    OUT / "Figure_Spatial_Adjustment_Comparison.pdf",
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# ---------------------------------------------------------------
# 8. Export model-input table
# ---------------------------------------------------------------

df.to_csv(
    OUT / "Spatial_Model_Input_1995_2005.csv",
    index=False
)

metadata = {
    "step": 11,
    "climate_period": "1995-2005",
    "land_cover_transition_period": "1992-2020",
    "n_grid_cells": int(len(df)),
    "block_size_degrees": BLOCK_DEGREES,
    "bootstrap_replicates": N_BOOTSTRAPS,
    "random_seed": SEED,
    "fraction_scale": (
        "Regression coefficient per 10 percentage points "
        "of basin-covered area mapped as changed"
    ),
    "geographical_controls": [
        "Latitude",
        "Longitude",
        "Latitude squared",
        "Longitude squared",
        "Latitude-longitude interaction",
        "Log of CHIRPS mean annual precipitation"
    ],
    "interpretation": (
        "Cross-sectional descriptive association only; "
        "not a causal land-cover effect."
    ),
    "limitations": [
        "Land-cover period extends beyond climate period",
        "CHIRPS-based climate covariate may share "
        "error structure with CHIRPS-referenced outcomes",
        "Block bootstrap does not remove omitted confounding",
        "Only total changed fraction is modeled",
        "Spatial dependence remains possible beyond 2 degrees"
    ]
}

with open(OUT / "Step11_Metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 78)
print("STEP 11 COMPLETED")
print("=" * 78)
print("Output directory:", OUT)
print("Original source files unchanged.")
