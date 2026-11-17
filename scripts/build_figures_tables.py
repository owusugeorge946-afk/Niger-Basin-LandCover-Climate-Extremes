# Published-source reproduction script (adapted from executed NIGER notebook cell 14).
# Numeric operations, source column names, and plotting methods remain unchanged.

# ================================================================
# NIGER RIVER BASIN — REVIEWER 1 REVISION
# COMPLETE FIGURE AND TABLE GENERATION PACKAGE
#
# Inputs: Previously verified outputs from Steps 4–13
# Outputs:
#   01_Figures/      PNG (900 dpi) + vector PDF
#   02_Tables/       CSV + consolidated Excel workbook
#   03_Audit/        File inventory and generation log
#
# All original scientific data remain unchanged.
# ================================================================

import os
from pathlib import Path
if Path("/content/drive").exists() and not (Path("/content/drive/MyDrive").exists()):
    from google.colab import drive
    drive.mount("/content/drive")

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

# ================================================================
# 1. PATHS AND JOURNAL FIGURE SETTINGS
# ================================================================

ROOT = Path(os.environ.get("NIGER_DRIVE_ROOT", "/content/drive/MyDrive"))
BASE = Path(os.environ.get("NIGER_ANALYSIS_BASE", str(ROOT / "Niger_Basin_Reviewer1_Revision")))

S4  = BASE / "04_Basin_Adjusted_Fractions"
S5  = BASE / "05_Grid_Coverage_QC"
S10 = BASE / "10_Full_Basin_CHIRPS_Validation"
S11 = BASE / "11_Spatial_Confounding_Audit"
S12 = BASE / "12_Recomputed_Absolute_Extremes"
S13 = BASE / "13_Percentile_Temperature_Extremes"

SURFACE = (
    ROOT / "Niger_Figure_6_Data" /
    "Niger_ERA5Land_Surface_Mechanisms_1995_2005.tif"
)

OUT = Path(os.environ.get("NIGER_OUTPUT_DIR", str(BASE / "RSASE_Final_Submission")))
FIG = OUT / "01_Figures"
TAB = OUT / "02_Tables"
AUD = OUT / "03_Audit"

for folder in [FIG, TAB, AUD]:
    folder.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

DPI = int(os.environ.get("NIGER_FIGURE_DPI", "900"))

# Scientific palette: blue negative / red positive.
BIAS_CMAP = "RdBu_r"

generated_figures = []
generated_tables = {}
issues = []

def heading(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required source result is missing:\n{path}"
        )
    return pd.read_csv(path)

def save_figure(fig, name):
    png = FIG / f"{name}.png"
    pdf = FIG / f"{name}.pdf"

    fig.savefig(
        png, dpi=DPI,
        bbox_inches="tight",
        facecolor="white"
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white"
    )

    generated_figures.append(name)
    print("Saved:", name)

    if os.environ.get("NIGER_SHOW_FIGURES", "0") == "1":
        plt.show()
    plt.close(fig)

def save_table(data, name):
    data.to_csv(
        TAB / f"{name}.csv",
        index=False
    )
    generated_tables[name] = data.copy()
    print("Saved table:", name, "| rows:", len(data))

def weighted_mean(values, weights):
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    good = (
        np.isfinite(x) &
        np.isfinite(w) &
        (w > 0)
    )

    if not good.any():
        return np.nan

    return float(
        np.average(x[good], weights=w[good])
    )

def make_grid_map(
    values, rows, cols, shape, nodata=np.nan
):
    out = np.full(shape, nodata, dtype=np.float64)
    out[
        np.asarray(rows, dtype=int),
        np.asarray(cols, dtype=int)
    ] = np.asarray(values, dtype=float)
    return out

def plot_map(
    ax, image, extent, title, unit,
    diverging=False, vmax=None
):
    vals = np.asarray(image, dtype=float)
    valid = vals[np.isfinite(vals)]

    if valid.size == 0:
        ax.set_axis_off()
        return None

    if diverging:
        if vmax is None:
            vmax = float(
                np.percentile(np.abs(valid), 98)
            )
        vmax = max(vmax, 1e-8)
        im = ax.imshow(
            np.ma.masked_invalid(vals),
            origin="upper",
            extent=extent,
            cmap=BIAS_CMAP,
            norm=TwoSlopeNorm(
                vmin=-vmax,
                vcenter=0,
                vmax=vmax
            ),
            interpolation="nearest"
        )
    else:
        im = ax.imshow(
            np.ma.masked_invalid(vals),
            origin="upper",
            extent=extent,
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            interpolation="nearest"
        )

    ax.set_title(title)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(
        linestyle=":",
        linewidth=0.45,
        alpha=0.22
    )

    cb = ax.figure.colorbar(
        im, ax=ax,
        shrink=0.75,
        pad=0.02
    )
    cb.set_label(unit)

    return im

# ================================================================
# 2. LOAD VERIFIED RESULTS
# ================================================================

heading("SOURCE DATA VALIDATION")

grid_all = load_csv(
    S5 / "Main_Analysis_Grid_Coverage90.csv"
)

fraction_all = load_csv(
    S4 / "Basin_Adjusted_Grid_Fractions.csv"
)

transitions = load_csv(
    S4 / "Basin_Adjusted_Transition_Fractions.csv"
)

transition_summary = load_csv(
    S4 / "Transition_Area_Summary.csv"
)

boundary_sensitivity = load_csv(
    S5 / "Boundary_Coverage_Sensitivity.csv"
)

chirps = load_csv(
    S10 / "CHIRPS_Full_Basin_Spatial_Climatology.csv"
)

chirps_metrics = load_csv(
    S10 / "CHIRPS_Full_Basin_Period_Metrics.csv"
)

chirps_zones = load_csv(
    S10 / "CHIRPS_Latitudinal_Zone_Comparison.csv"
)

regression = load_csv(
    S11 / "Spatially_Adjusted_Associations.csv"
)

outlier = load_csv(
    S11 / "Outlier_Sensitivity.csv"
)

extreme_five = load_csv(
    S12 / "Five_Extreme_Index_Bias_Summary.csv"
)

extreme_spatial = load_csv(
    S12 / "Recomputed_Climatological_Extreme_Biases.csv"
)

extreme_latitude = load_csv(
    S12 / "Latitude_Zone_Extreme_Biases.csv"
)

original_audit = load_csv(
    S12 / "Original_vs_Recomputed_Extremes.csv"
)

percentile_spatial = load_csv(
    S13 / "Recomputed_Percentile_Index_Biases.csv"
)

percentile_summary = load_csv(
    S13 / "Four_Percentile_Index_Bias_Summary.csv"
)

daily_audit = load_csv(
    BASE / "06_Climate_Unit_Audit" /
    "All_150_Daily_File_Calendar_Audit.csv"
)

# Verify the common main analytical sample.
assert len(grid_all) == 2661
assert grid_all.grid_cell_id.is_unique
assert extreme_spatial.grid_cell_id.is_unique

MAIN_IDS = set(grid_all.grid_cell_id)

assert MAIN_IDS == set(extreme_spatial.grid_cell_id)

# Use the archived ERA5 climate grid for map geometry.
REF_RASTER = (
    ROOT / "Niger_ERA5_Daily_1995_2005" /
    "ERA5_TX_1995.tif"
)

with rasterio.open(REF_RASTER) as src:
    SHAPE = (src.height, src.width)
    BOUNDS = src.bounds
    TRANSFORM = src.transform

EXTENT = [
    BOUNDS.left, BOUNDS.right,
    BOUNDS.bottom, BOUNDS.top
]

print("Analysis cells:", len(grid_all))
print("Climate grid:", SHAPE)
print("Output directory:", OUT)

# ================================================================
# FIGURE R1 — BASIN COVERAGE AND FRACTIONAL CHANGE
# ================================================================

heading("FIGURE R1 — FRACTIONAL LAND-COVER CHANGE")

coverage = load_csv(
    S5 / "Main_Analysis_Grid_Coverage90.csv"
)

all_fraction = fraction_all.copy()

# Full basin for mapping, not just the >=90% analysis mask.
map_coverage = make_grid_map(
    all_fraction["basin_area_km2"].to_numpy() /
    all_fraction["full_grid_area_km2"].to_numpy()
    if "full_grid_area_km2" in all_fraction.columns
    else np.full(len(all_fraction), np.nan),
    all_fraction.grid_row,
    all_fraction.grid_col,
    SHAPE
)

# Use Step 5 GeoTIFF as authoritative for coverage.
with rasterio.open(
    S5 / "Niger_Grid_Coverage_QC.tif"
) as src:
    map_coverage = src.read(1).astype(float)
    map_changed = src.read(2).astype(float)

    map_coverage[map_coverage == src.nodata] = np.nan
    map_changed[map_changed == src.nodata] = np.nan

fig, axes = plt.subplots(
    1, 2,
    figsize=(13.5, 5.7),
    constrained_layout=True
)

plot_map(
    axes[0], map_coverage, EXTENT,
    "(a) Basin coverage per climate-grid cell",
    "Basin coverage fraction",
    vmax=1
)

plot_map(
    axes[1], map_changed, EXTENT,
    "(b) Land-cover change fraction, 1992–2020",
    "Changed fraction of basin-covered area",
    vmax=1
)

save_figure(
    fig,
    "R1_Fractional_Land_Cover_Change"
)

# ================================================================
# FIGURE R2 — MAJOR LAND-COVER TRANSITIONS
# ================================================================

heading("FIGURE R2 — TRANSITION COMPOSITION")

names = {
    1: "Cropland",
    2: "Forest/tree cover",
    3: "Shrubland",
    4: "Grassland",
    5: "Sparse vegetation",
    6: "Flooded vegetation",
    7: "Urban",
    8: "Bare land",
    9: "Water"
}

tr = transition_summary.copy()

tr["transition_code"] = tr[
    "transition_code"
].astype(int)

tr["transition"] = tr["transition_code"].map(
    lambda x:
    names.get(x // 100, str(x // 100)) +
    " → " +
    names.get(x % 100, str(x % 100))
)

tr = tr.sort_values(
    "transition_area_km2",
    ascending=False
)

total_changed = tr[
    "transition_area_km2"
].sum()

tr["percentage_of_changed_area"] = (
    100 * tr["transition_area_km2"] /
    total_changed
)

save_table(
    tr,
    "T1_Land_Cover_Transition_Composition"
)

top = tr.head(12).iloc[::-1]

fig, ax = plt.subplots(
    figsize=(11.5, 7),
    constrained_layout=True
)

bars = ax.barh(
    top["transition"],
    top["transition_area_km2"],
    color="#326DA8"
)

ax.set_xlabel("Mapped transition area (km²)")
ax.set_title(
    "Twelve largest land-cover transitions, "
    "Niger River Basin (1992–2020)"
)

ax.grid(
    axis="x",
    linestyle=":",
    alpha=0.25
)
ax.set_axisbelow(True)

for bar, p in zip(
    bars,
    top["percentage_of_changed_area"]
):
    ax.text(
        bar.get_width() + 300,
        bar.get_y() + bar.get_height()/2,
        f"{p:.1f}%",
        va="center",
        fontsize=8
    )

ax.set_xlim(
    0,
    top["transition_area_km2"].max() * 1.22
)

save_figure(
    fig,
    "R2_Major_Land_Cover_Transitions"
)

# ================================================================
# TABLE T2 — BOUNDARY-COVERAGE SENSITIVITY
# ================================================================

heading("TABLE T2 — BOUNDARY SENSITIVITY")

save_table(
    boundary_sensitivity,
    "T2_Boundary_Coverage_Sensitivity"
)

# ================================================================
# FIGURE R3 — FULL-BASIN CHIRPS PRECIPITATION BIAS
# ================================================================

heading("FIGURE R3 — CHIRPS PRECIPITATION BENCHMARK")

# Steps 10 used an en dash for the period label.
p = chirps.loc[
    chirps["period"].astype(str).str.replace(
        "–", "-", regex=False
    ) == "1995-2005"
].copy()

assert len(p) == len(grid_all)

grid_locations = grid_all[
    ["grid_cell_id", "grid_row", "grid_col"]
]

p = p.merge(
    grid_locations,
    on="grid_cell_id",
    validate="one_to_one"
)

fig, axes = plt.subplots(
    1, 2,
    figsize=(13.5, 5.7),
    constrained_layout=True
)

for ax, column, title in [
    (
        axes[0],
        "ERA5_bias_mm",
        "(a) ERA5 minus CHIRPS"
    ),
    (
        axes[1],
        "CMCC_bias_mm",
        "(b) CMCC-ESM2 minus CHIRPS"
    )
]:

    image = make_grid_map(
        p[column],
        p.grid_row,
        p.grid_col,
        SHAPE
    )

    plot_map(
        ax, image, EXTENT,
        title,
        "Mean annual precipitation bias (mm/year)",
        diverging=True,
        vmax=750
    )

save_figure(
    fig,
    "R3_CHIRPS_Referenced_Precipitation_Bias"
)

# ================================================================
# FIGURE R4 — REGIONAL PRECIPITATION-BIAS STRUCTURE
# ================================================================

heading("FIGURE R4 — LATITUDINAL PRECIPITATION BIAS")

zone = chirps_zones.copy()

fig, ax = plt.subplots(
    figsize=(11, 5.8),
    constrained_layout=True
)

x = np.arange(len(zone))
width = 0.34

ax.bar(
    x - width/2,
    zone["ERA5_bias_mm"],
    width,
    label="ERA5 − CHIRPS",
    color="#326DA8"
)

ax.bar(
    x + width/2,
    zone["CMCC_bias_mm"],
    width,
    label="CMCC-ESM2 − CHIRPS",
    color="#D27742"
)

ax.axhline(0, color="black", linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(zone["latitude_zone"])
ax.set_ylabel("Mean annual precipitation bias (mm/year)")
ax.set_title(
    "Geographic structure of precipitation bias, 1995–2005"
)
ax.legend(frameon=False)
ax.grid(axis="y", linestyle=":", alpha=0.3)
ax.set_axisbelow(True)

save_figure(
    fig,
    "R4_Latitudinal_Precipitation_Bias"
)

save_table(
    chirps_metrics,
    "T3_Full_Basin_CHIRPS_Validation"
)

save_table(
    chirps_zones,
    "T3B_CHIRPS_Latitudinal_Zones"
)

# ================================================================
# FIGURE R5 — SPATIAL CONFOUNDING DIAGNOSTIC
# ================================================================

heading("FIGURE R5 — GEOGRAPHIC ADJUSTMENT")

reg = regression.copy()

reg["figure_label"] = (
    reg["outcome"] + "\n" + reg["model"]
)

est = reg[
    "change_coefficient_mm_per_10pp"
].to_numpy(dtype=float)

low = reg[
    "ci95_lower"
].to_numpy(dtype=float)

high = reg[
    "ci95_upper"
].to_numpy(dtype=float)

fig, ax = plt.subplots(
    figsize=(11, 7.2),
    constrained_layout=True
)

y = np.arange(len(reg))

ax.errorbar(
    est,
    y,
    xerr=[
        est - low,
        high - est
    ],
    fmt="o",
    color="#225E94",
    ecolor="#225E94",
    capsize=4,
    linewidth=1.5,
    markersize=5
)

ax.axvline(
    0,
    color="#333333",
    linewidth=1,
    linestyle="--"
)

ax.set_yticks(y)
ax.set_yticklabels(reg["figure_label"])
ax.invert_yaxis()

ax.set_xlabel(
    "Precipitation-bias association "
    "(mm/year per 10-percentage-point land-cover change)"
)

ax.set_title(
    "Land-cover association before and after geographic adjustment"
)

ax.grid(axis="x", linestyle=":", alpha=0.3)
ax.set_axisbelow(True)

save_figure(
    fig,
    "R5_Spatial_Confounding_Regression"
)

save_table(
    regression,
    "T4_Spatial_Association_Bootstrap"
)

save_table(
    outlier,
    "T4B_Outlier_Sensitivity"
)

# ================================================================
# FIGURE R6 — FIVE VERIFIED CLIMATE-EXTREME BIASES
# ================================================================

heading("FIGURE R6 — FIVE VERIFIED CLIMATE INDICES")

indices5 = [
    "TXx",
    "TNx",
    "R10mm",
    "PRCPTOT",
    "CDD"
]

units5 = {
    "TXx": "°C",
    "TNx": "°C",
    "R10mm": "days/year",
    "PRCPTOT": "mm/year",
    "CDD": "days"
}

fig, axes = plt.subplots(
    2, 3,
    figsize=(16, 10),
    constrained_layout=True
)

axes = axes.ravel()

for i, index in enumerate(indices5):

    im = make_grid_map(
        extreme_spatial[f"{index}_bias"],
        extreme_spatial["grid_row"],
        extreme_spatial["grid_col"],
        SHAPE
    )

    plot_map(
        axes[i], im, EXTENT,
        f"({chr(97+i)}) {index}",
        f"CMCC − ERA5 ({units5[index]})",
        diverging=True
    )

axes[5].axis("off")

fig.suptitle(
    "Recomputed climate-extreme discrepancies, 1995–2005",
    fontsize=15
)

save_figure(
    fig,
    "R6_Five_Verified_Climate_Extreme_Bias_Maps"
)

save_table(
    extreme_five,
    "T5_Five_Verified_Extreme_Index_Biases"
)

save_table(
    original_audit,
    "T5B_Original_Recalculated_Index_Audit"
)

save_table(
    extreme_latitude,
    "T7_Latitudinal_Extreme_Biases"
)

# ================================================================
# FIGURE R7 — FOUR PERCENTILE-BASED INDEX BIASES
# ================================================================

heading("FIGURE R7 — FOUR PERCENTILE-BASED INDICES")

# Primary diagnostic: 1981–1994 threshold reference.
reference = "independent_1981_1994"

temp = percentile_spatial.loc[
    percentile_spatial["baseline"] == reference
].copy()

assert len(temp) == len(grid_all)

temp = temp.merge(
    grid_locations,
    on="grid_cell_id",
    validate="one_to_one"
)

indices4 = [
    "TX90p",
    "TN90p",
    "WSDI",
    "CSDI"
]

units4 = {
    "TX90p": "percentage points",
    "TN90p": "percentage points",
    "WSDI": "days/year",
    "CSDI": "days/year"
}

fig, axes = plt.subplots(
    2, 2,
    figsize=(13, 10),
    constrained_layout=True
)

for i, index in enumerate(indices4):

    ax = axes.ravel()[i]

    im = make_grid_map(
        temp[f"{index}_bias"],
        temp.grid_row,
        temp.grid_col,
        SHAPE
    )

    plot_map(
        ax, im, EXTENT,
        f"({chr(97+i)}) {index}",
        f"CMCC − ERA5 ({units4[index]})",
        diverging=True
    )

fig.suptitle(
    "Percentile-based temperature extremes: "
    "1981–1994 reference, 1995–2005 evaluation",
    fontsize=14
)

save_figure(
    fig,
    "R7_Percentile_Temperature_Extreme_Bias_Maps"
)

# ================================================================
# FIGURE R8 — THRESHOLD REFERENCE-PERIOD SENSITIVITY
# ================================================================

heading("FIGURE R8 — REFERENCE-PERIOD SENSITIVITY")

summ = percentile_summary.copy()

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 9),
    constrained_layout=True
)

for ax, index in zip(
    axes.ravel(),
    indices4
):

    sub = summ.loc[
        summ["index"] == index
    ]

    a = sub.loc[
        sub["baseline"] == "independent_1981_1994",
        "CMCC_minus_ERA5"
    ].iloc[0]

    b = sub.loc[
        sub["baseline"] == "overlapping_1981_2005",
        "CMCC_minus_ERA5"
    ].iloc[0]

    ax.bar(
        ["1981–1994\nindependent", "1981–2005\noverlapping"],
        [a, b],
        color=["#326DA8", "#D27742"],
        width=0.56
    )

    ax.axhline(
        0, color="black", linewidth=1
    )

    ax.set_title(index)
    ax.set_ylabel(
        f"CMCC − ERA5 ({units4[index]})"
    )

    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.28
    )
    ax.set_axisbelow(True)

fig.suptitle(
    "Sensitivity of percentile-based extremes "
    "to the temperature reference period",
    fontsize=14
)

save_figure(
    fig,
    "R8_Temperature_Reference_Period_Sensitivity"
)

save_table(
    percentile_summary,
    "T6_Percentile_Index_Baseline_Sensitivity"
)

# ================================================================
# FIGURE R9 — ERA5-LAND SURFACE CONDITIONS
# ================================================================

heading("FIGURE R9 — ERA5-LAND SURFACE CONDITIONS")

# These are climatological reanalysis fields.
# They are not measured land-cover feedback responses.

if SURFACE.exists():

    with rasterio.open(SURFACE) as ds:

        surface_names = [
            str(n) if n else f"Band_{i+1}"
            for i, n in enumerate(ds.descriptions)
        ]

        surface_stack = ds.read(masked=True)

        surface_extent = [
            ds.bounds.left,
            ds.bounds.right,
            ds.bounds.bottom,
            ds.bounds.top
        ]

        surface_rows = []

        fig, axes = plt.subplots(
            2, 3,
            figsize=(16, 10),
            constrained_layout=True
        )

        for i, name in enumerate(surface_names[:5]):

            ax = axes.ravel()[i]

            values = np.asarray(
                np.ma.filled(
                    surface_stack[i], np.nan
                ),
                dtype=float
            )

            if not np.isfinite(values).any():
                ax.axis("off")
                continue

            im = ax.imshow(
                np.ma.masked_invalid(values),
                origin="upper",
                extent=surface_extent,
                cmap="viridis",
                interpolation="nearest"
            )

            ax.set_title(
                f"({chr(97+i)}) {name}"
            )
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")

            ax.grid(
                linestyle=":",
                alpha=0.22,
                linewidth=0.45
            )

            cb = fig.colorbar(
                im, ax=ax,
                shrink=0.75
            )

            # Units must be confirmed from archived processing.
            cb.set_label(
                f"{name} (source raster units)"
            )

            valid = values[np.isfinite(values)]

            surface_rows.append({
                "variable": name,
                "minimum": float(np.min(valid)),
                "mean_unweighted": float(np.mean(valid)),
                "maximum": float(np.max(valid)),
                "valid_pixels": int(len(valid)),
                "units": "Pending provenance verification"
            })

        axes.ravel()[-1].axis("off")

        fig.suptitle(
            "ERA5-Land surface conditions, 1995–2005",
            fontsize=15
        )

        save_figure(
            fig,
            "R9_ERA5Land_Surface_Conditions"
        )

        save_table(
            pd.DataFrame(surface_rows),
            "T9_Surface_Conditions_Metadata_Audit"
        )

else:
    issues.append(
        "ERA5-Land surface raster not found; R9 omitted."
    )
    print("ERA5-Land raster not found; R9 omitted.")

# ================================================================
# TABLE T8 — CLIMATE SOURCE-DATA AUDIT
# ================================================================

heading("TABLE T8 — REPRODUCIBILITY AUDIT")

audit_summary = (
    daily_audit
    .groupby(
        ["product", "variable"],
        as_index=False
    )
    .agg(
        annual_files=("year", "count"),
        years_min=("year", "min"),
        years_max=("year", "max"),
        calendar_complete=("calendar_valid", "all")
    )
)

save_table(
    audit_summary,
    "T8_Climate_Data_Reproducibility"
)

# ================================================================
# 3. COMBINE ALL TABLES INTO ONE EXCEL WORKBOOK
# ================================================================

heading("CONSOLIDATED EXCEL TABLE WORKBOOK")

workbook = TAB / "Niger_Basin_Revised_Tables.xlsx"

with pd.ExcelWriter(
    workbook,
    engine="openpyxl"
) as writer:

    for name, data in generated_tables.items():

        # Excel worksheet names are limited to 31 characters.
        sheet = name[:31]

        data.to_excel(
            writer,
            sheet_name=sheet,
            index=False
        )

        ws = writer.sheets[sheet]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        from openpyxl.styles import (
            Font, PatternFill, Alignment
        )
        from openpyxl.utils import get_column_letter

        for cell in ws[1]:
            cell.font = Font(
                bold=True,
                color="FFFFFF"
            )
            cell.fill = PatternFill(
                fill_type="solid",
                fgColor="17365D"
            )
            cell.alignment = Alignment(
                wrap_text=True
            )

        for column in ws.columns:

            letter = get_column_letter(
                column[0].column
            )

            max_len = max(
                len(str(cell.value or ""))
                for cell in list(column)[:500]
            )

            ws.column_dimensions[
                letter
            ].width = min(
                max(max_len + 2, 12),
                40
            )

print("Saved workbook:", workbook)

# ================================================================
# 4. FIGURE AND TABLE INVENTORY
# ================================================================

heading("PACKAGE INVENTORY")

figure_inventory = pd.DataFrame({
    "figure": generated_figures,
    "PNG_file": [
        f"01_Figures/{n}.png"
        for n in generated_figures
    ],
    "PDF_file": [
        f"01_Figures/{n}.pdf"
        for n in generated_figures
    ]
})

table_inventory = pd.DataFrame({
    "table": list(generated_tables),
    "CSV_file": [
        f"02_Tables/{n}.csv"
        for n in generated_tables
    ]
})

figure_inventory.to_csv(
    AUD / "Figure_Inventory.csv",
    index=False
)

table_inventory.to_csv(
    AUD / "Table_Inventory.csv",
    index=False
)

manifest = {
    "project": "Niger Basin Reviewer 1 Revision",
    "figure_count": len(generated_figures),
    "table_count": len(generated_tables),
    "figures": generated_figures,
    "tables": list(generated_tables),
    "issues": issues,
    "scientific_constraints": [
        "Land-cover changes are from 1992 to 2020",
        "Climate biases are evaluated over 1995 to 2005",
        "Land-cover associations are noncausal",
        "1981-1994 temperature baseline is only 14 years",
        "1981-2005 sensitivity overlaps evaluation years",
        "In-base percentile bootstrap has not been implemented",
        "ERA5-Land surface conditions are not proof of feedback",
        "Precipitation and heat-flux provenance must be documented",
        "Single-model CMCC results do not establish multi-model robustness"
    ]
}

with open(
    AUD / "Revision_Package_Manifest.json",
    "w"
) as f:
    json.dump(manifest, f, indent=2)

print("\n" + "=" * 78)
print("MANUSCRIPT REVISION PACKAGE GENERATED")
print("=" * 78)

print("Publication figures:", len(generated_figures))
print("Exported tables:", len(generated_tables))
print("Excel workbook:", workbook)

print("\nFIGURE FILES:")
for name in generated_figures:
    print(" ", name)

print("\nTABLE FILES:")
for name in generated_tables:
    print(" ", name)

if issues:
    print("\nISSUES:")
    for issue in issues:
        print(" ", issue)

print("\nSAVED TO:")
print(OUT)
print("\nOriginal research files were not modified.")
