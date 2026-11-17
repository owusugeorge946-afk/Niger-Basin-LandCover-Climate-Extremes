# Niger River Basin — land cover and climate-product comparisons

**Reproducibility code for the RSASE manuscript.** Original study: land-cover transitions (1992–2020) and ERA5 / CMCC-ESM2 extreme-index comparisons (1995–2005), with CHIRPS validation and ERA5-Land surface climatologies.

> **Interpretive restriction:** Associations reported in this study are **spatial, observational, and noncausal**. The 1992–2020 land-cover transition period does not align with the 1995–2005 climate evaluation period. The CMCC-ESM2 NEX-GDDP-CMIP6 product is statistically downscaled/bias-corrected; these results are not controlled land–atmosphere perturbation experiments.

## What this repository contains

- `run.py` — preflight validator and publication-build entry point.
- `scripts/build_figures_tables.py` — the manuscript's **executed original figure/table code**, adapted only to support environment-selected Drive input/output paths and disable interactive plotting by default.
- `scripts/step04_*.py` through `scripts/step13_*.py` — source code extracted from the authors' original Colab notebook. The corrected Step 9 script is included rather than the superseded initial attempt. These scripts retain original Colab-specific paths and **have not all been retested as standalone scripts**.
- `docs/input_manifest.json` — required filenames and audit expectations.
- `docs/METHODS_AND_LIMITATIONS.md` — scientific method, validation and noncausal interpretation.
- `docs/DATA_AVAILABILITY_TEMPLATE.md` — data deposition checklist. No repository DOI or public data URL has been invented.
- `requirements.txt` — Python dependency ranges; pin versions for an exact computational environment.

**This repository intentionally contains no original geospatial rasters, no third-party climate data, no personal files, and no saved Earth Engine authorization tokens.** The `.gitignore` excludes these data types. To reproduce the study, the user needs permitted access to the datasets or archived derived CSVs in their own Drive.

## Reproduce figures and tables in Google Colab

1. Create a Colab notebook, mount the Google Drive account that contains `Niger_Basin_Reviewer1_Revision` and the `Niger_ERA5_Daily_1995_2005` and `Niger_Figure_6_Data` directories.
2. Clone this repository (replace the placeholder with the **actual repository URL after you upload it**):

   ```bash
   !git clone https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
   %cd YOUR-REPOSITORY
   !pip install -r requirements.txt
   ```

3. Run:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

   ```bash
   !python run.py --check --validate-columns
   !python run.py --build
   ```

   All generated outputs are placed in:

   `/content/drive/MyDrive/Niger_Basin_Reviewer1_Revision/RSASE_Final_Submission/`

   including **nine figures** (900-dpi PNG + vector PDF), **twelve tables** (CSV, plus consolidated workbook), and **audit inventories**.

4. If your Drive folders are elsewhere, use `--drive-root`, `--analysis-base`, or `--output`.

   ```bash
   !python run.py --build --analysis-base '/content/drive/MyDrive/Niger_Basin_Reviewer1_Revision' --output '/content/drive/MyDrive/Niger_Basin_Reviewer1_Revision/RSASE_Final_Submission'
   ```

The helper is designed to **stop rather than fabricate or silently replace missing sources**. `--check` only verifies presence and selected columns; it does not validate every transformation. The full publication script is the one previously executed successfully in Colab, but the repository wrapper has not been run against a live Drive in this environment.

## Re-run upstream analyses (optional)

The source scripts for Steps 4–13 are provided for methodological transparency. **You do not need to rerun them** to reproduce figures from the validated archived outputs. If upstream datasets are changed, run source stages in numeric order, respecting the separate Step 10A Earth Engine export and Step 10B analysis. The source scripts contain study-specific directory structures and may require Google Earth Engine authentication (project `nana213`) and adjustment to your authorized project.

## Source-to-raster provenance checks (completed separately)

- **Step 15:** 108 paired comparisons (ERA5 and CMCC-ESM2 TX/TN/PR; 3 dates × 6 grid cells), with differences at expected GeoTIFF rounding precision.
- **Step 16:** six matched ERA5-Land sample pixels; equal-month climatology verified for albedo, radiation, sensible/latent heat and soil water. Radiation is positive downward; sensible/latent fluxes are upward-positive after a sign reversal. The six sample tests do not certify every pixel or prove causality.

Detailed numerical audit tables remain in the authors' Drive archives and should be deposited with an appropriate repository if permissible. Document provenance and licenses of ERA5, CHIRPS, ESA land cover and NEX-GDDP before redistributing source files.

## Ethical use, limitations and publication

This repository is for reproducibility, **not** a claim of scientifically independent validation of land-cover forcing. The 1981–1994 percentile reference is 14 years; the 1981–2005 sensitivity reference overlaps the 1995–2005 evaluation period and is **not** a canonical ETCCDI in-base bootstrap. Single-model CMCC comparisons do not establish multi-model robustness. The release does not imply that the data are publicly deposited or that the manuscript was accepted.

### Licensing

No software license has been assigned automatically. Before making the repository public, the authors should agree on a suitable license (e.g. MIT or BSD for original code) and confirm third-party data and code rights. Without a license, standard copyright restrictions apply.

### Suggested citation

Owusu Amoah, G., Mostafiz, R.B., Olawoyin, R., & Quayson, F. *Niger River Basin land-cover and climate-product comparison scripts* [software]. GitHub repository (unpublished, version to be assigned). **Do not invent a DOI or describe the manuscript as published.**
