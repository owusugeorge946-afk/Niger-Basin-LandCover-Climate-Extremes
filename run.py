#!/usr/bin/env python3
"""Preflight and publication build for the verified Niger Basin study.

Examples:
    python run.py --check --drive-root /content/drive/MyDrive
    python run.py --build --drive-root /content/drive/MyDrive
    python run.py --check --drive-root /content/drive/MyDrive --validate-columns

No source data are modified. All publication outputs go to RSASE_Final_Submission.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
MANIFEST = json.loads((REPO / 'docs' / 'input_manifest.json').read_text(encoding='utf-8'))


def preflight(root: Path, base: Path, validate_columns: bool = False) -> tuple[bool, list[str]]:
    missing = []
    for subdir, files in MANIFEST['required_files'].items():
        for filename in files:
            path = base / subdir / filename
            if not path.is_file(): missing.append(str(path))
    for relative in MANIFEST['external_files']:
        path = root / relative
        if not path.is_file(): missing.append(str(path))
    if validate_columns and not missing:
        import pandas as pd
        expectations = {
            '05_Grid_Coverage_QC/Main_Analysis_Grid_Coverage90.csv': {'grid_cell_id','grid_row','grid_col'},
            '12_Recomputed_Absolute_Extremes/Recomputed_Climatological_Extreme_Biases.csv': {'grid_cell_id'},
            '04_Basin_Adjusted_Fractions/Transition_Area_Summary.csv': {'transition_code','transition_area_km2'},
        }
        for filename, columns in expectations.items():
            d = pd.read_csv(base / filename,nrows=1)
            lacking=columns-set(d.columns)
            if lacking: missing.append(f'{filename} missing columns: {sorted(lacking)}')
        grid = pd.read_csv(base/'05_Grid_Coverage_QC'/'Main_Analysis_Grid_Coverage90.csv', usecols=['grid_cell_id'])
        if len(grid)!=MANIFEST['main_grid_cells'] or not grid.grid_cell_id.is_unique:
            missing.append('Expected 2,661 unique analysis grid cells')
    return not missing, missing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument('--check',action='store_true',help='Print an input inventory without generating figures')
    action.add_argument('--build',action='store_true',help='Regenerate all nine figures and twelve tables')
    p.add_argument('--drive-root',type=Path,default=Path(os.environ.get('NIGER_DRIVE_ROOT','/content/drive/MyDrive')))
    p.add_argument('--analysis-base',type=Path,default=None)
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--dpi',type=int,default=900)
    p.add_argument('--validate-columns',action='store_true',help='Check selected known column names and analysis cell count')
    a=p.parse_args()
    root=a.drive_root.expanduser().resolve()
    base=(a.analysis_base or root/'Niger_Basin_Reviewer1_Revision').expanduser().resolve()
    output=(a.output or base/'RSASE_Final_Submission').expanduser().resolve()
    print(f'Drive root: {root}\nAnalysis base: {base}\nOutput: {output}')
    ok, problems=preflight(root,base,validate_columns=a.validate_columns or a.build)
    if not ok:
        print(f'\nSTOP: {len(problems)} required inputs or schema checks failed:', file=sys.stderr)
        for i in problems:print(' - '+i,file=sys.stderr)
        print('Review docs/input_manifest.json and your Google Drive paths.',file=sys.stderr)
        return 2
    print('PASS: required input files'+(' and sampled schemas' if (a.validate_columns or a.build) else '')+' found.')
    if a.check:return 0
    output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update({'NIGER_DRIVE_ROOT':str(root),'NIGER_ANALYSIS_BASE':str(base),'NIGER_OUTPUT_DIR':str(output),'NIGER_FIGURE_DPI':str(a.dpi),'MPLBACKEND':'Agg','NIGER_SHOW_FIGURES':'0'})
    print('Generating publication package from the original tested notebook script ...')
    proc=subprocess.run([sys.executable,str(REPO/'scripts'/'build_figures_tables.py')],env=env,check=False)
    if proc.returncode:
        print('Publication build failed: inspect the traceback above.',file=sys.stderr)
        return proc.returncode
    figs=list((output/'01_Figures').glob('R*.png'))
    pdfs=list((output/'01_Figures').glob('R*.pdf'))
    tables=list((output/'02_Tables').glob('T*.csv'))
    print(f'Build finished: {len(figs)} PNG, {len(pdfs)} PDF, {len(tables)} CSV tables')
    if len(figs)!=9 or len(pdfs)!=9 or len(tables)!=12:
        print('WARNING: expected 9 figures and 12 tables. Inspect output and report. ', file=sys.stderr)
        return 3
    return 0

if __name__=='__main__':raise SystemExit(main())
