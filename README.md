# PET/CT Body Composition Extractor - Windows Edition

Windows-ready local software for PET/CT-derived whole-body composition parameter extraction and optional research-only OS/PFS risk scoring.

The app runs as a local Windows desktop window. It accepts anonymized CT and PET NIfTI files, performs TotalSegmentator-based tissue segmentation, extracts CT- and PET-derived body-composition parameters, displays the results, and exports the feature table as CSV.

If clinical inputs are supplied, the app can also calculate exploratory OS/PFS high-low risk groups using a frozen reduced Clinical+CT+PET ridge-Cox model specification. Risk outputs are for research stratification only and are **not** treatment recommendations.

## Recommended Start Method

On Windows, double-click:

```text
one_click_start_petct_bodycomp_windows.bat
```

or:

```text
start_petct_bodycomp_windows.bat
```

## What The Launcher Does

The Windows launcher:

1. Searches for Anaconda or Miniconda.
2. If conda is available, creates a separate environment named `petct_bodycomp`.
3. If conda is not available, creates a local `.venv` environment in this project folder.
4. Installs required packages from `requirements.txt`.
5. Starts the local desktop window.

The first run may take a long time because Python packages and TotalSegmentator model weights may need to be downloaded.

## GitHub Repository Contents

This repository is intended to contain source code, launcher scripts, package metadata, and documentation only. Real patient images, NIfTI files, generated masks, logs, and run outputs are ignored by `.gitignore` and should not be uploaded.

## Minimum Requirements

- Windows 10 or Windows 11
- Anaconda or Miniconda recommended
- Python 3.10+ if conda is not installed
- Internet access during the first run
- At least 16 GB RAM recommended for whole-body segmentation
- GPU is optional; CPU mode works but is slower

## Inputs

- CT NIfTI file: `.nii` or `.nii.gz`
- PET NIfTI file: `.nii` or `.nii.gz`
- Body weight in kg
- Injected activity in MBq
- Optional height in cm; required when risk scoring is enabled
- Optional clinical inputs for research risk scoring: age, sex, clinical stage, histology, and tumor SUVmax
- Optional existing TotalSegmentator mask folders if segmentation has already been run

Only anonymized files should be used. Do not upload or share DICOM files containing identifiers.

If automatic segmentation fails, the app will look under the configured output folder for a complete existing mask pair whose folder name matches the CT case identifier. This fallback is intended for rerunning a case after segmentation was completed previously. For a new case, use `Skip segmentation` only when you can manually select the matching `seg_total` and `seg_4tissue` folders.

## Outputs

The app displays and exports 24 body-composition parameters:

- Skeletal muscle: volume index, density mean, density SD, SUR mean, SUR SD
- Subcutaneous fat: volume index, density mean, density SD, SUR mean, SUR SD
- Intermuscular fat: volume index, IMAT/SM ratio, density mean, density SD, SUR mean, SUR SD
- Torso fat: volume index, torso-fat/SAT ratio, density mean, density SD, SUR mean, SUR SD
- Bone: density mean, density SD

The exported CSV file is named:

```text
petct_body_composition_parameters.csv
```

When research risk scoring is enabled, the app also exports:

```text
petct_research_risk_scores.csv
petct_body_composition_and_research_risk_scores.csv
```

The risk-score CSV includes OS and PFS risk scores, high-low risk groups using the locked development-cohort median reference, model-estimated event risks at the stored time horizons, and external validation C-index metadata from the frozen model specification.

## Main Files

- `one_click_start_petct_bodycomp_windows.bat` - recommended one-click launcher
- `start_petct_bodycomp_windows.bat` - full Windows launcher
- `bootstrap_windows.py` - environment checker, package installer, and app launcher
- `app.py` - desktop GUI-compatible root entry
- `petct_bodycomp/gui.py` - main desktop window interface
- `petct_bodycomp/segmentation.py` - TotalSegmentator wrapper
- `petct_bodycomp/feature_extraction.py` - parameter extraction logic
- `petct_bodycomp/risk_prediction.py` - research-only OS/PFS risk-score calculation
- `petct_bodycomp/risk_model_spec.json` - frozen reduced ridge-Cox model specification
- `requirements.txt` - runtime dependencies
- `USER_GUIDE.md` - detailed English user guide

## Local Command-Line Use

After the environment is ready, the app can also be launched with:

```bash
python bootstrap_windows.py
```

Check dependencies without launching:

```bash
python bootstrap_windows.py --check-only
```

Run the desktop GUI directly:

```bash
python main.py
```

## Privacy And Regulatory Notes

This is a research-use local extraction and risk-score prototype. Output CSV files and generated masks are saved under a timestamped folder in `outputs` by default. Regulated or identifiable clinical data should only be processed on institutionally approved secure computers. Risk-score outputs are exploratory and should not be used for clinical treatment decisions.

## Troubleshooting

- If installation fails, check the internet connection and run the launcher again.
- If TotalSegmentator fails due to memory, enable `Fast mode` and `Force split`.
- On Windows CPU-only environments, TotalSegmentator may fail before producing masks. If matching masks already exist, the app can reuse them automatically; otherwise rerun on a GPU workstation or provide existing mask folders with `Skip segmentation`.
- If GPU mode fails, switch to CPU mode.
- If the window does not appear, run `python bootstrap_windows.py --check-only` in this folder and check the printed error.
