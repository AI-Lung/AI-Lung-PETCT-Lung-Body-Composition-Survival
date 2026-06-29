# PET/CT Body Composition Extractor - Windows Edition

Windows-ready local software for PET/CT-derived whole-body composition parameter extraction.

The app runs as a local Windows desktop window. It accepts anonymized CT and PET NIfTI files, performs TotalSegmentator-based tissue segmentation, extracts CT- and PET-derived body-composition parameters, displays the results, and exports the feature table as CSV.

This tool does **not** generate survival predictions, risk groups, survival curves, or treatment recommendations.

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
- Optional height in cm
- Optional existing TotalSegmentator mask folders if segmentation has already been run

Only anonymized files should be used. Do not upload or share DICOM files containing identifiers.

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

## Main Files

- `one_click_start_petct_bodycomp_windows.bat` - recommended one-click launcher
- `start_petct_bodycomp_windows.bat` - full Windows launcher
- `bootstrap_windows.py` - environment checker, package installer, and app launcher
- `app.py` - desktop GUI-compatible root entry
- `petct_bodycomp/gui.py` - main desktop window interface
- `petct_bodycomp/segmentation.py` - TotalSegmentator wrapper
- `petct_bodycomp/feature_extraction.py` - parameter extraction logic
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

This is a research-use local extraction prototype. Output CSV files and generated masks are saved under a timestamped folder in `outputs` by default. Regulated or identifiable clinical data should only be processed on institutionally approved secure computers.

## Troubleshooting

- If installation fails, check the internet connection and run the launcher again.
- If TotalSegmentator fails due to memory, enable `Fast mode` and `Force split`.
- If GPU mode fails, switch to CPU mode.
- If the window does not appear, run `python bootstrap_windows.py --check-only` in this folder and check the printed error.
