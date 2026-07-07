# PET/CT Body Composition Extractor User Guide

This folder contains a Windows-ready local research application for PET/CT-derived body-composition parameter extraction.

The software can:

- Load anonymized CT and PET NIfTI files
- Run TotalSegmentator tissue segmentation
- Extract CT volume, CT density, and PET liver-normalized SUR parameters
- Optionally calculate research-only OS/PFS high-low risk groups from extracted body-composition features plus clinical inputs
- Display results by tissue compartment
- Export feature, risk-score, and combined CSV tables

## One-Click Start

On Windows, double-click:

```text
one_click_start_petct_bodycomp_windows.bat
```

You can also double-click:

```text
start_petct_bodycomp_windows.bat
```

## First-Run Behavior

The launcher automatically performs these steps:

1. Searches for Anaconda or Miniconda.
2. If conda is available, creates an isolated environment named:

```text
petct_bodycomp
```

3. If conda is not available, creates a local virtual environment in this project folder:

```text
.venv
```

4. Installs dependencies from `requirements.txt`.
5. Starts the local desktop window.

The first run may take a long time because Python packages and TotalSegmentator model weights may need to be downloaded.

## Recommended Computer Configuration

- Windows 10 or Windows 11
- Anaconda or Miniconda recommended
- Python 3.10 or later if conda is not installed
- Internet access during the first run
- At least 16 GB RAM recommended
- GPU optional; CPU mode is supported but slower

## Input Files

Prepare the following inputs:

- CT NIfTI file in `.nii` or `.nii.gz` format
- PET NIfTI file in `.nii` or `.nii.gz` format
- Body weight in kg
- Injected activity in MBq
- Optional height in cm; required when research risk scoring is enabled

For research risk scoring, also prepare:

- Age
- Sex
- Clinical stage
- Histology
- Optional tumor SUVmax

If TotalSegmentator has already been run, enable `Skip segmentation` and provide existing mask folders:

- `seg_total`
- `seg_4tissue`

If automatic segmentation fails, the software checks the configured output folder for a complete existing mask pair whose folder name matches the CT case identifier. This is only a rerun convenience for a case that was segmented previously. For new cases, manually select matching masks with `Skip segmentation` or rerun segmentation on a more suitable workstation.

## Output Parameters

The software exports 24 body-composition parameters:

- Skeletal muscle: volume index, density mean, density SD, SUR mean, SUR SD
- Subcutaneous fat: volume index, density mean, density SD, SUR mean, SUR SD
- Intermuscular fat: volume index, IMAT/SM ratio, density mean, density SD, SUR mean, SUR SD
- Torso fat: volume index, torso-fat/SAT ratio, density mean, density SD, SUR mean, SUR SD
- Bone: density mean, density SD

The exported CSV file is named:

```text
petct_body_composition_parameters.csv
```

When research risk scoring is enabled, the software also exports:

```text
petct_research_risk_scores.csv
petct_body_composition_and_research_risk_scores.csv
```

The risk-score table includes OS and PFS risk scores, high-low risk groups using the locked development-cohort median reference, model-estimated event risks at the stored time horizons, and external validation C-index metadata from the frozen model specification.

Risk-score outputs are for research stratification only. They are not diagnoses, survival guarantees, treatment recommendations, or substitutes for clinical review.

## Risk Prediction Workflow

1. Open the `Risk Prediction` tab.
2. Enable `Calculate OS/PFS high-low risk groups after feature extraction`.
3. Enter age, sex, clinical stage, histology, and tumor SUVmax if available.
4. Enter height in the `Input Data` tab because the model uses height-normalized body-composition features.
5. Run the pipeline normally.

## Privacy Notes

Use only anonymized NIfTI files. Do not process DICOM files that contain patient names, medical record numbers, accession numbers, or other identifiers unless the data are handled on an institutionally approved secure computer.

This is a local research-use extraction and risk-score tool. Output CSV files and generated masks are saved under a timestamped folder in `outputs` by default. Do not use the exported risk scores for treatment decisions.

## Troubleshooting

### The first launch is slow

The first launch may need to install Python packages and download TotalSegmentator model weights. Later launches should be faster.

### TotalSegmentator reports insufficient memory

Enable these options in the Segmentation Options tab:

```text
Force split (low RAM mode)
Fast mode (3 mm resolution)
```

Close other memory-intensive applications before rerunning segmentation.

### Automatic segmentation fails on CPU

Windows CPU-only runs can fail before TotalSegmentator writes the final masks. If matching masks already exist under `outputs`, the app can reuse them automatically. Otherwise, enable `Skip segmentation` and provide matching `seg_total` and `seg_4tissue` folders, or run the case on a GPU workstation.

### GPU mode fails

Disable:

```text
Use GPU for segmentation
```

Then rerun the analysis in CPU mode.

### The desktop window does not open

Run the dependency check from a terminal in this folder:

```bash
python bootstrap_windows.py --check-only
```

### Check dependencies without launching the app

Open a terminal in this folder and run:

```bash
python bootstrap_windows.py --check-only
```
