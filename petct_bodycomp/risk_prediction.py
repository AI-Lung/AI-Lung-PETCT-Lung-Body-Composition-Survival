# -*- coding: utf-8 -*-
"""
Research-only OS/PFS risk scoring for the revised Figure 6 ridge-Cox model.

The model specification is stored as JSON so the local software can calculate
risk scores without shipping source cohort data or retraining the model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


SPEC_PATH = Path(__file__).resolve().with_name("risk_model_spec.json")


def load_risk_model_spec(path: Optional[Path] = None) -> Dict:
    """Load the frozen Figure 6 model specification."""
    spec_path = path or SPEC_PATH
    with open(spec_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_clinical_inputs(
    age: float,
    sex_label: str,
    clinical_stage_label: str,
    histology_label: str,
    tumor_suvmax: Optional[float],
    spec: Optional[Dict] = None,
) -> Dict[str, object]:
    """Map reader-facing clinical inputs to the encoded fields used by the model."""
    model_spec = spec or load_risk_model_spec()
    mapping = model_spec["clinical_input_mapping"]

    tumor_missing = tumor_suvmax is None or not np.isfinite(float(tumor_suvmax))
    return {
        "Age": float(age),
        "Tumor_SUVmax": np.nan if tumor_missing else float(tumor_suvmax),
        "Tumor_SUVmax_missing": 1.0 if tumor_missing else 0.0,
        "Gender": mapping["Gender"][sex_label],
        "Cli": mapping["Cli"][clinical_stage_label],
        "Path": mapping["Path"][histology_label],
    }


def _get_raw_value(row: Dict[str, object], name: str, endpoint_spec: Dict) -> float:
    value = row.get(name, np.nan)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = np.nan
    if not np.isfinite(value):
        value = float(endpoint_spec["medians"][name])
    return value


def _encoded_vector(row: Dict[str, object], endpoint_spec: Dict) -> Dict[str, float]:
    """Reproduce the preprocessing used by the revised Figure 6 analysis."""
    encoded = {}
    categorical_cols = set(endpoint_spec["categorical_cols"])

    for column in endpoint_spec["encoded_columns"]:
        if column in categorical_cols:
            # The final encoded design uses drop-first dummy columns, so the
            # original categorical fields should not appear directly.
            continue

        if column in endpoint_spec["continuous_cols"]:
            raw_value = _get_raw_value(row, column, endpoint_spec)
        elif "_" in column:
            base, level = column.split("_", 1)
            observed = str(row.get(base, endpoint_spec["modes"].get(base, "")))
            raw_value = 1.0 if observed == level else 0.0
        else:
            raw_value = 0.0

        mean = float(endpoint_spec["prestandardization_means"][column])
        std = float(endpoint_spec["prestandardization_stds"][column]) or 1.0
        encoded[column] = (float(raw_value) - mean) / std

    return encoded


def _partial_hazard(encoded: Dict[str, float], endpoint_spec: Dict) -> float:
    linear = 0.0
    for column in endpoint_spec["encoded_columns"]:
        value = encoded.get(column, 0.0)
        centered = value - float(endpoint_spec["lifelines_norm_mean"].get(column, 0.0))
        linear += centered * float(endpoint_spec["cox_coefficients"][column])
    return float(math.exp(float(np.clip(linear, -50.0, 50.0))))


def _event_risks(partial_hazard: float, endpoint_spec: Dict) -> Dict[str, float]:
    risks = {}
    for horizon, item in endpoint_spec["baseline_survival"].items():
        baseline_survival = float(item["baseline_survival"])
        survival = baseline_survival ** partial_hazard
        risks[f"{int(float(horizon))}_month_event_risk"] = float(np.clip(1.0 - survival, 0.0, 1.0))
    return risks


def predict_risk(
    features_df: pd.DataFrame,
    clinical_inputs: Dict[str, object],
    cutoff_reference: str = "development_median",
    spec: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Calculate OS and PFS risk scores from extracted features and clinical inputs.

    Parameters
    ----------
    features_df:
        One-row DataFrame returned by feature extraction.
    clinical_inputs:
        Dict returned by build_clinical_inputs.
    cutoff_reference:
        "development_median" or "external_validation_median".
    spec:
        Optional preloaded model specification.
    """
    if features_df.empty:
        raise ValueError("features_df is empty.")

    model_spec = spec or load_risk_model_spec()
    row = features_df.iloc[0].to_dict()
    row.update(clinical_inputs)

    results = []
    for endpoint, endpoint_spec in model_spec["endpoints"].items():
        encoded = _encoded_vector(row, endpoint_spec)
        risk_score = _partial_hazard(encoded, endpoint_spec)
        cutoffs = endpoint_spec["cutoffs"]
        selected_cutoff = cutoffs[cutoff_reference]["cutoff"]
        development_cutoff = cutoffs["development_median"]["cutoff"]
        external_cutoff = cutoffs["external_validation_median"]["cutoff"]
        risk_group = "High predicted risk" if risk_score >= selected_cutoff else "Low predicted risk"

        result = {
            "Endpoint": endpoint,
            "Risk score": risk_score,
            "Selected cutoff": float(selected_cutoff),
            "Selected cutoff reference": cutoff_reference,
            "Risk group": risk_group,
            "Development median group": "High predicted risk" if risk_score >= development_cutoff else "Low predicted risk",
            "External validation median group": "High predicted risk" if risk_score >= external_cutoff else "Low predicted risk",
            "External C-index": float(endpoint_spec["performance"]["external_c_index"]),
        }
        result.update(_event_risks(risk_score, endpoint_spec))
        results.append(result)

    return pd.DataFrame(results)

