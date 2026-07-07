# -*- coding: utf-8 -*-
"""
Feature extraction from CT and PET NIfTI images with tissue masks.
Produces PET/CT body-composition parameters for the local extraction prototype.
"""

import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from typing import Dict, Optional, Tuple, List

# Tissue mask name mapping (TotalSegmentator 4tissue output)
TISSUE_MASK_MAP = {
    "skeletal_muscle": "SM",
    "subcutaneous_fat": "SAT",
    "intermuscular_fat": "IMAT",
    "torso_fat": "TAT",
}

# Bone keywords for aggregating bone volume (from whole-body segmentation)
BONE_KEYWORDS = [
    "vertebrae", "rib", "hip", "femur", "humerus",
    "clavicula", "scapula", "skull", "sacrum", "sternum",
    "costal_cartilages",
]

# Ordered feature names expected by models (full set)
ALL_FEATURE_NAMES = [
    "SM_volume_index", "SM_density_mean", "SM_density_sd",
    "SAT_volume_index", "SAT_density_mean", "SAT_density_sd",
    "IMAT_volume_index", "IMF_ratio", "IMAT_density_mean", "IMAT_density_sd",
    "TAT_volume_index", "TF_SAT_ratio", "TAT_density_mean", "TAT_density_sd",
    "Bone_density_mean", "Bone_density_sd",
    "SM_SURmean", "SM_SURsd", "SAT_SURmean", "SAT_SURsd",
    "IMAT_SURmean", "IMAT_SURsd", "TAT_SURmean", "TAT_SURsd",
]

# Additional extracted values used internally by the Figure 6 risk model.
# These are kept separate from the standard exported 24-parameter panel.
MODEL_SUPPORT_FEATURE_NAMES = [
    "SM_volume",
]


def load_nifti(filepath: str, dtype=None):
    """Load NIfTI file; return data, affine, header."""
    try:
        img = nib.load(filepath)
        data = img.get_fdata()
        if dtype is not None:
            data = data.astype(dtype)
        return data, img.affine, img.header
    except Exception as e:
        raise RuntimeError(f"Error loading {filepath}: {e}") from e


def resample_mask_to_target(mask_path: str, target_img: nib.Nifti1Image) -> np.ndarray:
    """Resample a binary mask NIfTI to match the target image space (nearest-neighbor)."""
    from nibabel.processing import resample_from_to
    mask_img = nib.load(mask_path)
    if mask_img.shape == target_img.shape and np.allclose(mask_img.affine, target_img.affine, atol=1e-3):
        return mask_img.get_fdata()
    resampled = resample_from_to(mask_img, target_img, order=0)
    return resampled.get_fdata()


def get_voxel_volume_mm3(header) -> float:
    """Voxel volume in mm^3 from NIfTI header."""
    zooms = header.get_zooms()[:3]
    return float(np.prod(zooms))


def extract_ct_stats_for_mask(ct_data: np.ndarray, mask_data: np.ndarray, voxel_volume_mm3: float) -> Dict:
    """Compute volume (cm3), mean HU, std HU for one mask."""
    mask_bool = (mask_data > 0).astype(bool)
    if not np.any(mask_bool):
        return {"volume_cm3": 0.0, "mean_hu": 0.0, "std_hu": 0.0}
    ct_values = ct_data[mask_bool]
    voxel_count = np.sum(mask_bool)
    volume_cm3 = (voxel_count * voxel_volume_mm3) / 1000.0
    return {
        "volume_cm3": float(volume_cm3),
        "mean_hu": float(np.mean(ct_values)),
        "std_hu": float(np.std(ct_values)) if ct_values.size > 1 else 0.0,
    }


def extract_pet_stats_for_mask(pet_data: np.ndarray, mask_data: np.ndarray) -> Dict:
    """Compute mean and std of PET values (SUV or counts) within mask."""
    mask_bool = (mask_data > 0).astype(bool)
    if not np.any(mask_bool):
        return {"mean_val": 0.0, "std_val": 0.0}
    values = pet_data[mask_bool]
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean_val": 0.0, "std_val": 0.0}
    return {
        "mean_val": float(np.mean(values)),
        "std_val": float(np.std(values)) if values.size > 1 else 0.0,
    }


def pet_to_suv(pet_data: np.ndarray, weight_kg: float, dose_mbq: float) -> np.ndarray:
    """Convert PET image in Bq/ml to SUV: SUV = (activity * weight_kg) / dose_MBq.
    Standard formula: SUV = (voxel_value_Bq_ml * weight_kg) / (dose_Bq / 1000) with dose in MBq.
    So dose_Bq = dose_mbq * 1e6, and SUV = (pet_Bq_ml * weight_kg) / (dose_mbq * 1e3).
    Common form: SUV = (pet_Bq_ml * weight_kg * 1000) / dose_mbq  [if dose in MBq].
    """
    if dose_mbq <= 0 or weight_kg <= 0:
        return pet_data.astype(np.float64)
    # SUV = (concentration_Bq_ml * weight_kg) / (dose_MBq * 1e3) so that dose in MBq -> Bq = dose_mbq*1e6, then 1e3 factor for g
    # Actually: SUV = (C * W) / D where C=Bq/ml, W=kg, D=Bq. So D = dose_mbq * 1e6. SUV = C * W / (dose_mbq*1e6)
    suv = (pet_data.astype(np.float64) * weight_kg) / (dose_mbq * 1e3)
    return np.clip(suv, 0, None)


def collect_tissue_masks(mask_dir_4tissue: str) -> Dict[str, str]:
    """Return dict tissue_key -> path for skeletal_muscle, subcutaneous_fat, intermuscular_fat, torso_fat."""
    out = {}
    for f in glob.glob(os.path.join(mask_dir_4tissue, "*.nii*")):
        name = os.path.basename(f).replace(".nii.gz", "").replace(".nii", "")
        if name in TISSUE_MASK_MAP:
            out[name] = f
    return out


def collect_bone_and_liver_masks(mask_dir_total: str) -> Tuple[List[str], Optional[str]]:
    """From TotalSegmentator 'total' output: list of bone mask paths, and liver mask path."""
    bone_paths = []
    liver_path = None
    for f in glob.glob(os.path.join(mask_dir_total, "*.nii*")):
        name = os.path.basename(f).replace(".nii.gz", "").replace(".nii", "")
        if name == "liver":
            liver_path = f
        else:
            for kw in BONE_KEYWORDS:
                if kw in name.lower():
                    bone_paths.append(f)
                    break
    return bone_paths, liver_path


def compute_ct_features(
    ct_path: str,
    mask_dir_4tissue: str,
    mask_dir_total: str,
    height_cm: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute CT-derived features: volumes, mean/SD HU, volume indices, ratios.
    SM, SAT, IMAT normalized to bone volume; TAT normalized by height^2; IMF_ratio, TF_SAT_ratio.
    """
    ct_data, _, ct_header = load_nifti(ct_path)
    voxel_vol = get_voxel_volume_mm3(ct_header)

    # 4 tissue stats
    tissue_paths = collect_tissue_masks(mask_dir_4tissue)
    tissue_stats = {}
    for mask_name, path in tissue_paths.items():
        mask_data, _, _ = load_nifti(path)
        tissue_stats[mask_name] = extract_ct_stats_for_mask(ct_data, mask_data, voxel_vol)

    # Bone aggregate from total segmentation: combined volume and CT stats over all bone voxels
    bone_paths, _ = collect_bone_and_liver_masks(mask_dir_total)
    bone_vol_cm3 = 0.0
    bone_combined_mask = None
    for bp in bone_paths:
        mask_data, _, _ = load_nifti(bp)
        mask_bool = (mask_data > 0).astype(bool)
        if not np.any(mask_bool):
            continue
        st = extract_ct_stats_for_mask(ct_data, mask_data, voxel_vol)
        bone_vol_cm3 += st["volume_cm3"]
        if bone_combined_mask is None:
            bone_combined_mask = mask_bool.copy()
        else:
            bone_combined_mask = np.logical_or(bone_combined_mask, mask_bool)
    if bone_combined_mask is not None and np.any(bone_combined_mask):
        bone_hu_values = ct_data[bone_combined_mask]
        bone_mean_hu = float(np.mean(bone_hu_values))
        bone_std_hu = float(np.std(bone_hu_values)) if bone_hu_values.size > 1 else 0.0
    else:
        bone_mean_hu = 0.0
        bone_std_hu = 0.0

    if bone_vol_cm3 <= 0:
        bone_vol_cm3 = 1.0  # avoid div by zero

    # Tissue shorthand
    sm = tissue_stats.get("skeletal_muscle", {"volume_cm3": 0, "mean_hu": 0, "std_hu": 0})
    sat = tissue_stats.get("subcutaneous_fat", {"volume_cm3": 0, "mean_hu": 0, "std_hu": 0})
    imat = tissue_stats.get("intermuscular_fat", {"volume_cm3": 0, "mean_hu": 0, "std_hu": 0})
    tat = tissue_stats.get("torso_fat", {"volume_cm3": 0, "mean_hu": 0, "std_hu": 0})

    height_m2 = None
    if height_cm is not None and height_cm > 0:
        height_m2 = (height_cm / 100.0) ** 2

    sm_vol = sm["volume_cm3"]
    sat_vol = sat["volume_cm3"]
    imat_vol = imat["volume_cm3"]
    tat_vol = tat["volume_cm3"]

    features = {}
    features["SM_volume"] = sm_vol
    # Volume indices: SM, SAT, IMAT / bone_volume
    features["SM_volume_index"] = sm_vol / bone_vol_cm3 if bone_vol_cm3 else 0.0
    features["SAT_volume_index"] = sat_vol / bone_vol_cm3 if bone_vol_cm3 else 0.0
    features["IMAT_volume_index"] = imat_vol / bone_vol_cm3 if bone_vol_cm3 else 0.0
    # TAT / height^2
    if height_m2 and height_m2 > 0:
        features["TAT_volume_index"] = tat_vol / height_m2
    else:
        features["TAT_volume_index"] = tat_vol  # fallback: raw cm3

    features["SM_density_mean"] = sm["mean_hu"]
    features["SM_density_sd"] = sm["std_hu"]
    features["SAT_density_mean"] = sat["mean_hu"]
    features["SAT_density_sd"] = sat["std_hu"]
    features["IMAT_density_mean"] = imat["mean_hu"]
    features["IMAT_density_sd"] = imat["std_hu"]
    features["TAT_density_mean"] = tat["mean_hu"]
    features["TAT_density_sd"] = tat["std_hu"]
    features["Bone_density_mean"] = bone_mean_hu
    features["Bone_density_sd"] = bone_std_hu

    features["IMF_ratio"] = (imat_vol / sm_vol) if sm_vol > 0 else 0.0
    features["TF_SAT_ratio"] = (tat_vol / sat_vol) if sat_vol > 0 else 0.0

    return features


def compute_pet_sur_features(
    pet_path: str,
    mask_dir_4tissue: str,
    mask_dir_total: str,
    weight_kg: float,
    dose_mbq: float,
    pet_is_suv: bool = False,
) -> Dict[str, float]:
    """
    Compute PET SUR (standardized uptake ratio to liver) for each tissue.
    Masks are resampled from CT space to PET space automatically.
    """
    pet_img = nib.load(pet_path)
    pet_data = pet_img.get_fdata()
    if not pet_is_suv and weight_kg > 0 and dose_mbq > 0:
        pet_data = pet_to_suv(pet_data, weight_kg, dose_mbq)

    _, liver_path = collect_bone_and_liver_masks(mask_dir_total)
    if liver_path is None:
        liver_mean = float(np.nanmedian(pet_data[pet_data > 0])) if np.any(pet_data > 0) else 1.0
        if liver_mean <= 0:
            liver_mean = 1.0
    else:
        liver_mask = resample_mask_to_target(liver_path, pet_img)
        st = extract_pet_stats_for_mask(pet_data, liver_mask)
        liver_mean = st["mean_val"]
        if liver_mean <= 0:
            liver_mean = 1.0

    tissue_paths = collect_tissue_masks(mask_dir_4tissue)
    sur_features = {}
    for mask_name, path in tissue_paths.items():
        mask_data = resample_mask_to_target(path, pet_img)
        st = extract_pet_stats_for_mask(pet_data, mask_data)
        prefix = TISSUE_MASK_MAP[mask_name]
        sur_features[f"{prefix}_SURmean"] = st["mean_val"] / liver_mean
        sur_features[f"{prefix}_SURsd"] = st["std_val"] / liver_mean if liver_mean else 0.0

    return sur_features


def apply_combat_optional(features_df: pd.DataFrame, batch: Optional[str] = None) -> pd.DataFrame:
    """Optionally apply ComBat harmonization. For single subject, pass or use batch='inference'."""
    try:
        import pycombat
        if batch is None or features_df.shape[0] < 2:
            return features_df
        # Only PET-derived columns
        pet_cols = [c for c in features_df.columns if "SUR" in c]
        if not pet_cols:
            return features_df
        # pyComBat expects a design matrix; for one row we typically skip
        return features_df
    except ImportError:
        return features_df


def extract_all_features(
    ct_path: str,
    pet_path: str,
    mask_dir_4tissue: str,
    mask_dir_total: str,
    weight_kg: float,
    dose_mbq: float,
    height_cm: Optional[float] = None,
    pet_is_suv: bool = False,
) -> pd.DataFrame:
    """
    Extract the 24 standard output features plus internal model-support values.
    Use ALL_FEATURE_NAMES when writing the public body-composition CSV.
    """
    ct_feat = compute_ct_features(ct_path, mask_dir_4tissue, mask_dir_total, height_cm)
    pet_feat = compute_pet_sur_features(
        pet_path, mask_dir_4tissue, mask_dir_total, weight_kg, dose_mbq, pet_is_suv
    )
    combined = {**ct_feat, **pet_feat}
    # Ensure order and fill missing with 0. Standard parameters remain first;
    # model-support values follow so UI exports can choose either subset.
    ordered_names = ALL_FEATURE_NAMES + MODEL_SUPPORT_FEATURE_NAMES
    row = {k: combined.get(k, 0.0) for k in ordered_names}
    return pd.DataFrame([row])[ordered_names]
