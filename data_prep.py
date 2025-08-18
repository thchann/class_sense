"""
data_prep.py
------------
Utilities for preparing labels and loading the fused MARLIN + OpenFace dataset.

Changes vs. Original:
- Added prepare_labels() to generate final_labels.csv cleanly.
- Replaced double-split logic (StratifiedShuffleSplit + train_test_split) 
  with a single deterministic train/val/test split.
- Fixed oversampling bug: in old code oversampled arrays were created 
  but then re-split and discarded. Now oversampling is applied only to train.
- Ensured MARLIN always has shape (N,1,768).
- Made paths relative (no /Users/... hardcoding).
- Added FusionSplits dataclass for cleaner structured access.
- Preserved old function name/data_loader_fusion for drop-in behavior.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Constants (replace with config.py if you want central config)
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # relative to remainder after test split


# -----------------------
# 1) Label preparation
# -----------------------
def prepare_labels(raw_csv: str, save_path: str = "data/final_labels.csv") -> None:
    """
    Convert DAiSEE AllLabels.csv to a compact file with columns: [video_id, label].
    """
    df = pd.read_csv(raw_csv)
    df.columns = [c.strip() for c in df.columns]

    # Build video_id column without .avi extension
    df["video_id"] = df["ClipID"].str.replace(".avi", "", regex=False)

    # Keep only video_id + Engagement, rename for consistency
    df = df[["video_id", "Engagement"]].rename(columns={"Engagement": "label"})

    # Drop rows without labels
    df = df.dropna(subset=["label"])

    # Ensure output folder exists
    dirn = os.path.dirname(save_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)

    df.to_csv(save_path, index=False)
    print(f"✅ Saved {save_path} with {len(df)} engagement labels")


# -----------------------
# 2) Fusion dataset API
# -----------------------
@dataclass
class FusionSplits:
    """
    A structured return object for cleaner downstream use.
    Contains train/val/test splits for MARLIN (x1), OpenFace (x2), and labels (y).
    """
    train_x1: np.ndarray
    train_x2: np.ndarray
    train_y: np.ndarray
    val_x1:   np.ndarray
    val_x2:   np.ndarray
    val_y:    np.ndarray
    test_x1:  np.ndarray
    test_x2:  np.ndarray
    test_y:   np.ndarray


def _unpack_Xy_fusion(Xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts the raw loaded fusion array into proper numpy arrays.

    Input format: each element is [video_id, marlin, openface, label].
    Ensures MARLIN has shape (1,768) for each sample.
    """
    x1_list, x2_list, y_list = [], [], []
    for clip_id, marlin, openface, label in Xy:
        marlin = np.asarray(marlin)
        if marlin.ndim == 1:
            marlin = marlin[None, :]  # ensure (1,768)
        x1_list.append(marlin)
        x2_list.append(np.asarray(openface))
        y_list.append(label)

    x1 = np.stack(x1_list, axis=0)  # (N,1,768)
    x2 = np.stack(x2_list, axis=0)  # (N,T,D)
    y  = np.asarray(y_list)
    return x1, x2, y


def load_fusion_splits(
    fused_path: str = "data/Xy_fusion.npy",
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    oversample: bool = True,
    random_state: int = RANDOM_STATE,
) -> FusionSplits:
    """
    Loads the fused dataset and produces train/val/test splits.

    Improvements over old version:
    - No redundant double-split: we split test once, then val from remainder.
    - Oversampling is actually applied (old code discarded it by accident).
    - Random state is fixed for reproducibility.
    """
    # Load pre-fused MARLIN+OpenFace dataset
    Xy = np.load(fused_path, allow_pickle=True)
    x1, x2, y = _unpack_Xy_fusion(Xy)

    # 1) First split out the test set
    x1_temp, x1_test, x2_temp, x2_test, y_temp, y_test = train_test_split(
        x1, x2, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # 2) Split remaining into train/val
    val_rel = val_size / (1.0 - test_size)  # val proportion of remaining
    x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(
        x1_temp, x2_temp, y_temp,
        test_size=val_rel,
        stratify=y_temp,
        random_state=random_state,
    )

    # 3) Optional oversampling on train set
    if oversample:
        n_train, _, marlin_dim = x1_train.shape
        T, D = x2_train.shape[1], x2_train.shape[2]

        # Flatten MARLIN + OpenFace for oversampler
        x1_flat = x1_train.reshape(n_train, marlin_dim)
        x2_flat = x2_train.reshape(n_train, T * D)
        X_flat  = np.concatenate([x1_flat, x2_flat], axis=1)

        sampler = RandomOverSampler(sampling_strategy="not majority",
                                    random_state=random_state)
        X_res, y_res = sampler.fit_resample(X_flat, y_train)

        # Recover shapes
        x1_train = X_res[:, :marlin_dim].reshape(-1, 1, marlin_dim)
        x2_train = X_res[:, marlin_dim:].reshape(-1, T, D)
        y_train  = y_res

    return FusionSplits(
        train_x1=x1_train, train_x2=x2_train, train_y=y_train,
        val_x1=x1_val,     val_x2=x2_val,     val_y=y_val,
        test_x1=x1_test,   test_x2=x2_test,   test_y=y_test,
    )


# -----------------------
# 3) Drop-in wrapper for old code
# -----------------------
def data_loader_fusion(feature_type="fusion", val=True, base_dir="data"):
    """
    Drop-in replacement for your old data_loader_fusion.
    Same signature + return format, so your training scripts work unchanged.
    Under the hood it uses load_fusion_splits (the bug-fixed version).
    """
    splits = load_fusion_splits(
        fused_path=f"{base_dir}/Xy_{feature_type}.npy",
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        oversample=True,
        random_state=RANDOM_STATE,
    )
    if val:
        return (
            (splits.train_x1, splits.train_x2, splits.train_y),
            (splits.val_x1,   splits.val_x2,   splits.val_y),
            (splits.test_x1,  splits.test_x2,  splits.test_y),
        )
    else:
        return (
            (splits.train_x1, splits.train_x2, splits.train_y),
            (splits.test_x1,  splits.test_x2,  splits.test_y),
        )


# -----------------------
# 4) Quick self-test
# -----------------------
if __name__ == "__main__":
    print("✅ Preparing demo splits from fused dataset ...")
    (train_x1, train_x2, train_y), (val_x1, val_x2, val_y), (test_x1, test_x2, test_y) = \
        data_loader_fusion("fusion", val=True)

    print("Train:", train_x1.shape, train_x2.shape, train_y.shape)
    print("Val:  ", val_x1.shape,   val_x2.shape,   val_y.shape)
    print("Test: ", test_x1.shape,  test_x2.shape,  test_y.shape)
