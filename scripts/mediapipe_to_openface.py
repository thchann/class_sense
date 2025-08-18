"""
Runtime helper: loads the trained converter and exposes a function
`mesh_frames_to_openface(mesh_seq_9x1434) -> (1,9,768)` for live use.
"""
from __future__ import annotations
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "mediapipe_to_openface.h5"
STATS_PATH = MODELS_DIR / "mediapipe_input_stats.npz"

_model = None
_mu = None
_sigma = None

def _lazy_load():
    global _model, _mu, _sigma
    if _model is None:
        _model = load_model(MODEL_PATH)
        st = np.load(STATS_PATH)
        _mu = st["mean"]
        _sigma = st["std"]

def mesh_frames_to_openface(mesh_seq_9x1434: np.ndarray) -> np.ndarray:
    """mesh_seq_9x1434: np.ndarray of shape (9, 1434) -> returns (1, 9, 768)"""
    _lazy_load()
    # z-score per frame using dataset stats
    Xz = (mesh_seq_9x1434 - _mu) / _sigma
    out = _model.predict(Xz, verbose=0)  # (9,768)
    return out[None, :, :]