# scripts/train_proxy.py
"""
Train a lightweight MLP that maps MediaPipe FaceMesh landmarks (1434) -> OpenFace-style projected embeddings (768) per frame.
Adds:
- gather_files() + preflight() to find matches and print a summary
- CLI flags: --dry-run, --max-videos, --progress-every
- Progress logs during dataset build
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ---------------------------
# CONFIG
# ---------------------------
VIDEOS_DIR = Path("/Users/theodorechan/Downloads/DAiSEE/DataSet/Test")
OPENFACE_DIR = Path("/Users/theodorechan/Downloads/openface_output")
PROJECTION_MATRIX_PATH = Path("/Users/theodorechan/external_libs/engagenet_baselines/openface_projection_matrix.npy")
SELECTED_COLUMNS = [
    'pose_Rx', 'pose_Ry', 'pose_Rz',
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
    'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
    'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
]
VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv")
SKIP_STEMS = {"tmp_clip", "temp_clip"}

N_FRAMES = 9
LANDMARK_DIM = 478 * 3
EMBED_DIM = 768
BATCH_SIZE = 256
EPOCHS = 15
VAL_SPLIT = 0.1
OUT_DIR = Path("models"); OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# MediaPipe
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.6)

# ---------------------------
# Utils: sampling + loaders
# ---------------------------

def evenly_spaced_indices(total: int, n: int) -> List[int]:
    if total <= 0:
        return [0]*n
    step = max(total // n, 1)
    idx = [min(i*step, total-1) for i in range(n)]
    return idx[:n] if len(idx) >= n else idx + [idx[-1]]*(n-len(idx))

def sample_frames(video_path: Path, n: int = N_FRAMES) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = evenly_spaced_indices(total, n) if total > 0 else list(range(n))
    for target in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    while len(frames) < n and len(frames) > 0:
        frames.append(frames[-1])
    return frames

def extract_mesh_vec(frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_face_mesh.process(rgb)
    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()
    return np.zeros((LANDMARK_DIM,), dtype=np.float32)

# ---------------------------
# File discovery + preflight
# ---------------------------

def gather_files(videos_root: Path, openface_root: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    vids: Dict[str, Path] = {}
    ofcs: Dict[str, Path] = {}
    for ext in VIDEO_EXTS:
        for vp in videos_root.rglob(ext):
            stem = vp.stem
            if stem in SKIP_STEMS:
                continue
            vids[stem] = vp
    for cp in openface_root.rglob("*.csv"):
        stem = cp.stem
        if stem in SKIP_STEMS:
            continue
        ofcs[stem] = cp
    return vids, ofcs

def preflight(max_videos: int | None = None) -> Tuple[Dict[str, Path], Dict[str, Path], List[str]]:
    vids, ofcs = gather_files(VIDEOS_DIR, OPENFACE_DIR)
    shared = sorted(set(vids.keys()) & set(ofcs.keys()))
    if max_videos is not None:
        shared = shared[:max_videos]
    print(f"Videos found: {len(vids)} | OpenFace CSVs found: {len(ofcs)} | Matches: {len(shared)}")
    for s in shared[:5]:
        print(f"  match: {s}\n    video: {vids[s]}\n    csv:   {ofcs[s]}")
    return vids, ofcs, shared

# ---------------------------
# OpenFace target loader
# ---------------------------

def load_openface_target(of_csv: Path, proj: np.ndarray | None) -> np.ndarray | None:
    df = pd.read_csv(of_csv)
    cols = [c for c in SELECTED_COLUMNS if c in df.columns]
    if not cols:
        # if the CSV is already 768D per frame, accept it directly
        if df.shape[1] == EMBED_DIM:
            arr = df.values.astype(np.float32)
        else:
            return None
    else:
        arr = df[cols].values.astype(np.float32)
        if proj is None:
            return None
    # pad/trim to N_FRAMES
    if arr.shape[0] < N_FRAMES:
        arr = np.pad(arr, ((0, N_FRAMES - arr.shape[0]), (0, 0)))
    elif arr.shape[0] > N_FRAMES:
        arr = arr[:N_FRAMES]
    # if raw AUs, z-score and project to 768
    if arr.shape[1] != EMBED_DIM:
        mu = arr.mean(axis=0, keepdims=True)
        sigma = arr.std(axis=0, keepdims=True) + 1e-6
        arr = (arr - mu) / sigma
        if proj.shape[0] != arr.shape[1]:
            if proj.shape[0] < arr.shape[1]:
                arr = arr[:, :proj.shape[0]]
            else:
                arr = np.pad(arr, ((0, 0), (0, proj.shape[0] - arr.shape[1])))
        arr = np.dot(arr, proj).astype(np.float32)
    return arr.astype(np.float32)

# ---------------------------
# Build dataset
# ---------------------------

def build_dataset(max_videos: int | None = None, progress_every: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    try:
        proj = np.load(PROJECTION_MATRIX_PATH)
    except Exception:
        proj = None
        print("⚠️ Projection matrix not found. Will try to use pre-projected 768D CSVs.")

    vids, ofcs, shared = preflight(max_videos=max_videos)
    if not shared:
        raise RuntimeError("No matches between videos and OpenFace CSVs. Check OPENFACE_DIR and filenames.")

    X_list, Y_list = [], []
    t0 = time.time()
    for i, stem in enumerate(shared):
        if i % progress_every == 0:
            pct = 100.0 * i / max(1, len(shared))
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 1e-6 else 0.0
            eta = (len(shared) - i) / rate if rate > 0 else float('inf')
            print(f"[INFO] {i}/{len(shared)} ({pct:4.1f}%) videos processed | {rate:5.1f} vid/s | ETA ~ {eta:5.1f}s")

        vp = vids[stem]
        of_csv = ofcs[stem]
        frames = sample_frames(vp, N_FRAMES)
        if not frames:
            continue
        mesh_seq = np.stack([extract_mesh_vec(f) for f in frames])  # (9,1434)
        target_seq = load_openface_target(of_csv, proj)             # (9,768) or None
        if target_seq is None:
            continue
        if not np.any(mesh_seq):
            continue
        X_list.append(mesh_seq)
        Y_list.append(target_seq)

    if not X_list:
        raise RuntimeError("No paired samples built. Ensure OpenFace CSVs are present and aligned with videos.")

    X = np.concatenate(X_list, axis=0)  # (N*9, 1434)
    Y = np.concatenate(Y_list, axis=0)  # (N*9, 768)
    print(f"✅ Built dataset: X={X.shape}, Y={Y.shape}")
    return X, Y

# ---------------------------
# Model
# ---------------------------

def build_model(input_dim: int = LANDMARK_DIM, output_dim: int = EMBED_DIM) -> tf.keras.Model:
    inp = layers.Input(shape=(input_dim,), name="mesh_frame")
    x = layers.LayerNormalization()(inp)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(output_dim, activation=None)(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="mse",
                  metrics=[tf.keras.metrics.CosineSimilarity(name="cos_sim")])
    return model

# ---------------------------
# Train
# ---------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Only scan & report counts; do not extract/train.")
    ap.add_argument("--max-videos", type=int, default=None, help="Limit number of matched videos for faster runs.")
    ap.add_argument("--progress-every", type=int, default=50, help="Print progress every N videos.")
    args = ap.parse_args()

    if args.dry_run:
        _vids, _ofcs, _shared = preflight(max_videos=args.max_videos)
        print("(dry-run) Nothing else to do.")
        raise SystemExit(0)

    X, Y = build_dataset(max_videos=args.max_videos, progress_every=args.progress_every)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-6
    np.savez(OUT_DIR / "mediapipe_input_stats.npz", mean=mu, std=sigma)
    Xz = (X - mu) / sigma

    model = build_model()
    ckpt = callbacks.ModelCheckpoint(str(OUT_DIR / "mediapipe_to_openface.h5"),
                                     monitor="val_cos_sim", mode="max",
                                     save_best_only=True, verbose=1)
    early = callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_cos_sim", mode="max")
    print("▶ Starting training…")
    hist = model.fit(Xz, Y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VAL_SPLIT,
                     callbacks=[ckpt, early], verbose=1)

    n_eval = min(2048, len(Xz))
    pred = model.predict(Xz[:n_eval], verbose=0)
    cs = np.sum(pred * Y[:n_eval], axis=1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(Y[:n_eval], axis=1) + 1e-8)
    pd.DataFrame({"cosine_sim": cs}).to_csv(REPORTS_DIR / "proxy_eval.csv", index=False)
    print("Saved:", OUT_DIR / "mediapipe_to_openface.h5")
