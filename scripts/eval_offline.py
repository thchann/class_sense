#!/usr/bin/env python3
# scripts/eval_offline.py
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import cv2
import torch
import mediapipe as mp
from marlin_pytorch import Marlin
from tensorflow.keras.models import load_model as tf_load_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))
from mediapipe_to_openface import mesh_frames_to_openface  # type: ignore

CLASS_NAMES = ["Disengaged", "Bored", "Attentive", "Confused"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--marlin", required=True)
    ap.add_argument("--engagenet", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--temp", type=float, default=4.0)
    ap.add_argument("--mc", type=int, default=5)
    ap.add_argument("--hist", type=int, default=9)
    ap.add_argument("--outdir", default=str(ROOT / "runs_offline"))
    return ap.parse_args()

def resolve_video(path_str: str) -> Path | None:
    p = Path(path_str)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if p.exists():
        return p
    # try extension swaps
    cand = []
    if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        base = p.with_suffix("")
        cand = [base.with_suffix(ext) for ext in [".mov", ".mp4", ".avi", ".mkv"]]
    else:
        cand = [p.with_suffix(ext) for ext in [".mov", ".mp4", ".avi", ".mkv"]]
    for c in cand:
        if c.exists():
            return c
    return None

def open_video_any(path: Path):
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap
    # macOS fallback (AVFoundation)
    cap = cv2.VideoCapture(str(path), cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        return cap
    return None

def sample_every_n(cap, total_frames, n=9):
    if total_frames and total_frames > 0:
        step = max(total_frames // n, 1)
        idxs = [min(i * step, total_frames - 1) for i in range(n)]
    else:
        idxs = list(range(n))
    frames = []
    for t in idxs:
        if total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    while len(frames) < n and frames:
        frames.append(frames[-1])
    return frames

def extract_mesh_seq(frames, face_mesh):
    feats = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            vec = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()
        else:
            vec = np.zeros((478*3,), dtype=np.float32)
        feats.append(vec)
    return np.stack(feats)

def write_temp_clip(frames, path, fps=30, size=(224,224)):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), fps, size)
    for f in frames:
        out.write(cv2.resize(f, size))
    out.release()

def softmax_np(x, axis=-1, eps=1e-8):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + eps)

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    marlin = Marlin.from_file("marlin_vit_base_ytf", args.marlin).to(device)
    engage = tf_load_model(args.engagenet, compile=False)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    )

    for vid_in in args.videos:
        vp = resolve_video(vid_in)
        if vp is None:
            print(f"[ERROR] Video not found: {vid_in} (cwd={Path.cwd()})")
            continue
        cap = open_video_any(vp)
        if cap is None:
            print(f"[ERROR] OpenCV could not open: {vp}")
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps if fps and fps > 0 else 30.0)

        frames = sample_every_n(cap, total, n=args.hist)
        cap.release()
        if not frames:
            print(f"[WARN] No frames read from {vp}")
            # still write empty CSV for consistency
            (outdir / f"{vp.stem}_eval.csv").write_text("t_sec,c0,c1,c2,c3\n")
            continue

        mesh_seq = extract_mesh_seq(frames, face_mesh)                        # (9,1434)
        prox = mesh_frames_to_openface(mesh_seq)                              # (1,9,768)

        tmp_clip = outdir / f"{vp.stem}_tmp.avi"
        write_temp_clip(frames, tmp_clip, fps=fps, size=(224,224))
        feats = marlin.extract_video(str(tmp_clip), crop_face=False)          # torch or np
        if isinstance(feats, torch.Tensor):
            feats = feats.cpu().numpy()
        marlin_in = feats[-1][None, None, :]                                  # (1,1,768)
        try:
            os.remove(tmp_clip)
        except: pass

        eps = 1e-8
        raw = np.mean([engage([marlin_in, prox], training=True).numpy() for _ in range(args.mc)], axis=0)
        logits = np.log(raw + eps) / args.temp
        probs = softmax_np(logits, axis=-1)

        csv_path = outdir / f"{vp.stem}_eval.csv"
        with open(csv_path, "w") as f:
            f.write("t_sec,c0,c1,c2,c3\n")
            # one prediction per 9-frame window -> use last timestamp of window for display
            t_sec = (args.hist - 1) / fps
            c0, c1, c2, c3 = probs[0].tolist()
            f.write(f"{t_sec:.3f},{c0:.6f},{c1:.6f},{c2:.6f},{c3:.6f}\n")

        mean_probs = probs[0]
        print(f"Saved: {csv_path}")
        print("Summary (mean probs):", dict(zip(CLASS_NAMES, [float(x) for x in mean_probs])))

if __name__ == "__main__":
    main()
