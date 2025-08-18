# live_marlin_openface_combination.py
import os, sys, cv2, torch, numpy as np, mediapipe as mp, warnings, logging
from marlin_pytorch import Marlin
from tensorflow.keras.models import load_model as tf_load_model
from threading import Thread
from collections import deque
from queue import Queue
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", module=r"torchvision\.io\._video_deprecation_warning")
warnings.filterwarnings("ignore", module=r"torchvision\.io\.video", message=r".*pts_unit 'pts'.*")
logging.getLogger("absl").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "scripts"))
from mediapipe_to_openface import mesh_frames_to_openface

CAMERA_INDEX = 1
FPS = 30
BUFFER_FRAMES = 9
FRAME_W, FRAME_H = 224, 224
MARLIN_MODEL_PATH = "/Users/theodorechan/external_libs/MARLIN/.marlin/marlin_vit_base_ytf.encoder.pt"
CHECKPOINT_PATH = "/Users/theodorechan/external_libs/engagenet_baselines/checkpoints/fusion_best.keras"
TEMP_MARLIN_CLIP = ROOT / "temp_marlin.avi"
MAX_QUEUE = 1

TEMPERATURE = 4.0
ALPHA_OPENFACE = 0.25
MIN_MESH_NORM = 1e-4
FACE_MIN_FRAMES = 2
NOFACE_CONSEC_FOR_UI = 3

USE_MARLIN = True
USE_OPENFACE = True
USE_ONLINE_PRIOR = True
PRIOR_DECAY = 0.98
CLASS_LOGIT_PRIOR = None

pred_history = deque(maxlen=5)
frame_queue = Queue()
prediction_queue = Queue()
last_pred = None
noface_streak = 0

print("▶ Loading MARLIN...")
device = "cuda" if torch.cuda.is_available() else "cpu"
marlin_model = Marlin.from_file("marlin_vit_base_ytf", MARLIN_MODEL_PATH).to(device)

print("▶ Loading EngageNet...")
engage_model = tf_load_model(CHECKPOINT_PATH, compile=False)
print(engage_model.summary())

det_conf = 0.35
def make_det(conf):
    return mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=conf)
face_detector = make_det(det_conf)
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.35)

CLASS_NAMES = ["Disengaged", "Bored", "Attentive", "Confused"]
CLASS_REMAP = [2, 1, 0, 3]

def purge_queue(q: Queue):
    try:
        while True:
            q.get_nowait()
    except Exception:
        pass

def crop_faces(frames):
    faces, det_hits = [], 0
    black = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    for f in frames[:BUFFER_FRAMES]:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        res = face_detector.process(rgb)
        if res.detections:
            det_hits += 1
            bb = res.detections[0].location_data.relative_bounding_box
            h, w = f.shape[:2]
            x1 = max(0, int(bb.xmin * w))
            y1 = max(0, int(bb.ymin * h))
            x2 = min(w, x1 + int(bb.width * w))
            y2 = min(h, y1 + int(bb.height * h))
            if x2 > x1 and y2 > y1:
                crop = cv2.resize(f[y1:y2, x1:x2], (FRAME_W, FRAME_H))
                faces.append(crop)
            else:
                faces.append(black)
        else:
            faces.append(black)
    while len(faces) < BUFFER_FRAMES:
        faces.append(black)
    return np.stack(faces), det_hits

def extract_mesh_seq(frames):
    feats, mesh_hits = [], 0
    for f in frames[:BUFFER_FRAMES]:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            vec = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()
            mesh_hits += 1
        else:
            vec = np.zeros((478*3,), dtype=np.float32)
        feats.append(vec)
    while len(feats) < BUFFER_FRAMES:
        feats.append(feats[-1])
    return np.stack(feats), mesh_hits

def extract_marlin_tensor(frames):
    faces, det_hits = crop_faces(frames)
    if det_hits < FACE_MIN_FRAMES:
        return None, det_hits
    out = cv2.VideoWriter(str(TEMP_MARLIN_CLIP), cv2.VideoWriter_fourcc(*"XVID"), FPS, (FRAME_W, FRAME_H))
    for f in faces:
        out.write(f)
    out.release()
    feats = marlin_model.extract_video(str(TEMP_MARLIN_CLIP), crop_face=False)
    if isinstance(feats, torch.Tensor):
        feats = feats.cpu().numpy()
    try:
        os.remove(TEMP_MARLIN_CLIP)
    except Exception:
        pass
    return feats[-1][None, None, :], det_hits

def _update_online_prior(logits):
    global CLASS_LOGIT_PRIOR
    if CLASS_LOGIT_PRIOR is None:
        CLASS_LOGIT_PRIOR = np.zeros_like(logits[0], dtype=np.float32)
    CLASS_LOGIT_PRIOR[:] = PRIOR_DECAY * CLASS_LOGIT_PRIOR + (1.0 - PRIOR_DECAY) * logits[0]
    CLASS_LOGIT_PRIOR[:] -= CLASS_LOGIT_PRIOR.mean()

def draw_dashboard(frame, avg_pred, entropy, face_hits, cls_id):
    h, w = frame.shape[:2]
    panel_w, panel_h = 320, 160
    x0, y0 = 20, 20
    cv2.rectangle(frame, (x0, y0), (x0+panel_w, y0+panel_h), (0, 0, 0), -1)
    cv2.addWeighted(frame[y0:y0+panel_h, x0:x0+panel_w], 0.6,
                    np.zeros((panel_h, panel_w, 3), dtype=np.uint8), 0.4, 0,
                    frame[y0:y0+panel_h, x0:x0+panel_w])
    cv2.putText(frame, f"Class: {CLASS_NAMES[cls_id]}", (x0+10, y0+25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"H:{entropy:.2f}  face:{face_hits}/9", (x0+10, y0+50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    bar_x, bar_y = x0+10, y0+70
    bar_w, bar_h, gap = 280, 18, 6
    for i, p in enumerate(avg_pred):
        px = int(p * bar_w)
        y = bar_y + i * (bar_h + gap)
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + bar_h), (80,80,80), 1)
        cv2.rectangle(frame, (bar_x, y), (bar_x + px, y + bar_h), (0,255,0), -1)
        cv2.putText(frame, f"C{i}: {p:.2f}", (bar_x + bar_w + 6, y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def draw_bottom_status(frame, text):
    h, w = frame.shape[:2]
    pad, bar_h = 10, 36
    y1, y2 = h - bar_h - pad, h - pad
    cv2.rectangle(frame, (0, y1), (w, y2), (0, 0, 0), -1)
    overlay = frame.copy()
    cv2.addWeighted(overlay[y1:y2, 0:w], 0.5, frame[y1:y2, 0:w], 0.5, 0, frame[y1:y2, 0:w])
    cv2.putText(frame, text, (16, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

def process_buffer():
    eps = 1e-8
    while True:
        frames = frame_queue.get()
        if frames is None:
            break

        mesh_seq, mesh_hits = extract_mesh_seq(frames)
        marlin_feats, det_hits = extract_marlin_tensor(frames)
        mesh_norm = float(np.linalg.norm(mesh_seq))

        mesh_ok   = (mesh_hits >= FACE_MIN_FRAMES) and (mesh_norm > MIN_MESH_NORM)
        marlin_ok = (marlin_feats is not None) and (det_hits >= FACE_MIN_FRAMES)

        if not (mesh_ok and marlin_ok):
            pred_history.clear()
            prediction_queue.put(("No face", 0.0, np.array([np.nan]*4), np.nan, min(det_hits, mesh_hits)))
            continue

        prox = mesh_frames_to_openface(mesh_seq) if USE_OPENFACE else np.zeros((1, 9, 768), np.float32)
        openface_feats = ALPHA_OPENFACE * prox

        raw_probs = np.mean(
            [engage_model([marlin_feats, openface_feats], training=True).numpy() for _ in range(5)],
            axis=0
        )  # shape (1, 4) in model's native class order

        logits = np.log(raw_probs + eps)
        if USE_ONLINE_PRIOR:
            _update_online_prior(logits)
            logits = logits - CLASS_LOGIT_PRIOR
        logits = logits / TEMPERATURE
        logits -= np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(logits); probs /= (np.sum(probs, axis=-1, keepdims=True) + eps)

        # -------- REMAP HERE --------
        probs = probs[:, CLASS_REMAP]  # now order matches CLASS_NAMES
        # ----------------------------

        pred_history.append(probs[0])
        avg_pred = np.mean(pred_history, axis=0)
        cls = int(np.argmax(avg_pred))
        entropy = float(-np.sum(avg_pred * np.log(avg_pred + eps)))
        prediction_queue.put((cls, float(np.max(avg_pred)), avg_pred, entropy, max(det_hits, mesh_hits)))

def live_loop():
    global last_pred, noface_streak, face_detector, det_conf
    cap = cv2.VideoCapture(CAMERA_INDEX)
    frames = []
    print("✅ Live prediction started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det = face_detector.process(rgb)
        has_face_now = bool(det.detections)
        noface_streak = 0 if has_face_now else (noface_streak + 1)

        if has_face_now and det.detections:
            bb = det.detections[0].location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x1 = max(0, int(bb.xmin * w)); y1 = max(0, int(bb.ymin * h))
            x2 = min(w, x1 + int(bb.width * w)); y2 = min(h, y1 + int(bb.height * h))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        if noface_streak >= 30 and det_conf > 0.25:
            det_conf = 0.25
            face_detector = make_det(det_conf)
        elif has_face_now and det_conf < 0.35:
            det_conf = 0.35
            face_detector = make_det(det_conf)

        frames.append(frame)
        if len(frames) >= BUFFER_FRAMES:
            while frame_queue.qsize() >= MAX_QUEUE:
                try: frame_queue.get_nowait()
                except: break
            frame_queue.put(frames[:BUFFER_FRAMES])
            frames = []

        while not prediction_queue.empty():
            last_pred = prediction_queue.get()

        if last_pred is None:
            cv2.putText(frame, "Warming up …", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            draw_bottom_status(frame, "Class: --")
        else:
            cls, conf, avg_pred, entropy, face_hits = last_pred
            if (noface_streak >= NOFACE_CONSEC_FOR_UI) or isinstance(cls, str):
                pred_history.clear()
                draw_bottom_status(frame, "🚫 No face. Holding last prediction — Class: None")
                cv2.putText(frame, "No face detected", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            else:
                draw_dashboard(frame, avg_pred, entropy, face_hits, cls)
                draw_bottom_status(frame, f"Class: {CLASS_NAMES[cls]}")

        cv2.imshow("Live Engagement", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    t = Thread(target=process_buffer, daemon=True); t.start()
    live_loop()
