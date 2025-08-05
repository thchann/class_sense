import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
from sklearn.preprocessing import StandardScaler

MARLIN_DIR = "/Users/theodorechan/Downloads/marlin_embeddings"
OPENFACE_DIR = "/Users/theodorechan/Downloads/openface_output"
LABELS_CSV = "data/final_labels.csv"
SAVE_PATH = "data/Xy_fusion.npy"

PROJECT_TO_DIM = 768

SELECTED_COLUMNS = [
    'pose_Rx', 'pose_Ry', 'pose_Rz',
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
    'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
    'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
]

def find_openface_csv(video_id, base_dir):
    pattern = os.path.join(base_dir, "**", "**", "**", f"{video_id}.csv")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None

def load_openface(file_path):
    try:
        df = pd.read_csv(file_path)
        df = df[SELECTED_COLUMNS]
        if len(df) > 9:
            df = df.iloc[:9]
        elif len(df) < 9:
            pad_len = 9 - len(df)
            padding = np.zeros((pad_len, len(SELECTED_COLUMNS)))
            df = pd.concat([df, pd.DataFrame(padding, columns=SELECTED_COLUMNS)], ignore_index=True)
        return df.values
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def project_openface(features, target_dim=768):
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    projection_matrix = np.random.randn(features.shape[1], target_dim)
    return np.dot(features, projection_matrix)

def load_marlin(file_path):
    arr = np.load(file_path)
    if arr.shape == (9, 768):
        return np.mean(arr, axis=0)
    elif arr.shape == (768,):
        return arr
    else:
        return None

def main():
    # Checking: Missing OpenFace list
    marlin_ids = set([os.path.splitext(f)[0] for f in os.listdir(MARLIN_DIR) if f.endswith('.npy')])
    openface_ids = set([os.path.splitext(f)[0] for f in os.listdir(OPENFACE_DIR) if f.endswith('.csv')])
    missing_openface_ids = sorted(marlin_ids - openface_ids)

    print(f"📊 Total MARLIN files: {len(marlin_ids)}")
    print(f"📊 Total OpenFace CSVs: {len(openface_ids)}")
    print(f"❌ Missing OpenFace CSVs: {len(missing_openface_ids)}")
    if missing_openface_ids:
        print("Examples:", missing_openface_ids[:10])

    labels_df = pd.read_csv(LABELS_CSV)
    samples = []

    missing_files = 0
    bad_marlin = 0
    bad_openface = 0
    saved = 0

    for i, (_, row) in enumerate(tqdm(labels_df.iterrows(), total=len(labels_df))):
        video_id = str(row["video_id"])
        label = row["label"]

        marlin_path = os.path.join(MARLIN_DIR, f"{video_id}.npy")
        openface_path = find_openface_csv(video_id, OPENFACE_DIR)

        if openface_path is None or not os.path.exists(marlin_path):
            missing_files += 1
            continue

        marlin = load_marlin(marlin_path)
        if marlin is None:
            bad_marlin += 1
            continue

        openface_raw = load_openface(openface_path)
        if openface_raw is None:
            bad_openface += 1
            continue

        openface_proj = project_openface(openface_raw, PROJECT_TO_DIM)
        marlin = marlin.reshape(1, -1)
        combined = np.vstack([marlin, openface_proj])

        samples.append([video_id, marlin, openface_proj, label])
        saved += 1

        if saved <= 5:
            print(f"\nSample {saved} - Clip ID: {video_id}")
            print(f"  ➤ MARLIN shape:    {marlin.shape}")
            print(f"  ➤ OpenFace shape: {openface_proj.shape}")
            print(f"  ➤ Combined shape: {combined.shape}")

    np.save(SAVE_PATH, np.array(samples, dtype=object))
    print(f"\n✅ Done. Total saved: {saved}")
    print(f"❌ Skipped due to missing files: {missing_files}")
    print(f"❌ Skipped due to bad MARLIN shape: {bad_marlin}")
    print(f"❌ Skipped due to OpenFace errors: {bad_openface}")

if __name__ == "__main__":
    main()
