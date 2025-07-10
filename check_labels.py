import numpy as np

Xy = np.load("data/Xy_fusion.npy", allow_pickle=True)

print("✅ Total samples:", len(Xy))

for i in range(3):
    clip_id, marlin, openface, label = Xy[i]
    print(f"\nSample {i} - Clip ID: {clip_id}")
    print(f"  ➤ MARLIN shape:    {marlin.shape}")
    print(f"  ➤ OpenFace shape: {openface.shape}")
    print(f"  ➤ Label:           {label}")
