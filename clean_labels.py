import numpy as np

Xy = np.load('data/Xy_engage_gaze+hp+au_marlin.npy', allow_pickle=True)
print(f"Total entries in Xy_fusion: {len(Xy)}")

bad = []
for i, entry in enumerate(Xy):
    if len(entry) != 3:
        bad.append((i, "wrong tuple length", len(entry)))
        continue
    video_id, marlin, openface = entry
    if marlin.shape != (1, 768) or openface.shape != (9, 768):
        bad.append((i, marlin.shape, openface.shape))

print(f"\nMalformed samples: {len(bad)}")
for i, m_shape, o_shape in bad[:5]:  # Show up to 5 issues
    print(f"Sample {i}: MARLIN={m_shape}, OpenFace={o_shape}")
