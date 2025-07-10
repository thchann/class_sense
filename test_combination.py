import numpy as np

Xy = np.load("data/Xy_fusion.npy", allow_pickle=True)

print(f"✅ Total samples: {len(Xy)}")
print("🔍 Sample 0 contents:")
for i, elem in enumerate(Xy[0]):
    print(f"  Element {i}: type={type(elem)}, shape={getattr(elem, 'shape', 'N/A')}, value={elem if i != 1 else '[matrix]'}")
