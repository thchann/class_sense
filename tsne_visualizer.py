import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

# Load data
data = np.load('data/Xy_fusion.npy', allow_pickle=True)

# Sample 2000 points randomly for faster t-SNE
random.seed(42)
sampled_data = random.sample(list(data), 2000)

features = []
labels = []

for sample in sampled_data:
    clip_id, marlin, openface, label = sample
    marlin = np.array(marlin).reshape(-1)           # (768,)
    openface = np.array(openface).reshape(-1)       # (9, 768) → (6912,)
    fused = np.concatenate([marlin, openface])      # (7680,)
    features.append(fused)
    labels.append(label)

features = np.array(features)
labels = np.array(labels)

# Run t-SNE
print("⏳ Running t-SNE on 2000 samples (this may take 1–2 min)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(features)

# Plot
plt.figure(figsize=(10, 8))
colors = ['blue', 'orange', 'green', 'red']
for i in range(4):
    idx = np.array(labels) == i
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=f'Class {i}', alpha=0.5, s=20, c=colors[i])
plt.legend()
plt.title("🌀 t-SNE Projection of Fused MARLIN + OpenFace Embeddings")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True)
plt.tight_layout()
plt.show()
