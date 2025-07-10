import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load your fused feature file
data = np.load('data/Xy_fusion.npy', allow_pickle=True)

features = []
labels = []

for sample in data:
    clip_id, marlin, openface, label = sample
    marlin = np.array(marlin).reshape(-1)           # (768,)
    openface = np.array(openface).reshape(-1)       # (9, 768) → (6912,)
    fused = np.concatenate([marlin, openface])      # (7680,)
    features.append(fused)
    labels.append(label)

features = np.array(features)
labels = np.array(labels)

# -------- PCA Visualization --------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(features)

plt.figure(figsize=(8, 6))
for i in range(4):
    idx = labels == i
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f'Class {i}', alpha=0.5)
plt.legend()
plt.title("📉 PCA Projection of Fused Features")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- t-SNE Visualization --------
print("Running t-SNE... (may take 1–2 minutes)")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(features)

plt.figure(figsize=(8, 6))
for i in range(4):
    idx = labels == i
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=f'Class {i}', alpha=0.5)
plt.legend()
plt.title("🌀 t-SNE Projection of Fused Features")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.grid(True)
plt.tight_layout()
plt.show()
