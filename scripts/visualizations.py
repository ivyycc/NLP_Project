"""


Just extra utilities to visualize token embeddings using t-SNE, UMAP, and SOM (MiniSom).
Saves PNG plots to a provided output directory.

Usage:
    from embedding_visualizations import plot_tsne_umap_som
    plot_tsne_umap_som(embeddings, labels_dict, out_dir="evaluation_plots/Features", prefix="emb")
Where:
    embeddings: np.array shape (N, D)
    labels_dict: dict of {name: label_array} e.g. {'color': colors, 'shape': shapes, 'quantity': quantities}

These plots are useful for qualitatively assessing how well the model has learned to cluster different card features in the embedding space, 
not all these plots are in the report, but they are in the evaluation_plots folder for reference.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


try:
    import umap
    _UMAP_AVAILABLE = True
except Exception:
    _UMAP_AVAILABLE = False

try:
    from minisom import MiniSom
    _MINISOM_AVAILABLE = True
except Exception:
    _MINISOM_AVAILABLE = False

sns.set_style("whitegrid")
sns.set_palette("husl")


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

# Helper to do PCA reduction if needed
def _pca_reduce_if_needed(X, n_components=50):

    if X.shape[1] > n_components:
        pca = PCA(n_components=n_components, random_state=0)
        Xr = pca.fit_transform(X)
        return Xr, pca
    return X, None

# Generic 2D scatter plot
def plot_scatter_2d(X2, labels, title, outpath, cmap=None, marker='o'):

    plt.figure(figsize=(6, 5))
    unique = np.unique(labels)
    for u in unique:
        mask = labels == u
        plt.scatter(X2[mask, 0], X2[mask, 1], label=str(u), alpha=0.8, s=60, marker=marker)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")

# t-SNE plot
def tsne_plot(embeddings, labels, outpath, perplexity=15, n_iter=1000, random_state=0, pca_pre=30):

    X, _ = _pca_reduce_if_needed(embeddings, n_components=min(pca_pre, embeddings.shape[1]))
    ts = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state, init='pca')
    X2 = ts.fit_transform(X)
    plot_scatter_2d(X2, np.array(labels), f"t-SNE (perp={perplexity})", outpath)

# Umap plot
def umap_plot(embeddings, labels, outpath, n_neighbors=10, min_dist=0.1, random_state=0, pca_pre=30):

    if not _UMAP_AVAILABLE:
        print("UMAP not available (install 'umap-learn'), skipping UMAP plot.")
        return
    X, _ = _pca_reduce_if_needed(embeddings, n_components=min(pca_pre, embeddings.shape[1]))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    X2 = reducer.fit_transform(X)
    plot_scatter_2d(X2, np.array(labels), f"UMAP (n_neighbors={n_neighbors})", outpath)

from collections import Counter

# SOM plot
def som_plot(embeddings, labels, outpath, grid_x=8, grid_y=8, sigma=1.0, learning_rate=0.5, num_iteration=1000, random_seed=0):
    if not _MINISOM_AVAILABLE:
        print("MiniSom not available (install 'minisom'), skipping SOM plot.")
        return

    X = embeddings.copy()
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    som = MiniSom(grid_x, grid_y, X.shape[1], sigma=sigma, learning_rate=learning_rate, random_seed=random_seed)
    som.random_weights_init(X)
    som.train_random(X, num_iteration)

    umatrix = som.distance_map()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(umatrix.T, cmap='bone_r')
    ax[0].set_title('SOM U-Matrix')
    plt.colorbar(im, ax=ax[0])

    # Fixed label mapping
    cell_labels = [[[] for _ in range(grid_y)] for _ in range(grid_x)]
    for vec, lab in zip(X, labels):
        w = som.winner(vec)
        cell_labels[w[0]][w[1]].append(int(lab))

    # Count samples per cell
    counts = np.zeros((grid_x, grid_y), dtype=int)
    for i in range(grid_x):
        for j in range(grid_y):
            counts[i, j] = len(cell_labels[i][j])
    
    im2 = ax[1].imshow(counts.T, cmap='viridis')
    ax[1].set_title('SOM sample counts per cell')
    plt.colorbar(im2, ax=ax[1])

    plt.suptitle('SOM visualization')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")

# Produce all plots
def plot_tsne_umap_som(embeddings, labels_dict, out_dir, prefix="emb", random_state=0):

    _ensure_dir(out_dir)
    for label_name, labels in labels_dict.items():
        lab = np.array(labels)
        # t-SNE
        tsne_path = os.path.join(out_dir, f"{prefix}_{label_name}_tsne.png")
        tsne_plot(embeddings, lab, tsne_path, perplexity=12, n_iter=1000, random_state=random_state, pca_pre=30)
        # UMAP
        umap_path = os.path.join(out_dir, f"{prefix}_{label_name}_umap.png")
        umap_plot(embeddings, lab, umap_path, n_neighbors=8, min_dist=0.1, random_state=random_state, pca_pre=30)
        # SOM (smaller grid for small N)
        som_path = os.path.join(out_dir, f"{prefix}_{label_name}_som.png")
        som_plot(embeddings, lab, som_path, grid_x=6, grid_y=6, num_iteration=500, random_seed=random_state)

    print("All visualization plots saved.")