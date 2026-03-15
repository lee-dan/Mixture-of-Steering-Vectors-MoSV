"""visualize.py — t-SNE, PCA, and UMAP plots of K-means vs router assignments with decision boundaries."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLUSTER_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]

plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_tsne(X, perplexity=30, n_iter=1000, seed=42):
    n = X.shape[0]
    pca_dim = min(50, n - 1, X.shape[1])
    X_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    perp = min(perplexity, max(5, (n - 1) // 3))
    return TSNE(n_components=2, perplexity=perp, max_iter=n_iter, random_state=seed).fit_transform(X_pca)


def compute_pca_2d(X, seed=42):
    pca = PCA(n_components=2, random_state=seed)
    X_2d = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    return X_2d, var[0], var[1]


def compute_umap(X, n_neighbors=30, min_dist=0.1, seed=42):
    import umap
    pca_dim = min(50, X.shape[0] - 1, X.shape[1])
    X_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=seed)
    return reducer.fit_transform(X_pca)


def get_router_predictions(router, prompt_vectors, best_layer_pos, device):
    router.eval()
    X = torch.from_numpy(prompt_vectors[:, best_layer_pos, :].astype(np.float32)).to(device)
    with torch.no_grad():
        logits = router._mlp(X)
    return logits.argmax(dim=-1).cpu().numpy()


def plot_side_by_side(X_2d, kmeans_labels, router_labels, K, tag, save_path,
                      xlabel="t-SNE 1", ylabel="t-SNE 2"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, labels, title in zip(axes,
                                  [kmeans_labels, router_labels],
                                  [f"K-means clusters (K={K})", f"Router predictions (K={K})"]):
        for k in range(K):
            mask = labels == k
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
                       label=f"Cluster {k}", alpha=0.5, s=18, edgecolors="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, markerscale=1.5, fontsize=9)

    # Mark misrouted points on the router panel (right)
    mismatch = kmeans_labels != router_labels
    if mismatch.any():
        axes[1].scatter(X_2d[mismatch, 0], X_2d[mismatch, 1],
                        marker="x", c="black", s=25, linewidths=0.8,
                        alpha=0.6, label=f"Misrouted ({mismatch.sum()})")
        axes[1].legend(frameon=False, markerscale=1.5, fontsize=9)

    pct_correct = (kmeans_labels == router_labels).mean() * 100
    fig.suptitle(f"{tag} — Router agreement with K-means: {pct_correct:.1f}%", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}  (router agreement={pct_correct:.1f}%, misrouted={mismatch.sum()})")


def plot_router_decision_boundary(X_2d, router_labels, K, tag, save_path,
                                   xlabel="t-SNE 1", ylabel="t-SNE 2"):
    """Visualize router decision regions via KNN flood-fill over a 2D embedding."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Fit KNN on the 2D t-SNE points using the router's cluster assignments
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_2d, router_labels)

    # Build a fine meshgrid covering the t-SNE extent with padding
    pad = 2.0
    x_min, x_max = X_2d[:, 0].min() - pad, X_2d[:, 0].max() + pad
    y_min, y_max = X_2d[:, 1].min() - pad, X_2d[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    zz = knn.predict(grid_points).reshape(xx.shape)

    # Build colormap arrays aligned to cluster indices
    from matplotlib.colors import ListedColormap
    colors_bg = [CLUSTER_COLORS[k % len(CLUSTER_COLORS)] for k in range(K)]
    cmap_bg = ListedColormap(colors_bg)

    # Flood-fill background regions
    ax.pcolormesh(xx, yy, zz, cmap=cmap_bg, alpha=0.25, shading="auto")

    # Draw decision boundary contour lines
    ax.contour(xx, yy, zz, levels=np.arange(-0.5, K, 1), colors="black",
               linewidths=0.5, alpha=0.7)

    # Overlay actual data points
    for k in range(K):
        mask = router_labels == k
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
                   label=f"Cluster {k}", alpha=0.6, s=15, edgecolors="none")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, markerscale=1.5, fontsize=9, loc="best")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    pct_points = {k: (router_labels == k).mean() * 100 for k in range(K)}
    subtitle = "  ".join(f"C{k}:{pct_points[k]:.0f}%" for k in range(K))
    ax.set_title(f"Router Decision Regions (K={K})\n{subtitle}", fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved boundary plot: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/defan.yaml")
    parser.add_argument("--k_tags", nargs="+", default=["K2", "K3", "K4", "K5"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cpu")

    from mosv.activation.extract import load_activations
    from mosv.clustering.cluster import load_clustering
    from mosv.routing.train import load_router

    act_dir = cfg["paths"]["activations_dir"]
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    fig_dir = cfg["paths"].get("figures_dir", cfg["paths"]["outputs_dir"])
    layers = cfg["activation"]["layers_to_probe"]

    print("Loading activations...")
    diff_vectors, prompt_vectors = load_activations(act_dir)

    # Compute t-SNE once using the best layer from the first available K variant
    first_tag = args.k_tags[0]
    _, _, first_meta = load_clustering(os.path.join(act_dir, f"sweep_{first_tag}"))
    best_layer = first_meta["best_layer_idx"]
    best_layer_pos = layers.index(best_layer)
    X = diff_vectors[:, best_layer_pos, :]

    print(f"Computing t-SNE on {X.shape[0]} points (layer {best_layer})...")
    X_tsne = compute_tsne(X)
    print("t-SNE done.")

    print("Computing PCA (2D)...")
    X_pca2d, var1, var2 = compute_pca_2d(X)
    pca_xlabel = f"PC1 ({var1*100:.1f}%)"
    pca_ylabel = f"PC2 ({var2*100:.1f}%)"
    print(f"PCA done. Variance explained: PC1={var1*100:.1f}%, PC2={var2*100:.1f}%")

    print("Computing UMAP...")
    X_umap = compute_umap(X)
    print("UMAP done.")

    for tag in args.k_tags:
        sub_act_dir  = os.path.join(act_dir,  f"sweep_{tag}")
        sub_ckpt_dir = os.path.join(ckpt_dir, f"sweep_{tag}")

        if not os.path.exists(sub_ckpt_dir):
            print(f"Skipping {tag} — no checkpoint dir found")
            continue

        _, cluster_labels, meta = load_clustering(sub_act_dir)
        K = meta["K"]

        router = load_router(sub_ckpt_dir, device)
        router_preds = get_router_predictions(router, prompt_vectors, best_layer_pos, device)

        # t-SNE plots
        tsne_path = os.path.join(fig_dir, f"router_tsne_{tag}.png")
        plot_side_by_side(X_tsne, cluster_labels, router_preds, K, tag, tsne_path)

        tsne_boundary_path = os.path.join(fig_dir, f"router_boundary_K{K}_tsne.png")
        plot_router_decision_boundary(X_tsne, router_preds, K, tag, tsne_boundary_path)

        # PCA plots
        pca_path = os.path.join(fig_dir, f"router_pca_{tag}.png")
        plot_side_by_side(X_pca2d, cluster_labels, router_preds, K, tag, pca_path,
                          xlabel=pca_xlabel, ylabel=pca_ylabel)

        pca_boundary_path = os.path.join(fig_dir, f"router_boundary_K{K}_pca.png")
        plot_router_decision_boundary(X_pca2d, router_preds, K, tag, pca_boundary_path,
                                      xlabel=pca_xlabel, ylabel=pca_ylabel)

        # UMAP plots
        umap_path = os.path.join(fig_dir, f"router_umap_{tag}.png")
        plot_side_by_side(X_umap, cluster_labels, router_preds, K, tag, umap_path,
                          xlabel="UMAP 1", ylabel="UMAP 2")

        umap_boundary_path = os.path.join(fig_dir, f"router_boundary_K{K}_umap.png")
        plot_router_decision_boundary(X_umap, router_preds, K, tag, umap_boundary_path,
                                      xlabel="UMAP 1", ylabel="UMAP 2")

    print("\nAll done.")


if __name__ == "__main__":
    main()
