import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CLUSTER_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]
SYSTEM_COLORS = {
    "vanilla": "#888888",
    "single-vec": "#4e79a7",
    "MoSV-hard": "#f28e2b",
    "MoSV-soft": "#e15759",
    "oracle": "#59a14f",
}


def plot_tsne_clusters(
    diff_vectors: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    perplexity: int = 30,
    n_iter: int = 1000,
) -> None:
    n_samples = diff_vectors.shape[0]
    # PCA pre-reduction for speed (TSNE recommendation: ≤50 dims)
    pca_dim = min(50, n_samples - 1, diff_vectors.shape[1])
    X_pca = PCA(n_components=pca_dim, random_state=42).fit_transform(diff_vectors)

    perp = min(perplexity, max(5, (n_samples - 1) // 3))
    X_2d = TSNE(n_components=2, perplexity=perp, max_iter=n_iter, random_state=42).fit_transform(X_pca)

    K = int(cluster_labels.max()) + 1
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in range(K):
        mask = cluster_labels == k
        label = cluster_names[k] if cluster_names else f"Cluster {k}"
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
                   label=label, alpha=0.6, s=20, edgecolors="none")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Activation Diff Vectors — t-SNE (K={K})")
    ax.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


def plot_activation_clusters(
    diff_vectors: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(diff_vectors)
    K = int(cluster_labels.max()) + 1

    fig, ax = plt.subplots(figsize=(7, 5))
    for k in range(K):
        mask = cluster_labels == k
        label = cluster_names[k] if cluster_names else f"Cluster {k}"
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
                   label=label, alpha=0.6, s=30, edgecolors="none")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Activation Difference Vectors (PCA)")
    ax.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


def plot_overall_metrics(
    system_results: Dict[str, dict],
    metric: str = "pct_T_and_I",
    save_path: Optional[str] = None,
) -> None:
    systems = list(system_results.keys())
    values = [system_results[s].get(metric, 0) * 100 for s in systems]
    colors = [SYSTEM_COLORS.get(s, "#aaaaaa") for s in systems]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(systems, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    metric_label = {
        "pct_T_and_I": "%Truthful & Informative",
        "pct_truthful": "%Truthful",
        "pct_informative": "%Informative",
        "refusal_rate": "Refusal Rate (%)",
    }.get(metric, metric)

    ax.set_ylabel(metric_label)
    ax.set_title(f"Overall {metric_label} by System")
    ax.set_ylim(0, max(values) * 1.15 + 5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


def plot_per_cluster_comparison(
    cluster_results: Dict[str, Dict[int, dict]],
    K: int,
    cluster_names: Optional[List[str]] = None,
    metric: str = "pct_T_and_I",
    save_path: Optional[str] = None,
) -> None:
    systems = list(cluster_results.keys())
    x = np.arange(K)
    width = 0.8 / len(systems)

    fig, ax = plt.subplots(figsize=(max(6, K * 2), 5))
    for i, system in enumerate(systems):
        vals = [
            cluster_results[system].get(k, {}).get(metric, 0) * 100
            for k in range(K)
        ]
        offset = (i - len(systems) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width,
               label=system, color=SYSTEM_COLORS.get(system, "#aaaaaa"), alpha=0.85)

    names = cluster_names or [f"Cluster {k}" for k in range(K)]
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("%Truthful & Informative" if metric == "pct_T_and_I" else metric)
    ax.set_title("Per-Cluster Performance by System")
    ax.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


def plot_alpha_sweep(
    alpha_values: List[float],
    system_curves: Dict[str, List[float]],
    metric: str = "pct_T_and_I",
    save_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for system, values in system_curves.items():
        ax.plot(alpha_values, [v * 100 for v in values],
                marker="o", label=system, color=SYSTEM_COLORS.get(system, "#aaaaaa"))

    ax.set_xlabel("Steering coefficient (alpha)")
    ax.set_ylabel("%Truthful & Informative")
    ax.set_title("Effect of Steering Coefficient on Performance")
    ax.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


def plot_router_confusion_matrix(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    cluster_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    from sklearn.metrics import confusion_matrix
    K = int(max(true_labels.max(), pred_labels.max())) + 1
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(K)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    names = cluster_names or [f"C{k}" for k in range(K)]
    fig, ax = plt.subplots(figsize=(max(4, K), max(4, K)))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=names, yticklabels=names, ax=ax,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted cluster")
    ax.set_ylabel("True cluster")
    ax.set_title("Router Confusion Matrix (normalized)")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


def plot_layer_probe_scores(
    layer_scores: Dict[int, float],
    save_path: Optional[str] = None,
) -> None:
    layers = sorted(layer_scores.keys())
    scores = [layer_scores[l] for l in layers]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, scores, marker="o", color="#4e79a7")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Linear probe accuracy")
    ax.set_title("Layer-wise Hallucination Discriminability (Linear Probe)")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()
