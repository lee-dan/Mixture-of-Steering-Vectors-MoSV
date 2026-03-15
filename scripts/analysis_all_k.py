"""analysis_all_k.py — Silhouette, t-SNE, cluster charts, and per-domain heatmap across all K variants."""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRATCH = os.environ.get("SCRATCH", os.path.join("/scratch/users", os.environ.get("USER", "")))
BASE_DIR = os.path.join(SCRATCH, "MoSV-Mixture-of-Steering-Vectors")

ACT_DIR = os.path.join(BASE_DIR, "activations")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUT_BASE = os.path.join(BASE_DIR, "outputs")
ANALYSIS_DIR = os.path.join(OUT_BASE, "analysis")
EVAL_RESULTS = os.path.join(OUT_BASE, "defan_accuracy_results.json")
TRAIN_JSONL = os.path.join(BASE_DIR, "data", "defan", "mc_train.jsonl")

DIFF_VECTORS_PATH = os.path.join(ACT_DIR, "diff_vectors.npy")
PROMPT_VECTORS_PATH = os.path.join(ACT_DIR, "prompt_vectors.npy")

# Layer config
LAYERS_PROBED = [8, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22]
BEST_LAYER = 14
LAYER_POS = 4  # index of layer 14 in LAYERS_PROBED

# Available K variants (gracefully skips missing ones)
K_CANDIDATES = [2, 4, 6, 8, 10, 15, 20, 35, 50]

VANILLA_ACC = 0.197

# Category colors — consistent across all plots
CATEGORY_COLORS = {
    "math":       "#e41a1c",
    "qs_rank":    "#377eb8",
    "census":     "#4daf4a",
    "nobel":      "#984ea3",
    "oscars":     "#ff7f00",
    "un_dates":   "#a65628",
    "conference": "#f781bf",
    "fifa":       "#999999",
}
CATEGORIES = list(CATEGORY_COLORS.keys())

os.makedirs(ANALYSIS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_available_k():
    available = []
    for k in K_CANDIDATES:
        cluster_path = os.path.join(ACT_DIR, f"sweep_K{k}", "cluster_labels.npy")
        if os.path.exists(cluster_path):
            available.append(k)
    return available


def load_train_categories():
    """Load category label for each training item in order."""
    categories = []
    with open(TRAIN_JSONL, "r") as f:
        for line in f:
            item = json.loads(line)
            categories.append(item.get("category", "unknown"))
    return np.array(categories)


def load_cluster_labels(k):
    path = os.path.join(ACT_DIR, f"sweep_K{k}", "cluster_labels.npy")
    return np.load(path)


def load_cluster_meta(k):
    path = os.path.join(ACT_DIR, f"sweep_K{k}", "cluster_meta.json")
    with open(path) as f:
        return json.load(f)


def load_router(k, device="cpu"):
    from mosv.routing.train import load_router as _load_router
    ckpt_dir = os.path.join(CKPT_DIR, f"sweep_K{k}")
    if not os.path.exists(os.path.join(ckpt_dir, "router.pt")):
        return None
    return _load_router(ckpt_dir, device)


# ---------------------------------------------------------------------------
# 1. Silhouette scores
# ---------------------------------------------------------------------------

def compute_silhouette_scores(available_ks):
    print("\n=== [1/6] Silhouette Scores ===")
    from sklearn.metrics import silhouette_score

    print(f"Loading diff_vectors from {DIFF_VECTORS_PATH} (mmap)...")
    diff_vecs = np.load(DIFF_VECTORS_PATH, mmap_mode="r")
    # shape: (N, 11, 4096) → take best layer
    N = diff_vecs.shape[0]
    X_full = diff_vecs[:, LAYER_POS, :]  # (N, 4096)

    SAMPLE_SIZE = 5000
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(N, size=min(SAMPLE_SIZE, N), replace=False)
    X_sample = np.array(X_full[sample_idx], dtype=np.float32)

    results = {}
    for k in available_ks:
        print(f"  K={k}: computing silhouette...")
        labels = load_cluster_labels(k)
        labels_sample = labels[sample_idx]
        n_unique = len(np.unique(labels_sample))
        if n_unique < 2:
            print(f"    K={k}: only {n_unique} cluster(s) in sample, skipping.")
            results[f"K{k}"] = None
            continue
        score = silhouette_score(X_sample, labels_sample, metric="euclidean")
        results[f"K{k}"] = float(score)
        print(f"    K={k}: silhouette = {score:.4f}")

    out_path = os.path.join(ANALYSIS_DIR, "silhouette_scores.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved silhouette scores to {out_path}")
    return results


# ---------------------------------------------------------------------------
# 2. Cluster domain composition bar charts
# ---------------------------------------------------------------------------

def plot_cluster_domain_bars(available_ks, train_categories):
    print("\n=== [2/6] Cluster Domain Composition Bar Charts ===")

    cat_list = CATEGORIES
    color_list = [CATEGORY_COLORS[c] for c in cat_list]

    # Compute per-K composition data
    all_data = {}
    for k in available_ks:
        labels = load_cluster_labels(k)
        composition = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            total = mask.sum()
            if total == 0:
                counts = {c: 0.0 for c in cat_list}
            else:
                counts = {}
                for c in cat_list:
                    counts[c] = float((train_categories[mask] == c).sum()) / total
            composition.append(counts)
        all_data[k] = composition

    # Save individual bar charts
    for k in available_ks:
        print(f"  Plotting domain bars for K={k}...")
        composition = all_data[k]
        fig, ax = plt.subplots(figsize=(10, max(4, k * 0.5 + 2)))

        for cluster_id, counts in enumerate(composition):
            left = 0.0
            for c, color in zip(cat_list, color_list):
                val = counts[c]
                if val > 0:
                    ax.barh(cluster_id, val, left=left, color=color,
                            height=0.7, edgecolor="white", linewidth=0.5)
                left += val

        ax.set_xlabel("Fraction of cluster")
        ax.set_ylabel("Cluster ID")
        ax.set_yticks(range(k))
        ax.set_yticklabels([f"C{i}" for i in range(k)])
        ax.set_xlim(0, 1)
        ax.set_title(f"Cluster Domain Composition — K={k}")

        patches = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c) for c in cat_list]
        ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        plt.tight_layout()

        out_path = os.path.join(ANALYSIS_DIR, f"cluster_domain_bars_K{k}.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {out_path}")

    # Combined figure: all K side by side
    print("  Plotting combined bar chart figure...")
    n_ks = len(available_ks)
    fig, axes = plt.subplots(1, n_ks, figsize=(6 * n_ks, 10), squeeze=False)
    axes = axes[0]

    for ax_idx, k in enumerate(available_ks):
        ax = axes[ax_idx]
        composition = all_data[k]
        for cluster_id, counts in enumerate(composition):
            left = 0.0
            for c, color in zip(cat_list, color_list):
                val = counts[c]
                if val > 0:
                    ax.barh(cluster_id, val, left=left, color=color,
                            height=0.7, edgecolor="white", linewidth=0.3)
                left += val
        ax.set_xlabel("Fraction")
        ax.set_xlim(0, 1)
        ax.set_yticks(range(k))
        ax.set_yticklabels([f"C{i}" for i in range(k)], fontsize=7)
        ax.set_title(f"K={k}", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Cluster ID")

    patches = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c) for c in cat_list]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Cluster Domain Composition by K", fontsize=13)
    plt.tight_layout()

    combined_path = os.path.join(ANALYSIS_DIR, "cluster_domain_bars_combined.png")
    fig.savefig(combined_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved combined figure: {combined_path}")

    return all_data


# ---------------------------------------------------------------------------
# 3. t-SNE of training set
# ---------------------------------------------------------------------------

def compute_or_load_tsne(train_categories):
    print("\n=== [3/6] t-SNE of Training Set ===")
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    tsne_coords_path = os.path.join(ANALYSIS_DIR, "tsne_coords_train.npy")
    tsne_idx_path = os.path.join(ANALYSIS_DIR, "tsne_sample_idx.npy")

    SAMPLE_SIZE = 5000

    if os.path.exists(tsne_coords_path) and os.path.exists(tsne_idx_path):
        print("  Loading cached t-SNE coords...")
        tsne_coords = np.load(tsne_coords_path)
        sample_idx = np.load(tsne_idx_path)
    else:
        print(f"  Loading prompt_vectors from {PROMPT_VECTORS_PATH} (mmap)...")
        prompt_vecs = np.load(PROMPT_VECTORS_PATH, mmap_mode="r")
        N = prompt_vecs.shape[0]

        # Stratified sample by category
        print(f"  Stratified sampling {SAMPLE_SIZE} items...")
        cat_unique = np.unique(train_categories)
        n_per_cat = SAMPLE_SIZE // len(cat_unique)
        rng = np.random.default_rng(42)
        sample_idx_list = []
        for cat in cat_unique:
            cat_idx = np.where(train_categories == cat)[0]
            chosen = rng.choice(cat_idx, size=min(n_per_cat, len(cat_idx)), replace=False)
            sample_idx_list.append(chosen)
        # top up to SAMPLE_SIZE
        sample_idx = np.concatenate(sample_idx_list)
        if len(sample_idx) < SAMPLE_SIZE:
            remaining = np.setdiff1d(np.arange(N), sample_idx)
            extra = rng.choice(remaining, size=SAMPLE_SIZE - len(sample_idx), replace=False)
            sample_idx = np.concatenate([sample_idx, extra])
        sample_idx = sample_idx[:SAMPLE_SIZE]
        np.random.shuffle(sample_idx)

        print(f"  Extracting {len(sample_idx)} prompt vectors at layer_pos={LAYER_POS}...")
        X = np.array(prompt_vecs[sample_idx, LAYER_POS, :], dtype=np.float32)

        print("  Running PCA(50)...")
        pca = PCA(n_components=50, random_state=42)
        X_pca = pca.fit_transform(X)

        print("  Running t-SNE(perplexity=40, max_iter=1000)...")
        tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42,
                    n_jobs=min(8, os.cpu_count() or 1))
        tsne_coords = tsne.fit_transform(X_pca)

        np.save(tsne_coords_path, tsne_coords)
        np.save(tsne_idx_path, sample_idx)
        print(f"  Saved t-SNE coords to {tsne_coords_path}")
        print(f"  Saved sample indices to {tsne_idx_path}")

    sample_cats = train_categories[sample_idx]

    # Plot by category
    print("  Plotting t-SNE colored by category...")
    fig, ax = plt.subplots(figsize=(10, 8))
    for cat in CATEGORIES:
        mask = sample_cats == cat
        if mask.sum() == 0:
            continue
        ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                   c=CATEGORY_COLORS[cat], label=cat, s=4, alpha=0.6, rasterized=True)
    ax.set_title("t-SNE of Training Set — Colored by Category")
    ax.legend(markerscale=3, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.axis("off")
    plt.tight_layout()
    cat_path = os.path.join(ANALYSIS_DIR, "tsne_train_by_category.png")
    fig.savefig(cat_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {cat_path}")

    return tsne_coords, sample_idx, sample_cats


def plot_tsne_by_cluster_and_router(available_ks, tsne_coords, sample_idx, sample_cats,
                                    train_categories):
    print("  Plotting t-SNE by cluster and router assignment for each K...")
    import torch

    for k in available_ks:
        # --- By cluster label ---
        labels = load_cluster_labels(k)
        cluster_labels_sample = labels[sample_idx]

        cmap = plt.get_cmap("tab20" if k > 10 else "tab10")
        colors_k = [cmap(i / max(k - 1, 1)) for i in range(k)]

        fig, ax = plt.subplots(figsize=(10, 8))
        # faint category background
        for cat in CATEGORIES:
            mask = sample_cats == cat
            if mask.sum() == 0:
                continue
            ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                       c=CATEGORY_COLORS[cat], s=4, alpha=0.1, rasterized=True)
        for cl in range(k):
            mask = cluster_labels_sample == cl
            if mask.sum() == 0:
                continue
            ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                       c=[colors_k[cl]], label=f"C{cl}", s=6, alpha=0.7, rasterized=True)
        ax.set_title(f"t-SNE — Cluster Assignment (K={k})")
        ax.legend(markerscale=2, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left",
                  ncol=max(1, k // 10))
        ax.axis("off")
        plt.tight_layout()
        out_path = os.path.join(ANALYSIS_DIR, f"tsne_train_by_cluster_K{k}.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved cluster plot K={k}: {out_path}")

        # --- By router assignment ---
        router = load_router(k)
        if router is None:
            print(f"    K={k}: no router checkpoint found, skipping router plot.")
            continue

        try:
            # Need prompt vectors for the sample
            prompt_vecs = np.load(PROMPT_VECTORS_PATH, mmap_mode="r")
            X_sample = np.array(prompt_vecs[sample_idx, LAYER_POS, :], dtype=np.float32)
            X_tensor = torch.from_numpy(X_sample)
            with torch.no_grad():
                logits = router.route_logits(X_tensor)  # (N_sample, K)
            top1 = logits.argmax(dim=-1).numpy()

            fig, ax = plt.subplots(figsize=(10, 8))
            for cat in CATEGORIES:
                mask = sample_cats == cat
                if mask.sum() == 0:
                    continue
                ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                           c=CATEGORY_COLORS[cat], s=4, alpha=0.1, rasterized=True)
            for cl in range(k):
                mask = top1 == cl
                if mask.sum() == 0:
                    continue
                ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                           c=[colors_k[cl]], label=f"C{cl}", s=6, alpha=0.7, rasterized=True)
            ax.set_title(f"t-SNE — Router Top-1 Assignment (K={k})")
            ax.legend(markerscale=2, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left",
                      ncol=max(1, k // 10))
            ax.axis("off")
            plt.tight_layout()
            out_path = os.path.join(ANALYSIS_DIR, f"tsne_train_by_router_K{k}.png")
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved router plot K={k}: {out_path}")
        except Exception as e:
            print(f"    K={k}: router plot failed — {e}")


# ---------------------------------------------------------------------------
# 4. Cluster interpretability
# ---------------------------------------------------------------------------

def compute_cluster_interpretability(available_ks, train_categories, domain_comp_data):
    print("\n=== [4/6] Cluster Interpretability Analysis ===")
    from scipy.stats import entropy as scipy_entropy

    results = {}
    for k in available_ks:
        composition = domain_comp_data[k]
        clusters_info = []
        n_interpretable = 0
        for cluster_id, counts in enumerate(composition):
            vals = np.array([counts.get(c, 0.0) for c in CATEGORIES])
            if vals.sum() == 0:
                clusters_info.append({
                    "cluster_id": cluster_id,
                    "dominant_category": None,
                    "dominant_pct": 0.0,
                    "entropy": float("nan"),
                    "interpretable": False,
                })
                continue
            dominant_idx = int(np.argmax(vals))
            dominant_cat = CATEGORIES[dominant_idx]
            dominant_pct = float(vals[dominant_idx])
            ent = float(scipy_entropy(vals + 1e-12, base=2))
            interpretable = dominant_pct > 0.60
            if interpretable:
                n_interpretable += 1
            clusters_info.append({
                "cluster_id": cluster_id,
                "dominant_category": dominant_cat,
                "dominant_pct": round(dominant_pct, 4),
                "entropy": round(ent, 4),
                "interpretable": interpretable,
            })
        results[f"K{k}"] = {
            "clusters": clusters_info,
            "n_interpretable": n_interpretable,
            "pct_interpretable": round(n_interpretable / k, 4) if k > 0 else 0.0,
        }

    out_path = os.path.join(ANALYSIS_DIR, "cluster_interpretability.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved interpretability data to {out_path}")

    # Human-readable summary
    print("\n--- Interpretability Summary ---")
    for k in available_ks:
        info = results[f"K{k}"]
        pct = info["pct_interpretable"] * 100
        print(f"  K={k}: {info['n_interpretable']}/{k} interpretable clusters ({pct:.0f}%)")
        for cl in info["clusters"]:
            tag = " [INTERPRETABLE]" if cl["interpretable"] else ""
            dom = cl["dominant_category"] or "?"
            print(f"    C{cl['cluster_id']}: dominant={dom} ({cl['dominant_pct']*100:.1f}%), "
                  f"entropy={cl['entropy']:.2f}{tag}")

    return results


# ---------------------------------------------------------------------------
# 5. Correlation analysis
# ---------------------------------------------------------------------------

def run_correlation_analysis(available_ks, silhouette_data):
    print("\n=== [5/6] Correlation Analysis ===")

    if not os.path.exists(EVAL_RESULTS):
        print(f"  WARNING: {EVAL_RESULTS} not found, skipping correlation analysis.")
        return {}

    with open(EVAL_RESULTS) as f:
        eval_data = json.load(f)

    rows = []
    for k in available_ks:
        key = f"K{k}"
        sil = silhouette_data.get(key)
        mosv_acc = eval_data.get(key)
        if mosv_acc is None:
            # Try lowercase
            mosv_acc = eval_data.get(key.lower())
        if sil is None or mosv_acc is None:
            print(f"  K={k}: missing silhouette or accuracy, skipping.")
            continue

        # cluster_sep_acc from cluster_meta
        try:
            meta = load_cluster_meta(k)
            # silhouette_scores in meta are per-layer; use best_layer_idx
            best_layer_idx = meta.get("best_layer_idx", LAYER_POS)
            sep_scores = meta.get("silhouette_scores", {})
            # silhouette_scores may be keyed by layer index as string
            cluster_sil = sep_scores.get(str(best_layer_idx), sep_scores.get(best_layer_idx))
            if cluster_sil is None and isinstance(sep_scores, list):
                cluster_sil = sep_scores[best_layer_idx] if best_layer_idx < len(sep_scores) else None
        except Exception as e:
            print(f"  K={k}: cluster_meta error — {e}")
            cluster_sil = None

        rows.append({
            "K": k,
            "silhouette": float(sil),
            "cluster_sep_sil": float(cluster_sil) if cluster_sil is not None else None,
            "mosv_acc": float(mosv_acc),
        })

    if len(rows) < 2:
        print("  Not enough data points for correlation. Skipping plots.")
        return {}

    ks = np.array([r["K"] for r in rows])
    sils = np.array([r["silhouette"] for r in rows])
    accs = np.array([r["mosv_acc"] for r in rows])

    from scipy.stats import pearsonr

    corr_sil_acc, p_sil_acc = pearsonr(sils, accs) if len(rows) >= 2 else (float("nan"), float("nan"))
    corr_k_acc, p_k_acc = pearsonr(ks, accs) if len(rows) >= 2 else (float("nan"), float("nan"))

    print(f"  Pearson r (silhouette vs MoSV acc): {corr_sil_acc:.3f} (p={p_sil_acc:.3f})")
    print(f"  Pearson r (K vs MoSV acc):          {corr_k_acc:.3f} (p={p_k_acc:.3f})")

    corr_results = {
        "pearson_silhouette_vs_acc": {"r": round(corr_sil_acc, 4), "p": round(p_sil_acc, 4)},
        "pearson_K_vs_acc": {"r": round(corr_k_acc, 4), "p": round(p_k_acc, 4)},
        "rows": rows,
    }

    # cluster_sep_sil vs acc
    sep_rows = [r for r in rows if r["cluster_sep_sil"] is not None]
    if len(sep_rows) >= 2:
        sep_sils = np.array([r["cluster_sep_sil"] for r in sep_rows])
        sep_accs = np.array([r["mosv_acc"] for r in sep_rows])
        corr_sep, p_sep = pearsonr(sep_sils, sep_accs)
        corr_results["pearson_cluster_sep_sil_vs_acc"] = {
            "r": round(corr_sep, 4), "p": round(p_sep, 4)
        }
        print(f"  Pearson r (cluster_sep_sil vs MoSV acc): {corr_sep:.3f} (p={p_sep:.3f})")

    # Scatter: silhouette vs acc
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.scatter(sils, accs, c="steelblue", s=60, zorder=3)
    for r in rows:
        ax.annotate(f"K{r['K']}", (r["silhouette"], r["mosv_acc"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.axhline(VANILLA_ACC, color="gray", linestyle="--", linewidth=1, label=f"vanilla ({VANILLA_ACC:.3f})")
    ax.set_xlabel("Silhouette Score")
    ax.set_ylabel("MoSV Accuracy")
    ax.set_title(f"Silhouette vs MoSV Acc\nr={corr_sil_acc:.3f}")
    ax.legend(fontsize=8)

    ax = axes[1]
    if len(sep_rows) >= 2:
        ax.scatter(sep_sils, sep_accs, c="tomato", s=60, zorder=3)
        for r in sep_rows:
            ax.annotate(f"K{r['K']}", (r["cluster_sep_sil"], r["mosv_acc"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.axhline(VANILLA_ACC, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Cluster Sep Silhouette")
        ax.set_ylabel("MoSV Accuracy")
        r_val = corr_results.get("pearson_cluster_sep_sil_vs_acc", {}).get("r", float("nan"))
        ax.set_title(f"Cluster Sep Sil vs MoSV Acc\nr={r_val:.3f}")
    else:
        ax.set_title("Cluster Sep Sil vs MoSV Acc\n(insufficient data)")

    ax = axes[2]
    ax.scatter(ks, accs, c="seagreen", s=60, zorder=3)
    for r in rows:
        ax.annotate(f"K{r['K']}", (r["K"], r["mosv_acc"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.axhline(VANILLA_ACC, color="gray", linestyle="--", linewidth=1, label=f"vanilla ({VANILLA_ACC:.3f})")
    ax.set_xlabel("K")
    ax.set_ylabel("MoSV Accuracy")
    ax.set_title(f"K vs MoSV Acc\nr={corr_k_acc:.3f}")
    ax.legend(fontsize=8)

    plt.suptitle("Correlation Analysis — DefAn MoSV", fontsize=12)
    plt.tight_layout()

    corr_plot_path = os.path.join(ANALYSIS_DIR, "correlation_analysis.png")
    fig.savefig(corr_plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved correlation plot: {corr_plot_path}")

    out_path = os.path.join(ANALYSIS_DIR, "correlation_analysis.json")
    with open(out_path, "w") as f:
        json.dump(corr_results, f, indent=2)
    print(f"  Saved correlation data: {out_path}")

    return corr_results


# ---------------------------------------------------------------------------
# 6. Per-domain accuracy heatmap
# ---------------------------------------------------------------------------

def run_per_domain_analysis(available_ks):
    print("\n=== [6/6] Per-Domain Accuracy Breakdown ===")

    if not os.path.exists(EVAL_RESULTS):
        print(f"  WARNING: {EVAL_RESULTS} not found, skipping.")
        return {}

    with open(EVAL_RESULTS) as f:
        eval_data = json.load(f)

    domains = CATEGORIES
    all_deltas = {}  # key = "K{k}", value = {domain: delta}
    per_domain_acc = {}  # key = "K{k}", value = {domain: acc}

    for k in available_ks:
        key = f"K{k}"
        domain_key = f"{key}_by_domain"
        if domain_key not in eval_data:
            # Try lowercase
            domain_key_lc = domain_key.lower()
            if domain_key_lc in eval_data:
                domain_key = domain_key_lc
            else:
                print(f"  K={k}: no per-domain data found, skipping.")
                continue
        domain_data = eval_data[domain_key]
        deltas = {}
        accs = {}
        for dom in domains:
            acc = domain_data.get(dom)
            if acc is not None:
                deltas[dom] = round(float(acc) - VANILLA_ACC, 4)
                accs[dom] = round(float(acc), 4)
            else:
                deltas[dom] = None
                accs[dom] = None
        all_deltas[key] = deltas
        per_domain_acc[key] = accs

    out_data = {"vanilla_acc": VANILLA_ACC, "per_domain_delta": all_deltas,
                "per_domain_acc": per_domain_acc}
    out_path = os.path.join(ANALYSIS_DIR, "per_domain_accuracy.json")
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"  Saved per-domain data: {out_path}")

    if not all_deltas:
        print("  No per-domain data available for heatmap.")
        return out_data

    # Build heatmap matrix
    k_labels = sorted(all_deltas.keys(), key=lambda x: int(x[1:]))
    n_rows = len(k_labels)
    n_cols = len(domains)
    matrix = np.full((n_rows, n_cols), np.nan)
    for row_i, kl in enumerate(k_labels):
        for col_j, dom in enumerate(domains):
            v = all_deltas[kl].get(dom)
            if v is not None:
                matrix[row_i, col_j] = v

    vmax = max(0.05, np.nanmax(np.abs(matrix)))
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.2), max(4, n_rows * 0.8 + 2)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(domains, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(k_labels, fontsize=9)
    ax.set_xlabel("Domain")
    ax.set_ylabel("K variant")
    ax.set_title(f"Per-Domain Accuracy Delta vs Vanilla ({VANILLA_ACC:.3f})")
    plt.colorbar(im, ax=ax, label="Δ accuracy (pp)")

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        fontsize=7, color="black")

    plt.tight_layout()
    heatmap_path = os.path.join(ANALYSIS_DIR, "per_domain_heatmap.png")
    fig.savefig(heatmap_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap: {heatmap_path}")

    return out_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Analysis output directory: {ANALYSIS_DIR}")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    available_ks = get_available_k()
    if not available_ks:
        print("ERROR: No K variants found in activations directory. Check ACT_DIR path.")
        sys.exit(1)
    print(f"Available K variants: {available_ks}")

    # Load training categories once (needed by multiple steps)
    if not os.path.exists(TRAIN_JSONL):
        print(f"ERROR: Training JSONL not found at {TRAIN_JSONL}")
        sys.exit(1)
    print("Loading training categories...")
    train_categories = load_train_categories()
    print(f"Loaded {len(train_categories)} items.")

    # 1. Silhouette scores
    try:
        silhouette_data = compute_silhouette_scores(available_ks)
    except Exception as e:
        print(f"[1] Silhouette computation failed: {e}")
        silhouette_data = {}

    # 2. Domain composition bars
    try:
        domain_comp_data = plot_cluster_domain_bars(available_ks, train_categories)
    except Exception as e:
        print(f"[2] Domain bar charts failed: {e}")
        domain_comp_data = {k: [] for k in available_ks}

    # 3. t-SNE
    try:
        tsne_coords, sample_idx, sample_cats = compute_or_load_tsne(train_categories)
        plot_tsne_by_cluster_and_router(available_ks, tsne_coords, sample_idx, sample_cats,
                                        train_categories)
    except Exception as e:
        import traceback
        print(f"[3] t-SNE failed: {e}")
        traceback.print_exc()

    # 4. Interpretability
    try:
        interp_data = compute_cluster_interpretability(available_ks, train_categories,
                                                        domain_comp_data)
    except Exception as e:
        print(f"[4] Interpretability analysis failed: {e}")
        interp_data = {}

    # 5. Correlation
    try:
        corr_data = run_correlation_analysis(available_ks, silhouette_data)
    except Exception as e:
        import traceback
        print(f"[5] Correlation analysis failed: {e}")
        traceback.print_exc()
        corr_data = {}

    # 6. Per-domain
    try:
        run_per_domain_analysis(available_ks)
    except Exception as e:
        print(f"[6] Per-domain analysis failed: {e}")

    print("\n=== All analyses complete ===")
    print(f"Results saved to: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
