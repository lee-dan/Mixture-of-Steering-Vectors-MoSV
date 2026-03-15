"""plot_router_animation.py — GIF animation of router cluster assignments from K2 to K50."""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
TRAIN_JSONL = os.path.join(BASE_DIR, "data", "defan", "mc_train.jsonl")
PROMPT_VECTORS_PATH = os.path.join(ACT_DIR, "prompt_vectors.npy")

LAYER_POS = 4  # layer index 14 in layers_probed
K_CANDIDATES = [2, 4, 6, 8, 10, 15, 20, 35, 50]

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

FRAMES_DIR = os.path.join(ANALYSIS_DIR, "router_frames")
os.makedirs(FRAMES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_train_categories():
    categories = []
    with open(TRAIN_JSONL, "r") as f:
        for line in f:
            item = json.loads(line)
            categories.append(item.get("category", "unknown"))
    return np.array(categories)


def get_available_k_with_router():
    available = []
    for k in K_CANDIDATES:
        router_path = os.path.join(CKPT_DIR, f"sweep_K{k}", "router.pt")
        if os.path.exists(router_path):
            available.append(k)
    return available


def load_router(k, device="cpu"):
    from mosv.routing.train import load_router as _load_router
    ckpt_dir = os.path.join(CKPT_DIR, f"sweep_K{k}")
    return _load_router(ckpt_dir, device)


def get_router_top1(k, X_tensor):
    import torch
    router = load_router(k)
    router.eval()
    with torch.no_grad():
        logits = router.route_logits(X_tensor)
    return logits.argmax(dim=-1).numpy()


def make_discrete_colormap(k):
    """Return a list of k colors from tab10/tab20."""
    if k <= 10:
        cmap = plt.get_cmap("tab10")
    elif k <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        # Cycle tab20 for larger k
        cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(k)]


def draw_frame(ax, tsne_coords, top1, k, sample_cats):
    """Draw a single animation frame onto ax."""
    ax.clear()

    # Faint background of category coloring
    for cat in CATEGORIES:
        mask = sample_cats == cat
        if mask.sum() == 0:
            continue
        ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                   c=CATEGORY_COLORS[cat], s=5, alpha=0.10, rasterized=True,
                   linewidths=0)

    # Router top-1 foreground
    colors_k = make_discrete_colormap(k)
    for cl in range(k):
        mask = top1 == cl
        if mask.sum() == 0:
            continue
        ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                   c=[colors_k[cl]], s=8, alpha=0.75, rasterized=True,
                   label=f"C{cl}", linewidths=0)

    ax.set_title(f"Router Assignment — K={k}", fontsize=13, pad=8)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Check for cached t-SNE coords
    tsne_coords_path = os.path.join(ANALYSIS_DIR, "tsne_coords_train.npy")
    tsne_idx_path = os.path.join(ANALYSIS_DIR, "tsne_sample_idx.npy")

    if not os.path.exists(tsne_coords_path) or not os.path.exists(tsne_idx_path):
        print("ERROR: t-SNE coordinates not found. Run analysis_all_k.py first to generate:")
        print(f"  {tsne_coords_path}")
        print(f"  {tsne_idx_path}")
        sys.exit(1)

    print(f"Loading t-SNE coords from {tsne_coords_path}...")
    tsne_coords = np.load(tsne_coords_path)
    sample_idx = np.load(tsne_idx_path)
    print(f"  {len(sample_idx)} sample points.")

    if not os.path.exists(TRAIN_JSONL):
        print(f"ERROR: Training JSONL not found at {TRAIN_JSONL}")
        sys.exit(1)

    print("Loading training categories...")
    train_categories = load_train_categories()
    sample_cats = train_categories[sample_idx]

    print(f"Loading prompt vectors (mmap) for {len(sample_idx)} points...")
    try:
        prompt_vecs = np.load(PROMPT_VECTORS_PATH, mmap_mode="r")
        X_sample = np.array(prompt_vecs[sample_idx, LAYER_POS, :], dtype=np.float32)
    except Exception as e:
        print(f"ERROR: Could not load prompt vectors: {e}")
        sys.exit(1)

    import torch
    X_tensor = torch.from_numpy(X_sample)

    # Get available K variants with routers
    available_ks = get_available_k_with_router()
    if not available_ks:
        print("ERROR: No router checkpoints found. Run sweep_K stages first.")
        sys.exit(1)
    print(f"Available K variants with routers: {available_ks}")

    # Pre-compute top-1 assignments for all K
    top1_per_k = {}
    for k in available_ks:
        print(f"  Computing router top-1 for K={k}...")
        try:
            top1 = get_router_top1(k, X_tensor)
            top1_per_k[k] = top1
        except Exception as e:
            print(f"  K={k}: router failed — {e}")

    ks_with_data = sorted([k for k in available_ks if k in top1_per_k])
    if not ks_with_data:
        print("ERROR: No router assignments could be computed.")
        sys.exit(1)

    # Save individual frame PNGs
    print("Saving individual frame PNGs...")
    for k in ks_with_data:
        fig, ax = plt.subplots(figsize=(9, 7))
        draw_frame(ax, tsne_coords, top1_per_k[k], k, sample_cats)

        # Category legend (small inset via legend)
        cat_patches = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c, alpha=0.6)
                       for c in CATEGORIES]
        ax.legend(handles=cat_patches, title="Domain (bg)", fontsize=7,
                  loc="lower left", framealpha=0.8, title_fontsize=7)

        frame_path = os.path.join(FRAMES_DIR, f"frame_K{k}.png")
        fig.savefig(frame_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved frame: {frame_path}")

    # Create GIF animation
    print("Creating GIF animation...")
    fig, ax = plt.subplots(figsize=(9, 7))

    def init():
        ax.clear()
        ax.axis("off")
        return []

    def animate(frame_idx):
        k = ks_with_data[frame_idx]
        draw_frame(ax, tsne_coords, top1_per_k[k], k, sample_cats)
        cat_patches = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c, alpha=0.6)
                       for c in CATEGORIES]
        ax.legend(handles=cat_patches, title="Domain (bg)", fontsize=7,
                  loc="lower left", framealpha=0.8, title_fontsize=7)
        return []

    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(ks_with_data),
        interval=1000,   # ms between frames
        blit=False,
        repeat=True,
    )

    gif_path = os.path.join(ANALYSIS_DIR, "router_animation.gif")
    writer = animation.PillowWriter(fps=1)
    ani.save(gif_path, writer=writer, dpi=100)
    plt.close(fig)
    print(f"Saved GIF animation: {gif_path}")
    print("Done.")


if __name__ == "__main__":
    main()
