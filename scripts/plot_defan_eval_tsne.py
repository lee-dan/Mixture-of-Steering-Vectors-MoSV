"""plot_defan_eval_tsne.py — t-SNE of DefAn eval set prompt activations colored by domain."""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DOMAIN_COLORS = {
    "math":       "#e41a1c",
    "qs_rank":    "#377eb8",
    "census":     "#4daf4a",
    "nobel":      "#984ea3",
    "oscars":     "#ff7f00",
    "un_dates":   "#a65628",
    "conference": "#f781bf",
    "fifa":       "#999999",
}


def extract_prompt_activations(model, tokenizer, items, layer_idx, device, batch_size=32):
    activations = []

    for i in tqdm(range(0, len(items), batch_size), desc="Extracting activations"):
        batch = items[i:i + batch_size]
        prompts = [it["prompt"] for it in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)

        captured = {}

        def hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            captured["h"] = hidden[:, -1, :].detach().float().cpu()

        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            model(**enc)
        handle.remove()

        activations.append(captured["h"].numpy())

    return np.concatenate(activations, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/defan.yaml")
    parser.add_argument("--out", default="outputs/figures/defan_eval_tsne_by_domain.png")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pca_components", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=40)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["paths"]["data_dir"]
    act_dir  = cfg["paths"]["activations_dir"]
    layers   = cfg["activation"]["layers_to_probe"]

    eval_path = os.path.join(data_dir, "defan_eval.jsonl")
    items = [json.loads(l) for l in open(eval_path)]
    domains = [it.get("domain", "unknown") for it in items]
    print(f"Loaded {len(items)} eval items")
    unique_domains = sorted(set(domains))
    print(f"Domains: {unique_domains}")

    # Get best layer from K2 clustering metadata
    from mosv.clustering.cluster import load_clustering
    sub_act = os.path.join(act_dir, "sweep_K2")
    _, _, meta = load_clustering(sub_act)
    best_layer = meta["best_layer_idx"]
    print(f"Best layer: {best_layer}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg["model"]["name"]

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto", "dtype": torch.float16}
    if cfg["model"].get("load_in_8bit"):
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs.pop("dtype", None)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    print("Model loaded.")

    X = extract_prompt_activations(model, tokenizer, items, best_layer, device, args.batch_size)
    print(f"Activations shape: {X.shape}")

    print("Running PCA...")
    X_pca = PCA(n_components=args.pca_components, random_state=42).fit_transform(X)

    print("Running t-SNE...")
    X_tsne = TSNE(n_components=2, perplexity=args.perplexity,
                  max_iter=1000, random_state=42).fit_transform(X_pca)

    print("Plotting...")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    domains_arr = np.array(domains)

    for dom in unique_domains:
        mask = domains_arr == dom
        color = DOMAIN_COLORS.get(dom, "#333333")
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=color, label=dom, s=4, alpha=0.5, linewidths=0)

    ax.legend(markerscale=4, fontsize=11, loc="best")
    ax.set_title("DefAn Eval Set — Prompt Activations t-SNE by Domain\n"
                 f"(10,615 held-out questions, layer {best_layer})", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
