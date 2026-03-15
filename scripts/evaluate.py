"""evaluate.py — Exact-match accuracy eval on held-out DefAn for vanilla, single-vec, and MoSV-K variants."""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_correct(response, ground_truth):
    return normalize(ground_truth) in normalize(response)


def generate_responses(system, items, batch_size=16, max_new_tokens=80):
    prompts = [item["question"] for item in items]
    responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating", leave=False):
        batch = prompts[i:i + batch_size]
        responses.extend(system.generate_batch(batch, max_new_tokens=max_new_tokens, do_sample=False))
    return responses


def score(responses, items):
    correct = sum(is_correct(r, it["correct_answer"]) for r, it in zip(responses, items))
    return correct / len(items) if items else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/defan.yaml")
    parser.add_argument("--k_tags", nargs="+", default=["K2", "K4", "K6", "K10", "K15", "K20"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--vanilla_acc", type=float, default=None,
                        help="Skip vanilla generation and use this pre-computed accuracy")
    parser.add_argument("--singlevec_acc", type=float, default=None,
                        help="Skip single-vec generation and use this pre-computed accuracy")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir   = cfg["paths"]["data_dir"]
    act_dir    = cfg["paths"]["activations_dir"]
    ckpt_dir   = cfg["paths"]["checkpoints_dir"]
    out_dir    = cfg["paths"]["outputs_dir"]
    layers     = cfg["activation"]["layers_to_probe"]
    batch_size = cfg["evaluation"].get("mc_batch_size", 16)
    os.makedirs(out_dir, exist_ok=True)

    eval_path = os.path.join(data_dir, "defan_eval.jsonl")
    items = [json.loads(l) for l in open(eval_path)]
    print(f"Loaded {len(items)} eval items")

    from mosv.activation.extract import load_activations
    from mosv.clustering.cluster import load_clustering
    from mosv.routing.train import load_router
    from mosv.steering.mosv import MoSV
    from mosv.steering.baselines import VanillaBaseline, SingleVecBaseline
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import BitsAndBytesConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg["model"]["name"]

    print(f"Loading model {model_name}...")
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

    diff_vectors, prompt_vectors = load_activations(act_dir)

    # Best layer from first available K
    first_tag = args.k_tags[0]
    sub_act = os.path.join(act_dir, f"sweep_{first_tag}")
    _, _, meta = load_clustering(sub_act)
    best_layer = meta["best_layer_idx"]
    best_layer_pos = layers.index(best_layer)

    # Vanilla
    if args.vanilla_acc is not None:
        van_acc = args.vanilla_acc
        print(f"\n--- Vanilla (pre-computed) ---")
        print(f"  Accuracy: {van_acc*100:.1f}%")
    else:
        print("\n--- Vanilla ---")
        vanilla = VanillaBaseline(model, tokenizer)
        van_resp = generate_responses(vanilla, items, batch_size=cfg["evaluation"].get("mc_batch_size", 16))
        van_acc = score(van_resp, items)
        print(f"  Accuracy: {van_acc*100:.1f}%")

    # Single-vec
    if args.singlevec_acc is not None:
        sv_acc = args.singlevec_acc
        print(f"--- Single-vec (pre-computed) ---")
        print(f"  Accuracy: {sv_acc*100:.1f}%")
    else:
        print("--- Single-vec ---")
        sv = SingleVecBaseline(model, tokenizer, diff_vectors,
                               steer_layer=best_layer, layer_pos=best_layer_pos, alpha=args.alpha)
        sv_resp = generate_responses(sv, items, batch_size=cfg["evaluation"].get("mc_batch_size", 16))
        sv_acc = score(sv_resp, items)
    print(f"  Accuracy: {sv_acc*100:.1f}%")

    results = {"vanilla": van_acc, "single-vec": sv_acc}

    for tag in args.k_tags:
        sub_act_dir  = os.path.join(act_dir,  f"sweep_{tag}")
        sub_ckpt_dir = os.path.join(ckpt_dir, f"sweep_{tag}")
        if not os.path.exists(sub_ckpt_dir):
            print(f"  Skipping {tag} — no checkpoint")
            continue

        steering_vectors, _, meta = load_clustering(sub_act_dir)
        K = meta["K"]
        router = load_router(sub_ckpt_dir, device)

        mosv = MoSV(model, tokenizer, router, steering_vectors,
                    steer_layer=best_layer, alpha=args.alpha)

        print(f"--- MoSV-{tag} (K={K}) ---")
        mosv_resp = generate_responses(mosv, items, batch_size=batch_size)
        acc = score(mosv_resp, items)
        print(f"  Accuracy: {acc*100:.1f}%  (Δ vs vanilla: {(acc-van_acc)*100:+.1f}pp)")
        results[tag] = acc

        # Per-domain breakdown
        domain_results = defaultdict(lambda: {"correct": 0, "total": 0})
        for resp, item in zip(mosv_resp, items):
            d = item.get("domain", "unknown")
            domain_results[d]["total"] += 1
            if is_correct(resp, item["correct_answer"]):
                domain_results[d]["correct"] += 1
        results[f"{tag}_by_domain"] = {
            d: v["correct"]/v["total"] for d, v in domain_results.items()
        }

    # Summary
    print(f"\n{'='*50}")
    print(f"{'System':<15} {'Accuracy':>10} {'Delta':>8}")
    print("-" * 35)
    for sys_name, acc in results.items():
        if "_by_domain" in sys_name:
            continue
        delta = f"{(acc-van_acc)*100:+.1f}pp" if sys_name != "vanilla" else "—"
        print(f"  {sys_name:<13} {acc*100:>9.1f}%  {delta:>8}")

    out_path = os.path.join(out_dir, "defan_accuracy_results.json")
    # Merge with existing results so parallel runs don't overwrite each other
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        existing.update(results)
        results = existing
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
