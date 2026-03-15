"""prepare_data.py — Download DefAn, split per-domain 85/15, run LLaMA inference, output contrastive pairs."""

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAN_BASE = "https://raw.githubusercontent.com/ashikiut/DefAn/main/DefAn-public"
DOMAIN_FILES = {
    "fifa":       "QA_domain_1_public.json",
    "census":     "QA_domain_2_public.json",
    "nobel":      "QA_domain_3_public.json",
    "oscars":     "QA_domain_4_public.json",
    "un_dates":   "QA_domain_5_public.json",
    "qs_rank":    "QA_domain_6_public.json",
    "conference": "QA_domain_7_public.json",
    "math":       "QA_domain_8_public.json",
}


def download_defan(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    by_domain = defaultdict(list)
    for domain, fname in DOMAIN_FILES.items():
        local = os.path.join(cache_dir, fname)
        if not os.path.exists(local):
            url = f"{DEFAN_BASE}/{fname}"
            print(f"  Downloading {url}...")
            urllib.request.urlretrieve(url, local)
        with open(local) as f:
            data = json.load(f)
        items = data if isinstance(data, list) else list(data.values())
        for item in items:
            by_domain[domain].append({
                "question": item.get("questions", "").strip(),
                "answer":   str(item.get("answer", "")).strip(),
                "type":     item.get("type", ""),
                "domain":   domain,
            })
        print(f"  {domain}: {len(items)} rows")
    return by_domain


def split_domain(rows, eval_fraction, seed=42):
    import random
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_fraction))
    return shuffled[n_eval:], shuffled[:n_eval]   # train, eval


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_hallucination(resp, gt, answer_type):
    resp_norm = normalize(resp)
    gt_norm = normalize(gt)
    if not gt_norm:
        return False
    return gt_norm not in resp_norm


def build_prompt(question, tokenizer):
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_inference(rows, model, tokenizer, device, batch_size=16, max_new_tokens=60):
    results = []
    for i in tqdm(range(0, len(rows), batch_size), desc="LLaMA inference"):
        batch = rows[i:i + batch_size]
        prompts = [build_prompt(r["question"], tokenizer) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        for j, row in enumerate(batch):
            new_ids = out[j, input_len:]
            resp = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            results.append({**row, "llama_response": resp})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/defan.yaml")
    parser.add_argument("--eval_fraction", type=float, default=0.15)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["paths"]["data_dir"]
    cache_dir = os.path.join("data/defan/raw")  # reuse cached downloads
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "mc_train.jsonl")
    eval_path  = os.path.join(data_dir, "defan_eval.jsonl")

    if os.path.exists(train_path) and os.path.exists(eval_path):
        print(f"Both files exist — skipping. Train: {sum(1 for _ in open(train_path))}, Eval: {sum(1 for _ in open(eval_path))}")
        return

    model_name = cfg["model"]["name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading DefAn data...")
    by_domain = download_defan(cache_dir)
    total = sum(len(v) for v in by_domain.values())
    print(f"Total rows: {total}")

    # Split per domain BEFORE any inference
    train_rows, eval_rows = [], []
    for domain, rows in by_domain.items():
        tr, ev = split_domain(rows, args.eval_fraction)
        train_rows.extend(tr)
        eval_rows.extend(ev)
        print(f"  {domain}: {len(tr)} train, {len(ev)} eval")
    print(f"Total — train: {len(train_rows)}, eval: {len(eval_rows)}")

    # Load model for training inference only
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Re-save eval with proper prompts (tokenizer now loaded)
    with open(eval_path, "w") as f:
        for r in eval_rows:
            f.write(json.dumps({
                "prompt":         build_prompt(r["question"], tokenizer),
                "question":       r["question"],
                "correct_answer": r["answer"],
                "domain":         r["domain"],
                "type":           r["type"],
                "source_dataset": "defan",
            }) + "\n")
    print(f"Saved {len(eval_rows)} eval items -> {eval_path}")

    load_kwargs = {"device_map": "auto", "dtype": torch.float16}
    if cfg["model"].get("load_in_8bit"):
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs.pop("dtype", None)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    print("Model loaded.")

    # Run inference on TRAIN rows only
    batch_size = cfg["activation"].get("batch_size", 16)
    results = run_inference(train_rows, model, tokenizer, device, batch_size=batch_size)

    # Build contrastive pairs
    pairs, stats = [], {"hallucinated": 0, "correct": 0}
    for r in results:
        if is_hallucination(r["llama_response"], r["answer"], r["type"]):
            pairs.append({
                "prompt":           r["question"],
                "correct_answer":   r["answer"],
                "incorrect_answer": r["llama_response"],
                "question":         r["question"],
                "category":         r["domain"],
                "source_dataset":   "defan",
            })
            stats["hallucinated"] += 1
        else:
            stats["correct"] += 1

    print(f"Train pairs: {stats['hallucinated']} hallucination, {stats['correct']} correct (yield {stats['hallucinated']/len(results)*100:.1f}%)")

    with open(train_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Saved {len(pairs)} training pairs -> {train_path}")


if __name__ == "__main__":
    main()
