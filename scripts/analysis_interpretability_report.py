"""analysis_interpretability_report.py — Text interpretability report: cluster purity, silhouette, correlation, per-domain accuracy."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRATCH = os.environ.get("SCRATCH", os.path.join("/scratch/users", os.environ.get("USER", "")))
BASE_DIR = os.path.join(SCRATCH, "MoSV-Mixture-of-Steering-Vectors")

OUT_BASE = os.path.join(BASE_DIR, "outputs")
ANALYSIS_DIR = os.path.join(OUT_BASE, "analysis")
EVAL_RESULTS = os.path.join(OUT_BASE, "defan_accuracy_results.json")

INTERP_JSON = os.path.join(ANALYSIS_DIR, "cluster_interpretability.json")
SIL_JSON = os.path.join(ANALYSIS_DIR, "silhouette_scores.json")
CORR_JSON = os.path.join(ANALYSIS_DIR, "correlation_analysis.json")
DOMAIN_JSON = os.path.join(ANALYSIS_DIR, "per_domain_accuracy.json")

VANILLA_ACC = 0.197
CATEGORIES = ["math", "qs_rank", "census", "nobel", "oscars", "un_dates", "conference", "fifa"]

REPORT_PATH = os.path.join(ANALYSIS_DIR, "interpretability_report.txt")


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def load_json_safe(path, label):
    if not os.path.exists(path):
        print(f"WARNING: {label} not found at {path}")
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def format_delta(delta):
    if delta is None:
        return "N/A"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.1f}pp"


def generate_report(lines_out):
    """Generate report; lines_out is a list to which we append strings."""

    def emit(s=""):
        lines_out.append(s)
        print(s)

    interp_data = load_json_safe(INTERP_JSON, "cluster_interpretability.json")
    sil_data = load_json_safe(SIL_JSON, "silhouette_scores.json")
    corr_data = load_json_safe(CORR_JSON, "correlation_analysis.json")
    domain_data = load_json_safe(DOMAIN_JSON, "per_domain_accuracy.json")
    eval_data = load_json_safe(EVAL_RESULTS, "defan_accuracy_results.json")

    per_domain_delta = domain_data.get("per_domain_delta", {})
    per_domain_acc = domain_data.get("per_domain_acc", {})

    # Figure out sorted K list
    all_k_keys = sorted(
        set(interp_data.keys()) | set(sil_data.keys()),
        key=lambda x: int(x[1:]) if x[1:].isdigit() else 9999
    )

    emit("=" * 70)
    emit("MoSV DefAn — Cluster Interpretability & Analysis Report")
    emit("=" * 70)
    emit(f"Vanilla accuracy: {VANILLA_ACC * 100:.1f}%")
    emit(f"K variants analysed: {', '.join(all_k_keys)}")
    emit()

    # Per-K sections
    for key in all_k_keys:
        k_num_str = key[1:]
        if not k_num_str.isdigit():
            continue
        k = int(k_num_str)

        emit(f"=== {key} ===")

        # MoSV accuracy
        mosv_acc = eval_data.get(key)
        if mosv_acc is None:
            mosv_acc = eval_data.get(key.lower())
        if mosv_acc is not None:
            delta_pp = (float(mosv_acc) - VANILLA_ACC) * 100
            sign = "+" if delta_pp >= 0 else ""
            emit(f"MoSV accuracy: {float(mosv_acc) * 100:.1f}%  "
                 f"(Δ vs vanilla: {sign}{delta_pp:.1f}pp)")
        else:
            emit("MoSV accuracy: N/A")

        # Silhouette
        sil = sil_data.get(key)
        sil_str = f"{sil:.3f}" if sil is not None else "N/A"

        # Cluster sep acc from eval_data or N/A
        # (We store it in corr_data rows if available)
        cluster_sep_str = "N/A"
        if corr_data:
            for row in corr_data.get("rows", []):
                if row.get("K") == k:
                    cs = row.get("cluster_sep_sil")
                    if cs is not None:
                        cluster_sep_str = f"{float(cs) * 100:.1f}%"
                    break

        emit(f"Silhouette: {sil_str} | Cluster sep sil: {cluster_sep_str}")

        # Clusters
        k_interp = interp_data.get(key, {})
        clusters = k_interp.get("clusters", [])
        if clusters:
            emit("Clusters:")
            for cl in clusters:
                cid = cl["cluster_id"]
                dom = cl.get("dominant_category") or "?"
                pct = cl.get("dominant_pct", 0.0) * 100
                ent = cl.get("entropy", float("nan"))
                tag = " [INTERPRETABLE]" if cl.get("interpretable") else ""
                ent_str = f"{ent:.2f}" if not (ent != ent) else "NaN"
                emit(f"  C{cid}: dominant={dom} ({pct:.1f}%), entropy={ent_str}{tag}")
        else:
            emit("Clusters: (no interpretability data)")

        # Per-domain delta
        k_deltas = per_domain_delta.get(key, {})
        if k_deltas:
            delta_parts = []
            for dom in CATEGORIES:
                d = k_deltas.get(dom)
                if d is not None:
                    sign = "+" if d >= 0 else ""
                    delta_parts.append(f"{dom}: {sign}{d * 100:.1f}pp")
                else:
                    delta_parts.append(f"{dom}: N/A")
            emit(f"Per-domain delta vs vanilla ({VANILLA_ACC * 100:.1f}%):")
            emit(f"  {' | '.join(delta_parts)}")
        else:
            emit("Per-domain delta: (no data)")

        emit()

    # Summary section
    emit("=" * 70)
    emit("SUMMARY")
    emit("=" * 70)

    # Best K by accuracy
    best_k = None
    best_acc = -1.0
    for key in all_k_keys:
        acc = eval_data.get(key) or eval_data.get(key.lower())
        if acc is not None and float(acc) > best_acc:
            best_acc = float(acc)
            best_k = key
    if best_k is not None:
        delta_best = (best_acc - VANILLA_ACC) * 100
        sign = "+" if delta_best >= 0 else ""
        emit(f"Best K by accuracy: {best_k}  ({best_acc * 100:.1f}%, {sign}{delta_best:.1f}pp vs vanilla)")
    else:
        emit("Best K by accuracy: N/A (no eval results)")

    # Correlation silhouette vs accuracy
    if corr_data:
        sil_corr = corr_data.get("pearson_silhouette_vs_acc", {})
        r = sil_corr.get("r")
        p = sil_corr.get("p")
        if r is not None:
            emit(f"Correlation (silhouette vs accuracy): Pearson r={r:.3f}  (p={p:.3f})")
        k_corr = corr_data.get("pearson_K_vs_acc", {})
        rk = k_corr.get("r")
        pk = k_corr.get("p")
        if rk is not None:
            emit(f"Correlation (K vs accuracy):          Pearson r={rk:.3f}  (p={pk:.3f})")
    else:
        emit("Correlation: (no data)")

    # Most interpretable K
    best_interp_k = None
    best_interp_pct = -1.0
    for key in all_k_keys:
        pct = interp_data.get(key, {}).get("pct_interpretable", 0.0)
        if pct > best_interp_pct:
            best_interp_pct = pct
            best_interp_k = key
    if best_interp_k is not None:
        n_int = interp_data.get(best_interp_k, {}).get("n_interpretable", 0)
        k_val = int(best_interp_k[1:]) if best_interp_k[1:].isdigit() else "?"
        emit(f"Most interpretable K: {best_interp_k}  "
             f"({n_int}/{k_val} clusters, {best_interp_pct * 100:.0f}% interpretable)")
    else:
        emit("Most interpretable K: N/A")

    # Which domains benefit most
    domain_gains = {dom: [] for dom in CATEGORIES}
    for key in all_k_keys:
        k_deltas = per_domain_delta.get(key, {})
        for dom in CATEGORIES:
            d = k_deltas.get(dom)
            if d is not None:
                domain_gains[dom].append(d)
    emit("Average per-domain delta vs vanilla (across all K):")
    sorted_doms = sorted(
        [d for d in CATEGORIES if domain_gains[d]],
        key=lambda d: -sum(domain_gains[d]) / len(domain_gains[d])
    )
    for dom in sorted_doms:
        vals = domain_gains[dom]
        avg = sum(vals) / len(vals)
        sign = "+" if avg >= 0 else ""
        emit(f"  {dom:12s}: avg {sign}{avg * 100:.1f}pp  "
             f"(best: {max(vals) * 100:+.1f}pp, worst: {min(vals) * 100:+.1f}pp)")
    for dom in CATEGORIES:
        if not domain_gains[dom]:
            emit(f"  {dom:12s}: no data")

    emit()
    emit("Report complete.")


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    lines = []
    generate_report(lines)

    report_text = "\n".join(lines) + "\n"
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
