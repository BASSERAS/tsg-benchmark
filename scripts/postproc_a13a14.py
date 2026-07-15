#!/usr/bin/env python3
"""Post-process A13/A14 for PT methods using saved .npy samples + TF1 env.
Computes Discriminative Score (A13) and Predictive Score (A14) for
Fourier-flows, CSDI, GT-GAN, Diffusion-TS using TimeGAN's TF1 metrics.

Usage: ./miniconda3/envs/tf1_env/bin/python scripts/postproc_a13a14.py
"""
import json, os, sys, warnings, numpy as np, glob as globmod
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SAMPLES = os.path.join(ROOT, "results", "samples")
RESULTS = os.path.join(ROOT, "results", "all_results.jsonl")

# Add TimeGAN repo to path for metrics module
_tg_dir = os.path.join(ROOT, "repos", "TimeGAN")
sys.path.insert(0, _tg_dir)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
# Import directly from file path since TimeGAN's metrics/ dir has no __init__.py
import importlib.util
_disc_spec = importlib.util.spec_from_file_location(
    "discriminative_metrics",
    os.path.join(_tg_dir, "metrics", "discriminative_metrics.py"))
_disc_mod = importlib.util.module_from_spec(_disc_spec)
_disc_spec.loader.exec_module(_disc_mod)
discriminative_score_metrics = _disc_mod.discriminative_score_metrics

_pred_spec = importlib.util.spec_from_file_location(
    "predictive_metrics",
    os.path.join(_tg_dir, "metrics", "predictive_metrics.py"))
_pred_mod = importlib.util.module_from_spec(_pred_spec)
_pred_spec.loader.exec_module(_pred_mod)
predictive_score_metrics = _pred_mod.predictive_score_metrics

def main():
    results = []
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            results = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(results)} results")

    # Find all generated sample files
    sample_files = [f for f in globmod.glob(os.path.join(SAMPLES, "*.npy"))
                    if not os.path.basename(f).startswith("real_")]
    real_files = {os.path.basename(f).replace("real_", "").replace(".npy", ""): f
                  for f in globmod.glob(os.path.join(SAMPLES, "real_*.npy"))}
    print(f"Found {len(sample_files)} generated samples, {len(real_files)} real data files")

    pt_methods = {"fourierflows", "csdi", "gtgan", "diffusionts"}
    updated = 0

    for gf in sorted(sample_files):
        basename = os.path.basename(gf).replace(".npy", "")
        # Parse method_dataset_s{seq_len}_seed{N}
        parts = basename.split("_")
        method = parts[0]
        if method not in pt_methods:
            continue

        # Find dataset part (between method and "s{num}")
        dataset_parts = []
        for p in parts[1:]:
            if p.startswith("s") and p[1:].isdigit():
                break
            dataset_parts.append(p)
        dataset = "_".join(dataset_parts)

        # Get seq_len
        seq_len = None
        for p in parts:
            if p.startswith("s") and p[1:].isdigit():
                seq_len = int(p[1:])
                break

        # Get seed
        seed = None
        for p in parts:
            if p.startswith("seed"):
                seed = int(p.replace("seed", ""))
                break

        if None in (seq_len, seed):
            continue

        # Find matching real data
        real_key = f"{dataset}_s{seq_len}"
        if real_key not in real_files:
            continue

        gen = np.load(gf)
        real = np.load(real_files[real_key])

        n = min(len(gen), len(real))
        if n < 2:
            continue

        # Ensure 3D (n, T, d)
        if gen.ndim == 2:
            gen = gen.reshape(n, -1, 1)
        if real.ndim == 2:
            real = real.reshape(n, -1, 1)

        gen_data = gen[:n]
        real_data = real[:n]

        print(f"  {method:15s} {dataset:12s} s{seq_len} seed={seed}: n={n}", end="")

        try:
            disc = discriminative_score_metrics(real_data, gen_data)
            pred = predictive_score_metrics(real_data, gen_data)
            print(f" A13={disc:.4f} A14={pred:.4f}")
        except Exception as e:
            print(f" FAILED: {str(e)[:50]}")
            disc = float('nan')
            pred = float('nan')

        # Update result entry
        for r in results:
            if (r.get("method") == method and r.get("dataset") == dataset
                    and r.get("seq_len") == seq_len and r.get("seed") == seed):
                r["discriminative_score"] = float(disc) if not np.isnan(disc) else float('nan')
                r["predictive_score"] = float(pred) if not np.isnan(pred) else float('nan')
                r["a13a14_source"] = "tf1_postproc"
                updated += 1
                break

    # Also update remaining NaN entries that had no saved samples
    print(f"\nUpdated {updated} entries with A13/A14")

    # Compute and show mean/std for completed method/dataset
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in results:
        if r.get("status") == "OK":
            m = r["method"]
            if m in pt_methods:
                key = (m, r.get("dataset"), r.get("seq_len"))
                d = r.get("discriminative_score")
                p = r.get("predictive_score")
                if d is not None and not (isinstance(d, float) and np.isnan(d)):
                    by_key[key].append((d, p))

    print("\nA13/A14 Summary:")
    for key in sorted(by_key):
        vals = by_key[key]
        if len(vals) >= 3:
            disc_vals = [v[0] for v in vals]
            pred_vals = [v[1] for v in vals]
            print(f"  {key[0]:15s} {key[1]:12s} s{key[2]}: "
                  f"A13={np.mean(disc_vals):.4f}±{np.std(disc_vals):.4f} "
                  f"A14={np.mean(pred_vals):.4f}±{np.std(pred_vals):.4f} "
                  f"({len(vals)} seeds)")

    # Save
    with open(RESULTS, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved updated results to {RESULTS}")

if __name__ == "__main__":
    main()
