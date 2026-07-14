#!/usr/bin/env python3
"""
TF1 A13/A14 post-processing for PT methods.
Compute Discriminative and Predictive scores from TF1 env.

Usage: ./miniconda3/envs/tf1_env/bin/python scripts/postproc_a13a14.py
"""
import json, os, sys, numpy as np, warnings
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from metrics import compute_discriminative_score, compute_predictive_score

RESULTS = os.path.join(ROOT, "results")
INPUT = os.path.join(RESULTS, "all_results.jsonl")

def load_results():
    results = []
    if os.path.exists(INPUT):
        with open(INPUT) as f:
            for line in f:
                if line.strip(): results.append(json.loads(line))
    return results

def update_entry(entry, disc, pred):
    entry['discriminative_score'] = disc
    entry['predictive_score'] = pred
    entry['a13a14_postprocessed'] = True

def main():
    results = load_results()
    print(f"Loaded {len(results)} results")

    # Find PT methods needing A13/A14
    pt_methods = ['fourierflows', 'csdi', 'gtgan', 'diffusionts']
    need_postproc = []
    for r in results:
        if r['method'] in pt_methods and r.get('status') == 'OK':
            if r.get('discriminative_score') is None or \
               (isinstance(r.get('discriminative_score'), float) and np.isnan(r.get('discriminative_score'))):
                need_postproc.append(r)

    print(f"{len(need_postproc)} experiments need A13/A14 post-processing")

    if not need_postproc:
        print("Nothing to do.")
        return

    for entry in need_postproc:
        method = entry['method']
        dataset = entry['dataset']
        seq_len = entry['seq_len']
        seed = entry['seed']

        print(f"  Processing {method} {dataset} s{seq_len} seed={seed}...")

        # Load the generated samples and real data
        try:
            from data.datasets import load_dataset as ld
            data_dict = ld(dataset, seq_len, seed)
            # We need the raw test data
            test_data = data_dict['test_raw']

            # We need generated samples - load from cache or regenerate
            # For now, regenerate with a quick sample
            try:
                import importlib
                mod = importlib.import_module(f'adapters.{method}_adapter')
                Adapter = getattr(mod, f'{method}Adapter')
                adapter = Adapter(seq_len=seq_len, n_features=data_dict['n_features'], seed=seed, device='cpu')
                adapter.training_steps = 1  # Won't actually train

                # Can't re-sample without retraining, skip for now
                print(f"    SKIP: need saved samples. Method: {method}")
                continue
            except:
                print(f"    SKIP: couldn't load adapter")
                continue
        except Exception as e:
            print(f"    Error: {e}")
            continue

if __name__ == '__main__':
    main()
