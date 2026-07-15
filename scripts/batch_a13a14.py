#!/usr/bin/env python3
"""Batch A13/A14 post-processing for PT methods using saved .npy samples.
Run: ./miniconda3/envs/tf1_env/bin/python scripts/batch_a13a14.py
"""
import os, sys, time, json, numpy as np, warnings
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# Add TimeGAN paths FIRST
sys.path.insert(0, os.path.join(ROOT, 'repos', 'TimeGAN'))
sys.path.insert(0, os.path.join(ROOT, 'repos', 'TimeGAN', 'metrics'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
_tf1 = tensorflow.compat.v1
_tf1.disable_eager_execution()
_tf1.disable_v2_behavior()
_tf = tensorflow
for _n in ['reset_default_graph','placeholder','Session','set_random_seed','global_variables_initializer',
           'variable_scope','get_variable','get_collection','GraphKeys','AUTO_REUSE','all_variables',
           'trainable_variables','get_default_graph','get_default_session']:
    if not hasattr(_tf, _n) and hasattr(_tf1, _n):
        setattr(_tf, _n, getattr(_tf1, _n))
if not hasattr(_tf.nn, 'dynamic_rnn'):
    _tf.nn.dynamic_rnn = _tf1.nn.dynamic_rnn
if not hasattr(_tf.nn, 'rnn_cell'):
    _tf.nn.rnn_cell = _tf1.nn.rnn_cell
if not hasattr(_tf.nn, 'moments'):
    _tf.nn.moments = _tf1.nn.moments
_tf.__dict__['train'] = _tf1.train
_tf.__dict__['losses'] = _tf1.losses
if not hasattr(_tf, 'contrib') or not hasattr(_tf.contrib, 'layers') or not hasattr(_tf.contrib.layers, 'fully_connected'):
    if not hasattr(_tf, 'contrib'):
        _tf.contrib = type('contrib', (), {})()
    if not hasattr(_tf.contrib, 'layers'):
        _tf.contrib.layers = type('layers', (), {})()
    def _fc(inputs, num_outputs, activation_fn=None, **kw):
        act = activation_fn if activation_fn else (lambda x: x)
        return _tf.keras.layers.Dense(num_outputs, activation=act)(inputs)
    _tf.contrib.layers.fully_connected = _fc

# Import metrics modules via sys.path
from discriminative_metrics import discriminative_score_metrics as disc_fn
from predictive_metrics import predictive_score_metrics as pred_fn
print("Imports OK")

# Find samples
SAMPLES = os.path.join(ROOT, 'benchmark_other', 'samples')
if not os.path.exists(SAMPLES):
    SAMPLES = os.path.join(ROOT, 'results', 'samples')

# Load results
RESULTS = os.path.join(ROOT, 'benchmark_other', 'all_results.jsonl')
results = []
if os.path.exists(RESULTS):
    with open(RESULTS) as f:
        results = [json.loads(l) for l in f if l.strip()]

sample_files = sorted([f for f in os.listdir(SAMPLES) if f.endswith('.npy') and not f.startswith('real_')])
real_files = {f.replace('real_', ''): f for f in os.listdir(SAMPLES) if f.startswith('real_')}
print(f"{len(sample_files)} samples, {len(real_files)} real data files, {len(results)} results")

pt_methods = {'fourierflows', 'csdi', 'diffusionts', 'gtgan', 'rgan', 'timevae'}
updated = 0
t_start = time.time()

for sf in sample_files:
    parts = sf.replace('.npy', '').split('_')
    method = parts[0]
    if method not in pt_methods:
        continue
    dataset_parts = []
    seq_len = None
    for p in parts[1:]:
        if p.startswith('s') and p[1:].isdigit():
            seq_len = int(p[1:])
            break
        dataset_parts.append(p)
    seed = None
    for p in parts:
        if p.startswith('seed'):
            seed = int(p.replace('seed', ''))
    if seq_len is None or seed is None:
        continue
    dataset = '_'.join(dataset_parts)

    gen = np.load(os.path.join(SAMPLES, sf))
    rk = dataset + '_s' + str(seq_len) + '.npy'
    if rk not in real_files:
        continue
    real = np.load(os.path.join(SAMPLES, real_files[rk]))
    n = min(len(gen), len(real))
    if n < 2:
        continue
    g, r = gen[:n], real[:n]
    if g.ndim == 2:
        g = g.reshape(n, -1, 1)
    if r.ndim == 2:
        r = r.reshape(n, -1, 1)

    t0 = time.time()
    try:
        d = disc_fn(r, g)
        p = pred_fn(r, g)
        elapsed = time.time() - t0
        print(f"{method:12s} {dataset:12s} s{seq_len} seed={seed}: A13={d:.4f} A14={p:.4f} ({elapsed:.0f}s)")
    except Exception as e:
        print(f"{method:12s} {dataset:12s} s{seq_len} seed={seed}: FAILED {str(e)[:80]}")
        d = float('nan')
        p = float('nan')

    for r in results:
        if (r.get('method') == method and r.get('dataset') == dataset
                and r.get('seq_len') == seq_len and r.get('seed') == seed):
            r['discriminative_score'] = d if not (isinstance(d, float) and np.isnan(d)) else None
            r['predictive_score'] = p if not (isinstance(p, float) and np.isnan(p)) else None
            r['a13a14_source'] = 'batch_tf1_postproc'
            updated += 1
            break

total_time = time.time() - t_start
with open(RESULTS, 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')
print(f"\nDone. Updated {updated}/{len(sample_files)} entries in {total_time:.0f}s")
