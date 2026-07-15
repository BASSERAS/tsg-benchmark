#!/usr/bin/env python3
"""A13/A14 post-processing using saved .npy samples.
Run via: ./miniconda3/envs/tf1_env/bin/python scripts/postproc_a13a14.py
"""
import os, sys, json, warnings, numpy as np
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# CRITICAL: Force TF1 mode on the GLOBAL tensorflow module
# We patch the module that metrics modules will import as 'import tensorflow as tf'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
_tf1 = tensorflow.compat.v1
_tf1.disable_eager_execution()
_tf1.disable_v2_behavior()

# Patch TF1 APIs onto the main tensorflow module (not compat.v1)
# because the TimeGAN metrics modules do "import tensorflow as tf"
_tf = tensorflow  # the module the metrics modules see
_TF1_APIS = ['reset_default_graph','placeholder','Session','set_random_seed',
             'global_variables_initializer','variable_scope','get_variable',
             'get_collection','GraphKeys','AUTO_REUSE','all_variables',
             'trainable_variables','get_default_graph','get_default_session']
for _n in _TF1_APIS:
    if not hasattr(_tf, _n) and hasattr(_tf1, _n):
        setattr(_tf, _n, getattr(_tf1, _n))

# Patch nn submodule
if not hasattr(_tf.nn, 'dynamic_rnn'): _tf.nn.dynamic_rnn = _tf1.nn.dynamic_rnn
if not hasattr(_tf.nn, 'rnn_cell'): _tf.nn.rnn_cell = _tf1.nn.rnn_cell
if not hasattr(_tf.nn, 'moments'): _tf.nn.moments = _tf1.nn.moments

# Copy specific loss functions
if hasattr(_tf1.losses, 'sigmoid_cross_entropy') and not hasattr(_tf.losses, 'sigmoid_cross_entropy'):
    for _fn in ['sigmoid_cross_entropy','mean_squared_error','absolute_difference']:
        if hasattr(_tf1.losses, _fn):
            setattr(_tf.losses, _fn, getattr(_tf1.losses, _fn))

# Replace entire tf.train with compat.v1.train
try:
    import types
    _tf.__dict__['train'] = _tf1.train
except:
    for _opt_name in ['AdamOptimizer','GradientDescentOptimizer','RMSPropOptimizer',
                       'MomentumOptimizer','AdagradOptimizer']:
        if hasattr(_tf1.train, _opt_name):
            setattr(_tf.train, _opt_name, getattr(_tf1.train, _opt_name))

# Patch contrib
if not hasattr(_tf, 'contrib') or not hasattr(_tf.contrib, 'layers') or not hasattr(_tf.contrib.layers, 'fully_connected'):
    if not hasattr(_tf, 'contrib'): _tf.contrib = type('contrib',(),{})()
    if not hasattr(_tf.contrib, 'layers'): _tf.contrib.layers = type('layers',(),{})()
    def _fc(inputs, num_outputs, activation_fn=None, **kw):
        act = activation_fn if activation_fn else (lambda x: x)
        return _tf.keras.layers.Dense(num_outputs, activation=act)(inputs)
    _tf.contrib.layers.fully_connected = _fc

# Now import TimeGAN metrics
sys.path.insert(0, os.path.join(ROOT, 'repos', 'TimeGAN'))
sys.path.insert(0, os.path.join(ROOT, 'repos', 'TimeGAN', 'metrics'))

# Import metrics functions via importlib to avoid module caching issues
import importlib.util
_disc_spec = importlib.util.spec_from_file_location('dm', os.path.join(ROOT, 'repos/TimeGAN/metrics/discriminative_metrics.py'))
_disc_mod = importlib.util.module_from_spec(_disc_spec)
_disc_spec.loader.exec_module(_disc_mod)
discriminative_score_metrics = _disc_mod.discriminative_score_metrics

_pred_spec = importlib.util.spec_from_file_location('pm', os.path.join(ROOT, 'repos/TimeGAN/metrics/predictive_metrics.py'))
_pred_mod = importlib.util.module_from_spec(_pred_spec)
_pred_spec.loader.exec_module(_pred_mod)
predictive_score_metrics = _pred_mod.predictive_score_metrics

SAMPLES = os.path.join(ROOT, 'benchmark_other', 'samples')
RESULTS = os.path.join(ROOT, 'benchmark_other', 'all_results.jsonl')

def main():
    samples_dir = SAMPLES
    if not os.path.exists(samples_dir):
        samples_alt = os.path.join(ROOT, 'results', 'samples')
        if os.path.exists(samples_alt):
            samples_dir = samples_alt
        else:
            print(f'No samples dir at {samples_dir} or {samples_alt}')
            return

    sample_files = sorted([f for f in os.listdir(samples_dir) if f.endswith('.npy') and not f.startswith('real_')])
    real_files = {f.replace('real_',''): f for f in os.listdir(samples_dir) if f.startswith('real_')}
    print(f'Found {len(sample_files)} generated samples, {len(real_files)} real files')

    results = []
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            results = [json.loads(l) for l in f if l.strip()]

    updated = 0
    for sf in sample_files:
        parts = sf.replace('.npy','').split('_')
        method = parts[0]
        dataset_parts, seq_len, seed = [], None, None
        for p in parts[1:]:
            if p.startswith('s') and p[1:].isdigit():
                seq_len = int(p[1:]); break
            dataset_parts.append(p)
        for p in parts:
            if p.startswith('seed'): seed = int(p.replace('seed',''))
        dataset = '_'.join(dataset_parts)
        if None in (seq_len, seed): continue

        gen = np.load(os.path.join(samples_dir, sf))
        real_key = f'{dataset}_s{seq_len}.npy'
        if real_key not in real_files: continue
        real = np.load(os.path.join(samples_dir, real_files[real_key]))

        if gen.ndim == 2: gen = gen.reshape(gen.shape[0], -1, 1) if gen.shape[0] >= gen.shape[1] else gen.reshape(-1, gen.shape[0], 1)
        if real.ndim == 2: real = real.reshape(real.shape[0], -1, 1) if real.shape[0] >= real.shape[1] else real.reshape(-1, real.shape[0], 1)
        if gen.ndim == 2: gen = gen.reshape(-1, gen.shape[0], 1)
        if real.ndim == 2: real = real.reshape(-1, real.shape[0], 1)

        n = min(len(gen), len(real))
        if n < 2: continue
        gen_a, real_a = gen[:n], real[:n]

        print(f'  {method:15s} {dataset:12s} s{seq_len} seed={seed} (n={n})', end='')
        try:
            disc = discriminative_score_metrics(real_a, gen_a)
            pred = predictive_score_metrics(real_a, gen_a)
            print(f' A13={disc:.4f} A14={pred:.4f}')
        except Exception as e:
            print(f' FAILED: {str(e)[:60]}')
            disc = float('nan')
            pred = float('nan')

        for r in results:
            if (r.get('method')==method and r.get('dataset')==dataset and r.get('seq_len')==seq_len and r.get('seed')==seed):
                r['discriminative_score'] = float(disc) if not np.isnan(disc) else None
                r['predictive_score'] = float(pred) if not np.isnan(pred) else None
                r['a13a14_source'] = 'tf1_postproc'
                updated += 1
                break

    with open(RESULTS, 'w') as f:
        for r in results: f.write(json.dumps(r)+'\n')
    print(f'\n✅ Updated {updated} entries')

if __name__ == '__main__':
    main()
