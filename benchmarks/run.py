"""Run label-blind feature adaptation + Isolation Forest with frozen POT.

Training and calibration functions never receive labels. Evaluate all held-out
rows at their natural class ratio; unknown Elliptic targets are masked only by
the final metric function. Local cache/checkpoints/scores are ignored by Git.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from sklearn.ensemble import IsolationForest
from threadpoolctl import threadpool_limits

from .data import DATASETS, load
from .evaluation import calibrate, evaluate
from .models import AdapterDetector, Preprocessor


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(2**20):
            h.update(b)
    return h.hexdigest()


def sample(x, cap, seed):
    # Uniform without replacement, no target-dependent sampling/filtering.
    indices = np.sort(np.random.default_rng(seed).choice(len(x), min(cap, len(x)), replace=False))
    return np.asarray(x[indices]), indices


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(args, name, method, seed):
    out = args.output / name / f'{method}_seed{seed}'
    config = {'dataset': name, 'method': method, 'seed': seed, 'device': args.device,
              'steps': args.steps, 'train_cap': args.train_cap,
              'calibration_cap': args.calibration_cap, 'risk': args.risk,
              'initial_quantile': args.initial_quantile, 'batch_size': args.batch_size,
              'weights_sha256': args.weights_hash,
              'data_manifest_sha256': digest(args.cache / name / 'manifest.json')}
    if (out / 'result.json').exists():
        previous = json.loads((out / 'result.json').read_text())
        if previous.get('run_config') != config:
            raise ValueError('Existing run has a different configuration; choose a new output directory')
        return previous
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    xtrain, train_indices = sample(load(args.cache, name, 'train'), args.train_cap, seed)
    prep = Preprocessor().fit(xtrain)
    train = prep.transform(xtrain); del xtrain
    prep.save(out / 'preprocessing.npz')
    np.save(out / 'train_indices.npy', train_indices)
    print(f'START {name} {method} seed={seed} train={len(train)} dims={train.shape[1]}', flush=True)
    if method == 'isolation_forest':
        detector = IsolationForest(n_estimators=100, max_samples=256, contamination='auto',
                                   random_state=seed, n_jobs=2).fit(np.clip(train, -8, 8))
        # Nonnegative path-length anomaly score. Higher always means anomalous.
        score = lambda z: -detector.score_samples(np.clip(z, -8, 8))
        training = {'trees': 100, 'max_samples': 256}
        import joblib
        joblib.dump(detector, out / 'model.joblib')
    else:
        detector = AdapterDetector(args.weights, seed, method == 'random_adapter',
                                   args.device, args.steps).fit(train)
        score = detector.score
        detector.save(out / 'adapter.pt')
        training = {'steps': args.steps, 'losses': detector.losses,
                    'trainable_parameters': detector.trainable_parameters,
                    'frozen_full_model_parameters': detector.frozen_parameters}
    del train; gc.collect()
    train_seconds = time.monotonic()-start
    calibration, cal_indices = sample(load(args.cache, name, 'calibration'), args.calibration_cap, seed+100)
    cal_scores = np.concatenate([score(prep.transform(calibration[i:i+args.batch_size]))
                                 for i in range(0, len(calibration), args.batch_size)])
    del calibration
    np.save(out / 'calibration_indices.npy', cal_indices)
    np.save(out / 'calibration_scores.npy', cal_scores)
    threshold = calibrate(cal_scores, risk=args.risk, quantile=args.initial_quantile)
    # Threshold is fitted and persisted BEFORE opening any target file.
    write_json(out / 'threshold.json', threshold.to_dict())
    xtest = load(args.cache, name, 'test')
    scores = np.lib.format.open_memmap(out / 'test_scores.npy', mode='w+', dtype=np.float64,
                                      shape=(len(xtest),))
    for i in range(0, len(xtest), args.batch_size):
        scores[i:i+args.batch_size] = score(prep.transform(xtest[i:i+args.batch_size]))
    scores.flush()
    labels = load(args.cache, name, 'test', targets=True)
    metrics = evaluate(labels, scores, threshold.threshold)
    manifest = json.loads((args.cache / name / 'manifest.json').read_text())
    result = {'dataset': name, 'method': method, 'seed': seed, 'status': 'complete', 'run_config': config,
              'train_rows_used': len(train_indices), 'calibration_rows_used': len(cal_indices),
              'split_counts': manifest['splits'], 'feature_count_raw': len(manifest['features']),
              'protocol': manifest['protocol'], 'threshold': threshold.to_dict(),
              'training': training, 'train_seconds': train_seconds,
              'total_seconds': time.monotonic()-start, 'metrics': metrics,
              'test_scores_sha256': digest(out / 'test_scores.npy'),
              'data_manifest_sha256': digest(args.cache / name / 'manifest.json')}
    write_json(out / 'result.json', result)
    print(f'DONE {name} {method} seed={seed} F1={metrics["f1"]:.6f} AP={metrics["average_precision"]} elapsed={result["total_seconds"]:.1f}s', flush=True)
    del detector, scores, xtest, labels; gc.collect()
    if args.device == 'mps':
        import torch
        torch.mps.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=['sapienza_15S', 'tstr', 'sd_stock', 'sd_crypto', 'ulb', 'elliptic', 'paysim', 'tabformer', 'ibm_aml', 'ieee_cis'])
    p.add_argument('--methods', nargs='+', choices=['isolation_forest', 'kronos_adapter', 'random_adapter'], default=['isolation_forest', 'kronos_adapter', 'random_adapter'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--weights', type=Path, default=Path('weights/Kronos-mini'))
    p.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--train-cap', type=int, default=50000)
    p.add_argument('--calibration-cap', type=int, default=100000)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--risk', type=float, default=.01)
    p.add_argument('--initial-quantile', type=float, default=.85)
    args = p.parse_args()
    args.weights_hash = digest(args.weights / 'model.safetensors')
    args.output.mkdir(parents=True, exist_ok=True); args.output.chmod(0o700)
    if args.steps < 1 or args.train_cap < 1 or args.calibration_cap < 1 or args.batch_size < 1:
        p.error('All caps, batch size and steps must be positive')
    import torch
    torch.set_num_threads(2)
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
                   'packages': {v: importlib.metadata.version(v) for v in ('numpy', 'pandas', 'scipy', 'torch', 'scikit-learn', 'safetensors')},
                   'weights_sha256': args.weights_hash,
                   'argv': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   'code_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   'labels_used_for_training_or_thresholds': False}
    write_json(args.output / 'environment.json', environment)
    with threadpool_limits(limits=2):
        for name in args.datasets:
            for method in args.methods:
                for seed in args.seeds:
                    run(args, name, method, seed)


if __name__ == '__main__':
    main()
