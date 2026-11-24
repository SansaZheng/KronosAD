"""Unsteered forecasts with per-feature detail, to analyse what limits the residual score.

Stores, per sampled window, the normalised forecast and target and the normalisation scale, so
residuals can be re-expressed with a different scale floor without running the model again.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load, sample, windows
from .engine import Engine, load_models


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--klines', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--lookback', type=int, default=64)
    p.add_argument('--calibration', type=int, default=15000)
    p.add_argument('--test', type=int, default=30000)
    p.add_argument('--samples', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    k = pd.read_pickle(a.klines) if a.klines.endswith('.pkl') else load(a.klines)
    eng = Engine(*load_models(a.weights), samples=a.samples, seed=a.seed)
    for split, n in (('calibration', a.calibration), ('test', a.test)):
        rows, y, w, _ = sample(k, split, a.lookback, n, a.seed + 1)
        keep = {'pred': [], 'target': [], 'scale': [], 'mean': []}
        for i in range(0, len(rows), 256):
            b = eng.base(*windows(k, rows[i:i + 256], a.lookback))
            o = eng.variant(b)
            for key, v in (('pred', o['pred']), ('target', b.target), ('scale', b.scale), ('mean', b.mean)):
                keep[key].append(v.numpy())
        np.savez(Path(a.output)/f'diagnose_{split}.npz', rows=rows, y=y, w=w, **{key: np.concatenate(v) for key, v in keep.items()})
        print(split, 'done', flush=True)


if __name__ == '__main__':
    main()
