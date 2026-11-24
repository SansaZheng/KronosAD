"""Sampled multi-step Kronos forecast paths for the calibration and test samples."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load, sample
from .engine import FEATURES, load_models, normalise, window_ref
from .paths import PathSampler


def future_windows(k, rows, lookback, horizon):
    """Context and the next ``horizon`` bars (the scored hour first) of the same series.

    Bars past the end of a series repeat the last available one; ``valid`` counts real bars.
    ``pump_ahead[:, h]`` marks a labelled pump at hour t+h.
    """
    v = k[FEATURES].to_numpy(np.float64)
    series, pump = k.series.to_numpy(), k.pump.to_numpy()
    end = pd.Series(np.arange(len(k))).groupby(series).max().to_numpy()[series[rows]]
    fut = np.minimum(rows[:, None] + np.arange(horizon)[None], end[:, None])
    ctx = rows[:, None] + np.arange(-lookback, 0)[None]
    valid = np.minimum(end - rows + 1, horizon)
    ahead = (pump[fut] == 1) & (np.arange(horizon)[None] < valid[:, None])
    ts = k.timestamp.dt.tz_localize(None).to_numpy()

    def stamps(t):
        i = pd.DatetimeIndex(t.ravel())
        return np.stack([i.minute, i.hour, i.weekday, i.day, i.month], 1).astype(np.float32).reshape(*t.shape, 5)

    cs = stamps(ts[ctx])
    fs = stamps(ts[rows][:, None] + np.arange(horizon - 1)[None]*np.timedelta64(1, 'h'))
    return v[ctx], v[fut], cs, fs, valid, ahead


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--klines', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--lookback', type=int, default=64)
    p.add_argument('--calibration', type=int, default=15000)
    p.add_argument('--test', type=int, default=30000)
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--horizon', type=int, default=6)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    k = pd.read_pickle(a.klines) if a.klines.endswith('.pkl') else load(a.klines)
    sampler = PathSampler(*load_models(a.weights), samples=a.samples, horizon=a.horizon, seed=a.seed)
    for split, n in (('calibration', a.calibration), ('test', a.test)):
        rows, y, w, _ = sample(k, split, a.lookback, n, a.seed + 1)
        done = Path(a.output)/f'paths_{split}.npz'
        if done.exists():
            continue
        ctx, fut, cs, fs, valid, ahead = future_windows(k, rows, a.lookback, a.horizon)
        ref = window_ref(ctx)
        x, _, mean, scale = normalise(np.concatenate([ctx, fut[:, :1]], 1))
        target = ((fut/ref - mean[:, None])/scale[:, None]).astype(np.float32)
        paths = np.zeros((len(rows), a.samples, a.horizon, 6), np.float16)
        start = time.time()
        for i in range(0, len(rows), a.batch):
            s = slice(i, i + a.batch)
            paths[s] = sampler(x[s, :-1], cs[s], fs[s]).permute(1, 0, 2, 3).numpy()
            if (i//a.batch) % 10 == 0:
                print(split, i, len(rows), f'{time.time() - start:.0f}s', flush=True)
        np.savez(done, rows=rows, y=y, w=w, paths=paths, target=target, last=x[:, -2], mean=mean, scale=scale,
                 ref=ref[:, 0], valid=valid, pump_ahead=ahead)
    print('done', flush=True)


if __name__ == '__main__':
    main()
