"""Knob experiment: discover boundaries on label-free training-period windows, then score
calibration and test windows under every steering variant.  Labels are only read by knob.report."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .boundary import Dense, discover
from .data import load, rules, sample, windows
from .engine import Engine, load_models

TAUS = [-8, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 8]
CONTROL_TAUS = [-4, 4]
DENSE_TAUS = [-4, -2, -1, 1, 2, 4]
BOUNDARIES = {'predictable': 'nll', 'smooth': 'jump'}
KEEP = ['nll', 'nll_s1', 'entropy_s1', 'resid_max', 'resid_mean', 'resid_price', 'jump', 'range']


def variants():
    v = [('base', None, None, 0.0)]
    for b in BOUNDARIES:
        v += [(f'{b}/knob/normal/{t:+g}', b, ('knob', 'normal'), t) for t in TAUS]
        v += [(f'{b}/knob/{d}/{t:+g}', b, ('knob', d), t) for d in ('tangent', 'random') for t in CONTROL_TAUS]
        v += [(f'{b}/random_dims/normal/{t:+g}', b, ('random_dims', 'normal'), t) for t in CONTROL_TAUS]
        v += [(f'{b}/dense/normal/{t:+g}', b, ('dense', 'normal'), t) for t in DENSE_TAUS]
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--klines', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--lookback', type=int, default=64)
    p.add_argument('--discovery', type=int, default=20000)
    p.add_argument('--calibration', type=int, default=15000)
    p.add_argument('--test', type=int, default=30000)
    p.add_argument('--samples', type=int, default=6)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    k = pd.read_pickle(a.klines) if a.klines.endswith('.pkl') else load(a.klines)
    tok, model = load_models(a.weights)
    eng = Engine(tok, model, samples=a.samples, seed=a.seed)

    disc = out/'discovery.npz'
    if not disc.exists():
        rows, _, _, info = sample(k, 'train', a.lookback, a.discovery, a.seed, live_only=True)
        H, V = [], {n: [] for n in BOUNDARIES.values()}
        for i in range(0, len(rows), a.batch):
            b = eng.base(*windows(k, rows[i:i + a.batch], a.lookback))
            o = eng.variant(b)
            H.append(b.hidden.numpy())
            for n in V:
                V[n].append(o[n].numpy())
        np.savez(disc, rows=rows, hidden=np.concatenate(H), **{n: np.concatenate(v) for n, v in V.items()})
    d = np.load(disc)
    steer, info = {}, {}
    for name, key in BOUNDARIES.items():
        knob, rnd, info[name] = discover(d['hidden'], d[key], seed=a.seed)
        steer[name] = {'knob': knob, 'random_dims': rnd, 'dense': Dense(d['hidden'], d[key], a.seed)}
    (out/'knobs.json').write_text(json.dumps(info, indent=1))
    names = [v[0] for v in variants()]
    config = {**vars(a), 'variants': names, 'taus': TAUS, 'temperature': eng.T, 'top_p': eng.top_p,
              'torch': torch.__version__, 'splits': {}}

    for split, n in (('calibration', a.calibration), ('test', a.test)):
        rows, y, w, config['splits'][split] = sample(k, split, a.lookback, n, a.seed + 1)
        done = out/f'{split}.npz'
        if done.exists():
            continue
        res = {key: np.zeros((len(names), len(rows)), np.float32) for key in KEEP}
        hidden, start = np.zeros((len(rows), model.n_layers + 1, model.d_model), np.float16), time.time()
        for i in range(0, len(rows), a.batch):
            b = eng.base(*windows(k, rows[i:i + a.batch], a.lookback))
            hidden[i:i + a.batch] = b.hidden.numpy()
            for j, (_, bd, how, tau) in enumerate(variants()):
                patch = None if bd is None else steer[bd][how[0]].patch(b.hidden, tau, how[1])
                o = eng.variant(b, patch)
                for key in KEEP:
                    res[key][j, i:i + a.batch] = o[key].numpy()
            if (i//a.batch) % 10 == 0:
                print(split, i, len(rows), f'{time.time() - start:.0f}s', flush=True)
        np.savez(done, rows=rows, y=y, w=w, hidden=hidden, **res, **rules(k, rows))
    (out/'config.json').write_text(json.dumps(config, indent=1))
    print('done', flush=True)


if __name__ == '__main__':
    main()
