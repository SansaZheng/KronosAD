"""What limits the residual score: forecast quality or the normalisation scale?

Re-expresses the unsteered residual with a relative scale floor c (error / (window sigma + c), in
units of the window mean price) and compares the Kronos forecast with a no-model persistence
forecast (previous bar).  c is selected on the calibration period; labels are read here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load, windows
from .engine import EPS, normalise
from .report import bootstrap, metrics, subtle_mask

FLOORS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]


def scores(d, k, lookback):
    x, target, mean, scale = normalise(windows(k, d['rows'], lookback)[0])
    assert np.allclose(target, d['target'], atol=1e-4)
    sigma = scale - EPS
    err = {'kronos': np.abs(d['target'] - d['pred'])*scale, 'persistence': np.abs(target - x[:, -2])*scale}
    out = {}
    for name, e in err.items():
        for c in FLOORS:
            z = e/(sigma + max(c, EPS))
            out[f'{name}/price/{c:g}'] = z[:, :4].max(1)
            out[f'{name}/all/{c:g}'] = z.max(1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--klines', required=True)
    p.add_argument('--runs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--lookback', type=int, default=64)
    a = p.parse_args()
    k = pd.read_pickle(a.klines) if a.klines.endswith('.pkl') else load(a.klines)
    res, sc, meta = {}, {}, {}
    for split in ('calibration', 'test'):
        d = np.load(Path(a.runs)/f'diagnose_{split}.npz')
        full = np.load(Path(a.runs)/f'{split}.npz')
        assert (d['rows'] == full['rows']).all()
        meta[split] = (d['y'], d['w'], subtle_mask(full))
        sc[split] = scores(d, k, a.lookback)
        res[split] = {n: metrics(s, *meta[split]) for n, s in sc[split].items()}
    sel = {}
    for name in ('kronos', 'persistence'):
        for feat in ('price', 'all'):
            keys = [f'{name}/{feat}/{c:g}' for c in FLOORS]
            best = max(keys, key=lambda n: res['calibration'][n]['ap'])
            y, w, _ = meta['test']
            sel[f'{name}/{feat}'] = {'selected': best, 'floor': float(best.rsplit('/', 1)[1]), 'test': res['test'][best],
                                     'test_no_floor': res['test'][keys[0]],
                                     'test_ap_gain_ci95': bootstrap(sc['test'][best], sc['test'][keys[0]], y, w, 1000, 7)}
    y, w, _ = meta['test']
    for feat in ('price', 'all'):
        a_, b_ = sel[f'kronos/{feat}']['selected'], sel[f'persistence/{feat}']['selected']
        sel[f'kronos_minus_persistence/{feat}'] = {'test_ap_gain_ci95': bootstrap(sc['test'][a_], sc['test'][b_], y, w, 1000, 7)}
    Path(a.output).mkdir(parents=True, exist_ok=True)
    (Path(a.output)/'scale_floor.json').write_text(json.dumps({'floors': FLOORS, 'metrics': res, 'selection': sel}, indent=1))
    for n, v in sel.items():
        print(n, json.dumps(v)[:330])
    for n in res['test']:
        print(f"{n:28s} cal AP {100*res['calibration'][n]['ap']:6.2f}  test AP {100*res['test'][n]['ap']:6.2f} R@1% {100*res['test'][n]['recall_fpr1']:5.1f} R@0.1% {100*res['test'][n]['recall_fpr01']:5.1f} subtle miss {100*res['test'][n]['subtle_miss_fpr1']:5.1f}")


if __name__ == '__main__':
    main()
