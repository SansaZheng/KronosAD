"""Does the model help once its predictive distribution or a multi-step path is used?

D  one-step, distribution-aware: error scaled by the spread of the sampled forecasts, or the
   distance outside the sampled 10-90% interval.
M  multi-step: largest scaled deviation of the realised path from the mean sampled path over the
   scored hour and the following hours.  This is delayed detection: horizon h needs h-1 later bars.
Each is compared with the same statistic computed from a previous-bar forecast and with the
model-free high/low range over the same hours.  Choices are made by calibration-period AP.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load
from .engine import EPS, FEATURES
from .improve import WEIGHTS, amount_score, market_index, parts, series_scale, unit
from .report import bootstrap, metrics, subtle_mask

FLOORS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, np.inf]
HORIZONS = [1, 2, 3, 6]


def build(d, k):
    paths = d['paths'].astype(np.float32)
    scale = d['scale'][:, None, :4]
    raw_last = k[FEATURES].to_numpy(np.float64)[d['rows'] - 1]
    last = ((raw_last/d['ref'] - d['mean'])/d['scale'])[:, None, :4]
    t = d['target'][:, :, :4]
    mean = paths.mean(1)[:, :, :4]
    q10, q90 = np.quantile(paths[..., :4], [0.1, 0.9], axis=1)
    rel = d['target'].astype(np.float64)*d['scale'][:, None] + d['mean'][:, None]
    return {'err/kronos': np.abs(t - mean)*scale, 'err/persistence': np.abs(t - last)*scale,
            'spread': paths.std(1)[:, :, :4]*scale, 'exceed': np.maximum(np.maximum(t - q90, q10 - t), 0)*scale,
            'width': (q90 - q10)*scale, 'sigma': np.broadcast_to(d['scale'][:, None, :4] - EPS, t.shape),
            'range': np.log(np.maximum(rel[:, :, 1], 1e-300)/np.maximum(rel[:, :, 2], 1e-300))}


def score(b, kind, c, h):
    num, den = {'kronos/window': ('err/kronos', 'sigma'), 'kronos/spread': ('err/kronos', 'spread'),
                'kronos/interval': ('exceed', 'width'), 'persistence/window': ('err/persistence', 'sigma')}[kind]
    e = b[num][:, :h]
    z = e if np.isinf(c) else e/(b[den][:, :h] + c)
    return z.reshape(len(z), -1).max(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--klines', required=True)
    p.add_argument('--runs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--lookback', type=int, default=64)
    p.add_argument('--bootstrap', type=int, default=1000)
    a = p.parse_args()
    k = pd.read_pickle(a.klines) if a.klines.endswith('.pkl') else load(a.klines)
    sscale, market = series_scale(k), market_index(k)
    B, M, A = {}, {}, {}
    for split in ('calibration', 'test'):
        d = np.load(Path(a.runs)/f'paths_{split}.npz')
        full = np.load(Path(a.runs)/f'{split}.npz')
        assert (d['rows'] == full['rows']).all()
        B[split] = build(d, k)
        M[split] = {'y': d['y'], 'w': d['w'], 'subtle': subtle_mask(full), 'ahead': d['pump_ahead']}
        A[split] = parts(np.load(Path(a.runs)/f'diagnose_{split}.npz'), k, a.lookback, sscale, market)

    def keep(split, h):
        m = M[split]
        return (m['y'] == 1) | ~m['ahead'][:, 1:h].any(1)

    def evaluate(split, s, h):
        m, kp = M[split], keep(split, h)
        return metrics(s[kp], m['y'][kp], m['w'][kp], m['subtle'][kp])

    def ci(x, y_, h):
        m, kp = M['test'], keep('test', h)
        return bootstrap(x[kp], y_[kp], m['y'][kp], m['w'][kp], a.bootstrap, 7)

    res = {'horizons': {}, 'samples': int(np.load(Path(a.runs)/'paths_test.npz')['paths'].shape[1])}
    final = {}
    for h in HORIZONS:
        out = {'negatives_dropped_test': int((~keep('test', h)).sum()),
               'rule_range': evaluate('test', B['test']['range'][:, :h].max(1), h)}
        chosen = {}
        for kind in ('kronos/window', 'kronos/spread', 'kronos/interval', 'persistence/window'):
            c = max(FLOORS, key=lambda v: evaluate('calibration', score(B['calibration'], kind, v, h), h)['ap'])
            chosen[kind] = c
            out[kind] = {'floor': str(c), 'calibration': evaluate('calibration', score(B['calibration'], kind, c, h), h),
                         'test': evaluate('test', score(B['test'], kind, c, h), h)}
        s = {kind: score(B['test'], kind, c, h) for kind, c in chosen.items()}
        out['ci95'] = {'spread_minus_window': ci(s['kronos/spread'], s['kronos/window'], h),
                       'interval_minus_window': ci(s['kronos/interval'], s['kronos/window'], h),
                       'kronos_minus_persistence': ci(s['kronos/window'], s['persistence/window'], h),
                       'kronos_minus_rule': ci(s['kronos/window'], B['test']['range'][:, :h].max(1), h)}
        # add the one-step traded-amount score, as in knob.improve
        yc, wc = M['calibration']['y'], M['calibration']['w']
        for name in ('kronos', 'persistence'):
            best_kind = max([kd for kd in chosen if kd.startswith(name)], key=lambda kd: out[kd]['calibration']['ap'])
            pr = {sp: score(B[sp], best_kind, chosen[best_kind], h) for sp in B}
            ca = max([0.01, 0.1, 1.0, 10.0, np.inf], key=lambda v: evaluate('calibration', amount_score(A['calibration'], name, v), 1)['ap'])
            am = {sp: amount_score(A[sp], name, ca) for sp in B}
            up = {sp: unit(pr[sp], pr['calibration'], yc, wc) for sp in B}
            ua = {sp: unit(am[sp], am['calibration'], yc, wc) for sp in B}
            combos = {'price_only': 1.0, **{f'sum_{g:g}': g for g in WEIGHTS}}
            comb = lambda sp, g: g*up[sp] + (1 - g)*ua[sp]
            g = max(combos, key=lambda n: evaluate('calibration', comb('calibration', combos[n]), h)['ap'])
            final[(name, h)] = comb('test', combos[g])
            out[f'{name}/with_amount'] = {'price_score': best_kind, 'combination': g,
                                          'calibration': evaluate('calibration', comb('calibration', combos[g]), h),
                                          'test': evaluate('test', final[(name, h)], h)}
        out['ci95']['with_amount_kronos_minus_persistence'] = ci(final[('kronos', h)], final[('persistence', h)], h)
        out['ci95']['with_amount_kronos_minus_rule'] = ci(final[('kronos', h)], B['test']['range'][:, :h].max(1), h)
        res['horizons'][str(h)] = out
    Path(a.output).mkdir(parents=True, exist_ok=True)
    (Path(a.output)/'distribution_and_horizon.json').write_text(json.dumps(res, indent=1))
    pc = lambda v: f'{100*v:6.2f}'
    for h, out in res['horizons'].items():
        print(f"== horizon {h} (dropped {out['negatives_dropped_test']})  rule: AP {pc(out['rule_range']['ap'])} R@1% {pc(out['rule_range']['recall_fpr1'])} subtle miss {pc(out['rule_range']['subtle_miss_fpr1'])}")
        for kind in ('kronos/window', 'kronos/spread', 'kronos/interval', 'persistence/window', 'kronos/with_amount', 'persistence/with_amount'):
            v = out[kind]
            t = v['test']
            print(f"  {kind:24s} {str(v.get('floor', v.get('combination'))):9s} cal AP {pc(v['calibration']['ap'])} | test AP {pc(t['ap'])} R@1% {pc(t['recall_fpr1'])} R@0.1% {pc(t['recall_fpr01'])} subtle miss {pc(t['subtle_miss_fpr1'])}")
        print('  ci', {n: [round(100*x, 1) for x in v] for n, v in out['ci95'].items()})


if __name__ == '__main__':
    main()
