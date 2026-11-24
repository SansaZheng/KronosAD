"""Score-side improvements that need no new model inference.

Stage 1  scale: replace "error / window sigma" by a floored window scale or a per-series scale
         estimated on the training period.
Stage 6  market: subtract a same-hour, model-free market activity index from the price error.
Stage 3  volume: score traded amount separately (one-sided, in BTC) and combine with price.

Every choice is made by calibration-period AP; the test period is only reported.  The same
pipeline is run with the previous bar as the forecast, to show what the model contributes.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load, windows
from .engine import EPS, FEATURES
from .report import bootstrap, metrics, subtle_mask

FLOORS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, np.inf]
AMOUNT_FLOORS = [0.01, 0.1, 1.0, 10.0, np.inf]
LAMBDAS = [0.25, 0.5, 1.0, 2.0, 4.0]
WEIGHTS = [0.25, 0.5, 0.75]


def series_scale(k):
    """Training-period std of the hourly relative close change, per series (label-free, past only)."""
    c = k.close.to_numpy()
    prev = np.concatenate([[c[0]], c[:-1]])
    r = np.where((k.pos.to_numpy() > 0) & (prev > 0), c/np.maximum(prev, 1e-300) - 1, 0.0)
    train = (k.split == 'train').to_numpy()
    s = pd.Series(r[train]).groupby(k.series.to_numpy()[train]).std()
    return s.reindex(range(k.series.max() + 1)).fillna(0.0).to_numpy()


def market_index(k):
    """Same-hour activity of all series: mean and 90th percentile of log(high/low), and the mean
    over series that traded in that hour."""
    rng = np.log(np.maximum(k.high.to_numpy(), 1e-300)/np.maximum(k.low.to_numpy(), 1e-300))
    g = pd.DataFrame({'t': k.timestamp.to_numpy(), 'r': rng, 'a': k.volume.to_numpy() > 0})
    out = g.groupby('t').r.agg(mean='mean', q90=lambda v: v.quantile(0.9))
    out['active'] = g[g.a].groupby('t').r.mean().reindex(out.index).fillna(0.0)
    return out


def parts(d, k, lookback, sscale, market):
    """Per-window error terms in comparable units, for the Kronos and previous-bar forecasts."""
    rows = d['rows']
    bars = windows(k, rows, lookback)[0]
    ctx = bars[:, :-1]
    amount_ref = np.where(ctx[:, :, 5].mean(1) > 0, ctx[:, :, 5].mean(1), 1.0)
    sigma = d['scale'] - EPS                                   # relative to the window mean close
    target = d['target']
    ref = np.ones((len(rows), 6))
    ref[:, :4] = np.where(ctx[:, :, 3].mean(1) > 0, ctx[:, :, 3].mean(1), 1.0)[:, None]
    for j in (4, 5):
        m = ctx[:, :, j].mean(1)
        ref[:, j] = np.where(m > 0, m, 1.0)
    last = (bars[:, -2]/ref - d['mean'])/d['scale']
    out = {'sigma_price': sigma[:, :4], 'series': sscale[k.series.to_numpy()[rows]],
           'amount_true': bars[:, -1, 5], 'sigma_amount': sigma[:, 5]*amount_ref,
           'market': market.reindex(k.timestamp.to_numpy()[rows]).to_numpy()}
    for name, pred in (('kronos', d['pred']), ('persistence', last)):
        out[f'{name}/err_price'] = np.abs(target - pred)[:, :4]*d['scale'][:, :4]
        amount_pred = np.maximum(pred[:, 5]*d['scale'][:, 5] + d['mean'][:, 5], 0)*amount_ref
        out[f'{name}/err_amount'] = np.maximum(out['amount_true'] - amount_pred, 0)
    return out


def price_score(p, name, kind, c, lam=0.0, index=0):
    e = np.maximum(p[f'{name}/err_price'] - lam*p['market'][:, [index]], 0)
    if np.isinf(c):
        return e.max(1)
    if kind == 'window':
        return (e/(p['sigma_price'] + c)).max(1)
    return e.max(1)/(p['series'] + c)


def amount_score(p, name, c):
    e = p[f'{name}/err_amount']
    return e if np.isinf(c) else e/(p['sigma_amount'] + c)


def unit(score, ref_score, y, w):
    """log1p(score / weighted 99th percentile of calibration non-pump hours)."""
    neg = ref_score[y == 0]
    order = np.argsort(neg)
    cum = np.cumsum(w[y == 0][order])/w[y == 0].sum()
    q = neg[order][min(np.searchsorted(cum, 0.99), len(neg) - 1)]
    return np.log1p(score/max(q, 1e-12))


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
    P, M = {}, {}
    for split in ('calibration', 'test'):
        d = np.load(Path(a.runs)/f'diagnose_{split}.npz')
        full = np.load(Path(a.runs)/f'{split}.npz')
        P[split] = parts(d, k, a.lookback, sscale, market)
        M[split] = (d['y'], d['w'], subtle_mask(full), full['rule_range'])
    yc, wc, sc, _ = M['calibration']
    yt, wt, st, rule = M['test']
    ap = lambda s: metrics(s, yc, wc, sc)['ap']
    res = {'grid': {'floors': [str(c) for c in FLOORS], 'amount_floors': [str(c) for c in AMOUNT_FLOORS], 'lambdas': LAMBDAS,
                    'market_indices': list(market.columns), 'weights': WEIGHTS},
           'rule_range_test': metrics(rule, yt, wt, st), 'pipelines': {}}
    for name in ('kronos', 'persistence'):
        stages, scores = {}, {}

        def record(stage, choice, fn, tried):
            cal, test = fn('calibration'), fn('test')
            scores[stage] = test
            stages[stage] = {'choice': choice, 'configurations_tried': tried, 'calibration': metrics(cal, yc, wc, sc),
                             'test': metrics(test, yt, wt, st)}

        base = lambda s: (P[s][f'{name}/err_price']/(P[s]['sigma_price'] + EPS)).max(1)
        record('0_original', {}, base, 1)
        cands = [(kind, c) for kind in ('window', 'series') for c in FLOORS if not (kind == 'series' and np.isinf(c))]
        kind, c = max(cands, key=lambda kc: ap(price_score(P['calibration'], name, *kc)))
        record('1_scale', {'scale': kind, 'floor': str(c)}, lambda s: price_score(P[s], name, kind, c), len(cands))
        mc = [(lam, i) for lam in LAMBDAS for i in range(market.shape[1])] + [(0.0, 0)]
        lam, idx = max(mc, key=lambda li: ap(price_score(P['calibration'], name, kind, c, *li)))
        price = lambda s: price_score(P[s], name, kind, c, lam, idx)
        record('2_scale_market', {'lambda': lam, 'index': market.columns[idx]}, price, len(mc))
        ca = max(AMOUNT_FLOORS, key=lambda v: ap(amount_score(P['calibration'], name, v)))
        amount = lambda s: amount_score(P[s], name, ca)
        record('amount_only', {'floor_btc': str(ca)}, amount, len(AMOUNT_FLOORS))
        up = lambda s: unit(price(s), price('calibration'), yc, wc)
        ua = lambda s: unit(amount(s), amount('calibration'), yc, wc)
        combos = {'price_only': lambda s: up(s), 'min': lambda s: np.minimum(up(s), ua(s)), 'max': lambda s: np.maximum(up(s), ua(s))}
        for wgt in WEIGHTS:
            combos[f'sum_{wgt:g}'] = lambda s, wgt=wgt: wgt*up(s) + (1 - wgt)*ua(s)
        best = max(combos, key=lambda n: ap(combos[n]('calibration')))
        record('3_scale_market_volume', {'combination': best}, combos[best], len(combos))
        for stage in stages:
            if stage != '0_original':
                stages[stage]['test_ap_gain_vs_original_ci95'] = bootstrap(scores[stage], scores['0_original'], yt, wt, a.bootstrap, 7)
            if stage == '3_scale_market_volume':
                stages[stage]['test_ap_gain_vs_scale_only_ci95'] = bootstrap(scores[stage], scores['1_scale'], yt, wt, a.bootstrap, 7)
            stages[stage]['test_ap_gain_vs_rule_range_ci95'] = bootstrap(scores[stage], rule, yt, wt, a.bootstrap, 7)
        res['pipelines'][name] = stages
        res.setdefault('_final', {})[name] = scores['3_scale_market_volume']
    f = res.pop('_final')
    res['kronos_minus_persistence_final_ci95'] = bootstrap(f['kronos'], f['persistence'], yt, wt, a.bootstrap, 7)
    Path(a.output).mkdir(parents=True, exist_ok=True)
    (Path(a.output)/'score_improvements.json').write_text(json.dumps(res, indent=1))
    pc = lambda v: f'{100*v:6.2f}'
    print('rule_range test', {m: pc(v) for m, v in res['rule_range_test'].items()})
    for name, stages in res['pipelines'].items():
        for stage, v in stages.items():
            t = v['test']
            print(f"{name:11s} {stage:22s} {json.dumps(v['choice']):48s} cal AP {pc(v['calibration']['ap'])} | test AP {pc(t['ap'])} AUROC {pc(t['auroc'])} R@1% {pc(t['recall_fpr1'])} R@0.1% {pc(t['recall_fpr01'])} subtle miss {pc(t['subtle_miss_fpr1'])} | vs orig {[round(100*x, 2) for x in v.get('test_ap_gain_vs_original_ci95', [])]} vs rule {[round(100*x, 2) for x in v['test_ap_gain_vs_rule_range_ci95']]}")
    print('kronos - persistence (final)', [round(100*x, 2) for x in res['kronos_minus_persistence_final_ci95']])


if __name__ == '__main__':
    main()
