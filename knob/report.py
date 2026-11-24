"""Evaluate the Knob experiment.  This is the only module that reads pump labels."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

SCORES = ['nll', 'resid_max', 'resid_price']
RULES = ['rule_range', 'rule_amount_24h', 'rule_return_z64']


def recall_at_fpr(score, y, w, fpr, mask=None):
    neg = y == 0
    order = np.argsort(-score[neg])
    cum = np.cumsum(w[neg][order])/w[neg].sum()
    i = np.searchsorted(cum, fpr, side='right')
    thr = score[neg][order][min(i, len(order) - 1)]
    pos = (y == 1) if mask is None else mask
    return float((score[pos] > thr).mean())


def metrics(score, y, w, subtle):
    return {'ap': float(average_precision_score(y, score, sample_weight=w)),
            'auroc': float(roc_auc_score(y, score, sample_weight=w)),
            'recall_fpr1': recall_at_fpr(score, y, w, 0.01),
            'recall_fpr01': recall_at_fpr(score, y, w, 0.001),
            'subtle_recall_fpr1': recall_at_fpr(score, y, w, 0.01, subtle),
            'subtle_miss_fpr1': 1 - recall_at_fpr(score, y, w, 0.01, subtle)}


def subtle_mask(d):
    """Pump hours whose own high/low range is below the median pump: least visible to a range rule."""
    y, r = d['y'], d['rule_range']
    return (y == 1) & (r <= np.median(r[y == 1]))


def bootstrap(a, b, y, w, reps, seed):
    """Paired stratified bootstrap of AP(a) - AP(b)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = []
    for _ in range(reps):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        out.append(average_precision_score(y[i], a[i], sample_weight=w[i]) - average_precision_score(y[i], b[i], sample_weight=w[i]))
    return [float(q) for q in np.quantile(out, [0.025, 0.5, 0.975])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--bootstrap', type=int, default=1000)
    a = p.parse_args()
    runs, out = Path(a.runs), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((runs/'config.json').read_text())
    names = config['variants']
    data = {s: np.load(runs/f'{s}.npz') for s in ('calibration', 'test')}
    subtle = {s: subtle_mask(d) for s, d in data.items()}
    res = {'config': config, 'knobs': json.loads((runs/'knobs.json').read_text()), 'splits': {}, 'metrics': {},
           'effects': {}, 'selection': {}, 'rules': {}}
    for s, d in data.items():
        y, w = d['y'], d['w']
        res['splits'][s] = {'pump_hours': int(y.sum()), 'sampled_other_hours': int((y == 0).sum()),
                            'represented_other_hours': float(w[y == 0].sum()),
                            'prevalence': float(y.sum()/w.sum()), 'subtle_pump_hours': int(subtle[s].sum())}
        res['rules'][s] = {r: metrics(d[r], y, w, subtle[s]) for r in RULES}
        res['metrics'][s] = {n: {k: metrics(d[k][j], y, w, subtle[s]) for k in SCORES} for j, n in enumerate(names)}
    # steering effect on forecasts, measured on non-pump calibration hours (no labels needed beyond exclusion)
    d = data['calibration']
    neg = d['y'] == 0
    for j, n in enumerate(names):
        res['effects'][n] = {'delta_nll': float((d['nll'][j] - d['nll'][0])[neg].mean()),
                             'share_nll_improved': float(((d['nll'][j] < d['nll'][0])[neg]).mean()),
                             'jump_mean': float(d['jump'][j][neg].mean()), 'jump_median': float(np.median(d['jump'][j][neg])),
                             'range_mean': float(d['range'][j][neg].mean()),
                             'entropy_mean': float(d['entropy_s1'][j][neg].mean()),
                             'resid_price_median': float(np.median(d['resid_price'][j][neg]))}
    for b in ('predictable', 'smooth'):
        for how in ('knob', 'dense'):
            grid = [(0.0, 0)] + sorted((float(n.rsplit('/', 1)[1]), j) for j, n in enumerate(names) if n.startswith(f'{b}/{how}/normal/'))
            grid.sort()
            taus = [t for t, _ in grid]
            mono = {m: float(spearmanr(taus, [res['effects'][names[j]][m] for _, j in grid])[0]) for m in ('jump_mean', 'entropy_mean', 'delta_nll', 'range_mean')}
            sel = {'taus': taus, 'monotonicity_spearman': mono}
            for k in SCORES:
                cal = [res['metrics']['calibration'][names[j]][k]['ap'] for _, j in grid]
                best = int(np.argmax(cal))
                tau, j = grid[best]
                t = data['test']
                sel[k] = {'calibration_ap_by_tau': cal, 'test_ap_by_tau': [res['metrics']['test'][names[jj]][k]['ap'] for _, jj in grid],
                          'selected_tau': tau, 'selected_variant': names[j],
                          'test_base': res['metrics']['test']['base'][k], 'test_selected': res['metrics']['test'][names[j]][k],
                          'test_ap_gain_ci95': [0.0, 0.0, 0.0] if j == 0 else bootstrap(t[k][j], t[k][0], t['y'], t['w'], a.bootstrap, 7)}
            res['selection'][f'{b}/{how}'] = sel
    (out/'results.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({'splits': res['splits'], 'rules_test': {r: v['ap'] for r, v in res['rules']['test'].items()},
                      'base_test': {k: v['ap'] for k, v in res['metrics']['test']['base'].items()}}, indent=1))


if __name__ == '__main__':
    main()
