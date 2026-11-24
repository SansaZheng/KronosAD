"""Windows of the Xu & Livshits hourly OHLCVA pump-and-dump candles (same task as xu_pump_1h)."""
import numpy as np
import pandas as pd

from .engine import FEATURES

BOUNDS = (pd.Timestamp('2018-08-24', tz='UTC'), pd.Timestamp('2018-09-26 08:00', tz='UTC'))
TOLERANCE = 2


def load(path):
    k = pd.read_csv(path, parse_dates=['timestamp'])
    k = k[k.exchange != 'CCCAGG'].sort_values(['coin', 'exchange', 'timestamp'], kind='stable').reset_index(drop=True)
    k['series'] = k.groupby(['coin', 'exchange'], sort=False).ngroup()
    k['pos'] = k.groupby('series').cumcount()
    hours = k.timestamp.diff().dt.total_seconds().div(3600)
    k['contiguous'] = (hours == 1) | (k.pos == 0)
    near = np.zeros(len(k), bool)
    p = np.flatnonzero(k.pump.to_numpy() == 1)
    s = k.series.to_numpy()
    for d in range(-TOLERANCE, TOLERANCE + 1):
        q = p + d
        q = q[(q >= 0) & (q < len(k))]
        near[q[s[q] == s[np.clip(q - d, 0, len(k) - 1)]]] = True
    k['near_pump'] = near
    k['split'] = np.where(k.timestamp < BOUNDS[0], 'train', np.where(k.timestamp < BOUNDS[1], 'calibration', 'test'))
    return k


def sample(k, split, lookback, negatives, seed, live_only=False):
    """All pump hours of the split plus uniformly sampled other hours.

    Returns row indices, labels and inverse-probability weights (weight 1 for pumps), so weighted
    metrics estimate the full natural-ratio split.  Hours within +/-2 bars of a pump are not sampled
    as negatives.  Rows need ``lookback`` earlier bars of the same series.
    """
    ok = (k.split == split).to_numpy() & (k.pos >= lookback).to_numpy()
    pos = np.flatnonzero(ok & (k.pump == 1).to_numpy())
    pool = ok & ~k.near_pump.to_numpy()
    if live_only:
        c = k.close.to_numpy()
        prev = np.concatenate([[np.nan], c[:-1]])
        moved = pd.Series((c != prev) & (k.pos > 0).to_numpy()).groupby(k.series.to_numpy()).rolling(lookback, min_periods=1).sum()
        pool &= np.concatenate([[0], moved.to_numpy()[:-1]]) >= lookback//4
    pool = np.flatnonzero(pool)
    neg = np.sort(np.random.default_rng(seed).choice(pool, min(negatives, len(pool)), replace=False))
    rows = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)])
    w = np.concatenate([np.ones(len(pos)), np.full(len(neg), len(pool)/len(neg))])
    return rows, y, w, {'pump_hours': int(len(pos)), 'negative_pool': int(len(pool)), 'negatives_sampled': int(len(neg))}


def windows(k, rows, lookback):
    v = k[FEATURES].to_numpy(np.float64)
    ts = k.timestamp
    stamp = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)
    off = np.arange(-lookback, 1)
    idx = rows[:, None] + off[None]
    return v[idx], stamp[idx[:, :-1]]


def rules(k, rows):
    """Training-free reference scores computed from the scored bar and strictly earlier bars."""
    v = k[FEATURES].to_numpy(np.float64)
    o, h, l, c, a = v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 5]
    rng = np.log(np.maximum(h, 1e-300)/np.maximum(l, 1e-300))
    past = np.stack([a[rows - d] for d in range(1, 25)], 1).mean(1)
    lr = np.log(np.maximum(c, 1e-300)) - np.log(np.maximum(np.concatenate([[c[0]], c[:-1]]), 1e-300))
    hist = np.stack([lr[rows - d] for d in range(1, 65)], 1)
    return {'rule_range': rng[rows], 'rule_amount_24h': a[rows]/(past + 1e-12),
            'rule_return_z64': np.abs(lr[rows])/(hist.std(1) + 1e-6)}
