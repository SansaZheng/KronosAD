"""Offline benchmark features. Targets are saved separately and never fitted.

Transaction rows and sequence summaries are NOT OHLC bars. No fabricated bars
are created. Temporal splits have fixed calendar/time boundaries, not label-
dependent boundaries. The original grouped market-data partitions are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data.load_supervised_data import load_sapienza, load_sd_fmm, load_tstr
from scripts.data.prepare_supervised_data import FEATURES
from scripts.data.transaction_data import iter_table, map_labels
from scripts.data.ieee_cis_local import iter_transactions

SPLITS = ('train', 'calibration', 'test')
DATASETS = ('sapienza_5S', 'sapienza_15S', 'sapienza_25S', 'tstr',
            'sd_stock', 'sd_crypto', 'ulb', 'elliptic', 'paysim',
            'tabformer', 'ibm_aml', 'ieee_cis')


def temporal_split(time, first, second):
    return np.where(np.asarray(time) <= first, 0,
                    np.where(np.asarray(time) <= second, 1, 2)).astype(np.int8)


def signed_log(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def hash_onehot(values, bins=16):
    """Stable nominal hashing, independent of both labels and data partitions."""
    strings = pd.Series(values).fillna('<missing>').astype(str)
    mapping = {v: int.from_bytes(hashlib.sha256(v.encode()).digest()[:4], 'little') % bins
               for v in strings.unique()}
    return np.eye(bins, dtype=np.float32)[strings.map(mapping).to_numpy()]


def time_boundaries(asset, column):
    """60/80 percent cumulative timestamp counts; never read target columns.

    Equal timestamp groups remain whole. A time-only pass avoids assuming a
    synthetic dataset has a uniform transaction rate across its nominal range.
    """
    counts = pd.Series(dtype='int64')
    for chunk in iter_table(asset, chunksize=200000, usecols=[column]):
        times = chunk[column]
        if asset == 'ibm_aml':
            times = pd.to_datetime(times, format='%Y/%m/%d %H:%M')
        counts = counts.add(times.value_counts(), fill_value=0)
    counts = counts.sort_index()
    cumulative = counts.cumsum().to_numpy()
    return [counts.index[np.searchsorted(cumulative, cumulative[-1]*fraction)]
            for fraction in (.6, .8)]


def sequence_summary(x):
    """Whole-sequence, retrospective features; not a causal online detector."""
    x = np.asarray(x, dtype=np.float64)[..., 0]
    # Relative changes discard price denomination without labelling any point.
    r = np.diff(np.log(np.maximum(x, 1e-12)), axis=1)
    parts = [r.mean(1), r.std(1), np.mean(np.abs(r), 1),
             np.sqrt(np.mean(r*r, 1)), np.sum(r, 1)]
    names = ['return_mean', 'return_std', 'return_abs_mean', 'return_rms', 'log_change']
    for q in (0, .01, .1, .25, .5, .75, .9, .99, 1):
        parts.append(np.quantile(r, q, axis=1)); names.append(f'return_q{q}')
    for width in (10, 50, 100):
        rolling = pd.DataFrame(r.T).rolling(width, min_periods=width)
        parts.extend([rolling.mean().max().to_numpy(), rolling.std(ddof=0).max().to_numpy()])
        names.extend([f'max_return_mean_{width}', f'max_return_std_{width}'])
    return np.column_stack(parts).astype(np.float32), names


def point_features(x):
    """Past/current values only within each source sequence; no label features.

    SD-FMM's publisher has already normalized its complete sequences, so these
    features cannot establish a causally deployable raw-price experiment.
    """
    x = np.asarray(x, dtype=np.float64)[..., 0]
    parts = [x, np.diff(x, prepend=x[:, :1], axis=1)]
    names = ['publisher_value', 'change']
    frame = pd.DataFrame(x.T)
    for width in (4, 16, 32):
        roll = frame.rolling(width, min_periods=1)
        parts.extend([roll.mean().to_numpy().T, roll.std(ddof=0).to_numpy().T,
                      roll.max().to_numpy().T-roll.min().to_numpy().T])
        names.extend([f'mean_{width}', f'std_{width}', f'range_{width}'])
    return np.stack(parts, -1).reshape(-1, len(parts)).astype(np.float32), names


class Writer:
    def __init__(self, root, name, protocol):
        self.path = Path(root) / name
        self.path.mkdir(parents=True, exist_ok=False)
        self.meta = {'dataset': name, 'protocol': protocol, 'splits': {}}
        self.handles = {}
        for split in SPLITS:
            self.handles[split] = [(self.path / f'{split}.{ext}').open('wb') for ext in ('x.f32', 'y.i8')]
            self.meta['splits'][split] = {'rows': 0, 'positive': 0, 'normal': 0, 'unknown': 0}

    def append(self, x, y, split, names):
        x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int8)
        if x.ndim != 2 or len(x) != len(y) or len(names) != x.shape[1]:
            raise ValueError('Invalid feature/target shape')
        if not np.isin(y, [-1, 0, 1]).all() or np.isinf(x).any():
            raise ValueError('Invalid targets/infinite features')
        if 'features' in self.meta and self.meta['features'] != names:
            raise ValueError('Feature schema changed between chunks')
        self.meta['features'] = names
        split = np.broadcast_to(np.asarray(split), (len(x),))
        if not np.isin(split, [0, 1, 2]).all():
            raise ValueError('Invalid split')
        for i, name in enumerate(SPLITS):
            mask = split == i
            if not mask.any():
                continue
            x[mask].tofile(self.handles[name][0]); y[mask].tofile(self.handles[name][1])
            counts = self.meta['splits'][name]
            counts['rows'] += int(mask.sum())
            for label, key in ((1, 'positive'), (0, 'normal'), (-1, 'unknown')):
                counts[key] += int((y[mask] == label).sum())

    def close(self):
        for pair in self.handles.values():
            for handle in pair:
                handle.close()
        if any(not v['rows'] for v in self.meta['splits'].values()):
            raise ValueError(f'Empty partition: {self.meta}')
        (self.path / 'manifest.json').write_text(json.dumps(self.meta, indent=2) + '\n')
        print(json.dumps({'prepared': self.meta['dataset'], 'features': len(self.meta['features']),
                          'splits': self.meta['splits']}), flush=True)


def prepare(name, root, ieee_dir=None):
    if name.startswith('sapienza_') or name in ('tstr', 'sd_stock', 'sd_crypto'):
        protocol = ('Existing coin-group split; same event/coin never split by row.' if name.startswith('sapienza_') else
                    'Existing deduplicated sequence-group split; offline retrospective task.')
        w = Writer(root, name, protocol)
        for i, split in enumerate(('train', 'validation', 'test')):
            if name.startswith('sapienza_'):
                x, y, _ = load_sapienza(name.split('_')[1], split); names = FEATURES
            elif name == 'tstr':
                x, y, _ = load_tstr(split); x, names = sequence_summary(x)
            else:
                x, y, _ = load_sd_fmm('stock' if name == 'sd_stock' else 'crypto', split)
                x, names = point_features(x); y = y.reshape(-1)
            w.append(x, y, i, names)
        w.close(); return

    protocols = {
        'ulb': 'Time<=86400 train; <=129600 calibration; later test. Time/label excluded from features.',
        'elliptic': 'Time step<=29 train; <=34 calibration; 35..49 test. Unknown nodes retained unlabelled during fitting/calibration; masked only for evaluation. No graph/message passing.',
        'paysim': 'Chronological 60/20/20 by cumulative time-only row counts, whole step groups. No account IDs, new balances, isFlaggedFraud features.',
        'tabformer': 'Year<=2016 train; Year==2017 calibration; Year>=2018 test. No User/Card/merchant IDs. Original row order is not chronological; split uses explicit Year.',
        'ibm_aml': 'Chronological 60/20/20 by cumulative time-only row counts, whole timestamp groups. No account/entity IDs or laundering pattern annotations.',
        'ieee_cis': 'Official labelled training transactions only: TransactionDT<=9504000 train; <=12614400 calibration; later internal test. Official competition test has no available truth. Numeric transaction features only; no identity table or raw transaction ID.'}
    w = Writer(root, name, protocols[name])
    if name in ('paysim', 'ibm_aml'):
        bounds = time_boundaries(name, 'step' if name == 'paysim' else 'Timestamp')
        w.meta['time_boundaries'] = [str(v) for v in bounds]
    if name == 'elliptic':
        labels = pd.concat(iter_table('elliptic_classes'), ignore_index=True).set_index('txId')['class']
        iterator = iter_table('elliptic_features', chunksize=50000)
    elif name == 'ieee_cis':
        if ieee_dir is None:
            raise ValueError('IEEE-CIS requires --ieee-dir with locally authorized data')
        iterator = iter_transactions(ieee_dir, chunksize=25000, with_identity=False)
    else:
        iterator = iter_table(name, chunksize=100000, low_memory=False)
    numeric_columns = None
    for d in iterator:
        if name == 'ulb':
            names = [f'V{i}' for i in range(1, 29)] + ['log_Amount']
            x = np.column_stack([d[[f'V{i}' for i in range(1, 29)]], signed_log(d.Amount)])
            y = map_labels(name, d.Class); split = temporal_split(d.Time, 86400, 129600)
        elif name == 'elliptic':
            x = d.iloc[:, 2:].to_numpy(); names = [f'feature_{i}' for i in range(2, 167)]
            y = map_labels(name, labels.reindex(d[0]).to_numpy())
            split = temporal_split(d[1], 29, 34)
        elif name == 'paysim':
            types = ('CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER')
            names = ['log_amount', 'log_oldbalanceOrg', 'log_oldbalanceDest'] + ['type_'+t for t in types]
            x = np.column_stack([signed_log(d[['amount', 'oldbalanceOrg', 'oldbalanceDest']]),
                                 np.column_stack([d.type.eq(t) for t in types])])
            y = map_labels(name, d.isFraud); split = temporal_split(d.step, *bounds)
        elif name == 'tabformer':
            minutes = pd.to_timedelta(d.Time.astype(str) + ':00').dt.total_seconds().to_numpy() / 60
            amount = pd.to_numeric(d.Amount.astype(str).str.replace('$', '', regex=False), errors='raise')
            x = np.column_stack([signed_log(amount), d.Month, d.Day, np.sin(minutes*2*np.pi/1440),
                                 np.cos(minutes*2*np.pi/1440), d['Errors?'].notna(),
                                 d['Merchant State'].isna(), d.Zip.isna(),
                                 hash_onehot(d['Use Chip'], 8), hash_onehot(d.MCC, 32)])
            names = ['log_amount', 'month', 'day', 'minute_sin', 'minute_cos', 'errors_present',
                     'state_missing', 'zip_missing'] + [f'chip_hash_{i}' for i in range(8)] + [f'mcc_hash_{i}' for i in range(32)]
            y = map_labels(name, d['Is Fraud?']); split = temporal_split(d.Year, 2016, 2017)
        elif name == 'ibm_aml':
            date = pd.to_datetime(d.Timestamp, format='%Y/%m/%d %H:%M')
            x = np.column_stack([signed_log(d[['Amount Received', 'Amount Paid']]),
                                 d['From Bank'].eq(d['To Bank']),
                                 d['Receiving Currency'].eq(d['Payment Currency']),
                                 hash_onehot(d['Receiving Currency']), hash_onehot(d['Payment Currency']),
                                 hash_onehot(d['Payment Format'])])
            names = ['log_received', 'log_paid', 'same_bank', 'same_currency'] + [f'{c}_hash_{i}' for c in ('receiving_currency', 'payment_currency', 'payment_format') for i in range(16)]
            y = map_labels(name, d['Is Laundering'])
            split = temporal_split(date, *bounds)
        else:
            if numeric_columns is None:
                excluded = {'TransactionID', 'TransactionDT', 'isFraud', 'ProductCD', 'card4',
                            'card6', 'P_emaildomain', 'R_emaildomain'} | {f'M{i}' for i in range(1, 10)}
                numeric_columns = [c for c in d if c not in excluded]
            names = numeric_columns
            x = d[names].apply(pd.to_numeric, errors='raise').to_numpy()
            y = map_labels('ulb', d.isFraud); split = temporal_split(d.TransactionDT, 9504000, 12614400)
        w.append(x, y, split, names)
    w.close()


def load(root, name, split, targets=False):
    """Features can be loaded without ever opening a target file."""
    path = Path(root) / name
    m = json.loads((path / 'manifest.json').read_text())
    n, p = m['splits'][split]['rows'], len(m['features'])
    if targets:
        return np.memmap(path / f'{split}.y.i8', mode='r', dtype='int8', shape=(n,))
    return np.memmap(path / f'{split}.x.f32', mode='r', dtype='float32', shape=(n, p))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--ieee-dir', type=Path)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.cache.chmod(0o700)
    for name in args.datasets:
        if (args.cache / name / 'manifest.json').exists():
            print(f'Already prepared: {name}', flush=True)
        else:
            prepare(name, args.cache, args.ieee_dir)
