"""Download audited public sources and build contiguous Kronos input series.

Run from repository root: python scripts/data/prepare_financial_data.py
Network access is only to fixed public research/market-data endpoints. Existing
raw files are SHA-checked against the previous manifest on repeat runs.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import pandas as pd

SAP_COMMIT = 'd71250d4cb055dde2d415c8cba38a0dcd6eb6e16'
SAP_ROOT = f'https://raw.githubusercontent.com/SystemsLab-Sapienza/pump-and-dump-dataset/{SAP_COMMIT}'
MEME_URL = 'https://zenodo.org/api/records/17830944/files/meme_coin-main.zip/content'
MEME_MD5 = 'd6676f869dec06a624f2528c20dfcee7'
BINANCE_ROOT = 'https://data.binance.vision/data/spot/daily/klines'
FEATURES = ['open', 'high', 'low', 'close', 'volume', 'amount']
BINANCE_COLUMNS = ['open_time', 'open', 'high', 'low', 'close', 'volume',
                   'close_time', 'amount', 'trades', 'buy_volume', 'buy_amount', 'ignore']


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(url, path, known=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if known is not None and digest(path) != known:
            raise ValueError(f'Existing source checksum changed: {path}')
        return path
    partial = path.with_name(path.name + '.partial')
    command = ['curl', '--fail', '--location', '--silent', '--show-error',
               '--connect-timeout', '10', '--max-time', '50', '--retry', '2',
               '--continue-at', '-', '--output', str(partial), url]
    subprocess.run(command, check=True)
    if known is not None and digest(partial) != known:
        raise ValueError(f'Redownloaded source differs from pinned manifest: {path}')
    partial.replace(path)
    return path


def canonical_binance(raw):
    """2025+ SPOT archives use microseconds; older archives use milliseconds."""
    data = raw.copy()
    ticks = pd.to_numeric(data['open_time'], errors='raise')
    if ticks.empty:
        raise ValueError('Empty Binance archive')
    unit = 'us' if ticks.median() >= 10**14 else 'ms'
    result = pd.DataFrame({'timestamps': pd.to_datetime(ticks, unit=unit, utc=True)})
    for feature in FEATURES:
        result[feature] = pd.to_numeric(data[feature], errors='raise')
    return result


def valid_bars(frame):
    """Prices must be positive, volume/turnover nonnegative and OHLC coherent."""
    fields = [f for f in FEATURES if f in frame]
    values = frame[fields].to_numpy(dtype=float)
    mask = np.isfinite(values).all(axis=1)
    mask &= (frame[['open', 'high', 'low', 'close']] > 0).all(axis=1).to_numpy()
    mask &= (frame.high >= frame[['open', 'low', 'close']].max(axis=1)).to_numpy()
    mask &= (frame.low <= frame[['open', 'high', 'close']].min(axis=1)).to_numpy()
    for feature in ('volume', 'amount'):
        if feature in frame:
            mask &= (frame[feature] >= 0).to_numpy()
    return mask


def rescale_quote(frame, multiplier):
    """Change quote units consistently; base asset quantity stays unchanged.

    KronosPredictor adds an absolute 1e-5 before input normalization. Quoting
    tiny token prices in 1e-8 USD/BTC avoids that constant dominating the price.
    This multiplier is fixed by dataset, never estimated from test observations.
    """
    data = frame.copy()
    for field in ('open', 'high', 'low', 'close', 'amount'):
        if field in data:
            data[field] *= multiplier
    return data


def meme_quote_multiplier(frame):
    """Choose a decimal quote unit from the first 64 context bars only."""
    reference = float(frame.open.iloc[:64].median())
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError('Invalid initial reference price')
    return float(10.0 ** np.ceil(np.log10(100.0 / reference)))


def contiguous_segments(frame, frequency, minimum_rows=600):
    """Drop bad bars, split at real time gaps, never invent/impute OHLC bars."""
    ordered = frame.sort_values('timestamps').reset_index(drop=True)
    if ordered.timestamps.isna().any() or ordered.timestamps.duplicated().any():
        raise ValueError('Missing/duplicate timestamps require source inspection')
    good = ordered.loc[valid_bars(ordered)].reset_index(drop=True)
    groups = good.timestamps.diff().ne(pd.Timedelta(frequency)).cumsum()
    return [g.reset_index(drop=True) for _, g in good.groupby(groups) if len(g) >= minimum_rows]


def interval_labels(timestamps, events):
    """Author-derived intervals are inclusive bar-open timestamps in UTC."""
    labels = np.zeros(len(timestamps), dtype=np.int8)
    for event in events.itertuples():
        labels[(timestamps >= event.pump_start) & (timestamps <= event.dump_end)] = 1
    return labels


def weak_event_labels(timestamps, event_times, frequency='5min'):
    """Announcement bar plus an explicit [-2 min, +30 min) weak interval.

    Positive window labels include any bar intersecting the interval. This
    includes announcement uncertainty and is not manual per-bar ground truth.
    """
    onset = np.zeros(len(timestamps), dtype=np.int8)
    window = np.zeros(len(timestamps), dtype=np.int8)
    ends = timestamps + pd.Timedelta(frequency)
    for event in event_times:
        onset[(timestamps <= event) & (event < ends)] = 1
        window[(timestamps < event + pd.Timedelta('30min')) &
               (ends > event - pd.Timedelta('2min'))] = 1
    return onset, window


def save_series(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic gzip header: rerunning does not rewrite Git history by date.
    compression = {'method': 'gzip', 'mtime': 0} if path.suffix == '.gz' else None
    data.to_csv(path, index=False, compression=compression, float_format='%.17g')


def catalog_entry(frame, path, root, dataset, series_id, frequency, license_name, labels,
                  quote_multiplier=None):
    fields = [f for f in FEATURES if f in frame]
    positives = np.flatnonzero(frame.label.to_numpy() == 1)
    # Allow a 24-bar buffer before any KNOWN positive in calibration.
    boundary = min(500, int(positives[0]) - 24) if len(positives) else 500
    if boundary - 64 < 220:
        boundary = 0  # Not enough pre-event calibration history; do not auto-run.
    if boundary >= len(frame):
        boundary = 0
    reference = frame.iloc[:boundary] if boundary else frame.iloc[:64]
    epsilon = {f: max(float(reference[f].abs().median()) * 1e-6, 1e-18)
               for f in fields}
    if quote_multiplier is None:
        quote_multiplier = 1e8 if dataset == 'sapienza_binance' else 1.0
    quote_unit = {'sapienza_binance': 'satoshi', 'recent_binance': 'USDT'}.get(
        dataset, f'{1.0 / quote_multiplier:.0e}_USD')
    return dict(dataset=dataset, series_id=series_id, file=path.relative_to(root).as_posix(),
                rows=len(frame), frequency=frequency, start=frame.timestamps.iloc[0].isoformat(),
                end=frame.timestamps.iloc[-1].isoformat(), features=','.join(fields),
                positive_rows=int((frame.label == 1).sum()),
                unknown_rows=int((frame.label == -1).sum()),
                label_semantics=labels, dimension_ground_truth=False,
                calibration_end=boundary, epsilon_json=json.dumps(epsilon, sort_keys=True),
                quote_multiplier=quote_multiplier, quote_unit=quote_unit,
                license=license_name, sha256=digest(path))


def build(root, minimum_rows=600, offline=False):
    root = Path(root).resolve()
    old_manifest = root / 'raw_manifest.json'
    old = json.loads(old_manifest.read_text()) if old_manifest.exists() else {'files': []}
    known = {x['path']: x['sha256'] for x in old['files']}
    sources = []

    def acquire(url, relative, license_name):
        path = root / relative
        if offline and not path.exists():
            raise FileNotFoundError(f'Offline source missing: {relative}')
        download(url, path, known.get(relative))
        sources.append(dict(path=relative, url=url, bytes=path.stat().st_size,
                            sha256=digest(path), license=license_name))
        return path

    events_path = acquire(SAP_ROOT + '/pump_telegram.csv', 'raw/sapienza/pump_telegram.csv', 'MIT')
    acquire(SAP_ROOT + '/LICENSE', 'raw/sapienza/LICENSE', 'MIT')
    acquire('https://raw.githubusercontent.com/binance/binance-public-data/master/TERMS_AND_CONDITIONS.md',
            'raw/binance/TERMS_AND_CONDITIONS.md', 'CC-BY-NC-SA-4.0 + Binance Vision Dataset Terms')
    record_path = acquire('https://zenodo.org/api/records/17830944',
                          'raw/meme2026/zenodo_record.json', 'CC-BY-4.0')
    metadata = json.loads(record_path.read_text())
    if metadata['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('Unexpected Zenodo license; inspect before redistribution')
    archive_path = acquire(MEME_URL, 'raw/meme2026/meme_coin-main.zip', 'CC-BY-4.0')
    if digest(archive_path, 'md5') != MEME_MD5:
        raise ValueError('Zenodo archive does not match publisher MD5')

    events = pd.read_csv(events_path)
    events['event_time'] = pd.to_datetime(events.date + ' ' + events.hour, utc=True)
    binance_events = events[events.exchange.str.lower() == 'binance'].copy()
    binance_events = binance_events.drop_duplicates(['symbol', 'event_time']).sort_values('event_time')
    chosen = []
    for year, group in binance_events.groupby(binance_events.event_time.dt.year):
        # Deterministic small starter set: first/middle/last event in each year.
        chosen.extend(group.iloc[sorted({0, len(group) // 2, len(group) - 1})].to_dict('records'))
    chosen = pd.DataFrame(chosen)
    chosen.to_csv(root / 'sapienza_selected_events.csv', index=False)

    requests = {}
    for event in chosen.itertuples():
        for date in pd.date_range(event.event_time.normalize() - pd.Timedelta('3d'),
                                  event.event_time.normalize() + pd.Timedelta('1d'), freq='1d'):
            requests[(event.symbol + 'BTC', date.strftime('%Y-%m-%d'))] = None
    # Latest completed UTC days confirmed downloadable at acquisition, not future data.
    recent_days = ['2026-10-29', '2026-10-30', '2026-10-31']
    for symbol in ('BTCUSDT', 'ETHUSDT'):
        for day in recent_days:
            requests[(symbol, day)] = None

    def get_daily(key):
        symbol, day = key
        filename = f'{symbol}-5m-{day}.zip'
        relative = f'raw/binance/{symbol}/{filename}'
        url = f'{BINANCE_ROOT}/{symbol}/5m/{filename}'
        archive = acquire(url, relative, 'CC-BY-NC-SA-4.0 + Binance Vision Dataset Terms')
        checksum = acquire(url + '.CHECKSUM', relative + '.CHECKSUM',
                           'CC-BY-NC-SA-4.0 + Binance Vision Dataset Terms')
        expected = checksum.read_text().split()[0]
        if digest(archive) != expected:
            raise ValueError(f'Binance publisher SHA256 mismatch: {archive}')
        with zipfile.ZipFile(archive) as zipped:
            members = [f for f in zipped.namelist() if f.endswith('.csv')]
            if len(members) != 1:
                raise ValueError(f'Unexpected Binance archive members: {archive}')
            data = pd.read_csv(zipped.open(members[0]), header=None, names=BINANCE_COLUMNS)
        return canonical_binance(data)

    daily = {}
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(get_daily, key): key for key in requests}
        for number, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            try:
                daily[key] = future.result()
            except Exception as error:
                if offline:
                    raise
                failures.append({'symbol': key[0], 'day': key[1], 'reason': str(error)})
            if number % 10 == 0 or number == len(futures):
                print(f'Checked Binance archives: {number}/{len(futures)}', flush=True)
    catalog = []
    skipped = list(failures)
    for event in chosen.itertuples():
        symbol = event.symbol + 'BTC'
        days = pd.date_range(event.event_time.normalize() - pd.Timedelta('3d'),
                             event.event_time.normalize() + pd.Timedelta('1d'), freq='1d')
        keys = [(symbol, day.strftime('%Y-%m-%d')) for day in days]
        if not all(key in daily for key in keys):
            skipped.append({'case': symbol + '_' + event.date, 'reason': 'Missing source day; no fabricated bars'})
            continue
        data = rescale_quote(pd.concat([daily[key] for key in keys], ignore_index=True), 1e8)
        relevant = binance_events[binance_events.symbol == event.symbol].event_time
        onset, window = weak_event_labels(data.timestamps, relevant)
        data['label_event_onset'] = onset
        data['label'] = window
        for i, segment in enumerate(contiguous_segments(data, '5min', minimum_rows)):
            case_id = f'{symbol}_{event.date}_s{i:02d}'
            path = root / 'prepared/sapienza_binance' / (case_id + '.csv.gz')
            save_series(segment, path)
            catalog.append(catalog_entry(segment, path, root, 'sapienza_binance', case_id,
                                         '5min', 'CC-BY-NC-SA-4.0 (derived OHLCVA; event list MIT)',
                                         'weak_announcement_window_minus2_plus30min'))
    for symbol in ('BTCUSDT', 'ETHUSDT'):
        keys = [(symbol, day) for day in recent_days]
        if not all(key in daily for key in keys):
            raise ValueError(f'Recent {symbol} data incomplete; inspect download log')
        data = pd.concat([daily[key] for key in keys], ignore_index=True)
        data['label'] = -1  # Unknown, not normal.
        for i, segment in enumerate(contiguous_segments(data, '5min', minimum_rows)):
            series_id = f'{symbol}_20261029_20261031_s{i:02d}'
            path = root / 'prepared/recent_binance' / (series_id + '.csv.gz')
            save_series(segment, path)
            catalog.append(catalog_entry(segment, path, root, 'recent_binance', series_id, '5min',
                                         'CC-BY-NC-SA-4.0 + Binance Vision Dataset Terms', 'unlabeled_minus1'))

    meme_quality = []
    with zipfile.ZipFile(archive_path) as outer:
        prefix = outer.namelist()[0].split('/')[0]
        events_name = prefix + '/data/pump_and_dumps_on_performing_tokens_after_mid_january.csv'
        event_bytes = outer.read(events_name)
        (root / 'raw/meme2026/pump_and_dump_events.csv').write_bytes(event_bytes)
        author_events = pd.read_csv(io.BytesIO(event_bytes))
        for field in ('pump_start', 'pump_peak', 'dump_end'):
            author_events[field] = pd.to_datetime(author_events[field], utc=True)
        # The publisher lowercased filenames, including Solana. Use those aliases
        # only to join this archive; do NOT treat them as canonical on-chain addresses.
        author_events['source_alias'] = (author_events.platform.str.lower() + '_' +
                                          author_events.address.str.lower())
        collisions = author_events.groupby('source_alias').address.nunique()
        if (collisions > 1).any():
            raise ValueError('Ambiguous lowercased address alias in source event labels')
        hour_bytes = outer.read(prefix + '/data/ohlcv_data_hour_until_mid_march.zip')
        with zipfile.ZipFile(io.BytesIO(hour_bytes)) as hourly:
            for member in sorted(hourly.namelist()):
                if not member.endswith('.csv'):
                    continue
                alias = Path(member).stem
                raw = pd.read_csv(hourly.open(member))
                raw = raw.rename(columns={'highest': 'high', 'lowest': 'low', 'closed': 'close'})
                times = pd.to_datetime(raw.timestamp, unit='s', utc=True)
                data = pd.DataFrame({'timestamps': times})
                for field in ('open', 'high', 'low', 'close'):
                    data[field] = pd.to_numeric(raw[field], errors='coerce')
                # GeckoTerminal trading volume is quote/USD turnover, not token
                # quantity. Preserve it as amount; base-volume remains absent.
                data['amount'] = pd.to_numeric(raw.volume, errors='coerce')
                relevant = author_events[author_events.source_alias == alias]
                data['label'] = interval_labels(data.timestamps, relevant)
                good_count = int(valid_bars(data).sum()) if len(data) else 0
                segments = contiguous_segments(data, '1h', minimum_rows)
                kept = 0
                for i, segment in enumerate(segments):
                    multiplier = meme_quote_multiplier(segment)
                    segment = rescale_quote(segment, multiplier)
                    series_id = f'{alias}_s{i:02d}'
                    path = root / 'prepared/meme2026' / (series_id + '.csv.gz')
                    save_series(segment, path)
                    kept += len(segment)
                    catalog.append(catalog_entry(segment, path, root, 'meme2026', series_id, '1h',
                                                 'CC-BY-4.0', 'author_algorithm_pump_start_through_dump_end',
                                                 quote_multiplier=multiplier))
                meme_quality.append(dict(source_member=member, source_rows=len(data),
                                         invalid_rows=len(data)-good_count,
                                         retained_rows=kept, retained_segments=len(segments),
                                         source_events=len(relevant)))
    pd.DataFrame(meme_quality).to_csv(root / 'meme_quality_report.csv', index=False)
    pd.DataFrame(skipped).to_json(root / 'download_issues.json', orient='records', indent=2)
    catalog_frame = pd.DataFrame(catalog).sort_values(['dataset', 'series_id']).reset_index(drop=True)
    catalog_frame.to_csv(root / 'catalog.csv', index=False)
    raw_manifest = dict(schema_version=1, acquired_date='2026-11-02',
                        files=sorted(sources, key=lambda entry: entry['path']))
    (root / 'raw_manifest.json').write_text(json.dumps(raw_manifest, indent=2) + '\n')
    summary = dict(raw_files=len(sources), raw_bytes=sum(s['bytes'] for s in sources),
                   sapienza_source_events=len(events), selected_events=len(chosen),
                   meme_source_hourly_files=len(meme_quality),
                   meme_source_hourly_rows=sum(x['source_rows'] for x in meme_quality),
                   meme_author_events=len(author_events), download_failures=len(failures),
                   datasets={})
    for name, group in catalog_frame.groupby('dataset'):
        summary['datasets'][name] = dict(series=len(group), rows=int(group.rows.sum()),
                                        positive_rows=int(group.positive_rows.sum()),
                                        runnable_series=int((group.calibration_end > 0).sum()))
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return catalog_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/financial_anomalies'))
    parser.add_argument('--minimum-rows', type=int, default=600)
    parser.add_argument('--offline', action='store_true', help='Rebuild from committed raw files only')
    args = parser.parse_args()
    if args.minimum_rows < 600:
        parser.error('minimum-rows must be >=600 to preserve useful POT calibration history')
    build(args.root, args.minimum_rows, args.offline)


if __name__ == '__main__':
    main()
