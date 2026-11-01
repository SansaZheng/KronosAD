"""Validate committed data, source SHA256s, OHLC consistency and temporal splits."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anomaly.scoring import validate_frame
from scripts.data.prepare_financial_data import digest, valid_bars


def validate(root):
    root = Path(root)
    manifest = json.loads((root / 'raw_manifest.json').read_text())
    for entry in manifest['files']:
        path = root / entry['path']
        assert path.stat().st_size == entry['bytes'], str(path)
        assert digest(path) == entry['sha256'], str(path)
    catalog = pd.read_csv(root / 'catalog.csv')
    assert not catalog.file.duplicated().any()
    assert not catalog.series_id.duplicated().any()
    files = {p.relative_to(root).as_posix() for p in (root/'prepared').rglob('*.csv.gz')}
    assert files == set(catalog.file), 'Missing/orphan prepared series'
    for entry in catalog.itertuples():
        path = root / entry.file
        assert digest(path) == entry.sha256, str(path)
        data = validate_frame(pd.read_csv(path))
        assert len(data) == entry.rows and len(data) >= 600
        assert valid_bars(data).all(), entry.series_id
        assert data.timestamps.diff().iloc[1:].eq(pd.Timedelta(entry.frequency)).all(), entry.series_id
        assert set(data.label.unique()) <= {-1, 0, 1}
        assert int((data.label == 1).sum()) == entry.positive_rows
        assert int((data.label == -1).sum()) == entry.unknown_rows
        assert list(f for f in ('open','high','low','close','volume','amount') if f in data) == entry.features.split(',')
        if entry.calibration_end:
            assert entry.calibration_end - 64 >= 220
            assert not (data.label.iloc[:entry.calibration_end] == 1).any(), entry.series_id
        if entry.dataset == 'recent_binance':
            assert (data.label == -1).all(), 'Unknown must not become normal'
        if entry.dataset == 'meme2026':
            assert 'volume' not in data and 'amount' in data
        floors = json.loads(entry.epsilon_json)
        assert all(np.isfinite(v) and v > 0 for v in floors.values())
        assert np.isfinite(entry.quote_multiplier) and entry.quote_multiplier > 0
    result = {'validated_raw_files': len(manifest['files']), 'validated_series': len(catalog),
              'validated_prepared_rows': int(catalog.rows.sum()), 'all_checks_passed': True}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/financial_anomalies'))
    validate(parser.parse_args().root)
