"""Run a prepared series with its documented split and per-feature scale floors."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anomaly.__main__ import build_parser, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/financial_anomalies'))
    parser.add_argument('--series-id', default='AMBBTC_2019-01-01_s00')
    parser.add_argument('--model', default='NeoQuasar/Kronos-mini')
    parser.add_argument('--tokenizer', default='NeoQuasar/Kronos-Tokenizer-2k')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--sample-count', type=int, default=1)
    parser.add_argument('--test-end', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    catalog = pd.read_csv(args.root / 'catalog.csv')
    selected = catalog[catalog.series_id == args.series_id]
    if len(selected) != 1:
        parser.error('series-id must match exactly one row in catalog.csv')
    entry = selected.iloc[0]
    if int(entry.calibration_end) <= 64:
        parser.error('This series has insufficient pre-event calibration history')
    argv = ['--csv', str(args.root / entry.file), '--calibration-end', str(entry.calibration_end),
            '--model', args.model, '--tokenizer', args.tokenizer, '--device', args.device,
            '--lookback', '64', '--sample-count', str(args.sample_count), '--output',
            str(args.output or Path('anomaly_results') / args.series_id)]
    for name, value in json.loads(entry.epsilon_json).items():
        argv += ['--epsilon', f'{name}={value}']
    if args.local_files_only:
        argv.append('--local-files-only')
    if args.test_end is not None:
        argv += ['--test-end', str(args.test_end)]
    if args.plot:
        argv.append('--plot')
    run(build_parser().parse_args(argv))


if __name__ == '__main__':
    main()
