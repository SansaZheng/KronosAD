"""Independently check saved scores, sample indices and confusion matrices."""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import load
from .run import digest, write_json


def verify(runs, cache):
    counts = 0
    total_evaluated = 0
    for path in sorted(runs.glob('*/*/result.json')):
        result = json.loads(path.read_text()); directory = path.parent
        name = result['dataset']
        scores = np.load(directory / 'test_scores.npy', mmap_mode='r')
        labels = load(cache, name, 'test', targets=True)
        threshold = json.loads((directory / 'threshold.json').read_text())
        assert threshold == result['threshold'], str(path)
        assert scores.shape == labels.shape and np.isfinite(scores).all()
        assert digest(directory / 'test_scores.npy') == result['test_scores_sha256']
        assert digest(cache / name / 'manifest.json') == result['data_manifest_sha256']
        cal = np.load(directory / 'calibration_scores.npy')
        assert len(cal) == threshold['n_calibration']
        assert int((cal > threshold['initial_threshold']).sum()) == threshold['n_peaks']
        for split, filename in [('train', 'train_indices.npy'), ('calibration', 'calibration_indices.npy')]:
            indices = np.load(directory / filename)
            n = result['split_counts'][split]['rows']
            assert len(indices) == len(np.unique(indices)) and (indices >= 0).all() and (indices < n).all()
        observed = dict(tp=0, fp=0, fn=0, tn=0)
        for i in range(0, len(scores), 500000):
            y, s = labels[i:i+500000], scores[i:i+500000]
            alarm = s >= threshold['threshold']
            observed['tp'] += int(np.count_nonzero(alarm & (y == 1)))
            observed['fp'] += int(np.count_nonzero(alarm & (y == 0)))
            observed['fn'] += int(np.count_nonzero(~alarm & (y == 1)))
            observed['tn'] += int(np.count_nonzero(~alarm & (y == 0)))
        for key, value in observed.items():
            assert value == result['metrics'][key], (str(path), key)
        total = sum(observed.values())
        assert total == result['metrics']['evaluated']
        numerator = 2 * observed['tp']
        denominator = numerator + observed['fp'] + observed['fn']
        assert np.isclose(numerator/denominator if denominator else 0., result['metrics']['f1'])
        total_evaluated += total; counts += 1
    return {'runs_verified': counts, 'evaluated_labels_across_runs': total_evaluated,
            'score_sha256_verified': True, 'confusion_matrices_independently_recomputed': True,
            'sampling_indices_unique_and_in_bounds': True, 'pot_records_match_calibration_scores': True,
            'raw_rows_in_report': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = verify(args.runs, args.cache)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
