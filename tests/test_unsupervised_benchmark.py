"""Protocol checks use invented data, never restricted competition records."""
import json

import numpy as np
import pytest

from benchmarks.data import temporal_split, point_features, hash_onehot, Writer, load
from benchmarks.evaluation import calibrate, evaluate
from benchmarks.models import Preprocessor, make_adapter
from benchmarks.run import sample


def test_unknown_mask_and_unadjusted_confusion():
    result = evaluate([1, 0, -1, 1, 0], [.9, .8, 100, .1, .2], .5)
    assert [result[k] for k in ('tp', 'fp', 'fn', 'tn')] == [1, 1, 1, 1]
    assert result['f1'] == .5 and result['unknown_excluded'] == 1
    assert result['fpr'] == .5
    assert result['evaluated'] == 4


def test_no_label_based_sampling_or_calibration():
    x = np.arange(200).reshape(100, 2)
    a, i = sample(x, 17, 42); b, j = sample(x, 17, 42)
    np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(i, j)
    assert len(np.unique(i)) == 17
    with pytest.raises(ValueError):
        calibrate(np.ones(1000))  # No silent threshold fallback.


def test_preprocessing_train_only_missing_and_extremes():
    train = np.array([[1., 2., np.nan], [2., np.nan, np.nan], [3., 2., np.nan]])
    prep = Preprocessor().fit(train)
    center = prep.center.copy()
    z = prep.transform([[1000000, np.nan, 4.]])
    assert z.shape == (1, 3) and z[0, 0] > 10000
    assert z[0, -1] == 1
    np.testing.assert_array_equal(prep.center, center)


def test_fixed_temporal_boundaries_and_nominal_hash():
    np.testing.assert_array_equal(temporal_split([1, 10, 11, 20, 21], 10, 20), [0, 0, 1, 1, 2])
    np.testing.assert_array_equal(hash_onehot(['a', 'b']), hash_onehot(['b', 'a'])[::-1])


def test_point_features_do_not_read_later_points():
    x = np.arange(40., dtype=float).reshape(1, 40, 1)
    a, names = point_features(x)
    x[:, 20:] = 100000
    b, _ = point_features(x)
    np.testing.assert_array_equal(a[:20], b[:20])
    assert len(names) == a.shape[1]


def test_cache_targets_are_separate_and_schema_checked(tmp_path):
    writer = Writer(tmp_path, 'toy', 'invented rows')
    for split in range(3):
        writer.append([[1, 2], [3, 4]], [0, 1], split, ['a', 'b'])
    writer.close()
    (tmp_path / 'toy/train.y.i8').unlink()
    assert load(tmp_path, 'toy', 'train').shape == (2, 2)
    with pytest.raises(FileNotFoundError):
        load(tmp_path, 'toy', 'train', targets=True)


def test_adapter_freezes_backbone_but_trains_input(tmp_path):
    torch = pytest.importorskip('torch')
    config = dict(s1_bits=2, s2_bits=2, n_layers=1, d_model=8, n_heads=2, ff_dim=16,
                  ffn_dropout_p=.2, attn_dropout_p=0., resid_dropout_p=.2,
                  token_dropout_p=0., learn_te=True)
    (tmp_path / 'config.json').write_text(json.dumps(config))
    model = make_adapter(4, tmp_path, 42, random_backbone=True)
    model.train()
    assert not model.backbone.training
    original = {k: v.detach().clone() for k, v in model.backbone.state_dict().items()}
    original_encoder = model.encoder[0].weight.detach().clone()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.01)
    x = torch.randn(8, 4)
    loss = (model(x)-x).square().mean(); loss.backward(); optimizer.step()
    assert not torch.equal(model.encoder[0].weight, original_encoder)
    for k, v in model.backbone.state_dict().items():
        assert torch.equal(v, original[k])
    assert all(p.grad is None for p in model.backbone.parameters())
