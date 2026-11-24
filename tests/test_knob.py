"""Offline tests for the Knob engine and steering.  Random weights; no download needed."""
import numpy as np
import pytest
import torch

from knob.boundary import GRAD_EPS, Knob, Pooled, discover, tercile_labels
from knob.engine import Engine, normalise
from knob.report import recall_at_fpr
from model import Kronos, KronosTokenizer

L, B = 24, 5


@pytest.fixture(scope='module')
def setup():
    torch.manual_seed(0)
    tok = KronosTokenizer(d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=3, n_dec_layers=3, ffn_dropout_p=0.0,
                          attn_dropout_p=0.0, resid_dropout_p=0.0, s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0, gamma=1.1,
                          zeta=0.05, group_size=4).eval()
    model = Kronos(s1_bits=4, s2_bits=4, n_layers=3, d_model=32, n_heads=2, ff_dim=64, ffn_dropout_p=0.0, attn_dropout_p=0.0,
                   resid_dropout_p=0.0, token_dropout_p=0.0, learn_te=True).eval()
    rng = np.random.default_rng(0)
    price = 1e-6*np.exp(np.cumsum(rng.normal(0, 0.02, (B, L + 1)), 1))
    bars = np.stack([price, price*1.01, price*0.99, price*1.002, rng.uniform(1, 9, (B, L + 1)), rng.uniform(1, 9, (B, L + 1))], -1)
    stamps = np.stack([np.zeros((B, L)), np.tile(np.arange(L) % 24, (B, 1)), np.ones((B, L)), np.ones((B, L)), np.ones((B, L))], -1).astype(np.float32)
    return tok, model, bars, stamps


def reference(tok, model, bars, stamps, patch=None):
    x = torch.from_numpy(normalise(bars)[0])
    s1, s2 = tok.encode(x, half=True)
    h = model.embedding([s1[:, :-1], s2[:, :-1]]) + model.time_emb(torch.from_numpy(stamps))
    if patch is not None:
        h[:, -1] += patch[:, 0]
    for i, block in enumerate(model.transformer):
        h = block(h)
        if patch is not None:
            h[:, -1] += patch[:, i + 1]
    ctx = model.norm(h)
    lp1 = torch.log_softmax(model.head(ctx)[:, -1], -1)
    lp2 = torch.log_softmax(model.decode_s2(ctx, s1[:, -1:])[:, -1], -1)
    nll = -lp1.gather(-1, s1[:, -1:])[:, 0] - lp2.gather(-1, s2[:, -1:])[:, 0]
    return nll, tok.decode([s1, s2], half=True)[:, -1], s1, s2


def test_last_token_engine_matches_full_forward(setup):
    tok, model, bars, stamps = setup
    eng = Engine(tok, model, samples=2)
    b = eng.base(bars, stamps)
    nll, decoded, s1, s2 = reference(tok, model, bars, stamps)
    assert torch.allclose(eng.variant(b, predict=False)['nll'], nll, atol=1e-4)
    assert torch.allclose(eng._decode(b, s1[:, -1], s2[:, -1]), decoded, atol=1e-4)


def test_patched_variant_matches_hooked_full_forward(setup):
    tok, model, bars, stamps = setup
    eng = Engine(tok, model, samples=2)
    b = eng.base(bars, stamps)
    patch = 0.5*torch.randn(B, model.n_layers + 1, model.d_model)
    nll = reference(tok, model, bars, stamps, patch)[0]
    got = eng.variant(b, patch, predict=False)['nll']
    assert torch.allclose(got, nll, atol=1e-4)
    assert not torch.allclose(got, eng.variant(b, predict=False)['nll'], atol=1e-4)


def test_zero_patch_is_identity_and_sampling_is_shared(setup):
    tok, model, bars, stamps = setup
    eng = Engine(tok, model, samples=3)
    b = eng.base(bars, stamps)
    a, z = eng.variant(b), eng.variant(b, torch.zeros(B, model.n_layers + 1, model.d_model))
    assert torch.equal(a['pred'], z['pred']) and torch.equal(a['nll'], z['nll'])


def test_scored_bar_does_not_reach_hidden_state_or_forecast(setup):
    tok, model, bars, stamps = setup
    eng = Engine(tok, model, samples=2)
    spiked = bars.copy()
    spiked[:, -1, 1] *= 3
    a, s = eng.base(bars, stamps), eng.base(spiked, stamps)
    assert torch.equal(a.hidden, s.hidden)
    assert torch.equal(eng.variant(a)['pred'], eng.variant(s)['pred'])
    assert (eng.variant(s)['resid_max'] > eng.variant(a)['resid_max']).all()


def test_normalise_is_price_scale_free(setup):
    bars = setup[2]
    a, b = normalise(bars), normalise(bars*np.array([1e6, 1e6, 1e6, 1e6, 1, 1]))
    assert np.allclose(a[0], b[0], atol=1e-3) and np.allclose(a[1], b[1], atol=1e-3)


def make_knob(layers=4, d=32, k=3):
    torch.manual_seed(1)
    m = Pooled(layers, k)
    m.w.data = torch.randn(layers)
    return Knob([2, 7, 11], m.requires_grad_(False), np.zeros((layers, d), np.float32), np.full((layers, d), 2.0, np.float32))


def test_patch_moves_pooled_features_by_the_requested_step():
    knob = make_knob()
    h = torch.randn(6, 4, 32)
    z = knob.pooled(h).requires_grad_(True)
    grad, = torch.autograd.grad(knob.model.mlp(z).sum(), z)
    for direction in ('normal', 'tangent', 'random'):
        p = knob.patch(h, 0.7, direction)
        step = knob.pooled(h + p) - knob.pooled(h)
        assert torch.allclose(step.norm(dim=-1), 0.7/(grad.norm(dim=-1) + GRAD_EPS), atol=1e-5)
        mask = torch.ones(32, dtype=torch.bool)
        mask[knob.dims] = False
        assert (p[:, :, mask] == 0).all()
        cos = torch.nn.functional.cosine_similarity(step, grad, dim=-1)
        if direction == 'normal':
            assert torch.allclose(cos, torch.ones(6), atol=1e-4)
        if direction == 'tangent':
            assert cos.abs().max() < 1e-4


def test_normal_patch_changes_score_in_the_requested_direction():
    knob = make_knob()
    h = torch.randn(64, 4, 32)
    assert (knob.score(h + knob.patch(h, 0.2)) > knob.score(h)).float().mean() > 0.95
    assert (knob.score(h + knob.patch(h, -0.2)) < knob.score(h)).float().mean() > 0.95


def test_discover_recovers_planted_dimensions():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(3000, 3, 20)).astype(np.float32)
    value = rng.normal(size=3000)
    h[:, 2, 5] -= 2*value
    h[:, 2, 13] -= 2*value
    knob, rnd, info = discover(h, value, max_k=4, random_subsets=3)
    assert set(info['top16_dims'][:2]) == {5, 13}
    assert info['topk_bal_acc'][1] > 0.9 > info['random_subsets_bal_acc_mean']
    assert not set(rnd.dims) & set(knob.dims)


def test_tercile_labels_are_balanced_and_low_value_is_class_one():
    v = np.arange(90.0)
    idx, y = tercile_labels(v, np.random.default_rng(0))
    assert y.mean() == 0.5 and v[idx[y == 1]].max() < v[idx[y == 0]].min()


def test_weighted_recall_at_fpr():
    score = np.array([9.0, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    w = np.array([1.0, 10, 1, 10, 10, 10, 10, 10, 10, 10])
    assert recall_at_fpr(score, y, w, 0.125) == 1.0   # threshold at the first negative's score excluded
    assert recall_at_fpr(score, y, w, 0.01) == 0.5
