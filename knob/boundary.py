"""Boundary probe, knob-dimension selection and steering patches (after Huang et al., EMNLP 2026).

Differences from the paper, all forced by the setting: the boundary label is label-free (terciles of
a per-window quantity, not answer correctness); the full-state reference probe is logistic
regression instead of a multi-view CNN; the distilled predictor is a small tanh MLP instead of a KAN.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

GRAD_EPS = 0.05


def tercile_labels(value, rng):
    """1 for the lowest third, 0 for the highest third; the middle third is unused.  Balanced."""
    lo, hi = np.quantile(value, [1/3, 2/3])
    one, zero = np.flatnonzero(value <= lo), np.flatnonzero(value >= hi)
    n = min(len(one), len(zero))
    idx = np.concatenate([rng.choice(one, n, replace=False), rng.choice(zero, n, replace=False)])
    y = np.concatenate([np.ones(n, np.int64), np.zeros(n, np.int64)])
    p = rng.permutation(len(idx))
    return idx[p], y[p]


class Pooled(torch.nn.Module):
    """z = sum_l softmax(w)_l * H[l, J];  score = MLP(z)."""

    def __init__(self, layers, k, hidden=8):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(layers))
        self.mlp = torch.nn.Sequential(torch.nn.Linear(k, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))

    def alpha(self):
        return self.w.softmax(0)

    def pool(self, h):
        return torch.einsum('l,...lk->...k', self.alpha(), h)

    def forward(self, h):
        return self.mlp(self.pool(h))[..., 0]


def fit_pooled(h, y, seed, steps=600):
    torch.manual_seed(seed)
    m = Pooled(h.shape[1], h.shape[2])
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    h, y = torch.as_tensor(h), torch.as_tensor(y, dtype=torch.float32)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(m(h), y).backward()
        opt.step()
    return m.requires_grad_(False)


def accuracy(m, h, y):
    with torch.no_grad():
        return float(balanced_accuracy_score(y, (m(torch.as_tensor(h)) > 0).numpy()))


class Knob:
    """A fitted knob: dimensions J, layer weights, score function and standardisation."""

    def __init__(self, dims, model, mean, std, seed=0):
        self.dims, self.model = np.asarray(dims), model
        self.mean, self.std = torch.as_tensor(mean), torch.as_tensor(std)
        g = torch.Generator().manual_seed(seed)
        self.fixed = torch.randn(2, len(self.dims), generator=g)   # tangent seed, random direction

    def pooled(self, hidden):
        hs = (hidden[:, :, self.dims] - self.mean[:, self.dims])/self.std[:, self.dims]
        return self.model.pool(hs)

    def score(self, hidden):
        with torch.no_grad():
            return self.model.mlp(self.pooled(hidden))[..., 0]

    def patch(self, hidden, tau, direction='normal'):
        """Minimum-norm layer-wise patch that moves the pooled knob features by tau*d/(|grad|+eps).

        hidden: [B, layers, d] last-token states.  Positive tau moves towards class 1.
        """
        z = self.pooled(hidden).detach().requires_grad_(True)
        with torch.enable_grad():
            grad, = torch.autograd.grad(self.model.mlp(z).sum(), z)
        norm = grad.norm(dim=-1, keepdim=True)
        n = grad/(norm + 1e-12)
        if direction == 'normal':
            d = n
        elif direction == 'tangent':
            t = self.fixed[0].expand_as(n)
            t = t - (t*n).sum(-1, keepdim=True)*n
            d = t/(t.norm(dim=-1, keepdim=True) + 1e-12)
        elif direction == 'random':
            d = (self.fixed[1]/self.fixed[1].norm()).expand_as(n)
        else:
            raise ValueError(direction)
        dz = tau*d/(norm + GRAD_EPS)
        a = self.model.alpha()
        per_layer = (a/(a**2).sum())[None, :, None]*dz[:, None, :]*self.std[:, self.dims][None]
        out = torch.zeros_like(hidden)
        out[:, :, self.dims] = per_layer.to(hidden.dtype)
        return out


def discover(hidden, value, seed=0, max_k=16, delta=0.01, random_subsets=30):
    """hidden [N, layers, d] last-token states; value: per-window quantity defining the boundary."""
    rng = np.random.default_rng(seed)
    idx, y = tercile_labels(value, rng)
    cut = int(0.8*len(idx))
    tr, te = idx[:cut], idx[cut:]
    ytr, yte = y[:cut], y[cut:]
    mean, std = hidden[tr].mean(0), hidden[tr].std(0) + 1e-6
    hs = (hidden - mean)/std
    full = LogisticRegression(C=0.1, max_iter=300).fit(hs[tr].reshape(len(tr), -1), ytr)
    acc_full = float(balanced_accuracy_score(yte, full.predict(hs[te].reshape(len(te), -1))))
    delta_a = hs[tr][ytr == 1].mean(0) - hs[tr][ytr == 0].mean(0)
    score = np.abs(delta_a).sum(0)
    order = np.argsort(-score)
    late = float(np.abs(delta_a)[delta_a.shape[0]//2 + 1:].sum()/np.abs(delta_a).sum())
    curve, models = [], {}
    for k in range(1, max_k + 1):
        dims = order[:k]
        m = fit_pooled(hs[tr][:, :, dims], ytr, seed)
        models[k] = m
        curve.append(accuracy(m, hs[te][:, :, dims], yte))
    reached = [k for k, a in enumerate(curve, 1) if a >= acc_full - delta]
    k = reached[0] if reached else int(np.argmax(curve)) + 1
    rand = []
    for i in range(random_subsets):
        dims = rng.choice(hidden.shape[2], k, replace=False)
        rand.append(accuracy(fit_pooled(hs[tr][:, :, dims], ytr, seed), hs[te][:, :, dims], yte))
    rdims = np.sort(rng.choice(np.setdiff1d(np.arange(hidden.shape[2]), order[:k]), k, replace=False))
    rmodel = fit_pooled(hs[tr][:, :, rdims], ytr, seed)
    info = {'train': int(len(tr)), 'test': int(len(te)), 'full_probe_bal_acc': acc_full,
            'topk_bal_acc': curve, 'k': int(k), 'reached_full_minus_delta': bool(reached), 'delta': delta,
            'dims': [int(j) for j in order[:k]], 'top16_dims': [int(j) for j in order[:16]],
            'dim_scores': [float(score[j]) for j in order[:16]], 'late_ratio': late,
            'layer_weights': [float(a) for a in models[k].alpha()],
            'random_subsets_bal_acc_mean': float(np.mean(rand)), 'random_subsets_bal_acc_max': float(np.max(rand)),
            'random_dims': [int(j) for j in rdims], 'random_dims_bal_acc': accuracy(rmodel, hs[te][:, :, rdims], yte)}
    return Knob(order[:k], models[k], mean, std, seed), Knob(rdims, rmodel, mean, std, seed), info


class Dense:
    """Reference steering without dimension selection: shift the final-layer state along the
    difference of class means over all dimensions.  tau = 4 equals the full gap between the means."""

    def __init__(self, hidden, value, seed=0):
        idx, y = tercile_labels(value, np.random.default_rng(seed))
        cut = int(0.8*len(idx))
        h = hidden[idx[:cut], -1]
        self.delta = torch.as_tensor(h[y[:cut] == 1].mean(0) - h[y[:cut] == 0].mean(0))

    def patch(self, hidden, tau, direction='normal'):
        out = torch.zeros_like(hidden)
        out[:, -1] = 0.25*tau*self.delta
        return out
