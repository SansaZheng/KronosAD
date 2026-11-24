"""Exact last-token engine for Kronos one-step scoring with hidden-state patches.

A patch only touches the hidden state of the last context token.  Because every attention in the
predictor and in the tokenizer decoder is causal, all earlier positions are unchanged, so one
full ("base") forward per window is cached and every patched variant recomputes a single position.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from model import Kronos, KronosTokenizer

FEATURES = ['open', 'high', 'low', 'close', 'volume', 'amount']
CLIP = 5.0
EPS = 1e-5


def load_models(weights):
    weights = Path(weights)
    out = []
    for cls, name in ((KronosTokenizer, 'Kronos-Tokenizer-2k'), (Kronos, 'Kronos-mini')):
        m = cls(**json.loads((weights/name/'config.json').read_text()))
        m.load_state_dict(load_file(str(weights/name/'model.safetensors')), strict=True)
        out.append(m.eval().requires_grad_(False))
    return out


def window_ref(ctx):
    """[B, 1, 6] divisors: window mean close for OHLC, window mean volume / amount (1 if zero)."""
    ref = np.ones((len(ctx), 1, 6))
    price = ctx[:, :, 3].mean(1)
    ref[:, 0, :4] = np.where(price > 0, price, 1.0)[:, None]
    for j in (4, 5):
        m = ctx[:, :, j].mean(1)
        ref[:, 0, j] = np.where(m > 0, m, 1.0)
    return ref


def normalise(bars):
    """bars: [B, L+1, 6] raw OHLCVA, the last row is the bar to score.

    Prices are divided by the window mean close and volume/amount by their window means first, so
    BTC-quoted prices (~1e-7) are not flattened by the official ``std + 1e-5`` normalisation.  Then
    the official KronosPredictor normalisation is applied with statistics of the L context bars only.
    Returns clipped model input [B, L+1, 6], the unclipped normalised target [B, 6] and the
    (mean, std + eps) of the rescaled context, which map normalised values back to price-relative units.
    """
    x = np.asarray(bars, dtype=np.float64)
    ref = window_ref(x[:, :-1])
    x = x/ref
    mean, std = x[:, :-1].mean(1, keepdims=True), x[:, :-1].std(1, keepdims=True)
    xn = (x - mean)/(std + EPS)
    return (np.clip(xn, -CLIP, CLIP).astype(np.float32), xn[:, -1].astype(np.float32),
            mean[:, 0].astype(np.float32), (std[:, 0] + EPS).astype(np.float32))


def _rot(x, pos, inv_freq):
    ang = pos*inv_freq
    ang = torch.cat([ang, ang])
    x1, x2 = x.chunk(2, dim=-1)
    return x*ang.cos() + torch.cat([-x2, x1], dim=-1)*ang.sin()


def _split(x, h):
    return x.view(*x.shape[:-1], h, x.shape[-1]//h)


class BlockCache:
    """Rotated keys and values of the cached positions of one causal TransformerBlock."""

    def __init__(self, block, x):
        a = block.self_attn
        n = block.norm1(x)
        k = _split(a.k_proj(n), a.n_heads).transpose(1, 2)
        v = _split(a.v_proj(n), a.n_heads).transpose(1, 2)
        pos = torch.arange(x.shape[1], dtype=torch.float32)[:, None]*a.rotary.inv_freq[None]
        ang = torch.cat([pos, pos], -1)[None, None]
        k1, k2 = k.chunk(2, dim=-1)
        self.k = k*ang.cos() + torch.cat([-k2, k1], dim=-1)*ang.sin()
        self.v = v

    def last(self, block, x, pos):
        """x: [..., B, d] hidden state of the token at position ``pos`` (after the cached ones)."""
        a = block.self_attn
        n = block.norm1(x)
        q = _rot(_split(a.q_proj(n), a.n_heads), pos, a.rotary.inv_freq)
        k = _rot(_split(a.k_proj(n), a.n_heads), pos, a.rotary.inv_freq)
        v = _split(a.v_proj(n), a.n_heads)
        s = torch.einsum('bhpd,...bhd->...bhp', self.k, q)
        s = torch.cat([s, (q*k).sum(-1, keepdim=True)], -1)/math.sqrt(a.head_dim)
        w = s.softmax(-1)
        o = torch.einsum('...bhp,bhpd->...bhd', w[..., :-1], self.v) + w[..., -1:]*v
        x = x + a.out_proj(o.reshape(*x.shape))
        return x + block.ffn(block.norm2(x))


class Base:
    """Cached base forward of a batch of windows."""


class Engine:
    def __init__(self, tokenizer, model, temperature=0.6, top_p=0.9, samples=8, seed=0):
        self.tok, self.model = tokenizer, model
        self.T, self.top_p, self.samples, self.seed = temperature, top_p, samples, seed
        self.n_layers = len(model.transformer)

    @torch.no_grad()
    def base(self, bars, stamps):
        """bars [B, L+1, 6] raw, stamps [B, L, 5] (minute, hour, weekday, day, month) of the context."""
        tok, model = self.tok, self.model
        x, target, mean, scale = normalise(bars)
        b = Base()
        b.mean, b.scale = torch.from_numpy(mean), torch.from_numpy(scale)
        b.target = torch.from_numpy(target)
        b.last_close = torch.from_numpy(x[:, -2, 3].copy())
        s1, s2 = tok.encode(torch.from_numpy(x), half=True)
        b.s1_true, b.s2_true = s1[:, -1], s2[:, -1]
        s1, s2 = s1[:, :-1], s2[:, :-1]
        L = s1.shape[1]
        h = model.embedding([s1, s2]) + model.time_emb(torch.as_tensor(stamps))
        b.hidden = [h[:, -1].clone()]          # layer 0 = embedding output, then one per block
        b.caches = []
        for block in model.transformer:
            b.caches.append(BlockCache(block, h[:, :-1]))
            h = block(h)
            b.hidden.append(h[:, -1].clone())
        b.hidden = torch.stack(b.hidden, 1)    # [B, n_layers+1, d]
        ctx = model.norm(h)
        ca = model.dep_layer.cross_attn
        b.dep_k = _split(ca.k_proj(ctx[:, :-1]), ca.n_heads).transpose(1, 2)
        b.dep_v = _split(ca.v_proj(ctx[:, :-1]), ca.n_heads).transpose(1, 2)
        z = tok.post_quant_embed(tok.indices_to_bits([s1, s2], half=True))
        b.dec = []
        for block in tok.decoder:
            b.dec.append(BlockCache(block, z))
            z = block(z)
        b.L = L
        g = torch.Generator().manual_seed(self.seed)
        b.gumbel = -torch.log(-torch.log(torch.rand(self.samples, 2, len(x), model.s1_vocab_size, generator=g).clamp_min(1e-12)))
        return b

    def _final(self, b, patch):
        """patch: [..., B, n_layers+1, d] or None -> last-token state after the final block."""
        h = b.hidden[:, 0] if patch is None else b.hidden[:, 0] + patch[..., 0, :]
        for i, block in enumerate(self.model.transformer):
            h = b.caches[i].last(block, h, b.L - 1)
            if patch is not None:
                h = h + patch[..., i + 1, :]
        return h

    def _s2_logits(self, b, ctx, s1):
        model = self.model
        ca = model.dep_layer.cross_attn
        q = _split(ca.q_proj(model.embedding.emb_s1(s1)), ca.n_heads)
        k, v = _split(ca.k_proj(ctx), ca.n_heads), _split(ca.v_proj(ctx), ca.n_heads)
        s = torch.einsum('bhpd,...bhd->...bhp', b.dep_k, q)
        s = torch.cat([s, (q*k).sum(-1, keepdim=True)], -1)/math.sqrt(ca.head_dim)
        w = s.softmax(-1)
        o = torch.einsum('...bhp,bhpd->...bhd', w[..., :-1], b.dep_v) + w[..., -1:]*v
        return model.head.cond_forward(model.dep_layer.norm(ctx + ca.out_proj(o.reshape(*ctx.shape))))

    def _decode(self, b, s1, s2):
        tok = self.tok
        z = tok.post_quant_embed(tok.indices_to_bits([s1, s2], half=True))
        for i, block in enumerate(tok.decoder):
            z = b.dec[i].last(block, z, b.L)
        return tok.head(z)

    def _nucleus(self, logits):
        logits = logits/self.T
        srt, idx = torch.sort(logits, descending=True, dim=-1)
        drop = torch.cumsum(F.softmax(srt, -1), -1) > self.top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        return logits.masked_fill(torch.zeros_like(drop).scatter(-1, idx, drop), -float('inf'))

    @torch.no_grad()
    def variant(self, b, patch=None, predict=True):
        """Scores of one patched variant.  All outputs are per window."""
        model = self.model
        ctx = model.norm(self._final(b, patch))
        lg1 = model.head(ctx)
        lp1 = F.log_softmax(lg1, -1)
        out = {'nll_s1': -lp1.gather(-1, b.s1_true[:, None])[:, 0],
               'entropy_s1': -(lp1.exp()*lp1).sum(-1)}
        lp2 = F.log_softmax(self._s2_logits(b, ctx, b.s1_true), -1)
        out['nll_s2'] = -lp2.gather(-1, b.s2_true[:, None])[:, 0]
        out['nll'] = out['nll_s1'] + out['nll_s2']
        if predict:
            f1 = self._nucleus(lg1)
            preds = []
            for s in range(self.samples):      # common random numbers across variants
                a1 = (f1 + b.gumbel[s, 0]).argmax(-1)
                a2 = (self._nucleus(self._s2_logits(b, ctx, a1)) + b.gumbel[s, 1]).argmax(-1)
                preds.append(self._decode(b, a1, a2))
            preds = torch.stack(preds)
            out['pred'] = preds.mean(0)
            out['pred_std'] = preds.std(0)
            err = (b.target - out['pred'])**2
            out['resid_max'], out['resid_mean'] = err.max(-1).values, err.mean(-1)
            out['resid_price'] = err[:, :4].max(-1).values
            out['jump'] = (out['pred'][:, 3] - b.last_close).abs()
            rel = out['pred']*b.scale + b.mean          # price-relative units (window mean close = 1)
            out['range'] = rel[:, 1] - rel[:, 2]
        return out
