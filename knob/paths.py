"""Sampled multi-step forecast paths with key/value caching.

One full forward per window builds the caches; every further step and sample only computes the
newly appended position.  Matches the official full forward (see tests).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .engine import BlockCache, _rot, _split, normalise


def _step(cache, block, x, pos, ek, ev):
    """x [N, B, d] token state at ``pos``; ek/ev [N, B, h, E, hd] rotated keys/values of the positions
    between the cache and ``pos``.  Returns the block output and this position's rotated key and value."""
    a = block.self_attn
    n = block.norm1(x)
    q = _rot(_split(a.q_proj(n), a.n_heads), pos, a.rotary.inv_freq)
    k = _rot(_split(a.k_proj(n), a.n_heads), pos, a.rotary.inv_freq)
    v = _split(a.v_proj(n), a.n_heads)
    keys = [torch.einsum('bhpd,nbhd->nbhp', cache.k, q)]
    if ek is not None:
        keys.append(torch.einsum('nbhed,nbhd->nbhe', ek, q))
    keys.append((q*k).sum(-1, keepdim=True))
    w = (torch.cat(keys, -1)/math.sqrt(a.head_dim)).softmax(-1)
    P = cache.k.shape[2]
    o = torch.einsum('nbhp,bhpd->nbhd', w[..., :P], cache.v) + w[..., -1:]*v
    if ek is not None:
        o = o + torch.einsum('nbhe,nbhed->nbhd', w[..., P:-1], ev)
    x = x + a.out_proj(o.reshape(*x.shape))
    return x + block.ffn(block.norm2(x)), k, v


def _append(old, new):
    new = new.unsqueeze(3)
    return new if old is None else torch.cat([old, new], 3)


class PathSampler:
    def __init__(self, tokenizer, model, temperature=0.6, top_p=0.9, samples=20, horizon=6, seed=0):
        self.tok, self.model = tokenizer, model
        self.T, self.top_p, self.N, self.H = temperature, top_p, samples, horizon
        self.gen = torch.Generator().manual_seed(seed)

    def _sample(self, logits, forced):
        if forced is not None:
            return forced
        logits = logits/self.T
        srt, idx = torch.sort(logits, descending=True, dim=-1)
        drop = torch.cumsum(F.softmax(srt, -1), -1) > self.top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        logits = logits.masked_fill(torch.zeros_like(drop).scatter(-1, idx, drop), -float('inf'))
        g = -torch.log(-torch.log(torch.rand(logits.shape, generator=self.gen).clamp_min(1e-12)))
        return (logits + g).argmax(-1)

    def _s2(self, dep_k, dep_v, ek, ev, ctx, s1):
        model = self.model
        ca = model.dep_layer.cross_attn
        q = _split(ca.q_proj(model.embedding.emb_s1(s1)), ca.n_heads)
        k, v = _split(ca.k_proj(ctx), ca.n_heads), _split(ca.v_proj(ctx), ca.n_heads)
        keys = [torch.einsum('bhpd,nbhd->nbhp', dep_k, q)]
        if ek is not None:
            keys.append(torch.einsum('nbhed,nbhd->nbhe', ek, q))
        keys.append((q*k).sum(-1, keepdim=True))
        w = (torch.cat(keys, -1)/math.sqrt(ca.head_dim)).softmax(-1)
        P = dep_k.shape[2]
        o = torch.einsum('nbhp,bhpd->nbhd', w[..., :P], dep_v) + w[..., -1:]*v
        if ek is not None:
            o = o + torch.einsum('nbhe,nbhed->nbhd', w[..., P:-1], ev)
        logits = model.head.cond_forward(model.dep_layer.norm(ctx + ca.out_proj(o.reshape(*ctx.shape))))
        return logits, k, v

    @torch.no_grad()
    def __call__(self, context, stamps, future_stamps, forced=None, return_logits=False):
        """context [B, L, 6] clipped normalised bars; stamps [B, L, 5]; future_stamps [B, H-1, 5].

        Returns sampled normalised paths [N, B, H, 6].  ``forced`` = (s1, s2) of shape [H, N, B]
        replaces sampling (used to verify against the official forward).
        """
        tok, model, N = self.tok, self.model, self.N
        s1, s2 = tok.encode(torch.as_tensor(context), half=True)
        B, L = s1.shape
        h = model.embedding([s1, s2]) + model.time_emb(torch.as_tensor(stamps))
        caches, state = [], []
        for block in model.transformer:
            caches.append(BlockCache(block, h[:, :-1]))
            state.append(h[:, -1])
            h = block(h)
        ctx_all = model.norm(h)
        ca = model.dep_layer.cross_attn
        dep_k = _split(ca.k_proj(ctx_all[:, :-1]), ca.n_heads).transpose(1, 2)
        dep_v = _split(ca.v_proj(ctx_all[:, :-1]), ca.n_heads).transpose(1, 2)
        z = tok.post_quant_embed(tok.indices_to_bits([s1, s2], half=True))
        dec = []
        for block in tok.decoder:
            dec.append(BlockCache(block, z))
            z = block(z)
        ek = [None]*len(caches)
        ev = [None]*len(caches)
        dk = dv = None
        gk, gv = [None]*len(dec), [None]*len(dec)
        x = state[0].expand(N, B, -1)
        out, logs = [], []
        future = model.time_emb(torch.as_tensor(future_stamps)) if self.H > 1 else None
        for j in range(self.H):
            pos = L - 1 + j
            for i, block in enumerate(model.transformer):
                x, k, v = _step(caches[i], block, x, pos, ek[i], ev[i])
                ek[i], ev[i] = _append(ek[i], k), _append(ev[i], v)
            ctx = model.norm(x)
            lg1 = model.head(ctx)
            a1 = self._sample(lg1, None if forced is None else forced[0][j])
            lg2, k, v = self._s2(dep_k, dep_v, dk, dv, ctx, a1)
            dk, dv = _append(dk, k), _append(dv, v)
            a2 = self._sample(lg2, None if forced is None else forced[1][j])
            logs.append((lg1, lg2))
            zt = tok.post_quant_embed(tok.indices_to_bits([a1, a2], half=True))
            for i, block in enumerate(tok.decoder):
                zt, k, v = _step(dec[i], block, zt, L + j, gk[i], gv[i])
                gk[i], gv[i] = _append(gk[i], k), _append(gv[i], v)
            out.append(tok.head(zt))
            if j + 1 < self.H:
                emb = model.embedding([a1.reshape(-1, 1), a2.reshape(-1, 1)]).reshape(N, B, -1)
                x = emb + future[:, j]
        paths = torch.stack(out, 2)
        return (paths, logs) if return_logits else paths
