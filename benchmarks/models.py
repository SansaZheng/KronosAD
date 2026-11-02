"""Feature autoencoder adaptation, NOT the upstream OHLC forecasting pipeline.

Each feature vector becomes ONE continuous token. A narrow trainable encoder
feeds the frozen Kronos Transformer blocks/norm and a trainable linear decoder.
The tokenizer, token embeddings, timestamps and generation heads are bypassed.
At sequence length one attention has no temporal modelling capability. This is
a transfer experiment, not evidence of Kronos time-series forecasting quality.
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np


class Preprocessor:
    """Train-only medians/scales, active dimensions and missingness indicators."""
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.keep = np.isfinite(x).any(0)
        x = x[:, self.keep]
        self.missing = np.isnan(x).any(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.center = np.nanmedian(x, axis=0)
            q1, q3 = np.nanquantile(x, [.25, .75], axis=0)
            std = np.nanstd(x, axis=0)
        self.scale = (q3-q1)/1.349
        self.scale = np.where(self.scale > 1e-8, self.scale, np.where(std > 1e-8, std, 1.))
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)[:, self.keep]
        missing = np.isnan(x)
        z = (np.where(missing, self.center, x)-self.center)/self.scale
        return np.column_stack([z, missing[:, self.missing]]).astype(np.float32)

    def save(self, path):
        np.savez(path, keep=self.keep, missing=self.missing, center=self.center, scale=self.scale)


def make_adapter(features, weights, seed, random_backbone=False):
    import torch
    from torch import nn
    from model.kronos import Kronos

    torch.manual_seed(seed)
    config = json.loads((Path(weights) / 'config.json').read_text())
    backbone = Kronos(**config) if random_backbone else Kronos.from_pretrained(str(weights), local_files_only=True)
    for p in backbone.parameters():
        p.requires_grad_(False)
    # Same adapter initialization for pretrained/random controls within a seed.
    torch.manual_seed(seed + 10000)

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.encoder = nn.Sequential(nn.Linear(features, min(8, max(1, features//2))),
                                         nn.Tanh(),
                                         nn.Linear(min(8, max(1, features//2)), backbone.d_model))
            self.decoder = nn.Linear(backbone.d_model, features)
            self.backbone.eval()

        def train(self, mode=True):
            super().train(mode)
            self.backbone.eval()  # Freeze dropout as well as parameters.
            return self

        def forward(self, z):
            h = self.encoder(z.clamp(-8, 8)).unsqueeze(1)
            for layer in self.backbone.transformer:
                h = layer(h)
            return self.decoder(self.backbone.norm(h)[:, 0])

    return Adapter()


class AdapterDetector:
    def __init__(self, weights, seed, random_backbone=False, device='cpu', steps=400):
        self.weights, self.seed, self.random_backbone = weights, seed, random_backbone
        self.device, self.steps = device, steps

    def fit(self, z):
        import torch
        self.model = make_adapter(z.shape[1], self.weights, self.seed, self.random_backbone).to(self.device)
        self.model.train()
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                     lr=1e-3, weight_decay=1e-4)
        rng = np.random.default_rng(self.seed)
        batch_size = min(256, len(z))
        self.losses = []
        for step in range(self.steps):
            indices = rng.integers(0, len(z), size=batch_size)
            batch = torch.as_tensor(z[indices], device=self.device)
            prediction = self.model(batch)
            # Robust unsupervised objective; raw test residuals stay unclipped.
            loss = torch.nn.functional.smooth_l1_loss(prediction, batch.clamp(-8, 8))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.)
            optimizer.step()
            if step % 100 == 0 or step == self.steps-1:
                self.losses.append({'step': step+1, 'loss': float(loss.detach().cpu())})
        self.model.eval()
        self.frozen_parameters = sum(p.numel() for p in self.model.backbone.parameters())
        self.trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return self

    def score(self, z):
        import torch
        with torch.inference_mode():
            out = self.model(torch.as_tensor(z, device=self.device)).cpu().numpy()
        return np.mean((z.astype(np.float64)-out)**2, axis=1)

    def save(self, path):
        import torch
        # No training rows, targets, transaction IDs or identity information.
        torch.save({'encoder': self.model.encoder.state_dict(), 'decoder': self.model.decoder.state_dict(),
                    'seed': self.seed, 'random_backbone': self.random_backbone}, path)
