"""Fixed POT calibration on held-out residuals (not SPOT/online updating)."""

from dataclasses import asdict, dataclass
from typing import Mapping
import warnings

import numpy as np
from scipy.stats import genpareto

from .scoring import ScoredSequence


@dataclass(frozen=True)
class POTThreshold:
    threshold: float
    initial_threshold: float
    shape: float
    scale: float
    n_calibration: int
    n_peaks: int
    initial_quantile: float
    risk: float

    def to_dict(self):
        return asdict(self)

    @classmethod
    def fit(cls, scores, initial_quantile=0.9, risk=0.001, min_peaks=20):
        values = np.asarray(scores, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("POT needs a nonempty finite one-dimensional score sequence.")
        if (values < 0).any():
            raise ValueError("Residual scores must be nonnegative.")
        if not 0 < initial_quantile < 1 or not 0 < risk < 1 - initial_quantile:
            raise ValueError("Need 0 < risk < 1 - initial_quantile < 1.")
        if not isinstance(min_peaks, int) or min_peaks < 3:
            raise ValueError("min_peaks must be an integer >= 3 (20+ recommended).")
        u = float(np.quantile(values, initial_quantile))
        peaks = values[values > u] - u
        if len(peaks) < min_peaks or np.ptp(peaks) == 0:
            raise ValueError(
                f"POT has {len(peaks)} peaks; need >= {min_peaks} nondegenerate peaks. "
                "Use a longer mostly-normal calibration segment or a lower initial quantile."
            )
        tail_mass = len(peaks) / len(values)
        if risk >= tail_mass:
            raise ValueError("risk must be smaller than the observed exceedance fraction.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                shape, _, scale = genpareto.fit(peaks, floc=0)
            except (RuntimeWarning, ValueError, FloatingPointError) as exc:
                raise ValueError("GPD fitting failed; inspect calibration residuals.") from exc
        if not np.isfinite([shape, scale]).all() or scale <= 0:
            raise ValueError("Invalid GPD parameters.")
        if not np.isfinite(genpareto.logpdf(peaks, shape, loc=0, scale=scale)).all():
            raise ValueError("GPD support does not contain the calibration peaks.")
        # P(S > D) ~= (n_peaks/n) * P(Z > D-u). isf handles shape -> 0.
        threshold = u + float(genpareto.isf(risk / tail_mass, shape, loc=0, scale=scale))
        if not np.isfinite(threshold) or threshold <= u:
            raise ValueError("Invalid POT threshold; inspect calibration/tail risk.")
        return cls(threshold, u, float(shape), float(scale), len(values), len(peaks),
                   float(initial_quantile), float(risk))


def fit_thresholds(calibration: ScoredSequence, initial_quantile=0.9,
                   risk=0.01, min_peaks=20, risk_mode="family"):
    """`family` allocates risk/m per feature; `per-feature` uses risk for each.

    Bonferroni controls a nominal union tail probability only if the fitted
    marginal tails are calibrated. Financial dependence/drift may invalidate it.
    """
    if risk_mode not in ("family", "per-feature"):
        raise ValueError("risk_mode must be family or per-feature.")
    if not calibration.features or not 0 < risk < 1:
        raise ValueError("Need scored features and risk in (0, 1).")
    marginal_risk = risk / len(calibration.features) if risk_mode == "family" else risk
    thresholds = {}
    for f in calibration.features:
        try:
            thresholds[f] = POTThreshold.fit(
                calibration.frame[f"score_{f}"], initial_quantile,
                marginal_risk, min_peaks,
            )
        except ValueError as exc:
            raise ValueError(f"POT calibration failed for {f}: {exc}") from exc
    return thresholds


def apply_thresholds(scores: ScoredSequence, thresholds: Mapping[str, POTThreshold]):
    """Frozen thresholds, >= comparisons, dimension labels and logical OR."""
    if set(scores.features) != set(thresholds):
        raise ValueError("Threshold features must match scored features exactly.")
    frame = scores.frame.copy()
    labels = []
    ratios = []
    for f in scores.features:
        threshold = thresholds[f].threshold
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"Invalid threshold for {f}.")
        values = frame[f"score_{f}"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Scores must be finite and nonnegative.")
        frame[f"threshold_{f}"] = threshold
        frame[f"is_anomaly_{f}"] = (values >= threshold).astype(int)
        labels.append(frame[f"is_anomaly_{f}"].to_numpy(dtype=bool))
        ratios.append(values / threshold)
    matrix = np.stack(labels, axis=1)
    frame["anomaly_score"] = np.max(np.stack(ratios, axis=1), axis=1)
    frame["is_anomaly"] = matrix.any(axis=1).astype(int)
    frame["anomaly_dimensions"] = [
        ",".join(f for f, flagged in zip(scores.features, row) if flagged)
        for row in matrix
    ]
    return frame
