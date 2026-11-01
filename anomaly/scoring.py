"""Causal, one-step prediction residuals in original observation units."""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

PRICE_COLUMNS = ("open", "high", "low", "close")
FEATURE_COLUMNS = PRICE_COLUMNS + ("volume", "amount")


@dataclass
class ScoredSequence:
    frame: pd.DataFrame
    features: tuple


def validate_frame(df: pd.DataFrame, timestamp: str = "timestamps") -> pd.DataFrame:
    """Reject ambiguous timestamps and missing/nonfinite observed features."""
    if timestamp not in df or not set(PRICE_COLUMNS).issubset(df.columns):
        raise ValueError(f"CSV needs {timestamp!r} and {PRICE_COLUMNS} columns.")
    result = df.copy().reset_index(drop=True)
    result[timestamp] = pd.to_datetime(result[timestamp], errors="raise")
    times = result[timestamp]
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("Timestamps must be nonmissing, unique and strictly increasing.")
    observed = [f for f in FEATURE_COLUMNS if f in result]
    result[observed] = result[observed].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(result[observed].to_numpy(dtype=float)).all():
        raise ValueError("Observed features must be finite; split/clean missing bars first.")
    return result


def rolling_scores(
    df: pd.DataFrame,
    predictor,
    lookback: int = 64,
    timestamp: str = "timestamps",
    features: Optional[Sequence[str]] = None,
    epsilon: Optional[Mapping[str, float]] = None,
    sample_count: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.9,
    start: Optional[int] = None,
    progress: bool = False,
) -> ScoredSequence:
    """Score t using only df[t-lookback:t]; the target is never clipped.

    Optional volume/amount may be filled for the model, but only originally
    observed columns can be scored. No labels are inspected. Rows before start
    are context only. `predictor` follows the upstream KronosPredictor API.
    """
    if not isinstance(lookback, int) or lookback < 2:
        raise ValueError("lookback must be an integer >= 2.")
    if lookback > predictor.max_context:
        raise ValueError("lookback exceeds predictor.max_context; silent truncation is forbidden.")
    if not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer.")
    if not np.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive and top_p in (0, 1].")
    data = validate_frame(df, timestamp)
    observed = tuple(f for f in FEATURE_COLUMNS if f in data)
    selected = observed if features is None else tuple(features)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(observed):
        raise ValueError("Scored features must be unique, originally observed OHLCVA columns.")
    eps = {f: 1e-5 for f in selected}
    if epsilon is not None:
        if not set(epsilon) <= set(selected):
            raise ValueError("epsilon contains an unscored feature.")
        eps.update(epsilon)
    if any(not np.isfinite(v) or v <= 0 for v in eps.values()):
        raise ValueError("Every epsilon must be finite and positive.")
    first = lookback if start is None else start
    if not isinstance(first, int) or first < lookback or first >= len(data):
        raise ValueError("start must leave lookback history and at least one target.")

    # Pass explicit model columns to avoid upstream replacing observed amount
    # when volume is absent. Fabricated columns never enter `selected`.
    model_data = data[list(observed)].copy()
    if "volume" not in model_data:
        model_data["volume"] = 0.0
    if "amount" not in model_data:
        model_data["amount"] = (
            model_data["volume"] * model_data[list(PRICE_COLUMNS)].mean(axis=1)
        )
    model_data = model_data[list(FEATURE_COLUMNS)]
    positions = range(first, len(data))
    if progress:
        from tqdm import tqdm
        positions = tqdm(positions, desc="Rolling one-step forecasts")
    records = []
    for t in positions:
        history = model_data.iloc[t - lookback:t].reset_index(drop=True).copy()
        history_times = data[timestamp].iloc[t - lookback:t].reset_index(drop=True)
        target_times = data[timestamp].iloc[t:t + 1].reset_index(drop=True)
        forecast = predictor.predict(
            df=history,
            x_timestamp=history_times,
            y_timestamp=target_times,
            pred_len=1,
            sample_count=sample_count,
            T=temperature,
            top_p=top_p,
            verbose=False,
        )
        if len(forecast) != 1 or not set(selected) <= set(forecast.columns):
            raise ValueError("Predictor must return one row with all scored features.")
        prediction = forecast[list(selected)].iloc[0].to_numpy(dtype=float)
        if not np.isfinite(prediction).all():
            raise ValueError(f"Nonfinite forecast at row {t}.")
        actual = data[list(selected)].iloc[t].to_numpy(dtype=float)
        # Population std agrees with Kronos preprocessing; use original history,
        # not a clipped/normalized target or future-dependent statistics.
        scale = data[list(selected)].iloc[t - lookback:t].std(ddof=0).to_numpy()
        denominator = scale + np.array([eps[f] for f in selected])
        with np.errstate(over="ignore", invalid="ignore"):
            scores = np.square((actual - prediction) / denominator)
        if not np.isfinite(scores).all():
            raise ValueError(f"Residual score overflow at row {t}; check units/epsilon.")
        record = {"row": t, "timestamp": data[timestamp].iloc[t]}
        for i, f in enumerate(selected):
            record.update({
                f"observed_{f}": actual[i],
                f"predicted_{f}": prediction[i],
                f"scale_{f}": denominator[i],
                f"score_{f}": scores[i],
            })
        records.append(record)
    return ScoredSequence(pd.DataFrame(records), selected)
