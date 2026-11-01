"""Offline tests for causal scoring and POT; no model downloads needed."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from anomaly import ScoredSequence, POTThreshold, apply_thresholds, fit_thresholds, rolling_scores
from anomaly.scoring import FEATURE_COLUMNS, validate_frame
from anomaly.__main__ import build_parser, parse_epsilon, run


class LastValuePredictor:
    max_context = 2048

    def __init__(self):
        self.calls = []

    def predict(self, df, x_timestamp, y_timestamp, **kwargs):
        self.calls.append((df.copy(), x_timestamp.copy(), y_timestamp.copy(), kwargs))
        return df.iloc[[-1]].reset_index(drop=True)


def frame(n=20, optional=True):
    rng = np.random.default_rng(12)
    base = 100 + np.cumsum(rng.normal(size=n))
    data = pd.DataFrame({"timestamps": pd.date_range("2025-01-01", periods=n, freq="5min")})
    for i, f in enumerate(FEATURE_COLUMNS):
        if optional or i < 4:
            data[f] = base + i
    return data


def test_score_formula_history_only_and_unclipped_target():
    data = frame(8)
    data.loc[4, "close"] = 10000
    predictor = LastValuePredictor()
    scores = rolling_scores(data, predictor, lookback=4, epsilon={"close": 0.01})
    expected = ((10000 - data.close.iloc[3]) / (data.close.iloc[:4].std(ddof=0) + 0.01)) ** 2
    assert scores.frame.score_close.iloc[0] == pytest.approx(expected)
    assert scores.frame.score_close.iloc[0] > 25  # no +/-5 clipping of the target
    for j, (history, xt, yt, kwargs) in enumerate(predictor.calls):
        t = 4 + j
        np.testing.assert_array_equal(history.close, data.close.iloc[t-4:t])
        assert xt.max() < yt.iloc[0] == data.timestamps.iloc[t]
        assert kwargs["pred_len"] == 1
        assert kwargs["sample_count"] == 1


def test_future_changes_cannot_change_earlier_scores_or_forecasts():
    original = frame(30)
    modified = original.copy()
    modified.loc[20:, "close"] += 9999
    first = rolling_scores(original, LastValuePredictor(), lookback=5)
    second = rolling_scores(modified, LastValuePredictor(), lookback=5)
    pd.testing.assert_frame_equal(first.frame.iloc[:15], second.frame.iloc[:15])


def test_optional_columns_not_scored_and_observed_amount_preserved():
    data = frame(optional=False)
    scores = rolling_scores(data, LastValuePredictor(), lookback=5)
    assert scores.features == FEATURE_COLUMNS[:4]
    assert "score_volume" not in scores.frame and "score_amount" not in scores.frame
    with pytest.raises(ValueError, match="originally observed"):
        rolling_scores(data, LastValuePredictor(), 5, features=["volume"])
    data["amount"] = np.arange(len(data)) + 500
    predictor = LastValuePredictor()
    scores = rolling_scores(data, predictor, lookback=5)
    assert "amount" in scores.features and "volume" not in scores.features
    np.testing.assert_array_equal(predictor.calls[0][0].amount, data.amount.iloc[:5])


@pytest.mark.parametrize("damage", ["duplicate", "reverse", "nan", "inf", "nat"])
def test_invalid_observations_rejected(damage):
    data = frame()
    if damage == "duplicate":
        data.loc[1, "timestamps"] = data.loc[0, "timestamps"]
    elif damage == "reverse":
        data = data.iloc[::-1]
    elif damage == "nat":
        data.loc[0, "timestamps"] = pd.NaT
    else:
        data.loc[0, "close"] = np.nan if damage == "nan" else np.inf
    with pytest.raises(ValueError):
        validate_frame(data)


def test_zero_variance_is_finite_and_context_limit_enforced():
    data = frame()
    data["close"] = 1.0
    result = rolling_scores(data, LastValuePredictor(), 5)
    assert (result.frame.score_close == 0).all()
    predictor = LastValuePredictor()
    predictor.max_context = 4
    with pytest.raises(ValueError, match="truncation"):
        rolling_scores(data, predictor, 5)


def test_start_uses_reference_history_without_scoring_it():
    result = rolling_scores(frame(), LastValuePredictor(), 5, start=12)
    assert result.frame.row.tolist() == list(range(12, 20))


def test_pot_matches_exponential_tail_formula_and_fit_uses_only_excesses():
    values = np.linspace(0, 100, 1001)
    u = np.quantile(values, 0.9)
    peaks = values[values > u] - u
    with patch("anomaly.pot.genpareto.fit", return_value=(0, 0, 2)) as fit:
        result = POTThreshold.fit(values, 0.9, 0.001)
    np.testing.assert_array_equal(fit.call_args.args[0], peaks)
    assert fit.call_args.kwargs == {"floc": 0}
    expected = u + 2 * np.log(len(peaks) / (len(values) * 0.001))
    assert result.threshold == pytest.approx(expected)


def test_pot_real_fit_recovers_heldout_exponential_tail():
    rng = np.random.default_rng(7)
    calibration = rng.exponential(2, size=30000)
    result = POTThreshold.fit(calibration, 0.9, 0.01)
    assert result.threshold == pytest.approx(-2 * np.log(0.01), rel=0.12)
    test = rng.exponential(2, size=30000)
    assert np.mean(test >= result.threshold) == pytest.approx(0.01, abs=0.004)


@pytest.mark.parametrize("values", [[], [np.nan], [-1, 2], np.ones(1000), [1, 2, 3]])
def test_bad_pot_tail_raises_instead_of_silent_fallback(values):
    with pytest.raises(ValueError):
        POTThreshold.fit(values)


def test_dimension_or_equality_and_frozen_calibration():
    rng = np.random.default_rng(9)
    calibration = ScoredSequence(pd.DataFrame({
        "score_close": rng.exponential(size=3000),
        "score_volume": rng.exponential(size=3000),
    }), ("close", "volume"))
    thresholds = fit_thresholds(calibration, risk=0.01)
    assert all(v.risk == 0.005 for v in thresholds.values())
    before = {f: v.to_dict() for f, v in thresholds.items()}
    d1, d2 = [thresholds[f].threshold for f in calibration.features]
    target = ScoredSequence(pd.DataFrame({
        "score_close": [0, d1, 0, d1 * 1e6],
        "score_volume": [0, 0, d2 * 2, 0],
    }), calibration.features)
    result = apply_thresholds(target, thresholds)
    assert result.is_anomaly.tolist() == [0, 1, 1, 1]
    assert result.anomaly_dimensions.tolist() == ["", "close", "volume", "close"]
    assert result.anomaly_score.iloc[1] == 1
    assert before == {f: v.to_dict() for f, v in thresholds.items()}
    with pytest.raises(ValueError, match="match"):
        apply_thresholds(target, {"close": thresholds["close"]})
    with pytest.raises(ValueError, match="Invalid threshold"):
        apply_thresholds(target, {**thresholds, "close": replace(thresholds["close"], threshold=0)})


def test_cli_fails_before_model_download_on_short_calibration(tmp_path):
    path = tmp_path / "data.csv"
    frame(100).to_csv(path, index=False)
    args = build_parser().parse_args(["--csv", str(path), "--lookback", "10",
                                      "--calibration-end", "50"])
    with pytest.raises(ValueError, match="too short"):
        run(args)
    assert parse_epsilon(["close=0.01", "volume=1"]) == {"close": 0.01, "volume": 1}
    with pytest.raises(ValueError):
        parse_epsilon(["close=0"])
