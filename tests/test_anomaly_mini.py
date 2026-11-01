"""Opt-in integration: KRONOS_MINI_WEIGHTS_DIR=/path/to/weights python -m pytest ..."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anomaly import POTThreshold, apply_thresholds, rolling_scores
from anomaly.__main__ import build_parser, run


@pytest.mark.skipif(not os.getenv("KRONOS_MINI_WEIGHTS_DIR"), reason="Local mini weights not configured")
def test_real_mini_offline_pipeline_and_target_isolation(tmp_path):
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer

    weights = Path(os.environ["KRONOS_MINI_WEIGHTS_DIR"]).resolve()
    source = Path(__file__).parent / "data" / "regression_input.csv"
    output = tmp_path / "mini"
    args = build_parser().parse_args([
        "--csv", str(source), "--model", str(weights / "Kronos-mini"),
        "--tokenizer", str(weights / "Kronos-Tokenizer-2k"), "--local-files-only",
        "--lookback", "64", "--calibration-end", "500", "--test-end", "520",
        "--output", str(output), "--plot",
    ])
    metadata = run(args)
    assert metadata["summary"]["calibration_rows"] == 436
    assert metadata["summary"]["test_rows"] == 20
    result = pd.read_csv(output / "test_anomalies.csv")
    assert result.row.tolist() == list(range(500, 520))
    np.testing.assert_array_equal(result.is_anomaly, (result.anomaly_score >= 1).astype(int))
    assert (output / "anomalies.png").stat().st_size > 1000
    stored = json.loads((output / "thresholds.json").read_text())
    assert stored["thresholds"] == metadata["thresholds"]
    assert len(pd.read_csv(output / "calibration_scores.csv")) == 436

    thresholds = {f: POTThreshold(**v) for f, v in stored["thresholds"].items()}
    tokenizer = KronosTokenizer.from_pretrained(str(weights / "Kronos-Tokenizer-2k"),
                                               local_files_only=True).eval()
    model = Kronos.from_pretrained(str(weights / "Kronos-mini"), local_files_only=True).eval()
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=2048)
    data = pd.read_csv(source).iloc[:501].copy()
    injected = data.copy()
    injected.loc[500, "high"] += 10
    torch.manual_seed(123)
    baseline = rolling_scores(data, predictor, lookback=64, start=500)
    torch.manual_seed(123)
    spike = rolling_scores(injected, predictor, lookback=64, start=500)
    for f in baseline.features:
        np.testing.assert_array_equal(baseline.frame[f"predicted_{f}"], spike.frame[f"predicted_{f}"])
        if f != "high":
            np.testing.assert_array_equal(baseline.frame[f"score_{f}"], spike.frame[f"score_{f}"])
    labeled = apply_thresholds(spike, thresholds)
    assert labeled.is_anomaly_high.iloc[0] == 1
    assert labeled.is_anomaly.iloc[0] == 1
    assert labeled.anomaly_dimensions.iloc[0] == "high"
