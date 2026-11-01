"""Run with python -m anomaly from the Kronos repository root."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd

from .scoring import FEATURE_COLUMNS, rolling_scores, validate_frame
from .pot import apply_thresholds, fit_thresholds


def build_parser():
    parser = argparse.ArgumentParser(description=(
        "Kronos rolling residual anomaly detection with held-out, fixed POT thresholds."
    ))
    parser.add_argument("--csv", type=Path, required=True, help="One asset, in timestamp order.")
    parser.add_argument("--calibration-end", type=int, required=True,
                        help="Exclusive zero-based row boundary; test starts at this row.")
    parser.add_argument("--test-end", type=int, help="Exclusive row boundary; default: CSV length.")
    parser.add_argument("--timestamp", default="timestamps")
    parser.add_argument("--features", nargs="+", choices=FEATURE_COLUMNS,
                        help="Default: all originally present OHLCVA columns.")
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--model", default="NeoQuasar/Kronos-mini")
    parser.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-2k")
    parser.add_argument("--model-revision", help="Optional Hugging Face commit/tag.")
    parser.add_argument("--tokenizer-revision", help="Optional Hugging Face commit/tag.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1, help="Torch CPU threads.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--epsilon", action="append", default=[], metavar="FEATURE=VALUE",
                        help="Per-feature positive scale floor addition; default 1e-5.")
    parser.add_argument("--initial-quantile", type=float, default=0.9)
    parser.add_argument("--risk", type=float, default=0.01,
                        help="Nominal tail probability; family-wise by default.")
    parser.add_argument("--risk-mode", choices=("family", "per-feature"), default="family")
    parser.add_argument("--min-peaks", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("anomaly_results/run"),
                        help="Must be a new directory, preventing accidental overwrites.")
    parser.add_argument("--plot", action="store_true", help="Save test close-price plot as PNG.")
    return parser


def parse_epsilon(items):
    result = {}
    for item in items:
        try:
            name, value = item.split("=", 1)
            value = float(value)
        except ValueError as exc:
            raise ValueError("epsilon must be FEATURE=VALUE.") from exc
        if name not in FEATURE_COLUMNS or name in result or not np.isfinite(value) or value <= 0:
            raise ValueError("epsilon needs a unique OHLCVA feature and finite positive value.")
        result[name] = value
    return result


def save_plot(test_output, destination):
    """Use observed bar positions so market closures are not drawn as trends."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.arange(len(test_output))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(positions, test_output.observed_close, label="Observed close")
    ax.plot(positions, test_output.predicted_close, label="One-step forecast")
    flagged = test_output.is_anomaly.to_numpy() == 1
    if flagged.any():
        ax.scatter(positions[flagged], test_output.observed_close[flagged],
                   c="red", marker="x", label="Any-dimension alarm")
    ticks = np.unique(np.linspace(0, len(test_output) - 1, min(6, len(test_output))).astype(int))
    times = pd.to_datetime(test_output.timestamp)
    ax.set_xticks(ticks, times.iloc[ticks].dt.strftime("%m-%d %H:%M"), rotation=25, ha="right")
    ax.set(title=f"Kronos residual anomalies (fixed POT; {int(flagged.sum())} alarms)",
           ylabel="Close price", xlabel="Observed bars (market closures omitted)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def run(args):
    epsilon = parse_epsilon(args.epsilon)
    data = validate_frame(pd.read_csv(args.csv), args.timestamp)
    end = len(data) if args.test_end is None else args.test_end
    if not 2 <= args.lookback < args.calibration_end < end <= len(data):
        raise ValueError("Need 2 <= lookback < calibration-end < test-end <= CSV length.")
    if args.max_context < args.lookback or args.threads < 1:
        raise ValueError("max-context must cover lookback; threads must be positive.")
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}. Choose a new --output directory.")
    features = tuple(args.features or [f for f in FEATURE_COLUMNS if f in data])
    if len(set(features)) != len(features) or not set(features) <= set(data):
        raise ValueError("features must be unique and originally present in the CSV.")
    if not set(epsilon) <= set(features):
        raise ValueError("epsilon contains an unscored feature.")
    if args.plot and "close" not in features:
        raise ValueError("--plot requires close in scored features.")
    if args.sample_count < 1 or not np.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("sample-count and temperature must be positive.")
    if not 0 < args.top_p <= 1:
        raise ValueError("top-p must be in (0,1].")
    # Check obvious POT configuration/sample-size errors before loading weights.
    marginal_risk = args.risk / len(features) if args.risk_mode == "family" else args.risk
    if not 0 < args.risk < 1 or not 0 < marginal_risk < 1 - args.initial_quantile < 1:
        raise ValueError("Need risk in (0,1) and marginal risk < 1 - initial-quantile < 1.")
    if args.min_peaks < 3:
        raise ValueError("min-peaks must be >= 3 (20+ recommended).")
    n_cal = args.calibration_end - args.lookback
    if int(np.ceil(n_cal * (1 - args.initial_quantile))) < args.min_peaks:
        raise ValueError("Calibration segment is too short for the requested POT tail/min-peaks.")

    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    tokenizer_options = {"local_files_only": args.local_files_only}
    model_options = {"local_files_only": args.local_files_only}
    if args.model_revision:
        model_options["revision"] = args.model_revision
    if args.tokenizer_revision:
        tokenizer_options["revision"] = args.tokenizer_revision
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer, **tokenizer_options)
    model = Kronos.from_pretrained(args.model, **model_options)
    tokenizer.eval()
    model.eval()
    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=args.max_context)
    options = dict(lookback=args.lookback, timestamp=args.timestamp, features=features,
                   epsilon=epsilon, sample_count=args.sample_count,
                   temperature=args.temperature, top_p=args.top_p, progress=True)
    print(f"Calibrating on rows [{args.lookback}, {args.calibration_end})", flush=True)
    calibration = rolling_scores(data.iloc[:args.calibration_end], predictor, **options)
    thresholds = fit_thresholds(calibration, args.initial_quantile, args.risk,
                                args.min_peaks, args.risk_mode)
    # Thresholds are frozen before any test predictions or scores are computed.
    print(f"Testing on rows [{args.calibration_end}, {end})", flush=True)
    offset = args.calibration_end - args.lookback
    test = rolling_scores(data.iloc[offset:end], predictor, **options)
    test.frame["row"] += offset
    calibration_output = apply_thresholds(calibration, thresholds)
    test_output = apply_thresholds(test, thresholds)
    metadata = {
        "schema_version": 1,
        "method": "kronos_one_step_standardized_squared_residual_fixed_pot",
        "upstream_extension": True,
        "features": list(features),
        "score": "((observed - predicted) / (history_population_std + epsilon)) ** 2",
        "thresholds": {f: v.to_dict() for f, v in thresholds.items()},
        "epsilon": {f: epsilon.get(f, 1e-5) for f in features},
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__},
        "split": {"warmup": [0, args.lookback],
                  "calibration": [args.lookback, args.calibration_end],
                  "test": [args.calibration_end, end], "end_exclusive": True},
        "summary": {
            "calibration_rows": len(calibration_output),
            "calibration_alarm_fraction_in_sample": float(calibration_output.is_anomaly.mean()),
            "test_rows": len(test_output),
            "test_anomaly_count": int(test_output.is_anomaly.sum()),
            "test_alarm_fraction": float(test_output.is_anomaly.mean()),
        },
        "limitations": [
            "No ground-truth labels evaluated; alarm fraction is not accuracy or F1.",
            "Fixed POT, not dynamically updated SPOT; tail fit may fail under market drift.",
            "Family risk uses Bonferroni allocation, not an empirical false-positive guarantee.",
            "Prior test observations enter later contexts, including prior anomalies.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    calibration_output.to_csv(args.output / "calibration_scores.csv", index=False)
    test_output.to_csv(args.output / "test_anomalies.csv", index=False)
    (args.output / "thresholds.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if args.plot:
        save_plot(test_output, args.output / "anomalies.png")
    print(json.dumps(metadata["summary"], indent=2))
    print(f"Saved to {args.output.resolve()}")
    return metadata


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
