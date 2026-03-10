#!/usr/bin/env python3
"""Evaluate an equal-weight average of multiple trained LSTM+XGB models.

Threshold selection happens on validation only. Test remains an untouched final holdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from xgboost import XGBClassifier

from train_lstm_from_sequences import (
    build_augmented_samples,
    frame_to_raw_sequences,
    load_dataframe,
    resolve_device,
    transform_samples,
)
from train_lstm_xgboost_ensemble import (
    MomentumSiameseLSTM,
    attach_weight_class,
    blend_specialist_predictions,
    build_oriented_static_matrix,
    build_oriented_weight_classes,
    choose_best_threshold,
    evaluate_probs,
    predict_momentum,
    split_with_locked_holdout,
)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate an equal-weight probability average across multiple trained ensemble runs."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=root_dir / "data" / "ufc_lstm_sequences.csv",
        help="Sequence CSV path.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        nargs="+",
        required=True,
        help="Metrics JSON files from trained runs to include in averaging.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root_dir / "data" / "model_cache" / "oof_model_average_ensemble_metrics.json",
        help="Where to save average-ensemble metrics.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation split fraction.")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Test split fraction.")
    parser.add_argument("--max-fights", type=int, default=None, help="Optional cap for fights.")
    parser.add_argument(
        "--holdout-manifest-path",
        type=Path,
        default=root_dir / "data" / "model_cache" / "final_holdout_fight_ids.txt",
        help="Locked holdout fight manifest path.",
    )
    parser.add_argument(
        "--refresh-holdout-manifest",
        action="store_true",
        help="Regenerate holdout manifest before evaluation.",
    )
    parser.add_argument(
        "--drop-empty-history",
        dest="drop_empty_history",
        action="store_true",
        help="Drop fights with fully empty history (default).",
    )
    parser.add_argument(
        "--keep-empty-history",
        dest="drop_empty_history",
        action="store_false",
        help="Keep fights with fully empty history.",
    )
    parser.set_defaults(drop_empty_history=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Inference dataloader workers.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def load_run_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        out = json.load(f)
    required = [
        "momentum_model_path",
        "momentum_scaler_path",
        "xgb_model_path",
        "xgb_specialists_path",
        "momentum_config",
        "xgb_config",
    ]
    missing = [k for k in required if k not in out]
    if missing:
        raise ValueError(f"Metrics {path} missing required keys: {missing}")
    return out


def load_scalers(path: Path) -> tuple[Any, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload["seq_scaler"], payload["static_scaler"]


def load_specialists(path: Path) -> tuple[dict[str, Any], float]:
    if not path.exists():
        return {}, 0.0
    with path.open("rb") as f:
        payload = pickle.load(f)
    models = payload.get("models", {}) if isinstance(payload, dict) else {}
    alpha = float(payload.get("blend_alpha", 0.0)) if isinstance(payload, dict) else 0.0
    return models, alpha


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    device = resolve_device(args.device)
    logging.info("Using device: %s", device)

    df, seq_len, num_stats, f1_cols, f2_cols = load_dataframe(
        args.input_csv,
        args.max_fights,
        drop_empty_history=args.drop_empty_history,
    )
    df = attach_weight_class(df, args.input_csv)
    train_df, val_df, test_df, holdout_info = split_with_locked_holdout(
        df,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        holdout_manifest_path=args.holdout_manifest_path,
        refresh_holdout_manifest=bool(args.refresh_holdout_manifest),
    )
    logging.info(
        "Split fights -> train: %d | val: %d | test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    val_f1, val_f2 = frame_to_raw_sequences(val_df, seq_len, num_stats, f1_cols, f2_cols)
    test_f1, test_f2 = frame_to_raw_sequences(test_df, seq_len, num_stats, f1_cols, f2_cols)
    weight_val = build_oriented_weight_classes(val_df)
    weight_test = build_oriented_weight_classes(test_df)

    sample_cache: dict[tuple[str, float], tuple[list[Any], list[Any]]] = {}
    static_cache: dict[tuple[bool, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    all_val_prob: list[np.ndarray] = []
    all_test_prob: list[np.ndarray] = []
    component_rows: list[dict[str, Any]] = []
    y_val_ref: np.ndarray | None = None
    y_test_ref: np.ndarray | None = None

    project_root = Path(__file__).resolve().parents[1]

    for metrics_path in args.metrics_json:
        metrics = load_run_metrics(metrics_path)

        momentum_cfg = metrics["momentum_config"]
        xgb_cfg = metrics["xgb_config"]
        recency_mode = str(momentum_cfg.get("static_recency_mode", "ema"))
        ema_alpha = float(momentum_cfg.get("ema_alpha", 0.75))
        trend_enabled = bool(xgb_cfg.get("use_trend_static_features", True))
        trend_alpha = float(xgb_cfg.get("trend_ema_alpha", 0.7))
        enhanced_context = bool(xgb_cfg.get("use_enhanced_context_static_features", False))

        sample_key = (recency_mode, ema_alpha)
        if sample_key not in sample_cache:
            val_samples_raw = build_augmented_samples(
                val_df,
                val_f1,
                val_f2,
                static_recency_mode=recency_mode,
                ema_alpha=ema_alpha,
            )
            test_samples_raw = build_augmented_samples(
                test_df,
                test_f1,
                test_f2,
                static_recency_mode=recency_mode,
                ema_alpha=ema_alpha,
            )
            sample_cache[sample_key] = (val_samples_raw, test_samples_raw)
        else:
            val_samples_raw, test_samples_raw = sample_cache[sample_key]

        static_key = (trend_enabled, trend_alpha, enhanced_context)
        if static_key not in static_cache:
            x_val_static, y_val = build_oriented_static_matrix(
                val_df,
                f1_raw=val_f1,
                f2_raw=val_f2,
                trend_ema_alpha=trend_alpha,
                use_trend_static_features=trend_enabled,
                use_enhanced_context_static_features=enhanced_context,
            )
            x_test_static, y_test = build_oriented_static_matrix(
                test_df,
                f1_raw=test_f1,
                f2_raw=test_f2,
                trend_ema_alpha=trend_alpha,
                use_trend_static_features=trend_enabled,
                use_enhanced_context_static_features=enhanced_context,
            )
            static_cache[static_key] = (x_val_static, y_val, x_test_static, y_test)
        x_val_static, y_val, x_test_static, y_test = static_cache[static_key]

        if y_val_ref is None:
            y_val_ref = y_val.astype(np.float32)
            y_test_ref = y_test.astype(np.float32)
        else:
            if not np.array_equal(y_val_ref.astype(np.int64), y_val.astype(np.int64)):
                raise ValueError(f"Label mismatch in validation set for {metrics_path}")
            if not np.array_equal(y_test_ref.astype(np.int64), y_test.astype(np.int64)):
                raise ValueError(f"Label mismatch in test set for {metrics_path}")

        momentum_model_path = resolve_path(project_root, str(metrics["momentum_model_path"]))
        scaler_path = resolve_path(project_root, str(metrics["momentum_scaler_path"]))
        xgb_model_path = resolve_path(project_root, str(metrics["xgb_model_path"]))
        xgb_specialists_path = resolve_path(project_root, str(metrics["xgb_specialists_path"]))

        seq_scaler, static_scaler = load_scalers(scaler_path)
        val_data = transform_samples(val_samples_raw, seq_scaler, static_scaler, seq_len)
        test_data = transform_samples(test_samples_raw, seq_scaler, static_scaler, seq_len)
        seq_dim = val_data[0].seq_a.shape[1]

        momentum_model = MomentumSiameseLSTM(
            seq_dim=seq_dim,
            hidden_size=int(momentum_cfg["hidden_size"]),
            num_layers=int(momentum_cfg["num_layers"]),
            dropout=float(momentum_cfg["dropout"]),
            bidirectional=bool(momentum_cfg["bidirectional"]),
            use_cross_attention=bool(momentum_cfg["use_cross_attention"]),
            attention_heads=int(momentum_cfg["attention_heads"]),
            attention_dropout=float(momentum_cfg["attention_dropout"]),
        ).to(device)
        checkpoint = torch.load(momentum_model_path, map_location=device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        momentum_model.load_state_dict(state_dict)

        momentum_val_prob, _ = predict_momentum(
            momentum_model,
            val_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        momentum_test_prob, _ = predict_momentum(
            momentum_model,
            test_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        x_val = np.concatenate([x_val_static, momentum_val_prob.reshape(-1, 1)], axis=1)
        x_test = np.concatenate([x_test_static, momentum_test_prob.reshape(-1, 1)], axis=1)
        xgb_model = XGBClassifier()
        xgb_model.load_model(str(xgb_model_path))
        val_prob = xgb_model.predict_proba(x_val)[:, 1].astype(np.float32)
        test_prob = xgb_model.predict_proba(x_test)[:, 1].astype(np.float32)

        use_specialists = bool(metrics.get("specialists", {}).get("enabled", False))
        specialist_models, blend_alpha = load_specialists(xgb_specialists_path)
        if use_specialists and specialist_models:
            val_prob = blend_specialist_predictions(
                base_prob=val_prob,
                x_matrix=x_val,
                weight_classes=weight_val,
                specialist_models=specialist_models,
                alpha=blend_alpha,
            )
            test_prob = blend_specialist_predictions(
                base_prob=test_prob,
                x_matrix=x_test,
                weight_classes=weight_test,
                specialist_models=specialist_models,
                alpha=blend_alpha,
            )

        all_val_prob.append(val_prob)
        all_test_prob.append(test_prob)
        component_rows.append(
            {
                "metrics_json": str(metrics_path),
                "ensemble_val_auc": float(evaluate_probs(y_val_ref, val_prob, 0.5)["auc"]),
                "ensemble_test_auc": float(evaluate_probs(y_test_ref, test_prob, 0.5)["auc"]),
                "use_specialists": bool(use_specialists and specialist_models),
            }
        )
        logging.info("Loaded %s", metrics_path)

    if not all_val_prob or y_val_ref is None or y_test_ref is None:
        raise ValueError("No model probabilities were produced.")

    val_avg = np.mean(np.stack(all_val_prob, axis=0), axis=0).astype(np.float32)
    test_avg = np.mean(np.stack(all_test_prob, axis=0), axis=0).astype(np.float32)
    threshold, val_metrics = choose_best_threshold(y_val_ref, val_avg)
    test_metrics = evaluate_probs(y_test_ref, test_avg, threshold)
    val_metrics_05 = evaluate_probs(y_val_ref, val_avg, threshold=0.5)
    test_metrics_05 = evaluate_probs(y_test_ref, test_avg, threshold=0.5)

    report = {
        "input_csv": str(args.input_csv),
        "holdout": holdout_info,
        "splits": {
            "train_fights": int(len(train_df)),
            "val_fights": int(len(val_df)),
            "test_fights": int(len(test_df)),
            "val_samples": int(len(y_val_ref)),
            "test_samples": int(len(y_test_ref)),
        },
        "num_models": int(len(args.metrics_json)),
        "model_components": component_rows,
        "average_strategy": "equal_weight_probability_mean",
        "ensemble_threshold": float(threshold),
        "ensemble_val_metrics_global_threshold_0_5": val_metrics_05,
        "ensemble_test_metrics_global_threshold_0_5": test_metrics_05,
        "ensemble_val_metrics_at_best_threshold": val_metrics,
        "ensemble_test_metrics": test_metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("Saved average-ensemble metrics: %s", args.output_json)
    logging.info("Average-ensemble test AUC: %.6f", float(test_metrics["auc"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
