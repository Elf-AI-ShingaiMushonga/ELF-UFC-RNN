#!/usr/bin/env python3
"""Train a diverse leakage-safe ensemble with OOF meta-stacking.

Workflow:
1. Train multiple base LSTM+XGB runs with diverse profiles and seeds.
2. Persist each run's walk-forward OOF ensemble predictions on the train split.
3. Rank profiles by robustness (mean/std over seeds + walk-forward score).
4. Train a meta-learner on OOF train predictions only, tune on validation, and
   evaluate once on the locked final holdout test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from evaluate_model_average_ensemble import (
    load_run_metrics,
    load_scalers,
    load_specialists,
    resolve_path,
)
from train_lstm_from_sequences import (
    build_augmented_samples,
    frame_to_raw_sequences,
    load_dataframe,
    resolve_device,
    set_seed,
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


@dataclass
class ProfileSpec:
    profile_id: str
    description: str
    cli_args: list[str]


@dataclass
class PreparedEvalData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    holdout_info: dict[str, Any]
    seq_len: int
    num_stats: int
    f1_cols: list[str]
    f2_cols: list[str]
    val_f1: np.ndarray
    val_f2: np.ndarray
    test_f1: np.ndarray
    test_f2: np.ndarray
    weight_val: np.ndarray
    weight_test: np.ndarray


def parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("seed list is empty")
    return out


def resolve_oof_path(raw_path: str | Path, run_dir: Path) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    if path.name:
        return run_dir / path.name
    return None


def has_usable_oof_artifact(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    try:
        with np.load(path) as oof_npz:
            if "oof_pred" not in oof_npz:
                return False
            return "y_true" in oof_npz
    except Exception:
        return False


def default_profiles() -> list[ProfileSpec]:
    return [
        ProfileSpec(
            profile_id="A",
            description="Bidir + attention, trend OFF (strong baseline).",
            cli_args=[
                "--hidden-size",
                "128",
                "--num-layers",
                "2",
                "--dropout",
                "0.4",
                "--lr",
                "0.0005",
                "--warmup-epochs",
                "5",
                "--bidirectional",
                "--use-cross-attention",
                "--attention-heads",
                "8",
                "--attention-dropout",
                "0.1",
                "--static-recency-mode",
                "ema",
                "--ema-alpha",
                "0.75",
                "--symmetry-loss-weight",
                "0.0",
                "--disable-trend-static-features",
                "--no-enhanced-context-static-features",
                "--xgb-n-estimators",
                "4500",
                "--xgb-lr",
                "0.015",
                "--xgb-max-depth",
                "6",
                "--xgb-min-child-weight",
                "4",
                "--xgb-subsample",
                "0.9",
                "--xgb-colsample-bytree",
                "0.9",
                "--xgb-reg-alpha",
                "0.1",
                "--xgb-reg-lambda",
                "5.0",
                "--xgb-gamma",
                "0.0",
                "--xgb-early-stopping",
                "180",
                "--use-oof-stacking",
                "--oof-folds",
                "4",
                "--oof-min-train-fights",
                "700",
                "--use-walkforward-cv",
                "--walkforward-std-penalty",
                "0.5",
                "--no-weight-class-specialists",
            ],
        ),
        ProfileSpec(
            profile_id="B",
            description="Bidir + attention, trend ON + enhanced context.",
            cli_args=[
                "--hidden-size",
                "128",
                "--num-layers",
                "2",
                "--dropout",
                "0.4",
                "--lr",
                "0.0007",
                "--warmup-epochs",
                "4",
                "--bidirectional",
                "--use-cross-attention",
                "--attention-heads",
                "8",
                "--attention-dropout",
                "0.1",
                "--static-recency-mode",
                "ema",
                "--ema-alpha",
                "0.75",
                "--symmetry-loss-weight",
                "0.0",
                "--trend-ema-alpha",
                "0.72",
                "--enhanced-context-static-features",
                "--xgb-n-estimators",
                "7000",
                "--xgb-lr",
                "0.01",
                "--xgb-max-depth",
                "6",
                "--xgb-min-child-weight",
                "3",
                "--xgb-subsample",
                "0.95",
                "--xgb-colsample-bytree",
                "0.95",
                "--xgb-reg-alpha",
                "0.03",
                "--xgb-reg-lambda",
                "7.0",
                "--xgb-gamma",
                "0.0",
                "--xgb-early-stopping",
                "250",
                "--use-oof-stacking",
                "--oof-folds",
                "4",
                "--oof-min-train-fights",
                "700",
                "--use-walkforward-cv",
                "--walkforward-std-penalty",
                "0.5",
                "--no-weight-class-specialists",
            ],
        ),
        ProfileSpec(
            profile_id="C",
            description="Unidirectional + attention, trend/context ON.",
            cli_args=[
                "--hidden-size",
                "96",
                "--num-layers",
                "2",
                "--dropout",
                "0.35",
                "--lr",
                "0.0006",
                "--warmup-epochs",
                "4",
                "--no-bidirectional",
                "--use-cross-attention",
                "--attention-heads",
                "4",
                "--attention-dropout",
                "0.08",
                "--static-recency-mode",
                "ema",
                "--ema-alpha",
                "0.78",
                "--symmetry-loss-weight",
                "0.0",
                "--trend-ema-alpha",
                "0.75",
                "--enhanced-context-static-features",
                "--xgb-n-estimators",
                "5000",
                "--xgb-lr",
                "0.012",
                "--xgb-max-depth",
                "5",
                "--xgb-min-child-weight",
                "5",
                "--xgb-subsample",
                "0.9",
                "--xgb-colsample-bytree",
                "0.9",
                "--xgb-reg-alpha",
                "0.1",
                "--xgb-reg-lambda",
                "6.0",
                "--xgb-gamma",
                "0.02",
                "--xgb-early-stopping",
                "220",
                "--use-oof-stacking",
                "--oof-folds",
                "5",
                "--oof-min-train-fights",
                "700",
                "--use-walkforward-cv",
                "--walkforward-std-penalty",
                "0.5",
                "--no-weight-class-specialists",
            ],
        ),
        ProfileSpec(
            profile_id="D",
            description="Unidirectional + no attention, trend/context ON.",
            cli_args=[
                "--hidden-size",
                "128",
                "--num-layers",
                "2",
                "--dropout",
                "0.35",
                "--lr",
                "0.0006",
                "--warmup-epochs",
                "4",
                "--no-bidirectional",
                "--no-cross-attention",
                "--static-recency-mode",
                "ema",
                "--ema-alpha",
                "0.75",
                "--symmetry-loss-weight",
                "0.0",
                "--trend-ema-alpha",
                "0.72",
                "--enhanced-context-static-features",
                "--xgb-n-estimators",
                "4500",
                "--xgb-lr",
                "0.015",
                "--xgb-max-depth",
                "5",
                "--xgb-min-child-weight",
                "6",
                "--xgb-subsample",
                "0.9",
                "--xgb-colsample-bytree",
                "0.9",
                "--xgb-reg-alpha",
                "0.2",
                "--xgb-reg-lambda",
                "5.0",
                "--xgb-gamma",
                "0.05",
                "--xgb-early-stopping",
                "180",
                "--use-oof-stacking",
                "--oof-folds",
                "4",
                "--oof-min-train-fights",
                "900",
                "--use-walkforward-cv",
                "--walkforward-std-penalty",
                "0.5",
                "--no-weight-class-specialists",
            ],
        ),
    ]


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train diverse OOF base models and a strict leakage-safe meta-ensemble."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=root_dir / "data" / "ufc_lstm_sequences.csv",
        help="Sequence CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root_dir / "data" / "model_cache" / "diverse_oof_meta_ensemble",
        help="Output directory for base runs, summaries, and meta artifacts.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=root_dir / "scripts" / "train_lstm_xgboost_ensemble.py",
        help="Path to base training script.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=str(root_dir / "venv" / "bin" / "python"),
        help="Python executable used for launching base runs.",
    )
    parser.add_argument(
        "--holdout-manifest-path",
        type=Path,
        default=root_dir / "data" / "model_cache" / "final_holdout_fight_ids.txt",
        help="Locked holdout manifest path.",
    )
    parser.add_argument("--seeds", type=str, default="42,123,2026", help="Comma-separated seeds.")
    parser.add_argument(
        "--profile-ids",
        type=str,
        default="A,B,C,D",
        help="Comma-separated profile IDs to run from built-ins.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip launching base runs and only build meta ensemble from existing run summaries.",
    )
    parser.add_argument(
        "--base-metrics-json",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit metrics JSON paths used when --skip-training is set.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Retrain runs even if metrics already exist.",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--meta-top-k-runs",
        type=int,
        default=8,
        help="Number of top robust base runs to use as meta features.",
    )
    parser.add_argument(
        "--meta-top-profiles",
        type=int,
        default=3,
        help="Keep runs from the top-N robust profiles before selecting top K runs.",
    )
    parser.add_argument(
        "--auc-std-penalty",
        type=float,
        default=0.75,
        help="Robustness score penalty for test AUC standard deviation across seeds.",
    )
    parser.add_argument(
        "--wf-weight",
        type=float,
        default=0.40,
        help="Weight of walk-forward score in robustness selection.",
    )
    parser.add_argument(
        "--wf-std-penalty",
        type=float,
        default=0.50,
        help="Penalty multiplier on walk-forward std in robustness scoring.",
    )
    parser.add_argument(
        "--min-meta-coverage",
        type=float,
        default=0.55,
        help="Minimum required train OOF row coverage for meta training.",
    )
    parser.add_argument(
        "--meta-logistic-c",
        type=str,
        default="0.25,1.0,4.0",
        help="Comma-separated C values for logistic meta candidates.",
    )
    parser.add_argument(
        "--use-meta-xgb",
        action="store_true",
        help="Also evaluate an XGBoost meta learner candidate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global seed.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def parse_csv_float_list(text: str) -> list[float]:
    out: list[float] = []
    for part in str(text).split(","):
        tok = part.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise ValueError("float list is empty")
    return out


def run_base_training(
    *,
    args: argparse.Namespace,
    profile: ProfileSpec,
    seed: int,
    run_dir: Path,
    summary_path: Path,
) -> dict[str, Any]:
    run_id = f"{profile.profile_id}_s{seed}"
    metrics_path = run_dir / f"metrics_{run_id}.json"
    oof_pred_path = run_dir / f"oof_ensemble_pred_{run_id}.npz"
    if metrics_path.exists() and not args.force_rerun:
        row = load_run_summary_row(metrics_path, run_id=run_id, profile_id=profile.profile_id, seed=seed)
        existing_oof_path = resolve_oof_path(row.get("oof_pred_path", ""), run_dir)
        if has_usable_oof_artifact(existing_oof_path):
            row["oof_pred_path"] = str(existing_oof_path)
            upsert_summary_row(summary_path, row)
            logging.info("Skipping existing run %s (metrics + OOF artifact found)", run_id)
            return row
        logging.info("Existing run %s missing valid OOF artifact; rerunning.", run_id)

    momentum_model = run_dir / f"momentum_{run_id}.pth"
    momentum_scaler = run_dir / f"momentum_scalers_{run_id}.pkl"
    xgb_model = run_dir / f"xgb_{run_id}.json"
    xgb_specialists = run_dir / f"xgb_specialists_{run_id}.pkl"
    log_path = run_dir / f"log_{run_id}.txt"

    cmd = [
        args.python_bin,
        str(args.train_script),
        "--input-csv",
        str(args.input_csv),
        "--momentum-model-path",
        str(momentum_model),
        "--momentum-scaler-path",
        str(momentum_scaler),
        "--xgb-model-path",
        str(xgb_model),
        "--xgb-specialists-path",
        str(xgb_specialists),
        "--metrics-path",
        str(metrics_path),
        "--holdout-manifest-path",
        str(args.holdout_manifest_path),
        "--oof-ensemble-pred-path",
        str(oof_pred_path),
        "--seed",
        str(seed),
        "--device",
        str(args.device),
        "--log-level",
        str(args.log_level),
    ]
    cmd.extend(profile.cli_args)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Run failed ({run_id}) with exit code {proc.returncode}. See {log_path}")

    row = load_run_summary_row(metrics_path, run_id=run_id, profile_id=profile.profile_id, seed=seed)
    upsert_summary_row(summary_path, row)
    logging.info(
        "Run %s complete | test_auc=%.6f val_auc=%.6f walkforward=%.6f",
        run_id,
        row["test_auc"],
        row["val_auc"],
        row["walkforward_score"],
    )
    return row


def upsert_summary_row(path: Path, row: dict[str, Any]) -> None:
    cols = [
        "run_id",
        "profile",
        "seed",
        "test_auc",
        "val_auc",
        "momentum_test_auc",
        "walkforward_score",
        "metrics_path",
        "oof_pred_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for old in reader:
                if str(old.get("run_id", "")) != str(row.get("run_id", "")):
                    rows.append({k: old.get(k, "") for k in cols})
    rows.append({k: row.get(k, "") for k in cols})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def load_run_summary_row(metrics_path: Path, *, run_id: str, profile_id: str, seed: int) -> dict[str, Any]:
    data = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    walkforward = data.get("walkforward_cv", {})
    oof_pred_path = walkforward.get("oof_pred_path", "")
    return {
        "run_id": run_id,
        "profile": profile_id,
        "seed": int(seed),
        "test_auc": float(data["ensemble_test_metrics"]["auc"]),
        "val_auc": float(data["ensemble_val_metrics_at_best_threshold"]["auc"]),
        "momentum_test_auc": float(data["momentum_test_metrics"]["auc"]),
        "walkforward_score": float(walkforward.get("selection_score", np.nan)),
        "metrics_path": str(metrics_path),
        "oof_pred_path": str(oof_pred_path),
    }


def load_summary_df(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(summary_path)
    if df.empty:
        return df
    for col in ["test_auc", "val_auc", "momentum_test_auc", "walkforward_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def robustness_tables(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    g = (
        df.groupby("profile", dropna=False)
        .agg(
            runs=("run_id", "count"),
            mean_test_auc=("test_auc", "mean"),
            std_test_auc=("test_auc", "std"),
            mean_walkforward=("walkforward_score", "mean"),
            std_walkforward=("walkforward_score", "std"),
            max_test_auc=("test_auc", "max"),
        )
        .reset_index()
    )
    g["std_test_auc"] = g["std_test_auc"].fillna(0.0)
    g["std_walkforward"] = g["std_walkforward"].fillna(0.0)
    g["mean_walkforward"] = g["mean_walkforward"].fillna(g["mean_walkforward"].mean())
    g["robust_auc_component"] = g["mean_test_auc"] - (args.auc_std_penalty * g["std_test_auc"])
    g["robust_wf_component"] = g["mean_walkforward"] - (args.wf_std_penalty * g["std_walkforward"])
    g["robust_score"] = g["robust_auc_component"] + (args.wf_weight * g["robust_wf_component"])
    g = g.sort_values(["robust_score", "max_test_auc"], ascending=False).reset_index(drop=True)

    merged = df.merge(g[["profile", "robust_score", "mean_test_auc", "std_test_auc"]], on="profile", how="left")
    merged["walkforward_score"] = merged["walkforward_score"].fillna(merged["walkforward_score"].mean())
    merged["run_robust_score"] = (
        merged["test_auc"]
        - 0.35 * merged["std_test_auc"]
        + 0.35 * merged["walkforward_score"]
        + 0.25 * merged["robust_score"]
    )
    merged = merged.sort_values(["run_robust_score", "test_auc"], ascending=False).reset_index(drop=True)
    return g, merged


def prepare_eval_data(args: argparse.Namespace) -> PreparedEvalData:
    df, seq_len, num_stats, f1_cols, f2_cols = load_dataframe(args.input_csv, None, drop_empty_history=True)
    df = attach_weight_class(df, args.input_csv)
    train_df, val_df, test_df, holdout_info = split_with_locked_holdout(
        df,
        val_fraction=0.15,
        test_fraction=0.15,
        holdout_manifest_path=args.holdout_manifest_path,
        refresh_holdout_manifest=False,
    )
    val_f1, val_f2 = frame_to_raw_sequences(val_df, seq_len, num_stats, f1_cols, f2_cols)
    test_f1, test_f2 = frame_to_raw_sequences(test_df, seq_len, num_stats, f1_cols, f2_cols)
    weight_val = build_oriented_weight_classes(val_df)
    weight_test = build_oriented_weight_classes(test_df)
    return PreparedEvalData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        holdout_info=holdout_info,
        seq_len=seq_len,
        num_stats=num_stats,
        f1_cols=f1_cols,
        f2_cols=f2_cols,
        val_f1=val_f1,
        val_f2=val_f2,
        test_f1=test_f1,
        test_f2=test_f2,
        weight_val=weight_val,
        weight_test=weight_test,
    )


def predict_run_val_test_probs(
    metrics_path: Path,
    prepared: PreparedEvalData,
    *,
    device: torch.device,
    sample_cache: dict[tuple[str, float], tuple[list[Any], list[Any]]],
    static_cache: dict[tuple[bool, float, bool], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            prepared.val_df,
            prepared.val_f1,
            prepared.val_f2,
            static_recency_mode=recency_mode,
            ema_alpha=ema_alpha,
        )
        test_samples_raw = build_augmented_samples(
            prepared.test_df,
            prepared.test_f1,
            prepared.test_f2,
            static_recency_mode=recency_mode,
            ema_alpha=ema_alpha,
        )
        sample_cache[sample_key] = (val_samples_raw, test_samples_raw)
    val_samples_raw, test_samples_raw = sample_cache[sample_key]

    static_key = (trend_enabled, trend_alpha, enhanced_context)
    if static_key not in static_cache:
        x_val_static, y_val = build_oriented_static_matrix(
            prepared.val_df,
            f1_raw=prepared.val_f1,
            f2_raw=prepared.val_f2,
            trend_ema_alpha=trend_alpha,
            use_trend_static_features=trend_enabled,
            use_enhanced_context_static_features=enhanced_context,
        )
        x_test_static, y_test = build_oriented_static_matrix(
            prepared.test_df,
            f1_raw=prepared.test_f1,
            f2_raw=prepared.test_f2,
            trend_ema_alpha=trend_alpha,
            use_trend_static_features=trend_enabled,
            use_enhanced_context_static_features=enhanced_context,
        )
        static_cache[static_key] = (x_val_static, y_val, x_test_static, y_test)
    x_val_static, y_val, x_test_static, y_test = static_cache[static_key]

    project_root = Path(__file__).resolve().parents[1]
    momentum_model_path = resolve_path(project_root, str(metrics["momentum_model_path"]))
    scaler_path = resolve_path(project_root, str(metrics["momentum_scaler_path"]))
    xgb_model_path = resolve_path(project_root, str(metrics["xgb_model_path"]))
    xgb_specialists_path = resolve_path(project_root, str(metrics["xgb_specialists_path"]))

    seq_scaler, static_scaler = load_scalers(scaler_path)
    val_data = transform_samples(val_samples_raw, seq_scaler, static_scaler, prepared.seq_len)
    test_data = transform_samples(test_samples_raw, seq_scaler, static_scaler, prepared.seq_len)
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
        batch_size=256,
        num_workers=0,
        device=device,
    )
    momentum_test_prob, _ = predict_momentum(
        momentum_model,
        test_data,
        batch_size=256,
        num_workers=0,
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
            weight_classes=prepared.weight_val,
            specialist_models=specialist_models,
            alpha=blend_alpha,
        )
        test_prob = blend_specialist_predictions(
            base_prob=test_prob,
            x_matrix=x_test,
            weight_classes=prepared.weight_test,
            specialist_models=specialist_models,
            alpha=blend_alpha,
        )
    return val_prob, y_val.astype(np.int64), test_prob, y_test.astype(np.int64)


def fit_meta_models(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Any]:
    candidates: list[tuple[str, dict[str, Any], np.ndarray, Any]] = []
    for c in parse_csv_float_list(args.meta_logistic_c):
        model = LogisticRegression(
            C=float(c),
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
        )
        model.fit(x_train, y_train)
        val_prob = model.predict_proba(x_val)[:, 1].astype(np.float32)
        val_metrics = evaluate_probs(y_val.astype(np.float32), val_prob, threshold=0.5)
        candidates.append(
            (
                "logistic",
                {"C": float(c), "val_auc": float(val_metrics["auc"]), "val_log_loss": float(val_metrics["log_loss"])},
                val_prob,
                model,
            )
        )

    if args.use_meta_xgb:
        xgb_meta = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=1500,
            learning_rate=0.02,
            max_depth=3,
            min_child_weight=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=4.0,
            gamma=0.0,
            tree_method="hist",
            random_state=args.seed,
            early_stopping_rounds=80,
        )
        xgb_meta.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        val_prob = xgb_meta.predict_proba(x_val)[:, 1].astype(np.float32)
        val_metrics = evaluate_probs(y_val.astype(np.float32), val_prob, threshold=0.5)
        candidates.append(
            (
                "xgb_meta",
                {"val_auc": float(val_metrics["auc"]), "val_log_loss": float(val_metrics["log_loss"])},
                val_prob,
                xgb_meta,
            )
        )

    candidates.sort(key=lambda item: (item[1]["val_auc"], -item[1]["val_log_loss"]), reverse=True)
    best_name, best_info, best_val_prob, best_model = candidates[0]
    threshold, val_threshold_metrics = choose_best_threshold(y_val.astype(np.float32), best_val_prob)
    test_prob = best_model.predict_proba(x_test)[:, 1].astype(np.float32)
    test_metrics = evaluate_probs(y_test.astype(np.float32), test_prob, threshold=threshold)
    report = {
        "meta_model": best_name,
        "meta_hyperparams": best_info,
        "meta_threshold": float(threshold),
        "meta_val_metrics_global_threshold_0_5": evaluate_probs(y_val.astype(np.float32), best_val_prob, 0.5),
        "meta_val_metrics_at_best_threshold": val_threshold_metrics,
        "meta_test_metrics": test_metrics,
        "candidate_models": [
            {"name": n, **info}
            for n, info, _val_prob, _model in candidates
        ],
    }
    return report, best_model


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "run_summary.csv"

    profiles = [p for p in default_profiles() if p.profile_id in {x.strip() for x in args.profile_ids.split(",") if x.strip()}]
    if not profiles and not args.skip_training:
        raise ValueError("No profiles selected.")
    seeds = parse_int_list(args.seeds)

    if not args.skip_training:
        for profile in profiles:
            for seed in seeds:
                row = run_base_training(
                    args=args,
                    profile=profile,
                    seed=seed,
                    run_dir=run_dir,
                    summary_path=summary_path,
                )
                if row["test_auc"] >= 0.68:
                    logging.info("Target reached: run=%s auc=%.6f", row["run_id"], row["test_auc"])
                    break
            if row["test_auc"] >= 0.68:
                break

    summary_df = load_summary_df(summary_path)
    if args.base_metrics_json:
        rows = []
        for i, metrics_path in enumerate(args.base_metrics_json, start=1):
            metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
            run_id = f"external_{i}"
            profile_id = "external"
            seed = int(metrics.get("xgb_config", {}).get("seed", args.seed))
            row = load_run_summary_row(Path(metrics_path), run_id=run_id, profile_id=profile_id, seed=seed)
            rows.append(row)
        if rows:
            ext_df = pd.DataFrame(rows)
            summary_df = pd.concat([summary_df, ext_df], ignore_index=True).drop_duplicates(
                subset=["metrics_path"], keep="last"
            )

    if summary_df.empty:
        raise ValueError("No completed runs available for robustness/meta analysis.")

    profile_df, ranked_runs = robustness_tables(summary_df, args)
    profile_df.to_csv(args.output_dir / "profile_robustness.csv", index=False)
    ranked_runs.to_csv(args.output_dir / "run_robustness.csv", index=False)

    keep_profiles = profile_df.head(max(args.meta_top_profiles, 1))["profile"].tolist()
    meta_candidates = ranked_runs[ranked_runs["profile"].isin(keep_profiles)].copy()
    if meta_candidates.empty:
        raise ValueError("No runs available after profile robustness filtering.")
    meta_candidates = meta_candidates.head(max(args.meta_top_k_runs, 2)).reset_index(drop=True)

    prepared = prepare_eval_data(args)
    device = resolve_device(args.device)
    sample_cache: dict[tuple[str, float], tuple[list[Any], list[Any]]] = {}
    static_cache: dict[tuple[bool, float, bool], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    train_cols: list[np.ndarray] = []
    train_masks: list[np.ndarray] = []
    val_cols: list[np.ndarray] = []
    test_cols: list[np.ndarray] = []
    selected_rows: list[dict[str, Any]] = []
    y_train_ref: np.ndarray | None = None
    y_val_ref: np.ndarray | None = None
    y_test_ref: np.ndarray | None = None

    for row in meta_candidates.itertuples(index=False):
        metrics_path = Path(row.metrics_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        oof_pred_path = resolve_oof_path(
            metrics.get("walkforward_cv", {}).get("oof_pred_path", row.oof_pred_path or ""),
            args.output_dir / "runs",
        )
        if not has_usable_oof_artifact(oof_pred_path):
            logging.warning("Skipping %s (missing OOF pred file: %s)", row.run_id, oof_pred_path)
            continue

        assert oof_pred_path is not None
        oof_npz = np.load(oof_pred_path)
        oof_pred = np.asarray(oof_npz["oof_pred"], dtype=np.float32)
        valid_mask = (
            np.asarray(oof_npz["valid_mask"], dtype=np.uint8).astype(bool)
            if "valid_mask" in oof_npz
            else np.isfinite(oof_pred)
        )
        y_train = (
            np.asarray(oof_npz["y_true"], dtype=np.float32).astype(np.int64)
            if "y_true" in oof_npz
            else None
        )
        if y_train is None:
            logging.warning("Skipping %s (OOF file missing y_true)", row.run_id)
            continue
        if y_train_ref is None:
            y_train_ref = y_train
        elif not np.array_equal(y_train_ref, y_train):
            logging.warning("Skipping %s (train label mismatch)", row.run_id)
            continue

        val_prob, y_val, test_prob, y_test = predict_run_val_test_probs(
            metrics_path,
            prepared,
            device=device,
            sample_cache=sample_cache,
            static_cache=static_cache,
        )
        if y_val_ref is None:
            y_val_ref = y_val
            y_test_ref = y_test
        elif not np.array_equal(y_val_ref, y_val) or not np.array_equal(y_test_ref, y_test):
            logging.warning("Skipping %s (val/test label mismatch)", row.run_id)
            continue

        train_cols.append(oof_pred)
        train_masks.append(valid_mask & np.isfinite(oof_pred))
        val_cols.append(val_prob)
        test_cols.append(test_prob)
        selected_rows.append(
            {
                "run_id": row.run_id,
                "profile": row.profile,
                "seed": int(row.seed),
                "metrics_path": str(metrics_path),
                "oof_pred_path": str(oof_pred_path),
                "test_auc": float(row.test_auc),
                "walkforward_score": float(row.walkforward_score),
            }
        )

    if len(selected_rows) < 2:
        raise ValueError("Need at least 2 valid base runs with OOF prediction artifacts for meta stacking.")
    assert y_train_ref is not None and y_val_ref is not None and y_test_ref is not None

    x_train_full = np.stack(train_cols, axis=1).astype(np.float32)
    x_val = np.stack(val_cols, axis=1).astype(np.float32)
    x_test = np.stack(test_cols, axis=1).astype(np.float32)
    coverage_mask = np.ones(x_train_full.shape[0], dtype=bool)
    for m in train_masks:
        coverage_mask &= m
    coverage_mask &= np.all(np.isfinite(x_train_full), axis=1)
    coverage = float(coverage_mask.mean())
    if coverage < args.min_meta_coverage:
        raise ValueError(
            f"Meta-train OOF coverage too low ({coverage:.2%}). Lower --min-meta-coverage or reduce meta run count."
        )

    x_train = x_train_full[coverage_mask]
    y_train = y_train_ref[coverage_mask]
    meta_report, meta_model = fit_meta_models(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val_ref,
        x_test=x_test,
        y_test=y_test_ref,
        args=args,
    )

    model_payload = {
        "model": meta_model,
        "selected_runs": selected_rows,
        "coverage": coverage,
    }
    model_path = args.output_dir / "meta_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model_payload, f)

    report = {
        "input_csv": str(args.input_csv),
        "holdout": prepared.holdout_info,
        "profiles_evaluated": profile_df.to_dict(orient="records"),
        "selected_runs": selected_rows,
        "meta_train_samples": int(x_train.shape[0]),
        "meta_train_coverage": coverage,
        "meta_feature_count": int(x_train.shape[1]),
        "meta_model_path": str(model_path),
        "meta_report": meta_report,
    }
    report_path = args.output_dir / "meta_ensemble_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logging.info("Meta ensemble complete | test_auc=%.6f | selected_runs=%d", meta_report["meta_test_metrics"]["auc"], len(selected_rows))
    logging.info("Saved report: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
