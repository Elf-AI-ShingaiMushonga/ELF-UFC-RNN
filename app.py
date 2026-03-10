#!/usr/bin/env python3
"""Flask app for UFC prediction, bet tracking, and maintenance workflows."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock, Thread
from typing import Any

from flask import Flask, Response, jsonify, render_template, request

APP_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = APP_ROOT / "scripts" / "train_lstm_xgboost_ensemble.py"
SCRAPER_SCRIPT = APP_ROOT / "scripts" / "scrape_ufc_fight_details.py"
SEQUENCE_SCRIPT = APP_ROOT / "scripts" / "build_fight_history_sequences.py"
AUDIT_SCRIPT = APP_ROOT / "scripts" / "audit_lstm_pipeline_data.py"

RAW_FIGHTS_CSV = APP_ROOT / "data" / "ufc_fight_details_lstm.csv"
METADATA_CSV = APP_ROOT / "data" / "ufc_fights_cleaned.csv"
SEQUENCE_CSV = APP_ROOT / "data" / "ufc_lstm_sequences.csv"
SCRAPER_CHECKPOINT_DB = APP_ROOT / "data" / "checkpoints" / "ufc_fight_details_checkpoint.sqlite"

BET_TRACKER_DB = APP_ROOT / "data" / "bet_tracker.sqlite3"
LEGACY_BET_TRACKER_JSON = APP_ROOT / "data" / "bet_tracker.json"
LEGACY_BETS_CSV = APP_ROOT / "data" / "bets_tracker.csv"
MODEL_STATE_PATH = APP_ROOT / "data" / "model_cache" / "site_model_state.json"
SITE_REFRESH_DIR = APP_ROOT / "data" / "model_cache" / "site_refresh_candidates"

BEST_TRAINING_RUN_INFO: dict[str, Any] = {
    "kind": "single_run",
    "run_id": "r11",
    "profile": "tuning_ensemble_068hunt",
    "label": "Best Single-Run Baseline (r11)",
    "test_auc": 0.6762168773046449,
    "val_auc": 0.6415739031958922,
    "selection_metric": "auc_plus_logloss_gain",
    "selection_score": 0.6871882089178986,
    "walkforward_score": 0.0,
    "walkforward_enabled": False,
    "metrics_path": "data/model_cache/tuning_ensemble_068hunt/metrics_r11.json",
    "momentum_model_path": "data/model_cache/tuning_ensemble_068hunt/momentum_r11.pth",
    "momentum_scaler_path": "data/model_cache/tuning_ensemble_068hunt/momentum_scalers_r11.pkl",
    "xgb_model_path": "data/model_cache/tuning_ensemble_068hunt/xgb_r11.json",
}

BEST_PREDICTOR_INFO: dict[str, Any] = {
    "kind": "weighted_ensemble",
    "run_id": "weighted_holdout_blend_v1",
    "profile": "auc068_runs",
    "label": "Weighted Ensemble (highest AUC)",
    "test_auc": 0.6834078294090559,
    "val_auc": 0.6452737138374545,
    "selection_metric": "holdout_tuned_weighted_probability_mean",
    "selection_score": 0.6834078294090559,
    "walkforward_score": 0.0,
    "walkforward_enabled": False,
    "metrics_path": "data/model_cache/auc068_runs/weighted_holdout_blend_v1.json",
}

DEFAULT_PARAMS: dict[str, Any] = {
    "input_csv": "data/ufc_lstm_sequences.csv",
    "momentum_model_path": BEST_TRAINING_RUN_INFO["momentum_model_path"],
    "momentum_scaler_path": BEST_TRAINING_RUN_INFO["momentum_scaler_path"],
    "xgb_model_path": BEST_TRAINING_RUN_INFO["xgb_model_path"],
    "metrics_path": BEST_TRAINING_RUN_INFO["metrics_path"],
    "epochs": 90,
    "patience": 20,
    "batch_size": 256,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.4,
    "lr": 0.0005,
    "warmup_epochs": 5,
    "min_epochs": 10,
    "min_delta": 0.0001,
    "weight_decay": 0.0001,
    "grad_clip": 1.0,
    "bidirectional": True,
    "use_cross_attention": True,
    "attention_heads": 8,
    "attention_dropout": 0.10,
    "static_recency_mode": "ema",
    "ema_alpha": 0.75,
    "val_fraction": 0.15,
    "test_fraction": 0.15,
    "max_fights": "",
    "seed": 42,
    "device": "auto",
    "num_workers": 0,
    "xgb_n_estimators": 5000,
    "xgb_lr": 0.012,
    "xgb_max_depth": 5,
    "xgb_min_child_weight": 4.0,
    "xgb_subsample": 0.95,
    "xgb_colsample_bytree": 0.95,
    "xgb_reg_alpha": 0.05,
    "xgb_reg_lambda": 6.0,
    "xgb_gamma": 0.0,
    "xgb_early_stopping": 200,
    "xgb_n_jobs": 0,
    "trend_ema_alpha": 0.70,
    "use_trend_static_features": False,
    "use_enhanced_context_static_features": False,
    "use_oof_stacking": False,
    "oof_folds": 4,
    "oof_min_train_fights": 700,
    "use_walkforward_cv": False,
    "walkforward_std_penalty": 0.5,
    "use_weight_class_specialists": False,
    "log_level": "INFO",
}


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def iso_from_timestamp(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return dt.datetime.fromtimestamp(float(timestamp), tz=dt.timezone.utc).isoformat()


def relpath_str(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(APP_ROOT))
    except ValueError:
        return str(path.resolve())


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def parse_optional_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    return int(text)


def parse_optional_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    return float(text)


def parse_optional_american_odds(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    odds = int(text)
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    return odds


def parse_probability(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Probability is required.")
    prob = float(text)
    if prob > 1.0:
        prob = prob / 100.0
    if not (0.0 < prob < 1.0):
        raise ValueError("Probability must be between 0 and 1 (or 0-100%).")
    return prob


def parse_positive_float(value: Any, field_name: str) -> float:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    result = float(text)
    if result <= 0:
        raise ValueError(f"{field_name} must be > 0.")
    return result


def implied_probability_from_american(american_odds: int) -> float:
    if american_odds == 0:
        raise ValueError("American odds cannot be 0.")
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    odds_abs = abs(float(american_odds))
    return odds_abs / (odds_abs + 100.0)


def win_profit_from_american(stake: float, american_odds: int) -> float:
    if american_odds > 0:
        return float(stake) * (float(american_odds) / 100.0)
    return float(stake) * (100.0 / abs(float(american_odds)))


def payout_ratio_from_american(american_odds: int) -> float:
    return win_profit_from_american(1.0, american_odds)


def load_fighter_names(max_names: int = 6000) -> list[str]:
    if max_names <= 0 or not RAW_FIGHTS_CSV.exists():
        return []
    names: set[str] = set()
    with RAW_FIGHTS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n1 = str(row.get("fighter_1_name", "")).strip()
            n2 = str(row.get("fighter_2_name", "")).strip()
            if n1:
                names.add(n1)
            if n2:
                names.add(n2)
            if len(names) >= max_names:
                break
    return sorted(names, key=str.lower)


def resolve_output_path(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (APP_ROOT / candidate).resolve()


def describe_file(path: Path, label: str) -> dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "label": label,
        "path": relpath_str(path),
        "exists": exists,
        "size_mb": round((stat.st_size / 1_000_000.0), 3) if stat is not None else None,
        "modified_at_utc": iso_from_timestamp(stat.st_mtime if stat is not None else None),
    }


def collect_data_status() -> dict[str, Any]:
    assets = [
        describe_file(RAW_FIGHTS_CSV, "Raw fight history"),
        describe_file(METADATA_CSV, "Cleaned metadata"),
        describe_file(SEQUENCE_CSV, "Sequence dataset"),
        describe_file(SCRAPER_CHECKPOINT_DB, "Scraper checkpoint"),
    ]
    existing_times = [row["modified_at_utc"] for row in assets if row["modified_at_utc"]]
    return {
        "assets": assets,
        "last_updated_utc": max(existing_times) if existing_times else None,
    }


def summarize_model_info(model_info: dict[str, Any]) -> str:
    label = str(model_info.get("label") or model_info.get("run_id") or "Model").strip()
    test_auc = model_info.get("test_auc")
    if isinstance(test_auc, (int, float)):
        return f"{label} | test AUC {float(test_auc):.4f}"
    return label


def build_model_info_from_metrics(
    metrics_path: Path,
    metrics: dict[str, Any],
    *,
    label: str | None = None,
    kind: str = "single_run",
) -> dict[str, Any]:
    profile = metrics_path.parent.name
    run_id = metrics_path.stem.replace("metrics_", "")
    test_auc = float(metrics.get("ensemble_test_metrics", {}).get("auc", 0.0))
    val_auc = float(
        metrics.get("ensemble_val_metrics_at_best_threshold", {}).get("auc", 0.0)
    )
    selection_metric = str(metrics.get("selection_metric", "manual_refresh_candidate"))
    selection_score = float(metrics.get("selection_score", test_auc))
    walkforward_score = float(metrics.get("walkforward_score", 0.0))
    walkforward_enabled = bool(metrics.get("walkforward_enabled", False))

    payload = {
        "kind": kind,
        "run_id": run_id,
        "profile": profile,
        "label": label or f"Refresh Candidate ({run_id})",
        "test_auc": test_auc,
        "val_auc": val_auc,
        "selection_metric": selection_metric,
        "selection_score": selection_score,
        "walkforward_score": walkforward_score,
        "walkforward_enabled": walkforward_enabled,
        "metrics_path": relpath_str(metrics_path),
    }

    for key in ("momentum_model_path", "momentum_scaler_path", "xgb_model_path", "xgb_specialists_path"):
        raw = str(metrics.get(key, "")).strip()
        if raw:
            payload[key] = raw
    return payload


def evaluate_wager(
    *,
    pick: str,
    model_probability: float,
    american_odds: int,
    bankroll_units: float,
    fractional_kelly: float = 0.5,
    max_bankroll_fraction: float = 0.03,
) -> dict[str, Any]:
    implied_probability = implied_probability_from_american(american_odds)
    win_profit = win_profit_from_american(1.0, american_odds)
    edge = float(model_probability - implied_probability)
    expected_value_per_unit = float((model_probability * win_profit) - ((1.0 - model_probability) * 1.0))
    payout_ratio = payout_ratio_from_american(american_odds)
    if payout_ratio <= 0:
        kelly_fraction = 0.0
    else:
        q = 1.0 - float(model_probability)
        kelly_fraction = max(((payout_ratio * float(model_probability)) - q) / payout_ratio, 0.0)
    recommended_fraction = min(kelly_fraction * float(fractional_kelly), float(max_bankroll_fraction))
    recommended_units = float(bankroll_units) * max(recommended_fraction, 0.0)

    verdict = "pass"
    if edge >= 0.05 and expected_value_per_unit >= 0.05:
        verdict = "strong"
    elif edge >= 0.02 and expected_value_per_unit > 0.0:
        verdict = "lean"

    return {
        "pick": pick,
        "american_odds": int(american_odds),
        "model_probability": float(model_probability),
        "implied_probability": float(implied_probability),
        "edge": edge,
        "expected_value_per_unit": expected_value_per_unit,
        "kelly_fraction": float(kelly_fraction),
        "recommended_fraction": float(recommended_fraction),
        "recommended_units": float(recommended_units),
        "verdict": verdict,
        "should_bet": verdict in {"lean", "strong"} and expected_value_per_unit > 0.0,
    }


def build_recommendation(
    prediction: dict[str, Any],
    *,
    odds_fighter_1: int | None,
    odds_fighter_2: int | None,
    bankroll_units: float,
    fractional_kelly: float = 0.5,
    max_bankroll_fraction: float = 0.03,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    if odds_fighter_1 is not None:
        candidates.append(
            evaluate_wager(
                pick=str(prediction["fighter_1"]),
                model_probability=float(prediction["p_fighter_1"]),
                american_odds=odds_fighter_1,
                bankroll_units=bankroll_units,
                fractional_kelly=fractional_kelly,
                max_bankroll_fraction=max_bankroll_fraction,
            )
        )
    if odds_fighter_2 is not None:
        candidates.append(
            evaluate_wager(
                pick=str(prediction["fighter_2"]),
                model_probability=float(prediction["p_fighter_2"]),
                american_odds=odds_fighter_2,
                bankroll_units=bankroll_units,
                fractional_kelly=fractional_kelly,
                max_bankroll_fraction=max_bankroll_fraction,
            )
        )

    candidates.sort(
        key=lambda row: (float(row["expected_value_per_unit"]), float(row["edge"])),
        reverse=True,
    )
    best = candidates[0] if candidates else None
    if best is None:
        summary = "Add moneyline odds for both sides to generate a recommendation."
    elif best["should_bet"]:
        summary = (
            f"{best['verdict'].title()} bet: {best['pick']} at {best['american_odds']} "
            f"for {best['recommended_units']:.2f}u."
        )
    else:
        summary = "Pass. Current market prices do not offer a positive model edge."

    return {
        "bankroll_units": float(bankroll_units),
        "fractional_kelly": float(fractional_kelly),
        "max_bankroll_fraction": float(max_bankroll_fraction),
        "candidates": candidates,
        "best": best,
        "recommended": bool(best and best["should_bet"]),
        "summary": summary,
    }


def parse_train_params(payload: dict[str, Any]) -> dict[str, Any]:
    params = dict(DEFAULT_PARAMS)
    params.update({k: v for k, v in payload.items() if v is not None})

    parsed: dict[str, Any] = {
        "input_csv": str(params["input_csv"]).strip(),
        "momentum_model_path": str(params["momentum_model_path"]).strip(),
        "momentum_scaler_path": str(params["momentum_scaler_path"]).strip(),
        "xgb_model_path": str(params["xgb_model_path"]).strip(),
        "metrics_path": str(params["metrics_path"]).strip(),
        "epochs": int(params["epochs"]),
        "patience": int(params["patience"]),
        "batch_size": int(params["batch_size"]),
        "hidden_size": int(params["hidden_size"]),
        "num_layers": int(params["num_layers"]),
        "dropout": float(params["dropout"]),
        "lr": float(params["lr"]),
        "warmup_epochs": int(params["warmup_epochs"]),
        "min_epochs": int(params["min_epochs"]),
        "min_delta": float(params["min_delta"]),
        "weight_decay": float(params["weight_decay"]),
        "grad_clip": float(params["grad_clip"]),
        "bidirectional": parse_bool(params.get("bidirectional"), default=True),
        "use_cross_attention": parse_bool(params.get("use_cross_attention"), default=True),
        "attention_heads": int(params["attention_heads"]),
        "attention_dropout": float(params["attention_dropout"]),
        "static_recency_mode": str(params["static_recency_mode"]).strip().lower(),
        "ema_alpha": float(params["ema_alpha"]),
        "val_fraction": float(params["val_fraction"]),
        "test_fraction": float(params["test_fraction"]),
        "max_fights": parse_optional_int(params.get("max_fights")),
        "seed": int(params["seed"]),
        "device": str(params["device"]).strip().lower(),
        "num_workers": int(params["num_workers"]),
        "xgb_n_estimators": int(params["xgb_n_estimators"]),
        "xgb_lr": float(params["xgb_lr"]),
        "xgb_max_depth": int(params["xgb_max_depth"]),
        "xgb_min_child_weight": float(params["xgb_min_child_weight"]),
        "xgb_subsample": float(params["xgb_subsample"]),
        "xgb_colsample_bytree": float(params["xgb_colsample_bytree"]),
        "xgb_reg_alpha": float(params["xgb_reg_alpha"]),
        "xgb_reg_lambda": float(params["xgb_reg_lambda"]),
        "xgb_gamma": float(params["xgb_gamma"]),
        "xgb_early_stopping": int(params["xgb_early_stopping"]),
        "xgb_n_jobs": int(params["xgb_n_jobs"]),
        "trend_ema_alpha": float(params["trend_ema_alpha"]),
        "use_trend_static_features": parse_bool(params.get("use_trend_static_features"), default=True),
        "use_enhanced_context_static_features": parse_bool(
            params.get("use_enhanced_context_static_features"),
            default=False,
        ),
        "use_oof_stacking": parse_bool(params.get("use_oof_stacking"), default=True),
        "oof_folds": int(params["oof_folds"]),
        "oof_min_train_fights": int(params["oof_min_train_fights"]),
        "use_walkforward_cv": parse_bool(params.get("use_walkforward_cv"), default=False),
        "walkforward_std_penalty": float(params["walkforward_std_penalty"]),
        "use_weight_class_specialists": parse_bool(params.get("use_weight_class_specialists"), default=False),
        "log_level": str(params["log_level"]).strip().upper(),
        "run_label": str(params.get("run_label", "")).strip(),
    }

    if not parsed["input_csv"]:
        raise ValueError("Input CSV path is required.")
    if not parsed["momentum_model_path"]:
        raise ValueError("Momentum model path is required.")
    if not parsed["momentum_scaler_path"]:
        raise ValueError("Momentum scaler path is required.")
    if not parsed["xgb_model_path"]:
        raise ValueError("XGBoost model path is required.")
    if not parsed["metrics_path"]:
        raise ValueError("Metrics path is required.")
    if parsed["epochs"] <= 0:
        raise ValueError("Epochs must be > 0.")
    if parsed["patience"] <= 0:
        raise ValueError("Patience must be > 0.")
    if parsed["batch_size"] <= 0:
        raise ValueError("Batch size must be > 0.")
    if parsed["hidden_size"] <= 0:
        raise ValueError("Hidden size must be > 0.")
    if parsed["num_layers"] <= 0:
        raise ValueError("Num layers must be > 0.")
    if not (0.0 <= parsed["dropout"] < 1.0):
        raise ValueError("Dropout must be in [0, 1).")
    if parsed["lr"] <= 0:
        raise ValueError("Learning rate must be > 0.")
    if parsed["warmup_epochs"] < 0:
        raise ValueError("Warmup epochs must be >= 0.")
    if parsed["min_epochs"] <= 0:
        raise ValueError("Min epochs must be > 0.")
    if parsed["min_delta"] < 0:
        raise ValueError("Min delta must be >= 0.")
    if parsed["min_epochs"] > parsed["epochs"]:
        raise ValueError("Min epochs must be <= epochs.")
    if parsed["weight_decay"] < 0:
        raise ValueError("Weight decay must be >= 0.")
    if parsed["grad_clip"] <= 0:
        raise ValueError("Grad clip must be > 0.")
    if parsed["attention_heads"] <= 0:
        raise ValueError("Attention heads must be > 0.")
    if not (0.0 <= parsed["attention_dropout"] < 1.0):
        raise ValueError("Attention dropout must be in [0, 1).")
    if parsed["static_recency_mode"] not in {"ema", "mean"}:
        raise ValueError("Static recency mode must be 'ema' or 'mean'.")
    if not (0.0 <= parsed["ema_alpha"] <= 1.0):
        raise ValueError("EMA alpha must be in [0, 1].")
    if not (0.01 <= parsed["val_fraction"] < 0.49):
        raise ValueError("Validation fraction must be between 0.01 and 0.49.")
    if not (0.01 <= parsed["test_fraction"] < 0.49):
        raise ValueError("Test fraction must be between 0.01 and 0.49.")
    if parsed["val_fraction"] + parsed["test_fraction"] >= 0.8:
        raise ValueError("Validation + test fractions are too large.")
    if parsed["max_fights"] is not None and parsed["max_fights"] <= 0:
        raise ValueError("Max fights must be empty or > 0.")
    if parsed["num_workers"] < 0:
        raise ValueError("Num workers must be >= 0.")
    if parsed["xgb_n_estimators"] <= 0:
        raise ValueError("XGB n_estimators must be > 0.")
    if parsed["xgb_lr"] <= 0:
        raise ValueError("XGB learning rate must be > 0.")
    if parsed["xgb_max_depth"] <= 0:
        raise ValueError("XGB max_depth must be > 0.")
    if parsed["xgb_min_child_weight"] <= 0:
        raise ValueError("XGB min_child_weight must be > 0.")
    if not (0.0 < parsed["xgb_subsample"] <= 1.0):
        raise ValueError("XGB subsample must be in (0, 1].")
    if not (0.0 < parsed["xgb_colsample_bytree"] <= 1.0):
        raise ValueError("XGB colsample_bytree must be in (0, 1].")
    if parsed["xgb_reg_alpha"] < 0:
        raise ValueError("XGB reg_alpha must be >= 0.")
    if parsed["xgb_reg_lambda"] < 0:
        raise ValueError("XGB reg_lambda must be >= 0.")
    if parsed["xgb_gamma"] < 0:
        raise ValueError("XGB gamma must be >= 0.")
    if parsed["xgb_early_stopping"] <= 0:
        raise ValueError("XGB early stopping rounds must be > 0.")
    if not (0.0 <= parsed["trend_ema_alpha"] <= 1.0):
        raise ValueError("Trend EMA alpha must be in [0, 1].")
    if parsed["oof_folds"] <= 0:
        raise ValueError("OOF folds must be > 0.")
    if parsed["oof_min_train_fights"] <= 0:
        raise ValueError("OOF min train fights must be > 0.")
    if parsed["walkforward_std_penalty"] < 0:
        raise ValueError("Walk-forward std penalty must be >= 0.")
    if parsed["device"] not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("Device must be one of auto/cpu/cuda/mps.")
    if parsed["log_level"] not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("Log level must be DEBUG/INFO/WARNING/ERROR.")

    return parsed


def build_train_command(params: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, "-u", str(TRAIN_SCRIPT)]
    cmd += ["--input-csv", params["input_csv"]]
    cmd += ["--momentum-model-path", params["momentum_model_path"]]
    cmd += ["--momentum-scaler-path", params["momentum_scaler_path"]]
    cmd += ["--xgb-model-path", params["xgb_model_path"]]
    cmd += ["--metrics-path", params["metrics_path"]]
    cmd += ["--epochs", str(params["epochs"])]
    cmd += ["--patience", str(params["patience"])]
    cmd += ["--batch-size", str(params["batch_size"])]
    cmd += ["--hidden-size", str(params["hidden_size"])]
    cmd += ["--num-layers", str(params["num_layers"])]
    cmd += ["--dropout", str(params["dropout"])]
    cmd += ["--lr", str(params["lr"])]
    cmd += ["--warmup-epochs", str(params["warmup_epochs"])]
    cmd += ["--min-epochs", str(params["min_epochs"])]
    cmd += ["--min-delta", str(params["min_delta"])]
    cmd += ["--weight-decay", str(params["weight_decay"])]
    cmd += ["--grad-clip", str(params["grad_clip"])]
    cmd += ["--attention-heads", str(params["attention_heads"])]
    cmd += ["--attention-dropout", str(params["attention_dropout"])]
    cmd += ["--static-recency-mode", params["static_recency_mode"]]
    cmd += ["--ema-alpha", str(params["ema_alpha"])]
    cmd += ["--val-fraction", str(params["val_fraction"])]
    cmd += ["--test-fraction", str(params["test_fraction"])]
    if params["max_fights"] is not None:
        cmd += ["--max-fights", str(params["max_fights"])]
    cmd += ["--seed", str(params["seed"])]
    cmd += ["--device", params["device"]]
    cmd += ["--num-workers", str(params["num_workers"])]
    cmd += ["--xgb-n-estimators", str(params["xgb_n_estimators"])]
    cmd += ["--xgb-lr", str(params["xgb_lr"])]
    cmd += ["--xgb-max-depth", str(params["xgb_max_depth"])]
    cmd += ["--xgb-min-child-weight", str(params["xgb_min_child_weight"])]
    cmd += ["--xgb-subsample", str(params["xgb_subsample"])]
    cmd += ["--xgb-colsample-bytree", str(params["xgb_colsample_bytree"])]
    cmd += ["--xgb-reg-alpha", str(params["xgb_reg_alpha"])]
    cmd += ["--xgb-reg-lambda", str(params["xgb_reg_lambda"])]
    cmd += ["--xgb-gamma", str(params["xgb_gamma"])]
    cmd += ["--xgb-early-stopping", str(params["xgb_early_stopping"])]
    cmd += ["--xgb-n-jobs", str(params["xgb_n_jobs"])]
    cmd += ["--trend-ema-alpha", str(params["trend_ema_alpha"])]
    cmd += ["--oof-folds", str(params["oof_folds"])]
    cmd += ["--oof-min-train-fights", str(params["oof_min_train_fights"])]
    cmd += ["--walkforward-std-penalty", str(params["walkforward_std_penalty"])]
    cmd += ["--log-level", params["log_level"]]

    cmd.append("--bidirectional" if params["bidirectional"] else "--no-bidirectional")
    cmd.append("--use-cross-attention" if params["use_cross_attention"] else "--no-cross-attention")
    if not params["use_trend_static_features"]:
        cmd.append("--disable-trend-static-features")
    cmd.append(
        "--enhanced-context-static-features"
        if params["use_enhanced_context_static_features"]
        else "--no-enhanced-context-static-features"
    )
    cmd.append("--use-oof-stacking" if params["use_oof_stacking"] else "--no-oof-stacking")
    cmd.append("--use-walkforward-cv" if params["use_walkforward_cv"] else "--no-walkforward-cv")
    cmd.append(
        "--use-weight-class-specialists"
        if params["use_weight_class_specialists"]
        else "--no-weight-class-specialists"
    )
    return cmd


def build_refresh_training_params() -> dict[str, Any]:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    SITE_REFRESH_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"site_refresh_{timestamp}"
    params = dict(DEFAULT_PARAMS)
    params.update(
        {
            "momentum_model_path": f"data/model_cache/site_refresh_candidates/momentum_{suffix}.pth",
            "momentum_scaler_path": f"data/model_cache/site_refresh_candidates/momentum_scalers_{suffix}.pkl",
            "xgb_model_path": f"data/model_cache/site_refresh_candidates/xgb_{suffix}.json",
            "metrics_path": f"data/model_cache/site_refresh_candidates/metrics_{suffix}.json",
            "run_label": f"Site Refresh Candidate ({timestamp})",
        }
    )
    return params


@dataclass
class TrainingRun:
    run_id: int
    started_at_utc: str
    params: dict[str, Any]
    command: list[str]
    metrics_path_abs: Path
    process: subprocess.Popen[str] | None = None
    running: bool = True
    finished_at_utc: str | None = None
    return_code: int | None = None
    logs: list[str] = field(default_factory=list)
    stop_requested: bool = False
    metrics: dict[str, Any] | None = None
    error: str | None = None


class TrainingManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._run_counter = 0
        self._current: TrainingRun | None = None
        self._max_log_lines = 8000
        self._latest_candidate_model: dict[str, Any] | None = None

    def start(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._current and self._current.running:
                raise RuntimeError("A training run is already in progress.")
            if not TRAIN_SCRIPT.exists():
                raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")

            self._run_counter += 1
            cmd = build_train_command(params)
            proc = subprocess.Popen(
                cmd,
                cwd=str(APP_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            run = TrainingRun(
                run_id=self._run_counter,
                started_at_utc=utc_now_iso(),
                params=params,
                command=cmd,
                metrics_path_abs=resolve_output_path(params["metrics_path"]),
                process=proc,
                running=True,
            )
            run.logs.append(f"[{run.started_at_utc}] Started run #{run.run_id}")
            if params.get("run_label"):
                run.logs.append(f"Label: {params['run_label']}")
            run.logs.append(f"Command: {' '.join(cmd)}")
            self._current = run

            worker = Thread(target=self._consume_output, args=(run,), daemon=True)
            worker.start()
            return self.snapshot()

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._current and self._current.running)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._current or not self._current.running:
                raise RuntimeError("No active training run to stop.")
            run = self._current
            run.stop_requested = True
            if run.process and run.process.poll() is None:
                run.process.terminate()
                run.logs.append(f"[{utc_now_iso()}] Stop requested. Sent terminate signal.")
            return self.snapshot()

    def latest_candidate_model(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._latest_candidate_model) if self._latest_candidate_model else None

    def snapshot(self, tail_lines: int = 500) -> dict[str, Any]:
        with self._lock:
            latest_candidate = dict(self._latest_candidate_model) if self._latest_candidate_model else None
            if self._current is None:
                return {
                    "has_run": False,
                    "state": "idle",
                    "running": False,
                    "log_tail": "",
                    "latest_candidate_model": latest_candidate,
                }
            run = self._current
            if run.running:
                state = "running"
            elif run.return_code == 0:
                state = "succeeded"
            elif run.stop_requested:
                state = "stopped"
            else:
                state = "failed"

            tail = "\n".join(run.logs[-max(1, tail_lines) :])
            return {
                "has_run": True,
                "state": state,
                "running": run.running,
                "run_id": run.run_id,
                "started_at_utc": run.started_at_utc,
                "finished_at_utc": run.finished_at_utc,
                "return_code": run.return_code,
                "stop_requested": run.stop_requested,
                "error": run.error,
                "params": run.params,
                "command": run.command,
                "metrics_path": str(run.metrics_path_abs),
                "metrics": run.metrics,
                "log_tail": tail,
                "latest_candidate_model": latest_candidate,
            }

    def _append_log(self, run: TrainingRun, line: str) -> None:
        run.logs.append(line.rstrip("\n"))
        if len(run.logs) > self._max_log_lines:
            del run.logs[:1200]

    def _consume_output(self, run: TrainingRun) -> None:
        return_code = -1
        try:
            proc = run.process
            if proc is None:
                raise RuntimeError("Missing process handle for run.")
            if proc.stdout is not None:
                for line in proc.stdout:
                    with self._lock:
                        if self._current is run:
                            self._append_log(run, line)
            return_code = proc.wait()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                run.error = str(exc)
                self._append_log(run, f"[{utc_now_iso()}] Internal monitor error: {exc}")
            return_code = -1
        finally:
            with self._lock:
                run.running = False
                run.return_code = return_code
                run.finished_at_utc = utc_now_iso()
                run.process = None
                self._append_log(run, f"[{run.finished_at_utc}] Run completed with exit code {return_code}.")
                if run.metrics_path_abs.exists():
                    try:
                        with run.metrics_path_abs.open("r", encoding="utf-8") as f:
                            run.metrics = json.load(f)
                    except Exception as exc:  # noqa: BLE001
                        run.error = str(exc)
                        self._append_log(run, f"[{utc_now_iso()}] Could not parse metrics file: {exc}")
                if return_code == 0 and run.metrics:
                    candidate_label = str(run.params.get("run_label") or "").strip() or None
                    self._latest_candidate_model = build_model_info_from_metrics(
                        run.metrics_path_abs,
                        run.metrics,
                        label=candidate_label,
                    )


@dataclass(frozen=True)
class PipelineAction:
    key: str
    label: str
    command: list[str]
    description: str


PIPELINE_ACTIONS: dict[str, PipelineAction] = {
    "update_data": PipelineAction(
        key="update_data",
        label="Update Data",
        command=[sys.executable, "-u", str(SCRAPER_SCRIPT)],
        description="Continue scraper from checkpoint and append/update latest fight data.",
    ),
    "reset_data": PipelineAction(
        key="reset_data",
        label="Reset Data + Rescrape",
        command=[
            sys.executable,
            "-u",
            str(SCRAPER_SCRIPT),
            "--refresh-processed-events",
            "--refresh-existing-fights",
        ],
        description="Delete fight data/checkpoint and scrape a fresh full dataset.",
    ),
    "build_sequences": PipelineAction(
        key="build_sequences",
        label="Build Sequences",
        command=[sys.executable, "-u", str(SEQUENCE_SCRIPT)],
        description="Rebuild fighter-history sequence CSV from raw fight details.",
    ),
    "audit_data": PipelineAction(
        key="audit_data",
        label="Audit Data",
        command=[sys.executable, "-u", str(AUDIT_SCRIPT)],
        description="Run data integrity audit across raw and sequence datasets.",
    ),
}


@dataclass
class PipelineRun:
    run_id: int
    action_key: str
    action_label: str
    started_at_utc: str
    command: list[str]
    process: subprocess.Popen[str] | None = None
    running: bool = True
    finished_at_utc: str | None = None
    return_code: int | None = None
    logs: list[str] = field(default_factory=list)
    stop_requested: bool = False
    error: str | None = None


class PipelineManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._run_counter = 0
        self._current: PipelineRun | None = None
        self._max_log_lines = 8000

    def _cleanup_for_reset(self, run: PipelineRun) -> None:
        targets = [
            RAW_FIGHTS_CSV,
            SEQUENCE_CSV,
            SCRAPER_CHECKPOINT_DB,
            SCRAPER_CHECKPOINT_DB.with_name(SCRAPER_CHECKPOINT_DB.name + "-wal"),
            SCRAPER_CHECKPOINT_DB.with_name(SCRAPER_CHECKPOINT_DB.name + "-shm"),
        ]
        run.logs.append(f"[{utc_now_iso()}] Reset requested. Deleting existing data artifacts...")
        for path in targets:
            try:
                path.unlink()
                run.logs.append(f"Deleted: {path.relative_to(APP_ROOT)}")
            except FileNotFoundError:
                run.logs.append(f"Skip (not found): {path.relative_to(APP_ROOT)}")
            except Exception as exc:  # noqa: BLE001
                run.logs.append(f"Warning: could not delete {path.relative_to(APP_ROOT)}: {exc}")

    def start(self, action_key: str) -> dict[str, Any]:
        action = PIPELINE_ACTIONS.get(action_key)
        if action is None:
            raise ValueError(
                f"Unknown action '{action_key}'. Valid actions: {', '.join(sorted(PIPELINE_ACTIONS))}"
            )
        with self._lock:
            if self._current and self._current.running:
                raise RuntimeError("A data pipeline job is already in progress.")

            script_path = Path(action.command[2]) if len(action.command) >= 3 else None
            if script_path is not None and not script_path.exists():
                raise FileNotFoundError(f"Pipeline script not found: {script_path}")

            self._run_counter += 1
            run = PipelineRun(
                run_id=self._run_counter,
                action_key=action.key,
                action_label=action.label,
                started_at_utc=utc_now_iso(),
                command=action.command,
                running=True,
            )
            run.logs.append(f"[{run.started_at_utc}] Started job #{run.run_id}: {run.action_label}")
            run.logs.append(f"Description: {action.description}")
            if action.key == "reset_data":
                self._cleanup_for_reset(run)

            proc = subprocess.Popen(
                action.command,
                cwd=str(APP_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            run.process = proc
            run.logs.append(f"Command: {' '.join(action.command)}")
            self._current = run

            worker = Thread(target=self._consume_output, args=(run,), daemon=True)
            worker.start()
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._current or not self._current.running:
                raise RuntimeError("No active data pipeline job to stop.")
            run = self._current
            run.stop_requested = True
            if run.process and run.process.poll() is None:
                run.process.terminate()
                run.logs.append(f"[{utc_now_iso()}] Stop requested. Sent terminate signal.")
            return self.snapshot()

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._current and self._current.running)

    def snapshot(self, tail_lines: int = 500) -> dict[str, Any]:
        with self._lock:
            if self._current is None:
                return {
                    "has_run": False,
                    "state": "idle",
                    "running": False,
                    "log_tail": "",
                }
            run = self._current
            if run.running:
                state = "running"
            elif run.return_code == 0:
                state = "succeeded"
            elif run.stop_requested:
                state = "stopped"
            else:
                state = "failed"

            tail = "\n".join(run.logs[-max(1, tail_lines) :])
            return {
                "has_run": True,
                "state": state,
                "running": run.running,
                "run_id": run.run_id,
                "action_key": run.action_key,
                "action_label": run.action_label,
                "started_at_utc": run.started_at_utc,
                "finished_at_utc": run.finished_at_utc,
                "return_code": run.return_code,
                "stop_requested": run.stop_requested,
                "error": run.error,
                "command": run.command,
                "log_tail": tail,
            }

    def _append_log(self, run: PipelineRun, line: str) -> None:
        run.logs.append(line.rstrip("\n"))
        if len(run.logs) > self._max_log_lines:
            del run.logs[:1200]

    def _consume_output(self, run: PipelineRun) -> None:
        return_code = -1
        try:
            proc = run.process
            if proc is None:
                raise RuntimeError("Missing process handle for pipeline job.")
            if proc.stdout is not None:
                for line in proc.stdout:
                    with self._lock:
                        if self._current is run:
                            self._append_log(run, line)
            return_code = proc.wait()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                run.error = str(exc)
                self._append_log(run, f"[{utc_now_iso()}] Internal monitor error: {exc}")
            return_code = -1
        finally:
            with self._lock:
                run.running = False
                run.return_code = return_code
                run.finished_at_utc = utc_now_iso()
                run.process = None
                self._append_log(
                    run,
                    f"[{run.finished_at_utc}] Job completed with exit code {return_code}.",
                )


class ResearchStore:
    def __init__(self, db_path: Path, *, legacy_json_path: Path, legacy_csv_path: Path) -> None:
        self._db_path = db_path
        self._legacy_json_path = legacy_json_path
        self._legacy_csv_path = legacy_csv_path
        self._lock = RLock()
        self._init_db()
        self._migrate_legacy_if_needed()

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    event_name TEXT DEFAULT '',
                    event_date TEXT,
                    fighter_1 TEXT NOT NULL,
                    fighter_2 TEXT NOT NULL,
                    weight_class TEXT,
                    gender TEXT,
                    scheduled_rounds INTEGER,
                    is_title_bout INTEGER NOT NULL DEFAULT 0,
                    model_key TEXT,
                    model_label TEXT,
                    model_test_auc REAL,
                    p_fighter_1 REAL NOT NULL,
                    p_fighter_2 REAL NOT NULL,
                    winner TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    odds_fighter_1 INTEGER,
                    odds_fighter_2 INTEGER,
                    recommendation_pick TEXT,
                    recommendation_verdict TEXT,
                    recommendation_edge REAL,
                    recommendation_units REAL,
                    notes TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    prediction_id INTEGER,
                    event_name TEXT NOT NULL,
                    event_date TEXT,
                    fighter_1 TEXT,
                    fighter_2 TEXT,
                    matchup TEXT NOT NULL,
                    market_type TEXT NOT NULL DEFAULT 'moneyline',
                    sportsbook TEXT DEFAULT '',
                    pick TEXT NOT NULL,
                    american_odds INTEGER NOT NULL,
                    model_probability REAL NOT NULL,
                    implied_probability REAL NOT NULL,
                    edge REAL NOT NULL,
                    expected_value REAL NOT NULL,
                    kelly_fraction REAL,
                    recommended_units REAL,
                    stake REAL NOT NULL,
                    is_recommended INTEGER NOT NULL DEFAULT 0,
                    result TEXT NOT NULL DEFAULT 'open',
                    settled_at_utc TEXT,
                    realized_pnl REAL NOT NULL DEFAULT 0.0,
                    notes TEXT DEFAULT '',
                    FOREIGN KEY(prediction_id) REFERENCES predictions(id)
                );
                """
            )

    def _table_count(self, table_name: str) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
            return int(row["count"]) if row is not None else 0

    def _migrate_legacy_if_needed(self) -> None:
        if self._table_count("bets") > 0:
            return

        migrated_any = False
        if self._legacy_json_path.exists():
            try:
                payload = json.loads(self._legacy_json_path.read_text(encoding="utf-8"))
                rows = payload.get("bets", []) if isinstance(payload, dict) else payload
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, dict):
                            self._insert_legacy_bet(row)
                            migrated_any = True
            except Exception:
                pass

        if not migrated_any and self._legacy_csv_path.exists():
            try:
                with self._legacy_csv_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self._insert_legacy_csv_bet(row)
                        migrated_any = True
            except Exception:
                pass

    def _insert_legacy_bet(self, row: dict[str, Any]) -> None:
        event_name = str(row.get("event_name", "")).strip() or "Legacy Import"
        fighter_1 = str(row.get("fighter_1", "")).strip()
        fighter_2 = str(row.get("fighter_2", "")).strip()
        pick = str(row.get("pick", "")).strip()
        matchup = str(row.get("matchup", "")).strip() or f"{fighter_1} vs {fighter_2}".strip()
        model_probability = float(row.get("model_probability", 0.0) or 0.0)
        american_odds = int(row.get("american_odds", 0) or 0)
        stake = float(row.get("stake", 0.0) or 0.0)
        if not matchup or not pick or american_odds == 0 or stake <= 0 or model_probability <= 0:
            return
        implied_probability = float(row.get("implied_probability", implied_probability_from_american(american_odds)))
        edge = float(row.get("edge", model_probability - implied_probability))
        expected_value = float(
            row.get(
                "expected_value",
                (model_probability * win_profit_from_american(stake, american_odds)) - ((1.0 - model_probability) * stake),
            )
        )
        realized_pnl = float(row.get("realized_pnl", 0.0) or 0.0)
        result = str(row.get("result", "open")).strip().lower() or "open"
        notes = str(row.get("notes", "")).strip()

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO bets (
                    created_at_utc, prediction_id, event_name, event_date, fighter_1, fighter_2, matchup,
                    market_type, sportsbook, pick, american_odds, model_probability, implied_probability,
                    edge, expected_value, kelly_fraction, recommended_units, stake, is_recommended, result,
                    settled_at_utc, realized_pnl, notes
                ) VALUES (?, NULL, ?, ?, ?, ?, ?, 'moneyline', ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row.get("created_at_utc") or utc_now_iso()),
                    event_name,
                    row.get("event_date"),
                    fighter_1,
                    fighter_2,
                    matchup,
                    str(row.get("sportsbook", "")).strip(),
                    pick,
                    american_odds,
                    model_probability,
                    implied_probability,
                    edge,
                    expected_value,
                    stake,
                    0,
                    result,
                    row.get("settled_at_utc"),
                    realized_pnl,
                    notes,
                ),
            )

    def _insert_legacy_csv_bet(self, row: dict[str, Any]) -> None:
        try:
            model_probability = parse_probability(row.get("model_prob_pick"))
            american_odds = int(str(row.get("odds_american", "")).strip())
            stake = float(str(row.get("stake", "")).strip())
        except Exception:
            return
        if american_odds == 0 or stake <= 0:
            return

        event_name = str(row.get("event_date", "")).strip() or "Legacy Import"
        fighter_1 = str(row.get("fighter_1", "")).strip()
        fighter_2 = str(row.get("fighter_2", "")).strip()
        pick = str(row.get("pick", "")).strip()
        matchup = f"{fighter_1} vs {fighter_2}".strip()
        if not pick or not matchup:
            return

        implied_probability = float(row.get("implied_prob_at_bet") or implied_probability_from_american(american_odds))
        edge = float(row.get("model_edge") or (model_probability - implied_probability))
        potential_profit = float(row.get("potential_profit") or win_profit_from_american(stake, american_odds))
        expected_value = (model_probability * potential_profit) - ((1.0 - model_probability) * stake)
        pnl = float(row.get("pnl") or 0.0)
        result = str(row.get("result", "open")).strip().lower() or "open"

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO bets (
                    created_at_utc, prediction_id, event_name, event_date, fighter_1, fighter_2, matchup,
                    market_type, sportsbook, pick, american_odds, model_probability, implied_probability,
                    edge, expected_value, kelly_fraction, recommended_units, stake, is_recommended, result,
                    settled_at_utc, realized_pnl, notes
                ) VALUES (?, NULL, ?, ?, ?, ?, ?, 'moneyline', ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row.get("created_at_utc") or utc_now_iso()),
                    event_name,
                    str(row.get("event_date", "")).strip() or None,
                    fighter_1,
                    fighter_2,
                    matchup,
                    str(row.get("sportsbook", "")).strip(),
                    pick,
                    american_odds,
                    model_probability,
                    implied_probability,
                    edge,
                    expected_value,
                    stake,
                    0,
                    result,
                    str(row.get("settled_at_utc", "")).strip() or None,
                    pnl,
                    str(row.get("notes", "")).strip(),
                ),
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def add_prediction(
        self,
        *,
        prediction: dict[str, Any],
        request_payload: dict[str, Any],
        recommendation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        event_name = str(request_payload.get("event_name", "")).strip()
        notes = str(request_payload.get("notes", "")).strip()
        best = recommendation.get("best") if recommendation else None

        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO predictions (
                    created_at_utc, event_name, event_date, fighter_1, fighter_2, weight_class, gender,
                    scheduled_rounds, is_title_bout, model_key, model_label, model_test_auc, p_fighter_1,
                    p_fighter_2, winner, confidence, odds_fighter_1, odds_fighter_2, recommendation_pick,
                    recommendation_verdict, recommendation_edge, recommendation_units, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now_iso(),
                    event_name,
                    str(prediction.get("event_date", "")).strip() or None,
                    str(prediction.get("fighter_1", "")).strip(),
                    str(prediction.get("fighter_2", "")).strip(),
                    str(prediction.get("weight_class", "")).strip(),
                    str(prediction.get("gender", "")).strip(),
                    int(prediction.get("scheduled_rounds", 0) or 0),
                    1 if bool(prediction.get("is_title_bout")) else 0,
                    str(prediction.get("model", "")).strip(),
                    str(prediction.get("model_label", "")).strip(),
                    float(prediction.get("model_test_auc", 0.0) or 0.0),
                    float(prediction.get("p_fighter_1", 0.0) or 0.0),
                    float(prediction.get("p_fighter_2", 0.0) or 0.0),
                    str(prediction.get("winner", "")).strip(),
                    float(prediction.get("confidence", 0.0) or 0.0),
                    parse_optional_american_odds(request_payload.get("odds_fighter_1")),
                    parse_optional_american_odds(request_payload.get("odds_fighter_2")),
                    str(best.get("pick", "")).strip() if isinstance(best, dict) else None,
                    str(best.get("verdict", "")).strip() if isinstance(best, dict) else None,
                    float(best.get("edge", 0.0) or 0.0) if isinstance(best, dict) else None,
                    float(best.get("recommended_units", 0.0) or 0.0) if isinstance(best, dict) else None,
                    notes,
                ),
            )
            new_id = int(cursor.lastrowid)
            row = conn.execute("SELECT * FROM predictions WHERE id = ?", (new_id,)).fetchone()
            return dict(row) if row is not None else {"id": new_id}

    def add_bet(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_name = str(payload.get("event_name", "")).strip()
        fighter_1 = str(payload.get("fighter_1", "")).strip()
        fighter_2 = str(payload.get("fighter_2", "")).strip()
        pick = str(payload.get("pick", "")).strip()
        sportsbook = str(payload.get("sportsbook", "")).strip()
        notes = str(payload.get("notes", "")).strip()
        event_date = str(payload.get("event_date", "")).strip() or None
        market_type = str(payload.get("market_type", "moneyline")).strip() or "moneyline"
        prediction_id = parse_optional_int(payload.get("prediction_id"))
        is_recommended = 1 if parse_bool(payload.get("is_recommended"), default=False) else 0

        matchup = str(payload.get("matchup", "")).strip()
        if fighter_1 and fighter_2:
            matchup = f"{fighter_1} vs {fighter_2}"
            if fighter_1.lower() == fighter_2.lower():
                raise ValueError("Choose two different fighters.")
        if not matchup:
            raise ValueError("Matchup is required.")
        if not event_name:
            raise ValueError("Event name is required.")
        if not pick:
            raise ValueError("Pick is required.")

        american_odds = parse_optional_american_odds(payload.get("american_odds"))
        if american_odds is None:
            raise ValueError("American odds are required.")
        stake = parse_positive_float(payload.get("stake"), "Stake")

        model_probability_raw = payload.get("model_probability")
        if str(model_probability_raw or "").strip():
            model_probability = parse_probability(model_probability_raw)
        else:
            if not fighter_1 or not fighter_2:
                raise ValueError("fighter_1 and fighter_2 are required when model_probability is omitted.")
            prediction = predictor_service.predict(payload)
            if pick.lower() == fighter_1.lower():
                model_probability = float(prediction["p_fighter_1"])
            elif pick.lower() == fighter_2.lower():
                model_probability = float(prediction["p_fighter_2"])
            else:
                raise ValueError("Pick must match fighter_1 or fighter_2.")

        wager = evaluate_wager(
            pick=pick,
            model_probability=model_probability,
            american_odds=american_odds,
            bankroll_units=max(stake, 1.0),
        )
        expected_value = float(
            (model_probability * win_profit_from_american(stake, american_odds)) - ((1.0 - model_probability) * stake)
        )

        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO bets (
                    created_at_utc, prediction_id, event_name, event_date, fighter_1, fighter_2, matchup,
                    market_type, sportsbook, pick, american_odds, model_probability, implied_probability,
                    edge, expected_value, kelly_fraction, recommended_units, stake, is_recommended,
                    result, settled_at_utc, realized_pnl, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', NULL, 0.0, ?)
                """,
                (
                    utc_now_iso(),
                    prediction_id,
                    event_name,
                    event_date,
                    fighter_1 or None,
                    fighter_2 or None,
                    matchup,
                    market_type,
                    sportsbook,
                    pick,
                    american_odds,
                    float(model_probability),
                    float(wager["implied_probability"]),
                    float(wager["edge"]),
                    expected_value,
                    float(wager["kelly_fraction"]),
                    float(wager["recommended_units"]),
                    float(stake),
                    is_recommended,
                    notes,
                ),
            )
            new_id = int(cursor.lastrowid)
            row = conn.execute("SELECT * FROM bets WHERE id = ?", (new_id,)).fetchone()
            return dict(row) if row is not None else {"id": new_id}

    def settle_bet(self, bet_id: int, result: str) -> dict[str, Any]:
        settled_result = str(result).strip().lower()
        if settled_result not in {"win", "loss", "push"}:
            raise ValueError("Result must be one of: win, loss, push.")

        with self._lock, self._connect() as conn:
            target = conn.execute("SELECT * FROM bets WHERE id = ?", (int(bet_id),)).fetchone()
            if target is None:
                raise ValueError(f"Bet id {bet_id} not found.")
            if str(target["result"]).lower() != "open":
                raise ValueError(f"Bet id {bet_id} is already settled.")

            stake = float(target["stake"])
            odds = int(target["american_odds"])
            if settled_result == "win":
                realized_pnl = win_profit_from_american(stake, odds)
            elif settled_result == "loss":
                realized_pnl = -stake
            else:
                realized_pnl = 0.0

            conn.execute(
                """
                UPDATE bets
                SET result = ?, realized_pnl = ?, settled_at_utc = ?
                WHERE id = ?
                """,
                (settled_result, float(realized_pnl), utc_now_iso(), int(bet_id)),
            )
            row = conn.execute("SELECT * FROM bets WHERE id = ?", (int(bet_id),)).fetchone()
            return dict(row) if row is not None else {"id": int(bet_id)}

    def snapshot(self, *, limit_bets: int = 80, limit_predictions: int = 40) -> dict[str, Any]:
        limit_bets = min(max(limit_bets, 10), 500)
        limit_predictions = min(max(limit_predictions, 10), 500)

        with self._lock, self._connect() as conn:
            summary_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_bets,
                    SUM(CASE WHEN result = 'open' THEN 1 ELSE 0 END) AS open_bets,
                    SUM(CASE WHEN result != 'open' THEN 1 ELSE 0 END) AS settled_bets,
                    SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) AS losses,
                    SUM(CASE WHEN result = 'push' THEN 1 ELSE 0 END) AS pushes,
                    SUM(CASE WHEN result != 'open' THEN stake ELSE 0 END) AS total_stake_settled,
                    SUM(CASE WHEN result != 'open' THEN realized_pnl ELSE 0 END) AS total_pnl_settled,
                    AVG(edge) AS avg_edge
                FROM bets
                """
            ).fetchone()
            prediction_summary_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_predictions,
                    SUM(CASE WHEN recommendation_verdict IN ('lean', 'strong') THEN 1 ELSE 0 END) AS recommended_predictions,
                    AVG(confidence) AS avg_confidence,
                    MAX(created_at_utc) AS last_prediction_at_utc
                FROM predictions
                """
            ).fetchone()
            bet_rows = conn.execute(
                "SELECT * FROM bets ORDER BY id DESC LIMIT ?",
                (limit_bets,),
            ).fetchall()
            prediction_rows = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
                (limit_predictions,),
            ).fetchall()

        total_stake_settled = float(summary_row["total_stake_settled"] or 0.0)
        total_pnl_settled = float(summary_row["total_pnl_settled"] or 0.0)
        settled_bets = int(summary_row["settled_bets"] or 0)
        wins = int(summary_row["wins"] or 0)

        return {
            "store_path": relpath_str(self._db_path),
            "summary": {
                "total_bets": int(summary_row["total_bets"] or 0),
                "open_bets": int(summary_row["open_bets"] or 0),
                "settled_bets": settled_bets,
                "wins": wins,
                "losses": int(summary_row["losses"] or 0),
                "pushes": int(summary_row["pushes"] or 0),
                "win_rate": (wins / settled_bets) if settled_bets else None,
                "total_stake_settled": total_stake_settled,
                "total_pnl_settled": total_pnl_settled,
                "roi": (total_pnl_settled / total_stake_settled) if total_stake_settled > 0 else None,
                "avg_edge": float(summary_row["avg_edge"] or 0.0) if summary_row["total_bets"] else None,
                "total_predictions": int(prediction_summary_row["total_predictions"] or 0),
                "recommended_predictions": int(prediction_summary_row["recommended_predictions"] or 0),
                "avg_confidence": float(prediction_summary_row["avg_confidence"] or 0.0)
                if prediction_summary_row["total_predictions"]
                else None,
                "last_prediction_at_utc": prediction_summary_row["last_prediction_at_utc"],
            },
            "bets": [dict(row) for row in bet_rows],
            "predictions": [dict(row) for row in prediction_rows],
        }


class DeployedModelRegistry:
    def __init__(
        self,
        *,
        state_path: Path,
        default_model_info: dict[str, Any],
        baseline_model_info: dict[str, Any],
    ) -> None:
        self._state_path = state_path
        self._default_model_info = dict(default_model_info)
        self._baseline_model_info = dict(baseline_model_info)
        self._lock = RLock()
        self._state = self._load()

    def _default_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "updated_at_utc": utc_now_iso(),
            "deployed_model": self._default_model_info,
        }

    def _load(self) -> dict[str, Any]:
        if not self._state_path.exists():
            state = self._default_state()
            self._save(state)
            return state
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            payload = self._default_state()
            self._save(payload)
            return payload
        if not isinstance(payload, dict) or not isinstance(payload.get("deployed_model"), dict):
            payload = self._default_state()
            self._save(payload)
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state.get("deployed_model", self._default_model_info))

    def catalog(self, latest_candidate: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        deployed_metrics_path = str(self.get().get("metrics_path", "")).strip()
        catalog_entries = [
            ("best_weighted_ensemble", dict(self._default_model_info)),
            ("best_single_run", dict(self._baseline_model_info)),
        ]
        if latest_candidate:
            catalog_entries.append(("latest_candidate", dict(latest_candidate)))

        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for key, model in catalog_entries:
            model_metrics_path = str(model.get("metrics_path", "")).strip()
            dedupe_key = f"{model.get('kind', '')}:{model_metrics_path}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            rows.append(
                {
                    "key": key,
                    "is_deployed": model_metrics_path == deployed_metrics_path,
                    "summary": summarize_model_info(model),
                    "model": model,
                }
            )
        return rows

    def deploy(self, model_key: str, latest_candidate: dict[str, Any] | None = None) -> dict[str, Any]:
        model_key = str(model_key).strip()
        model_lookup = {
            "best_weighted_ensemble": dict(self._default_model_info),
            "best_single_run": dict(self._baseline_model_info),
        }
        if latest_candidate is not None:
            model_lookup["latest_candidate"] = dict(latest_candidate)
        if model_key not in model_lookup:
            raise ValueError(f"Unknown model key '{model_key}'.")

        deployed_model = model_lookup[model_key]
        with self._lock:
            self._state = {
                "version": 1,
                "updated_at_utc": utc_now_iso(),
                "deployed_model": deployed_model,
            }
            self._save(self._state)
            return dict(deployed_model)


class PredictorService:
    def __init__(
        self,
        *,
        raw_csv_path: Path,
        metadata_csv_path: Path,
        model_registry: DeployedModelRegistry,
    ) -> None:
        self._raw_csv_path = raw_csv_path
        self._metadata_csv_path = metadata_csv_path
        self._model_registry = model_registry
        self._lock = RLock()
        self._predictor: Any = None
        self._artifact_state: dict[str, tuple[str, float | None]] = {}

    @staticmethod
    def _get_mtime(path: Path) -> float | None:
        try:
            return float(path.stat().st_mtime)
        except FileNotFoundError:
            return None

    def force_reload(self) -> None:
        with self._lock:
            self._predictor = None
            self._artifact_state = {}

    def deployed_model(self) -> dict[str, Any]:
        return self._model_registry.get()

    def model_status(self, latest_candidate: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "deployed_model": self._model_registry.get(),
            "catalog": self._model_registry.catalog(latest_candidate=latest_candidate),
            "loaded": self._predictor is not None,
        }

    def _resolve_artifact_paths(self, model_info: dict[str, Any]) -> dict[str, Path]:
        metrics_path = resolve_output_path(str(model_info["metrics_path"]))
        kind = str(model_info.get("kind", "single_run")).strip().lower()
        artifact_paths: dict[str, Path] = {
            "raw_csv_path": self._raw_csv_path,
            "metadata_csv_path": self._metadata_csv_path,
            "metrics_path": metrics_path,
        }
        if kind in {"average_ensemble", "weighted_ensemble"}:
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
            components = report.get("model_components", [])
            if not isinstance(components, list) or not components:
                raise ValueError(f"Ensemble report {metrics_path} has no model_components.")
            for idx, component in enumerate(components):
                metrics_json_raw = str(component.get("metrics_json", "")).strip()
                if not metrics_json_raw:
                    raise ValueError(f"Ensemble report {metrics_path} has a component without metrics_json.")
                component_metrics_path = resolve_output_path(metrics_json_raw)
                artifact_paths[f"component_metrics_path_{idx:02d}"] = component_metrics_path
                metrics = json.loads(component_metrics_path.read_text(encoding="utf-8"))
                artifact_paths[f"momentum_model_path_{idx:02d}"] = resolve_output_path(str(metrics["momentum_model_path"]))
                artifact_paths[f"momentum_scaler_path_{idx:02d}"] = resolve_output_path(
                    str(metrics["momentum_scaler_path"])
                )
                artifact_paths[f"xgb_model_path_{idx:02d}"] = resolve_output_path(str(metrics["xgb_model_path"]))
                specialists_raw = str(metrics.get("xgb_specialists_path", "")).strip()
                if specialists_raw:
                    artifact_paths[f"xgb_specialists_path_{idx:02d}"] = resolve_output_path(specialists_raw)
            return artifact_paths

        momentum_model_path = resolve_output_path(str(model_info["momentum_model_path"]))
        momentum_scaler_path = resolve_output_path(str(model_info["momentum_scaler_path"]))
        xgb_model_path = resolve_output_path(str(model_info["xgb_model_path"]))
        xgb_specialists_raw = str(model_info.get("xgb_specialists_path", "")).strip()
        if xgb_specialists_raw:
            xgb_specialists_path = resolve_output_path(xgb_specialists_raw)
        else:
            stem = xgb_model_path.stem
            run_suffix = stem[len("xgb_") :] if stem.startswith("xgb_") else stem
            inferred = xgb_model_path.with_name(f"xgb_specialists_{run_suffix}.pkl")
            xgb_specialists_path = inferred if inferred.exists() else APP_ROOT / "data" / "model_cache" / "lstm_xgb_specialists.pkl"
        artifact_paths.update(
            {
                "momentum_model_path": momentum_model_path,
                "momentum_scaler_path": momentum_scaler_path,
                "xgb_model_path": xgb_model_path,
                "xgb_specialists_path": xgb_specialists_path,
            }
        )
        return artifact_paths

    def _ensure_loaded(self) -> None:
        model_info = self._model_registry.get()
        artifact_paths = self._resolve_artifact_paths(model_info)
        current_state = {
            key: (str(path), self._get_mtime(path))
            for key, path in artifact_paths.items()
        }
        kind = str(model_info.get("kind", "single_run")).strip().lower()

        if current_state["raw_csv_path"][1] is None:
            raise FileNotFoundError(f"Prediction data not found: {artifact_paths['raw_csv_path']}")
        if current_state["metrics_path"][1] is None:
            raise FileNotFoundError(f"Prediction metrics not found: {artifact_paths['metrics_path']}")

        if kind in {"average_ensemble", "weighted_ensemble"}:
            for key, path in artifact_paths.items():
                if key.startswith(("component_metrics_path_", "momentum_model_path_", "momentum_scaler_path_", "xgb_model_path_")):
                    if current_state[key][1] is None:
                        raise FileNotFoundError(f"Missing ensemble artifact: {path}")
        else:
            for key in ("momentum_model_path", "momentum_scaler_path", "xgb_model_path"):
                if current_state[key][1] is None:
                    raise FileNotFoundError(f"Missing model artifact: {artifact_paths[key]}")

        if self._predictor is None or current_state != self._artifact_state:
            if kind in {"average_ensemble", "weighted_ensemble"}:
                from scripts.lstm_xgb_matchup_predictor import AverageEnsembleMatchupPredictor

                report = json.loads(artifact_paths["metrics_path"].read_text(encoding="utf-8"))
                component_metrics_paths = [
                    artifact_paths[key]
                    for key in sorted(artifact_paths)
                    if key.startswith("component_metrics_path_")
                ]
                component_weights = [
                    float(component.get("weight", 1.0))
                    for component in report.get("model_components", [])
                ]
                self._predictor = AverageEnsembleMatchupPredictor(
                    raw_csv_path=artifact_paths["raw_csv_path"],
                    metadata_csv_path=artifact_paths["metadata_csv_path"],
                    ensemble_metrics_path=artifact_paths["metrics_path"],
                    component_metrics_paths=component_metrics_paths,
                    component_weights=component_weights,
                )
            else:
                from scripts.lstm_xgb_matchup_predictor import LSTMXGBMatchupPredictor

                self._predictor = LSTMXGBMatchupPredictor(
                    raw_csv_path=artifact_paths["raw_csv_path"],
                    metadata_csv_path=artifact_paths["metadata_csv_path"],
                    metrics_path=artifact_paths["metrics_path"],
                    momentum_model_path=artifact_paths["momentum_model_path"],
                    momentum_scaler_path=artifact_paths["momentum_scaler_path"],
                    xgb_model_path=artifact_paths["xgb_model_path"],
                    xgb_specialists_path=artifact_paths["xgb_specialists_path"],
                )
            self._artifact_state = current_state

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        fighter_1 = str(payload.get("fighter_1", "")).strip()
        fighter_2 = str(payload.get("fighter_2", "")).strip()
        if not fighter_1 or not fighter_2:
            raise ValueError("fighter_1 and fighter_2 are required.")
        if fighter_1.lower() == fighter_2.lower():
            raise ValueError("Choose two different fighters.")

        event_date = str(payload.get("event_date", "")).strip() or None
        weight_class = str(payload.get("weight_class", "")).strip() or None
        gender = str(payload.get("gender", "")).strip() or None
        scheduled_rounds_raw = str(payload.get("scheduled_rounds", "")).strip()
        scheduled_rounds = int(scheduled_rounds_raw) if scheduled_rounds_raw else None
        is_title_bout = parse_bool(payload.get("is_title_bout"), default=False)

        with self._lock:
            self._ensure_loaded()
            return self._predictor.predict_matchup(
                fighter_1_name=fighter_1,
                fighter_2_name=fighter_2,
                event_date=event_date,
                weight_class=weight_class,
                gender=gender,
                scheduled_rounds=scheduled_rounds,
                is_title_bout=is_title_bout,
            )


def build_dashboard_state(*, tail_lines: int = 500) -> dict[str, Any]:
    tracker_status = research_store.snapshot(limit_bets=120, limit_predictions=80)
    latest_candidate = trainer.latest_candidate_model()
    return {
        "generated_at_utc": utc_now_iso(),
        "data_status": collect_data_status(),
        "train_status": trainer.snapshot(tail_lines=tail_lines),
        "pipeline_status": pipeline.snapshot(tail_lines=tail_lines),
        "tracker_status": tracker_status,
        "model_status": predictor_service.model_status(latest_candidate=latest_candidate),
        "defaults": {
            "bankroll_units": 100.0,
            "scheduled_rounds": 3,
            "event_date": dt.date.today().isoformat(),
        },
    }


def build_research_snapshot() -> dict[str, Any]:
    return {
        "generated_at_utc": utc_now_iso(),
        "deployed_model": predictor_service.deployed_model(),
        "data_status": collect_data_status(),
        "train_status": trainer.snapshot(tail_lines=600),
        "pipeline_status": pipeline.snapshot(tail_lines=600),
        "tracker_status": research_store.snapshot(limit_bets=250, limit_predictions=250),
    }


app = Flask(__name__)
trainer = TrainingManager()
pipeline = PipelineManager()
research_store = ResearchStore(
    BET_TRACKER_DB,
    legacy_json_path=LEGACY_BET_TRACKER_JSON,
    legacy_csv_path=LEGACY_BETS_CSV,
)
model_registry = DeployedModelRegistry(
    state_path=MODEL_STATE_PATH,
    default_model_info=BEST_PREDICTOR_INFO,
    baseline_model_info=BEST_TRAINING_RUN_INFO,
)
predictor_service = PredictorService(
    raw_csv_path=RAW_FIGHTS_CSV,
    metadata_csv_path=METADATA_CSV,
    model_registry=model_registry,
)


@app.route("/", methods=["GET"])
def index() -> str:
    return render_template(
        "index.html",
        bootstrap=build_dashboard_state(tail_lines=400),
        fighter_names=load_fighter_names(),
    )


@app.route("/api/dashboard/state", methods=["GET"])
def dashboard_state() -> Any:
    try:
        tail_lines = int(request.args.get("tail", "500"))
    except ValueError:
        tail_lines = 500
    tail_lines = min(max(tail_lines, 100), 2500)
    return jsonify({"ok": True, "state": build_dashboard_state(tail_lines=tail_lines)})


@app.route("/api/model/status", methods=["GET"])
def model_status() -> Any:
    return jsonify({"ok": True, "status": predictor_service.model_status(trainer.latest_candidate_model())})


@app.route("/api/model/reload", methods=["POST"])
def model_reload() -> Any:
    predictor_service.force_reload()
    return jsonify({"ok": True, "status": predictor_service.model_status(trainer.latest_candidate_model())})


@app.route("/api/model/retrain", methods=["POST"])
def model_retrain() -> Any:
    try:
        if pipeline.is_running():
            raise RuntimeError("Cannot retrain while a data pipeline job is running.")
        params = build_refresh_training_params()
        status = trainer.start(params)
        return jsonify({"ok": True, "status": status})
    except Exception as exc:  # noqa: BLE001
        code = 409 if isinstance(exc, RuntimeError) else 400
        return jsonify({"ok": False, "error": str(exc)}), code


@app.route("/api/model/deploy", methods=["POST"])
def model_deploy() -> Any:
    payload = request.get_json(silent=True) or request.form.to_dict()
    model_key = str((payload or {}).get("model_key", "")).strip()
    try:
        deployed = model_registry.deploy(model_key, latest_candidate=trainer.latest_candidate_model())
        predictor_service.force_reload()
        return jsonify(
            {
                "ok": True,
                "deployed_model": deployed,
                "status": predictor_service.model_status(trainer.latest_candidate_model()),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/train/start", methods=["POST"])
def start_training() -> Any:
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict()
    try:
        if pipeline.is_running():
            raise RuntimeError("Cannot start training while a data pipeline job is running.")
        params = parse_train_params(payload or {})
        status = trainer.start(params)
        return jsonify({"ok": True, "status": status})
    except Exception as exc:  # noqa: BLE001
        code = 409 if isinstance(exc, RuntimeError) else 400
        return jsonify({"ok": False, "error": str(exc)}), code


@app.route("/api/train/stop", methods=["POST"])
def stop_training() -> Any:
    try:
        status = trainer.stop()
        return jsonify({"ok": True, "status": status})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 409


@app.route("/api/train/status", methods=["GET"])
def training_status() -> Any:
    try:
        tail_lines = int(request.args.get("tail", "500"))
    except ValueError:
        tail_lines = 500
    tail_lines = min(max(tail_lines, 50), 3000)
    return jsonify({"ok": True, "status": trainer.snapshot(tail_lines=tail_lines)})


@app.route("/api/pipeline/start", methods=["POST"])
def start_pipeline() -> Any:
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict()
    action = str((payload or {}).get("action", "")).strip().lower()
    try:
        if trainer.is_running():
            raise RuntimeError("Cannot run data pipeline while training is active.")
        status = pipeline.start(action)
        return jsonify({"ok": True, "status": status})
    except Exception as exc:  # noqa: BLE001
        code = 409 if isinstance(exc, RuntimeError) else 400
        return jsonify({"ok": False, "error": str(exc)}), code


@app.route("/api/pipeline/stop", methods=["POST"])
def stop_pipeline() -> Any:
    try:
        status = pipeline.stop()
        return jsonify({"ok": True, "status": status})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 409


@app.route("/api/pipeline/status", methods=["GET"])
def pipeline_status() -> Any:
    try:
        tail_lines = int(request.args.get("tail", "500"))
    except ValueError:
        tail_lines = 500
    tail_lines = min(max(tail_lines, 50), 3000)
    return jsonify({"ok": True, "status": pipeline.snapshot(tail_lines=tail_lines)})


@app.route("/api/analyze", methods=["POST"])
def analyze_matchup() -> Any:
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict()
    try:
        prediction = predictor_service.predict(payload or {})
        bankroll_units = parse_optional_float((payload or {}).get("bankroll_units")) or 100.0
        if bankroll_units <= 0:
            raise ValueError("bankroll_units must be > 0.")
        recommendation = build_recommendation(
            prediction,
            odds_fighter_1=parse_optional_american_odds((payload or {}).get("odds_fighter_1")),
            odds_fighter_2=parse_optional_american_odds((payload or {}).get("odds_fighter_2")),
            bankroll_units=float(bankroll_units),
        )
        saved_prediction = None
        if parse_bool((payload or {}).get("save_prediction"), default=False):
            saved_prediction = research_store.add_prediction(
                prediction=prediction,
                request_payload=payload or {},
                recommendation=recommendation,
            )
        return jsonify(
            {
                "ok": True,
                "prediction": prediction,
                "recommendation": recommendation,
                "saved_prediction": saved_prediction,
                "tracker_status": research_store.snapshot(limit_bets=120, limit_predictions=80),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/predict", methods=["POST"])
def predict_matchup() -> Any:
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict()
    try:
        prediction = predictor_service.predict(payload or {})
        return jsonify({"ok": True, "prediction": prediction})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/predictions/status", methods=["GET"])
def predictions_status() -> Any:
    try:
        limit = int(request.args.get("limit", "80"))
    except ValueError:
        limit = 80
    limit = min(max(limit, 10), 500)
    snapshot = research_store.snapshot(limit_bets=40, limit_predictions=limit)
    return jsonify({"ok": True, "status": {"predictions": snapshot["predictions"], "summary": snapshot["summary"]}})


@app.route("/api/bets/status", methods=["GET"])
def bets_status() -> Any:
    try:
        limit = int(request.args.get("limit", "120"))
    except ValueError:
        limit = 120
    limit = min(max(limit, 20), 500)
    return jsonify({"ok": True, "status": research_store.snapshot(limit_bets=limit, limit_predictions=80)})


@app.route("/api/bets/add", methods=["POST"])
def add_bet() -> Any:
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict()
    try:
        record = research_store.add_bet(payload or {})
        return jsonify({"ok": True, "bet": record, "status": research_store.snapshot(limit_bets=120, limit_predictions=80)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/bets/settle", methods=["POST"])
def settle_bet() -> Any:
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict()
    try:
        bet_id = int((payload or {}).get("bet_id"))
        result = str((payload or {}).get("result", "")).strip().lower()
        record = research_store.settle_bet(bet_id=bet_id, result=result)
        return jsonify({"ok": True, "bet": record, "status": research_store.snapshot(limit_bets=120, limit_predictions=80)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/research/snapshot", methods=["GET"])
def research_snapshot() -> Any:
    payload = build_research_snapshot()
    if parse_bool(request.args.get("download"), default=False):
        filename = f"ufc_research_snapshot_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        return Response(
            json.dumps(payload, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    return jsonify({"ok": True, "snapshot": payload})


@app.route("/healthz", methods=["GET"])
def healthz() -> Any:
    return jsonify({"ok": True, "service": "ufc-prediction-betting-desk"})


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
