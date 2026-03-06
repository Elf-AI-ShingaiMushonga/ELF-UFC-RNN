#!/usr/bin/env python3
"""End-to-end UFC fight outcome pipeline aligned to the Siamese paper draft.

Implements:
- Leakage-safe temporal train/val/test split
- Baselines: Logistic Regression, Gradient Boosting, MLP
- Siamese architecture with shared GRU fighter encoder
- Calibration, subgroup metrics, and ablations
- Reproducible artifact export (metrics/predictions/config)
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

MISSING_ML_DEPS = False
try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    MISSING_ML_DEPS = True
    np = Any  # type: ignore[assignment]
    pd = Any  # type: ignore[assignment]
    torch = Any  # type: ignore[assignment]
    class _DummyNN:
        Module = object

    nn = _DummyNN()  # type: ignore[assignment]
    ColumnTransformer = Any  # type: ignore[assignment]
    HistGradientBoostingClassifier = Any  # type: ignore[assignment]
    SimpleImputer = Any  # type: ignore[assignment]
    LogisticRegression = Any  # type: ignore[assignment]
    accuracy_score = Any  # type: ignore[assignment]
    brier_score_loss = Any  # type: ignore[assignment]
    log_loss = Any  # type: ignore[assignment]
    roc_auc_score = Any  # type: ignore[assignment]
    MLPClassifier = Any  # type: ignore[assignment]
    Pipeline = Any  # type: ignore[assignment]
    OneHotEncoder = Any  # type: ignore[assignment]
    StandardScaler = Any  # type: ignore[assignment]
    pack_padded_sequence = Any  # type: ignore[assignment]
    DataLoader = Any  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]


POSITIVE_LABEL = "fighter_1_win"
NEGATIVE_LABEL = "fighter_2_win"
SPLIT_COL = "__split__"
TARGET_COL = "__target__"

SUPPORTED_POWER_PROFILES = {"standard", "max_power"}

NON_FEATURE_COLUMNS = {
    "fight_id",
    "event_id",
    "event_name",
    "event_date",
    "event_city",
    "event_state",
    "event_country",
    "fighter_1_id",
    "fighter_1_name",
    "fighter_1_dob",
    "fighter_2_id",
    "fighter_2_name",
    "fighter_2_dob",
    "winner_fighter_id",
    "winner_name",
    "outcome_label",
    "scrape_timestamp_utc",
    TARGET_COL,
    SPLIT_COL,
}

POST_FIGHT_COLUMNS = {
    "round_ended",
    "time_ended",
    "fight_duration_seconds",
    "result_method",
    "result_method_category",
}

BASELINE_CATEGORICAL = {
    "weight_class",
    "gender",
    "fighter_1_stance",
    "fighter_2_stance",
    "event_country",
    "time_format",
}

STATIC_NUMERIC_COLS = [
    "bout_index",
    "is_main_event",
    "is_title_bout",
    "scheduled_rounds",
    "age_days_diff_f1_minus_f2",
    "height_cm_diff_f1_minus_f2",
    "reach_cm_diff_f1_minus_f2",
    "wins_diff_f1_minus_f2",
    "losses_diff_f1_minus_f2",
    "total_fights_diff_f1_minus_f2",
    "win_streak_diff_f1_minus_f2",
    "win_rate_diff_f1_minus_f2",
    "finish_rate_diff_f1_minus_f2",
    "days_since_last_fight_diff_f1_minus_f2",
    "avg_fight_duration_sec_diff_f1_minus_f2",
    "avg_rounds_fought_diff_f1_minus_f2",
    "sig_str_landed_per_min_diff_f1_minus_f2",
    "sig_str_absorbed_per_min_diff_f1_minus_f2",
    "sig_str_accuracy_diff_f1_minus_f2",
    "sig_str_defense_diff_f1_minus_f2",
    "td_landed_per_15_diff_f1_minus_f2",
    "td_absorbed_per_15_diff_f1_minus_f2",
    "td_accuracy_diff_f1_minus_f2",
    "td_defense_diff_f1_minus_f2",
    "sub_attempts_per_15_diff_f1_minus_f2",
    "knockdowns_per_15_diff_f1_minus_f2",
    "control_time_per_min_diff_f1_minus_f2",
    "event_year",
    "event_month",
    "event_dayofweek",
]

STATIC_CATEGORICAL_COLS = [
    "weight_class",
    "gender",
    "event_country",
]

PHYSICAL_NUMERIC_COLS = [
    "fighter_1_height_cm",
    "fighter_1_reach_cm",
    "fighter_2_height_cm",
    "fighter_2_reach_cm",
]

PHYSICAL_CATEGORICAL_COLS = [
    "fighter_1_stance",
    "fighter_2_stance",
]

METHOD_CATEGORIES = ["ko_tko", "submission", "decision", "dq", "other", "unknown"]


@dataclasses.dataclass
class TemporalSplit:
    mode: str
    train_start_date: str
    train_end_date: str
    val_start_date: str
    val_end_date: str
    test_start_date: str
    counts: dict[str, int]


@dataclasses.dataclass
class BaselineResult:
    name: str
    val_probs: Any
    test_probs: Any
    metrics_test: dict[str, Any]
    metrics_val: dict[str, Any]


@dataclasses.dataclass
class SiameseConfig:
    max_seq_len: int = 8
    hidden_dim: int = 96
    static_hidden_dim: int = 48
    num_layers: int = 1
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5


@dataclasses.dataclass
class SiamesePreparedData:
    seq1_train: Any
    len1_train: Any
    seq2_train: Any
    len2_train: Any
    static_train: Any
    physical_train: Any
    y_train: Any
    seq1_val: Any
    len1_val: Any
    seq2_val: Any
    len2_val: Any
    static_val: Any
    physical_val: Any
    y_val: Any
    seq1_test: Any
    len1_test: Any
    seq2_test: Any
    len2_test: Any
    static_test: Any
    physical_test: Any
    y_test: Any
    test_meta: Any
    sequence_feature_names: list[str]
    static_feature_names: list[str]
    physical_feature_names: list[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _detect_system_ram_gb() -> float:
    try:
        page_size = float(os.sysconf("SC_PAGE_SIZE"))
        pages = float(os.sysconf("SC_PHYS_PAGES"))
        if page_size > 0 and pages > 0:
            return (page_size * pages) / (1024.0 ** 3)
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return 8.0


def _infer_capacity_tier(ram_gb: float) -> str:
    if ram_gb >= 32.0:
        return "ultra"
    if ram_gb >= 16.0:
        return "high"
    if ram_gb >= 8.0:
        return "medium"
    return "low"


def get_training_capacity_profile(power_profile: str = "standard", seed: int = 42) -> dict[str, Any]:
    profile = str(power_profile).strip().lower()
    if profile not in SUPPORTED_POWER_PROFILES:
        raise ValueError(
            f"Unsupported power profile: {power_profile}. "
            f"Choose one of: {', '.join(sorted(SUPPORTED_POWER_PROFILES))}."
        )

    cpu_count = max(1, int(os.cpu_count() or 1))
    ram_gb = float(_detect_system_ram_gb())
    capacity_tier = _infer_capacity_tier(ram_gb)
    tree_n_jobs = max(1, min(cpu_count - 1, 8))

    config: dict[str, Any] = {
        "profile": profile,
        "capacity_tier": capacity_tier,
        "cpu_count": cpu_count,
        "ram_gb": ram_gb,
        "seed": int(seed),
        "logistic_max_iter": 600,
        "logistic_c": 1.0,
        "gb_learning_rate": 0.05,
        "gb_max_depth": 6,
        "gb_max_iter": 300,
        "gb_max_leaf_nodes": 63,
        "gb_min_samples_leaf": 20,
        "gb_l2_regularization": 0.1,
        "mlp_hidden_layer_sizes": (192, 96),
        "mlp_alpha": 1e-4,
        "mlp_learning_rate_init": 1e-3,
        "mlp_batch_size": "auto",
        "mlp_max_iter": 350,
        "mlp_n_iter_no_change": 10,
        "mlp_validation_fraction": 0.1,
        "rf_n_estimators": 300,
        "rf_min_samples_leaf": 2,
        "et_n_estimators": 400,
        "et_min_samples_leaf": 2,
        "tree_n_jobs": 1,
        "siamese_hidden_dim": 96,
        "siamese_static_hidden_dim": 48,
        "siamese_num_layers": 1,
        "siamese_dropout": 0.1,
        "siamese_batch_size": 64,
        "siamese_epochs": 25,
        "siamese_lr": 1e-3,
        "siamese_weight_decay": 1e-4,
        "siamese_patience": 5,
        "momentum_embedding_dim": 32,
        "momentum_hidden_dim": 96,
        "momentum_num_layers": 1,
        "momentum_dropout": 0.1,
        "momentum_batch_size": 64,
        "momentum_epochs": 25,
        "momentum_lr": 1e-3,
        "momentum_weight_decay": 1e-4,
        "momentum_patience": 5,
        "xgb_n_estimators": 700,
        "xgb_learning_rate": 0.03,
        "xgb_max_depth": 4,
        "xgb_min_child_weight": 2.0,
        "xgb_subsample": 0.85,
        "xgb_colsample_bytree": 0.85,
        "xgb_reg_lambda": 1.0,
        "xgb_early_stopping_rounds": 50,
    }

    if profile == "standard":
        return config

    config["tree_n_jobs"] = tree_n_jobs
    config["logistic_max_iter"] = 2000
    config["logistic_c"] = 1.5

    if capacity_tier == "ultra":
        config.update(
            {
                "gb_learning_rate": 0.02,
                "gb_max_depth": 14,
                "gb_max_iter": 1800,
                "gb_max_leaf_nodes": 255,
                "gb_min_samples_leaf": 10,
                "gb_l2_regularization": 0.2,
                "mlp_hidden_layer_sizes": (1536, 768, 384),
                "mlp_alpha": 5e-5,
                "mlp_learning_rate_init": 6e-4,
                "mlp_batch_size": 320,
                "mlp_max_iter": 1000,
                "mlp_n_iter_no_change": 25,
                "mlp_validation_fraction": 0.12,
                "rf_n_estimators": 2200,
                "et_n_estimators": 2800,
                "siamese_hidden_dim": 224,
                "siamese_static_hidden_dim": 112,
                "siamese_num_layers": 3,
                "siamese_dropout": 0.2,
                "siamese_batch_size": 112,
                "siamese_epochs": 65,
                "siamese_lr": 8e-4,
                "siamese_weight_decay": 8e-5,
                "siamese_patience": 10,
                "momentum_embedding_dim": 96,
                "momentum_hidden_dim": 224,
                "momentum_num_layers": 3,
                "momentum_dropout": 0.2,
                "momentum_batch_size": 112,
                "momentum_epochs": 65,
                "momentum_lr": 8e-4,
                "momentum_weight_decay": 8e-5,
                "momentum_patience": 10,
                "xgb_n_estimators": 2200,
                "xgb_learning_rate": 0.015,
                "xgb_max_depth": 8,
                "xgb_min_child_weight": 1.0,
                "xgb_subsample": 0.9,
                "xgb_colsample_bytree": 0.9,
                "xgb_reg_lambda": 1.0,
                "xgb_early_stopping_rounds": 140,
            }
        )
    elif capacity_tier == "high":
        config.update(
            {
                "gb_learning_rate": 0.025,
                "gb_max_depth": 12,
                "gb_max_iter": 1400,
                "gb_max_leaf_nodes": 255,
                "gb_min_samples_leaf": 12,
                "gb_l2_regularization": 0.2,
                "mlp_hidden_layer_sizes": (1024, 512, 256),
                "mlp_alpha": 8e-5,
                "mlp_learning_rate_init": 7e-4,
                "mlp_batch_size": 256,
                "mlp_max_iter": 800,
                "mlp_n_iter_no_change": 20,
                "mlp_validation_fraction": 0.12,
                "rf_n_estimators": 1700,
                "et_n_estimators": 2200,
                "siamese_hidden_dim": 192,
                "siamese_static_hidden_dim": 96,
                "siamese_num_layers": 2,
                "siamese_dropout": 0.15,
                "siamese_batch_size": 96,
                "siamese_epochs": 55,
                "siamese_lr": 8e-4,
                "siamese_weight_decay": 1e-4,
                "siamese_patience": 9,
                "momentum_embedding_dim": 80,
                "momentum_hidden_dim": 192,
                "momentum_num_layers": 2,
                "momentum_dropout": 0.15,
                "momentum_batch_size": 96,
                "momentum_epochs": 55,
                "momentum_lr": 8e-4,
                "momentum_weight_decay": 1e-4,
                "momentum_patience": 9,
                "xgb_n_estimators": 1600,
                "xgb_learning_rate": 0.018,
                "xgb_max_depth": 7,
                "xgb_min_child_weight": 1.0,
                "xgb_subsample": 0.9,
                "xgb_colsample_bytree": 0.9,
                "xgb_reg_lambda": 1.0,
                "xgb_early_stopping_rounds": 120,
            }
        )
    elif capacity_tier == "medium":
        config.update(
            {
                "gb_learning_rate": 0.03,
                "gb_max_depth": 10,
                "gb_max_iter": 1000,
                "gb_max_leaf_nodes": 127,
                "gb_min_samples_leaf": 16,
                "gb_l2_regularization": 0.2,
                "mlp_hidden_layer_sizes": (768, 384, 192),
                "mlp_alpha": 1e-4,
                "mlp_learning_rate_init": 8e-4,
                "mlp_batch_size": 192,
                "mlp_max_iter": 650,
                "mlp_n_iter_no_change": 18,
                "mlp_validation_fraction": 0.12,
                "rf_n_estimators": 1200,
                "et_n_estimators": 1500,
                "siamese_hidden_dim": 160,
                "siamese_static_hidden_dim": 80,
                "siamese_num_layers": 2,
                "siamese_dropout": 0.15,
                "siamese_batch_size": 80,
                "siamese_epochs": 45,
                "siamese_lr": 9e-4,
                "siamese_weight_decay": 1e-4,
                "siamese_patience": 8,
                "momentum_embedding_dim": 64,
                "momentum_hidden_dim": 160,
                "momentum_num_layers": 2,
                "momentum_dropout": 0.15,
                "momentum_batch_size": 80,
                "momentum_epochs": 45,
                "momentum_lr": 9e-4,
                "momentum_weight_decay": 1e-4,
                "momentum_patience": 8,
                "xgb_n_estimators": 1200,
                "xgb_learning_rate": 0.02,
                "xgb_max_depth": 6,
                "xgb_min_child_weight": 1.0,
                "xgb_subsample": 0.9,
                "xgb_colsample_bytree": 0.9,
                "xgb_reg_lambda": 1.0,
                "xgb_early_stopping_rounds": 100,
            }
        )
    else:
        config.update(
            {
                "gb_learning_rate": 0.035,
                "gb_max_depth": 8,
                "gb_max_iter": 700,
                "gb_max_leaf_nodes": 127,
                "gb_min_samples_leaf": 20,
                "gb_l2_regularization": 0.2,
                "mlp_hidden_layer_sizes": (512, 256, 128),
                "mlp_alpha": 1e-4,
                "mlp_learning_rate_init": 9e-4,
                "mlp_batch_size": 128,
                "mlp_max_iter": 500,
                "mlp_n_iter_no_change": 16,
                "mlp_validation_fraction": 0.12,
                "rf_n_estimators": 800,
                "et_n_estimators": 1000,
                "siamese_hidden_dim": 128,
                "siamese_static_hidden_dim": 64,
                "siamese_num_layers": 2,
                "siamese_dropout": 0.15,
                "siamese_batch_size": 64,
                "siamese_epochs": 35,
                "siamese_lr": 9e-4,
                "siamese_weight_decay": 1e-4,
                "siamese_patience": 7,
                "momentum_embedding_dim": 48,
                "momentum_hidden_dim": 128,
                "momentum_num_layers": 2,
                "momentum_dropout": 0.15,
                "momentum_batch_size": 64,
                "momentum_epochs": 35,
                "momentum_lr": 9e-4,
                "momentum_weight_decay": 1e-4,
                "momentum_patience": 7,
                "xgb_n_estimators": 900,
                "xgb_learning_rate": 0.025,
                "xgb_max_depth": 5,
                "xgb_min_child_weight": 1.5,
                "xgb_subsample": 0.9,
                "xgb_colsample_bytree": 0.9,
                "xgb_reg_lambda": 1.0,
                "xgb_early_stopping_rounds": 80,
            }
        )

    return config


def infer_method_category(value: Any) -> str:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return "unknown"
    if "ko" in text or "tko" in text:
        return "ko_tko"
    if "sub" in text:
        return "submission"
    if "dec" in text:
        return "decision"
    if "dq" in text:
        return "dq"
    return "other"


def safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def resolve_prefight_win_rate(row: Any, fighter_prefix: str) -> float:
    """Return pre-fight win rate for a fighter from row fields, with fallback."""
    rate = safe_float(row.get(f"{fighter_prefix}win_rate_pre"))
    if not np.isnan(rate):
        return float(np.clip(rate, 0.0, 1.0))

    wins = safe_float(row.get(f"{fighter_prefix}wins_pre"))
    total = safe_float(row.get(f"{fighter_prefix}total_fights_pre"))
    if np.isnan(wins) or np.isnan(total) or total <= 0:
        return float("nan")
    return float(np.clip(wins / total, 0.0, 1.0))


def resolve_days_since_last_fight(
    raw_value: Any,
    current_event_date: Any,
    previous_event_date: Optional[Any],
) -> float:
    """Resolve inactivity days with CSV value first, then chronological fallback."""
    days = safe_float(raw_value)
    if not np.isnan(days):
        return max(days, 0.0)

    if previous_event_date is None:
        return float("nan")

    current = pd.Timestamp(current_event_date).normalize()
    delta_days = (current - previous_event_date).days
    if delta_days < 0:
        return float("nan")
    return float(delta_days)


def create_output_dir(base_dir: Path, run_name: str) -> Path:
    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def ensure_minimum_rows(df: Any, min_rows: int) -> None:
    if len(df) < min_rows:
        raise ValueError(
            f"Not enough rows for experiment: {len(df)} found, at least {min_rows} required."
        )


def load_and_prepare_dataframe(csv_path: Path) -> Any:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {csv_path}")
    required = {"event_date", "outcome_label", "fight_id", "fighter_1_id", "fighter_2_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"]).copy()
    df = df[df["outcome_label"].isin([POSITIVE_LABEL, NEGATIVE_LABEL])].copy()
    if df.empty:
        raise ValueError("No binary labeled fights (fighter_1_win / fighter_2_win) in dataset.")

    df[TARGET_COL] = (df["outcome_label"] == POSITIVE_LABEL).astype(int)
    df["event_year"] = df["event_date"].dt.year.astype(float)
    df["event_month"] = df["event_date"].dt.month.astype(float)
    df["event_dayofweek"] = df["event_date"].dt.dayofweek.astype(float)

    for col in df.columns:
        if col in NON_FEATURE_COLUMNS or col == "event_date":
            continue
        if col in BASELINE_CATEGORICAL or col in STATIC_CATEGORICAL_COLS:
            continue
        if col.startswith("fighter_1_stance") or col.startswith("fighter_2_stance"):
            continue
        if col in {"weight_class", "gender", "event_country", "time_format"}:
            continue
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                df[col] = converted

    if "bout_index" in df.columns:
        df["bout_index"] = pd.to_numeric(df["bout_index"], errors="coerce")
    else:
        df["bout_index"] = 999.0

    df = df.sort_values(["event_date", "bout_index", "fight_id"]).reset_index(drop=True)
    return df


def rebalance_binary_orientation_if_needed(df: Any, seed: int) -> tuple[Any, dict[str, Any]]:
    """Re-orient fighter sides when binary labels collapse to one class.

    Some scraped datasets store winner-first rows (fighter_1 is almost always winner),
    which makes the binary target degenerate. This keeps one row per fight and
    deterministically swaps fighter_1/fighter_2 sides for about half of fights.
    """
    counts_before = df["outcome_label"].value_counts(dropna=False).to_dict()
    binary_mask = df["outcome_label"].isin([POSITIVE_LABEL, NEGATIVE_LABEL])
    binary_df = df[binary_mask]
    unique_binary = set(binary_df["outcome_label"].unique().tolist())
    needs_rebalance = unique_binary == {POSITIVE_LABEL} or unique_binary == {NEGATIVE_LABEL}

    info: dict[str, Any] = {
        "applied": False,
        "seed": int(seed),
        "counts_before": {str(k): int(v) for k, v in counts_before.items()},
        "counts_after": {str(k): int(v) for k, v in counts_before.items()},
        "swapped_rows": 0,
    }
    if not needs_rebalance:
        return df, info

    out = df.copy(deep=True)

    def _flip_side(fight_id: Any) -> bool:
        key = f"{fight_id}|{seed}".encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()
        return (int(digest[:8], 16) % 2) == 1

    flip_mask = out.index.to_series().map(lambda _: False)
    flip_mask.loc[binary_mask] = out.loc[binary_mask, "fight_id"].map(_flip_side)

    paired_cols: list[tuple[str, str]] = []
    for col in out.columns:
        if not col.startswith("fighter_1_"):
            continue
        c2 = "fighter_2_" + col[len("fighter_1_") :]
        if c2 in out.columns:
            paired_cols.append((col, c2))
    for c1, c2 in paired_cols:
        tmp = out.loc[flip_mask, c1].copy()
        out.loc[flip_mask, c1] = out.loc[flip_mask, c2].values
        out.loc[flip_mask, c2] = tmp.values

    diff_cols = [c for c in out.columns if c.endswith("_diff_f1_minus_f2")]
    if diff_cols:
        out.loc[flip_mask, diff_cols] = -out.loc[flip_mask, diff_cols].apply(
            pd.to_numeric, errors="coerce"
        )

    pos_mask = flip_mask & (out["outcome_label"] == POSITIVE_LABEL)
    neg_mask = flip_mask & (out["outcome_label"] == NEGATIVE_LABEL)
    out.loc[pos_mask, "outcome_label"] = NEGATIVE_LABEL
    out.loc[neg_mask, "outcome_label"] = POSITIVE_LABEL
    out[TARGET_COL] = (out["outcome_label"] == POSITIVE_LABEL).astype(int)

    counts_after = out["outcome_label"].value_counts(dropna=False).to_dict()
    info["applied"] = True
    info["swapped_rows"] = int(flip_mask.sum())
    info["counts_after"] = {str(k): int(v) for k, v in counts_after.items()}
    return out, info


def _parse_date_arg(label: str, value: str) -> Any:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{label} must be a valid date in YYYY-MM-DD format. Got: {value!r}")
    return pd.Timestamp(parsed).normalize()


def _latest_full_year(df: Any, include_partial_latest_year: bool) -> int:
    max_date = pd.Timestamp(df["event_date"].max()).normalize()
    max_year = int(max_date.year)
    if include_partial_latest_year:
        return max_year
    is_full_year = max_date >= pd.Timestamp(year=max_year, month=12, day=31)
    return max_year if is_full_year else max_year - 1


def _infer_cutoffs_from_year_windows(
    df: Any,
    val_years: int,
    test_years: int,
    expanding_test_end_year: Optional[int],
    include_partial_latest_year: bool,
) -> tuple[Any, Any]:
    if val_years < 1:
        raise ValueError("--val-years must be >= 1.")
    if test_years < 1:
        raise ValueError("--test-years must be >= 1.")

    available_years = sorted(int(y) for y in df["event_date"].dt.year.dropna().unique())
    if not available_years:
        raise ValueError("No event years available for temporal split.")

    latest_full_year = _latest_full_year(df, include_partial_latest_year)
    if expanding_test_end_year is None:
        target_test_end_year = latest_full_year
    else:
        target_test_end_year = int(expanding_test_end_year)
        if target_test_end_year > latest_full_year and not include_partial_latest_year:
            raise ValueError(
                f"--expanding-test-end-year={target_test_end_year} exceeds latest full year "
                f"{latest_full_year}. Pass --include-partial-latest-year to allow it."
            )

    candidate_years = [y for y in available_years if y <= target_test_end_year]
    min_required = val_years + test_years + 1
    if len(candidate_years) < min_required:
        raise ValueError(
            "Not enough calendar years for split. Need at least "
            f"{min_required} distinct years, found {len(candidate_years)} "
            f"(up to {target_test_end_year})."
        )

    test_start_idx = len(candidate_years) - test_years
    val_start_idx = test_start_idx - val_years
    train_end_idx = val_start_idx - 1
    if train_end_idx < 0:
        raise ValueError("Year-window split left no training years. Increase data coverage.")

    train_end_year = candidate_years[train_end_idx]
    val_end_year = candidate_years[test_start_idx - 1]

    train_end = pd.Timestamp(year=train_end_year, month=12, day=31)
    val_end = pd.Timestamp(year=val_end_year, month=12, day=31)
    return train_end, val_end


def _apply_chronological_cutoffs(
    df: Any,
    train_end: Any,
    val_end: Any,
    mode: str,
) -> TemporalSplit:
    train_end = pd.Timestamp(train_end).normalize()
    val_end = pd.Timestamp(val_end).normalize()
    if train_end >= val_end:
        raise ValueError(
            f"Invalid temporal cutoffs: train_end ({train_end.date()}) must be earlier than "
            f"val_end ({val_end.date()})."
        )

    event_dates = df["event_date"].dt.normalize()
    split = np.where(
        event_dates <= train_end,
        "train",
        np.where(event_dates <= val_end, "val", "test"),
    )
    df[SPLIT_COL] = split
    counts = df[SPLIT_COL].value_counts().to_dict()
    if counts.get("train", 0) == 0 or counts.get("val", 0) == 0 or counts.get("test", 0) == 0:
        raise ValueError(
            "Temporal split produced an empty split. Adjust cutoffs or year-window arguments."
        )

    train_dates = df.loc[df[SPLIT_COL] == "train", "event_date"]
    val_dates = df.loc[df[SPLIT_COL] == "val", "event_date"]
    test_dates = df.loc[df[SPLIT_COL] == "test", "event_date"]
    return TemporalSplit(
        mode=mode,
        train_start_date=str(pd.Timestamp(train_dates.min()).date()),
        train_end_date=str(train_end.date()),
        val_start_date=str(pd.Timestamp(val_dates.min()).date()),
        val_end_date=str(val_end.date()),
        test_start_date=str(pd.Timestamp(test_dates.min()).date()),
        counts={k: int(v) for k, v in counts.items()},
    )


def temporal_split_dataframe(
    df: Any,
    split_mode: str,
    train_end_date: Optional[str],
    val_end_date: Optional[str],
    val_years: int,
    test_years: int,
    expanding_test_end_year: Optional[int],
    include_partial_latest_year: bool,
) -> TemporalSplit:
    mode = split_mode.strip().lower()
    if mode not in {"strict", "expanding_window"}:
        raise ValueError(f"--split-mode must be one of: strict, expanding_window. Got: {split_mode!r}")

    if train_end_date and not val_end_date:
        raise ValueError("--val-end-date is required when --train-end-date is provided.")
    if val_end_date and not train_end_date:
        raise ValueError("--train-end-date is required when --val-end-date is provided.")

    if train_end_date and val_end_date:
        train_end = _parse_date_arg("--train-end-date", train_end_date)
        val_end = _parse_date_arg("--val-end-date", val_end_date)
    else:
        train_end, val_end = _infer_cutoffs_from_year_windows(
            df=df,
            val_years=val_years,
            test_years=test_years,
            expanding_test_end_year=expanding_test_end_year,
            include_partial_latest_year=include_partial_latest_year,
        )

    return _apply_chronological_cutoffs(df, train_end, val_end, mode=mode)


def select_baseline_features(df: Any) -> tuple[list[str], list[str], list[str]]:
    candidate_cols = []
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if col in POST_FIGHT_COLUMNS:
            continue
        if col.startswith("result_method"):
            continue
        if col.startswith("winner_"):
            continue
        if col == "event_date":
            continue
        candidate_cols.append(col)

    categorical = [c for c in candidate_cols if c in BASELINE_CATEGORICAL]
    numeric = [c for c in candidate_cols if c not in categorical]
    return candidate_cols, numeric, categorical


def make_one_hot_encoder() -> Any:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_tabular_pipeline(
    model_name: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    power_profile: str = "standard",
    seed: int = 42,
) -> Any:
    cfg = get_training_capacity_profile(power_profile=power_profile, seed=seed)
    numeric_pipe = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[("impute", SimpleImputer(strategy="most_frequent")), ("onehot", make_one_hot_encoder())]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    if model_name == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=int(cfg["logistic_max_iter"]),
            class_weight="balanced",
            C=float(cfg["logistic_c"]),
            solver="lbfgs",
        )
    elif model_name == "gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            learning_rate=float(cfg["gb_learning_rate"]),
            max_depth=int(cfg["gb_max_depth"]),
            max_iter=int(cfg["gb_max_iter"]),
            max_leaf_nodes=int(cfg["gb_max_leaf_nodes"]),
            min_samples_leaf=int(cfg["gb_min_samples_leaf"]),
            l2_regularization=float(cfg["gb_l2_regularization"]),
            random_state=int(seed),
        )
    elif model_name == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=tuple(int(v) for v in cfg["mlp_hidden_layer_sizes"]),
            activation="relu",
            solver="adam",
            alpha=float(cfg["mlp_alpha"]),
            learning_rate_init=float(cfg["mlp_learning_rate_init"]),
            batch_size=cfg["mlp_batch_size"],
            max_iter=int(cfg["mlp_max_iter"]),
            n_iter_no_change=int(cfg["mlp_n_iter_no_change"]),
            validation_fraction=float(cfg["mlp_validation_fraction"]),
            early_stopping=True,
            random_state=int(seed),
        )
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")

    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def expected_calibration_error(y_true: Any, probs: Any, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left = bins[i]
        right = bins[i + 1]
        mask = (p >= left) & (p < right) if i < n_bins - 1 else (p >= left) & (p <= right)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(y[mask]))
        bin_conf = float(np.mean(p[mask]))
        weight = float(np.mean(mask))
        ece += abs(bin_acc - bin_conf) * weight
    return float(ece)


def compute_metrics(y_true: Any, probs: Any) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probs, dtype=float)
    if y.size == 0:
        return {
            "roc_auc": float("nan"),
            "log_loss": float("nan"),
            "accuracy": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
        }
    p = np.clip(p, 1e-6, 1 - 1e-6)
    preds = (p >= 0.5).astype(int)

    auc = float("nan")
    if len(np.unique(y)) > 1:
        auc = float(roc_auc_score(y, p))

    return {
        "roc_auc": auc,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, preds)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p, n_bins=10),
    }


def calibration_table(y_true: Any, probs: Any, n_bins: int = 10) -> Any:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        left = bins[i]
        right = bins[i + 1]
        mask = (p >= left) & (p < right) if i < n_bins - 1 else (p >= left) & (p <= right)
        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "count": 0,
                    "pred_mean": float("nan"),
                    "true_rate": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "count": count,
                "pred_mean": float(np.mean(p[mask])),
                "true_rate": float(np.mean(y[mask])),
            }
        )
    return pd.DataFrame(rows)


def _safe_logit(probs: Any) -> Any:
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def maybe_apply_platt_calibration(
    y_val: Any,
    val_probs: Any,
    test_probs: Any,
    min_improvement: float = 0.005,
) -> tuple[Any, Any, dict[str, Any]]:
    """Calibrate probabilities on validation and keep calibration only if it helps log loss."""
    y = np.asarray(y_val, dtype=int)
    v = np.asarray(val_probs, dtype=float)
    t = np.asarray(test_probs, dtype=float)
    raw_log_loss = compute_metrics(y, v)["log_loss"]

    if y.size == 0 or len(np.unique(y)) < 2:
        return v, t, {"applied": False, "reason": "val_single_class", "val_log_loss_raw": raw_log_loss}

    try:
        x_val = _safe_logit(v).reshape(-1, 1)
        calibrator = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=1000,
        )
        calibrator.fit(x_val, y)
        v_cal = calibrator.predict_proba(x_val)[:, 1]
        cal_log_loss = compute_metrics(y, v_cal)["log_loss"]
        improvement = raw_log_loss - cal_log_loss

        if np.isfinite(improvement) and improvement > min_improvement:
            x_test = _safe_logit(t).reshape(-1, 1)
            t_cal = calibrator.predict_proba(x_test)[:, 1]
            return v_cal, t_cal, {
                "applied": True,
                "method": "platt",
                "val_log_loss_raw": raw_log_loss,
                "val_log_loss_calibrated": cal_log_loss,
                "val_log_loss_improvement": improvement,
            }
    except Exception as exc:  # noqa: BLE001
        return v, t, {
            "applied": False,
            "reason": f"platt_failed:{type(exc).__name__}",
            "val_log_loss_raw": raw_log_loss,
        }

    return v, t, {
        "applied": False,
        "reason": "no_val_log_loss_gain",
        "val_log_loss_raw": raw_log_loss,
    }


def subgroup_metrics(df_test: Any, probs: Any, model_name: str, min_count: int = 25) -> Any:
    rows = []
    subgroup_cols = ["weight_class", "gender", "is_title_bout"]
    for subgroup in subgroup_cols:
        if subgroup not in df_test.columns:
            continue
        for value, group in df_test.groupby(subgroup):
            if len(group) < min_count:
                continue
            idx = group.index.to_numpy()
            metrics = compute_metrics(group[TARGET_COL].to_numpy(), probs[idx])
            rows.append(
                {
                    "model": model_name,
                    "subgroup": subgroup,
                    "value": str(value),
                    "count": int(len(group)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def train_baseline_models(
    df: Any,
    power_profile: str = "standard",
    seed: int = 42,
) -> tuple[list[BaselineResult], dict[str, Any]]:
    feature_cols, numeric_cols, categorical_cols = select_baseline_features(df)

    train_df = df[df[SPLIT_COL] == "train"].copy()
    val_df = df[df[SPLIT_COL] == "val"].copy()
    test_df = df[df[SPLIT_COL] == "test"].copy()

    # Drop features that are fully missing in train (common after adding new columns).
    feature_cols = [col for col in feature_cols if train_df[col].notna().any()]
    numeric_cols = [col for col in numeric_cols if col in feature_cols]
    categorical_cols = [col for col in categorical_cols if col in feature_cols]

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[TARGET_COL].to_numpy(dtype=int)
    y_val = val_df[TARGET_COL].to_numpy(dtype=int)
    y_test = test_df[TARGET_COL].to_numpy(dtype=int)

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "Train split contains only one class. Increase data coverage or adjust temporal split fractions."
        )
    if len(np.unique(y_val)) < 2:
        logging.warning("Validation split contains one class; ROC-AUC will be NaN for some models.")
    if len(np.unique(y_test)) < 2:
        logging.warning("Test split contains one class; ROC-AUC will be NaN for some models.")

    results: list[BaselineResult] = []
    models = ["logistic_regression", "gradient_boosting", "mlp"]
    for model_name in models:
        logging.info("Training baseline: %s", model_name)
        pipeline = build_tabular_pipeline(
            model_name=model_name,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            power_profile=power_profile,
            seed=seed,
        )
        pipeline.fit(X_train, y_train)
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        test_probs = pipeline.predict_proba(X_test)[:, 1]
        result = BaselineResult(
            name=model_name,
            val_probs=val_probs,
            test_probs=test_probs,
            metrics_val=compute_metrics(y_val, val_probs),
            metrics_test=compute_metrics(y_test, test_probs),
        )
        results.append(result)

    return results, {"feature_cols": feature_cols, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols}


def _safe_series_float(df: Any, col: str) -> Any:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def naive_prefight_winrate_probs(df: Any) -> Any:
    f1 = _safe_series_float(df, "fighter_1_win_rate_pre")
    f2 = _safe_series_float(df, "fighter_2_win_rate_pre")
    if f1.notna().sum() == 0 and f2.notna().sum() == 0:
        w1 = _safe_series_float(df, "fighter_1_wins_pre")
        t1 = _safe_series_float(df, "fighter_1_total_fights_pre")
        w2 = _safe_series_float(df, "fighter_2_wins_pre")
        t2 = _safe_series_float(df, "fighter_2_total_fights_pre")
        f1 = w1 / t1.replace(0, np.nan)
        f2 = w2 / t2.replace(0, np.nan)

    f1 = f1.fillna(0.5).clip(0.0, 1.0)
    f2 = f2.fillna(0.5).clip(0.0, 1.0)
    p = (f1 + 1e-6) / (f1 + f2 + 2e-6)
    return p.to_numpy(dtype=float)


class StaticFeatureEncoder:
    def __init__(self, numeric_cols: list[str], categorical_cols: list[str]) -> None:
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.numeric_fill: dict[str, float] = {}
        self.categories: dict[str, list[str]] = {}
        self.feature_names: list[str] = []

    def fit(self, df_train: Any) -> None:
        self.numeric_fill = {}
        for col in self.numeric_cols:
            if col not in df_train.columns:
                continue
            series = pd.to_numeric(df_train[col], errors="coerce")
            self.numeric_fill[col] = float(series.median()) if series.notna().any() else 0.0

        self.categories = {}
        for col in self.categorical_cols:
            if col not in df_train.columns:
                continue
            values = (
                df_train[col].astype(str).replace("nan", "__UNK__").replace("", "__UNK__").fillna("__UNK__")
            )
            cats = sorted(set(values.tolist()))
            if "__UNK__" not in cats:
                cats.append("__UNK__")
            self.categories[col] = cats

        feature_names = []
        for col in self.numeric_cols:
            if col in self.numeric_fill:
                feature_names.append(col)
        for col in self.categorical_cols:
            cats = self.categories.get(col, [])
            for cat in cats:
                feature_names.append(f"{col}__{cat}")
        self.feature_names = feature_names

    def transform(self, df: Any) -> Any:
        numeric_matrix = []
        for col in self.numeric_cols:
            if col not in self.numeric_fill:
                continue
            values = pd.to_numeric(df[col], errors="coerce").fillna(self.numeric_fill[col]).to_numpy(dtype=float)
            numeric_matrix.append(values.reshape(-1, 1))
        if numeric_matrix:
            x_num = np.hstack(numeric_matrix)
        else:
            x_num = np.zeros((len(df), 0), dtype=float)

        one_hot_parts = []
        for col in self.categorical_cols:
            cats = self.categories.get(col)
            if not cats:
                continue
            mapping = {cat: idx for idx, cat in enumerate(cats)}
            values = (
                df[col].astype(str).replace("nan", "__UNK__").replace("", "__UNK__").fillna("__UNK__").to_numpy()
            )
            arr = np.zeros((len(df), len(cats)), dtype=float)
            for i, raw in enumerate(values):
                idx = mapping.get(raw, mapping.get("__UNK__", 0))
                arr[i, idx] = 1.0
            one_hot_parts.append(arr)

        if one_hot_parts:
            x_cat = np.hstack(one_hot_parts)
        else:
            x_cat = np.zeros((len(df), 0), dtype=float)

        if x_num.size == 0 and x_cat.size == 0:
            return np.zeros((len(df), 0), dtype=float)
        if x_num.size == 0:
            return x_cat
        if x_cat.size == 0:
            return x_num
        return np.hstack([x_num, x_cat])


class SequenceNormalizer:
    def __init__(self) -> None:
        self.mean: Optional[Any] = None
        self.std: Optional[Any] = None

    def fit(self, vectors: list[Any]) -> None:
        if not vectors:
            raise ValueError("Cannot fit sequence normalizer without vectors.")
        matrix = np.vstack(vectors)
        # Avoid nanmean/nanstd RuntimeWarnings on all-missing columns by computing
        # statistics only from finite values.
        finite = np.isfinite(matrix)
        counts = finite.sum(axis=0).astype(float)

        safe_matrix = np.where(finite, matrix, 0.0)
        sums = safe_matrix.sum(axis=0)
        mean = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)

        centered = np.where(finite, matrix - mean, 0.0)
        var_sums = (centered ** 2).sum(axis=0)
        var = np.divide(var_sums, counts, out=np.ones_like(var_sums, dtype=float), where=counts > 0)
        std = np.sqrt(np.maximum(var, 0.0))

        self.mean = np.where(np.isfinite(mean), mean, 0.0)
        self.std = np.where((~np.isfinite(std)) | (std == 0), 1.0, std)

    def transform_vector(self, vector: Any) -> Any:
        assert self.mean is not None and self.std is not None
        v = np.asarray(vector, dtype=float)
        v = np.where(np.isfinite(v), v, self.mean)
        return (v - self.mean) / self.std


def fighter_side_numeric_columns(df: Any, side_prefix: str) -> list[str]:
    cols = []
    excluded_suffixes = {"id", "name", "dob", "stance"}
    for col in df.columns:
        if not col.startswith(side_prefix):
            continue
        suffix = col[len(side_prefix) :]
        if suffix in excluded_suffixes:
            continue
        if suffix.endswith("_id") or suffix.endswith("_name"):
            continue
        # Inactivity is injected explicitly into sequence entries so it is always present
        # even if source CSV naming drifts.
        if suffix == "days_since_last_fight":
            continue
        # Height/reach/stance are static physical traits and are appended after
        # sequence encoding (not consumed as timestep inputs).
        if not (suffix.endswith("_pre") or suffix in {"age_days"}):
            continue
        cols.append(col)
    cols = sorted(cols)
    return cols


def pad_history(history_vectors: list[Any], max_len: int, feat_dim: int) -> tuple[Any, int]:
    arr = np.zeros((max_len, feat_dim), dtype=np.float32)
    if not history_vectors:
        return arr, 1
    effective = history_vectors[-max_len:]
    start = max_len - len(effective)
    arr[start:, :] = np.asarray(effective, dtype=np.float32)
    return arr, len(effective)


def standardize_dense_matrix(matrix: Any, train_mask: Any) -> Any:
    train_raw = matrix[train_mask]
    fill = np.nanmedian(train_raw, axis=0)
    fill = np.where(np.isnan(fill), 0.0, fill)
    train_filled = np.where(np.isnan(train_raw), fill, train_raw)
    matrix = np.where(np.isnan(matrix), fill, matrix)
    mean = train_filled.mean(axis=0)
    std = train_filled.std(axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where((std == 0) | np.isnan(std), 1.0, std)
    return (matrix - mean) / std


def build_siamese_dataset(df: Any, max_seq_len: int) -> SiamesePreparedData:
    f1_cols = fighter_side_numeric_columns(df, "fighter_1_")
    f2_cols = ["fighter_2_" + c[len("fighter_1_") :] for c in f1_cols]
    train_mask_df = (df[SPLIT_COL] == "train").to_numpy()

    static_numeric = [c for c in STATIC_NUMERIC_COLS if c in df.columns]
    static_categorical = [c for c in STATIC_CATEGORICAL_COLS if c in df.columns]
    static_encoder = StaticFeatureEncoder(static_numeric, static_categorical)
    static_encoder.fit(df[df[SPLIT_COL] == "train"])
    static_all = static_encoder.transform(df)

    physical_numeric = [c for c in PHYSICAL_NUMERIC_COLS if c in df.columns]
    physical_categorical = [c for c in PHYSICAL_CATEGORICAL_COLS if c in df.columns]
    physical_encoder = StaticFeatureEncoder(physical_numeric, physical_categorical)
    physical_encoder.fit(df[df[SPLIT_COL] == "train"])
    physical_all = physical_encoder.transform(df)

    histories: dict[str, list[Any]] = {}
    last_event_dates: dict[str, Any] = {}
    samples = []
    train_vectors: list[Any] = []

    for row_idx, row in df.iterrows():
        fighter_1_id = str(row["fighter_1_id"])
        fighter_2_id = str(row["fighter_2_id"])
        event_date = pd.Timestamp(row["event_date"]).normalize()

        seq1_hist = histories.get(fighter_1_id, [])
        seq2_hist = histories.get(fighter_2_id, [])

        samples.append(
            {
                "row_idx": row_idx,
                "split": row[SPLIT_COL],
                "fight_id": row["fight_id"],
                "event_date": row["event_date"],
                "weight_class": row.get("weight_class", ""),
                "gender": row.get("gender", ""),
                "is_title_bout": row.get("is_title_bout", 0),
                "y": int(row[TARGET_COL]),
                "seq1_hist": list(seq1_hist[-max_seq_len:]),
                "seq2_hist": list(seq2_hist[-max_seq_len:]),
            }
        )

        f1_vector = [safe_float(row.get(col)) for col in f1_cols]
        f2_vector = [safe_float(row.get(col)) for col in f2_cols]
        f1_days_since_last = resolve_days_since_last_fight(
            row.get("fighter_1_days_since_last_fight"),
            event_date,
            last_event_dates.get(fighter_1_id),
        )
        f2_days_since_last = resolve_days_since_last_fight(
            row.get("fighter_2_days_since_last_fight"),
            event_date,
            last_event_dates.get(fighter_2_id),
        )
        f1_opp_win_rate_pre = resolve_prefight_win_rate(row, "fighter_2_")
        f2_opp_win_rate_pre = resolve_prefight_win_rate(row, "fighter_1_")

        method_cat = infer_method_category(row.get("result_method_category", row.get("result_method", "")))
        method_flags = [1.0 if method_cat == c else 0.0 for c in METHOD_CATEGORIES]
        duration = safe_float(row.get("fight_duration_seconds"))
        rounds = safe_float(row.get("round_ended"))
        title = safe_float(row.get("is_title_bout"))
        scheduled = safe_float(row.get("scheduled_rounds"))

        f1_result = 1.0 if int(row[TARGET_COL]) == 1 else 0.0
        f2_result = 1.0 - f1_result

        f1_entry = np.asarray(
            f1_vector
            + [f1_days_since_last, f1_opp_win_rate_pre, f1_result, duration, rounds, title, scheduled]
            + method_flags,
            dtype=float,
        )
        f2_entry = np.asarray(
            f2_vector
            + [f2_days_since_last, f2_opp_win_rate_pre, f2_result, duration, rounds, title, scheduled]
            + method_flags,
            dtype=float,
        )

        histories.setdefault(fighter_1_id, []).append(f1_entry)
        histories.setdefault(fighter_2_id, []).append(f2_entry)
        last_event_dates[fighter_1_id] = event_date
        last_event_dates[fighter_2_id] = event_date

        if row[SPLIT_COL] == "train":
            train_vectors.append(f1_entry)
            train_vectors.append(f2_entry)

    if not train_vectors:
        raise ValueError("No training vectors were built for Siamese dataset.")

    normalizer = SequenceNormalizer()
    normalizer.fit(train_vectors)

    base_seq_names = [c[len("fighter_1_") :] for c in f1_cols]
    extra_names = [
        "days_since_last_fight",
        "opponent_win_rate_pre",
        "result_binary",
        "fight_duration_seconds",
        "round_ended",
        "is_title_bout",
        "scheduled_rounds",
    ]
    method_names = [f"method__{c}" for c in METHOD_CATEGORIES]
    seq_feature_names = base_seq_names + extra_names + method_names

    seq1_all = []
    len1_all = []
    seq2_all = []
    len2_all = []
    y_all = []
    split_all = []
    for sample in samples:
        seq1_vecs = [normalizer.transform_vector(v) for v in sample["seq1_hist"]]
        seq2_vecs = [normalizer.transform_vector(v) for v in sample["seq2_hist"]]
        pad1, l1 = pad_history(seq1_vecs, max_seq_len, len(seq_feature_names))
        pad2, l2 = pad_history(seq2_vecs, max_seq_len, len(seq_feature_names))
        seq1_all.append(pad1)
        len1_all.append(l1)
        seq2_all.append(pad2)
        len2_all.append(l2)
        y_all.append(sample["y"])
        split_all.append(sample["split"])

    seq1_all = np.asarray(seq1_all, dtype=np.float32)
    len1_all = np.asarray(len1_all, dtype=np.int64)
    seq2_all = np.asarray(seq2_all, dtype=np.float32)
    len2_all = np.asarray(len2_all, dtype=np.int64)
    y_all = np.asarray(y_all, dtype=np.float32)

    if static_all.shape[1] == 0:
        static_all = np.zeros((len(df), 1), dtype=float)
        static_feature_names = ["__no_static__"]
    else:
        static_feature_names = static_encoder.feature_names
    static_all = standardize_dense_matrix(static_all, train_mask_df)

    if physical_all.shape[1] == 0:
        physical_all = np.zeros((len(df), 1), dtype=float)
        physical_feature_names = ["__no_physical__"]
    else:
        physical_feature_names = physical_encoder.feature_names
    physical_all = standardize_dense_matrix(physical_all, train_mask_df)

    split_all = np.array(split_all)
    train_mask = split_all == "train"
    val_mask = split_all == "val"
    test_mask = split_all == "test"

    test_meta = df.loc[test_mask, ["fight_id", "event_date", "weight_class", "gender", "is_title_bout"]].copy()
    test_meta = test_meta.reset_index(drop=True)

    return SiamesePreparedData(
        seq1_train=seq1_all[train_mask],
        len1_train=len1_all[train_mask],
        seq2_train=seq2_all[train_mask],
        len2_train=len2_all[train_mask],
        static_train=static_all[train_mask].astype(np.float32),
        physical_train=physical_all[train_mask].astype(np.float32),
        y_train=y_all[train_mask],
        seq1_val=seq1_all[val_mask],
        len1_val=len1_all[val_mask],
        seq2_val=seq2_all[val_mask],
        len2_val=len2_all[val_mask],
        static_val=static_all[val_mask].astype(np.float32),
        physical_val=physical_all[val_mask].astype(np.float32),
        y_val=y_all[val_mask],
        seq1_test=seq1_all[test_mask],
        len1_test=len1_all[test_mask],
        seq2_test=seq2_all[test_mask],
        len2_test=len2_all[test_mask],
        static_test=static_all[test_mask].astype(np.float32),
        physical_test=physical_all[test_mask].astype(np.float32),
        y_test=y_all[test_mask],
        test_meta=test_meta,
        sequence_feature_names=seq_feature_names,
        static_feature_names=static_feature_names,
        physical_feature_names=physical_feature_names,
    )


class FightPairDataset(Dataset):
    def __init__(
        self,
        seq1: Any,
        len1: Any,
        seq2: Any,
        len2: Any,
        static: Any,
        physical: Any,
        y: Any,
    ) -> None:
        self.seq1 = torch.tensor(seq1, dtype=torch.float32)
        self.len1 = torch.tensor(len1, dtype=torch.int64)
        self.seq2 = torch.tensor(seq2, dtype=torch.float32)
        self.len2 = torch.tensor(len2, dtype=torch.int64)
        self.static = torch.tensor(static, dtype=torch.float32)
        self.physical = torch.tensor(physical, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "seq1": self.seq1[idx],
            "len1": self.len1[idx],
            "seq2": self.seq2[idx],
            "len2": self.len2[idx],
            "static": self.static[idx],
            "physical": self.physical[idx],
            "y": self.y[idx],
        }


class SiameseGRUModel(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        static_dim: int,
        physical_dim: int,
        hidden_dim: int,
        static_hidden_dim: int,
        num_layers: int,
        dropout: float,
        interaction_mode: str = "full",
    ) -> None:
        super().__init__()
        self.interaction_mode = interaction_mode
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, static_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if interaction_mode == "full":
            fusion_dim = hidden_dim * 4 + static_hidden_dim + physical_dim
        elif interaction_mode == "concat":
            fusion_dim = hidden_dim * 2 + static_hidden_dim + physical_dim
        else:
            raise ValueError(f"Unsupported interaction_mode: {interaction_mode}")

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_sequence(self, seq: Any, lengths: Any) -> Any:
        x = torch.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.relu(self.seq_proj(x))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.encoder(packed)
        return h[-1]

    def forward(
        self,
        seq1: Any,
        len1: Any,
        seq2: Any,
        len2: Any,
        static: Any,
        physical: Any,
    ) -> Any:
        h1 = self.encode_sequence(seq1, len1)
        h2 = self.encode_sequence(seq2, len2)
        s = self.static_mlp(static)
        if self.interaction_mode == "full":
            pair = torch.cat([h1, h2, torch.abs(h1 - h2), h1 * h2, s, physical], dim=1)
        else:
            pair = torch.cat([h1, h2, s, physical], dim=1)
        logits = self.head(pair).squeeze(1)
        return logits


def evaluate_siamese(
    model: Any,
    dataloader: Any,
    device: Any,
    criterion: Any,
    non_blocking: bool = False,
) -> tuple[float, Any, Any]:
    model.eval()
    losses = []
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            seq1 = batch["seq1"].to(device, non_blocking=non_blocking)
            len1 = batch["len1"].to(device, non_blocking=non_blocking)
            seq2 = batch["seq2"].to(device, non_blocking=non_blocking)
            len2 = batch["len2"].to(device, non_blocking=non_blocking)
            static = batch["static"].to(device, non_blocking=non_blocking)
            physical = batch["physical"].to(device, non_blocking=non_blocking)
            y = batch["y"].to(device, non_blocking=non_blocking)

            logits = model(seq1, len1, seq2, len2, static, physical)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits)

            losses.append(float(loss.item()))
            all_probs.append(probs.detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    probs_np = np.concatenate(all_probs) if all_probs else np.array([])
    targets_np = np.concatenate(all_targets) if all_targets else np.array([])
    loss_value = float(np.mean(losses)) if losses else float("nan")
    return loss_value, targets_np, probs_np


def train_siamese_model(
    prepared: SiamesePreparedData,
    config: SiameseConfig,
    interaction_mode: str,
    device: str,
    num_workers: int,
) -> tuple[Any, Any, dict[str, Any], list[dict[str, Any]]]:
    train_ds = FightPairDataset(
        prepared.seq1_train,
        prepared.len1_train,
        prepared.seq2_train,
        prepared.len2_train,
        prepared.static_train,
        prepared.physical_train,
        prepared.y_train,
    )
    val_ds = FightPairDataset(
        prepared.seq1_val,
        prepared.len1_val,
        prepared.seq2_val,
        prepared.len2_val,
        prepared.static_val,
        prepared.physical_val,
        prepared.y_val,
    )
    test_ds = FightPairDataset(
        prepared.seq1_test,
        prepared.len1_test,
        prepared.seq2_test,
        prepared.len2_test,
        prepared.static_test,
        prepared.physical_test,
        prepared.y_test,
    )

    dataloader_kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": max(0, int(num_workers)),
        "pin_memory": device == "cuda",
    }
    if dataloader_kwargs["num_workers"] > 0:
        dataloader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **dataloader_kwargs)
    non_blocking = bool(dataloader_kwargs["pin_memory"])

    model = SiameseGRUModel(
        seq_dim=prepared.seq1_train.shape[2],
        static_dim=prepared.static_train.shape[1],
        physical_dim=prepared.physical_train.shape[1],
        hidden_dim=config.hidden_dim,
        static_hidden_dim=config.static_hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        interaction_mode=interaction_mode,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-5,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val_score = float("inf")
    no_improve_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            seq1 = batch["seq1"].to(device, non_blocking=non_blocking)
            len1 = batch["len1"].to(device, non_blocking=non_blocking)
            seq2 = batch["seq2"].to(device, non_blocking=non_blocking)
            len2 = batch["len2"].to(device, non_blocking=non_blocking)
            static = batch["static"].to(device, non_blocking=non_blocking)
            physical = batch["physical"].to(device, non_blocking=non_blocking)
            y = batch["y"].to(device, non_blocking=non_blocking)

            optimizer.zero_grad()
            logits = model(seq1, len1, seq2, len2, static, physical)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss, y_val, p_val = evaluate_siamese(
            model,
            val_loader,
            device,
            criterion,
            non_blocking=non_blocking,
        )
        val_metrics = compute_metrics(y_val, p_val)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )
        logging.info(
            "Siamese[%s] epoch %s/%s | train_loss=%.4f | val_log_loss=%.4f | val_auc=%.4f",
            interaction_mode,
            epoch,
            config.epochs,
            train_loss,
            val_metrics["log_loss"],
            val_metrics["roc_auc"],
        )
        scheduler.step(val_metrics["log_loss"])

        if val_metrics["log_loss"] < best_val_score:
            best_val_score = val_metrics["log_loss"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= config.patience:
                logging.info("Early stopping triggered after %s epochs", epoch)
                break

    model.load_state_dict(best_state)
    _, y_val_best, p_val_best = evaluate_siamese(
        model,
        val_loader,
        device,
        criterion,
        non_blocking=non_blocking,
    )
    _, y_test_best, p_test_best = evaluate_siamese(
        model,
        test_loader,
        device,
        criterion,
        non_blocking=non_blocking,
    )
    val_metrics = compute_metrics(y_val_best, p_val_best)
    test_metrics = compute_metrics(y_test_best, p_test_best)
    return p_val_best, p_test_best, {"val": val_metrics, "test": test_metrics}, history


def indices_from_names(names: list[str], excludes: list[str]) -> list[int]:
    out = []
    for i, n in enumerate(names):
        if any(token in n for token in excludes):
            continue
        out.append(i)
    return out


def subset_prepared_data(
    prepared: SiamesePreparedData,
    seq_indices: list[int],
    static_indices: list[int],
) -> SiamesePreparedData:
    def _subset_seq(arr: Any) -> Any:
        if seq_indices:
            return arr[:, :, seq_indices]
        return np.zeros((arr.shape[0], arr.shape[1], 1), dtype=np.float32)

    def _subset_static(arr: Any) -> Any:
        if static_indices:
            return arr[:, static_indices]
        return np.zeros((arr.shape[0], 1), dtype=np.float32)

    sequence_names = (
        [prepared.sequence_feature_names[i] for i in seq_indices]
        if seq_indices
        else ["__no_sequence__"]
    )
    static_names = (
        [prepared.static_feature_names[i] for i in static_indices]
        if static_indices
        else ["__no_static__"]
    )
    return SiamesePreparedData(
        seq1_train=_subset_seq(prepared.seq1_train),
        len1_train=prepared.len1_train,
        seq2_train=_subset_seq(prepared.seq2_train),
        len2_train=prepared.len2_train,
        static_train=_subset_static(prepared.static_train),
        physical_train=prepared.physical_train,
        y_train=prepared.y_train,
        seq1_val=_subset_seq(prepared.seq1_val),
        len1_val=prepared.len1_val,
        seq2_val=_subset_seq(prepared.seq2_val),
        len2_val=prepared.len2_val,
        static_val=_subset_static(prepared.static_val),
        physical_val=prepared.physical_val,
        y_val=prepared.y_val,
        seq1_test=_subset_seq(prepared.seq1_test),
        len1_test=prepared.len1_test,
        seq2_test=_subset_seq(prepared.seq2_test),
        len2_test=prepared.len2_test,
        static_test=_subset_static(prepared.static_test),
        physical_test=prepared.physical_test,
        y_test=prepared.y_test,
        test_meta=prepared.test_meta.copy(),
        sequence_feature_names=sequence_names,
        static_feature_names=static_names,
        physical_feature_names=prepared.physical_feature_names,
    )


def run_siamese_and_ablations(
    prepared: SiamesePreparedData,
    config: SiameseConfig,
    run_ablations: bool,
    device: str,
    num_workers: int,
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    results = {}
    histories = []

    p_val, p_test, metrics, history = train_siamese_model(
        prepared=prepared,
        config=config,
        interaction_mode="full",
        device=device,
        num_workers=num_workers,
    )
    results["siamese_rnn"] = {
        "val_probs": p_val,
        "test_probs": p_test,
        "metrics": metrics,
    }
    histories.extend([{"model": "siamese_rnn", **row} for row in history])

    ablation_rows = []
    if run_ablations:
        variants = [
            ("siamese_no_context", "full", [], ["weight_class__", "gender__", "event_country__", "is_title_bout", "scheduled_rounds", "event_"]),
            ("siamese_no_efficiency", "full", ["sig_str", "td_", "sub_", "control", "knockdowns"], []),
            ("siamese_concat_only", "concat", [], []),
        ]
        all_seq_idx = list(range(len(prepared.sequence_feature_names)))
        all_static_idx = list(range(len(prepared.static_feature_names)))
        for variant_name, interaction_mode, seq_excludes, static_excludes in variants:
            seq_idx = all_seq_idx
            static_idx = all_static_idx
            if seq_excludes:
                seq_idx = indices_from_names(prepared.sequence_feature_names, seq_excludes)
            if static_excludes:
                static_idx = indices_from_names(prepared.static_feature_names, static_excludes)
            sub_data = subset_prepared_data(prepared, seq_idx, static_idx)
            logging.info("Running ablation: %s", variant_name)
            v_val, v_test, v_metrics, v_history = train_siamese_model(
                prepared=sub_data,
                config=config,
                interaction_mode=interaction_mode,
                device=device,
                num_workers=num_workers,
            )
            results[variant_name] = {"val_probs": v_val, "test_probs": v_test, "metrics": v_metrics}
            histories.extend([{"model": variant_name, **row} for row in v_history])
            ablation_rows.append(
                {
                    "model": variant_name,
                    **{f"val_{k}": v for k, v in v_metrics["val"].items()},
                    **{f"test_{k}": v for k, v in v_metrics["test"].items()},
                }
            )

    return results, pd.DataFrame(ablation_rows), histories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UFC Siamese network study pipeline (baselines + RNN Siamese + analyses)."
    )
    parser.add_argument("--input-csv", type=Path, default=Path("data/ufc_fights_rnn.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", type=str, default="ufc_siamese_run")
    parser.add_argument(
        "--split-mode",
        type=str,
        default="strict",
        choices=["strict", "expanding_window"],
        help=(
            "Temporal split policy. 'strict' uses explicit cutoff dates when provided; otherwise "
            "it infers year-based cutoffs. 'expanding_window' infers cutoffs from rolling year windows."
        ),
    )
    parser.add_argument(
        "--train-end-date",
        type=str,
        default=None,
        help="Train split end date (inclusive) in YYYY-MM-DD. Must be paired with --val-end-date.",
    )
    parser.add_argument(
        "--train-start-date",
        type=str,
        default=None,
        help="Optional earliest date (inclusive) kept for modeling in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--val-end-date",
        type=str,
        default=None,
        help="Validation split end date (inclusive) in YYYY-MM-DD. Test split starts after this date.",
    )
    parser.add_argument(
        "--val-years",
        type=int,
        default=1,
        help="Number of full calendar years in validation window when cutoffs are inferred.",
    )
    parser.add_argument(
        "--test-years",
        type=int,
        default=1,
        help="Number of full calendar years in test window when cutoffs are inferred.",
    )
    parser.add_argument(
        "--expanding-test-end-year",
        type=int,
        default=None,
        help="Anchor year for inferred expanding-window test end (for example, 2023).",
    )
    parser.add_argument(
        "--include-partial-latest-year",
        action="store_true",
        help="Allow inferred windows to include a partially observed latest calendar year.",
    )
    parser.add_argument(
        "--power-profile",
        type=str,
        default="max_power",
        choices=sorted(SUPPORTED_POWER_PROFILES),
        help="Model capacity preset. 'max_power' scales model size by available CPU/RAM.",
    )
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--static-hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes for Siamese training/evaluation.",
    )
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument(
        "--calibration-min-improvement",
        type=float,
        default=0.005,
        help=(
            "Minimum validation log-loss gain required before applying Platt calibration "
            "to held-out probabilities."
        ),
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
    if device_arg == "auto":
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"
    if device_arg == "cuda" and not cuda_ok:
        raise ValueError("CUDA requested but torch.cuda.is_available() is False.")
    if device_arg == "mps" and not mps_ok:
        raise ValueError("MPS requested but torch.backends.mps is unavailable.")
    return device_arg


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if MISSING_ML_DEPS:
        raise SystemExit(
            "Missing ML dependencies. Install with: pip install -r requirements.txt"
        )

    set_seed(args.seed)
    out_dir = create_output_dir(args.output_dir, args.run_name)
    logging.info("Output directory: %s", out_dir)
    capacity_cfg = get_training_capacity_profile(power_profile=args.power_profile, seed=args.seed)
    args.hidden_dim = int(
        args.hidden_dim if args.hidden_dim is not None else capacity_cfg["siamese_hidden_dim"]
    )
    args.static_hidden_dim = int(
        args.static_hidden_dim
        if args.static_hidden_dim is not None
        else capacity_cfg["siamese_static_hidden_dim"]
    )
    args.num_layers = int(
        args.num_layers if args.num_layers is not None else capacity_cfg["siamese_num_layers"]
    )
    args.dropout = float(args.dropout if args.dropout is not None else capacity_cfg["siamese_dropout"])
    args.batch_size = int(
        args.batch_size if args.batch_size is not None else capacity_cfg["siamese_batch_size"]
    )
    args.epochs = int(args.epochs if args.epochs is not None else capacity_cfg["siamese_epochs"])
    args.lr = float(args.lr if args.lr is not None else capacity_cfg["siamese_lr"])
    args.weight_decay = float(
        args.weight_decay if args.weight_decay is not None else capacity_cfg["siamese_weight_decay"]
    )
    args.patience = int(args.patience if args.patience is not None else capacity_cfg["siamese_patience"])
    logging.info(
        "Power profile: %s (tier=%s, cpu=%s, ram_gb=%.1f)",
        capacity_cfg["profile"],
        capacity_cfg["capacity_tier"],
        capacity_cfg["cpu_count"],
        capacity_cfg["ram_gb"],
    )

    df = load_and_prepare_dataframe(args.input_csv)
    if args.train_start_date:
        train_start = _parse_date_arg("--train-start-date", args.train_start_date)
        before = len(df)
        df = df[df["event_date"].dt.normalize() >= train_start].copy()
        logging.info(
            "Applied train-start-date filter (%s): kept %s/%s rows.",
            train_start.date(),
            len(df),
            before,
        )
    df, orientation_info = rebalance_binary_orientation_if_needed(df, seed=args.seed)
    if orientation_info["applied"]:
        logging.warning(
            "Detected collapsed binary orientation; swapped fighter sides on %s rows "
            "to rebalance classes. Outcome counts before=%s after=%s",
            orientation_info["swapped_rows"],
            orientation_info["counts_before"],
            orientation_info["counts_after"],
        )
    ensure_minimum_rows(df, min_rows=200)
    split_info = temporal_split_dataframe(
        df=df,
        split_mode=args.split_mode,
        train_end_date=args.train_end_date,
        val_end_date=args.val_end_date,
        val_years=args.val_years,
        test_years=args.test_years,
        expanding_test_end_year=args.expanding_test_end_year,
        include_partial_latest_year=args.include_partial_latest_year,
    )
    logging.info("Temporal split mode: %s", split_info.mode)
    logging.info("Temporal split counts: %s", split_info.counts)
    logging.info(
        "Train range: %s -> %s | Val range: %s -> %s | Test starts: %s",
        split_info.train_start_date,
        split_info.train_end_date,
        split_info.val_start_date,
        split_info.val_end_date,
        split_info.test_start_date,
    )

    baseline_results, baseline_meta = train_baseline_models(
        df=df,
        power_profile=args.power_profile,
        seed=args.seed,
    )

    val_df = df[df[SPLIT_COL] == "val"].copy()
    test_df = df[df[SPLIT_COL] == "test"].copy()
    naive_val_probs = naive_prefight_winrate_probs(val_df)
    naive_test_probs = naive_prefight_winrate_probs(test_df)
    baseline_results.append(
        BaselineResult(
            name="naive_prefight_winrate",
            val_probs=naive_val_probs,
            test_probs=naive_test_probs,
            metrics_val=compute_metrics(val_df[TARGET_COL].to_numpy(dtype=int), naive_val_probs),
            metrics_test=compute_metrics(test_df[TARGET_COL].to_numpy(dtype=int), naive_test_probs),
        )
    )
    y_val_local = val_df[TARGET_COL].to_numpy(dtype=int)
    y_test_local = test_df[TARGET_COL].to_numpy(dtype=int)

    probability_calibration: dict[str, Any] = {}
    recalibrated_baselines: list[BaselineResult] = []
    for baseline in baseline_results:
        val_probs_cal, test_probs_cal, cal_info = maybe_apply_platt_calibration(
            y_val_local,
            baseline.val_probs,
            baseline.test_probs,
            min_improvement=args.calibration_min_improvement,
        )
        probability_calibration[baseline.name] = cal_info
        if cal_info.get("applied", False):
            logging.info(
                "Applied Platt calibration to %s (val log loss %.4f -> %.4f)",
                baseline.name,
                cal_info.get("val_log_loss_raw", float("nan")),
                cal_info.get("val_log_loss_calibrated", float("nan")),
            )
        recalibrated_baselines.append(
            BaselineResult(
                name=baseline.name,
                val_probs=val_probs_cal,
                test_probs=test_probs_cal,
                metrics_val=compute_metrics(y_val_local, val_probs_cal),
                metrics_test=compute_metrics(y_test_local, test_probs_cal),
            )
        )
    baseline_results = recalibrated_baselines

    siamese_cfg = SiameseConfig(
        max_seq_len=args.max_seq_len,
        hidden_dim=args.hidden_dim,
        static_hidden_dim=args.static_hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )
    prepared = build_siamese_dataset(df, max_seq_len=siamese_cfg.max_seq_len)
    device = resolve_device(args.device)
    logging.info("Training device: %s", device)

    siamese_results, ablation_df, histories = run_siamese_and_ablations(
        prepared=prepared,
        config=siamese_cfg,
        run_ablations=not args.skip_ablations,
        device=device,
        num_workers=args.num_workers,
    )
    y_val_siam = np.asarray(prepared.y_val, dtype=int)
    y_test_siam = np.asarray(prepared.y_test, dtype=int)
    for model_name, payload in siamese_results.items():
        val_probs_cal, test_probs_cal, cal_info = maybe_apply_platt_calibration(
            y_val_siam,
            payload["val_probs"],
            payload["test_probs"],
            min_improvement=args.calibration_min_improvement,
        )
        payload["val_probs"] = val_probs_cal
        payload["test_probs"] = test_probs_cal
        payload["metrics"] = {
            "val": compute_metrics(y_val_siam, val_probs_cal),
            "test": compute_metrics(y_test_siam, test_probs_cal),
        }
        probability_calibration[model_name] = cal_info
        if cal_info.get("applied", False):
            logging.info(
                "Applied Platt calibration to %s (val log loss %.4f -> %.4f)",
                model_name,
                cal_info.get("val_log_loss_raw", float("nan")),
                cal_info.get("val_log_loss_calibrated", float("nan")),
            )
    if not ablation_df.empty and "model" in ablation_df.columns:
        for idx, row in ablation_df.iterrows():
            model_name = str(row["model"])
            if model_name not in siamese_results:
                continue
            metrics = siamese_results[model_name]["metrics"]
            for key, value in metrics["val"].items():
                ablation_df.loc[idx, f"val_{key}"] = value
            for key, value in metrics["test"].items():
                ablation_df.loc[idx, f"test_{key}"] = value

    metrics_rows = []
    test_predictions = prepared.test_meta.copy()
    test_predictions[TARGET_COL] = prepared.y_test.astype(int)
    test_df_local = df[df[SPLIT_COL] == "test"].copy().reset_index(drop=True)
    y_test_local = test_df_local[TARGET_COL].to_numpy(dtype=int)

    for baseline in baseline_results:
        metrics_rows.append(
            {
                "model": baseline.name,
                **{f"val_{k}": v for k, v in baseline.metrics_val.items()},
                **{f"test_{k}": v for k, v in baseline.metrics_test.items()},
            }
        )
        probs = baseline.test_probs
        test_predictions[f"pred_{baseline.name}"] = probs
        subgroup_df = subgroup_metrics(test_df_local, probs, baseline.name)
        if not subgroup_df.empty:
            subgroup_path = out_dir / f"subgroup_{baseline.name}.csv"
            subgroup_df.to_csv(subgroup_path, index=False)
        calibration_df = calibration_table(y_test_local, probs, n_bins=10)
        calibration_df.insert(0, "model", baseline.name)
        calibration_df.to_csv(out_dir / f"calibration_{baseline.name}.csv", index=False)

    for model_name, payload in siamese_results.items():
        metrics_rows.append(
            {
                "model": model_name,
                **{f"val_{k}": v for k, v in payload["metrics"]["val"].items()},
                **{f"test_{k}": v for k, v in payload["metrics"]["test"].items()},
            }
        )
        test_predictions[f"pred_{model_name}"] = payload["test_probs"]

        calibration_df = calibration_table(prepared.y_test, payload["test_probs"], n_bins=10)
        calibration_df.insert(0, "model", model_name)
        calibration_df.to_csv(out_dir / f"calibration_{model_name}.csv", index=False)

        subgroup_df = subgroup_metrics(test_df_local, payload["test_probs"], model_name)
        if not subgroup_df.empty:
            subgroup_df.to_csv(out_dir / f"subgroup_{model_name}.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("test_log_loss")
    metrics_df.to_csv(out_dir / "main_metrics.csv", index=False)

    test_predictions.to_csv(out_dir / "test_predictions.csv", index=False)
    pd.DataFrame(histories).to_csv(out_dir / "siamese_training_history.csv", index=False)
    if not ablation_df.empty:
        ablation_df.to_csv(out_dir / "ablation_metrics.csv", index=False)

    metadata = {
        "input_csv": str(args.input_csv),
        "rows_total": int(len(df)),
        "split": dataclasses.asdict(split_info),
        "split_request": {
            "split_mode": args.split_mode,
            "train_start_date": args.train_start_date,
            "train_end_date": args.train_end_date,
            "val_end_date": args.val_end_date,
            "val_years": args.val_years,
            "test_years": args.test_years,
            "expanding_test_end_year": args.expanding_test_end_year,
            "include_partial_latest_year": bool(args.include_partial_latest_year),
        },
        "orientation_rebalance": orientation_info,
        "probability_calibration": probability_calibration,
        "baseline_feature_counts": {
            "all": len(baseline_meta["feature_cols"]),
            "numeric": len(baseline_meta["numeric_cols"]),
            "categorical": len(baseline_meta["categorical_cols"]),
        },
        "siamese_config": dataclasses.asdict(siamese_cfg),
        "sequence_feature_count": len(prepared.sequence_feature_names),
        "static_feature_count": len(prepared.static_feature_names),
        "physical_feature_count": len(prepared.physical_feature_names),
        "device": device,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "power_profile": args.power_profile,
        "capacity": {
            "tier": capacity_cfg["capacity_tier"],
            "cpu_count": capacity_cfg["cpu_count"],
            "ram_gb": capacity_cfg["ram_gb"],
        },
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    logging.info("Completed. Main metrics: %s", out_dir / "main_metrics.csv")
    logging.info("Predictions: %s", out_dir / "test_predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
