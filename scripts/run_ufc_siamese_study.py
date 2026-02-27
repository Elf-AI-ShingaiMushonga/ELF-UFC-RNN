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
import json
import logging
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

METHOD_CATEGORIES = ["ko_tko", "submission", "decision", "dq", "other", "unknown"]


@dataclasses.dataclass
class TemporalSplit:
    train_end_date: str
    val_end_date: str
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
    y_train: Any
    seq1_val: Any
    len1_val: Any
    seq2_val: Any
    len2_val: Any
    static_val: Any
    y_val: Any
    seq1_test: Any
    len1_test: Any
    seq2_test: Any
    len2_test: Any
    static_test: Any
    y_test: Any
    test_meta: Any
    sequence_feature_names: list[str]
    static_feature_names: list[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def temporal_split_dataframe(
    df: Any,
    train_fraction: float,
    val_fraction: float,
) -> TemporalSplit:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("--train-fraction must be between 0 and 1.")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("--val-fraction must be between 0 and 1.")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("--train-fraction + --val-fraction must be < 1.0.")

    unique_dates = np.array(sorted(df["event_date"].dt.normalize().unique()))
    if len(unique_dates) < 3:
        raise ValueError("Need at least 3 unique event dates for train/val/test temporal split.")

    train_idx = max(0, min(len(unique_dates) - 3, int(len(unique_dates) * train_fraction) - 1))
    val_idx = max(
        train_idx + 1,
        min(len(unique_dates) - 2, int(len(unique_dates) * (train_fraction + val_fraction)) - 1),
    )
    train_end = unique_dates[train_idx]
    val_end = unique_dates[val_idx]

    split = np.where(
        df["event_date"].dt.normalize() <= train_end,
        "train",
        np.where(df["event_date"].dt.normalize() <= val_end, "val", "test"),
    )
    df[SPLIT_COL] = split
    counts = df[SPLIT_COL].value_counts().to_dict()

    if counts.get("train", 0) == 0 or counts.get("val", 0) == 0 or counts.get("test", 0) == 0:
        raise ValueError(
            "Temporal split produced an empty split. Adjust fractions or input date coverage."
        )

    return TemporalSplit(
        train_end_date=str(pd.Timestamp(train_end).date()),
        val_end_date=str(pd.Timestamp(val_end).date()),
        counts={k: int(v) for k, v in counts.items()},
    )


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


def build_tabular_pipeline(model_name: str, numeric_cols: list[str], categorical_cols: list[str]) -> Any:
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
        estimator = LogisticRegression(max_iter=600, class_weight="balanced")
    elif model_name == "gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            l2_regularization=0.1,
            random_state=42,
        )
    elif model_name == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=(192, 96),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=350,
            early_stopping=True,
            random_state=42,
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


def train_baseline_models(df: Any) -> tuple[list[BaselineResult], dict[str, Any]]:
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
        pipeline = build_tabular_pipeline(model_name, numeric_cols, categorical_cols)
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
        self.mean = np.nanmean(matrix, axis=0)
        self.std = np.nanstd(matrix, axis=0)
        self.mean = np.where(np.isnan(self.mean), 0.0, self.mean)
        self.std = np.where((self.std == 0) | np.isnan(self.std), 1.0, self.std)

    def transform_vector(self, vector: Any) -> Any:
        assert self.mean is not None and self.std is not None
        v = np.asarray(vector, dtype=float)
        v = np.where(np.isnan(v), self.mean, v)
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
        if not (suffix.endswith("_pre") or suffix in {"age_days", "height_cm", "reach_cm", "days_since_last_fight"}):
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


def build_siamese_dataset(df: Any, max_seq_len: int) -> SiamesePreparedData:
    f1_cols = fighter_side_numeric_columns(df, "fighter_1_")
    f2_cols = ["fighter_2_" + c[len("fighter_1_") :] for c in f1_cols]

    static_numeric = [c for c in STATIC_NUMERIC_COLS if c in df.columns]
    static_categorical = [c for c in STATIC_CATEGORICAL_COLS if c in df.columns]
    static_encoder = StaticFeatureEncoder(static_numeric, static_categorical)
    static_encoder.fit(df[df[SPLIT_COL] == "train"])
    static_all = static_encoder.transform(df)

    histories: dict[str, list[Any]] = {}
    samples = []
    train_vectors: list[Any] = []

    for row_idx, row in df.iterrows():
        fighter_1_id = str(row["fighter_1_id"])
        fighter_2_id = str(row["fighter_2_id"])

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

        method_cat = infer_method_category(row.get("result_method_category", row.get("result_method", "")))
        method_flags = [1.0 if method_cat == c else 0.0 for c in METHOD_CATEGORIES]
        duration = safe_float(row.get("fight_duration_seconds"))
        rounds = safe_float(row.get("round_ended"))
        title = safe_float(row.get("is_title_bout"))
        scheduled = safe_float(row.get("scheduled_rounds"))

        f1_result = 1.0 if int(row[TARGET_COL]) == 1 else 0.0
        f2_result = 1.0 - f1_result

        f1_entry = np.asarray(f1_vector + [f1_result, duration, rounds, title, scheduled] + method_flags, dtype=float)
        f2_entry = np.asarray(f2_vector + [f2_result, duration, rounds, title, scheduled] + method_flags, dtype=float)

        histories.setdefault(fighter_1_id, []).append(f1_entry)
        histories.setdefault(fighter_2_id, []).append(f2_entry)

        if row[SPLIT_COL] == "train":
            train_vectors.append(f1_entry)
            train_vectors.append(f2_entry)

    if not train_vectors:
        raise ValueError("No training vectors were built for Siamese dataset.")

    normalizer = SequenceNormalizer()
    normalizer.fit(train_vectors)

    base_seq_names = [c[len("fighter_1_") :] for c in f1_cols]
    extra_names = ["result_binary", "fight_duration_seconds", "round_ended", "is_title_bout", "scheduled_rounds"]
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

    static_train_raw = static_all[np.array(split_all) == "train"]
    static_fill = np.nanmedian(static_train_raw, axis=0)
    static_fill = np.where(np.isnan(static_fill), 0.0, static_fill)
    static_train_filled = np.where(np.isnan(static_train_raw), static_fill, static_train_raw)
    static_all = np.where(np.isnan(static_all), static_fill, static_all)
    static_mean = static_train_filled.mean(axis=0)
    static_std = static_train_filled.std(axis=0)
    static_mean = np.where(np.isnan(static_mean), 0.0, static_mean)
    static_std = np.where((static_std == 0) | np.isnan(static_std), 1.0, static_std)
    static_all = (static_all - static_mean) / static_std

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
        y_train=y_all[train_mask],
        seq1_val=seq1_all[val_mask],
        len1_val=len1_all[val_mask],
        seq2_val=seq2_all[val_mask],
        len2_val=len2_all[val_mask],
        static_val=static_all[val_mask].astype(np.float32),
        y_val=y_all[val_mask],
        seq1_test=seq1_all[test_mask],
        len1_test=len1_all[test_mask],
        seq2_test=seq2_all[test_mask],
        len2_test=len2_all[test_mask],
        static_test=static_all[test_mask].astype(np.float32),
        y_test=y_all[test_mask],
        test_meta=test_meta,
        sequence_feature_names=seq_feature_names,
        static_feature_names=static_feature_names,
    )


class FightPairDataset(Dataset):
    def __init__(
        self,
        seq1: Any,
        len1: Any,
        seq2: Any,
        len2: Any,
        static: Any,
        y: Any,
    ) -> None:
        self.seq1 = torch.tensor(seq1, dtype=torch.float32)
        self.len1 = torch.tensor(len1, dtype=torch.int64)
        self.seq2 = torch.tensor(seq2, dtype=torch.float32)
        self.len2 = torch.tensor(len2, dtype=torch.int64)
        self.static = torch.tensor(static, dtype=torch.float32)
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
            "y": self.y[idx],
        }


class SiameseGRUModel(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        static_dim: int,
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
            fusion_dim = hidden_dim * 4 + static_hidden_dim
        elif interaction_mode == "concat":
            fusion_dim = hidden_dim * 2 + static_hidden_dim
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

    def forward(self, seq1: Any, len1: Any, seq2: Any, len2: Any, static: Any) -> Any:
        h1 = self.encode_sequence(seq1, len1)
        h2 = self.encode_sequence(seq2, len2)
        s = self.static_mlp(static)
        if self.interaction_mode == "full":
            pair = torch.cat([h1, h2, torch.abs(h1 - h2), h1 * h2, s], dim=1)
        else:
            pair = torch.cat([h1, h2, s], dim=1)
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
            y = batch["y"].to(device, non_blocking=non_blocking)

            logits = model(seq1, len1, seq2, len2, static)
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
        prepared.y_train,
    )
    val_ds = FightPairDataset(
        prepared.seq1_val,
        prepared.len1_val,
        prepared.seq2_val,
        prepared.len2_val,
        prepared.static_val,
        prepared.y_val,
    )
    test_ds = FightPairDataset(
        prepared.seq1_test,
        prepared.len1_test,
        prepared.seq2_test,
        prepared.len2_test,
        prepared.static_test,
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
            y = batch["y"].to(device, non_blocking=non_blocking)

            optimizer.zero_grad()
            logits = model(seq1, len1, seq2, len2, static)
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
        y_train=prepared.y_train,
        seq1_val=_subset_seq(prepared.seq1_val),
        len1_val=prepared.len1_val,
        seq2_val=_subset_seq(prepared.seq2_val),
        len2_val=prepared.len2_val,
        static_val=_subset_static(prepared.static_val),
        y_val=prepared.y_val,
        seq1_test=_subset_seq(prepared.seq1_test),
        len1_test=prepared.len1_test,
        seq2_test=_subset_seq(prepared.seq2_test),
        len2_test=prepared.len2_test,
        static_test=_subset_static(prepared.static_test),
        y_test=prepared.y_test,
        test_meta=prepared.test_meta.copy(),
        sequence_feature_names=sequence_names,
        static_feature_names=static_names,
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
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--static-hidden-dim", type=int, default=48)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
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
            "Missing ML dependencies. Install with: pip install -r requirements-ml.txt"
        )

    set_seed(args.seed)
    out_dir = create_output_dir(args.output_dir, args.run_name)
    logging.info("Output directory: %s", out_dir)

    df = load_and_prepare_dataframe(args.input_csv)
    ensure_minimum_rows(df, min_rows=200)
    split_info = temporal_split_dataframe(df, args.train_fraction, args.val_fraction)
    logging.info("Temporal split counts: %s", split_info.counts)
    logging.info("Train end date: %s | Val end date: %s", split_info.train_end_date, split_info.val_end_date)

    baseline_results, baseline_meta = train_baseline_models(df)

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
        "baseline_feature_counts": {
            "all": len(baseline_meta["feature_cols"]),
            "numeric": len(baseline_meta["numeric_cols"]),
            "categorical": len(baseline_meta["categorical_cols"]),
        },
        "siamese_config": dataclasses.asdict(siamese_cfg),
        "sequence_feature_count": len(prepared.sequence_feature_names),
        "static_feature_count": len(prepared.static_feature_names),
        "device": device,
        "seed": args.seed,
        "num_workers": args.num_workers,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    logging.info("Completed. Main metrics: %s", out_dir / "main_metrics.csv")
    logging.info("Predictions: %s", out_dir / "test_predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
