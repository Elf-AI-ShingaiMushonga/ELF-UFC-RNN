#!/usr/bin/env python3
"""Train and deploy an LSTM model on raw UFC fight-detail data.

Input expected from scripts/scrape_ufc_fight_details.py:
  data/ufc_fight_details_lstm.csv
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset


RAW_REQUIRED_COLUMNS = [
    "fight_id",
    "event_id",
    "event_date",
    "bout_index",
    "is_main_event",
    "weight_class",
    "gender",
    "is_title_bout",
    "scheduled_rounds",
    "round_ended",
    "time_ended",
    "fight_duration_seconds",
    "result_method_category",
    "fighter_1_id",
    "fighter_1_status",
    "fighter_2_id",
    "fighter_2_status",
    "outcome_label",
    "kd_1",
    "kd_2",
    "sig_str_1_landed",
    "sig_str_1_attempted",
    "sig_str_2_landed",
    "sig_str_2_attempted",
    "td_1_landed",
    "td_1_attempted",
    "td_2_landed",
    "td_2_attempted",
    "sub_1",
    "sub_2",
    "ctrl_seconds_1",
    "ctrl_seconds_2",
]

RAW_NUMERIC_COLUMNS = [
    "bout_index",
    "is_main_event",
    "is_title_bout",
    "scheduled_rounds",
    "round_ended",
    "fight_duration_seconds",
    "kd_1",
    "kd_2",
    "sig_str_1_landed",
    "sig_str_1_attempted",
    "sig_str_2_landed",
    "sig_str_2_attempted",
    "td_1_landed",
    "td_1_attempted",
    "td_2_landed",
    "td_2_attempted",
    "sub_1",
    "sub_2",
    "ctrl_seconds_1",
    "ctrl_seconds_2",
]

POSITIVE_LABEL = "fighter_1_win"
NEGATIVE_LABEL = "fighter_2_win"

SEQ_FEATURE_NAMES = [
    "won",
    "finish_win",
    "finish_loss",
    "fight_minutes",
    "days_since_prev_log",
    "sig_str_landed_per_min",
    "sig_str_absorbed_per_min",
    "sig_str_accuracy",
    "sig_str_defense",
    "td_landed_per_15",
    "td_absorbed_per_15",
    "td_accuracy",
    "td_defense",
    "sub_attempts_per_15",
    "kd_for_per_15",
    "kd_against_per_15",
    "control_time_per_min",
    "title_bout",
    "scheduled_rounds_norm",
    "method_ko_win",
    "method_sub_win",
    "method_decision_win",
    "method_ko_loss",
    "method_sub_loss",
    "method_decision_loss",
]

STATIC_FEATURE_NAMES = [
    "log_fights_a",
    "log_fights_b",
    "log_fights_diff",
    "recent3_winrate_a",
    "recent3_winrate_b",
    "recent3_winrate_diff",
    "recent5_winrate_a",
    "recent5_winrate_b",
    "recent5_winrate_diff",
    "recent3_finish_a",
    "recent3_finish_b",
    "recent3_finish_diff",
    "recent3_sigacc_a",
    "recent3_sigacc_b",
    "recent3_sigacc_diff",
    "scheduled_rounds_norm",
    "is_title_bout",
    "is_main_event",
    "weight_lbs_norm",
    "is_female_bout",
]


@dataclasses.dataclass
class SequenceSample:
    sample_id: str
    event_date: pd.Timestamp
    seq_a: np.ndarray
    seq_b: np.ndarray
    static: np.ndarray
    target: int


@dataclasses.dataclass
class TransformedSample:
    sample_id: str
    event_date: pd.Timestamp
    seq_a: np.ndarray
    len_a: int
    seq_b: np.ndarray
    len_b: int
    static: np.ndarray
    target: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean, engineer, train, and deploy a UFC fight-details LSTM model."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/ufc_fight_details_lstm.csv"),
        help="Raw fight-details CSV from scrape_ufc_fight_details.py",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("data/model_cache/lstm_fight_details"),
        help="Directory for saved model, scalers, and metrics.",
    )
    parser.add_argument("--seq-len", type=int, default=12, help="Max sequence length per fighter.")
    parser.add_argument(
        "--min-history",
        type=int,
        default=2,
        help="Minimum prior fights required per fighter to build a sample.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=80, help="Max training epochs.")
    parser.add_argument("--patience", type=int, default=12, help="Early-stopping patience.")
    parser.add_argument("--hidden-size", type=int, default=96, help="LSTM hidden size per direction.")
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM layer count.")
    parser.add_argument("--dropout", type=float, default=0.28, help="Dropout probability.")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation split fraction.")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Test split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device selection.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of generated samples (debug/smoke runs).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 is safest across environments).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "mps":
        mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
        return torch.device("mps" if mps_ok else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if mps_ok:
        return torch.device("mps")
    return torch.device("cpu")


def safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def parse_weight_class_lbs(weight_class: str) -> float:
    text = (weight_class or "").lower()
    mapping = {
        "strawweight": 115,
        "flyweight": 125,
        "bantamweight": 135,
        "featherweight": 145,
        "lightweight": 155,
        "welterweight": 170,
        "middleweight": 185,
        "light heavyweight": 205,
        "heavyweight": 265,
        "catchweight": 180,
        "open weight": 265,
    }
    for label, lbs in mapping.items():
        if label in text:
            return float(lbs)
    return 170.0


def load_raw_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    for col in RAW_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in RAW_NUMERIC_COLUMNS:
        if col == "fight_duration_seconds":
            df[col] = df[col].fillna(0).clip(lower=0)
        else:
            df[col] = df[col].fillna(0).clip(lower=0)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"]).copy()
    df["outcome_label"] = df["outcome_label"].astype(str).str.strip()
    df["fighter_1_id"] = df["fighter_1_id"].astype(str).str.strip()
    df["fighter_2_id"] = df["fighter_2_id"].astype(str).str.strip()
    df["fight_id"] = df["fight_id"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["fight_id"]).copy()
    df = df.sort_values(["event_date", "bout_index", "fight_id"]).reset_index(drop=True)
    return df


def validate_raw_data(df: pd.DataFrame) -> None:
    if len(df) < 200:
        raise ValueError(f"Not enough fights for robust LSTM training. Found {len(df)} rows.")
    label_rows = df[df["outcome_label"].isin([POSITIVE_LABEL, NEGATIVE_LABEL])]
    if label_rows.empty:
        raise ValueError("No decisive outcomes found (fighter_1_win/fighter_2_win).")
    class_counts = label_rows["outcome_label"].value_counts().to_dict()
    if POSITIVE_LABEL not in class_counts or NEGATIVE_LABEL not in class_counts:
        logging.warning(
            "Raw decisive labels are one-sided (%s). "
            "This is acceptable when scraped rows are winner-first because mirrored samples "
            "later restore both classes.",
            class_counts,
        )
    key_attempt_cols = [
        "sig_str_1_attempted",
        "sig_str_2_attempted",
        "td_1_attempted",
        "td_2_attempted",
    ]
    coverage: dict[str, float] = {}
    for col in key_attempt_cols:
        coverage[col] = float((df[col] > 0).mean())
    logging.info("Raw scrape coverage (attempt columns): %s", coverage)
    if min(coverage.values()) < 0.25:
        raise ValueError(
            "Scraper output appears incomplete (very low attempted-stat coverage). "
            f"Coverage: {coverage}"
        )


def last_n_mean(history: list[dict[str, float]], key: str, n: int) -> float:
    if not history:
        return 0.0
    vals = [row[key] for row in history[-n:]]
    return float(np.mean(vals)) if vals else 0.0


def effective_fight_minutes(row: pd.Series) -> float:
    seconds = float(row.get("fight_duration_seconds", 0) or 0)
    if seconds <= 0:
        rounds = float(row.get("round_ended", 0) or 0)
        if rounds <= 0:
            rounds = float(row.get("scheduled_rounds", 3) or 3)
        seconds = max(rounds * 300.0, 300.0)
    return max(seconds / 60.0, 1e-3)


def build_fighter_fight_vector(
    row: pd.Series,
    *,
    side: int,
    previous_event_date: Optional[pd.Timestamp],
) -> tuple[np.ndarray, dict[str, float]]:
    event_date = pd.Timestamp(row["event_date"])
    minutes = effective_fight_minutes(row)

    if side == 1:
        sig_for_l = float(row["sig_str_1_landed"])
        sig_for_a = float(row["sig_str_1_attempted"])
        sig_against_l = float(row["sig_str_2_landed"])
        sig_against_a = float(row["sig_str_2_attempted"])
        td_for_l = float(row["td_1_landed"])
        td_for_a = float(row["td_1_attempted"])
        td_against_l = float(row["td_2_landed"])
        td_against_a = float(row["td_2_attempted"])
        kd_for = float(row["kd_1"])
        kd_against = float(row["kd_2"])
        sub_for = float(row["sub_1"])
        ctrl_for = float(row["ctrl_seconds_1"])
        won = 1.0 if row["outcome_label"] == POSITIVE_LABEL else 0.0
        lost = 1.0 if row["outcome_label"] == NEGATIVE_LABEL else 0.0
    else:
        sig_for_l = float(row["sig_str_2_landed"])
        sig_for_a = float(row["sig_str_2_attempted"])
        sig_against_l = float(row["sig_str_1_landed"])
        sig_against_a = float(row["sig_str_1_attempted"])
        td_for_l = float(row["td_2_landed"])
        td_for_a = float(row["td_2_attempted"])
        td_against_l = float(row["td_1_landed"])
        td_against_a = float(row["td_1_attempted"])
        kd_for = float(row["kd_2"])
        kd_against = float(row["kd_1"])
        sub_for = float(row["sub_2"])
        ctrl_for = float(row["ctrl_seconds_2"])
        won = 1.0 if row["outcome_label"] == NEGATIVE_LABEL else 0.0
        lost = 1.0 if row["outcome_label"] == POSITIVE_LABEL else 0.0

    method_cat = str(row.get("result_method_category", "")).lower()
    finish_win = 1.0 if won and method_cat in {"ko_tko", "submission"} else 0.0
    finish_loss = 1.0 if lost and method_cat in {"ko_tko", "submission"} else 0.0
    method_ko_win = 1.0 if won and method_cat == "ko_tko" else 0.0
    method_sub_win = 1.0 if won and method_cat == "submission" else 0.0
    method_decision_win = 1.0 if won and method_cat == "decision" else 0.0
    method_ko_loss = 1.0 if lost and method_cat == "ko_tko" else 0.0
    method_sub_loss = 1.0 if lost and method_cat == "submission" else 0.0
    method_decision_loss = 1.0 if lost and method_cat == "decision" else 0.0

    if previous_event_date is not None:
        days_since_prev = max((event_date - previous_event_date).days, 0)
    else:
        days_since_prev = 0

    sig_acc = safe_div(sig_for_l, sig_for_a)
    sig_def = safe_div(max(sig_against_a - sig_against_l, 0), sig_against_a)
    td_acc = safe_div(td_for_l, td_for_a)
    td_def = safe_div(max(td_against_a - td_against_l, 0), td_against_a)

    vector = np.array(
        [
            won,
            finish_win,
            finish_loss,
            minutes,
            math.log1p(days_since_prev),
            safe_div(sig_for_l, minutes),
            safe_div(sig_against_l, minutes),
            sig_acc,
            sig_def,
            safe_div(td_for_l * 15.0, minutes),
            safe_div(td_against_l * 15.0, minutes),
            td_acc,
            td_def,
            safe_div(sub_for * 15.0, minutes),
            safe_div(kd_for * 15.0, minutes),
            safe_div(kd_against * 15.0, minutes),
            safe_div(ctrl_for, minutes),
            float(row.get("is_title_bout", 0) or 0),
            safe_div(float(row.get("scheduled_rounds", 0) or 0), 5.0),
            method_ko_win,
            method_sub_win,
            method_decision_win,
            method_ko_loss,
            method_sub_loss,
            method_decision_loss,
        ],
        dtype=np.float32,
    )
    stats = {"won": won, "finish_win": finish_win, "sig_acc": sig_acc}
    return vector, stats


def build_static_features(
    hist_a: list[dict[str, float]],
    hist_b: list[dict[str, float]],
    row: pd.Series,
) -> np.ndarray:
    fights_a = float(len(hist_a))
    fights_b = float(len(hist_b))
    r3_win_a = last_n_mean(hist_a, "won", 3)
    r3_win_b = last_n_mean(hist_b, "won", 3)
    r5_win_a = last_n_mean(hist_a, "won", 5)
    r5_win_b = last_n_mean(hist_b, "won", 5)
    r3_fin_a = last_n_mean(hist_a, "finish_win", 3)
    r3_fin_b = last_n_mean(hist_b, "finish_win", 3)
    r3_sig_a = last_n_mean(hist_a, "sig_acc", 3)
    r3_sig_b = last_n_mean(hist_b, "sig_acc", 3)
    scheduled = float(row.get("scheduled_rounds", 0) or 0)
    wc_lbs = parse_weight_class_lbs(str(row.get("weight_class", "")))
    is_female = 1.0 if "women" in str(row.get("weight_class", "")).lower() else 0.0
    static = np.array(
        [
            math.log1p(fights_a),
            math.log1p(fights_b),
            math.log1p(fights_a) - math.log1p(fights_b),
            r3_win_a,
            r3_win_b,
            r3_win_a - r3_win_b,
            r5_win_a,
            r5_win_b,
            r5_win_a - r5_win_b,
            r3_fin_a,
            r3_fin_b,
            r3_fin_a - r3_fin_b,
            r3_sig_a,
            r3_sig_b,
            r3_sig_a - r3_sig_b,
            safe_div(scheduled, 5.0),
            float(row.get("is_title_bout", 0) or 0),
            float(row.get("is_main_event", 0) or 0),
            safe_div(wc_lbs, 265.0),
            is_female,
        ],
        dtype=np.float32,
    )
    return static


def build_sequence_samples(
    df: pd.DataFrame,
    *,
    seq_len: int,
    min_history: int,
    max_samples: Optional[int],
) -> list[SequenceSample]:
    history: dict[str, list[dict[str, Any]]] = {}
    last_event_date: dict[str, pd.Timestamp] = {}
    samples: list[SequenceSample] = []

    decisive = df[df["outcome_label"].isin([POSITIVE_LABEL, NEGATIVE_LABEL])].copy()
    decisive = decisive.sort_values(["event_date", "bout_index", "fight_id"]).reset_index(drop=True)

    for _, row in decisive.iterrows():
        fighter_a = row["fighter_1_id"]
        fighter_b = row["fighter_2_id"]
        hist_a = history.get(fighter_a, [])
        hist_b = history.get(fighter_b, [])
        event_date = pd.Timestamp(row["event_date"])

        if len(hist_a) >= min_history and len(hist_b) >= min_history:
            seq_a = np.asarray([h["vec"] for h in hist_a[-seq_len:]], dtype=np.float32)
            seq_b = np.asarray([h["vec"] for h in hist_b[-seq_len:]], dtype=np.float32)
            static_ab = build_static_features(hist_a, hist_b, row)
            target_ab = 1 if row["outcome_label"] == POSITIVE_LABEL else 0
            samples.append(
                SequenceSample(
                    sample_id=f"{row['fight_id']}_ab",
                    event_date=event_date,
                    seq_a=seq_a,
                    seq_b=seq_b,
                    static=static_ab,
                    target=target_ab,
                )
            )

            static_ba = build_static_features(hist_b, hist_a, row)
            samples.append(
                SequenceSample(
                    sample_id=f"{row['fight_id']}_ba",
                    event_date=event_date,
                    seq_a=seq_b,
                    seq_b=seq_a,
                    static=static_ba,
                    target=1 - target_ab,
                )
            )

            if max_samples is not None and len(samples) >= max_samples:
                break

        vec_a, stats_a = build_fighter_fight_vector(
            row, side=1, previous_event_date=last_event_date.get(fighter_a)
        )
        vec_b, stats_b = build_fighter_fight_vector(
            row, side=2, previous_event_date=last_event_date.get(fighter_b)
        )
        history.setdefault(fighter_a, []).append({"vec": vec_a, **stats_a})
        history.setdefault(fighter_b, []).append({"vec": vec_b, **stats_b})
        last_event_date[fighter_a] = event_date
        last_event_date[fighter_b] = event_date

    samples.sort(key=lambda s: (s.event_date, s.sample_id))
    return samples


def chronological_split(
    samples: list[SequenceSample], val_fraction: float, test_fraction: float
) -> tuple[list[SequenceSample], list[SequenceSample], list[SequenceSample]]:
    n = len(samples)
    if n < 200:
        raise ValueError(f"Not enough training samples after history filtering: {n}")
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val - n_test
    if n_train < 50:
        raise ValueError(
            f"Train split too small ({n_train} samples). Increase data or lower val/test fractions."
        )
    train = samples[:n_train]
    val = samples[n_train : n_train + n_val]
    test = samples[n_train + n_val :]
    return train, val, test


def fit_scalers(train_samples: list[SequenceSample]) -> tuple[RobustScaler, RobustScaler]:
    seq_rows: list[np.ndarray] = []
    static_rows: list[np.ndarray] = []
    for sample in train_samples:
        seq_rows.append(sample.seq_a)
        seq_rows.append(sample.seq_b)
        static_rows.append(sample.static)
    seq_matrix = np.vstack(seq_rows)
    static_matrix = np.vstack(static_rows)
    seq_scaler = RobustScaler(quantile_range=(10, 90))
    static_scaler = RobustScaler(quantile_range=(10, 90))
    seq_scaler.fit(seq_matrix)
    static_scaler.fit(static_matrix)
    return seq_scaler, static_scaler


def pad_sequence(seq: np.ndarray, seq_len: int, feat_dim: int) -> tuple[np.ndarray, int]:
    actual_len = int(min(len(seq), seq_len))
    if actual_len <= 0:
        out = np.zeros((seq_len, feat_dim), dtype=np.float32)
        return out, 1
    if len(seq) > seq_len:
        seq = seq[-seq_len:]
    out = np.zeros((seq_len, feat_dim), dtype=np.float32)
    out[-len(seq) :] = seq
    return out, actual_len


def transform_samples(
    samples: list[SequenceSample],
    *,
    seq_scaler: RobustScaler,
    static_scaler: RobustScaler,
    seq_len: int,
) -> list[TransformedSample]:
    transformed: list[TransformedSample] = []
    seq_dim = len(SEQ_FEATURE_NAMES)
    for sample in samples:
        seq_a_scaled = seq_scaler.transform(sample.seq_a)
        seq_b_scaled = seq_scaler.transform(sample.seq_b)
        static_scaled = static_scaler.transform(sample.static.reshape(1, -1)).astype(np.float32).reshape(-1)
        seq_a_pad, len_a = pad_sequence(seq_a_scaled.astype(np.float32), seq_len, seq_dim)
        seq_b_pad, len_b = pad_sequence(seq_b_scaled.astype(np.float32), seq_len, seq_dim)
        transformed.append(
            TransformedSample(
                sample_id=sample.sample_id,
                event_date=sample.event_date,
                seq_a=seq_a_pad,
                len_a=len_a,
                seq_b=seq_b_pad,
                len_b=len_b,
                static=static_scaled,
                target=sample.target,
            )
        )
    return transformed


class FightSequenceDataset(Dataset):
    def __init__(self, samples: list[TransformedSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        sample = self.samples[idx]
        return (
            torch.tensor(sample.seq_a, dtype=torch.float32),
            torch.tensor(sample.len_a, dtype=torch.int64),
            torch.tensor(sample.seq_b, dtype=torch.float32),
            torch.tensor(sample.len_b, dtype=torch.int64),
            torch.tensor(sample.static, dtype=torch.float32),
            torch.tensor(sample.target, dtype=torch.float32),
        )


class SiameseFightLSTM(nn.Module):
    def __init__(
        self,
        *,
        seq_dim: int,
        static_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_dim = seq_dim
        self.static_dim = static_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoder = nn.LSTM(
            input_size=seq_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        rep_dim = hidden_size * 4  # mean + max pooling of bidirectional output
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        head_in = rep_dim * 4 + 32
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def encode(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            seq,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=seq.size(1),
        )
        idx = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        mask = idx < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()
        denom = lengths.unsqueeze(1).clamp(min=1).float()
        mean_pool = (out * mask_f).sum(dim=1) / denom
        masked = out.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_pool = masked.max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=1)

    def forward(
        self,
        seq_a: torch.Tensor,
        len_a: torch.Tensor,
        seq_b: torch.Tensor,
        len_b: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        rep_a = self.encode(seq_a, len_a)
        rep_b = self.encode(seq_b, len_b)
        static_rep = self.static_net(static)
        x = torch.cat([rep_a, rep_b, rep_a - rep_b, rep_a * rep_b, static_rep], dim=1)
        logits = self.head(x).squeeze(1)
        return logits


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    losses: list[float] = []
    probs_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []

    for seq_a, len_a, seq_b, len_b, static, target in loader:
        seq_a = seq_a.to(device)
        len_a = len_a.to(device)
        seq_b = seq_b.to(device)
        len_b = len_b.to(device)
        static = static.to(device)
        target = target.to(device)

        logits = model(seq_a, len_a, seq_b, len_b, static)
        loss = criterion(logits, target)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        losses.append(float(loss.item()))
        probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
        targets_all.append(target.detach().cpu().numpy())

    probs = np.concatenate(probs_all) if probs_all else np.zeros(0, dtype=np.float32)
    targets = np.concatenate(targets_all) if targets_all else np.zeros(0, dtype=np.float32)
    return float(np.mean(losses) if losses else 0.0), probs, targets


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auc": safe_auc(y_true, y_prob),
    }
    clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    metrics["log_loss"] = float(log_loss(y_true, clipped, labels=[0, 1]))
    return metrics


def choose_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_score = -1.0
    best_metrics: dict[str, float] = evaluate_probs(y_true, y_prob, 0.5)
    for threshold in np.linspace(0.30, 0.70, 81):
        metrics = evaluate_probs(y_true, y_prob, float(threshold))
        score = metrics["balanced_accuracy"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def class_pos_weight(samples: list[TransformedSample]) -> float:
    targets = np.asarray([s.target for s in samples], dtype=np.float32)
    pos = float((targets == 1).sum())
    neg = float((targets == 0).sum())
    if pos <= 0 or neg <= 0:
        return 1.0
    return max(neg / pos, 1e-6)


def train_model(
    *,
    train_samples: list[TransformedSample],
    val_samples: list[TransformedSample],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, float, dict[str, Any]]:
    model = SiameseFightLSTM(
        seq_dim=len(SEQ_FEATURE_NAMES),
        static_dim=len(STATIC_FEATURE_NAMES),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    pos_weight = class_pos_weight(train_samples)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    train_loader = DataLoader(
        FightSequenceDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        FightSequenceDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    best_state: Optional[dict[str, Any]] = None
    best_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(
            model,
            train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_loss, val_prob, val_true = run_epoch(
            model,
            val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_auc = safe_auc(val_true, val_prob)
        scheduler.step(val_loss)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        logging.info(
            "Epoch %03d | train_loss=%.4f | val_loss=%.4f | val_auc=%.4f | lr=%.6f",
            epoch,
            train_loss,
            val_loss,
            val_auc,
            float(optimizer.param_groups[0]["lr"]),
        )

        improved = val_auc > best_auc
        if math.isnan(val_auc):
            improved = False
        if improved:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logging.info("Early stopping triggered at epoch %d", epoch)
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(history)
        best_auc = float("nan")
    model.load_state_dict(best_state)

    val_loader_eval = DataLoader(
        FightSequenceDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    _, val_prob_final, val_true_final = run_epoch(
        model,
        val_loader_eval,
        criterion=criterion,
        optimizer=None,
        device=device,
        grad_clip=args.grad_clip,
    )
    threshold, val_threshold_metrics = choose_best_threshold(val_true_final, val_prob_final)
    train_summary = {
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_auc),
        "pos_weight": float(pos_weight),
        "threshold": float(threshold),
        "val_metrics_at_best_threshold": val_threshold_metrics,
        "history": history,
    }
    return model, threshold, train_summary


def save_artifacts(
    *,
    artifacts_dir: Path,
    model: nn.Module,
    threshold: float,
    train_summary: dict[str, Any],
    test_metrics: dict[str, float],
    seq_scaler: RobustScaler,
    static_scaler: RobustScaler,
    args: argparse.Namespace,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "lstm_fight_model.pt"
    scalers_path = artifacts_dir / "lstm_fight_scalers.joblib"
    metrics_path = artifacts_dir / "lstm_fight_metrics.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "threshold": float(threshold),
            "seq_feature_names": SEQ_FEATURE_NAMES,
            "static_feature_names": STATIC_FEATURE_NAMES,
            "config": {
                "seq_len": int(args.seq_len),
                "hidden_size": int(args.hidden_size),
                "num_layers": int(args.num_layers),
                "dropout": float(args.dropout),
                "seq_dim": int(len(SEQ_FEATURE_NAMES)),
                "static_dim": int(len(STATIC_FEATURE_NAMES)),
            },
        },
        model_path,
    )
    joblib.dump(
        {
            "seq_scaler": seq_scaler,
            "static_scaler": static_scaler,
            "seq_feature_names": SEQ_FEATURE_NAMES,
            "static_feature_names": STATIC_FEATURE_NAMES,
            "seq_len": int(args.seq_len),
        },
        scalers_path,
    )

    report = {
        "input_csv": str(args.input_csv),
        "artifacts_dir": str(artifacts_dir),
        "threshold": float(threshold),
        "train_summary": train_summary,
        "test_metrics": test_metrics,
        "model_path": str(model_path),
        "scalers_path": str(scalers_path),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logging.info("Saved model to %s", model_path)
    logging.info("Saved scalers to %s", scalers_path)
    logging.info("Saved metrics to %s", metrics_path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    set_seed(args.seed)
    device = resolve_device(args.device)
    logging.info("Using device: %s", device)

    df = load_raw_dataframe(args.input_csv)
    validate_raw_data(df)
    logging.info("Loaded %d fights from %s", len(df), args.input_csv)

    samples = build_sequence_samples(
        df,
        seq_len=args.seq_len,
        min_history=args.min_history,
        max_samples=args.max_samples,
    )
    logging.info("Generated %d sequence samples", len(samples))
    if len(samples) < 200:
        raise ValueError(
            f"Insufficient sequence samples ({len(samples)}). "
            "Collect more fights or reduce --min-history."
        )

    train_raw, val_raw, test_raw = chronological_split(
        samples,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    logging.info(
        "Chronological split sizes -> train: %d | val: %d | test: %d",
        len(train_raw),
        len(val_raw),
        len(test_raw),
    )

    seq_scaler, static_scaler = fit_scalers(train_raw)
    train_data = transform_samples(
        train_raw, seq_scaler=seq_scaler, static_scaler=static_scaler, seq_len=args.seq_len
    )
    val_data = transform_samples(
        val_raw, seq_scaler=seq_scaler, static_scaler=static_scaler, seq_len=args.seq_len
    )
    test_data = transform_samples(
        test_raw, seq_scaler=seq_scaler, static_scaler=static_scaler, seq_len=args.seq_len
    )

    model, threshold, train_summary = train_model(
        train_samples=train_data,
        val_samples=val_data,
        args=args,
        device=device,
    )

    test_loader = DataLoader(
        FightSequenceDataset(test_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    criterion = nn.BCEWithLogitsLoss()
    _, test_prob, test_true = run_epoch(
        model,
        test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        grad_clip=args.grad_clip,
    )
    test_metrics = evaluate_probs(test_true, test_prob, threshold=threshold)
    logging.info("Test metrics @ threshold %.3f: %s", threshold, test_metrics)

    save_artifacts(
        artifacts_dir=args.artifacts_dir,
        model=model,
        threshold=threshold,
        train_summary=train_summary,
        test_metrics=test_metrics,
        seq_scaler=seq_scaler,
        static_scaler=static_scaler,
        args=args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
