#!/usr/bin/env python3
"""Train a high-signal Siamese LSTM from ufc_lstm_sequences.csv."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset


POSITIVE_LABEL = "fighter_1_win"
NEGATIVE_LABEL = "fighter_2_win"


@dataclass
class SequenceSample:
    fight_id: str
    event_date: pd.Timestamp
    seq_a: np.ndarray
    len_a: int
    seq_b: np.ndarray
    len_b: int
    static: np.ndarray
    target: int


@dataclass
class TransformedSample:
    fight_id: str
    event_date: pd.Timestamp
    seq_a: np.ndarray
    len_a: int
    seq_b: np.ndarray
    len_b: int
    static: np.ndarray
    target: int


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train an improved Siamese LSTM on prebuilt UFC sequence rows."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=root_dir / "data" / "ufc_lstm_sequences.csv",
        help="Path to sequence CSV created by build_fight_history_sequences.py",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=root_dir / "champion_lstm_model.pth",
        help="Path to save model checkpoint.",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=root_dir / "data" / "model_cache" / "lstm_sequence_scalers.pkl",
        help="Path to save fitted scalers.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=root_dir / "data" / "model_cache" / "lstm_sequence_metrics.json",
        help="Path to save train/eval metrics report.",
    )
    parser.add_argument("--epochs", type=int, default=120, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stop patience.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--hidden-size", type=int, default=96, help="LSTM hidden size.")
    parser.add_argument("--num-layers", type=int, default=1, help="LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.35, help="Dropout rate.")
    parser.add_argument("--lr", type=float, default=5e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument(
        "--bidirectional",
        dest="bidirectional",
        action="store_true",
        help="Use bidirectional LSTM encoder (recommended default).",
    )
    parser.add_argument(
        "--no-bidirectional",
        dest="bidirectional",
        action="store_false",
        help="Use unidirectional LSTM encoder for strict temporal directionality.",
    )
    parser.set_defaults(bidirectional=True)
    parser.add_argument(
        "--use-cross-attention",
        dest="use_cross_attention",
        action="store_true",
        help="Enable cross-attention between fighter timelines after LSTM encoding (default).",
    )
    parser.add_argument(
        "--no-cross-attention",
        dest="use_cross_attention",
        action="store_false",
        help="Disable cross-attention between fighter timelines.",
    )
    parser.set_defaults(use_cross_attention=True)
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=4,
        help="Number of attention heads when cross-attention is enabled.",
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        default=0.10,
        help="Attention dropout when cross-attention is enabled.",
    )
    parser.add_argument(
        "--static-recency-mode",
        type=str,
        default="ema",
        choices=["ema", "mean"],
        help="Recency summarization strategy for static features.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.65,
        help="EMA alpha for static recency features (used when mode=ema).",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.15, help="Validation split from fights."
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.15, help="Test split from fights."
    )
    parser.add_argument(
        "--max-fights",
        type=int,
        default=None,
        help="Optional cap for number of chronological fights (debug/smoke).",
    )
    parser.add_argument(
        "--drop-empty-history",
        dest="drop_empty_history",
        action="store_true",
        help="Drop fights where both fighter histories are fully empty at sequence time.",
    )
    parser.add_argument(
        "--keep-empty-history",
        dest="drop_empty_history",
        action="store_false",
        help="Keep fights with fully empty histories (mostly debut/debut matchups).",
    )
    parser.set_defaults(drop_empty_history=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 is safest).",
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


def safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return num / np.maximum(den, eps)


def detect_sequence_layout(columns: list[str]) -> tuple[int, int, list[str], list[str]]:
    pat_f1 = re.compile(r"^f1_seq_(\d+)_stat_(\d+)$")
    pat_f2 = re.compile(r"^f2_seq_(\d+)_stat_(\d+)$")
    f1_steps: set[int] = set()
    f2_steps: set[int] = set()
    f1_stats: set[int] = set()
    f2_stats: set[int] = set()

    for col in columns:
        m1 = pat_f1.match(col)
        if m1:
            f1_steps.add(int(m1.group(1)))
            f1_stats.add(int(m1.group(2)))
            continue
        m2 = pat_f2.match(col)
        if m2:
            f2_steps.add(int(m2.group(1)))
            f2_stats.add(int(m2.group(2)))

    if not f1_steps or not f1_stats:
        raise ValueError("No f1 sequence columns detected (f1_seq_{step}_stat_{idx}).")
    if f1_steps != f2_steps or f1_stats != f2_stats:
        raise ValueError("f1/f2 sequence column layout mismatch.")

    steps = sorted(f1_steps)
    stats = sorted(f1_stats)
    seq_len = len(steps)
    num_stats = len(stats)
    f1_cols = [f"f1_seq_{s}_stat_{i}" for s in steps for i in stats]
    f2_cols = [f"f2_seq_{s}_stat_{i}" for s in steps for i in stats]
    return seq_len, num_stats, f1_cols, f2_cols


def load_dataframe(
    path: Path,
    max_fights: Optional[int],
    drop_empty_history: bool,
) -> tuple[pd.DataFrame, int, int, list[str], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    required = ["fight_id", "event_date", "outcome_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    seq_len, num_stats, f1_cols, f2_cols = detect_sequence_layout(df.columns.tolist())
    numeric_cols = f1_cols + f2_cols
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"]).copy()
    df["fight_id"] = df["fight_id"].astype(str).str.strip()
    df["outcome_label"] = df["outcome_label"].astype(str).str.strip()
    df = df[df["outcome_label"].isin([POSITIVE_LABEL, NEGATIVE_LABEL])].copy()
    if df.empty:
        raise ValueError("No decisive rows found in input sequence CSV.")
    df = df.sort_values(["event_date", "fight_id"]).drop_duplicates(subset=["fight_id"]).reset_index(
        drop=True
    )
    zero_hist_mask = (df[f1_cols + f2_cols].abs().sum(axis=1) == 0)
    zero_hist_rate_before = float(zero_hist_mask.mean())
    if drop_empty_history:
        dropped = int(zero_hist_mask.sum())
        if dropped > 0:
            df = df.loc[~zero_hist_mask].copy().reset_index(drop=True)
            logging.info(
                "Dropped %d fully empty-history fights (%.2f%% of decisive rows).",
                dropped,
                zero_hist_rate_before * 100.0,
            )

    if max_fights is not None:
        df = df.head(max_fights).copy()
    if len(df) < 300:
        raise ValueError(f"Not enough fights after filtering: {len(df)} (need >= 300).")

    label_counts = df["outcome_label"].value_counts().to_dict()
    zero_hist_mask_after = (df[f1_cols + f2_cols].abs().sum(axis=1) == 0)
    zero_hist_rate = float(zero_hist_mask_after.mean())
    logging.info("Decisive label counts: %s", label_counts)
    if len(label_counts) == 1:
        only_label = next(iter(label_counts))
        logging.warning(
            "Only one decisive orientation label present (%s). AB/BA augmentation is required to balance classes.",
            only_label,
        )
    logging.info("Rows with fully empty sequence history: %.2f%%", zero_hist_rate * 100.0)
    if zero_hist_rate > 0.25:
        logging.warning(
            "High empty-history rate detected (%.2f%%). Dataset may be too sparse.",
            zero_hist_rate * 100.0,
        )

    return df, seq_len, num_stats, f1_cols, f2_cols


def compute_lengths(raw_seq: np.ndarray) -> np.ndarray:
    mask = np.abs(raw_seq).sum(axis=2) > 0
    return mask.sum(axis=1).astype(np.int64)


def extract_lengths(
    df: pd.DataFrame,
    raw_seq: np.ndarray,
    length_column: str,
) -> np.ndarray:
    if length_column in df.columns:
        lengths = pd.to_numeric(df[length_column], errors="coerce").fillna(0).astype(np.int64).to_numpy()
        return np.clip(lengths, 0, raw_seq.shape[1])
    return compute_lengths(raw_seq)


def engineer_features(raw_seq: np.ndarray) -> np.ndarray:
    if raw_seq.shape[2] < 12:
        raise ValueError(
            f"Expected at least 12 base stats per step, but found {raw_seq.shape[2]}."
        )
    base = raw_seq[:, :, :12]
    kd_for = base[:, :, 0]
    kd_against = base[:, :, 1]
    sig_for_l = base[:, :, 2]
    sig_for_a = base[:, :, 3]
    sig_against_l = base[:, :, 4]
    sig_against_a = base[:, :, 5]
    td_for_l = base[:, :, 6]
    td_for_a = base[:, :, 7]
    td_against_l = base[:, :, 8]
    td_against_a = base[:, :, 9]
    sub_for = base[:, :, 10]
    ctrl_for = base[:, :, 11]

    sig_acc = safe_div(sig_for_l, sig_for_a)
    sig_def = np.clip(safe_div(np.maximum(sig_against_a - sig_against_l, 0.0), sig_against_a), 0.0, 1.0)
    td_acc = safe_div(td_for_l, td_for_a)
    td_def = np.clip(safe_div(np.maximum(td_against_a - td_against_l, 0.0), td_against_a), 0.0, 1.0)
    activity = np.log1p(sig_for_a + td_for_a + (2.0 * sub_for) + (ctrl_for / 30.0))
    grappling = np.log1p(td_for_a + (2.0 * sub_for) + (ctrl_for / 60.0))
    kd_swing = kd_for - kd_against
    sig_diff_ratio = safe_div(sig_for_l - sig_against_l, sig_for_a + sig_against_a)
    td_diff_ratio = safe_div(td_for_l - td_against_l, td_for_a + td_against_a)

    base_log = np.log1p(np.clip(base, 0.0, None))
    derived = np.stack(
        [
            sig_acc,
            sig_def,
            td_acc,
            td_def,
            activity,
            grappling,
            kd_swing,
            sig_diff_ratio,
            td_diff_ratio,
        ],
        axis=2,
    ).astype(np.float32)
    engineered = np.concatenate([base_log.astype(np.float32), derived], axis=2)

    if raw_seq.shape[2] > 12:
        extra = np.log1p(np.clip(raw_seq[:, :, 12:], 0.0, None)).astype(np.float32)
        engineered = np.concatenate([engineered, extra], axis=2)

    return engineered.astype(np.float32)


def recent_mean(seq: np.ndarray, length: int, idx: int, window: int = 3) -> float:
    if length <= 0:
        return 0.0
    valid = seq[-length:]
    values = valid[-window:, idx]
    return float(np.mean(values)) if len(values) else 0.0


def recent_ema(seq: np.ndarray, length: int, idx: int, alpha: float) -> float:
    if length <= 0:
        return 0.0
    valid = seq[-length:, idx].astype(np.float64)
    if len(valid) == 0:
        return 0.0
    if alpha <= 0:
        return float(np.mean(valid))
    if alpha >= 1:
        return float(valid[-1])
    # Exponentially down-weight older fights so most recent performances dominate.
    powers = np.arange(len(valid) - 1, -1, -1, dtype=np.float64)
    weights = np.power(1.0 - alpha, powers)
    weights /= np.maximum(weights.sum(), 1e-12)
    return float(np.dot(valid, weights))


def summarize_recent(
    seq: np.ndarray,
    length: int,
    idx: int,
    mode: str,
    ema_alpha: float,
) -> float:
    if mode == "mean":
        return recent_mean(seq, length, idx)
    return recent_ema(seq, length, idx, alpha=ema_alpha)


def build_static_features(
    seq_a: np.ndarray,
    len_a: int,
    seq_b: np.ndarray,
    len_b: int,
    *,
    recency_mode: str,
    ema_alpha: float,
) -> np.ndarray:
    # Engineered feature positions after base 12 log stats.
    sig_acc_idx = 12
    td_acc_idx = 14
    activity_idx = 16
    kd_swing_idx = 18
    ctrl_idx = 11
    seq_len = max(seq_a.shape[0], 1)

    len_a_norm = float(len_a) / float(seq_len)
    len_b_norm = float(len_b) / float(seq_len)
    sig_a = summarize_recent(seq_a, len_a, sig_acc_idx, recency_mode, ema_alpha)
    sig_b = summarize_recent(seq_b, len_b, sig_acc_idx, recency_mode, ema_alpha)
    td_a = summarize_recent(seq_a, len_a, td_acc_idx, recency_mode, ema_alpha)
    td_b = summarize_recent(seq_b, len_b, td_acc_idx, recency_mode, ema_alpha)
    act_a = summarize_recent(seq_a, len_a, activity_idx, recency_mode, ema_alpha)
    act_b = summarize_recent(seq_b, len_b, activity_idx, recency_mode, ema_alpha)
    kd_a = summarize_recent(seq_a, len_a, kd_swing_idx, recency_mode, ema_alpha)
    kd_b = summarize_recent(seq_b, len_b, kd_swing_idx, recency_mode, ema_alpha)
    ctrl_a = summarize_recent(seq_a, len_a, ctrl_idx, recency_mode, ema_alpha)
    ctrl_b = summarize_recent(seq_b, len_b, ctrl_idx, recency_mode, ema_alpha)

    return np.array(
        [
            len_a_norm,
            len_b_norm,
            len_a_norm - len_b_norm,
            sig_a,
            sig_b,
            sig_a - sig_b,
            td_a,
            td_b,
            td_a - td_b,
            act_a,
            act_b,
            act_a - act_b,
            kd_a,
            kd_b,
            kd_a - kd_b,
            ctrl_a,
            ctrl_b,
            ctrl_a - ctrl_b,
        ],
        dtype=np.float32,
    )


def chronological_split(
    df: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val - n_test
    if n_train < 200:
        raise ValueError(f"Train split too small ({n_train} fights).")
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


def frame_to_raw_sequences(
    df: pd.DataFrame,
    seq_len: int,
    num_stats: int,
    f1_cols: list[str],
    f2_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    f1 = df[f1_cols].to_numpy(dtype=np.float32).reshape(len(df), seq_len, num_stats)
    f2 = df[f2_cols].to_numpy(dtype=np.float32).reshape(len(df), seq_len, num_stats)
    return f1, f2


def build_augmented_samples(
    df: pd.DataFrame,
    f1_raw: np.ndarray,
    f2_raw: np.ndarray,
    *,
    static_recency_mode: str,
    ema_alpha: float,
) -> list[SequenceSample]:
    f1_len = extract_lengths(df, f1_raw, "f1_history_len")
    f2_len = extract_lengths(df, f2_raw, "f2_history_len")
    f1_eng = engineer_features(f1_raw)
    f2_eng = engineer_features(f2_raw)
    labels = (df["outcome_label"].to_numpy() == POSITIVE_LABEL).astype(np.int64)

    samples: list[SequenceSample] = []
    for i in range(len(df)):
        fight_id = str(df.iloc[i]["fight_id"])
        event_date = pd.Timestamp(df.iloc[i]["event_date"])
        target = int(labels[i])

        static_ab = build_static_features(
            f1_eng[i],
            int(f1_len[i]),
            f2_eng[i],
            int(f2_len[i]),
            recency_mode=static_recency_mode,
            ema_alpha=ema_alpha,
        )
        samples.append(
            SequenceSample(
                fight_id=f"{fight_id}_ab",
                event_date=event_date,
                seq_a=f1_eng[i],
                len_a=int(f1_len[i]),
                seq_b=f2_eng[i],
                len_b=int(f2_len[i]),
                static=static_ab,
                target=target,
            )
        )

        static_ba = build_static_features(
            f2_eng[i],
            int(f2_len[i]),
            f1_eng[i],
            int(f1_len[i]),
            recency_mode=static_recency_mode,
            ema_alpha=ema_alpha,
        )
        samples.append(
            SequenceSample(
                fight_id=f"{fight_id}_ba",
                event_date=event_date,
                seq_a=f2_eng[i],
                len_a=int(f2_len[i]),
                seq_b=f1_eng[i],
                len_b=int(f1_len[i]),
                static=static_ba,
                target=1 - target,
            )
        )

    return samples


def fit_scalers(train_samples: list[SequenceSample]) -> tuple[RobustScaler, RobustScaler]:
    def tail_or_zero(seq: np.ndarray, length: int) -> np.ndarray:
        if length <= 0:
            return np.zeros((1, seq.shape[1]), dtype=np.float32)
        return seq[-length:]

    seq_rows: list[np.ndarray] = []
    static_rows: list[np.ndarray] = []
    for sample in train_samples:
        seq_rows.append(tail_or_zero(sample.seq_a, sample.len_a))
        seq_rows.append(tail_or_zero(sample.seq_b, sample.len_b))
        static_rows.append(sample.static)
    seq_matrix = np.vstack(seq_rows)
    static_matrix = np.vstack(static_rows)
    seq_scaler = RobustScaler(quantile_range=(10, 90))
    static_scaler = RobustScaler(quantile_range=(10, 90))
    seq_scaler.fit(seq_matrix)
    static_scaler.fit(static_matrix)
    return seq_scaler, static_scaler


def pad_sequence(valid_seq: np.ndarray, seq_len: int) -> tuple[np.ndarray, int]:
    actual_len = min(len(valid_seq), seq_len)
    if actual_len <= 0:
        return np.zeros((seq_len, valid_seq.shape[1]), dtype=np.float32), 1
    if len(valid_seq) > seq_len:
        valid_seq = valid_seq[-seq_len:]
        actual_len = seq_len
    out = np.zeros((seq_len, valid_seq.shape[1]), dtype=np.float32)
    out[: len(valid_seq)] = valid_seq
    return out, actual_len


def transform_samples(
    samples: list[SequenceSample],
    seq_scaler: RobustScaler,
    static_scaler: RobustScaler,
    seq_len: int,
) -> list[TransformedSample]:
    transformed: list[TransformedSample] = []
    for sample in samples:
        seq_a_scaled = seq_scaler.transform(sample.seq_a).astype(np.float32)
        seq_b_scaled = seq_scaler.transform(sample.seq_b).astype(np.float32)
        static_scaled = (
            static_scaler.transform(sample.static.reshape(1, -1)).astype(np.float32).reshape(-1)
        )
        seq_a_valid = (
            seq_a_scaled[-sample.len_a :]
            if sample.len_a > 0
            else np.zeros((0, seq_a_scaled.shape[1]), dtype=np.float32)
        )
        seq_b_valid = (
            seq_b_scaled[-sample.len_b :]
            if sample.len_b > 0
            else np.zeros((0, seq_b_scaled.shape[1]), dtype=np.float32)
        )
        seq_a_pad, len_a = pad_sequence(seq_a_valid, seq_len)
        seq_b_pad, len_b = pad_sequence(seq_b_valid, seq_len)
        transformed.append(
            TransformedSample(
                fight_id=sample.fight_id,
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


class FightDataset(Dataset):
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
        seq_dim: int,
        static_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        use_cross_attention: bool,
        attention_heads: int,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.seq_embed_dim = hidden_size * (2 if bidirectional else 1)
        self.use_cross_attention = bool(use_cross_attention)
        self.encoder = nn.LSTM(
            input_size=seq_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        if self.use_cross_attention:
            if attention_heads <= 0:
                raise ValueError("attention_heads must be >= 1 when cross-attention is enabled.")
            if self.seq_embed_dim % attention_heads != 0:
                raise ValueError(
                    f"seq embedding dim ({self.seq_embed_dim}) must be divisible by attention_heads ({attention_heads})."
                )
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.seq_embed_dim,
                num_heads=attention_heads,
                dropout=attention_dropout,
                batch_first=True,
            )
            self.cross_norm_a = nn.LayerNorm(self.seq_embed_dim)
            self.cross_norm_b = nn.LayerNorm(self.seq_embed_dim)
        else:
            self.cross_attention = None
            self.cross_norm_a = None
            self.cross_norm_b = None

        rep_dim = self.seq_embed_dim * 2
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 24),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(rep_dim * 4 + 24, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def encode(self, seq: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        return out, mask

    def apply_cross_attention(
        self,
        out_a: torch.Tensor,
        mask_a: torch.Tensor,
        out_b: torch.Tensor,
        mask_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cross_attention is None:
            return out_a, out_b
        attn_a, _ = self.cross_attention(
            query=out_a,
            key=out_b,
            value=out_b,
            key_padding_mask=~mask_b,
            need_weights=False,
        )
        attn_b, _ = self.cross_attention(
            query=out_b,
            key=out_a,
            value=out_a,
            key_padding_mask=~mask_a,
            need_weights=False,
        )
        out_a = self.cross_norm_a(out_a + attn_a)
        out_b = self.cross_norm_b(out_b + attn_b)
        return out_a, out_b

    @staticmethod
    def masked_pool(out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        mean_pool = (out * mask_f).sum(dim=1) / denom
        max_pool = out.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=1)

    def forward(
        self,
        seq_a: torch.Tensor,
        len_a: torch.Tensor,
        seq_b: torch.Tensor,
        len_b: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        out_a, mask_a = self.encode(seq_a, len_a)
        out_b, mask_b = self.encode(seq_b, len_b)
        out_a, out_b = self.apply_cross_attention(out_a, mask_a, out_b, mask_b)
        rep_a = self.masked_pool(out_a, mask_a)
        rep_b = self.masked_pool(out_b, mask_b)
        static_rep = self.static_net(static)
        x = torch.cat([rep_a, rep_b, rep_a - rep_b, rep_a * rep_b, static_rep], dim=1)
        return self.head(x).squeeze(1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
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
    clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auc": safe_auc(y_true, y_prob),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
    }


def choose_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    best_t = 0.5
    best_score = -1.0
    best_metrics = evaluate_probs(y_true, y_prob, 0.5)
    for threshold in np.linspace(0.30, 0.70, 81):
        metrics = evaluate_probs(y_true, y_prob, float(threshold))
        if metrics["balanced_accuracy"] > best_score:
            best_score = metrics["balanced_accuracy"]
            best_t = float(threshold)
            best_metrics = metrics
    return best_t, best_metrics


def class_pos_weight(samples: list[TransformedSample]) -> float:
    targets = np.asarray([s.target for s in samples], dtype=np.float32)
    pos = float((targets == 1).sum())
    neg = float((targets == 0).sum())
    if pos <= 0 or neg <= 0:
        return 1.0
    return max(neg / pos, 1e-6)


def train_model(
    train_samples: list[TransformedSample],
    val_samples: list[TransformedSample],
    seq_dim: int,
    static_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, float, dict[str, Any]]:
    model = SiameseFightLSTM(
        seq_dim=seq_dim,
        static_dim=static_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        use_cross_attention=args.use_cross_attention,
        attention_heads=args.attention_heads,
        attention_dropout=args.attention_dropout,
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
        patience=4,
        min_lr=1e-6,
    )

    train_loader = DataLoader(
        FightDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        FightDataset(val_samples),
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
        scheduler.step(val_loss)
        val_auc = safe_auc(val_true, val_prob)
        val_bal = evaluate_probs(val_true, val_prob, threshold=0.5)["balanced_accuracy"]
        score = val_auc if not math.isnan(val_auc) else val_bal

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_balanced_accuracy": val_bal,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        logging.info(
            "Epoch %03d | train_loss=%.4f | val_loss=%.4f | val_auc=%.4f | val_bal=%.4f | lr=%.6f",
            epoch,
            train_loss,
            val_loss,
            val_auc,
            val_bal,
            float(optimizer.param_groups[0]["lr"]),
        )

        if score > best_auc:
            best_auc = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logging.info("Early stopping at epoch %d", epoch)
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(history)
        best_auc = float("nan")
    model.load_state_dict(best_state)

    val_loader_eval = DataLoader(
        FightDataset(val_samples),
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
    summary = {
        "best_epoch": int(best_epoch),
        "best_score": float(best_auc),
        "pos_weight": float(pos_weight),
        "threshold": float(threshold),
        "val_metrics_at_best_threshold": val_threshold_metrics,
        "history": history,
    }
    return model, threshold, summary


def save_artifacts(
    model: nn.Module,
    seq_scaler: RobustScaler,
    static_scaler: RobustScaler,
    metrics_report: dict[str, Any],
    model_path: Path,
    scaler_path: Path,
    metrics_path: Path,
    config: dict[str, Any],
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
        },
        model_path,
    )
    with scaler_path.open("wb") as f:
        pickle.dump(
            {
                "seq_scaler": seq_scaler,
                "static_scaler": static_scaler,
                "config": config,
            },
            f,
        )
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    logging.info("Saved model checkpoint: %s", model_path)
    logging.info("Saved scalers: %s", scaler_path)
    logging.info("Saved metrics: %s", metrics_path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    set_seed(args.seed)
    device = resolve_device(args.device)
    logging.info("Using device: %s", device)
    logging.info(
        "Options | bidirectional=%s | cross_attention=%s | recency_mode=%s | ema_alpha=%.3f",
        args.bidirectional,
        args.use_cross_attention,
        args.static_recency_mode,
        args.ema_alpha,
    )

    df, seq_len, num_stats, f1_cols, f2_cols = load_dataframe(
        args.input_csv,
        args.max_fights,
        drop_empty_history=args.drop_empty_history,
    )
    logging.info(
        "Loaded %d fights | seq_len=%d | raw_stats_per_step=%d",
        len(df),
        seq_len,
        num_stats,
    )

    train_df, val_df, test_df = chronological_split(df, args.val_fraction, args.test_fraction)
    logging.info(
        "Fight splits -> train: %d | val: %d | test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    train_f1, train_f2 = frame_to_raw_sequences(train_df, seq_len, num_stats, f1_cols, f2_cols)
    val_f1, val_f2 = frame_to_raw_sequences(val_df, seq_len, num_stats, f1_cols, f2_cols)
    test_f1, test_f2 = frame_to_raw_sequences(test_df, seq_len, num_stats, f1_cols, f2_cols)

    train_samples = build_augmented_samples(
        train_df,
        train_f1,
        train_f2,
        static_recency_mode=args.static_recency_mode,
        ema_alpha=args.ema_alpha,
    )
    val_samples = build_augmented_samples(
        val_df,
        val_f1,
        val_f2,
        static_recency_mode=args.static_recency_mode,
        ema_alpha=args.ema_alpha,
    )
    test_samples = build_augmented_samples(
        test_df,
        test_f1,
        test_f2,
        static_recency_mode=args.static_recency_mode,
        ema_alpha=args.ema_alpha,
    )
    logging.info(
        "Augmented samples -> train: %d | val: %d | test: %d",
        len(train_samples),
        len(val_samples),
        len(test_samples),
    )

    seq_scaler, static_scaler = fit_scalers(train_samples)
    train_data = transform_samples(train_samples, seq_scaler, static_scaler, seq_len)
    val_data = transform_samples(val_samples, seq_scaler, static_scaler, seq_len)
    test_data = transform_samples(test_samples, seq_scaler, static_scaler, seq_len)

    seq_dim = train_data[0].seq_a.shape[1]
    static_dim = train_data[0].static.shape[0]
    model, threshold, train_summary = train_model(
        train_samples=train_data,
        val_samples=val_data,
        seq_dim=seq_dim,
        static_dim=static_dim,
        args=args,
        device=device,
    )

    test_loader = DataLoader(
        FightDataset(test_data),
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
    test_metrics = evaluate_probs(test_true, test_prob, threshold)
    logging.info("Test metrics @ threshold %.3f: %s", threshold, test_metrics)

    config = {
        "seq_len": seq_len,
        "raw_num_stats": num_stats,
        "seq_dim": seq_dim,
        "static_dim": static_dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "bidirectional": bool(args.bidirectional),
        "use_cross_attention": bool(args.use_cross_attention),
        "attention_heads": int(args.attention_heads),
        "attention_dropout": float(args.attention_dropout),
        "static_recency_mode": str(args.static_recency_mode),
        "ema_alpha": float(args.ema_alpha),
        "dropout": args.dropout,
        "threshold": float(threshold),
        "drop_empty_history": bool(args.drop_empty_history),
    }
    metrics_report = {
        "input_csv": str(args.input_csv),
        "splits": {
            "train_fights": int(len(train_df)),
            "val_fights": int(len(val_df)),
            "test_fights": int(len(test_df)),
            "train_samples": int(len(train_data)),
            "val_samples": int(len(val_data)),
            "test_samples": int(len(test_data)),
        },
        "train_summary": train_summary,
        "test_metrics": test_metrics,
        "config": config,
        "model_path": str(args.model_path),
        "scaler_path": str(args.scaler_path),
    }
    save_artifacts(
        model=model,
        seq_scaler=seq_scaler,
        static_scaler=static_scaler,
        metrics_report=metrics_report,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        metrics_path=args.metrics_path,
        config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
