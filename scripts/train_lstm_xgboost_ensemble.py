#!/usr/bin/env python3
"""Train an LSTM momentum model and an XGBoost static ensemble head.

By default, XGBoost is trained on chronological out-of-fold momentum scores to avoid
in-sample leakage in the stacking stage.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from train_lstm_from_sequences import (
    FightDataset,
    POSITIVE_LABEL,
    build_augmented_samples,
    chronological_split,
    class_pos_weight,
    fit_scalers,
    frame_to_raw_sequences,
    load_dataframe,
    resolve_device,
    set_seed,
    transform_samples,
)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train momentum-only Siamese LSTM + XGBoost static ensemble."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=root_dir / "data" / "ufc_lstm_sequences.csv",
        help="Path to sequence CSV created by build_fight_history_sequences.py",
    )
    parser.add_argument(
        "--momentum-model-path",
        type=Path,
        default=root_dir / "data" / "model_cache" / "lstm_momentum_model.pth",
        help="Path to save momentum LSTM checkpoint.",
    )
    parser.add_argument(
        "--momentum-scaler-path",
        type=Path,
        default=root_dir / "data" / "model_cache" / "lstm_momentum_scalers.pkl",
        help="Path to save sequence/static scalers used for momentum model inputs.",
    )
    parser.add_argument(
        "--xgb-model-path",
        type=Path,
        default=root_dir / "data" / "model_cache" / "lstm_xgb_ensemble.json",
        help="Path to save trained XGBoost model.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=root_dir / "data" / "model_cache" / "lstm_xgb_ensemble_metrics.json",
        help="Path to save train/eval metrics report.",
    )
    parser.add_argument("--epochs", type=int, default=90, help="Momentum model max epochs.")
    parser.add_argument("--patience", type=int, default=16, help="Momentum early-stop patience.")
    parser.add_argument("--batch-size", type=int, default=256, help="Momentum batch size.")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size.")
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.35, help="Momentum dropout.")
    parser.add_argument("--lr", type=float, default=7e-4, help="Momentum AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Momentum weight decay.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument(
        "--bidirectional",
        dest="bidirectional",
        action="store_true",
        help="Use bidirectional LSTM encoder.",
    )
    parser.add_argument(
        "--no-bidirectional",
        dest="bidirectional",
        action="store_false",
        help="Use unidirectional LSTM encoder.",
    )
    parser.set_defaults(bidirectional=True)
    parser.add_argument(
        "--use-cross-attention",
        dest="use_cross_attention",
        action="store_true",
        help="Enable cross-attention between fighter timelines.",
    )
    parser.add_argument(
        "--no-cross-attention",
        dest="use_cross_attention",
        action="store_false",
        help="Disable cross-attention between fighter timelines.",
    )
    parser.set_defaults(use_cross_attention=True)
    parser.add_argument("--attention-heads", type=int, default=4, help="Cross-attention heads.")
    parser.add_argument("--attention-dropout", type=float, default=0.10, help="Cross-attention dropout.")
    parser.add_argument("--warmup-epochs", type=int, default=4, help="Linear LR warmup epochs.")
    parser.add_argument("--min-epochs", type=int, default=10, help="Minimum epochs before early stop.")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum score improvement.")
    parser.add_argument(
        "--static-recency-mode",
        type=str,
        default="ema",
        choices=["ema", "mean"],
        help="Recency summarization strategy passed to augmentation.",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.75, help="EMA alpha for static recency.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation split fraction.")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Test split fraction.")
    parser.add_argument(
        "--holdout-manifest-path",
        type=Path,
        default=root_dir / "data" / "model_cache" / "final_holdout_fight_ids.txt",
        help="Path to locked chronological holdout fight IDs. Created on first run and reused thereafter.",
    )
    parser.add_argument(
        "--refresh-holdout-manifest",
        action="store_true",
        help="Regenerate holdout manifest from current chronology (overwrites existing holdout IDs).",
    )
    parser.add_argument(
        "--drop-empty-history",
        dest="drop_empty_history",
        action="store_true",
        help="Drop fights with fully empty history sequences.",
    )
    parser.add_argument(
        "--keep-empty-history",
        dest="drop_empty_history",
        action="store_false",
        help="Keep fights with fully empty history sequences.",
    )
    parser.set_defaults(drop_empty_history=True)
    parser.add_argument("--max-fights", type=int, default=None, help="Optional cap for fights.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--xgb-n-estimators", type=int, default=2500, help="XGB max trees.")
    parser.add_argument("--xgb-lr", type=float, default=0.02, help="XGB learning rate.")
    parser.add_argument("--xgb-max-depth", type=int, default=4, help="XGB max tree depth.")
    parser.add_argument("--xgb-min-child-weight", type=float, default=8.0, help="XGB min child weight.")
    parser.add_argument("--xgb-subsample", type=float, default=0.85, help="XGB subsample ratio.")
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.85, help="XGB colsample ratio.")
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.2, help="XGB L1 regularization.")
    parser.add_argument("--xgb-reg-lambda", type=float, default=3.0, help="XGB L2 regularization.")
    parser.add_argument("--xgb-gamma", type=float, default=0.2, help="XGB split gain threshold.")
    parser.add_argument("--xgb-early-stopping", type=int, default=120, help="XGB early stopping rounds.")
    parser.add_argument("--xgb-n-jobs", type=int, default=0, help="XGB CPU workers (0=all).")
    parser.add_argument(
        "--use-weight-class-specialists",
        dest="use_weight_class_specialists",
        action="store_true",
        help="Train per-weight-class specialist XGBoost models and blend with global model.",
    )
    parser.add_argument(
        "--no-weight-class-specialists",
        dest="use_weight_class_specialists",
        action="store_false",
        help="Disable per-weight-class specialist XGBoost models.",
    )
    parser.set_defaults(use_weight_class_specialists=True)
    parser.add_argument(
        "--specialist-min-train-samples",
        type=int,
        default=280,
        help="Minimum oriented train samples required to fit a specialist model.",
    )
    parser.add_argument(
        "--specialist-min-val-samples",
        type=int,
        default=80,
        help="Minimum oriented validation samples required for specialist early-stopping.",
    )
    parser.add_argument(
        "--specialist-blend-alpha",
        type=float,
        default=0.65,
        help="Blend factor for specialist predictions: final=(1-a)*global + a*specialist.",
    )
    parser.add_argument(
        "--xgb-specialists-path",
        type=Path,
        default=root_dir / "data" / "model_cache" / "lstm_xgb_specialists.pkl",
        help="Path to save specialist XGBoost models.",
    )
    parser.add_argument(
        "--use-oof-stacking",
        dest="use_oof_stacking",
        action="store_true",
        help="Train XGBoost on chronological out-of-fold momentum scores (leakage-safe; default).",
    )
    parser.add_argument(
        "--no-oof-stacking",
        dest="use_oof_stacking",
        action="store_false",
        help="Disable OOF stacking and train XGBoost on in-sample momentum scores (faster, leakage-prone).",
    )
    parser.set_defaults(use_oof_stacking=True)
    parser.add_argument(
        "--oof-folds",
        type=int,
        default=4,
        help="Number of expanding-window OOF folds used when OOF stacking is enabled.",
    )
    parser.add_argument(
        "--oof-min-train-fights",
        type=int,
        default=700,
        help="Minimum initial train fights before first OOF validation fold.",
    )
    parser.add_argument(
        "--use-walkforward-cv",
        dest="use_walkforward_cv",
        action="store_true",
        help="Compute walk-forward CV metrics from chronological OOF folds for robust model selection.",
    )
    parser.add_argument(
        "--no-walkforward-cv",
        dest="use_walkforward_cv",
        action="store_false",
        help="Disable additional walk-forward CV reporting.",
    )
    parser.set_defaults(use_walkforward_cv=True)
    parser.add_argument(
        "--walkforward-std-penalty",
        type=float,
        default=0.50,
        help="Selection score uses mean_auc - penalty * std_auc on walk-forward folds.",
    )
    parser.add_argument(
        "--symmetry-loss-weight",
        type=float,
        default=0.20,
        help="AB/BA consistency penalty weight for momentum model training.",
    )
    parser.add_argument(
        "--trend-ema-alpha",
        type=float,
        default=0.70,
        help="EMA alpha used for engineered trend/static features in the XGBoost head.",
    )
    parser.add_argument(
        "--disable-trend-static-features",
        dest="use_trend_static_features",
        action="store_false",
        help="Disable trend/volatility/opponent-adjusted static features for XGBoost.",
    )
    parser.set_defaults(use_trend_static_features=True)
    parser.add_argument(
        "--enhanced-context-static-features",
        dest="use_enhanced_context_static_features",
        action="store_true",
        help="Enable richer age/rust/quality interaction features for XGBoost.",
    )
    parser.add_argument(
        "--no-enhanced-context-static-features",
        dest="use_enhanced_context_static_features",
        action="store_false",
        help="Disable richer age/rust/quality interaction features for XGBoost.",
    )
    parser.set_defaults(use_enhanced_context_static_features=False)
    parser.add_argument(
        "--oof-ensemble-pred-path",
        type=Path,
        default=None,
        help="Optional .npz path to save walk-forward OOF ensemble probabilities for meta-stacking.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


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


def split_oriented_fight_id(fight_id: str) -> tuple[str, str]:
    text = str(fight_id).strip()
    if text.endswith("_ab"):
        return text[:-3], "ab"
    if text.endswith("_ba"):
        return text[:-3], "ba"
    # Fallback for older IDs that may use "-ab"/"-ba".
    if text.endswith("-ab"):
        return text[:-3], "ab"
    if text.endswith("-ba"):
        return text[:-3], "ba"
    return text, "ab"


class PairedFightDataset(torch.utils.data.Dataset):
    """Pairs AB/BA augmentations so symmetry loss can be enforced in each batch."""

    def __init__(self, samples: list[Any]) -> None:
        grouped: dict[str, dict[str, Any]] = {}
        for sample in samples:
            base_id, orient = split_oriented_fight_id(getattr(sample, "fight_id", ""))
            grouped.setdefault(base_id, {})[orient] = sample
        self.pairs: list[tuple[Any, Any]] = []
        for orient_map in grouped.values():
            if "ab" in orient_map and "ba" in orient_map:
                self.pairs.append((orient_map["ab"], orient_map["ba"]))
        if not self.pairs:
            raise ValueError("No AB/BA pairs found for symmetry training.")

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def to_tensors(sample: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(sample.seq_a, dtype=torch.float32),
            torch.tensor(sample.len_a, dtype=torch.long),
            torch.tensor(sample.seq_b, dtype=torch.float32),
            torch.tensor(sample.len_b, dtype=torch.long),
            torch.tensor(sample.target, dtype=torch.float32),
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        sample_ab, sample_ba = self.pairs[idx]
        return (*self.to_tensors(sample_ab), *self.to_tensors(sample_ba))


class MomentumSiameseLSTM(nn.Module):
    def __init__(
        self,
        seq_dim: int,
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
        self.head = nn.Sequential(
            nn.Linear(rep_dim * 4, 256),
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
    ) -> torch.Tensor:
        out_a, mask_a = self.encode(seq_a, len_a)
        out_b, mask_b = self.encode(seq_b, len_b)
        out_a, out_b = self.apply_cross_attention(out_a, mask_a, out_b, mask_b)
        rep_a = self.masked_pool(out_a, mask_a)
        rep_b = self.masked_pool(out_b, mask_b)
        x = torch.cat([rep_a, rep_b, rep_a - rep_b, rep_a * rep_b], dim=1)
        return self.head(x).squeeze(1)


def run_epoch_momentum(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
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

    for seq_a, len_a, seq_b, len_b, _static, target in loader:
        seq_a = seq_a.to(device)
        len_a = len_a.to(device)
        seq_b = seq_b.to(device)
        len_b = len_b.to(device)
        target = target.to(device)

        logits = model(seq_a, len_a, seq_b, len_b)
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


def run_epoch_momentum_paired(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    symmetry_loss_weight: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.train(mode=True)
    losses: list[float] = []
    sym_losses: list[float] = []
    probs_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []

    for (
        seq_a_ab,
        len_a_ab,
        seq_b_ab,
        len_b_ab,
        target_ab,
        seq_a_ba,
        len_a_ba,
        seq_b_ba,
        len_b_ba,
        target_ba,
    ) in loader:
        seq_a_ab = seq_a_ab.to(device)
        len_a_ab = len_a_ab.to(device)
        seq_b_ab = seq_b_ab.to(device)
        len_b_ab = len_b_ab.to(device)
        target_ab = target_ab.to(device)

        seq_a_ba = seq_a_ba.to(device)
        len_a_ba = len_a_ba.to(device)
        seq_b_ba = seq_b_ba.to(device)
        len_b_ba = len_b_ba.to(device)
        target_ba = target_ba.to(device)

        logits_ab = model(seq_a_ab, len_a_ab, seq_b_ab, len_b_ab)
        logits_ba = model(seq_a_ba, len_a_ba, seq_b_ba, len_b_ba)
        cls_loss = 0.5 * (criterion(logits_ab, target_ab) + criterion(logits_ba, target_ba))

        if symmetry_loss_weight > 0:
            # AB/BA consistency: p(AB) + p(BA) should be close to 1.
            sym_err = torch.sigmoid(logits_ab) + torch.sigmoid(logits_ba) - 1.0
            sym_loss = torch.mean(sym_err * sym_err)
            loss = cls_loss + (symmetry_loss_weight * sym_loss)
            sym_losses.append(float(sym_loss.item()))
        else:
            loss = cls_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        losses.append(float(loss.item()))
        probs_all.append(torch.sigmoid(logits_ab).detach().cpu().numpy())
        probs_all.append(torch.sigmoid(logits_ba).detach().cpu().numpy())
        targets_all.append(target_ab.detach().cpu().numpy())
        targets_all.append(target_ba.detach().cpu().numpy())

    probs = np.concatenate(probs_all) if probs_all else np.zeros(0, dtype=np.float32)
    targets = np.concatenate(targets_all) if targets_all else np.zeros(0, dtype=np.float32)
    sym_mean = float(np.mean(sym_losses) if sym_losses else 0.0)
    return float(np.mean(losses) if losses else 0.0), sym_mean, probs, targets


def predict_momentum(
    model: nn.Module,
    samples: list[Any],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = torch.utils.data.DataLoader(
        FightDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    probs_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    with torch.no_grad():
        for seq_a, len_a, seq_b, len_b, _static, target in loader:
            seq_a = seq_a.to(device)
            len_a = len_a.to(device)
            seq_b = seq_b.to(device)
            len_b = len_b.to(device)
            logits = model(seq_a, len_a, seq_b, len_b)
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
            targets_all.append(target.numpy())
    probs = np.concatenate(probs_all) if probs_all else np.zeros(0, dtype=np.float32)
    targets = np.concatenate(targets_all) if targets_all else np.zeros(0, dtype=np.float32)
    return probs.astype(np.float32), targets.astype(np.float32)


def train_momentum_model(
    train_samples: list[Any],
    val_samples: list[Any],
    seq_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, float, dict[str, Any]]:
    model = MomentumSiameseLSTM(
        seq_dim=seq_dim,
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

    use_symmetry = float(args.symmetry_loss_weight) > 0.0
    if use_symmetry:
        paired_dataset = PairedFightDataset(train_samples)
        train_loader = torch.utils.data.DataLoader(
            paired_dataset,
            batch_size=max(1, args.batch_size // 2),
            shuffle=True,
            num_workers=args.num_workers,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            FightDataset(train_samples),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    val_loader = torch.utils.data.DataLoader(
        FightDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    best_state: Optional[dict[str, Any]] = None
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    warmup_epochs = max(0, int(args.warmup_epochs))
    min_epochs = max(1, int(args.min_epochs))
    min_delta = float(max(args.min_delta, 0.0))
    base_lr = float(args.lr)

    for epoch in range(1, args.epochs + 1):
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_scale = float(epoch) / float(warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = max(base_lr * warmup_scale, 1e-7)

        if use_symmetry:
            train_loss, train_sym_loss, _, _ = run_epoch_momentum_paired(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                grad_clip=args.grad_clip,
                symmetry_loss_weight=float(args.symmetry_loss_weight),
            )
        else:
            train_loss, _, _ = run_epoch_momentum(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                grad_clip=args.grad_clip,
            )
            train_sym_loss = 0.0
        val_loss, val_prob, val_true = run_epoch_momentum(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            grad_clip=args.grad_clip,
        )
        if epoch > warmup_epochs:
            scheduler.step(val_loss)
        val_auc = safe_auc(val_true, val_prob)
        val_bal = evaluate_probs(val_true, val_prob, threshold=0.5)["balanced_accuracy"]
        score = val_auc if not math.isnan(val_auc) else val_bal

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_auc": float(val_auc),
                "val_balanced_accuracy": float(val_bal),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_symmetry_loss": float(train_sym_loss),
            }
        )
        logging.info(
            "Momentum Epoch %03d | train_loss=%.4f | sym=%.4f | val_loss=%.4f | val_auc=%.4f | val_bal=%.4f | lr=%.6f",
            epoch,
            train_loss,
            train_sym_loss,
            val_loss,
            val_auc,
            val_bal,
            float(optimizer.param_groups[0]["lr"]),
        )

        if score > (best_score + min_delta):
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            if epoch >= min_epochs:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    logging.info("Momentum early stopping at epoch %d", epoch)
                    break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(history)
        best_score = float("nan")
    model.load_state_dict(best_state)

    val_prob_final, val_true_final = predict_momentum(
        model=model,
        samples=val_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    threshold, val_threshold_metrics = choose_best_threshold(val_true_final, val_prob_final)
    summary = {
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "pos_weight": float(pos_weight),
        "symmetry_loss_weight": float(args.symmetry_loss_weight),
        "threshold": float(threshold),
        "val_metrics_at_best_threshold": val_threshold_metrics,
        "history": history,
    }
    return model, threshold, summary


def build_expanding_oof_windows(
    num_train_fights: int,
    requested_folds: int,
    requested_min_train_fights: int,
) -> tuple[list[tuple[int, int]], int]:
    if num_train_fights < 40:
        raise ValueError(
            f"Not enough train fights for OOF stacking: {num_train_fights} (need >= 40)."
        )
    folds = max(int(requested_folds), 1)
    min_train_fights = max(int(requested_min_train_fights), 20)

    if min_train_fights >= num_train_fights:
        min_train_fights = max(20, num_train_fights // 2)
    remaining_fights = num_train_fights - min_train_fights
    if remaining_fights < 1:
        raise ValueError(
            f"OOF requires at least 1 validation fight after min_train_fights; got {num_train_fights=} {min_train_fights=}."
        )
    if folds > remaining_fights:
        folds = remaining_fights

    base = remaining_fights // folds
    extra = remaining_fights % folds
    windows: list[tuple[int, int]] = []
    val_start = min_train_fights
    for i in range(folds):
        fold_size = base + (1 if i < extra else 0)
        if fold_size <= 0:
            continue
        val_end = min(val_start + fold_size, num_train_fights)
        if val_end > val_start:
            windows.append((val_start, val_end))
        val_start = val_end
    if not windows:
        raise ValueError("Failed to construct OOF windows.")
    return windows, min_train_fights


def build_oof_momentum_predictions(
    train_samples: list[Any],
    *,
    num_train_fights: int,
    seq_len: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    expected_samples = num_train_fights * 2
    if len(train_samples) != expected_samples:
        raise ValueError(
            f"Train sample count mismatch for OOF: {len(train_samples)} vs expected {expected_samples}."
        )

    windows, min_train_fights = build_expanding_oof_windows(
        num_train_fights=num_train_fights,
        requested_folds=args.oof_folds,
        requested_min_train_fights=args.oof_min_train_fights,
    )
    logging.info(
        "OOF stacking enabled | windows=%d | min_train_fights=%d | first_val_fight_idx=%d",
        len(windows),
        min_train_fights,
        windows[0][0],
    )

    oof_pred = np.full(expected_samples, np.nan, dtype=np.float32)
    fold_reports: list[dict[str, Any]] = []

    for fold_idx, (val_start_fight, val_end_fight) in enumerate(windows, start=1):
        # Expanding-window chronology: train on all fights strictly before this fold.
        train_end_fight = val_start_fight
        train_start_sample = 0
        train_end_sample = train_end_fight * 2
        val_start_sample = val_start_fight * 2
        val_end_sample = val_end_fight * 2

        fold_train_raw = train_samples[train_start_sample:train_end_sample]
        fold_val_raw = train_samples[val_start_sample:val_end_sample]
        if not fold_train_raw or not fold_val_raw:
            continue

        set_seed(args.seed + fold_idx)
        fold_seq_scaler, fold_static_scaler = fit_scalers(fold_train_raw)
        fold_train = transform_samples(fold_train_raw, fold_seq_scaler, fold_static_scaler, seq_len)
        fold_val = transform_samples(fold_val_raw, fold_seq_scaler, fold_static_scaler, seq_len)
        seq_dim = fold_train[0].seq_a.shape[1]

        logging.info(
            "OOF fold %d/%d | train_fights=%d | val_fights=%d",
            fold_idx,
            len(windows),
            train_end_fight,
            val_end_fight - val_start_fight,
        )
        fold_model, fold_threshold, fold_summary = train_momentum_model(
            train_samples=fold_train,
            val_samples=fold_val,
            seq_dim=seq_dim,
            args=args,
            device=device,
        )
        fold_prob, fold_true = predict_momentum(
            fold_model,
            fold_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        oof_pred[val_start_sample:val_end_sample] = fold_prob.astype(np.float32)
        fold_metrics = evaluate_probs(fold_true, fold_prob, fold_threshold)
        fold_reports.append(
            {
                "fold_index": int(fold_idx),
                "train_fights": int(train_end_fight),
                "val_fights": int(val_end_fight - val_start_fight),
                "val_start_fight_idx": int(val_start_fight),
                "val_end_fight_idx": int(val_end_fight),
                "threshold": float(fold_threshold),
                "best_epoch": int(fold_summary["best_epoch"]),
                "best_score": float(fold_summary["best_score"]),
                "val_metrics": fold_metrics,
            }
        )

    # Restore base seed before the final full-train momentum fit.
    set_seed(args.seed)
    oof_mask = ~np.isnan(oof_pred)
    coverage = float(oof_mask.mean())
    logging.info(
        "OOF coverage: %d/%d samples (%.2f%%)",
        int(oof_mask.sum()),
        int(len(oof_mask)),
        coverage * 100.0,
    )
    if coverage < 0.5:
        raise ValueError(
            f"OOF coverage too low ({coverage:.2%}); reduce --oof-min-train-fights or folds."
        )
    return oof_pred.astype(np.float32), oof_mask.astype(bool), fold_reports


def summarize_walkforward_fold_metrics(
    fold_reports: list[dict[str, Any]],
    *,
    std_penalty: float,
) -> dict[str, Any]:
    aucs = np.asarray(
        [float(fr.get("val_metrics", {}).get("auc", float("nan"))) for fr in fold_reports],
        dtype=np.float64,
    )
    aucs = aucs[np.isfinite(aucs)]
    if aucs.size == 0:
        return {
            "fold_count": 0,
            "mean_auc": float("nan"),
            "std_auc": float("nan"),
            "selection_score": float("nan"),
        }
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    score = float(mean_auc - (max(std_penalty, 0.0) * std_auc))
    return {
        "fold_count": int(aucs.size),
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "selection_score": score,
    }


def build_walkforward_ensemble_cv(
    *,
    x_train_static: np.ndarray,
    y_train: np.ndarray,
    oof_train_prob: np.ndarray,
    oof_fold_reports: list[dict[str, Any]],
    args: argparse.Namespace,
    oof_pred_output_path: Optional[Path] = None,
) -> dict[str, Any]:
    if not oof_fold_reports:
        return {
            "enabled": False,
            "reason": "no_oof_fold_reports",
            "folds_evaluated": 0,
        }

    n = len(y_train)
    if len(oof_train_prob) != n:
        raise ValueError("OOF momentum predictions misaligned for walk-forward CV.")
    stacked_all = np.concatenate([x_train_static, oof_train_prob.reshape(-1, 1)], axis=1)
    row_idx = np.arange(n, dtype=np.int64)
    fold_metrics: list[dict[str, Any]] = []
    fold_pred = np.full(n, np.nan, dtype=np.float32)

    for fold_i, fold in enumerate(oof_fold_reports, start=1):
        val_start = int(fold["val_start_fight_idx"]) * 2
        val_end = int(fold["val_end_fight_idx"]) * 2
        val_mask = (row_idx >= val_start) & (row_idx < val_end)
        train_mask = (row_idx < val_start) & np.isfinite(oof_train_prob)
        if int(train_mask.sum()) < 120 or int(val_mask.sum()) < 40:
            continue
        y_train_fold = y_train[train_mask]
        y_val_fold = y_train[val_mask]
        if len(np.unique(y_train_fold)) < 2 or len(np.unique(y_val_fold)) < 2:
            continue
        x_train_fold = stacked_all[train_mask]
        x_val_fold = stacked_all[val_mask]

        xgb_fold = fit_xgb_model(
            x_train=x_train_fold,
            y_train=y_train_fold,
            x_val=x_val_fold,
            y_val=y_val_fold,
            args=args,
            random_state=args.seed + 1000 + fold_i,
        )
        val_prob_fold = xgb_fold.predict_proba(x_val_fold)[:, 1].astype(np.float32)
        fold_pred[val_mask] = val_prob_fold
        fold_metrics.append(
            {
                "fold_index": int(fold_i),
                "train_samples": int(train_mask.sum()),
                "val_samples": int(val_mask.sum()),
                "val_start_sample": int(val_start),
                "val_end_sample": int(val_end),
                "val_metrics": evaluate_probs(y_val_fold.astype(np.float32), val_prob_fold, threshold=0.5),
            }
        )

    valid_mask = np.isfinite(fold_pred)
    if int(valid_mask.sum()) == 0:
        return {
            "enabled": True,
            "reason": "no_valid_walkforward_folds",
            "folds_evaluated": 0,
            "coverage": 0.0,
            "fold_metrics": fold_metrics,
        }

    y_true = y_train[valid_mask].astype(np.float32)
    y_prob = fold_pred[valid_mask]
    summary = summarize_walkforward_fold_metrics(fold_metrics, std_penalty=args.walkforward_std_penalty)
    summary.update(
        {
            "enabled": True,
            "folds_evaluated": int(len(fold_metrics)),
            "coverage": float(valid_mask.mean()),
            "metrics_at_0_5": evaluate_probs(y_true, y_prob, threshold=0.5),
        }
    )
    if oof_pred_output_path is not None:
        oof_pred_output_path = Path(oof_pred_output_path)
        oof_pred_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            oof_pred_output_path,
            oof_pred=fold_pred.astype(np.float32),
            valid_mask=valid_mask.astype(np.uint8),
            y_true=y_train.astype(np.float32),
        )
        summary["oof_pred_path"] = str(oof_pred_output_path)
    summary["fold_metrics"] = fold_metrics
    return summary


def row_float(row: pd.Series, name: str) -> float:
    if name not in row.index:
        return 0.0
    value = row[name]
    if pd.isna(value):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def split_with_locked_holdout(
    df: pd.DataFrame,
    *,
    val_fraction: float,
    test_fraction: float,
    holdout_manifest_path: Path,
    refresh_holdout_manifest: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ordered = df.sort_values(["event_date", "fight_id"]).reset_index(drop=True)
    holdout_manifest_path = Path(holdout_manifest_path)

    def build_from_manifest(fight_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        holdout_mask = ordered["fight_id"].astype(str).isin(fight_ids).to_numpy()
        if int(holdout_mask.sum()) == 0:
            raise ValueError("Locked holdout manifest had no matching fight_id values in current dataset.")
        test_df_local = ordered[holdout_mask].copy()
        remaining = ordered[~holdout_mask].copy()
        if remaining.empty:
            raise ValueError("Locked holdout consumed all fights; no data left for train/val.")
        val_count = max(1, int(round(len(ordered) * float(val_fraction))))
        val_count = min(val_count, max(1, len(remaining) - 1))
        train_df_local = remaining.iloc[:-val_count].copy()
        val_df_local = remaining.iloc[-val_count:].copy()
        if train_df_local.empty or val_df_local.empty:
            raise ValueError("Locked holdout split produced empty train/val partitions.")
        return train_df_local, val_df_local, test_df_local

    used_existing = False
    manifest_ids: list[str] = []
    if holdout_manifest_path.exists() and not refresh_holdout_manifest:
        try:
            text = holdout_manifest_path.read_text(encoding="utf-8")
            manifest_ids = [line.strip() for line in text.splitlines() if line.strip()]
            if manifest_ids:
                train_df, val_df, test_df = build_from_manifest(set(manifest_ids))
                used_existing = True
            else:
                logging.warning(
                    "Holdout manifest exists but is empty: %s. Regenerating holdout from chronology.",
                    holdout_manifest_path,
                )
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning(
                "Failed to load holdout manifest %s (%s). Regenerating holdout from chronology.",
                holdout_manifest_path,
                exc,
            )

    if not used_existing:
        train_df, val_df, test_df = chronological_split(ordered, val_fraction, test_fraction)
        holdout_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_ids = test_df["fight_id"].astype(str).tolist()
        holdout_manifest_path.write_text("\n".join(manifest_ids) + "\n", encoding="utf-8")
        logging.info(
            "Locked final holdout manifest %s with %d fights.",
            holdout_manifest_path,
            len(manifest_ids),
        )
    else:
        logging.info(
            "Using locked final holdout from %s (%d fights).",
            holdout_manifest_path,
            len(manifest_ids),
        )

    info = {
        "manifest_path": str(holdout_manifest_path),
        "used_existing_manifest": bool(used_existing),
        "refresh_requested": bool(refresh_holdout_manifest),
        "holdout_fights": int(len(test_df)),
        "holdout_fight_ids_preview": manifest_ids[:10],
    }
    return train_df, val_df, test_df, info


def normalize_weight_class(value: Any) -> str:
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "NULL"}:
        return "UNKNOWN"
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Z0-9_]+", "", text)
    return text or "UNKNOWN"


def attach_weight_class(df: pd.DataFrame, input_csv: Path) -> pd.DataFrame:
    out = df.copy()
    if "weight_class" in out.columns:
        out["weight_class"] = out["weight_class"].map(normalize_weight_class)
        return out

    details_csv = input_csv.with_name("ufc_fight_details_lstm.csv")
    if not details_csv.exists():
        logging.warning(
            "weight_class unavailable: %s not found. Specialist models will fall back to UNKNOWN.",
            details_csv,
        )
        out["weight_class"] = "UNKNOWN"
        return out

    try:
        details = pd.read_csv(details_csv, usecols=["fight_id", "weight_class"])
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning(
            "Failed to read weight_class from %s (%s). Using UNKNOWN fallback.",
            details_csv,
            exc,
        )
        out["weight_class"] = "UNKNOWN"
        return out

    if "fight_id" not in details.columns or "weight_class" not in details.columns:
        out["weight_class"] = "UNKNOWN"
        return out

    details = details.copy()
    details["fight_id"] = details["fight_id"].astype(str).str.strip()
    details["weight_class"] = details["weight_class"].map(normalize_weight_class)
    details = details.drop_duplicates(subset=["fight_id"], keep="last")

    out = out.merge(details, on="fight_id", how="left")
    out["weight_class"] = out["weight_class"].map(normalize_weight_class)
    coverage = float((out["weight_class"] != "UNKNOWN").mean())
    logging.info("weight_class coverage after merge: %.2f%%", coverage * 100.0)
    return out


def build_oriented_weight_classes(df: pd.DataFrame) -> np.ndarray:
    classes = df["weight_class"].map(normalize_weight_class).to_numpy(dtype=object)
    return np.repeat(classes, 2)


def safe_div_array(num: np.ndarray, den: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return num / np.maximum(den, eps)


def ema_1d(values: np.ndarray, alpha: float) -> float:
    if values.size == 0:
        return 0.0
    if alpha <= 0:
        return float(values.mean())
    if alpha >= 1:
        return float(values[-1])
    powers = np.arange(values.size - 1, -1, -1, dtype=np.float64)
    weights = np.power(1.0 - alpha, powers)
    weights /= np.maximum(weights.sum(), 1e-12)
    return float(np.dot(values, weights))


def slope_1d(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    x_center = x - x.mean()
    y_center = values.astype(np.float64) - values.mean()
    den = float(np.dot(x_center, x_center))
    if den <= 1e-12:
        return 0.0
    return float(np.dot(x_center, y_center) / den)


def volatility_1d(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    return float(np.std(np.diff(values.astype(np.float64))))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    w = np.maximum(weights.astype(np.float64), 1e-6)
    return float(np.average(values.astype(np.float64), weights=w))


def history_col(history: np.ndarray, idx: int) -> np.ndarray:
    if history.ndim != 2 or idx < 0 or idx >= history.shape[1]:
        return np.zeros(history.shape[0], dtype=np.float64)
    return history[:, idx].astype(np.float64)


def compute_fighter_trend_bundle(history: np.ndarray, length: int, ema_alpha: float) -> list[float]:
    if length <= 0 or history.size == 0:
        # 29 values (kept fixed for stable feature shapes).
        return [0.0] * 29
    valid = history[-length:].astype(np.float64)

    kd_for = history_col(valid, 0)
    kd_against = history_col(valid, 1)
    sig_landed = history_col(valid, 2)
    sig_attempted = history_col(valid, 3)
    sig_absorbed = history_col(valid, 4)
    sig_absorbed_attempted = history_col(valid, 5)
    td_landed = history_col(valid, 6)
    td_attempted = history_col(valid, 7)
    sub_attempted = history_col(valid, 10)
    ctrl_seconds = history_col(valid, 11)
    fight_secs = np.maximum(history_col(valid, 12), 1.0)
    opp_elo = np.maximum(history_col(valid, 13), 0.0)

    sig_acc = safe_div_array(sig_landed, sig_attempted)
    td_acc = safe_div_array(td_landed, td_attempted)
    sig_def = 1.0 - safe_div_array(sig_absorbed, sig_absorbed_attempted)
    ctrl_rate = safe_div_array(ctrl_seconds, fight_secs)
    activity = safe_div_array(sig_attempted + td_attempted + sub_attempted, fight_secs)
    kd_balance = kd_for - kd_against

    opp_weights = np.clip(opp_elo / 1500.0, 0.50, 1.80)
    sig_adj = weighted_mean(sig_acc, opp_weights)
    td_adj = weighted_mean(td_acc, opp_weights)
    sig_def_adj = weighted_mean(sig_def, opp_weights)

    metrics = [sig_acc, td_acc, sig_def, ctrl_rate, activity, kd_balance]
    ema_vals = [ema_1d(m, ema_alpha) for m in metrics]
    slope_vals = [slope_1d(m) for m in metrics]
    std_vals = [float(np.std(m)) for m in metrics]
    vol_vals = [volatility_1d(m) for m in metrics]

    return [
        *ema_vals,
        *slope_vals,
        *std_vals,
        *vol_vals,
        float(np.mean(opp_elo)),
        ema_1d(opp_elo, ema_alpha),
        sig_adj,
        td_adj,
        sig_def_adj,
    ]


def make_trend_static_features(
    seq_a: np.ndarray,
    len_a: int,
    seq_b: np.ndarray,
    len_b: int,
    *,
    ema_alpha: float,
) -> list[float]:
    trend_a = compute_fighter_trend_bundle(seq_a, len_a, ema_alpha)
    trend_b = compute_fighter_trend_bundle(seq_b, len_b, ema_alpha)
    trend_diff = [a - b for a, b in zip(trend_a, trend_b)]
    return [*trend_a, *trend_b, *trend_diff]


def make_static_features(
    *,
    elo_a: float,
    elo_b: float,
    days_a: float,
    days_b: float,
    age_a_days: float,
    age_b_days: float,
    age_gap_over_5y: float,
    height_a_cm: float,
    height_b_cm: float,
    reach_a_cm: float,
    reach_b_cm: float,
    career_abs_a: float,
    career_abs_b: float,
    career_over_1500_a: float,
    career_over_1500_b: float,
) -> list[float]:
    days_a = max(days_a, 0.0)
    days_b = max(days_b, 0.0)
    age_a_days = max(age_a_days, 0.0)
    age_b_days = max(age_b_days, 0.0)
    height_a_cm = max(height_a_cm, 0.0)
    height_b_cm = max(height_b_cm, 0.0)
    reach_a_cm = max(reach_a_cm, 0.0)
    reach_b_cm = max(reach_b_cm, 0.0)
    career_abs_a = max(career_abs_a, 0.0)
    career_abs_b = max(career_abs_b, 0.0)

    age_a_years = age_a_days / 365.25 if age_a_days > 0 else 0.0
    age_b_years = age_b_days / 365.25 if age_b_days > 0 else 0.0
    age_diff_years = age_a_years - age_b_years
    abs_age_gap_years = abs(age_diff_years)
    age_gap_over_5 = max(age_gap_over_5y, 1.0 if abs_age_gap_years >= 5.0 else 0.0)
    younger_a = 1.0 if age_diff_years < 0 else 0.0

    elo_diff = elo_a - elo_b
    days_diff = days_a - days_b
    reach_diff = reach_a_cm - reach_b_cm
    height_diff = height_a_cm - height_b_cm
    career_abs_diff = career_abs_a - career_abs_b

    return [
        elo_a,
        elo_b,
        elo_diff,
        days_a,
        days_b,
        days_diff,
        np.log1p(days_a),
        np.log1p(days_b),
        age_a_years,
        age_b_years,
        age_diff_years,
        abs_age_gap_years,
        age_gap_over_5,
        younger_a,
        height_a_cm,
        height_b_cm,
        height_diff,
        reach_a_cm,
        reach_b_cm,
        reach_diff,
        career_abs_a,
        career_abs_b,
        career_abs_diff,
        np.log1p(career_abs_a),
        np.log1p(career_abs_b),
        career_over_1500_a,
        career_over_1500_b,
    ]


def rust_bucket_features(days_since_last_fight: float) -> list[float]:
    days = max(float(days_since_last_fight), 0.0)
    if days <= 120.0:
        return [1.0, 0.0, 0.0, 0.0]
    if days <= 365.0:
        return [0.0, 1.0, 0.0, 0.0]
    if days <= 730.0:
        return [0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0, 1.0]


def safe_div_scalar(num: float, den: float, eps: float = 1e-6) -> float:
    return float(num) / float(max(float(den), eps))


def compute_quality_context(history: np.ndarray, length: int, *, ema_alpha: float = 0.70) -> list[float]:
    if length <= 0 or history.size == 0:
        return [0.0] * 8
    valid = history[-length:].astype(np.float64)
    opp_elo = np.maximum(history_col(valid, 13), 0.0)
    if opp_elo.size == 0:
        return [0.0] * 8
    recent = opp_elo[-min(3, int(opp_elo.size)) :]
    mean_all = float(np.mean(opp_elo))
    recent_mean = float(np.mean(recent))
    return [
        mean_all,
        float(np.std(opp_elo)),
        recent_mean,
        float(recent_mean - mean_all),
        slope_1d(opp_elo),
        ema_1d(opp_elo, ema_alpha),
        float(np.max(opp_elo)),
        float(np.min(opp_elo)),
    ]


def make_enhanced_context_features(
    *,
    elo_a: float,
    elo_b: float,
    days_a: float,
    days_b: float,
    age_a_days: float,
    age_b_days: float,
    height_a_cm: float,
    height_b_cm: float,
    reach_a_cm: float,
    reach_b_cm: float,
    career_abs_a: float,
    career_abs_b: float,
    quality_a: list[float],
    quality_b: list[float],
) -> list[float]:
    days_a = max(days_a, 0.0)
    days_b = max(days_b, 0.0)
    age_a_years = max(age_a_days, 0.0) / 365.25 if age_a_days > 0 else 0.0
    age_b_years = max(age_b_days, 0.0) / 365.25 if age_b_days > 0 else 0.0
    age_gap = age_a_years - age_b_years
    abs_age_gap = abs(age_gap)
    younger_a = 1.0 if age_gap < 0 else 0.0

    rust_a = rust_bucket_features(days_a)
    rust_b = rust_bucket_features(days_b)
    rust_gap_years = (days_a - days_b) / 365.25
    rust_ratio = safe_div_scalar(np.log1p(days_a), np.log1p(days_b))

    reach_height_a = safe_div_scalar(max(reach_a_cm, 0.0), max(height_a_cm, 1e-6))
    reach_height_b = safe_div_scalar(max(reach_b_cm, 0.0), max(height_b_cm, 1e-6))
    reach_diff = max(reach_a_cm, 0.0) - max(reach_b_cm, 0.0)
    height_diff = max(height_a_cm, 0.0) - max(height_b_cm, 0.0)

    dmg_a = max(career_abs_a, 0.0)
    dmg_b = max(career_abs_b, 0.0)
    dmg_per_year_a = safe_div_scalar(dmg_a, max(age_a_years, 18.0))
    dmg_per_year_b = safe_div_scalar(dmg_b, max(age_b_years, 18.0))

    elo_diff = elo_a - elo_b
    elo_rust_adj = safe_div_scalar(elo_diff, 1.0 + abs(rust_gap_years))
    elo_age_adj = safe_div_scalar(elo_diff, 1.0 + abs_age_gap)

    qa_mean, qa_std, qa_recent, qa_recent_delta, qa_slope, qa_ema, qa_max, qa_min = quality_a
    qb_mean, qb_std, qb_recent, qb_recent_delta, qb_slope, qb_ema, qb_max, qb_min = quality_b

    return [
        *rust_a,
        *rust_b,
        rust_gap_years,
        rust_ratio,
        age_gap,
        abs_age_gap,
        age_gap * age_gap,
        younger_a * abs_age_gap,
        max(age_a_years, age_b_years),
        reach_height_a,
        reach_height_b,
        reach_height_a - reach_height_b,
        reach_diff * younger_a,
        height_diff * younger_a,
        dmg_per_year_a,
        dmg_per_year_b,
        dmg_per_year_a - dmg_per_year_b,
        elo_rust_adj,
        elo_age_adj,
        qa_mean,
        qb_mean,
        qa_mean - qb_mean,
        qa_recent,
        qb_recent,
        qa_recent - qb_recent,
        qa_recent_delta - qb_recent_delta,
        qa_slope - qb_slope,
        qa_ema - qb_ema,
        qa_std - qb_std,
        qa_max - qb_max,
        qa_min - qb_min,
    ]


def build_oriented_static_matrix(
    df: pd.DataFrame,
    *,
    f1_raw: np.ndarray,
    f2_raw: np.ndarray,
    trend_ema_alpha: float,
    use_trend_static_features: bool,
    use_enhanced_context_static_features: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[int] = []
    for i, row in enumerate(df.itertuples(index=False)):
        row_s = pd.Series(row._asdict())
        target_ab = int(str(row_s["outcome_label"]) == POSITIVE_LABEL)

        elo_f1 = row_float(row_s, "f1_pre_fight_elo")
        elo_f2 = row_float(row_s, "f2_pre_fight_elo")
        days_f1 = row_float(row_s, "f1_days_since_last_fight")
        days_f2 = row_float(row_s, "f2_days_since_last_fight")
        age_days_f1 = row_float(row_s, "fighter_1_age_days")
        age_days_f2 = row_float(row_s, "fighter_2_age_days")
        age_gap_over_5y = row_float(row_s, "age_gap_over_5y")
        height_f1 = row_float(row_s, "fighter_1_height_cm")
        height_f2 = row_float(row_s, "fighter_2_height_cm")
        reach_f1 = row_float(row_s, "fighter_1_reach_cm")
        reach_f2 = row_float(row_s, "fighter_2_reach_cm")
        career_abs_f1 = row_float(row_s, "f1_career_significant_strikes_absorbed")
        career_abs_f2 = row_float(row_s, "f2_career_significant_strikes_absorbed")
        career_over_1500_f1 = row_float(row_s, "f1_career_significant_strikes_absorbed_over_1500")
        career_over_1500_f2 = row_float(row_s, "f2_career_significant_strikes_absorbed_over_1500")
        if abs(career_over_1500_f1) < 1e-9 and career_abs_f1 >= 1500.0:
            career_over_1500_f1 = 1.0
        if abs(career_over_1500_f2) < 1e-9 and career_abs_f2 >= 1500.0:
            career_over_1500_f2 = 1.0
        len_f1 = int(max(row_float(row_s, "f1_history_len"), 0.0))
        len_f2 = int(max(row_float(row_s, "f2_history_len"), 0.0))
        len_f1 = min(len_f1, int(f1_raw.shape[1]))
        len_f2 = min(len_f2, int(f2_raw.shape[1]))
        quality_f1 = compute_quality_context(f1_raw[i], len_f1)
        quality_f2 = compute_quality_context(f2_raw[i], len_f2)

        base_ab = make_static_features(
            elo_a=elo_f1,
            elo_b=elo_f2,
            days_a=days_f1,
            days_b=days_f2,
            age_a_days=age_days_f1,
            age_b_days=age_days_f2,
            age_gap_over_5y=age_gap_over_5y,
            height_a_cm=height_f1,
            height_b_cm=height_f2,
            reach_a_cm=reach_f1,
            reach_b_cm=reach_f2,
            career_abs_a=career_abs_f1,
            career_abs_b=career_abs_f2,
            career_over_1500_a=career_over_1500_f1,
            career_over_1500_b=career_over_1500_f2,
        )
        if use_trend_static_features:
            base_ab.extend(
                make_trend_static_features(
                    f1_raw[i],
                    len_f1,
                    f2_raw[i],
                    len_f2,
                    ema_alpha=trend_ema_alpha,
                )
            )
        if use_enhanced_context_static_features:
            base_ab.extend(
                make_enhanced_context_features(
                    elo_a=elo_f1,
                    elo_b=elo_f2,
                    days_a=days_f1,
                    days_b=days_f2,
                    age_a_days=age_days_f1,
                    age_b_days=age_days_f2,
                    height_a_cm=height_f1,
                    height_b_cm=height_f2,
                    reach_a_cm=reach_f1,
                    reach_b_cm=reach_f2,
                    career_abs_a=career_abs_f1,
                    career_abs_b=career_abs_f2,
                    quality_a=quality_f1,
                    quality_b=quality_f2,
                )
            )
        rows.append(base_ab)
        labels.append(target_ab)

        base_ba = make_static_features(
            elo_a=elo_f2,
            elo_b=elo_f1,
            days_a=days_f2,
            days_b=days_f1,
            age_a_days=age_days_f2,
            age_b_days=age_days_f1,
            age_gap_over_5y=age_gap_over_5y,
            height_a_cm=height_f2,
            height_b_cm=height_f1,
            reach_a_cm=reach_f2,
            reach_b_cm=reach_f1,
            career_abs_a=career_abs_f2,
            career_abs_b=career_abs_f1,
            career_over_1500_a=career_over_1500_f2,
            career_over_1500_b=career_over_1500_f1,
        )
        if use_trend_static_features:
            base_ba.extend(
                make_trend_static_features(
                    f2_raw[i],
                    len_f2,
                    f1_raw[i],
                    len_f1,
                    ema_alpha=trend_ema_alpha,
                )
            )
        if use_enhanced_context_static_features:
            base_ba.extend(
                make_enhanced_context_features(
                    elo_a=elo_f2,
                    elo_b=elo_f1,
                    days_a=days_f2,
                    days_b=days_f1,
                    age_a_days=age_days_f2,
                    age_b_days=age_days_f1,
                    height_a_cm=height_f2,
                    height_b_cm=height_f1,
                    reach_a_cm=reach_f2,
                    reach_b_cm=reach_f1,
                    career_abs_a=career_abs_f2,
                    career_abs_b=career_abs_f1,
                    quality_a=quality_f2,
                    quality_b=quality_f1,
                )
            )
        rows.append(base_ba)
        labels.append(1 - target_ab)

    matrix = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return matrix, y


def fit_xgb_model(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    random_state: int,
) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=args.xgb_n_estimators,
        learning_rate=args.xgb_lr,
        max_depth=args.xgb_max_depth,
        min_child_weight=args.xgb_min_child_weight,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        reg_alpha=args.xgb_reg_alpha,
        reg_lambda=args.xgb_reg_lambda,
        gamma=args.xgb_gamma,
        early_stopping_rounds=args.xgb_early_stopping,
        tree_method="hist",
        random_state=random_state,
        n_jobs=args.xgb_n_jobs if args.xgb_n_jobs != 0 else None,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    return model


def fit_specialist_models(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    weight_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    weight_val: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, XGBClassifier], list[dict[str, Any]]]:
    specialist_models: dict[str, XGBClassifier] = {}
    reports: list[dict[str, Any]] = []
    if not args.use_weight_class_specialists:
        return specialist_models, reports

    for weight_class in sorted({str(w) for w in weight_train.tolist()}):
        train_mask = weight_train == weight_class
        val_mask = weight_val == weight_class
        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())
        if n_train < args.specialist_min_train_samples or n_val < args.specialist_min_val_samples:
            continue
        y_tr = y_train[train_mask]
        y_va = y_val[val_mask]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue

        seed_offset = int(sum(ord(ch) for ch in weight_class) % 10000)
        specialist = fit_xgb_model(
            x_train=x_train[train_mask],
            y_train=y_tr,
            x_val=x_val[val_mask],
            y_val=y_va,
            args=args,
            random_state=args.seed + 2000 + seed_offset,
        )
        pred_val = specialist.predict_proba(x_val[val_mask])[:, 1].astype(np.float32)
        specialist_models[weight_class] = specialist
        reports.append(
            {
                "weight_class": weight_class,
                "train_samples": n_train,
                "val_samples": n_val,
                "val_metrics": evaluate_probs(y_va.astype(np.float32), pred_val, threshold=0.5),
            }
        )
    return specialist_models, reports


def blend_specialist_predictions(
    *,
    base_prob: np.ndarray,
    x_matrix: np.ndarray,
    weight_classes: np.ndarray,
    specialist_models: dict[str, XGBClassifier],
    alpha: float,
) -> np.ndarray:
    out = base_prob.astype(np.float32).copy()
    a = float(np.clip(alpha, 0.0, 1.0))
    if a <= 0.0 or not specialist_models:
        return out
    for weight_class, model in specialist_models.items():
        mask = weight_classes == weight_class
        if int(mask.sum()) == 0:
            continue
        specialist_prob = model.predict_proba(x_matrix[mask])[:, 1].astype(np.float32)
        out[mask] = ((1.0 - a) * out[mask]) + (a * specialist_prob)
    return out


def save_artifacts(
    momentum_model: nn.Module,
    seq_scaler: Any,
    static_scaler: Any,
    xgb_model: XGBClassifier,
    specialist_models: dict[str, XGBClassifier],
    metrics_report: dict[str, Any],
    args: argparse.Namespace,
    momentum_config: dict[str, Any],
) -> None:
    args.momentum_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.momentum_scaler_path.parent.mkdir(parents=True, exist_ok=True)
    args.xgb_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.xgb_specialists_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": momentum_model.state_dict(),
            "config": momentum_config,
        },
        args.momentum_model_path,
    )
    with args.momentum_scaler_path.open("wb") as f:
        pickle.dump(
            {
                "seq_scaler": seq_scaler,
                "static_scaler": static_scaler,
                "config": momentum_config,
            },
            f,
        )
    xgb_model.save_model(str(args.xgb_model_path))
    with args.metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)
    with args.xgb_specialists_path.open("wb") as f:
        pickle.dump(
            {
                "blend_alpha": float(args.specialist_blend_alpha),
                "models": specialist_models,
            },
            f,
        )

    logging.info("Saved momentum model: %s", args.momentum_model_path)
    logging.info("Saved momentum scalers: %s", args.momentum_scaler_path)
    logging.info("Saved XGBoost model: %s", args.xgb_model_path)
    logging.info("Saved XGBoost specialists: %s", args.xgb_specialists_path)
    logging.info("Saved metrics: %s", args.metrics_path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    set_seed(args.seed)
    device = resolve_device(args.device)
    logging.info("Using device: %s", device)
    if args.oof_folds < 1:
        raise ValueError("--oof-folds must be >= 1")
    if args.oof_min_train_fights < 1:
        raise ValueError("--oof-min-train-fights must be >= 1")
    if args.symmetry_loss_weight < 0:
        raise ValueError("--symmetry-loss-weight must be >= 0")
    if not (0.0 <= args.specialist_blend_alpha <= 1.0):
        raise ValueError("--specialist-blend-alpha must be within [0, 1]")
    if args.specialist_min_train_samples < 1 or args.specialist_min_val_samples < 1:
        raise ValueError("--specialist-min-train-samples and --specialist-min-val-samples must be >= 1")

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

    oof_fold_reports: list[dict[str, Any]] = []
    momentum_train_oof_prob: np.ndarray | None = None
    train_stack_mask: np.ndarray | None = None
    momentum_walkforward_summary: dict[str, Any] = {
        "fold_count": 0,
        "mean_auc": float("nan"),
        "std_auc": float("nan"),
        "selection_score": float("nan"),
    }
    if args.use_oof_stacking:
        momentum_train_oof_prob, train_stack_mask, oof_fold_reports = build_oof_momentum_predictions(
            train_samples,
            num_train_fights=len(train_df),
            seq_len=seq_len,
            args=args,
            device=device,
        )
        momentum_walkforward_summary = summarize_walkforward_fold_metrics(
            oof_fold_reports,
            std_penalty=float(args.walkforward_std_penalty),
        )
        if momentum_walkforward_summary["fold_count"] > 0:
            logging.info(
                "Momentum walk-forward summary | folds=%d | mean_auc=%.4f | std_auc=%.4f | score=%.4f",
                momentum_walkforward_summary["fold_count"],
                momentum_walkforward_summary["mean_auc"],
                momentum_walkforward_summary["std_auc"],
                momentum_walkforward_summary["selection_score"],
            )
    else:
        logging.warning(
            "OOF stacking disabled; XGBoost will train on in-sample momentum scores (leakage-prone)."
        )

    seq_scaler, static_scaler = fit_scalers(train_samples)
    train_data = transform_samples(train_samples, seq_scaler, static_scaler, seq_len)
    val_data = transform_samples(val_samples, seq_scaler, static_scaler, seq_len)
    test_data = transform_samples(test_samples, seq_scaler, static_scaler, seq_len)

    seq_dim = train_data[0].seq_a.shape[1]
    momentum_model, momentum_threshold, momentum_summary = train_momentum_model(
        train_samples=train_data,
        val_samples=val_data,
        seq_dim=seq_dim,
        args=args,
        device=device,
    )

    momentum_train_prob, momentum_train_true = predict_momentum(
        momentum_model,
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    momentum_val_prob, momentum_val_true = predict_momentum(
        momentum_model,
        val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    momentum_test_prob, momentum_test_true = predict_momentum(
        momentum_model,
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    momentum_test_metrics = evaluate_probs(momentum_test_true, momentum_test_prob, momentum_threshold)
    logging.info(
        "Momentum-only test metrics @ threshold %.3f: %s",
        momentum_threshold,
        momentum_test_metrics,
    )

    x_train_static, y_train = build_oriented_static_matrix(
        train_df,
        f1_raw=train_f1,
        f2_raw=train_f2,
        trend_ema_alpha=float(args.trend_ema_alpha),
        use_trend_static_features=bool(args.use_trend_static_features),
        use_enhanced_context_static_features=bool(args.use_enhanced_context_static_features),
    )
    x_val_static, y_val = build_oriented_static_matrix(
        val_df,
        f1_raw=val_f1,
        f2_raw=val_f2,
        trend_ema_alpha=float(args.trend_ema_alpha),
        use_trend_static_features=bool(args.use_trend_static_features),
        use_enhanced_context_static_features=bool(args.use_enhanced_context_static_features),
    )
    x_test_static, y_test = build_oriented_static_matrix(
        test_df,
        f1_raw=test_f1,
        f2_raw=test_f2,
        trend_ema_alpha=float(args.trend_ema_alpha),
        use_trend_static_features=bool(args.use_trend_static_features),
        use_enhanced_context_static_features=bool(args.use_enhanced_context_static_features),
    )
    weight_train_full = build_oriented_weight_classes(train_df)
    weight_val = build_oriented_weight_classes(val_df)
    weight_test = build_oriented_weight_classes(test_df)

    if (
        len(y_train) != len(momentum_train_prob)
        or len(y_val) != len(momentum_val_prob)
        or len(y_test) != len(momentum_test_prob)
    ):
        raise ValueError("Momentum predictions and static matrices are misaligned in length.")

    if args.use_oof_stacking:
        if momentum_train_oof_prob is None or train_stack_mask is None:
            raise ValueError("OOF predictions were not generated.")
        if len(momentum_train_oof_prob) != len(y_train):
            raise ValueError("OOF momentum predictions are misaligned with train labels.")
        train_stack_prob = momentum_train_oof_prob
        stack_mask = train_stack_mask
    else:
        train_stack_prob = momentum_train_prob
        stack_mask = np.ones(len(y_train), dtype=bool)

    x_train_full = np.concatenate([x_train_static, train_stack_prob.reshape(-1, 1)], axis=1)
    x_train = x_train_full[stack_mask]
    y_train_stacked = y_train[stack_mask]
    weight_train = weight_train_full[stack_mask]
    if len(np.unique(y_train_stacked)) < 2:
        raise ValueError("XGBoost training labels are degenerate after applying stack mask.")
    logging.info(
        "Stacked train samples for XGBoost: %d/%d (%.2f%%)",
        int(stack_mask.sum()),
        int(len(stack_mask)),
        100.0 * float(stack_mask.mean()),
    )

    x_val = np.concatenate([x_val_static, momentum_val_prob.reshape(-1, 1)], axis=1)
    x_test = np.concatenate([x_test_static, momentum_test_prob.reshape(-1, 1)], axis=1)
    xgb_model = fit_xgb_model(
        x_train=x_train,
        y_train=y_train_stacked,
        x_val=x_val,
        y_val=y_val,
        args=args,
        random_state=args.seed,
    )
    specialist_models, specialist_reports = fit_specialist_models(
        x_train=x_train,
        y_train=y_train_stacked,
        weight_train=weight_train,
        x_val=x_val,
        y_val=y_val,
        weight_val=weight_val,
        args=args,
    )

    walkforward_cv_report: dict[str, Any] = {"enabled": False, "reason": "disabled"}
    if args.use_walkforward_cv and args.use_oof_stacking and momentum_train_oof_prob is not None:
        walkforward_cv_report = build_walkforward_ensemble_cv(
            x_train_static=x_train_static,
            y_train=y_train,
            oof_train_prob=momentum_train_oof_prob,
            oof_fold_reports=oof_fold_reports,
            args=args,
            oof_pred_output_path=args.oof_ensemble_pred_path,
        )

    val_prob_global = xgb_model.predict_proba(x_val)[:, 1].astype(np.float32)
    test_prob_global = xgb_model.predict_proba(x_test)[:, 1].astype(np.float32)
    val_prob_xgb = blend_specialist_predictions(
        base_prob=val_prob_global,
        x_matrix=x_val,
        weight_classes=weight_val,
        specialist_models=specialist_models,
        alpha=float(args.specialist_blend_alpha),
    )
    test_prob_xgb = blend_specialist_predictions(
        base_prob=test_prob_global,
        x_matrix=x_test,
        weight_classes=weight_test,
        specialist_models=specialist_models,
        alpha=float(args.specialist_blend_alpha),
    )

    ensemble_threshold, val_threshold_metrics = choose_best_threshold(y_val.astype(np.float32), val_prob_xgb)
    test_metrics = evaluate_probs(y_test.astype(np.float32), test_prob_xgb, ensemble_threshold)
    val_metrics_global = evaluate_probs(y_val.astype(np.float32), val_prob_global, threshold=0.5)
    test_metrics_global = evaluate_probs(y_test.astype(np.float32), test_prob_global, threshold=0.5)
    logging.info(
        "Ensemble test metrics @ threshold %.3f: %s | specialists=%d",
        ensemble_threshold,
        test_metrics,
        len(specialist_models),
    )

    momentum_config = {
        "seq_len": int(seq_len),
        "raw_num_stats": int(num_stats),
        "seq_dim": int(seq_dim),
        "hidden_size": int(args.hidden_size),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "bidirectional": bool(args.bidirectional),
        "use_cross_attention": bool(args.use_cross_attention),
        "attention_heads": int(args.attention_heads),
        "attention_dropout": float(args.attention_dropout),
        "threshold": float(momentum_threshold),
        "static_recency_mode": str(args.static_recency_mode),
        "ema_alpha": float(args.ema_alpha),
        "symmetry_loss_weight": float(args.symmetry_loss_weight),
    }
    xgb_config = {
        "n_estimators": int(args.xgb_n_estimators),
        "learning_rate": float(args.xgb_lr),
        "max_depth": int(args.xgb_max_depth),
        "min_child_weight": float(args.xgb_min_child_weight),
        "subsample": float(args.xgb_subsample),
        "colsample_bytree": float(args.xgb_colsample_bytree),
        "reg_alpha": float(args.xgb_reg_alpha),
        "reg_lambda": float(args.xgb_reg_lambda),
        "gamma": float(args.xgb_gamma),
        "xgb_early_stopping": int(args.xgb_early_stopping),
        "use_oof_stacking": bool(args.use_oof_stacking),
        "oof_folds": int(args.oof_folds),
        "oof_min_train_fights": int(args.oof_min_train_fights),
        "use_walkforward_cv": bool(args.use_walkforward_cv),
        "walkforward_std_penalty": float(args.walkforward_std_penalty),
        "use_weight_class_specialists": bool(args.use_weight_class_specialists),
        "specialist_min_train_samples": int(args.specialist_min_train_samples),
        "specialist_min_val_samples": int(args.specialist_min_val_samples),
        "specialist_blend_alpha": float(args.specialist_blend_alpha),
        "trend_ema_alpha": float(args.trend_ema_alpha),
        "use_trend_static_features": bool(args.use_trend_static_features),
        "use_enhanced_context_static_features": bool(args.use_enhanced_context_static_features),
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
        "holdout": holdout_info,
        "oof_stacking": {
            "enabled": bool(args.use_oof_stacking),
            "samples_used_for_xgb_train": int(stack_mask.sum()),
            "samples_dropped_for_xgb_train": int(len(stack_mask) - int(stack_mask.sum())),
            "coverage": float(stack_mask.mean()),
            "fold_reports": oof_fold_reports,
        },
        "momentum_train_summary": momentum_summary,
        "momentum_walkforward_summary": momentum_walkforward_summary,
        "momentum_test_metrics": momentum_test_metrics,
        "walkforward_cv": walkforward_cv_report,
        "ensemble_val_metrics_global_threshold_0_5": val_metrics_global,
        "ensemble_test_metrics_global_threshold_0_5": test_metrics_global,
        "ensemble_val_metrics_at_best_threshold": val_threshold_metrics,
        "ensemble_test_metrics": test_metrics,
        "ensemble_threshold": float(ensemble_threshold),
        "specialists": {
            "enabled": bool(args.use_weight_class_specialists),
            "count": int(len(specialist_models)),
            "blend_alpha": float(args.specialist_blend_alpha),
            "reports": specialist_reports,
            "train_weight_class_counts": {
                str(k): int(v)
                for k, v in pd.Series(weight_train_full).value_counts().sort_index().to_dict().items()
            },
            "val_weight_class_counts": {
                str(k): int(v) for k, v in pd.Series(weight_val).value_counts().sort_index().to_dict().items()
            },
            "test_weight_class_counts": {
                str(k): int(v) for k, v in pd.Series(weight_test).value_counts().sort_index().to_dict().items()
            },
        },
        "momentum_config": momentum_config,
        "xgb_config": xgb_config,
        "momentum_model_path": str(args.momentum_model_path),
        "momentum_scaler_path": str(args.momentum_scaler_path),
        "xgb_model_path": str(args.xgb_model_path),
        "xgb_specialists_path": str(args.xgb_specialists_path),
    }
    save_artifacts(
        momentum_model=momentum_model,
        seq_scaler=seq_scaler,
        static_scaler=static_scaler,
        xgb_model=xgb_model,
        specialist_models=specialist_models,
        metrics_report=metrics_report,
        args=args,
        momentum_config=momentum_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
