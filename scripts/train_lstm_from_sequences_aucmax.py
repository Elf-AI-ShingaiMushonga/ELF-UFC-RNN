#!/usr/bin/env python3
"""Train an AUC-optimized Siamese LSTM on prebuilt UFC sequence rows."""

from __future__ import annotations

import argparse
import copy
import logging
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_lstm_from_sequences as base


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AUCMaxSiameseFightLSTM(nn.Module):
    """Siamese LSTM with cross-attention + learned temporal pooling."""

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
        fusion_dim: int,
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

        attn_hidden = max(16, self.seq_embed_dim // 2)
        self.pool_score = nn.Sequential(
            nn.Linear(self.seq_embed_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

        # mean + max + learned-attention pooling
        rep_dim = self.seq_embed_dim * 3

        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
        )

        head_in = (rep_dim * 4) + 32
        self.fusion_in = nn.Sequential(
            nn.Linear(head_in, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block1 = ResidualMLPBlock(fusion_dim, dropout)
        self.block2 = ResidualMLPBlock(fusion_dim, dropout)
        self.out = nn.Linear(fusion_dim, 1)

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

    def temporal_pool(self, out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        mean_pool = (out * mask_f).sum(dim=1) / denom
        max_pool = out.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values

        scores = self.pool_score(out).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        attn_pool = (out * weights.unsqueeze(-1)).sum(dim=1)

        return torch.cat([mean_pool, max_pool, attn_pool], dim=1)

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

        rep_a = self.temporal_pool(out_a, mask_a)
        rep_b = self.temporal_pool(out_b, mask_b)
        static_rep = self.static_net(static)

        x = torch.cat([rep_a, rep_b, rep_a - rep_b, rep_a * rep_b, static_rep], dim=1)
        x = self.fusion_in(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.out(x).squeeze(1)


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        if self.decay <= 0.0:
            return
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if self.decay <= 0.0:
            return
        one_minus = 1.0 - self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=one_minus)

    def apply(self, model: nn.Module) -> None:
        if self.decay <= 0.0:
            return
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module) -> None:
        if self.decay <= 0.0:
            return
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.backup:
                continue
            param.data.copy_(self.backup[name].data)
        self.backup = {}

    @contextmanager
    def average_parameters(self, model: nn.Module):
        self.apply(model)
        try:
            yield
        finally:
            self.restore(model)


def pairwise_auc_surrogate(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_pairs: int,
) -> torch.Tensor:
    """Differentiable ranking surrogate that pushes positive logits above negatives."""
    pos = logits[targets > 0.5]
    neg = logits[targets <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_tensor(0.0)

    total_pairs = int(pos.numel() * neg.numel())
    if total_pairs <= max_pairs:
        diffs = pos.unsqueeze(1) - neg.unsqueeze(0)
        return F.softplus(-diffs).mean()

    sample_size = max(1, int(max_pairs))
    pos_idx = torch.randint(0, pos.numel(), (sample_size,), device=logits.device)
    neg_idx = torch.randint(0, neg.numel(), (sample_size,), device=logits.device)
    diffs = pos[pos_idx] - neg[neg_idx]
    return F.softplus(-diffs).mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float,
    auc_loss_weight: float,
    auc_pair_limit: int,
    ema: Optional[ExponentialMovingAverage] = None,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    losses: list[float] = []
    bce_losses: list[float] = []
    auc_losses: list[float] = []
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
        bce = criterion(logits, target)
        auc_rank = pairwise_auc_surrogate(logits, target, max_pairs=auc_pair_limit)
        loss = bce + (auc_loss_weight * auc_rank)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)

        losses.append(float(loss.item()))
        bce_losses.append(float(bce.item()))
        auc_losses.append(float(auc_rank.item()))
        probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
        targets_all.append(target.detach().cpu().numpy())

    probs = np.concatenate(probs_all) if probs_all else np.zeros(0, dtype=np.float32)
    targets = np.concatenate(targets_all) if targets_all else np.zeros(0, dtype=np.float32)
    return (
        float(np.mean(losses) if losses else 0.0),
        float(np.mean(bce_losses) if bce_losses else 0.0),
        float(np.mean(auc_losses) if auc_losses else 0.0),
        probs,
        targets,
    )


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train an AUC-optimized Siamese LSTM on UFC sequence rows."
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
        default=root_dir / "champion_lstm_model_aucmax.pth",
        help="Path to save model checkpoint.",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=root_dir / "data" / "model_cache" / "lstm_sequence_scalers_aucmax.pkl",
        help="Path to save fitted scalers.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=root_dir / "data" / "model_cache" / "lstm_sequence_metrics_aucmax.json",
        help="Path to save train/eval metrics report.",
    )
    parser.add_argument("--epochs", type=int, default=120, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stop patience.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--hidden-size", type=int, default=48, help="LSTM hidden size.")
    parser.add_argument("--num-layers", type=int, default=3, help="LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.65, help="Dropout rate.")
    parser.add_argument("--lr", type=float, default=5e-4, help="AdamW learning rate.")
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=4,
        help="Linear LR warmup epochs before cosine scheduling.",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=8,
        help="Minimum epochs to train before early stopping can trigger.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum improvement in score to reset early-stop patience.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
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
        default=0.75,
        help="EMA alpha for static recency features (used when mode=ema).",
    )
    parser.add_argument(
        "--auc-loss-weight",
        type=float,
        default=0.35,
        help="Weight for pairwise AUC ranking loss added to BCE.",
    )
    parser.add_argument(
        "--auc-pair-limit",
        type=int,
        default=4096,
        help="Maximum sampled positive-negative pairs per batch for ranking loss.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.998,
        help="EMA decay for model weights; set <=0 to disable.",
    )
    parser.add_argument(
        "--eval-use-ema",
        dest="eval_use_ema",
        action="store_true",
        help="Evaluate on EMA weights when EMA is enabled (default).",
    )
    parser.add_argument(
        "--no-eval-use-ema",
        dest="eval_use_ema",
        action="store_false",
        help="Evaluate on raw training weights.",
    )
    parser.set_defaults(eval_use_ema=True)
    parser.add_argument(
        "--fusion-dim",
        type=int,
        default=320,
        help="Width of residual fusion head.",
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


def train_model(
    train_samples: list[base.TransformedSample],
    val_samples: list[base.TransformedSample],
    seq_dim: int,
    static_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, float, dict[str, Any]]:
    model = AUCMaxSiameseFightLSTM(
        seq_dim=seq_dim,
        static_dim=static_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        use_cross_attention=args.use_cross_attention,
        attention_heads=args.attention_heads,
        attention_dropout=args.attention_dropout,
        fusion_dim=args.fusion_dim,
    ).to(device)

    pos_weight = base.class_pos_weight(train_samples)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    base_lr = float(args.lr)
    warmup_epochs = max(0, int(args.warmup_epochs))
    min_epochs = max(1, int(args.min_epochs))
    min_delta = float(max(args.min_delta, 0.0))
    cosine_epochs = max(1, int(args.epochs - warmup_epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=max(base_lr * 0.02, 1e-6),
    )

    train_loader = DataLoader(
        base.FightDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        base.FightDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ema = ExponentialMovingAverage(model, args.ema_decay) if args.ema_decay > 0 else None
    use_ema_eval = bool(ema is not None and args.eval_use_ema)

    best_state: Optional[dict[str, Any]] = None
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_scale = float(epoch) / float(warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = max(base_lr * warmup_scale, 1e-7)

        train_loss, train_bce, train_rank, _, _ = run_epoch(
            model,
            train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            auc_loss_weight=args.auc_loss_weight,
            auc_pair_limit=args.auc_pair_limit,
            ema=ema,
        )

        if use_ema_eval:
            with ema.average_parameters(model):
                val_loss, val_bce, val_rank, val_prob, val_true = run_epoch(
                    model,
                    val_loader,
                    criterion=criterion,
                    optimizer=None,
                    device=device,
                    grad_clip=args.grad_clip,
                    auc_loss_weight=args.auc_loss_weight,
                    auc_pair_limit=args.auc_pair_limit,
                    ema=None,
                )
                candidate_state = copy.deepcopy(model.state_dict())
        else:
            val_loss, val_bce, val_rank, val_prob, val_true = run_epoch(
                model,
                val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                grad_clip=args.grad_clip,
                auc_loss_weight=args.auc_loss_weight,
                auc_pair_limit=args.auc_pair_limit,
                ema=None,
            )
            candidate_state = copy.deepcopy(model.state_dict())

        if epoch > warmup_epochs:
            scheduler.step()

        val_auc = base.safe_auc(val_true, val_prob)
        val_bal = base.evaluate_probs(val_true, val_prob, threshold=0.5)["balanced_accuracy"]
        score = val_auc if not math.isnan(val_auc) else val_bal

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_bce": train_bce,
                "train_rank_loss": train_rank,
                "val_loss": val_loss,
                "val_bce": val_bce,
                "val_rank_loss": val_rank,
                "val_auc": val_auc,
                "val_balanced_accuracy": val_bal,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        logging.info(
            "Epoch %03d | train_loss=%.4f (bce=%.4f rank=%.4f) | "
            "val_loss=%.4f (bce=%.4f rank=%.4f) | val_auc=%.4f | val_bal=%.4f | lr=%.6f",
            epoch,
            train_loss,
            train_bce,
            train_rank,
            val_loss,
            val_bce,
            val_rank,
            val_auc,
            val_bal,
            optimizer.param_groups[0]["lr"],
        )

        if score > (best_score + min_delta):
            best_score = score
            best_state = candidate_state
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epoch >= min_epochs and epochs_without_improvement >= args.patience:
                logging.info("Early stopping at epoch %d (best epoch %d).", epoch, best_epoch)
                break

    if best_state is None:
        if use_ema_eval:
            with ema.average_parameters(model):
                best_state = copy.deepcopy(model.state_dict())
        else:
            best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(history)
        best_score = float("nan")

    model.load_state_dict(best_state)

    val_loss_final, val_bce_final, val_rank_final, val_prob_final, val_true_final = run_epoch(
        model,
        val_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        grad_clip=args.grad_clip,
        auc_loss_weight=args.auc_loss_weight,
        auc_pair_limit=args.auc_pair_limit,
        ema=None,
    )
    threshold, val_threshold_metrics = base.choose_best_threshold(val_true_final, val_prob_final)

    summary = {
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "pos_weight": float(pos_weight),
        "threshold": float(threshold),
        "val_metrics_at_best_threshold": val_threshold_metrics,
        "val_loss_final": float(val_loss_final),
        "val_bce_final": float(val_bce_final),
        "val_rank_loss_final": float(val_rank_final),
        "history": history,
    }
    return model, threshold, summary


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    base.set_seed(args.seed)
    device = base.resolve_device(args.device)
    logging.info("Using device: %s", device)

    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be >= 0")
    if args.min_epochs < 1:
        raise ValueError("--min-epochs must be >= 1")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be >= 0")
    if args.auc_loss_weight < 0:
        raise ValueError("--auc-loss-weight must be >= 0")
    if args.auc_pair_limit < 1:
        raise ValueError("--auc-pair-limit must be >= 1")
    if args.ema_decay < 0 or args.ema_decay >= 1:
        raise ValueError("--ema-decay must be in [0, 1)")
    if args.min_epochs > args.epochs:
        logging.warning("min_epochs (%d) > epochs (%d); clamping to epochs.", args.min_epochs, args.epochs)
        args.min_epochs = args.epochs

    logging.info(
        "Options | bidirectional=%s | cross_attention=%s | recency_mode=%s | ema_alpha=%.3f | "
        "auc_loss_weight=%.3f | ema_decay=%.4f",
        args.bidirectional,
        args.use_cross_attention,
        args.static_recency_mode,
        args.ema_alpha,
        args.auc_loss_weight,
        args.ema_decay,
    )

    df, seq_len, num_stats, f1_cols, f2_cols = base.load_dataframe(
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

    train_df, val_df, test_df = base.chronological_split(df, args.val_fraction, args.test_fraction)
    logging.info(
        "Fight splits -> train: %d | val: %d | test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    train_f1, train_f2 = base.frame_to_raw_sequences(train_df, seq_len, num_stats, f1_cols, f2_cols)
    val_f1, val_f2 = base.frame_to_raw_sequences(val_df, seq_len, num_stats, f1_cols, f2_cols)
    test_f1, test_f2 = base.frame_to_raw_sequences(test_df, seq_len, num_stats, f1_cols, f2_cols)

    train_samples = base.build_augmented_samples(
        train_df,
        train_f1,
        train_f2,
        static_recency_mode=args.static_recency_mode,
        ema_alpha=args.ema_alpha,
    )
    val_samples = base.build_augmented_samples(
        val_df,
        val_f1,
        val_f2,
        static_recency_mode=args.static_recency_mode,
        ema_alpha=args.ema_alpha,
    )
    test_samples = base.build_augmented_samples(
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

    seq_scaler, static_scaler = base.fit_scalers(train_samples)
    train_data = base.transform_samples(train_samples, seq_scaler, static_scaler, seq_len)
    val_data = base.transform_samples(val_samples, seq_scaler, static_scaler, seq_len)
    test_data = base.transform_samples(test_samples, seq_scaler, static_scaler, seq_len)

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
        base.FightDataset(test_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    criterion = nn.BCEWithLogitsLoss()
    _, _, _, test_prob, test_true = run_epoch(
        model,
        test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        grad_clip=args.grad_clip,
        auc_loss_weight=args.auc_loss_weight,
        auc_pair_limit=args.auc_pair_limit,
        ema=None,
    )
    test_metrics = base.evaluate_probs(test_true, test_prob, threshold)
    logging.info("Test metrics @ threshold %.3f: %s", threshold, test_metrics)

    config = {
        "variant": "aucmax_pairwise_rank",
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
        "fusion_dim": int(args.fusion_dim),
        "static_recency_mode": str(args.static_recency_mode),
        "ema_alpha": float(args.ema_alpha),
        "dropout": float(args.dropout),
        "warmup_epochs": int(args.warmup_epochs),
        "min_epochs": int(args.min_epochs),
        "min_delta": float(args.min_delta),
        "auc_loss_weight": float(args.auc_loss_weight),
        "auc_pair_limit": int(args.auc_pair_limit),
        "ema_decay": float(args.ema_decay),
        "eval_use_ema": bool(args.eval_use_ema),
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
    base.save_artifacts(
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
