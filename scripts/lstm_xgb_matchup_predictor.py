#!/usr/bin/env python3
"""Matchup inference using trained LSTM momentum + XGBoost ensemble artifacts."""

from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_fight_history_sequences import (
    DEFAULT_PERFORMANCE_STATS,
    build_sequences,
    merge_optional_metadata,
    prepare_dataframe,
)
from train_lstm_from_sequences import (
    build_augmented_samples,
    frame_to_raw_sequences,
    transform_samples,
)
from train_lstm_xgboost_ensemble import (
    MomentumSiameseLSTM,
    blend_specialist_predictions,
    build_oriented_static_matrix,
    build_oriented_weight_classes,
    predict_momentum,
)


POSITIVE_LABEL = "fighter_1_win"
PROJECT_ROOT = SCRIPT_DIR.parent


@dataclass
class FighterSnapshot:
    fighter_id: str
    fighter_name: str
    last_event_date: pd.Timestamp
    age_days: float
    height_cm: float
    reach_cm: float
    weight_class: str
    gender: str


class LSTMXGBMatchupPredictor:
    def __init__(
        self,
        *,
        raw_csv_path: Path,
        metadata_csv_path: Path,
        metrics_path: Path,
        momentum_model_path: Path,
        momentum_scaler_path: Path,
        xgb_model_path: Path,
        xgb_specialists_path: Path | None = None,
    ) -> None:
        self.raw_csv_path = Path(raw_csv_path)
        self.metadata_csv_path = Path(metadata_csv_path)
        self.metrics_path = Path(metrics_path)
        self.momentum_model_path = Path(momentum_model_path)
        self.momentum_scaler_path = Path(momentum_scaler_path)
        self.xgb_model_path = Path(xgb_model_path)
        self.xgb_specialists_path = Path(xgb_specialists_path) if xgb_specialists_path else None
        self.device = torch.device("cpu")

        self.metrics = self._load_metrics()
        self.momentum_config = dict(self.metrics.get("momentum_config", {}))
        self.xgb_config = dict(self.metrics.get("xgb_config", {}))
        self.seq_len = int(self.momentum_config.get("seq_len", 5))
        self.raw_num_stats = int(self.momentum_config.get("raw_num_stats", len(DEFAULT_PERFORMANCE_STATS)))
        self.static_recency_mode = str(self.momentum_config.get("static_recency_mode", "ema"))
        self.ema_alpha = float(self.momentum_config.get("ema_alpha", 0.75))
        self.trend_ema_alpha = float(self.xgb_config.get("trend_ema_alpha", 0.70))
        self.use_trend_static_features = bool(self.xgb_config.get("use_trend_static_features", True))
        self.use_enhanced_context_static_features = bool(
            self.xgb_config.get("use_enhanced_context_static_features", False)
        )

        self.f1_seq_cols, self.f2_seq_cols = self._build_sequence_layout(self.seq_len, self.raw_num_stats)

        self.seq_scaler, self.static_scaler = self._load_scalers()
        self.momentum_model = self._load_momentum_model()
        self.xgb_model = self._load_xgb_model()
        self.specialist_blend_alpha, self.specialist_models = self._load_specialists()

        self.prepared_df, self.f1_stat_columns, self.f2_stat_columns = self._load_prepared_dataframe()
        self.fighters_by_name = self._build_fighter_snapshots(self.prepared_df)

    def _load_metrics(self) -> dict[str, Any]:
        if not self.metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_path}")
        return json.loads(self.metrics_path.read_text(encoding="utf-8"))

    def _load_scalers(self) -> tuple[Any, Any]:
        if not self.momentum_scaler_path.exists():
            raise FileNotFoundError(f"Momentum scaler file not found: {self.momentum_scaler_path}")
        with self.momentum_scaler_path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Invalid momentum scaler payload.")
        seq_scaler = payload.get("seq_scaler")
        static_scaler = payload.get("static_scaler")
        if seq_scaler is None or static_scaler is None:
            raise ValueError("Momentum scalers payload missing seq/static scalers.")
        return seq_scaler, static_scaler

    def _load_momentum_model(self) -> MomentumSiameseLSTM:
        if not self.momentum_model_path.exists():
            raise FileNotFoundError(f"Momentum model file not found: {self.momentum_model_path}")

        checkpoint = torch.load(self.momentum_model_path, map_location=self.device)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError("Invalid momentum checkpoint format.")

        cfg = dict(checkpoint.get("config") or self.momentum_config)
        model = MomentumSiameseLSTM(
            seq_dim=int(cfg.get("seq_dim", self.momentum_config.get("seq_dim", 24))),
            hidden_size=int(cfg.get("hidden_size", self.momentum_config.get("hidden_size", 128))),
            num_layers=int(cfg.get("num_layers", self.momentum_config.get("num_layers", 2))),
            dropout=float(cfg.get("dropout", self.momentum_config.get("dropout", 0.4))),
            bidirectional=bool(cfg.get("bidirectional", self.momentum_config.get("bidirectional", True))),
            use_cross_attention=bool(
                cfg.get("use_cross_attention", self.momentum_config.get("use_cross_attention", True))
            ),
            attention_heads=int(cfg.get("attention_heads", self.momentum_config.get("attention_heads", 4))),
            attention_dropout=float(
                cfg.get("attention_dropout", self.momentum_config.get("attention_dropout", 0.10))
            ),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def _load_xgb_model(self) -> XGBClassifier:
        if not self.xgb_model_path.exists():
            raise FileNotFoundError(f"XGBoost model file not found: {self.xgb_model_path}")
        model = XGBClassifier()
        model.load_model(str(self.xgb_model_path))
        return model

    def _load_specialists(self) -> tuple[float, dict[str, Any]]:
        if self.xgb_specialists_path is None:
            return 0.0, {}
        if not self.xgb_specialists_path.exists():
            return 0.0, {}
        with self.xgb_specialists_path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            return 0.0, {}
        models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
        alpha = float(payload.get("blend_alpha", self.xgb_config.get("specialist_blend_alpha", 0.0)))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return alpha, models

    @staticmethod
    def _build_sequence_layout(seq_len: int, num_stats: int) -> tuple[list[str], list[str]]:
        f1_cols = [f"f1_seq_{step}_stat_{idx}" for step in range(seq_len) for idx in range(num_stats)]
        f2_cols = [f"f2_seq_{step}_stat_{idx}" for step in range(seq_len) for idx in range(num_stats)]
        return f1_cols, f2_cols

    def _load_prepared_dataframe(self) -> tuple[pd.DataFrame, list[str], list[str]]:
        if not self.raw_csv_path.exists():
            raise FileNotFoundError(f"Raw fight CSV not found: {self.raw_csv_path}")

        raw_df = pd.read_csv(self.raw_csv_path)
        enriched_df = merge_optional_metadata(raw_df, self.metadata_csv_path)
        prepared_df, f1_stat_columns, f2_stat_columns = prepare_dataframe(
            enriched_df,
            performance_stats=list(DEFAULT_PERFORMANCE_STATS),
            drop_nonstandard_outcomes=False,
        )
        prepared_df = prepared_df.sort_values(["event_date", "bout_index", "fight_id"]).reset_index(drop=True)
        return prepared_df, f1_stat_columns, f2_stat_columns

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        if pd.isna(value):
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _build_fighter_snapshots(self, df: pd.DataFrame) -> dict[str, FighterSnapshot]:
        snapshots: dict[str, FighterSnapshot] = {}
        for row in df.itertuples(index=False):
            event_date = pd.Timestamp(getattr(row, "event_date"))
            weight_class = str(getattr(row, "weight_class", "") or "").strip()
            gender = str(getattr(row, "gender", "") or "").strip()
            for side in (1, 2):
                fighter_name = str(getattr(row, f"fighter_{side}_name", "") or "").strip()
                fighter_id = str(getattr(row, f"fighter_{side}_id", "") or "").strip()
                if not fighter_name or not fighter_id:
                    continue
                key = fighter_name.lower()
                snapshots[key] = FighterSnapshot(
                    fighter_id=fighter_id,
                    fighter_name=fighter_name,
                    last_event_date=event_date,
                    age_days=self._safe_float(getattr(row, f"fighter_{side}_age_days", 0.0), 0.0),
                    height_cm=self._safe_float(getattr(row, f"fighter_{side}_height_cm", 0.0), 0.0),
                    reach_cm=self._safe_float(getattr(row, f"fighter_{side}_reach_cm", 0.0), 0.0),
                    weight_class=weight_class,
                    gender=gender,
                )
        return snapshots

    def _resolve_fighter(self, fighter_name: str) -> FighterSnapshot:
        key = str(fighter_name).strip().lower()
        if key not in self.fighters_by_name:
            raise ValueError(f"Unknown fighter: {fighter_name}")
        return self.fighters_by_name[key]

    @staticmethod
    def _infer_context_value(primary: str, fallback: str) -> str:
        p = str(primary or "").strip()
        if p:
            return p
        return str(fallback or "").strip()

    @staticmethod
    def _default_rounds(raw_rounds: Any) -> int:
        try:
            rounds = int(raw_rounds)
            if rounds > 0:
                return rounds
        except (TypeError, ValueError):
            pass
        return 3

    def _age_at_prediction(self, snapshot: FighterSnapshot, pred_date: pd.Timestamp) -> float:
        base_age = max(float(snapshot.age_days), 0.0)
        if base_age <= 0.0:
            return 0.0
        delta_days = max((pred_date - snapshot.last_event_date).days, 0)
        return float(base_age + float(delta_days))

    def _build_prediction_row(
        self,
        fighter_1: FighterSnapshot,
        fighter_2: FighterSnapshot,
        *,
        pred_date: pd.Timestamp,
        weight_class: str,
        gender: str,
        scheduled_rounds: int,
        is_title_bout: bool,
    ) -> tuple[pd.DataFrame, str]:
        row = {col: np.nan for col in self.prepared_df.columns}

        pred_id = (
            f"pred_{fighter_1.fighter_id}_{fighter_2.fighter_id}_{pred_date.strftime('%Y%m%d')}_{int(time.time() * 1000)}"
        )
        row["fight_id"] = pred_id
        row["event_date"] = pred_date
        row["bout_index"] = 999
        row["fighter_1_id"] = fighter_1.fighter_id
        row["fighter_2_id"] = fighter_2.fighter_id
        row["fighter_1_name"] = fighter_1.fighter_name
        row["fighter_2_name"] = fighter_2.fighter_name
        row["weight_class"] = weight_class
        row["gender"] = gender
        row["scheduled_rounds"] = int(max(scheduled_rounds, 1))
        row["is_title_bout"] = 1 if is_title_bout else 0
        row["outcome_label"] = POSITIVE_LABEL

        age_1 = self._age_at_prediction(fighter_1, pred_date)
        age_2 = self._age_at_prediction(fighter_2, pred_date)
        row["fighter_1_age_days"] = age_1
        row["fighter_2_age_days"] = age_2
        row["age_days_diff_f1_minus_f2"] = age_1 - age_2
        row["age_diff_years_f1_minus_f2"] = (age_1 - age_2) / 365.25 if (age_1 > 0 or age_2 > 0) else 0.0
        row["age_gap_over_5y"] = float(abs(float(row["age_diff_years_f1_minus_f2"])) >= 5.0)

        row["fighter_1_height_cm"] = max(float(fighter_1.height_cm), 0.0)
        row["fighter_2_height_cm"] = max(float(fighter_2.height_cm), 0.0)
        row["height_cm_diff_f1_minus_f2"] = float(row["fighter_1_height_cm"]) - float(row["fighter_2_height_cm"])

        row["fighter_1_reach_cm"] = max(float(fighter_1.reach_cm), 0.0)
        row["fighter_2_reach_cm"] = max(float(fighter_2.reach_cm), 0.0)
        row["reach_cm_diff_f1_minus_f2"] = float(row["fighter_1_reach_cm"]) - float(row["fighter_2_reach_cm"])

        for col in self.f1_stat_columns:
            row[col] = 0.0
        for col in self.f2_stat_columns:
            row[col] = 0.0

        return pd.DataFrame([row]), pred_id

    def predict_matchup(
        self,
        *,
        fighter_1_name: str,
        fighter_2_name: str,
        event_date: str | None = None,
        weight_class: str | None = None,
        gender: str | None = None,
        scheduled_rounds: int | None = None,
        is_title_bout: bool = False,
    ) -> dict[str, Any]:
        f1_name = str(fighter_1_name).strip()
        f2_name = str(fighter_2_name).strip()
        if not f1_name or not f2_name:
            raise ValueError("fighter_1_name and fighter_2_name are required.")
        if f1_name.lower() == f2_name.lower():
            raise ValueError("Choose two different fighters.")

        fighter_1 = self._resolve_fighter(f1_name)
        fighter_2 = self._resolve_fighter(f2_name)
        pred_date = pd.Timestamp(event_date).normalize() if event_date else pd.Timestamp.today().normalize()

        inferred_weight = self._infer_context_value(weight_class, fighter_1.weight_class)
        if not inferred_weight:
            inferred_weight = self._infer_context_value(fighter_2.weight_class, "UNKNOWN")
        inferred_gender = self._infer_context_value(gender, fighter_1.gender)
        if not inferred_gender:
            inferred_gender = self._infer_context_value(fighter_2.gender, "")
        use_rounds = self._default_rounds(scheduled_rounds)

        prediction_row_df, pred_fight_id = self._build_prediction_row(
            fighter_1,
            fighter_2,
            pred_date=pred_date,
            weight_class=inferred_weight,
            gender=inferred_gender,
            scheduled_rounds=use_rounds,
            is_title_bout=bool(is_title_bout),
        )

        history_df = self.prepared_df[self.prepared_df["event_date"] <= pred_date].copy()
        combined_df = pd.concat([history_df, prediction_row_df], ignore_index=True)
        combined_df = combined_df.sort_values(["event_date", "bout_index", "fight_id"]).reset_index(drop=True)

        seq_df = build_sequences(
            combined_df,
            sequence_length=self.seq_len,
            f1_stat_columns=self.f1_stat_columns,
            f2_stat_columns=self.f2_stat_columns,
            elo_base=1500.0,
            elo_k_factor=24.0,
            elo_scale=400.0,
        )
        target_df = seq_df[seq_df["fight_id"] == pred_fight_id].copy()
        if target_df.empty:
            raise RuntimeError("Failed to construct prediction sequence row.")
        target_df["weight_class"] = inferred_weight

        f1_raw, f2_raw = frame_to_raw_sequences(
            target_df,
            self.seq_len,
            self.raw_num_stats,
            self.f1_seq_cols,
            self.f2_seq_cols,
        )
        samples = build_augmented_samples(
            target_df,
            f1_raw,
            f2_raw,
            static_recency_mode=self.static_recency_mode,
            ema_alpha=self.ema_alpha,
        )
        transformed = transform_samples(samples, self.seq_scaler, self.static_scaler, self.seq_len)
        momentum_prob, _ = predict_momentum(
            self.momentum_model,
            transformed,
            batch_size=32,
            num_workers=0,
            device=self.device,
        )

        x_static, _ = build_oriented_static_matrix(
            target_df,
            f1_raw=f1_raw,
            f2_raw=f2_raw,
            trend_ema_alpha=self.trend_ema_alpha,
            use_trend_static_features=self.use_trend_static_features,
            use_enhanced_context_static_features=self.use_enhanced_context_static_features,
        )
        x_pred = np.concatenate([x_static, momentum_prob.reshape(-1, 1)], axis=1)
        xgb_prob_global = self.xgb_model.predict_proba(x_pred)[:, 1].astype(np.float32)

        weight_classes = build_oriented_weight_classes(target_df)
        xgb_prob = blend_specialist_predictions(
            base_prob=xgb_prob_global,
            x_matrix=x_pred,
            weight_classes=weight_classes,
            specialist_models=self.specialist_models,
            alpha=self.specialist_blend_alpha,
        )
        if xgb_prob.shape[0] < 2:
            raise RuntimeError("Expected AB/BA oriented predictions but got fewer than 2 rows.")

        p_ab = float(np.clip(xgb_prob[0], 0.0, 1.0))
        p_ba = float(np.clip(xgb_prob[1], 0.0, 1.0))
        p_fighter_1 = float(np.clip(0.5 * (p_ab + (1.0 - p_ba)), 0.0, 1.0))
        p_fighter_2 = float(1.0 - p_fighter_1)

        winner = fighter_1.fighter_name if p_fighter_1 >= p_fighter_2 else fighter_2.fighter_name
        confidence = float(abs(p_fighter_1 - p_fighter_2))

        return {
            "fighter_1": fighter_1.fighter_name,
            "fighter_2": fighter_2.fighter_name,
            "fighter_1_id": fighter_1.fighter_id,
            "fighter_2_id": fighter_2.fighter_id,
            "event_date": str(pred_date.date()),
            "weight_class": inferred_weight,
            "gender": inferred_gender,
            "scheduled_rounds": int(use_rounds),
            "is_title_bout": bool(is_title_bout),
            "model": "lstm_xgb_ensemble",
            "model_label": "LSTM+XGBoost Ensemble",
            "model_test_auc": float(self.metrics.get("ensemble_test_metrics", {}).get("auc", 0.0)),
            "p_fighter_1": p_fighter_1,
            "p_fighter_2": p_fighter_2,
            "winner": winner,
            "confidence": confidence,
            "orientation_prob_ab": p_ab,
            "orientation_prob_ba": p_ba,
        }


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


class AverageEnsembleMatchupPredictor:
    """Blend multiple trained LSTM+XGB predictors at inference time."""

    def __init__(
        self,
        *,
        raw_csv_path: Path,
        metadata_csv_path: Path,
        ensemble_metrics_path: Path,
        component_metrics_paths: list[Path],
        component_weights: list[float] | None = None,
    ) -> None:
        self.raw_csv_path = Path(raw_csv_path)
        self.metadata_csv_path = Path(metadata_csv_path)
        self.ensemble_metrics_path = Path(ensemble_metrics_path)
        self.component_metrics_paths = [Path(path) for path in component_metrics_paths]
        if not self.component_metrics_paths:
            raise ValueError("Average ensemble requires at least one component metrics file.")
        if component_weights is not None and len(component_weights) != len(self.component_metrics_paths):
            raise ValueError("component_weights must align with component_metrics_paths.")

        self.ensemble_metrics = self._load_ensemble_metrics()
        if component_weights is None:
            self.component_weights = [1.0 / float(len(self.component_metrics_paths))] * len(self.component_metrics_paths)
        else:
            weight_array = np.asarray(component_weights, dtype=np.float64)
            if np.any(weight_array < 0.0):
                raise ValueError("component_weights must be non-negative.")
            weight_sum = float(weight_array.sum())
            if weight_sum <= 0.0:
                raise ValueError("component_weights must sum to a positive value.")
            self.component_weights = (weight_array / weight_sum).astype(np.float64).tolist()
        self.component_predictors = [self._build_component_predictor(path) for path in self.component_metrics_paths]

    def _load_ensemble_metrics(self) -> dict[str, Any]:
        if not self.ensemble_metrics_path.exists():
            raise FileNotFoundError(f"Average ensemble metrics not found: {self.ensemble_metrics_path}")
        return json.loads(self.ensemble_metrics_path.read_text(encoding="utf-8"))

    def _build_component_predictor(self, metrics_path: Path) -> LSTMXGBMatchupPredictor:
        if not metrics_path.exists():
            raise FileNotFoundError(f"Component metrics not found: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        required = ["momentum_model_path", "momentum_scaler_path", "xgb_model_path"]
        missing = [key for key in required if not str(metrics.get(key, "")).strip()]
        if missing:
            raise ValueError(f"Component metrics {metrics_path} missing required keys: {missing}")

        specialists_raw = str(metrics.get("xgb_specialists_path", "")).strip()
        specialists_path = resolve_project_path(specialists_raw) if specialists_raw else None

        return LSTMXGBMatchupPredictor(
            raw_csv_path=self.raw_csv_path,
            metadata_csv_path=self.metadata_csv_path,
            metrics_path=metrics_path,
            momentum_model_path=resolve_project_path(str(metrics["momentum_model_path"])),
            momentum_scaler_path=resolve_project_path(str(metrics["momentum_scaler_path"])),
            xgb_model_path=resolve_project_path(str(metrics["xgb_model_path"])),
            xgb_specialists_path=specialists_path,
        )

    def predict_matchup(
        self,
        *,
        fighter_1_name: str,
        fighter_2_name: str,
        event_date: str | None = None,
        weight_class: str | None = None,
        gender: str | None = None,
        scheduled_rounds: int | None = None,
        is_title_bout: bool = False,
    ) -> dict[str, Any]:
        predictions = [
            predictor.predict_matchup(
                fighter_1_name=fighter_1_name,
                fighter_2_name=fighter_2_name,
                event_date=event_date,
                weight_class=weight_class,
                gender=gender,
                scheduled_rounds=scheduled_rounds,
                is_title_bout=is_title_bout,
            )
            for predictor in self.component_predictors
        ]

        base = dict(predictions[0])
        weight_array = np.asarray(self.component_weights, dtype=np.float64)
        p_fighter_1 = float(np.dot(weight_array, np.asarray([float(pred["p_fighter_1"]) for pred in predictions])))
        p_fighter_2 = float(1.0 - p_fighter_1)
        p_ab = float(np.dot(weight_array, np.asarray([float(pred["orientation_prob_ab"]) for pred in predictions])))
        p_ba = float(np.dot(weight_array, np.asarray([float(pred["orientation_prob_ba"]) for pred in predictions])))
        winner = base["fighter_1"] if p_fighter_1 >= p_fighter_2 else base["fighter_2"]
        blend_name = str(self.ensemble_metrics.get("ensemble_strategy", "average_probability_mean"))
        model_label = "Weighted LSTM+XGBoost Ensemble" if "weight" in blend_name else "Average LSTM+XGBoost Ensemble"

        base.update(
            {
                "model": "weighted_lstm_xgb_ensemble" if "weight" in blend_name else "average_lstm_xgb_ensemble",
                "model_label": model_label,
                "model_test_auc": float(
                    self.ensemble_metrics.get("ensemble_test_metrics", {}).get("auc", 0.0)
                ),
                "model_val_auc": float(
                    self.ensemble_metrics.get("ensemble_val_metrics_at_best_threshold", {}).get("auc", 0.0)
                ),
                "ensemble_size": int(len(self.component_predictors)),
                "component_metrics_paths": [str(path) for path in self.component_metrics_paths],
                "component_weights": self.component_weights,
                "ensemble_strategy": blend_name,
                "p_fighter_1": p_fighter_1,
                "p_fighter_2": p_fighter_2,
                "winner": winner,
                "confidence": float(abs(p_fighter_1 - p_fighter_2)),
                "orientation_prob_ab": p_ab,
                "orientation_prob_ba": p_ba,
            }
        )
        return base
