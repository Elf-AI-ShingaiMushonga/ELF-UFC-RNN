#!/usr/bin/env python3
"""Model-backed UFC matchup prediction service."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from scripts.run_ufc_siamese_study import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    SPLIT_COL,
    TARGET_COL,
    FightPairDataset,
    SiameseConfig,
    SiameseGRUModel,
    build_tabular_pipeline,
    build_siamese_dataset,
    get_training_capacity_profile,
    indices_from_names,
    load_and_prepare_dataframe,
    make_one_hot_encoder,
    rebalance_binary_orientation_if_needed,
    resolve_device,
    select_baseline_features,
    subset_prepared_data,
)


# Highest-accuracy model options plus best-performing Siamese variant.
TOP_MODEL_RANKING: list[tuple[str, str, float]] = [
    ("accuracy_weighted_ensemble", "Accuracy-Weighted Ensemble", 0.6201413427561837),
    ("mlp", "MLP", 0.6130742049469965),
    ("logistic_regression", "Logistic Regression", 0.6113074204946997),
    ("random_forest", "Random Forest", 0.6042402826855123),
    ("extra_trees", "Extra Trees", 0.5848056537102474),
    ("siamese_no_context", "Siamese (No Context)", 0.4805653710247349),
]

# Fixed blend weights from best-accuracy experiments.
ACCURACY_ENSEMBLE_WEIGHTS: dict[str, float] = {
    "random_forest": 0.5950887472230991,
    "gradient_boosting": 0.3576107367526437,
    "extra_trees": 0.02593190460582457,
    "mlp": 0.018272243244888853,
    "logistic_regression": 0.0030963681735436442,
}

# Base tabular models needed to support ranked options (including ensemble components).
BASE_MODEL_KEYS = [
    "mlp",
    "logistic_regression",
    "gradient_boosting",
    "random_forest",
    "extra_trees",
]

CACHE_VERSION = 1
BASE_MODEL_CACHE_FILE = "base_models.joblib"
SIAMESE_CACHE_FILE = "siamese_no_context.pt"


@dataclass
class FighterProfile:
    fighter_id: str
    fighter_name: str
    last_event_date: pd.Timestamp
    features: dict[str, Any]


@dataclass
class ModelSpec:
    key: str
    label: str
    test_accuracy: float


class FightPredictor:
    def __init__(
        self,
        csv_path: Path,
        default_model: str = "mlp",
        power_profile: str = "max_power",
        seed: int = 42,
        model_cache_dir: Path | None = None,
    ) -> None:
        self.csv_path = csv_path
        self.seed = int(seed)
        self.power_profile = str(power_profile)
        self.model_cache_dir = (
            model_cache_dir
            if model_cache_dir is not None
            else (self.csv_path.parent / "model_cache")
        )
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self._base_model_cache_path = self.model_cache_dir / BASE_MODEL_CACHE_FILE
        self._siamese_cache_path = self.model_cache_dir / SIAMESE_CACHE_FILE
        self._cache_info: dict[str, Any] = {}
        self.capacity_cfg = get_training_capacity_profile(power_profile=self.power_profile, seed=self.seed)
        self.model_specs = [ModelSpec(*row) for row in TOP_MODEL_RANKING]
        self.model_spec_map = {spec.key: spec for spec in self.model_specs}
        self.model_keys = [spec.key for spec in self.model_specs]
        self.default_model = default_model if default_model in self.model_spec_map else self.model_keys[0]
        self.model_options = [
            {
                "key": spec.key,
                "label": f"{spec.label} ({spec.test_accuracy * 100:.2f}% test acc)",
            }
            for spec in self.model_specs
        ]

        self.df = load_and_prepare_dataframe(csv_path)
        self.df_aug = self._augment_with_swapped_rows(self.df)
        (
            self.base_models,
            self.feature_cols,
            self.numeric_cols,
            self.categorical_cols,
            loaded_from_cache,
            trained_at_utc,
        ) = self._load_or_train_base_models(self.df_aug)
        self._cache_info.update(
            {
                "base_models_loaded_from_cache": bool(loaded_from_cache),
                "base_models_trained_at_utc": trained_at_utc,
            }
        )

        self._refresh_runtime_state()

        # Siamese model (best-performing variant) is trained lazily on first use.
        # It is fit on all available historical rows and cached after first train.
        self._siamese_model: Any = None
        self._siamese_device: str = "cpu"
        self._siamese_max_seq_len: int = 8
        self._siamese_static_idx: list[int] = []
        self._siamese_seq_idx: list[int] = []
        self._siamese_config = SiameseConfig(
            max_seq_len=self._siamese_max_seq_len,
            hidden_dim=int(self.capacity_cfg["siamese_hidden_dim"]),
            static_hidden_dim=int(self.capacity_cfg["siamese_static_hidden_dim"]),
            num_layers=int(self.capacity_cfg["siamese_num_layers"]),
            dropout=float(self.capacity_cfg["siamese_dropout"]),
            batch_size=int(self.capacity_cfg["siamese_batch_size"]),
            epochs=int(self.capacity_cfg["siamese_epochs"]),
            lr=float(self.capacity_cfg["siamese_lr"]),
            weight_decay=float(self.capacity_cfg["siamese_weight_decay"]),
            patience=int(self.capacity_cfg["siamese_patience"]),
        )
        self._cache_info["siamese_loaded_from_cache"] = False
        self._cache_info["siamese_trained_at_utc"] = None
        self._siamese_base_df = self._prepare_siamese_base_dataframe()

    @staticmethod
    def _utc_now_iso() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()

    def _refresh_runtime_state(self) -> None:
        self.feature_set = set(self.feature_cols)
        self.feature_defaults = self._build_feature_defaults(self.df_aug)
        self.diff_cols = [c for c in self.feature_cols if c.endswith("_diff_f1_minus_f2")]
        self.fighters_by_name = self._build_fighter_profiles(self.df)
        self.fighter_names = sorted(
            {profile.fighter_name for profile in self.fighters_by_name.values()},
            key=str.lower,
        )
        self.weight_classes = self._sorted_unique_strings(self.df, "weight_class")
        self.genders = self._sorted_unique_strings(self.df, "gender")
        self.context_defaults = self._build_context_defaults(self.df)

    def _load_base_models_from_cache(
        self,
    ) -> tuple[dict[str, Any], list[str], list[str], list[str], str] | None:
        if not self._base_model_cache_path.exists():
            return None
        try:
            payload = joblib.load(self._base_model_cache_path)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if int(payload.get("cache_version", -1)) != CACHE_VERSION:
            return None
        if str(payload.get("power_profile", "")) != self.power_profile:
            return None
        if int(payload.get("seed", -1)) != self.seed:
            return None

        models = payload.get("models")
        feature_cols = payload.get("feature_cols")
        numeric_cols = payload.get("numeric_cols")
        categorical_cols = payload.get("categorical_cols")
        if not isinstance(models, dict):
            return None
        if not all(key in models for key in BASE_MODEL_KEYS):
            return None
        if not isinstance(feature_cols, list) or not isinstance(numeric_cols, list) or not isinstance(categorical_cols, list):
            return None
        trained_at_utc = str(payload.get("trained_at_utc", ""))
        return models, feature_cols, numeric_cols, categorical_cols, trained_at_utc

    def _save_base_models_to_cache(
        self,
        models: dict[str, Any],
        feature_cols: list[str],
        numeric_cols: list[str],
        categorical_cols: list[str],
        trained_at_utc: str,
    ) -> None:
        payload = {
            "cache_version": CACHE_VERSION,
            "power_profile": self.power_profile,
            "seed": self.seed,
            "trained_at_utc": trained_at_utc,
            "csv_path": str(self.csv_path),
            "models": models,
            "feature_cols": feature_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        }
        joblib.dump(payload, self._base_model_cache_path)

    def _load_or_train_base_models(
        self,
        df: pd.DataFrame,
    ) -> tuple[dict[str, Any], list[str], list[str], list[str], bool, str]:
        cached = self._load_base_models_from_cache()
        if cached is not None:
            models, feature_cols, numeric_cols, categorical_cols, trained_at_utc = cached
            return models, feature_cols, numeric_cols, categorical_cols, True, trained_at_utc

        models, feature_cols, numeric_cols, categorical_cols = self._train_models(df)
        trained_at_utc = self._utc_now_iso()
        self._save_base_models_to_cache(
            models=models,
            feature_cols=feature_cols,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            trained_at_utc=trained_at_utc,
        )
        return models, feature_cols, numeric_cols, categorical_cols, False, trained_at_utc

    def _clear_siamese_cache(self) -> None:
        self._siamese_model = None
        self._siamese_seq_idx = []
        self._siamese_static_idx = []
        if self._siamese_cache_path.exists():
            self._siamese_cache_path.unlink()

    def reload_data(self) -> dict[str, Any]:
        self.df = load_and_prepare_dataframe(self.csv_path)
        self.df_aug = self._augment_with_swapped_rows(self.df)
        self._siamese_base_df = self._prepare_siamese_base_dataframe()
        self._refresh_runtime_state()
        return self.model_cache_status()

    def retrain_models(self) -> dict[str, Any]:
        self.df = load_and_prepare_dataframe(self.csv_path)
        self.df_aug = self._augment_with_swapped_rows(self.df)
        (
            self.base_models,
            self.feature_cols,
            self.numeric_cols,
            self.categorical_cols,
        ) = self._train_models(self.df_aug)
        trained_at_utc = self._utc_now_iso()
        self._save_base_models_to_cache(
            models=self.base_models,
            feature_cols=self.feature_cols,
            numeric_cols=self.numeric_cols,
            categorical_cols=self.categorical_cols,
            trained_at_utc=trained_at_utc,
        )
        self._cache_info["base_models_loaded_from_cache"] = False
        self._cache_info["base_models_trained_at_utc"] = trained_at_utc
        self._siamese_base_df = self._prepare_siamese_base_dataframe()
        self._clear_siamese_cache()
        self._refresh_runtime_state()
        return self.model_cache_status()

    def model_cache_status(self) -> dict[str, Any]:
        data_updated_utc = None
        if self.csv_path.exists():
            data_updated_utc = dt.datetime.fromtimestamp(
                self.csv_path.stat().st_mtime,
                tz=dt.timezone.utc,
            ).isoformat()
        return {
            "cache_dir": str(self.model_cache_dir),
            "base_models_loaded_from_cache": bool(self._cache_info.get("base_models_loaded_from_cache", False)),
            "base_models_trained_at_utc": self._cache_info.get("base_models_trained_at_utc"),
            "siamese_loaded_from_cache": bool(self._cache_info.get("siamese_loaded_from_cache", False)),
            "siamese_trained_at_utc": self._cache_info.get("siamese_trained_at_utc"),
            "data_rows": int(len(self.df)),
            "data_updated_utc": data_updated_utc,
        }

    @staticmethod
    def _sorted_unique_strings(df: pd.DataFrame, col: str) -> list[str]:
        if col not in df.columns:
            return []
        vals = (
            df[col]
            .dropna()
            .astype(str)
            .map(str.strip)
            .replace("nan", "")
            .replace("", np.nan)
            .dropna()
            .unique()
            .tolist()
        )
        return sorted(vals)

    def _augment_with_swapped_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        mirrored = df.copy(deep=True)

        paired_cols: list[tuple[str, str]] = []
        for col in df.columns:
            if not col.startswith("fighter_1_"):
                continue
            suffix = col[len("fighter_1_") :]
            col2 = f"fighter_2_{suffix}"
            if col2 in df.columns:
                paired_cols.append((col, col2))
        for col1, col2 in paired_cols:
            mirrored[col1] = df[col2].values
            mirrored[col2] = df[col1].values

        diff_cols = [c for c in df.columns if c.endswith("_diff_f1_minus_f2")]
        for col in diff_cols:
            mirrored[col] = -pd.to_numeric(df[col], errors="coerce")

        mirrored["outcome_label"] = df["outcome_label"].map(
            {POSITIVE_LABEL: NEGATIVE_LABEL, NEGATIVE_LABEL: POSITIVE_LABEL}
        )
        mirrored[TARGET_COL] = (mirrored["outcome_label"] == POSITIVE_LABEL).astype(int)

        if "winner_fighter_id" in df.columns:
            mirrored["winner_fighter_id"] = np.where(
                df["outcome_label"] == POSITIVE_LABEL,
                df["fighter_2_id"],
                df["fighter_1_id"],
            )
        if "winner_name" in df.columns:
            mirrored["winner_name"] = np.where(
                df["outcome_label"] == POSITIVE_LABEL,
                df["fighter_2_name"],
                df["fighter_1_name"],
            )

        return pd.concat([df, mirrored], ignore_index=True)

    def _prepare_siamese_base_dataframe(self) -> pd.DataFrame:
        df = load_and_prepare_dataframe(self.csv_path)
        df, _ = rebalance_binary_orientation_if_needed(df, seed=self.seed)
        # Train Siamese on all currently available history for web inference.
        df[SPLIT_COL] = "train"
        return df

    def _train_siamese_model_for_web(self, prepared: Any, device: str) -> Any:
        train_ds = FightPairDataset(
            prepared.seq1_train,
            prepared.len1_train,
            prepared.seq2_train,
            prepared.len2_train,
            prepared.static_train,
            prepared.physical_train,
            prepared.y_train,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self._siamese_config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        model = SiameseGRUModel(
            seq_dim=prepared.seq1_train.shape[2],
            static_dim=prepared.static_train.shape[1],
            physical_dim=prepared.physical_train.shape[1],
            hidden_dim=self._siamese_config.hidden_dim,
            static_hidden_dim=self._siamese_config.static_hidden_dim,
            num_layers=self._siamese_config.num_layers,
            dropout=self._siamese_config.dropout,
            interaction_mode="full",
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self._siamese_config.lr,
            weight_decay=self._siamese_config.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(self._siamese_config.epochs):
            model.train()
            for batch in train_loader:
                seq1 = batch["seq1"].to(device)
                len1 = batch["len1"].to(device)
                seq2 = batch["seq2"].to(device)
                len2 = batch["len2"].to(device)
                static = batch["static"].to(device)
                physical = batch["physical"].to(device)
                y = batch["y"].to(device)

                optimizer.zero_grad()
                logits = model(seq1, len1, seq2, len2, static, physical)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        return model

    def _try_load_siamese_from_cache(
        self,
        prepared: Any,
        seq_idx: list[int],
        static_idx: list[int],
        device: str,
    ) -> Any | None:
        if not self._siamese_cache_path.exists():
            return None
        try:
            payload = torch.load(self._siamese_cache_path, map_location=device)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if int(payload.get("cache_version", -1)) != CACHE_VERSION:
            return None
        if payload.get("seq_idx") != seq_idx or payload.get("static_idx") != static_idx:
            return None
        if payload.get("siamese_config") != dataclasses.asdict(self._siamese_config):
            return None
        seq_dim = int(prepared.seq1_train.shape[2])
        static_dim = int(prepared.static_train.shape[1])
        physical_dim = int(prepared.physical_train.shape[1])
        if int(payload.get("seq_dim", -1)) != seq_dim:
            return None
        if int(payload.get("static_dim", -1)) != static_dim:
            return None
        if int(payload.get("physical_dim", -1)) != physical_dim:
            return None

        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            return None

        model = SiameseGRUModel(
            seq_dim=seq_dim,
            static_dim=static_dim,
            physical_dim=physical_dim,
            hidden_dim=self._siamese_config.hidden_dim,
            static_hidden_dim=self._siamese_config.static_hidden_dim,
            num_layers=self._siamese_config.num_layers,
            dropout=self._siamese_config.dropout,
            interaction_mode="full",
        ).to(device)
        try:
            model.load_state_dict(state_dict)
        except Exception:
            return None
        model.eval()
        self._cache_info["siamese_loaded_from_cache"] = True
        self._cache_info["siamese_trained_at_utc"] = payload.get("trained_at_utc")
        return model

    def _save_siamese_to_cache(
        self,
        model: Any,
        prepared: Any,
        seq_idx: list[int],
        static_idx: list[int],
    ) -> None:
        payload = {
            "cache_version": CACHE_VERSION,
            "seq_idx": seq_idx,
            "static_idx": static_idx,
            "siamese_config": dataclasses.asdict(self._siamese_config),
            "seq_dim": int(prepared.seq1_train.shape[2]),
            "static_dim": int(prepared.static_train.shape[1]),
            "physical_dim": int(prepared.physical_train.shape[1]),
            "trained_at_utc": self._utc_now_iso(),
            "state_dict": model.state_dict(),
        }
        torch.save(payload, self._siamese_cache_path)

    def _ensure_siamese_model(self) -> None:
        if self._siamese_model is not None:
            return

        self._siamese_device = resolve_device("auto")
        prepared = build_siamese_dataset(self._siamese_base_df.copy(), max_seq_len=self._siamese_max_seq_len)

        seq_idx = list(range(len(prepared.sequence_feature_names)))
        static_excludes = ["weight_class__", "gender__", "event_country__", "is_title_bout", "scheduled_rounds", "event_"]
        static_idx = indices_from_names(prepared.static_feature_names, static_excludes)
        prepared_sub = subset_prepared_data(prepared, seq_idx, static_idx)

        cached_model = self._try_load_siamese_from_cache(
            prepared=prepared_sub,
            seq_idx=seq_idx,
            static_idx=static_idx,
            device=self._siamese_device,
        )
        if cached_model is not None:
            self._siamese_model = cached_model
            self._siamese_seq_idx = seq_idx
            self._siamese_static_idx = static_idx
            return

        model = self._train_siamese_model_for_web(prepared_sub, device=self._siamese_device)
        self._save_siamese_to_cache(
            model=model,
            prepared=prepared_sub,
            seq_idx=seq_idx,
            static_idx=static_idx,
        )
        self._siamese_model = model
        self._siamese_seq_idx = seq_idx
        self._siamese_static_idx = static_idx
        self._cache_info["siamese_loaded_from_cache"] = False
        self._cache_info["siamese_trained_at_utc"] = self._utc_now_iso()

    def _build_siamese_prediction_row(
        self,
        x_row: pd.DataFrame,
        context: dict[str, Any],
        fighter_1_id: str,
        fighter_2_id: str,
        fight_id: str,
    ) -> dict[str, Any]:
        template = {col: np.nan for col in self._siamese_base_df.columns}
        for col, val in x_row.iloc[0].items():
            if col in template:
                template[col] = val

        template["fight_id"] = fight_id
        template["event_id"] = "web_prediction"
        template["event_name"] = "Web Predictor"
        template["event_date"] = pd.Timestamp(context["event_date"]).normalize()
        template["fighter_1_id"] = fighter_1_id
        template["fighter_2_id"] = fighter_2_id
        template["fighter_1_name"] = context["fighter_1"]
        template["fighter_2_name"] = context["fighter_2"]
        template["winner_fighter_id"] = ""
        template["winner_name"] = ""
        template["outcome_label"] = NEGATIVE_LABEL
        template[TARGET_COL] = 0
        template[SPLIT_COL] = "test"
        if "bout_index" in template and (pd.isna(template["bout_index"]) or template["bout_index"] == ""):
            template["bout_index"] = 999.0
        return template

    def _predict_siamese_pair_probabilities(
        self,
        x_ab: pd.DataFrame,
        x_ba: pd.DataFrame,
        context: dict[str, Any],
    ) -> tuple[float, float]:
        self._ensure_siamese_model()
        assert self._siamese_model is not None

        fid_ab = f"web_pred_ab_{context['fighter_1_id']}_{context['fighter_2_id']}_{context['event_date']}"
        fid_ba = f"web_pred_ba_{context['fighter_2_id']}_{context['fighter_1_id']}_{context['event_date']}"
        row_ab = self._build_siamese_prediction_row(
            x_row=x_ab,
            context=context,
            fighter_1_id=context["fighter_1_id"],
            fighter_2_id=context["fighter_2_id"],
            fight_id=fid_ab,
        )
        reverse_context = dict(context)
        reverse_context["fighter_1"] = context["fighter_2"]
        reverse_context["fighter_2"] = context["fighter_1"]
        reverse_context["fighter_1_id"] = context["fighter_2_id"]
        reverse_context["fighter_2_id"] = context["fighter_1_id"]
        row_ba = self._build_siamese_prediction_row(
            x_row=x_ba,
            context=reverse_context,
            fighter_1_id=context["fighter_2_id"],
            fighter_2_id=context["fighter_1_id"],
            fight_id=fid_ba,
        )

        df_pred = pd.concat([self._siamese_base_df, pd.DataFrame([row_ab, row_ba])], ignore_index=True)
        df_pred["event_date"] = pd.to_datetime(df_pred["event_date"], errors="coerce")
        df_pred = df_pred.sort_values(["event_date", "bout_index", "fight_id"]).reset_index(drop=True)

        prepared = build_siamese_dataset(df_pred, max_seq_len=self._siamese_max_seq_len)
        prepared = subset_prepared_data(prepared, self._siamese_seq_idx, self._siamese_static_idx)
        test_meta = prepared.test_meta.copy()
        idx_ab = test_meta.index[test_meta["fight_id"].astype(str) == fid_ab]
        idx_ba = test_meta.index[test_meta["fight_id"].astype(str) == fid_ba]
        if len(idx_ab) == 0 or len(idx_ba) == 0:
            raise ValueError("Failed to build Siamese inference samples for matchup.")
        i_ab = int(idx_ab[0])
        i_ba = int(idx_ba[0])

        batch_idx = np.array([i_ab, i_ba], dtype=int)
        seq1 = torch.tensor(prepared.seq1_test[batch_idx], dtype=torch.float32, device=self._siamese_device)
        len1 = torch.tensor(prepared.len1_test[batch_idx], dtype=torch.int64, device=self._siamese_device)
        seq2 = torch.tensor(prepared.seq2_test[batch_idx], dtype=torch.float32, device=self._siamese_device)
        len2 = torch.tensor(prepared.len2_test[batch_idx], dtype=torch.int64, device=self._siamese_device)
        static = torch.tensor(prepared.static_test[batch_idx], dtype=torch.float32, device=self._siamese_device)
        physical = torch.tensor(prepared.physical_test[batch_idx], dtype=torch.float32, device=self._siamese_device)

        self._siamese_model.eval()
        with torch.no_grad():
            logits = self._siamese_model(seq1, len1, seq2, len2, static, physical)
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(float)

        probs = np.clip(probs, 0.0, 1.0)
        return float(probs[0]), float(probs[1])

    def _build_tree_pipeline(self, model_name: str) -> Any:
        numeric_pipe = Pipeline(
            steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        categorical_pipe = Pipeline(
            steps=[("impute", SimpleImputer(strategy="most_frequent")), ("onehot", make_one_hot_encoder())]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, self.numeric_cols),
                ("cat", categorical_pipe, self.categorical_cols),
            ],
            remainder="drop",
        )

        if model_name == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=int(self.capacity_cfg["rf_n_estimators"]),
                max_depth=None,
                min_samples_leaf=int(self.capacity_cfg["rf_min_samples_leaf"]),
                class_weight="balanced_subsample",
                random_state=self.seed,
                n_jobs=int(self.capacity_cfg["tree_n_jobs"]),
            )
        elif model_name == "extra_trees":
            estimator = ExtraTreesClassifier(
                n_estimators=int(self.capacity_cfg["et_n_estimators"]),
                max_depth=None,
                min_samples_leaf=int(self.capacity_cfg["et_min_samples_leaf"]),
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=int(self.capacity_cfg["tree_n_jobs"]),
            )
        else:
            raise ValueError(f"Unsupported tree model: {model_name}")

        return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])

    def _train_models(
        self,
        df: pd.DataFrame,
    ) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
        feature_cols, numeric_cols, categorical_cols = select_baseline_features(df)
        feature_cols = [col for col in feature_cols if df[col].notna().any()]
        numeric_cols = [c for c in numeric_cols if c in feature_cols]
        categorical_cols = [c for c in categorical_cols if c in feature_cols]

        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        x = df[feature_cols].copy()
        y = df[TARGET_COL].to_numpy(dtype=int)

        for col in numeric_cols:
            x[col] = pd.to_numeric(x[col], errors="coerce")
        for col in categorical_cols:
            x[col] = x[col].astype(str).replace("nan", "__UNK__").replace("", "__UNK__")

        models: dict[str, Any] = {}
        for model_name in BASE_MODEL_KEYS:
            if model_name in {"random_forest", "extra_trees"}:
                model = self._build_tree_pipeline(model_name)
            else:
                model = build_tabular_pipeline(
                    model_name=model_name,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    power_profile=self.power_profile,
                    seed=self.seed,
                )
            model.fit(x, y)
            models[model_name] = model

        return models, feature_cols, numeric_cols, categorical_cols

    def _build_feature_defaults(self, df: pd.DataFrame) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        for col in self.feature_cols:
            if col not in df.columns:
                if col in self.numeric_cols:
                    defaults[col] = 0.0
                elif col in self.categorical_cols:
                    defaults[col] = "__UNK__"
                else:
                    defaults[col] = 0.0
                continue
            if col in self.numeric_cols:
                series = pd.to_numeric(df[col], errors="coerce")
                defaults[col] = float(series.median()) if series.notna().any() else 0.0
            elif col in self.categorical_cols:
                series = (
                    df[col]
                    .astype(str)
                    .replace("nan", "__UNK__")
                    .replace("", "__UNK__")
                    .fillna("__UNK__")
                )
                if series.empty:
                    defaults[col] = "__UNK__"
                else:
                    mode = series.mode(dropna=False)
                    defaults[col] = str(mode.iloc[0]) if not mode.empty else "__UNK__"
            else:
                defaults[col] = 0.0
        return defaults

    def _build_context_defaults(self, df: pd.DataFrame) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        defaults["weight_class"] = self.feature_defaults.get("weight_class", "")
        defaults["gender"] = self.feature_defaults.get("gender", "")
        defaults["event_country"] = self.feature_defaults.get("event_country", "")
        rounds = self.feature_defaults.get("scheduled_rounds", 3)
        defaults["scheduled_rounds"] = int(round(rounds)) if isinstance(rounds, (int, float)) else 3
        defaults["is_title_bout"] = 0
        return defaults

    def _build_fighter_profiles(self, df: pd.DataFrame) -> dict[str, FighterProfile]:
        profiles: dict[str, FighterProfile] = {}
        side_fields: dict[int, list[str]] = {1: [], 2: []}
        for side in (1, 2):
            prefix = f"fighter_{side}_"
            side_fields[side] = [c for c in df.columns if c.startswith(prefix)]

        sorted_df = df.sort_values(["event_date", "bout_index", "fight_id"])
        for _, row in sorted_df.iterrows():
            event_date = pd.Timestamp(row["event_date"]).normalize()
            for side in (1, 2):
                fighter_id = str(row.get(f"fighter_{side}_id", "")).strip()
                fighter_name = str(row.get(f"fighter_{side}_name", "")).strip()
                if not fighter_id or not fighter_name:
                    continue
                prefix = f"fighter_{side}_"
                feat: dict[str, Any] = {}
                for col in side_fields[side]:
                    suffix = col[len(prefix) :]
                    if suffix in {"id", "name", "dob"}:
                        continue
                    feat[suffix] = row.get(col)
                key = fighter_name.lower()
                profiles[key] = FighterProfile(
                    fighter_id=fighter_id,
                    fighter_name=fighter_name,
                    last_event_date=event_date,
                    features=feat,
                )
        return profiles

    def resolve_fighter(self, fighter_name: str) -> FighterProfile:
        key = fighter_name.strip().lower()
        if key not in self.fighters_by_name:
            raise ValueError(f"Unknown fighter: {fighter_name}")
        return self.fighters_by_name[key]

    @staticmethod
    def _as_float(value: Any) -> float:
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def _resolve_model(self, model_name: str | None) -> str:
        chosen = str(model_name or self.default_model).strip().lower()
        if chosen not in self.model_spec_map:
            raise ValueError(
                f"Unsupported model: {model_name}. Choose one of: {', '.join(self.model_keys)} or all."
            )
        return chosen

    def _predict_proba_with_model(self, model_name: str, x: pd.DataFrame) -> np.ndarray:
        if model_name == "accuracy_weighted_ensemble":
            probs = np.zeros(len(x), dtype=float)
            for key, weight in ACCURACY_ENSEMBLE_WEIGHTS.items():
                model = self.base_models[key]
                probs += float(weight) * model.predict_proba(x)[:, 1]
            return np.clip(probs, 0.0, 1.0)
        model = self.base_models[model_name]
        return model.predict_proba(x)[:, 1]

    def _apply_fighter_to_row(
        self,
        row: dict[str, Any],
        fighter: FighterProfile,
        fighter_prefix: str,
        event_date: pd.Timestamp,
    ) -> None:
        for suffix, value in fighter.features.items():
            col = f"{fighter_prefix}{suffix}"
            if col in self.feature_set:
                row[col] = value

        days_since_last = (event_date - fighter.last_event_date).days
        days_since_last = max(days_since_last, 0)
        days_col = f"{fighter_prefix}days_since_last_fight"
        if days_col in self.feature_set:
            row[days_col] = float(days_since_last)

        age_col = f"{fighter_prefix}age_days"
        if age_col in self.feature_set:
            age_at_last = self._as_float(fighter.features.get("age_days"))
            if not np.isnan(age_at_last):
                row[age_col] = float(age_at_last + days_since_last)

    def _build_feature_row(
        self,
        fighter_1: FighterProfile,
        fighter_2: FighterProfile,
        event_date: pd.Timestamp,
        weight_class: str,
        gender: str,
        scheduled_rounds: int,
        is_title_bout: bool,
    ) -> pd.DataFrame:
        row = dict(self.feature_defaults)

        if "event_year" in self.feature_set:
            row["event_year"] = float(event_date.year)
        if "event_month" in self.feature_set:
            row["event_month"] = float(event_date.month)
        if "event_dayofweek" in self.feature_set:
            row["event_dayofweek"] = float(event_date.dayofweek)
        if "weight_class" in self.feature_set and weight_class:
            row["weight_class"] = weight_class
        if "gender" in self.feature_set and gender:
            row["gender"] = gender
        if "scheduled_rounds" in self.feature_set:
            row["scheduled_rounds"] = float(scheduled_rounds)
        if "is_title_bout" in self.feature_set:
            row["is_title_bout"] = 1.0 if is_title_bout else 0.0

        self._apply_fighter_to_row(row, fighter_1, "fighter_1_", event_date)
        self._apply_fighter_to_row(row, fighter_2, "fighter_2_", event_date)

        for diff_col in self.diff_cols:
            base_name = diff_col[: -len("_diff_f1_minus_f2")]
            col1 = f"fighter_1_{base_name}"
            col2 = f"fighter_2_{base_name}"
            if col1 not in self.feature_set or col2 not in self.feature_set:
                continue
            v1 = self._as_float(row.get(col1))
            v2 = self._as_float(row.get(col2))
            if np.isnan(v1) or np.isnan(v2):
                continue
            row[diff_col] = float(v1 - v2)

        x = pd.DataFrame([row], columns=self.feature_cols)
        for col in self.numeric_cols:
            x[col] = pd.to_numeric(x[col], errors="coerce")
        for col in self.categorical_cols:
            x[col] = x[col].astype(str).replace("nan", "__UNK__").replace("", "__UNK__")
        return x

    def _prepare_matchup_features(
        self,
        fighter_1_name: str,
        fighter_2_name: str,
        event_date: str | None,
        weight_class: str | None,
        gender: str | None,
        scheduled_rounds: int | None,
        is_title_bout: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        if fighter_1_name.strip().lower() == fighter_2_name.strip().lower():
            raise ValueError("Choose two different fighters.")

        fighter_1 = self.resolve_fighter(fighter_1_name)
        fighter_2 = self.resolve_fighter(fighter_2_name)
        pred_date = pd.Timestamp(event_date).normalize() if event_date else pd.Timestamp.today().normalize()

        use_weight_class = weight_class or self.context_defaults["weight_class"]
        use_gender = gender or self.context_defaults["gender"]
        use_rounds = int(scheduled_rounds or self.context_defaults["scheduled_rounds"])

        x_ab = self._build_feature_row(
            fighter_1=fighter_1,
            fighter_2=fighter_2,
            event_date=pred_date,
            weight_class=use_weight_class,
            gender=use_gender,
            scheduled_rounds=use_rounds,
            is_title_bout=is_title_bout,
        )
        x_ba = self._build_feature_row(
            fighter_1=fighter_2,
            fighter_2=fighter_1,
            event_date=pred_date,
            weight_class=use_weight_class,
            gender=use_gender,
            scheduled_rounds=use_rounds,
            is_title_bout=is_title_bout,
        )
        context = {
            "fighter_1": fighter_1.fighter_name,
            "fighter_2": fighter_2.fighter_name,
            "fighter_1_id": fighter_1.fighter_id,
            "fighter_2_id": fighter_2.fighter_id,
            "event_date": str(pred_date.date()),
            "weight_class": use_weight_class,
            "gender": use_gender,
            "scheduled_rounds": use_rounds,
            "is_title_bout": bool(is_title_bout),
        }
        return x_ab, x_ba, context

    def _predict_from_feature_pair(
        self,
        model_name: str,
        x_ab: pd.DataFrame,
        x_ba: pd.DataFrame,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if model_name == "siamese_no_context":
            p_ab, p_ba = self._predict_siamese_pair_probabilities(x_ab=x_ab, x_ba=x_ba, context=context)
        else:
            p_ab = float(self._predict_proba_with_model(model_name, x_ab)[0])
            p_ba = float(self._predict_proba_with_model(model_name, x_ba)[0])
        p_fighter_1 = float(np.clip((p_ab + (1.0 - p_ba)) * 0.5, 0.0, 1.0))
        p_fighter_2 = 1.0 - p_fighter_1

        winner = context["fighter_1"] if p_fighter_1 >= p_fighter_2 else context["fighter_2"]
        confidence = abs(p_fighter_1 - p_fighter_2)
        spec = self.model_spec_map[model_name]
        return {
            **context,
            "model": model_name,
            "model_label": spec.label,
            "model_test_accuracy": spec.test_accuracy,
            "p_fighter_1": p_fighter_1,
            "p_fighter_2": p_fighter_2,
            "winner": winner,
            "confidence": confidence,
        }

    def predict_matchup(
        self,
        fighter_1_name: str,
        fighter_2_name: str,
        event_date: str | None = None,
        weight_class: str | None = None,
        gender: str | None = None,
        scheduled_rounds: int | None = None,
        is_title_bout: bool = False,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        chosen_model = self._resolve_model(model_name)
        x_ab, x_ba, context = self._prepare_matchup_features(
            fighter_1_name=fighter_1_name,
            fighter_2_name=fighter_2_name,
            event_date=event_date,
            weight_class=weight_class,
            gender=gender,
            scheduled_rounds=scheduled_rounds,
            is_title_bout=is_title_bout,
        )
        return self._predict_from_feature_pair(chosen_model, x_ab, x_ba, context)

    def predict_matchup_all(
        self,
        fighter_1_name: str,
        fighter_2_name: str,
        event_date: str | None = None,
        weight_class: str | None = None,
        gender: str | None = None,
        scheduled_rounds: int | None = None,
        is_title_bout: bool = False,
    ) -> list[dict[str, Any]]:
        x_ab, x_ba, context = self._prepare_matchup_features(
            fighter_1_name=fighter_1_name,
            fighter_2_name=fighter_2_name,
            event_date=event_date,
            weight_class=weight_class,
            gender=gender,
            scheduled_rounds=scheduled_rounds,
            is_title_bout=is_title_bout,
        )
        out = [self._predict_from_feature_pair(spec.key, x_ab, x_ba, context) for spec in self.model_specs]
        return out
