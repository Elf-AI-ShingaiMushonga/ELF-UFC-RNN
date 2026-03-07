#!/usr/bin/env python3
"""Build chronological fighter-history sequences for LSTM training."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd


BASE_PERFORMANCE_STATS = [
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
    "ctrl_seconds_1",
    "fight_duration_seconds",
]
# Chronology-derived features injected per historical fight.
DERIVED_PERFORMANCE_STATS = [
    "opponent_elo_1",
    "days_since_last_fight_1",
]
DEFAULT_PERFORMANCE_STATS = BASE_PERFORMANCE_STATS + DERIVED_PERFORMANCE_STATS

OPTIONAL_METADATA_COLUMNS = [
    "fighter_1_age_days",
    "fighter_2_age_days",
    "fighter_1_dob",
    "fighter_2_dob",
    "age_days_diff_f1_minus_f2",
    "fighter_1_height_cm",
    "fighter_2_height_cm",
    "fighter_1_reach_cm",
    "fighter_2_reach_cm",
    "height_cm_diff_f1_minus_f2",
    "reach_cm_diff_f1_minus_f2",
]

VALID_OUTCOMES = {"fighter_1_win", "fighter_2_win", "draw", "no_contest"}


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build fighter-history sequence rows from ufc_fight_details_lstm.csv"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=root_dir / "data" / "ufc_fight_details_lstm.csv",
        help="Raw fight-details CSV path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=root_dir / "data" / "ufc_lstm_sequences.csv",
        help="Output sequence CSV path.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=root_dir / "data" / "ufc_fights_cleaned.csv",
        help="Optional metadata CSV used to source age fields by fight_id.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Number of prior fights to keep per fighter.",
    )
    parser.add_argument(
        "--elo-base",
        type=float,
        default=1500.0,
        help="Initial ELO assigned to fighters with no prior fights.",
    )
    parser.add_argument(
        "--elo-k-factor",
        type=float,
        default=24.0,
        help="K-factor used for ELO updates after each decisive fight.",
    )
    parser.add_argument(
        "--elo-scale",
        type=float,
        default=400.0,
        help="Scale term in expected-score computation for ELO.",
    )
    parser.add_argument(
        "--drop-nonstandard-outcomes",
        action="store_true",
        help="Drop rows where outcome_label is not one of fighter_1_win/fighter_2_win/draw/no_contest.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def swap_fighter_side(col_name: str) -> str:
    if "_1_" in col_name:
        return col_name.replace("_1_", "_2_")
    if "_2_" in col_name:
        return col_name.replace("_2_", "_1_")
    if col_name.endswith("_1"):
        return col_name[:-2] + "_2"
    if col_name.endswith("_2"):
        return col_name[:-2] + "_1"
    return col_name


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_raw_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path)


def merge_optional_metadata(df: pd.DataFrame, metadata_csv: Path) -> pd.DataFrame:
    if not metadata_csv.exists():
        logging.warning(
            "Metadata CSV not found at %s; age-based features will default to 0.",
            metadata_csv,
        )
        return df

    usecols = {"fight_id"} | set(OPTIONAL_METADATA_COLUMNS)
    try:
        meta = pd.read_csv(metadata_csv, usecols=lambda c: c in usecols)
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Failed reading metadata CSV %s: %s", metadata_csv, exc)
        return df

    if "fight_id" not in meta.columns:
        logging.warning("Metadata CSV %s has no fight_id column; skipping merge.", metadata_csv)
        return df

    meta = meta.copy()
    meta["fight_id"] = meta["fight_id"].astype(str).str.strip()
    keep_cols = [c for c in OPTIONAL_METADATA_COLUMNS if c in meta.columns]
    if not keep_cols:
        logging.warning("Metadata CSV %s has no supported age columns; skipping merge.", metadata_csv)
        return df

    meta = meta[["fight_id"] + keep_cols]
    dupes = int(meta.duplicated(subset=["fight_id"], keep="last").sum())
    if dupes > 0:
        logging.info("Metadata has %d duplicate fight_id rows; keeping last.", dupes)
        meta = meta.drop_duplicates(subset=["fight_id"], keep="last")

    merged = df.merge(meta, on="fight_id", how="left")
    coverage = float(merged["fighter_1_age_days"].notna().mean()) if "fighter_1_age_days" in merged else 0.0
    logging.info("Merged metadata from %s | age coverage (fighter_1_age_days): %.2f%%", metadata_csv, coverage * 100.0)
    return merged


def prepare_dataframe(
    df: pd.DataFrame,
    performance_stats: list[str],
    drop_nonstandard_outcomes: bool,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    base_required = [
        "fight_id",
        "event_date",
        "bout_index",
        "fighter_1_id",
        "fighter_2_id",
        "outcome_label",
    ]
    f1_stat_columns = performance_stats
    f2_stat_columns = [swap_fighter_side(col) for col in performance_stats]
    derived_cols = set(DERIVED_PERFORMANCE_STATS + [swap_fighter_side(c) for c in DERIVED_PERFORMANCE_STATS])
    required_columns = sorted(set(base_required + f1_stat_columns + f2_stat_columns))

    df = df.copy()
    for col in derived_cols:
        if col not in df.columns:
            df[col] = 0.0

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"])

    df["fight_id"] = df["fight_id"].astype(str).str.strip()
    df["fighter_1_id"] = df["fighter_1_id"].astype(str).str.strip()
    df["fighter_2_id"] = df["fighter_2_id"].astype(str).str.strip()
    df["outcome_label"] = df["outcome_label"].astype(str).str.strip()

    df = df[
        df["fight_id"].ne("")
        & df["fighter_1_id"].ne("")
        & df["fighter_2_id"].ne("")
        & df["fighter_1_id"].ne(df["fighter_2_id"])
    ].copy()

    if drop_nonstandard_outcomes:
        before = len(df)
        df = df[df["outcome_label"].isin(VALID_OUTCOMES)].copy()
        dropped = before - len(df)
        if dropped > 0:
            logging.info("Dropped %d rows with nonstandard outcome_label values.", dropped)

    df["bout_index"] = pd.to_numeric(df["bout_index"], errors="coerce").fillna(999).astype(int)

    for col in sorted(set(f1_stat_columns + f2_stat_columns)):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df[col] = df[col].clip(lower=0.0)

    if "fighter_1_dob" in df.columns:
        df["fighter_1_dob"] = pd.to_datetime(df["fighter_1_dob"], errors="coerce")
    if "fighter_2_dob" in df.columns:
        df["fighter_2_dob"] = pd.to_datetime(df["fighter_2_dob"], errors="coerce")
    if "fighter_1_age_days" in df.columns:
        df["fighter_1_age_days"] = pd.to_numeric(df["fighter_1_age_days"], errors="coerce")
    if "fighter_2_age_days" in df.columns:
        df["fighter_2_age_days"] = pd.to_numeric(df["fighter_2_age_days"], errors="coerce")
    if "fighter_1_height_cm" in df.columns:
        df["fighter_1_height_cm"] = pd.to_numeric(df["fighter_1_height_cm"], errors="coerce")
    if "fighter_2_height_cm" in df.columns:
        df["fighter_2_height_cm"] = pd.to_numeric(df["fighter_2_height_cm"], errors="coerce")
    if "fighter_1_reach_cm" in df.columns:
        df["fighter_1_reach_cm"] = pd.to_numeric(df["fighter_1_reach_cm"], errors="coerce")
    if "fighter_2_reach_cm" in df.columns:
        df["fighter_2_reach_cm"] = pd.to_numeric(df["fighter_2_reach_cm"], errors="coerce")

    if "fighter_1_age_days" not in df.columns:
        df["fighter_1_age_days"] = pd.Series([math.nan] * len(df), index=df.index)
    if "fighter_2_age_days" not in df.columns:
        df["fighter_2_age_days"] = pd.Series([math.nan] * len(df), index=df.index)
    if "fighter_1_height_cm" not in df.columns:
        df["fighter_1_height_cm"] = pd.Series([math.nan] * len(df), index=df.index)
    if "fighter_2_height_cm" not in df.columns:
        df["fighter_2_height_cm"] = pd.Series([math.nan] * len(df), index=df.index)
    if "fighter_1_reach_cm" not in df.columns:
        df["fighter_1_reach_cm"] = pd.Series([math.nan] * len(df), index=df.index)
    if "fighter_2_reach_cm" not in df.columns:
        df["fighter_2_reach_cm"] = pd.Series([math.nan] * len(df), index=df.index)

    if "fighter_1_dob" in df.columns:
        missing_age_1 = df["fighter_1_age_days"].isna() & df["fighter_1_dob"].notna()
        df.loc[missing_age_1, "fighter_1_age_days"] = (
            df.loc[missing_age_1, "event_date"] - df.loc[missing_age_1, "fighter_1_dob"]
        ).dt.days
    if "fighter_2_dob" in df.columns:
        missing_age_2 = df["fighter_2_age_days"].isna() & df["fighter_2_dob"].notna()
        df.loc[missing_age_2, "fighter_2_age_days"] = (
            df.loc[missing_age_2, "event_date"] - df.loc[missing_age_2, "fighter_2_dob"]
        ).dt.days

    df["fighter_1_age_days"] = pd.to_numeric(df["fighter_1_age_days"], errors="coerce").fillna(0.0)
    df["fighter_2_age_days"] = pd.to_numeric(df["fighter_2_age_days"], errors="coerce").fillna(0.0)
    df["fighter_1_age_days"] = df["fighter_1_age_days"].clip(lower=0.0)
    df["fighter_2_age_days"] = df["fighter_2_age_days"].clip(lower=0.0)
    df["fighter_1_height_cm"] = pd.to_numeric(df["fighter_1_height_cm"], errors="coerce").fillna(0.0)
    df["fighter_2_height_cm"] = pd.to_numeric(df["fighter_2_height_cm"], errors="coerce").fillna(0.0)
    df["fighter_1_reach_cm"] = pd.to_numeric(df["fighter_1_reach_cm"], errors="coerce").fillna(0.0)
    df["fighter_2_reach_cm"] = pd.to_numeric(df["fighter_2_reach_cm"], errors="coerce").fillna(0.0)
    df["fighter_1_height_cm"] = df["fighter_1_height_cm"].clip(lower=0.0)
    df["fighter_2_height_cm"] = df["fighter_2_height_cm"].clip(lower=0.0)
    df["fighter_1_reach_cm"] = df["fighter_1_reach_cm"].clip(lower=0.0)
    df["fighter_2_reach_cm"] = df["fighter_2_reach_cm"].clip(lower=0.0)
    df["age_days_diff_f1_minus_f2"] = df["fighter_1_age_days"] - df["fighter_2_age_days"]
    df["age_diff_years_f1_minus_f2"] = df["age_days_diff_f1_minus_f2"] / 365.25
    df["age_gap_over_5y"] = (df["age_diff_years_f1_minus_f2"].abs() >= 5.0).astype(float)
    df["height_cm_diff_f1_minus_f2"] = df["fighter_1_height_cm"] - df["fighter_2_height_cm"]
    df["reach_cm_diff_f1_minus_f2"] = df["fighter_1_reach_cm"] - df["fighter_2_reach_cm"]

    df = df.sort_values(["event_date", "bout_index", "fight_id"]).reset_index(drop=True)

    duplicate_count = int(df.duplicated(subset=["fight_id"], keep="last").sum())
    if duplicate_count > 0:
        logging.warning("Found %d duplicate fight_id rows; keeping last occurrence.", duplicate_count)
        df = df.drop_duplicates(subset=["fight_id"], keep="last").copy()

    df = df.sort_values(["event_date", "bout_index", "fight_id"]).reset_index(drop=True)
    return df, f1_stat_columns, f2_stat_columns


def expected_score(rating_a: float, rating_b: float, scale: float) -> float:
    return 1.0 / (1.0 + (10.0 ** ((rating_b - rating_a) / scale)))


def resolve_sequence_stat(
    row: pd.Series,
    column: str,
    *,
    opponent_pre_fight_elo: float,
    days_since_last_fight: float,
) -> float:
    if column in {"opponent_elo_1", "opponent_elo_2"}:
        return float(max(opponent_pre_fight_elo, 0.0))
    if column in {"days_since_last_fight_1", "days_since_last_fight_2"}:
        return float(max(days_since_last_fight, 0.0))
    return float(max(as_float(row.get(column), 0.0), 0.0))


def build_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    f1_stat_columns: list[str],
    f2_stat_columns: list[str],
    *,
    elo_base: float,
    elo_k_factor: float,
    elo_scale: float,
) -> pd.DataFrame:
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    if elo_k_factor <= 0:
        raise ValueError("elo_k_factor must be > 0")
    if elo_scale <= 0:
        raise ValueError("elo_scale must be > 0")

    fighter_histories: dict[str, list[list[float]]] = {}
    fighter_last_fight_date: dict[str, pd.Timestamp] = {}
    fighter_elo: dict[str, float] = {}
    fighter_career_sig_absorbed: dict[str, float] = {}

    rows: list[dict[str, object]] = []
    num_features = len(f1_stat_columns)

    for _, row in df.iterrows():
        f1_id = row["fighter_1_id"]
        f2_id = row["fighter_2_id"]
        event_date = pd.Timestamp(row["event_date"])

        pre_elo_f1 = float(fighter_elo.get(f1_id, elo_base))
        pre_elo_f2 = float(fighter_elo.get(f2_id, elo_base))

        prev_date_f1 = fighter_last_fight_date.get(f1_id)
        prev_date_f2 = fighter_last_fight_date.get(f2_id)
        days_f1 = float(max((event_date - prev_date_f1).days, 0)) if prev_date_f1 is not None else 0.0
        days_f2 = float(max((event_date - prev_date_f2).days, 0)) if prev_date_f2 is not None else 0.0

        age_days_f1 = float(max(as_float(row.get("fighter_1_age_days"), 0.0), 0.0))
        age_days_f2 = float(max(as_float(row.get("fighter_2_age_days"), 0.0), 0.0))
        age_days_diff = age_days_f1 - age_days_f2
        age_diff_years = age_days_diff / 365.25
        age_gap_over_5y = float(abs(age_diff_years) >= 5.0)
        height_cm_f1 = float(max(as_float(row.get("fighter_1_height_cm"), 0.0), 0.0))
        height_cm_f2 = float(max(as_float(row.get("fighter_2_height_cm"), 0.0), 0.0))
        height_cm_diff = height_cm_f1 - height_cm_f2
        reach_cm_f1 = float(max(as_float(row.get("fighter_1_reach_cm"), 0.0), 0.0))
        reach_cm_f2 = float(max(as_float(row.get("fighter_2_reach_cm"), 0.0), 0.0))
        reach_cm_diff = reach_cm_f1 - reach_cm_f2
        career_sig_absorbed_f1 = float(max(fighter_career_sig_absorbed.get(f1_id, 0.0), 0.0))
        career_sig_absorbed_f2 = float(max(fighter_career_sig_absorbed.get(f2_id, 0.0), 0.0))
        career_sig_absorbed_diff = career_sig_absorbed_f1 - career_sig_absorbed_f2

        f1_history = fighter_histories.get(f1_id, [])
        f2_history = fighter_histories.get(f2_id, [])

        f1_seq = f1_history[-sequence_length:]
        f2_seq = f2_history[-sequence_length:]

        f1_history_len = len(f1_seq)
        f2_history_len = len(f2_seq)

        if len(f1_seq) < sequence_length:
            padding = [[0.0] * num_features for _ in range(sequence_length - len(f1_seq))]
            f1_seq = padding + f1_seq
        if len(f2_seq) < sequence_length:
            padding = [[0.0] * num_features for _ in range(sequence_length - len(f2_seq))]
            f2_seq = padding + f2_seq

        sequence_row: dict[str, object] = {
            "fight_id": row["fight_id"],
            "event_date": event_date.strftime("%Y-%m-%d"),
            "outcome_label": row["outcome_label"],
            "f1_id": f1_id,
            "f2_id": f2_id,
            "f1_history_len": int(f1_history_len),
            "f2_history_len": int(f2_history_len),
            "f1_pre_fight_elo": pre_elo_f1,
            "f2_pre_fight_elo": pre_elo_f2,
            "elo_diff_f1_minus_f2": pre_elo_f1 - pre_elo_f2,
            "f1_days_since_last_fight": days_f1,
            "f2_days_since_last_fight": days_f2,
            "days_since_last_fight_diff_f1_minus_f2": days_f1 - days_f2,
            "fighter_1_age_days": age_days_f1,
            "fighter_2_age_days": age_days_f2,
            "age_days_diff_f1_minus_f2": age_days_diff,
            "age_diff_years_f1_minus_f2": age_diff_years,
            "age_gap_over_5y": age_gap_over_5y,
            "fighter_1_height_cm": height_cm_f1,
            "fighter_2_height_cm": height_cm_f2,
            "height_cm_diff_f1_minus_f2": height_cm_diff,
            "fighter_1_reach_cm": reach_cm_f1,
            "fighter_2_reach_cm": reach_cm_f2,
            "reach_cm_diff_f1_minus_f2": reach_cm_diff,
            "f1_career_significant_strikes_absorbed": career_sig_absorbed_f1,
            "f2_career_significant_strikes_absorbed": career_sig_absorbed_f2,
            "career_significant_strikes_absorbed_diff_f1_minus_f2": career_sig_absorbed_diff,
            "f1_career_significant_strikes_absorbed_over_1500": float(career_sig_absorbed_f1 >= 1500.0),
            "f2_career_significant_strikes_absorbed_over_1500": float(career_sig_absorbed_f2 >= 1500.0),
        }

        for step, stats in enumerate(f1_seq):
            for stat_idx, value in enumerate(stats):
                sequence_row[f"f1_seq_{step}_stat_{stat_idx}"] = float(value)

        for step, stats in enumerate(f2_seq):
            for stat_idx, value in enumerate(stats):
                sequence_row[f"f2_seq_{step}_stat_{stat_idx}"] = float(value)

        rows.append(sequence_row)

        f1_today = [
            resolve_sequence_stat(
                row,
                col,
                opponent_pre_fight_elo=pre_elo_f2,
                days_since_last_fight=days_f1,
            )
            for col in f1_stat_columns
        ]
        f2_today = [
            resolve_sequence_stat(
                row,
                col,
                opponent_pre_fight_elo=pre_elo_f1,
                days_since_last_fight=days_f2,
            )
            for col in f2_stat_columns
        ]
        fighter_histories.setdefault(f1_id, []).append(f1_today)
        fighter_histories.setdefault(f2_id, []).append(f2_today)

        outcome = str(row["outcome_label"])
        score_f1: float | None
        if outcome == "fighter_1_win":
            score_f1 = 1.0
        elif outcome == "fighter_2_win":
            score_f1 = 0.0
        elif outcome == "draw":
            score_f1 = 0.5
        else:
            score_f1 = None

        if score_f1 is not None:
            expected_f1 = expected_score(pre_elo_f1, pre_elo_f2, elo_scale)
            expected_f2 = 1.0 - expected_f1
            score_f2 = 1.0 - score_f1
            fighter_elo[f1_id] = pre_elo_f1 + (elo_k_factor * (score_f1 - expected_f1))
            fighter_elo[f2_id] = pre_elo_f2 + (elo_k_factor * (score_f2 - expected_f2))
        else:
            fighter_elo[f1_id] = pre_elo_f1
            fighter_elo[f2_id] = pre_elo_f2

        fighter_last_fight_date[f1_id] = event_date
        fighter_last_fight_date[f2_id] = event_date
        # Update cumulative absorbed volume after this fight so next row remains strictly pre-fight.
        fighter_career_sig_absorbed[f1_id] = career_sig_absorbed_f1 + float(
            max(as_float(row.get("sig_str_2_landed"), 0.0), 0.0)
        )
        fighter_career_sig_absorbed[f2_id] = career_sig_absorbed_f2 + float(
            max(as_float(row.get("sig_str_1_landed"), 0.0), 0.0)
        )

    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("Loading raw UFC fight details from %s", args.input_csv)
    raw_df = load_raw_dataframe(args.input_csv)
    enriched_df = merge_optional_metadata(raw_df, args.metadata_csv)

    prepared_df, f1_stat_columns, f2_stat_columns = prepare_dataframe(
        enriched_df,
        performance_stats=DEFAULT_PERFORMANCE_STATS,
        drop_nonstandard_outcomes=args.drop_nonstandard_outcomes,
    )

    logging.info(
        "Prepared %d fights | date range %s -> %s",
        len(prepared_df),
        prepared_df["event_date"].min().date().isoformat() if not prepared_df.empty else "n/a",
        prepared_df["event_date"].max().date().isoformat() if not prepared_df.empty else "n/a",
    )
    age_coverage = float(
        ((prepared_df["fighter_1_age_days"] > 0) & (prepared_df["fighter_2_age_days"] > 0)).mean()
    )
    logging.info("Age availability (both fighters in bout): %.2f%%", age_coverage * 100.0)
    reach_coverage = float(
        ((prepared_df["fighter_1_reach_cm"] > 0) & (prepared_df["fighter_2_reach_cm"] > 0)).mean()
    )
    logging.info("Reach availability (both fighters in bout): %.2f%%", reach_coverage * 100.0)

    if prepared_df.empty:
        raise ValueError("No valid fights left after preprocessing.")

    seq_df = build_sequences(
        prepared_df,
        sequence_length=args.sequence_length,
        f1_stat_columns=f1_stat_columns,
        f2_stat_columns=f2_stat_columns,
        elo_base=args.elo_base,
        elo_k_factor=args.elo_k_factor,
        elo_scale=args.elo_scale,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_csv(args.output_csv, index=False)

    all_seq_cols = [c for c in seq_df.columns if c.startswith("f1_seq_") or c.startswith("f2_seq_")]
    zero_history_rows = int((seq_df[all_seq_cols].abs().sum(axis=1) == 0).sum())
    logging.info(
        "Saved %d sequence rows to %s | fully-empty history rows=%d (%.2f%%)",
        len(seq_df),
        args.output_csv,
        zero_history_rows,
        100.0 * zero_history_rows / max(len(seq_df), 1),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
