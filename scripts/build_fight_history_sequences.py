#!/usr/bin/env python3
"""Build chronological fighter-history sequences for LSTM training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


DEFAULT_PERFORMANCE_STATS = [
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
        "--sequence-length",
        type=int,
        default=5,
        help="Number of prior fights to keep per fighter.",
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


def load_raw_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path)


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
    required_columns = sorted(set(base_required + f1_stat_columns + f2_stat_columns))

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
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

    df = df.sort_values(["event_date", "bout_index", "fight_id"]).reset_index(drop=True)

    duplicate_count = int(df.duplicated(subset=["fight_id"], keep="last").sum())
    if duplicate_count > 0:
        logging.warning("Found %d duplicate fight_id rows; keeping last occurrence.", duplicate_count)
        df = df.drop_duplicates(subset=["fight_id"], keep="last").copy()

    df = df.sort_values(["event_date", "bout_index", "fight_id"]).reset_index(drop=True)
    return df, f1_stat_columns, f2_stat_columns


def build_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    f1_stat_columns: list[str],
    f2_stat_columns: list[str],
) -> pd.DataFrame:
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")

    fighter_histories: dict[str, list[list[float]]] = {}
    rows: list[dict[str, object]] = []
    num_features = len(f1_stat_columns)

    for _, row in df.iterrows():
        f1_id = row["fighter_1_id"]
        f2_id = row["fighter_2_id"]

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
            "event_date": row["event_date"].strftime("%Y-%m-%d"),
            "outcome_label": row["outcome_label"],
            "f1_id": f1_id,
            "f2_id": f2_id,
            "f1_history_len": int(f1_history_len),
            "f2_history_len": int(f2_history_len),
        }

        for step, stats in enumerate(f1_seq):
            for stat_idx, value in enumerate(stats):
                sequence_row[f"f1_seq_{step}_stat_{stat_idx}"] = float(value)

        for step, stats in enumerate(f2_seq):
            for stat_idx, value in enumerate(stats):
                sequence_row[f"f2_seq_{step}_stat_{stat_idx}"] = float(value)

        rows.append(sequence_row)

        f1_today = [float(row[col]) for col in f1_stat_columns]
        f2_today = [float(row[col]) for col in f2_stat_columns]
        fighter_histories.setdefault(f1_id, []).append(f1_today)
        fighter_histories.setdefault(f2_id, []).append(f2_today)

    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("Loading raw UFC fight details from %s", args.input_csv)
    raw_df = load_raw_dataframe(args.input_csv)

    prepared_df, f1_stat_columns, f2_stat_columns = prepare_dataframe(
        raw_df,
        performance_stats=DEFAULT_PERFORMANCE_STATS,
        drop_nonstandard_outcomes=args.drop_nonstandard_outcomes,
    )

    logging.info(
        "Prepared %d fights | date range %s -> %s",
        len(prepared_df),
        prepared_df["event_date"].min().date().isoformat() if not prepared_df.empty else "n/a",
        prepared_df["event_date"].max().date().isoformat() if not prepared_df.empty else "n/a",
    )

    if prepared_df.empty:
        raise ValueError("No valid fights left after preprocessing.")

    seq_df = build_sequences(
        prepared_df,
        sequence_length=args.sequence_length,
        f1_stat_columns=f1_stat_columns,
        f2_stat_columns=f2_stat_columns,
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
