#!/usr/bin/env python3
"""Audit raw UFC fight data and sequence data for LSTM training readiness."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


LANDED_ATTEMPTED_PAIRS = [
    ("sig_str_1_landed", "sig_str_1_attempted"),
    ("sig_str_2_landed", "sig_str_2_attempted"),
    ("td_1_landed", "td_1_attempted"),
    ("td_2_landed", "td_2_attempted"),
]


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Audit scraper + sequence outputs for LSTM quality")
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=root_dir / "data" / "ufc_fight_details_lstm.csv",
        help="Raw fight-details CSV",
    )
    parser.add_argument(
        "--seq-csv",
        type=Path,
        default=root_dir / "data" / "ufc_lstm_sequences.csv",
        help="Sequence CSV",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Expected sequence length",
    )
    return parser.parse_args()


def detect_seq_layout(columns: list[str]) -> tuple[int, int, list[str], list[str]]:
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
        raise ValueError("No sequence columns found.")
    if f1_steps != f2_steps or f1_stats != f2_stats:
        raise ValueError("f1/f2 sequence layout mismatch.")

    steps = sorted(f1_steps)
    stats = sorted(f1_stats)
    f1_cols = [f"f1_seq_{s}_stat_{i}" for s in steps for i in stats]
    f2_cols = [f"f2_seq_{s}_stat_{i}" for s in steps for i in stats]
    return len(steps), len(stats), f1_cols, f2_cols


def main() -> int:
    args = parse_args()

    raw = pd.read_csv(args.raw_csv)
    seq = pd.read_csv(args.seq_csv)

    print("== RAW DATA ==")
    print("rows:", len(raw), "unique_fight_id:", raw["fight_id"].nunique())
    print("date_min:", pd.to_datetime(raw["event_date"], errors="coerce").min())
    print("date_max:", pd.to_datetime(raw["event_date"], errors="coerce").max())
    print("outcome_counts:", raw["outcome_label"].value_counts(dropna=False).to_dict())

    missing_cols = [
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
    miss = raw[missing_cols].isna().mean().sort_values(ascending=False)
    print("top_missing_rates:")
    print(miss.head(8).to_string())

    for landed, attempted in LANDED_ATTEMPTED_PAIRS:
        bad = int(
            (
                pd.to_numeric(raw[landed], errors="coerce")
                > pd.to_numeric(raw[attempted], errors="coerce")
            )
            .fillna(False)
            .sum()
        )
        print(f"invalid_{landed}_gt_{attempted}:", bad)

    winner_f1 = int(
        (
            (raw["winner_fighter_id"].fillna("") == raw["fighter_1_id"].fillna(""))
            & raw["winner_fighter_id"].fillna("").ne("")
        ).sum()
    )
    winner_f2 = int(
        (
            (raw["winner_fighter_id"].fillna("") == raw["fighter_2_id"].fillna(""))
            & raw["winner_fighter_id"].fillna("").ne("")
        ).sum()
    )
    print("winner_eq_f1:", winner_f1, "winner_eq_f2:", winner_f2)

    print("\n== SEQUENCE DATA ==")
    seq_len, num_stats, f1_cols, f2_cols = detect_seq_layout(seq.columns.tolist())
    print("rows:", len(seq), "unique_fight_id:", seq["fight_id"].nunique())
    print("detected_seq_len:", seq_len, "stats_per_step:", num_stats)
    if seq_len != args.sequence_length:
        print(f"WARNING: expected sequence_length={args.sequence_length}, found {seq_len}")

    print("outcome_counts:", seq["outcome_label"].value_counts(dropna=False).to_dict())
    print("sequence_null_rate:", float(seq[f1_cols + f2_cols].isna().mean().mean()))
    all_zero_rate = float((seq[f1_cols + f2_cols].abs().sum(axis=1) == 0).mean())
    print("all_zero_sequence_rate:", all_zero_rate)

    if {"f1_history_len", "f2_history_len"}.issubset(seq.columns):
        f1_min = int(pd.to_numeric(seq["f1_history_len"], errors="coerce").fillna(0).min())
        f1_max = int(pd.to_numeric(seq["f1_history_len"], errors="coerce").fillna(0).max())
        f2_min = int(pd.to_numeric(seq["f2_history_len"], errors="coerce").fillna(0).min())
        f2_max = int(pd.to_numeric(seq["f2_history_len"], errors="coerce").fillna(0).max())
        print("f1_history_len_range:", (f1_min, f1_max))
        print("f2_history_len_range:", (f2_min, f2_max))

    print("\n== CROSS-DATA CHECKS ==")
    raw_ids = set(raw["fight_id"].astype(str))
    seq_ids = set(seq["fight_id"].astype(str))
    print("raw_not_in_seq:", len(raw_ids - seq_ids))
    print("seq_not_in_raw:", len(seq_ids - raw_ids))

    severe = 0
    severe += int(raw["fight_id"].nunique() != len(raw))
    severe += int(seq["fight_id"].nunique() != len(seq))
    severe += int(len(raw_ids - seq_ids) > 0 or len(seq_ids - raw_ids) > 0)
    severe += int(any(
        int(
            (
                pd.to_numeric(raw[l], errors="coerce")
                > pd.to_numeric(raw[a], errors="coerce")
            )
            .fillna(False)
            .sum()
        )
        > 0
        for l, a in LANDED_ATTEMPTED_PAIRS
    ))

    if severe > 0:
        print("\nAUDIT_RESULT: FAIL (severe issues detected)")
        return 1
    print("\nAUDIT_RESULT: PASS (no severe consistency violations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
