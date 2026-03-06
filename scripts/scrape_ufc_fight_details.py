#!/usr/bin/env python3
"""Resumable raw UFC fight-details scraper for sequence/LSTM training."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scrape_ufc_fights import (
    DEFAULT_USER_AGENT,
    MISSING_SCRAPER_DEPS,
    EventMeta,
    FightDetails,
    FightStub,
    HttpClient,
    clean_text,
    compute_fight_duration_seconds,
    method_category,
    parse_event_fights,
    parse_events_index,
    parse_fight_details,
    utc_now_iso,
    winner_from_status,
)


RAW_CSV_COLUMNS = [
    "fight_id",
    "event_id",
    "event_name",
    "event_date",
    "event_city",
    "event_state",
    "event_country",
    "bout_index",
    "is_main_event",
    "weight_class",
    "gender",
    "is_title_bout",
    "scheduled_rounds",
    "time_format",
    "round_ended",
    "time_ended",
    "fight_duration_seconds",
    "result_method",
    "result_method_category",
    "fighter_1_id",
    "fighter_1_name",
    "fighter_1_status",
    "fighter_2_id",
    "fighter_2_name",
    "fighter_2_status",
    "winner_fighter_id",
    "winner_name",
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
    "scrape_timestamp_utc",
]

RAW_TEXT_COLUMNS = {
    "fight_id",
    "event_id",
    "event_name",
    "event_date",
    "event_city",
    "event_state",
    "event_country",
    "weight_class",
    "gender",
    "time_format",
    "time_ended",
    "result_method",
    "result_method_category",
    "fighter_1_id",
    "fighter_1_name",
    "fighter_1_status",
    "fighter_2_id",
    "fighter_2_name",
    "fighter_2_status",
    "winner_fighter_id",
    "winner_name",
    "outcome_label",
    "scrape_timestamp_utc",
}

RAW_INTEGER_COLUMNS = {
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
}

LANDED_ATTEMPTED_PAIRS = [
    ("sig_str_1_landed", "sig_str_1_attempted"),
    ("sig_str_2_landed", "sig_str_2_attempted"),
    ("td_1_landed", "td_1_attempted"),
    ("td_2_landed", "td_2_attempted"),
]

NONNEGATIVE_COLUMNS = [
    "fight_duration_seconds",
    "round_ended",
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

VALID_OUTCOMES = {"fighter_1_win", "fighter_2_win", "draw", "no_contest", "unknown"}


def sanitize_raw_fight_row(row: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for col in NONNEGATIVE_COLUMNS:
        value = row.get(col)
        if value is None:
            continue
        if isinstance(value, (int, float)) and value < 0:
            row[col] = 0
            issues.append(f"{col}<0 coerced to 0")

    for landed_col, attempted_col in LANDED_ATTEMPTED_PAIRS:
        landed = row.get(landed_col)
        attempted = row.get(attempted_col)
        if landed is None or attempted is None:
            continue
        if attempted < landed:
            row[attempted_col] = landed
            issues.append(f"{attempted_col}<{landed_col} coerced to {landed}")

    winner_id = clean_text(str(row.get("winner_fighter_id") or ""))
    fighter_1_id = clean_text(str(row.get("fighter_1_id") or ""))
    fighter_2_id = clean_text(str(row.get("fighter_2_id") or ""))
    outcome_label = clean_text(str(row.get("outcome_label") or ""))
    if outcome_label not in VALID_OUTCOMES:
        row["outcome_label"] = "unknown"
        issues.append("invalid outcome_label coerced to unknown")
    if winner_id and winner_id not in {fighter_1_id, fighter_2_id}:
        row["winner_fighter_id"] = ""
        row["winner_name"] = ""
        if row.get("outcome_label") in {"fighter_1_win", "fighter_2_win"}:
            row["outcome_label"] = "unknown"
        issues.append("winner_fighter_id not in fighter IDs; winner cleared")
    return issues


def parse_date_filter(value: Optional[str]) -> Optional[dt.date]:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc


def coalesce_max_int(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def merged_fight_stats(stub: FightStub, details: FightDetails) -> dict[str, Optional[int]]:
    return {
        "kd_1": coalesce_max_int(details.kd_1, stub.kd_1),
        "kd_2": coalesce_max_int(details.kd_2, stub.kd_2),
        "sig_str_1_landed": coalesce_max_int(details.sig_str_1_landed, stub.sig_str_1_landed),
        "sig_str_1_attempted": coalesce_max_int(details.sig_str_1_attempted, stub.sig_str_1_attempted),
        "sig_str_2_landed": coalesce_max_int(details.sig_str_2_landed, stub.sig_str_2_landed),
        "sig_str_2_attempted": coalesce_max_int(details.sig_str_2_attempted, stub.sig_str_2_attempted),
        "td_1_landed": coalesce_max_int(details.td_1_landed, stub.td_1_landed),
        "td_1_attempted": coalesce_max_int(details.td_1_attempted, stub.td_1_attempted),
        "td_2_landed": coalesce_max_int(details.td_2_landed, stub.td_2_landed),
        "td_2_attempted": coalesce_max_int(details.td_2_attempted, stub.td_2_attempted),
        "sub_1": coalesce_max_int(details.sub_1, stub.sub_1),
        "sub_2": coalesce_max_int(details.sub_2, stub.sub_2),
        "ctrl_seconds_1": coalesce_max_int(details.ctrl_seconds_1, stub.ctrl_seconds_1),
        "ctrl_seconds_2": coalesce_max_int(details.ctrl_seconds_2, stub.ctrl_seconds_2),
    }


def build_raw_fight_row(event: EventMeta, stub: FightStub, details: FightDetails) -> dict[str, Any]:
    winner_id, winner_name, outcome_label = winner_from_status(
        stub.fighter_1_id,
        stub.fighter_1_name,
        stub.fighter_1_status,
        stub.fighter_2_id,
        stub.fighter_2_name,
        stub.fighter_2_status,
    )
    stats = merged_fight_stats(stub, details)
    fight_duration_seconds = compute_fight_duration_seconds(stub.round_ended, stub.time_ended)
    row: dict[str, Any] = {
        "fight_id": stub.fight_id,
        "event_id": event.event_id,
        "event_name": event.event_name,
        "event_date": event.event_date.isoformat(),
        "event_city": event.event_city,
        "event_state": event.event_state,
        "event_country": event.event_country,
        "bout_index": stub.bout_index,
        "is_main_event": 1 if stub.bout_index == 1 else 0,
        "weight_class": details.weight_class or stub.weight_class,
        "gender": details.gender,
        "is_title_bout": details.is_title_bout,
        "scheduled_rounds": details.scheduled_rounds,
        "time_format": details.time_format,
        "round_ended": stub.round_ended,
        "time_ended": stub.time_ended,
        "fight_duration_seconds": fight_duration_seconds,
        "result_method": stub.method,
        "result_method_category": method_category(stub.method),
        "fighter_1_id": stub.fighter_1_id,
        "fighter_1_name": stub.fighter_1_name,
        "fighter_1_status": stub.fighter_1_status,
        "fighter_2_id": stub.fighter_2_id,
        "fighter_2_name": stub.fighter_2_name,
        "fighter_2_status": stub.fighter_2_status,
        "winner_fighter_id": winner_id,
        "winner_name": winner_name,
        "outcome_label": outcome_label,
        "kd_1": stats["kd_1"],
        "kd_2": stats["kd_2"],
        "sig_str_1_landed": stats["sig_str_1_landed"],
        "sig_str_1_attempted": stats["sig_str_1_attempted"],
        "sig_str_2_landed": stats["sig_str_2_landed"],
        "sig_str_2_attempted": stats["sig_str_2_attempted"],
        "td_1_landed": stats["td_1_landed"],
        "td_1_attempted": stats["td_1_attempted"],
        "td_2_landed": stats["td_2_landed"],
        "td_2_attempted": stats["td_2_attempted"],
        "sub_1": stats["sub_1"],
        "sub_2": stats["sub_2"],
        "ctrl_seconds_1": stats["ctrl_seconds_1"],
        "ctrl_seconds_2": stats["ctrl_seconds_2"],
        "scrape_timestamp_utc": utc_now_iso(),
    }
    return row


class RawCheckpointStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._setup()

    def close(self) -> None:
        self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

    def _raw_column_sql_type(self, column: str) -> str:
        if column in RAW_TEXT_COLUMNS:
            return "TEXT"
        if column in RAW_INTEGER_COLUMNS:
            return "INTEGER"
        return "REAL"

    def _ensure_table_columns(self, table: str, column_defs: dict[str, str]) -> None:
        existing = {
            row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for column, column_type in column_defs.items():
            if column in existing:
                continue
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def _setup(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;

            CREATE TABLE IF NOT EXISTS processed_events (
                event_id TEXT PRIMARY KEY,
                event_url TEXT NOT NULL,
                event_name TEXT NOT NULL,
                event_date TEXT NOT NULL,
                processed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS raw_fights (
                fight_id TEXT PRIMARY KEY
            );
            """
        )
        self._ensure_table_columns(
            "raw_fights",
            {
                column: self._raw_column_sql_type(column)
                for column in RAW_CSV_COLUMNS
                if column != "fight_id"
            },
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_raw_fights_event_date
            ON raw_fights(event_date, bout_index, fight_id)
            """
        )
        self.conn.commit()

    def processed_event_ids(self) -> set[str]:
        rows = self.conn.execute("SELECT event_id FROM processed_events").fetchall()
        return {str(row["event_id"]) for row in rows}

    def mark_event_processed(self, event: EventMeta) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO processed_events
            (event_id, event_url, event_name, event_date, processed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.event_url,
                event.event_name,
                event.event_date.isoformat(),
                utc_now_iso(),
            ),
        )

    def existing_fight_ids_for_event(self, event_id: str) -> set[str]:
        rows = self.conn.execute(
            "SELECT fight_id FROM raw_fights WHERE event_id = ?",
            (event_id,),
        ).fetchall()
        return {str(row["fight_id"]) for row in rows}

    def insert_raw_fight(self, row: dict[str, Any]) -> None:
        placeholders = ", ".join(["?"] * len(RAW_CSV_COLUMNS))
        columns = ", ".join(RAW_CSV_COLUMNS)
        values = [row.get(column) for column in RAW_CSV_COLUMNS]
        update_columns = [column for column in RAW_CSV_COLUMNS if column != "fight_id"]
        update_clause = ", ".join([f"{column}=excluded.{column}" for column in update_columns])
        self.conn.execute(
            (
                f"INSERT INTO raw_fights ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(fight_id) DO UPDATE SET {update_clause}"
            ),
            values,
        )

    def raw_fights_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS c FROM raw_fights").fetchone()
        return int(row["c"])

    def export_csv(self, csv_path: Path) -> int:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        query = (
            "SELECT "
            + ", ".join(RAW_CSV_COLUMNS)
            + " FROM raw_fights ORDER BY event_date, bout_index, fight_id"
        )
        rows = self.conn.execute(query)
        count = 0
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(RAW_CSV_COLUMNS)
            for row in rows:
                writer.writerow([row[column] for column in RAW_CSV_COLUMNS])
                count += 1
        return count


def process_event(
    *,
    store: RawCheckpointStore,
    client: HttpClient,
    event: EventMeta,
    max_fights_remaining: Optional[int],
    commit_every: int,
    stop_on_fight_error: bool,
    refresh_existing_fights: bool,
) -> tuple[int, bool, bool]:
    stubs = parse_event_fights(client, event)
    if not stubs:
        logging.warning("No fights found for event %s (%s)", event.event_name, event.event_url)
        return 0, True, False

    existing_fight_ids = store.existing_fight_ids_for_event(event.event_id)
    inserted = 0
    pending_writes = 0
    uncommitted_fights = 0
    uncommitted_fight_ids: list[str] = []
    event_completed = True
    stopped_by_limit = False

    for stub in stubs:
        if max_fights_remaining is not None and max_fights_remaining <= 0:
            stopped_by_limit = True
            event_completed = False
            break
        if stub.fight_id in existing_fight_ids and not refresh_existing_fights:
            continue
        try:
            details = parse_fight_details(client, stub.fight_url, stub.weight_class)
            row = build_raw_fight_row(event, stub, details)
            issues = sanitize_raw_fight_row(row)
            if issues:
                logging.warning("Row sanitation for fight %s: %s", stub.fight_id, "; ".join(issues))
            store.insert_raw_fight(row)
            inserted += 1
            pending_writes += 1
            uncommitted_fights += 1
            uncommitted_fight_ids.append(stub.fight_id)
            existing_fight_ids.add(stub.fight_id)
            if max_fights_remaining is not None:
                max_fights_remaining -= 1
            if pending_writes >= commit_every:
                store.commit()
                pending_writes = 0
                uncommitted_fights = 0
                uncommitted_fight_ids.clear()
        except Exception as exc:
            store.rollback()
            if uncommitted_fights > 0:
                inserted = max(inserted - uncommitted_fights, 0)
                if max_fights_remaining is not None:
                    max_fights_remaining += uncommitted_fights
            for fight_id in uncommitted_fight_ids:
                existing_fight_ids.discard(fight_id)
            pending_writes = 0
            uncommitted_fights = 0
            uncommitted_fight_ids.clear()
            event_completed = False
            logging.error(
                "Failed fight %s (%s) in event %s: %s",
                stub.fight_id,
                stub.fight_url,
                event.event_name,
                exc,
            )
            if stop_on_fight_error:
                break
            continue

    if pending_writes > 0:
        store.commit()
    return inserted, event_completed, stopped_by_limit


def filter_events(
    events: Iterable[EventMeta],
    *,
    processed_event_ids: set[str],
    include_processed_events: bool,
    start_date: Optional[dt.date],
    end_date: Optional[dt.date],
    max_events: Optional[int],
) -> list[EventMeta]:
    today = dt.date.today()
    filtered: list[EventMeta] = []
    for event in events:
        if event.event_date > today:
            continue
        if start_date and event.event_date < start_date:
            continue
        if end_date and event.event_date > end_date:
            continue
        if (not include_processed_events) and event.event_id in processed_event_ids:
            continue
        filtered.append(event)
        if max_events is not None and len(filtered) >= max_events:
            break
    return filtered


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable UFC raw fight-details scraper for sequence/LSTM datasets."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/ufc_fight_details_lstm.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--checkpoint-db",
        type=Path,
        default=Path("data/checkpoints/ufc_fight_details_checkpoint.sqlite"),
        help="SQLite checkpoint path.",
    )
    parser.add_argument("--max-events", type=int, default=None, help="Max events to process this run.")
    parser.add_argument("--max-fights", type=int, default=None, help="Max fights to process this run.")
    parser.add_argument(
        "--start-date",
        type=parse_date_filter,
        default=None,
        help="Only process events on/after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date_filter,
        default=None,
        help="Only process events on/before this date (YYYY-MM-DD).",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.05, help="Delay after each request.")
    parser.add_argument("--timeout-seconds", type=int, default=30, help="Per-request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per request.")
    parser.add_argument("--backoff-seconds", type=float, default=1.0, help="Retry backoff base seconds.")
    parser.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT, help="HTTP User-Agent.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity level.",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=10,
        help="Commit every N processed fights.",
    )
    parser.add_argument(
        "--stop-on-fight-error",
        action="store_true",
        help="Stop processing current event when one fight fails to parse.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip scraping and export the current checkpoint DB to CSV.",
    )
    parser.add_argument(
        "--refresh-processed-events",
        action="store_true",
        help="Reprocess events even if already marked as processed in checkpoint DB.",
    )
    parser.add_argument(
        "--refresh-existing-fights",
        action="store_true",
        help="Reparse fights even if fight_id already exists in checkpoint DB (upsert row values).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    store = RawCheckpointStore(args.checkpoint_db)
    try:
        if args.export_only:
            rows = store.export_csv(args.output_csv)
            logging.info("Exported %s rows to %s", rows, args.output_csv)
            return 0

        commit_every = max(1, int(args.commit_every))
        if MISSING_SCRAPER_DEPS:
            raise SystemExit("Missing dependencies. Run: pip install -r requirements.txt")

        client = HttpClient(
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            backoff_seconds=args.backoff_seconds,
            sleep_seconds=args.sleep_seconds,
            user_agent=args.user_agent,
        )

        logging.info("Fetching completed UFC events index...")
        all_events = parse_events_index(client)
        if not all_events:
            logging.error("No events discovered. Site structure may have changed.")
            return 1

        pending_events = filter_events(
            all_events,
            processed_event_ids=store.processed_event_ids(),
            include_processed_events=args.refresh_processed_events,
            start_date=args.start_date,
            end_date=args.end_date,
            max_events=args.max_events,
        )
        if not pending_events:
            logging.info("No new events to scrape. Refreshing CSV export.")
            rows = store.export_csv(args.output_csv)
            logging.info("CSV refreshed with %s rows.", rows)
            return 0

        logging.info(
            "Processing %s event(s). Existing fights in checkpoint DB: %s | refresh_processed_events=%s | refresh_existing_fights=%s",
            len(pending_events),
            store.raw_fights_count(),
            args.refresh_processed_events,
            args.refresh_existing_fights,
        )

        total_rows_written = 0
        processed_events = 0
        max_fights_remaining = args.max_fights

        for event in pending_events:
            logging.info(
                "Event: %s (%s) - %s",
                event.event_name,
                event.event_date.isoformat(),
                event.event_id,
            )
            inserted, event_completed, stopped_by_limit = process_event(
                store=store,
                client=client,
                event=event,
                max_fights_remaining=max_fights_remaining,
                commit_every=commit_every,
                stop_on_fight_error=args.stop_on_fight_error,
                refresh_existing_fights=args.refresh_existing_fights,
            )
            total_rows_written += inserted
            if max_fights_remaining is not None:
                max_fights_remaining -= inserted

            if event_completed:
                store.mark_event_processed(event)
                store.commit()
                processed_events += 1
                logging.info("Event completed. Rows written: %s", inserted)
            else:
                logging.warning("Event not fully completed, checkpoint retained at fight-level.")

            if stopped_by_limit:
                logging.info("Stopped early because --max-fights limit was reached.")
                break

        rows = store.export_csv(args.output_csv)
        logging.info(
            "Run complete. Rows written: %s | Events completed this run: %s | Total CSV rows: %s",
            total_rows_written,
            processed_events,
            rows,
        )
        logging.info("CSV path: %s", args.output_csv)
        logging.info("Checkpoint DB path: %s", args.checkpoint_db)
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
