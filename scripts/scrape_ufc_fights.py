#!/usr/bin/env python3
"""Resumable UFCStats scraper for fight-level, RNN-ready training data.

This script scrapes UFC fight history from UFCStats and writes a CSV with
pre-fight features plus outcome labels.

Key behavior:
- Checkpointed in SQLite so reruns continue from where they stopped.
- Chunk controls (`--max-events`, `--max-fights`) for long/history runs.
- Per-fight dedupe via `fight_id` primary key.
- Feature set is focused on pre-fight, leakage-safe modeling signals.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import logging
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

MISSING_SCRAPER_DEPS = False
try:
    import requests
    from requests.adapters import HTTPAdapter
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - runtime guard
    requests = None  # type: ignore[assignment]
    HTTPAdapter = Any  # type: ignore[assignment]
    BeautifulSoup = Any  # type: ignore[assignment]
    MISSING_SCRAPER_DEPS = True


EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"
DATE_RE = re.compile(r"[A-Za-z]+\s+\d{1,2},\s+\d{4}")
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

CSV_COLUMNS = [
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
    "fighter_1_dob",
    "fighter_1_age_days",
    "fighter_1_height_cm",
    "fighter_1_reach_cm",
    "fighter_1_stance",
    "fighter_1_wins_pre",
    "fighter_1_losses_pre",
    "fighter_1_draws_pre",
    "fighter_1_no_contests_pre",
    "fighter_1_total_fights_pre",
    "fighter_1_win_streak_pre",
    "fighter_1_days_since_last_fight",
    "fighter_1_win_rate_pre",
    "fighter_1_finish_rate_pre",
    "fighter_1_ko_wins_pre",
    "fighter_1_sub_wins_pre",
    "fighter_1_dec_wins_pre",
    "fighter_1_ko_losses_pre",
    "fighter_1_sub_losses_pre",
    "fighter_1_dec_losses_pre",
    "fighter_1_avg_fight_duration_sec_pre",
    "fighter_1_avg_rounds_fought_pre",
    "fighter_1_sig_str_landed_per_min_pre",
    "fighter_1_sig_str_absorbed_per_min_pre",
    "fighter_1_sig_str_accuracy_pre",
    "fighter_1_sig_str_defense_pre",
    "fighter_1_td_landed_per_15_pre",
    "fighter_1_td_absorbed_per_15_pre",
    "fighter_1_td_accuracy_pre",
    "fighter_1_td_defense_pre",
    "fighter_1_sub_attempts_per_15_pre",
    "fighter_1_knockdowns_per_15_pre",
    "fighter_1_control_time_per_min_pre",
    "fighter_2_id",
    "fighter_2_name",
    "fighter_2_dob",
    "fighter_2_age_days",
    "fighter_2_height_cm",
    "fighter_2_reach_cm",
    "fighter_2_stance",
    "fighter_2_wins_pre",
    "fighter_2_losses_pre",
    "fighter_2_draws_pre",
    "fighter_2_no_contests_pre",
    "fighter_2_total_fights_pre",
    "fighter_2_win_streak_pre",
    "fighter_2_days_since_last_fight",
    "fighter_2_win_rate_pre",
    "fighter_2_finish_rate_pre",
    "fighter_2_ko_wins_pre",
    "fighter_2_sub_wins_pre",
    "fighter_2_dec_wins_pre",
    "fighter_2_ko_losses_pre",
    "fighter_2_sub_losses_pre",
    "fighter_2_dec_losses_pre",
    "fighter_2_avg_fight_duration_sec_pre",
    "fighter_2_avg_rounds_fought_pre",
    "fighter_2_sig_str_landed_per_min_pre",
    "fighter_2_sig_str_absorbed_per_min_pre",
    "fighter_2_sig_str_accuracy_pre",
    "fighter_2_sig_str_defense_pre",
    "fighter_2_td_landed_per_15_pre",
    "fighter_2_td_absorbed_per_15_pre",
    "fighter_2_td_accuracy_pre",
    "fighter_2_td_defense_pre",
    "fighter_2_sub_attempts_per_15_pre",
    "fighter_2_knockdowns_per_15_pre",
    "fighter_2_control_time_per_min_pre",
    "age_days_diff_f1_minus_f2",
    "height_cm_diff_f1_minus_f2",
    "reach_cm_diff_f1_minus_f2",
    "wins_diff_f1_minus_f2",
    "losses_diff_f1_minus_f2",
    "total_fights_diff_f1_minus_f2",
    "win_streak_diff_f1_minus_f2",
    "win_rate_diff_f1_minus_f2",
    "finish_rate_diff_f1_minus_f2",
    "days_since_last_fight_diff_f1_minus_f2",
    "avg_fight_duration_sec_diff_f1_minus_f2",
    "avg_rounds_fought_diff_f1_minus_f2",
    "sig_str_landed_per_min_diff_f1_minus_f2",
    "sig_str_absorbed_per_min_diff_f1_minus_f2",
    "sig_str_accuracy_diff_f1_minus_f2",
    "sig_str_defense_diff_f1_minus_f2",
    "td_landed_per_15_diff_f1_minus_f2",
    "td_absorbed_per_15_diff_f1_minus_f2",
    "td_accuracy_diff_f1_minus_f2",
    "td_defense_diff_f1_minus_f2",
    "sub_attempts_per_15_diff_f1_minus_f2",
    "knockdowns_per_15_diff_f1_minus_f2",
    "control_time_per_min_diff_f1_minus_f2",
    "winner_fighter_id",
    "winner_name",
    "outcome_label",
    "scrape_timestamp_utc",
]

TEXT_FIGHT_COLUMNS = {
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
    "fighter_1_dob",
    "fighter_1_stance",
    "fighter_2_id",
    "fighter_2_name",
    "fighter_2_dob",
    "fighter_2_stance",
    "winner_fighter_id",
    "winner_name",
    "outcome_label",
    "scrape_timestamp_utc",
}

INTEGER_FIGHT_COLUMNS = {
    "bout_index",
    "is_main_event",
    "is_title_bout",
    "scheduled_rounds",
    "round_ended",
    "fight_duration_seconds",
    "fighter_1_wins_pre",
    "fighter_1_losses_pre",
    "fighter_1_draws_pre",
    "fighter_1_no_contests_pre",
    "fighter_1_total_fights_pre",
    "fighter_1_win_streak_pre",
    "fighter_1_ko_wins_pre",
    "fighter_1_sub_wins_pre",
    "fighter_1_dec_wins_pre",
    "fighter_1_ko_losses_pre",
    "fighter_1_sub_losses_pre",
    "fighter_1_dec_losses_pre",
    "fighter_2_wins_pre",
    "fighter_2_losses_pre",
    "fighter_2_draws_pre",
    "fighter_2_no_contests_pre",
    "fighter_2_total_fights_pre",
    "fighter_2_win_streak_pre",
    "fighter_2_ko_wins_pre",
    "fighter_2_sub_wins_pre",
    "fighter_2_dec_wins_pre",
    "fighter_2_ko_losses_pre",
    "fighter_2_sub_losses_pre",
    "fighter_2_dec_losses_pre",
}


@dataclasses.dataclass(frozen=True)
class EventMeta:
    event_id: str
    event_url: str
    event_name: str
    event_date: dt.date
    event_city: str
    event_state: str
    event_country: str


@dataclasses.dataclass(frozen=True)
class FightStub:
    fight_id: str
    fight_url: str
    bout_index: int
    fighter_1_id: str
    fighter_1_name: str
    fighter_1_url: str
    fighter_2_id: str
    fighter_2_name: str
    fighter_2_url: str
    fighter_1_status: str
    fighter_2_status: str
    weight_class: str
    method: str
    round_ended: Optional[int]
    time_ended: str
    kd_1: Optional[int]
    kd_2: Optional[int]
    sig_str_1_landed: Optional[int]
    sig_str_1_attempted: Optional[int]
    sig_str_2_landed: Optional[int]
    sig_str_2_attempted: Optional[int]
    td_1_landed: Optional[int]
    td_1_attempted: Optional[int]
    td_2_landed: Optional[int]
    td_2_attempted: Optional[int]
    sub_1: Optional[int]
    sub_2: Optional[int]
    ctrl_seconds_1: Optional[int]
    ctrl_seconds_2: Optional[int]


@dataclasses.dataclass(frozen=True)
class FighterProfile:
    fighter_id: str
    fighter_url: str
    full_name: str
    dob: Optional[dt.date]
    height_cm: Optional[float]
    reach_cm: Optional[float]
    stance: str


@dataclasses.dataclass
class FighterState:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    no_contests: int = 0
    total_fights: int = 0
    win_streak: int = 0
    ko_wins: int = 0
    sub_wins: int = 0
    dec_wins: int = 0
    ko_losses: int = 0
    sub_losses: int = 0
    dec_losses: int = 0
    total_rounds_fought: int = 0
    total_fight_seconds: int = 0
    total_knockdowns: int = 0
    total_sub_attempts: int = 0
    total_control_seconds: int = 0
    sig_str_landed: int = 0
    sig_str_attempted: int = 0
    sig_str_absorbed: int = 0
    sig_str_faced: int = 0
    td_landed: int = 0
    td_attempted: int = 0
    td_absorbed: int = 0
    td_faced: int = 0
    last_fight_date: Optional[dt.date] = None


@dataclasses.dataclass(frozen=True)
class FightDetails:
    weight_class: str
    gender: str
    is_title_bout: int
    time_format: str
    scheduled_rounds: Optional[int]


@dataclasses.dataclass(frozen=True)
class PreFightMetrics:
    win_rate: Optional[float]
    finish_rate: Optional[float]
    avg_fight_duration_sec: Optional[float]
    avg_rounds_fought: Optional[float]
    sig_str_landed_per_min: Optional[float]
    sig_str_absorbed_per_min: Optional[float]
    sig_str_accuracy: Optional[float]
    sig_str_defense: Optional[float]
    td_landed_per_15: Optional[float]
    td_absorbed_per_15: Optional[float]
    td_accuracy: Optional[float]
    td_defense: Optional[float]
    sub_attempts_per_15: Optional[float]
    knockdowns_per_15: Optional[float]
    control_time_per_min: Optional[float]


class HttpClient:
    def __init__(
        self,
        *,
        timeout_seconds: int,
        max_retries: int,
        backoff_seconds: float,
        sleep_seconds: float,
        user_agent: str,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        # Reuse TCP connections across many requests during long historical runs.
        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=0)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_soup(self, url: str) -> BeautifulSoup:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout_seconds)
                response.raise_for_status()
                return BeautifulSoup(response.text, "html.parser")
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                wait_seconds = self.backoff_seconds * (2 ** (attempt - 1))
                logging.warning(
                    "Request failed (%s/%s) for %s: %s. Retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    url,
                    exc,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
            finally:
                if self.sleep_seconds > 0:
                    time.sleep(self.sleep_seconds)
        raise RuntimeError(f"Failed to fetch {url}: {last_error}")


class CheckpointStore:
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

    def _fight_column_sql_type(self, column: str) -> str:
        if column in TEXT_FIGHT_COLUMNS:
            return "TEXT"
        if column in INTEGER_FIGHT_COLUMNS:
            return "INTEGER"
        return "REAL"

    def _ensure_table_columns(self, table: str, column_defs: dict[str, str]) -> None:
        existing = {
            row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for column, column_type in column_defs.items():
            if column in existing:
                continue
            self.conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
            )

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

            CREATE TABLE IF NOT EXISTS fighter_profiles (
                fighter_id TEXT PRIMARY KEY,
                fighter_url TEXT NOT NULL,
                full_name TEXT,
                dob TEXT,
                height_cm REAL,
                reach_cm REAL,
                stance TEXT,
                scraped_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fighter_state (
                fighter_id TEXT PRIMARY KEY,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                draws INTEGER NOT NULL DEFAULT 0,
                no_contests INTEGER NOT NULL DEFAULT 0,
                total_fights INTEGER NOT NULL DEFAULT 0,
                win_streak INTEGER NOT NULL DEFAULT 0,
                ko_wins INTEGER NOT NULL DEFAULT 0,
                sub_wins INTEGER NOT NULL DEFAULT 0,
                dec_wins INTEGER NOT NULL DEFAULT 0,
                ko_losses INTEGER NOT NULL DEFAULT 0,
                sub_losses INTEGER NOT NULL DEFAULT 0,
                dec_losses INTEGER NOT NULL DEFAULT 0,
                total_rounds_fought INTEGER NOT NULL DEFAULT 0,
                total_fight_seconds INTEGER NOT NULL DEFAULT 0,
                total_knockdowns INTEGER NOT NULL DEFAULT 0,
                total_sub_attempts INTEGER NOT NULL DEFAULT 0,
                total_control_seconds INTEGER NOT NULL DEFAULT 0,
                sig_str_landed INTEGER NOT NULL DEFAULT 0,
                sig_str_attempted INTEGER NOT NULL DEFAULT 0,
                sig_str_absorbed INTEGER NOT NULL DEFAULT 0,
                sig_str_faced INTEGER NOT NULL DEFAULT 0,
                td_landed INTEGER NOT NULL DEFAULT 0,
                td_attempted INTEGER NOT NULL DEFAULT 0,
                td_absorbed INTEGER NOT NULL DEFAULT 0,
                td_faced INTEGER NOT NULL DEFAULT 0,
                last_fight_date TEXT
            );

            CREATE TABLE IF NOT EXISTS fights (
                fight_id TEXT PRIMARY KEY
            );
            """
        )
        self._ensure_table_columns(
            "fighter_profiles",
            {
                "fighter_url": "TEXT",
                "full_name": "TEXT",
                "dob": "TEXT",
                "height_cm": "REAL",
                "reach_cm": "REAL",
                "stance": "TEXT",
                "scraped_at": "TEXT",
            },
        )
        self._ensure_table_columns(
            "fighter_state",
            {
                "wins": "INTEGER",
                "losses": "INTEGER",
                "draws": "INTEGER",
                "no_contests": "INTEGER",
                "total_fights": "INTEGER",
                "win_streak": "INTEGER",
                "ko_wins": "INTEGER",
                "sub_wins": "INTEGER",
                "dec_wins": "INTEGER",
                "ko_losses": "INTEGER",
                "sub_losses": "INTEGER",
                "dec_losses": "INTEGER",
                "total_rounds_fought": "INTEGER",
                "total_fight_seconds": "INTEGER",
                "total_knockdowns": "INTEGER",
                "total_sub_attempts": "INTEGER",
                "total_control_seconds": "INTEGER",
                "sig_str_landed": "INTEGER",
                "sig_str_attempted": "INTEGER",
                "sig_str_absorbed": "INTEGER",
                "sig_str_faced": "INTEGER",
                "td_landed": "INTEGER",
                "td_attempted": "INTEGER",
                "td_absorbed": "INTEGER",
                "td_faced": "INTEGER",
                "last_fight_date": "TEXT",
            },
        )
        self._ensure_table_columns(
            "fights",
            {
                column: self._fight_column_sql_type(column)
                for column in CSV_COLUMNS
                if column != "fight_id"
            },
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_fights_event_date
            ON fights(event_date, bout_index, fight_id)
            """
        )
        self.conn.commit()

    def event_processed(self, event_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM processed_events WHERE event_id = ? LIMIT 1", (event_id,)
        ).fetchone()
        return row is not None

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

    def fight_exists(self, fight_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM fights WHERE fight_id = ? LIMIT 1", (fight_id,)
        ).fetchone()
        return row is not None

    def insert_fight(self, row: dict[str, Any]) -> None:
        placeholders = ", ".join(["?"] * len(CSV_COLUMNS))
        columns = ", ".join(CSV_COLUMNS)
        values = [row.get(column) for column in CSV_COLUMNS]
        self.conn.execute(
            f"INSERT OR IGNORE INTO fights ({columns}) VALUES ({placeholders})", values
        )

    def existing_fight_ids_for_event(self, event_id: str) -> set[str]:
        rows = self.conn.execute(
            "SELECT fight_id FROM fights WHERE event_id = ?",
            (event_id,),
        ).fetchall()
        return {str(row["fight_id"]) for row in rows}

    def fights_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS c FROM fights").fetchone()
        return int(row["c"])

    def upsert_fighter_profile(self, profile: FighterProfile) -> None:
        self.conn.execute(
            """
            INSERT INTO fighter_profiles
            (fighter_id, fighter_url, full_name, dob, height_cm, reach_cm, stance, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fighter_id) DO UPDATE SET
                fighter_url=excluded.fighter_url,
                full_name=excluded.full_name,
                dob=excluded.dob,
                height_cm=excluded.height_cm,
                reach_cm=excluded.reach_cm,
                stance=excluded.stance,
                scraped_at=excluded.scraped_at
            """,
            (
                profile.fighter_id,
                profile.fighter_url,
                profile.full_name,
                profile.dob.isoformat() if profile.dob else None,
                profile.height_cm,
                profile.reach_cm,
                profile.stance,
                utc_now_iso(),
            ),
        )

    def get_fighter_profile(self, fighter_id: str) -> Optional[FighterProfile]:
        row = self.conn.execute(
            """
            SELECT fighter_id, fighter_url, full_name, dob, height_cm, reach_cm, stance
            FROM fighter_profiles
            WHERE fighter_id = ?
            """,
            (fighter_id,),
        ).fetchone()
        if row is None:
            return None
        return FighterProfile(
            fighter_id=row["fighter_id"],
            fighter_url=row["fighter_url"],
            full_name=row["full_name"] or "",
            dob=parse_iso_date(row["dob"]),
            height_cm=row["height_cm"],
            reach_cm=row["reach_cm"],
            stance=row["stance"] or "",
        )

    def get_fighter_state(self, fighter_id: str) -> FighterState:
        row = self.conn.execute(
            """
            SELECT
                wins, losses, draws, no_contests, total_fights, win_streak,
                ko_wins, sub_wins, dec_wins, ko_losses, sub_losses, dec_losses,
                total_rounds_fought, total_fight_seconds, total_knockdowns,
                total_sub_attempts, total_control_seconds, sig_str_landed,
                sig_str_attempted, sig_str_absorbed, sig_str_faced, td_landed,
                td_attempted, td_absorbed, td_faced, last_fight_date
            FROM fighter_state
            WHERE fighter_id = ?
            """,
            (fighter_id,),
        ).fetchone()
        if row is None:
            return FighterState()
        return FighterState(
            wins=int(row["wins"]),
            losses=int(row["losses"]),
            draws=int(row["draws"]),
            no_contests=int(row["no_contests"]),
            total_fights=int(row["total_fights"]),
            win_streak=int(row["win_streak"]),
            ko_wins=int(row["ko_wins"]),
            sub_wins=int(row["sub_wins"]),
            dec_wins=int(row["dec_wins"]),
            ko_losses=int(row["ko_losses"]),
            sub_losses=int(row["sub_losses"]),
            dec_losses=int(row["dec_losses"]),
            total_rounds_fought=int(row["total_rounds_fought"]),
            total_fight_seconds=int(row["total_fight_seconds"]),
            total_knockdowns=int(row["total_knockdowns"]),
            total_sub_attempts=int(row["total_sub_attempts"]),
            total_control_seconds=int(row["total_control_seconds"]),
            sig_str_landed=int(row["sig_str_landed"]),
            sig_str_attempted=int(row["sig_str_attempted"]),
            sig_str_absorbed=int(row["sig_str_absorbed"]),
            sig_str_faced=int(row["sig_str_faced"]),
            td_landed=int(row["td_landed"]),
            td_attempted=int(row["td_attempted"]),
            td_absorbed=int(row["td_absorbed"]),
            td_faced=int(row["td_faced"]),
            last_fight_date=parse_iso_date(row["last_fight_date"]),
        )

    def upsert_fighter_state(self, fighter_id: str, state: FighterState) -> None:
        self.conn.execute(
            """
            INSERT INTO fighter_state
            (
                fighter_id, wins, losses, draws, no_contests, total_fights, win_streak,
                ko_wins, sub_wins, dec_wins, ko_losses, sub_losses, dec_losses,
                total_rounds_fought, total_fight_seconds, total_knockdowns,
                total_sub_attempts, total_control_seconds, sig_str_landed,
                sig_str_attempted, sig_str_absorbed, sig_str_faced, td_landed,
                td_attempted, td_absorbed, td_faced, last_fight_date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fighter_id) DO UPDATE SET
                wins=excluded.wins,
                losses=excluded.losses,
                draws=excluded.draws,
                no_contests=excluded.no_contests,
                total_fights=excluded.total_fights,
                win_streak=excluded.win_streak,
                ko_wins=excluded.ko_wins,
                sub_wins=excluded.sub_wins,
                dec_wins=excluded.dec_wins,
                ko_losses=excluded.ko_losses,
                sub_losses=excluded.sub_losses,
                dec_losses=excluded.dec_losses,
                total_rounds_fought=excluded.total_rounds_fought,
                total_fight_seconds=excluded.total_fight_seconds,
                total_knockdowns=excluded.total_knockdowns,
                total_sub_attempts=excluded.total_sub_attempts,
                total_control_seconds=excluded.total_control_seconds,
                sig_str_landed=excluded.sig_str_landed,
                sig_str_attempted=excluded.sig_str_attempted,
                sig_str_absorbed=excluded.sig_str_absorbed,
                sig_str_faced=excluded.sig_str_faced,
                td_landed=excluded.td_landed,
                td_attempted=excluded.td_attempted,
                td_absorbed=excluded.td_absorbed,
                td_faced=excluded.td_faced,
                last_fight_date=excluded.last_fight_date
            """,
            (
                fighter_id,
                state.wins,
                state.losses,
                state.draws,
                state.no_contests,
                state.total_fights,
                state.win_streak,
                state.ko_wins,
                state.sub_wins,
                state.dec_wins,
                state.ko_losses,
                state.sub_losses,
                state.dec_losses,
                state.total_rounds_fought,
                state.total_fight_seconds,
                state.total_knockdowns,
                state.total_sub_attempts,
                state.total_control_seconds,
                state.sig_str_landed,
                state.sig_str_attempted,
                state.sig_str_absorbed,
                state.sig_str_faced,
                state.td_landed,
                state.td_attempted,
                state.td_absorbed,
                state.td_faced,
                state.last_fight_date.isoformat() if state.last_fight_date else None,
            ),
        )

    def export_csv(self, csv_path: Path) -> int:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        query = (
            "SELECT "
            + ", ".join(CSV_COLUMNS)
            + " FROM fights ORDER BY event_date, bout_index, fight_id"
        )
        rows = self.conn.execute(query)
        count = 0
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            for row in rows:
                writer.writerow([row[column] for column in CSV_COLUMNS])
                count += 1
        return count


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def parse_human_date(value: str) -> Optional[dt.date]:
    value = clean_text(value)
    if not value:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def parse_iso_date(value: Optional[str]) -> Optional[dt.date]:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def parse_mmss_to_seconds(value: str) -> Optional[int]:
    value = clean_text(value)
    if not value or ":" not in value:
        return None
    parts = value.split(":")
    if len(parts) != 2:
        return None
    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
    except ValueError:
        return None
    if minutes < 0 or seconds < 0 or seconds >= 60:
        return None
    return minutes * 60 + seconds


def parse_control_to_seconds(value: str) -> Optional[int]:
    value = clean_text(value)
    if not value or value in {"--", "-"}:
        return None
    return parse_mmss_to_seconds(value)


def parse_landed_attempted(value: str) -> tuple[Optional[int], Optional[int]]:
    value = clean_text(value)
    if not value or value in {"--", "-"}:
        return (None, None)
    match = re.search(r"(\d+)\s*(?:of|/|-|\s)\s*(\d+)", value, flags=re.IGNORECASE)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    nums = re.findall(r"\d+", value)
    if len(nums) >= 2:
        return (int(nums[0]), int(nums[1]))
    return (None, None)


def parse_int_from_text(value: str) -> Optional[int]:
    value = clean_text(value)
    if not value or value in {"--", "-"}:
        return None
    match = re.search(r"-?\d+", value)
    if not match:
        return None
    return int(match.group(0))


def parse_height_to_cm(raw_value: str) -> Optional[float]:
    raw_value = clean_text(raw_value)
    match = re.search(r"(\d+)\s*'\s*(\d+)", raw_value)
    if not match:
        return None
    feet = int(match.group(1))
    inches = int(match.group(2))
    return round(feet * 30.48 + inches * 2.54, 2)


def parse_reach_to_cm(raw_value: str) -> Optional[float]:
    raw_value = clean_text(raw_value)
    match = re.search(r"(\d+(?:\.\d+)?)", raw_value)
    if not match:
        return None
    inches = float(match.group(1))
    return round(inches * 2.54, 2)


def parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    value = clean_text(str(value))
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return num / den


def rounded(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(value, digits)


def method_category(method_text: str) -> str:
    text = clean_text(method_text).lower()
    if not text:
        return "unknown"
    if "ko" in text or "tko" in text:
        return "ko_tko"
    if "sub" in text:
        return "submission"
    if "dec" in text:
        return "decision"
    if "dq" in text:
        return "dq"
    return "other"


def compute_fight_duration_seconds(round_ended: Optional[int], time_ended: str) -> Optional[int]:
    if round_ended is None or round_ended <= 0:
        return None
    elapsed_last_round = parse_mmss_to_seconds(time_ended)
    if elapsed_last_round is None:
        return None
    return (round_ended - 1) * 300 + elapsed_last_round


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def extract_id_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def split_location(location_text: str) -> tuple[str, str, str]:
    location_text = clean_text(location_text)
    if not location_text:
        return ("", "", "")
    parts = [part.strip() for part in location_text.split(",") if part.strip()]
    if len(parts) >= 3:
        city = ", ".join(parts[:-2])
        state = parts[-2]
        country = parts[-1]
        return (city, state, country)
    if len(parts) == 2:
        return (parts[0], "", parts[1])
    return (parts[0], "", "")


def normalize_status(raw_status: str) -> str:
    value = clean_text(raw_status).upper()
    if value in {"W", "WIN"}:
        return "W"
    if value in {"L", "LOSS"}:
        return "L"
    if value in {"D", "DRAW"}:
        return "D"
    if value in {"NC", "N/C", "NO CONTEST"}:
        return "NC"
    return ""


def infer_gender(weight_class: str) -> str:
    text = (weight_class or "").lower()
    if "women" in text:
        return "female"
    if text:
        return "male"
    return "unknown"


def parse_scheduled_rounds(time_format: str) -> Optional[int]:
    text = clean_text(time_format)
    if not text:
        return None
    match = re.search(r"(\d+)\s*rnd", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def parse_events_index(client: HttpClient) -> list[EventMeta]:
    soup = client.get_soup(EVENTS_URL)
    table = soup.select_one("table.b-statistics__table")
    rows = soup.select("tr.b-statistics__table-row") if table is None else table.select(
        "tr.b-statistics__table-row"
    )
    header_map: dict[str, int] = {}
    if table is not None:
        headers = [normalize_header(h.get_text(" ", strip=True)) for h in table.select("thead th")]
        header_map = {name: idx for idx, name in enumerate(headers)}
    events: list[EventMeta] = []
    for row in rows:
        link = row.select_one("a[href*='/event-details/']")
        if link is None:
            continue
        event_url = clean_text(link.get("href", ""))
        if not event_url:
            continue
        event_name = clean_text(link.get_text(" ", strip=True))
        row_text = clean_text(" ".join(row.stripped_strings))
        date_match = DATE_RE.search(row_text)
        if date_match is None:
            continue
        event_date = parse_human_date(date_match.group(0))
        if event_date is None:
            continue
        cells = row.find_all("td")
        location_text = ""
        location_idx = header_map.get("location")
        if location_idx is not None and location_idx < len(cells):
            location_text = clean_text(cells[location_idx].get_text(" ", strip=True))
        elif len(cells) >= 3:
            location_text = clean_text(cells[2].get_text(" ", strip=True))
        elif len(cells) >= 2:
            location_text = clean_text(cells[-1].get_text(" ", strip=True))
        city, state, country = split_location(location_text)
        events.append(
            EventMeta(
                event_id=extract_id_from_url(event_url),
                event_url=event_url,
                event_name=event_name,
                event_date=event_date,
                event_city=city,
                event_state=state,
                event_country=country,
            )
        )
    events.sort(key=lambda event: (event.event_date, event.event_id))
    return events


def td_values(td: Optional[Any]) -> list[str]:
    if td is None:
        return []
    paragraphs = td.find_all("p")
    if paragraphs:
        values = [clean_text(p.get_text(" ", strip=True)) for p in paragraphs]
        return [value for value in values if value]
    text = clean_text(td.get_text(" ", strip=True))
    return [text] if text else []


def td_pair_values(td: Optional[Any]) -> tuple[str, str]:
    values = td_values(td)
    first = values[0] if len(values) > 0 else ""
    second = values[1] if len(values) > 1 else ""
    return first, second


def parse_event_fights(client: HttpClient, event: EventMeta) -> list[FightStub]:
    soup = client.get_soup(event.event_url)
    table = soup.select_one("table.b-fight-details__table")
    if table is None:
        return []
    header_cells = table.select("thead th")
    headers = [normalize_header(cell.get_text(" ", strip=True)) for cell in header_cells]
    rows = table.select("tbody tr.b-fight-details__table-row")
    if not rows:
        rows = table.select("tr.b-fight-details__table-row")
    stubs: list[FightStub] = []
    for idx, row in enumerate(rows, start=1):
        fight_url = clean_text(row.get("data-link", ""))
        if not fight_url:
            link = row.select_one("a[href*='/fight-details/']")
            if link:
                fight_url = clean_text(link.get("href", ""))
        if not fight_url:
            continue
        cells = row.find_all("td", recursive=False)
        if not cells or not headers:
            continue
        mapped: dict[str, Any] = {}
        for h_idx, header in enumerate(headers):
            if h_idx < len(cells):
                mapped[header] = cells[h_idx]
        fighter_cell = mapped.get("fighter")
        status_cell = mapped.get("wl")
        weight_class_cell = mapped.get("weightclass")
        method_cell = mapped.get("method")
        round_cell = mapped.get("round")
        time_cell = mapped.get("time")
        kd_cell = mapped.get("kd")
        sig_str_cell = mapped.get("sigstr")
        td_cell = mapped.get("td")
        sub_cell = mapped.get("sub")
        ctrl_cell = mapped.get("ctrl")
        fighter_links = []
        if fighter_cell is not None:
            fighter_links = fighter_cell.select("a[href*='/fighter-details/']")
        if len(fighter_links) < 2:
            continue
        fighter_1_url = clean_text(fighter_links[0].get("href", ""))
        fighter_2_url = clean_text(fighter_links[1].get("href", ""))
        if not fighter_1_url or not fighter_2_url:
            continue
        fighter_1_name = clean_text(fighter_links[0].get_text(" ", strip=True))
        fighter_2_name = clean_text(fighter_links[1].get_text(" ", strip=True))
        status_values = td_values(status_cell)
        fighter_1_status = status_values[0] if len(status_values) > 0 else ""
        fighter_2_status = status_values[1] if len(status_values) > 1 else ""
        weight_class_values = td_values(weight_class_cell)
        method_values = td_values(method_cell)
        round_values = td_values(round_cell)
        time_values = td_values(time_cell)
        kd_1_raw, kd_2_raw = td_pair_values(kd_cell)
        sig_str_1_raw, sig_str_2_raw = td_pair_values(sig_str_cell)
        td_1_raw, td_2_raw = td_pair_values(td_cell)
        sub_1_raw, sub_2_raw = td_pair_values(sub_cell)
        ctrl_1_raw, ctrl_2_raw = td_pair_values(ctrl_cell)
        sig_str_1_landed, sig_str_1_attempted = parse_landed_attempted(sig_str_1_raw)
        sig_str_2_landed, sig_str_2_attempted = parse_landed_attempted(sig_str_2_raw)
        td_1_landed, td_1_attempted = parse_landed_attempted(td_1_raw)
        td_2_landed, td_2_attempted = parse_landed_attempted(td_2_raw)
        stubs.append(
            FightStub(
                fight_id=extract_id_from_url(fight_url),
                fight_url=fight_url,
                bout_index=idx,
                fighter_1_id=extract_id_from_url(fighter_1_url),
                fighter_1_name=fighter_1_name,
                fighter_1_url=fighter_1_url,
                fighter_2_id=extract_id_from_url(fighter_2_url),
                fighter_2_name=fighter_2_name,
                fighter_2_url=fighter_2_url,
                fighter_1_status=fighter_1_status,
                fighter_2_status=fighter_2_status,
                weight_class=weight_class_values[0] if weight_class_values else "",
                method=method_values[0] if method_values else "",
                round_ended=parse_optional_int(round_values[0] if round_values else None),
                time_ended=time_values[0] if time_values else "",
                kd_1=parse_int_from_text(kd_1_raw),
                kd_2=parse_int_from_text(kd_2_raw),
                sig_str_1_landed=sig_str_1_landed,
                sig_str_1_attempted=sig_str_1_attempted,
                sig_str_2_landed=sig_str_2_landed,
                sig_str_2_attempted=sig_str_2_attempted,
                td_1_landed=td_1_landed,
                td_1_attempted=td_1_attempted,
                td_2_landed=td_2_landed,
                td_2_attempted=td_2_attempted,
                sub_1=parse_int_from_text(sub_1_raw),
                sub_2=parse_int_from_text(sub_2_raw),
                ctrl_seconds_1=parse_control_to_seconds(ctrl_1_raw),
                ctrl_seconds_2=parse_control_to_seconds(ctrl_2_raw),
            )
        )
    return stubs


def parse_fight_details(client: HttpClient, fight_url: str, fallback_weight_class: str) -> FightDetails:
    soup = client.get_soup(fight_url)
    title_element = soup.select_one(".b-fight-details__fight-title")
    title = clean_text(title_element.get_text(" ", strip=True)) if title_element else ""
    info: dict[str, str] = {}
    for item in soup.select(".b-fight-details__text-item, .b-fight-details__text-item_first"):
        text = clean_text(item.get_text(" ", strip=True))
        if ":" not in text:
            continue
        label, value = text.split(":", 1)
        info[normalize_header(label)] = clean_text(value)
    weight_class = info.get("weightclass") or clean_text(fallback_weight_class)
    if not weight_class and title:
        weight_class = title.replace("Bout", "").strip()
    time_format = info.get("timeformat", "")
    scheduled_rounds = parse_scheduled_rounds(time_format)
    is_title_bout = 1 if "title" in title.lower() or "title" in weight_class.lower() else 0
    gender = infer_gender(weight_class)
    return FightDetails(
        weight_class=weight_class,
        gender=gender,
        is_title_bout=is_title_bout,
        time_format=time_format,
        scheduled_rounds=scheduled_rounds,
    )


def parse_fighter_profile(client: HttpClient, fighter_id: str, fighter_url: str) -> FighterProfile:
    soup = client.get_soup(fighter_url)
    name_element = soup.select_one(".b-content__title-highlight")
    full_name = clean_text(name_element.get_text(" ", strip=True)) if name_element else ""
    details: dict[str, str] = {}
    for item in soup.select("li.b-list__box-list-item"):
        text = clean_text(item.get_text(" ", strip=True))
        if ":" not in text:
            continue
        label, value = text.split(":", 1)
        details[normalize_header(label)] = clean_text(value)
    dob = parse_human_date(details.get("dob", "")) if details.get("dob") else None
    return FighterProfile(
        fighter_id=fighter_id,
        fighter_url=fighter_url,
        full_name=full_name,
        dob=dob,
        height_cm=parse_height_to_cm(details.get("height", "")),
        reach_cm=parse_reach_to_cm(details.get("reach", "")),
        stance=details.get("stance", ""),
    )


def get_or_fetch_profile(
    store: CheckpointStore,
    client: HttpClient,
    fighter_id: str,
    fighter_url: str,
    fallback_name: str,
    profile_cache: dict[str, FighterProfile],
) -> FighterProfile:
    cached_in_memory = profile_cache.get(fighter_id)
    if cached_in_memory is not None:
        if not cached_in_memory.full_name and fallback_name:
            cached_in_memory = dataclasses.replace(cached_in_memory, full_name=fallback_name)
            profile_cache[fighter_id] = cached_in_memory
        return cached_in_memory
    cached = store.get_fighter_profile(fighter_id)
    if cached is not None:
        if not cached.full_name and fallback_name:
            cached = dataclasses.replace(cached, full_name=fallback_name)
        profile_cache[fighter_id] = cached
        return cached
    profile = parse_fighter_profile(client, fighter_id, fighter_url)
    if not profile.full_name and fallback_name:
        profile = dataclasses.replace(profile, full_name=fallback_name)
    store.upsert_fighter_profile(profile)
    profile_cache[fighter_id] = profile
    return profile


def age_days_at_fight(dob: Optional[dt.date], fight_date: dt.date) -> Optional[int]:
    if dob is None:
        return None
    return (fight_date - dob).days


def days_since_last_fight(last_date: Optional[dt.date], fight_date: dt.date) -> Optional[int]:
    if last_date is None:
        return None
    return (fight_date - last_date).days


def winner_from_status(
    fighter_1_id: str,
    fighter_1_name: str,
    fighter_1_status: str,
    fighter_2_id: str,
    fighter_2_name: str,
    fighter_2_status: str,
) -> tuple[str, str, str]:
    status_1 = normalize_status(fighter_1_status)
    status_2 = normalize_status(fighter_2_status)
    if status_1 == "W":
        return fighter_1_id, fighter_1_name, "fighter_1_win"
    if status_2 == "W":
        return fighter_2_id, fighter_2_name, "fighter_2_win"
    if status_1 == "D" or status_2 == "D":
        return "", "", "draw"
    if status_1 == "NC" or status_2 == "NC":
        return "", "", "no_contest"
    return "", "", "unknown"


def prefight_metrics_from_state(state: FighterState) -> PreFightMetrics:
    fight_minutes = state.total_fight_seconds / 60 if state.total_fight_seconds > 0 else 0
    finish_wins = state.ko_wins + state.sub_wins
    return PreFightMetrics(
        win_rate=rounded(safe_div(state.wins, state.total_fights)),
        finish_rate=rounded(safe_div(finish_wins, state.wins)),
        avg_fight_duration_sec=rounded(
            safe_div(state.total_fight_seconds, state.total_fights)
        ),
        avg_rounds_fought=rounded(safe_div(state.total_rounds_fought, state.total_fights)),
        sig_str_landed_per_min=rounded(safe_div(state.sig_str_landed, fight_minutes)),
        sig_str_absorbed_per_min=rounded(safe_div(state.sig_str_absorbed, fight_minutes)),
        sig_str_accuracy=rounded(safe_div(state.sig_str_landed, state.sig_str_attempted)),
        sig_str_defense=rounded(
            safe_div(
                max(state.sig_str_faced - state.sig_str_absorbed, 0),
                state.sig_str_faced,
            )
        ),
        td_landed_per_15=rounded(safe_div(state.td_landed * 15, fight_minutes)),
        td_absorbed_per_15=rounded(safe_div(state.td_absorbed * 15, fight_minutes)),
        td_accuracy=rounded(safe_div(state.td_landed, state.td_attempted)),
        td_defense=rounded(
            safe_div(max(state.td_faced - state.td_absorbed, 0), state.td_faced)
        ),
        sub_attempts_per_15=rounded(safe_div(state.total_sub_attempts * 15, fight_minutes)),
        knockdowns_per_15=rounded(safe_div(state.total_knockdowns * 15, fight_minutes)),
        control_time_per_min=rounded(safe_div(state.total_control_seconds, fight_minutes)),
    )


def numeric_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(a - b, 4)


def apply_result_to_state(
    *,
    state: FighterState,
    status: str,
    fight_date: dt.date,
    result_method_category: str,
    round_ended: Optional[int],
    fight_duration_seconds: Optional[int],
    knockdowns_for: Optional[int],
    sub_attempts_for: Optional[int],
    control_seconds_for: Optional[int],
    sig_str_landed_for: Optional[int],
    sig_str_attempted_for: Optional[int],
    sig_str_landed_against: Optional[int],
    sig_str_attempted_against: Optional[int],
    td_landed_for: Optional[int],
    td_attempted_for: Optional[int],
    td_landed_against: Optional[int],
    td_attempted_against: Optional[int],
) -> FighterState:
    normalized = normalize_status(status)
    updated = dataclasses.replace(state)
    if normalized == "W":
        updated.wins += 1
        updated.win_streak += 1
        updated.total_fights += 1
        updated.last_fight_date = fight_date
        if result_method_category == "ko_tko":
            updated.ko_wins += 1
        elif result_method_category == "submission":
            updated.sub_wins += 1
        elif result_method_category == "decision":
            updated.dec_wins += 1
    elif normalized == "L":
        updated.losses += 1
        updated.win_streak = 0
        updated.total_fights += 1
        updated.last_fight_date = fight_date
        if result_method_category == "ko_tko":
            updated.ko_losses += 1
        elif result_method_category == "submission":
            updated.sub_losses += 1
        elif result_method_category == "decision":
            updated.dec_losses += 1
    elif normalized == "D":
        updated.draws += 1
        updated.win_streak = 0
        updated.total_fights += 1
        updated.last_fight_date = fight_date
    elif normalized == "NC":
        updated.no_contests += 1
        updated.total_fights += 1
        updated.last_fight_date = fight_date

    if round_ended is not None and round_ended > 0:
        updated.total_rounds_fought += round_ended
    if fight_duration_seconds is not None and fight_duration_seconds >= 0:
        updated.total_fight_seconds += fight_duration_seconds

    if knockdowns_for is not None:
        updated.total_knockdowns += max(knockdowns_for, 0)
    if sub_attempts_for is not None:
        updated.total_sub_attempts += max(sub_attempts_for, 0)
    if control_seconds_for is not None:
        updated.total_control_seconds += max(control_seconds_for, 0)

    if sig_str_landed_for is not None:
        updated.sig_str_landed += max(sig_str_landed_for, 0)
    if sig_str_attempted_for is not None:
        updated.sig_str_attempted += max(sig_str_attempted_for, 0)
    if sig_str_landed_against is not None:
        updated.sig_str_absorbed += max(sig_str_landed_against, 0)
    if sig_str_attempted_against is not None:
        updated.sig_str_faced += max(sig_str_attempted_against, 0)

    if td_landed_for is not None:
        updated.td_landed += max(td_landed_for, 0)
    if td_attempted_for is not None:
        updated.td_attempted += max(td_attempted_for, 0)
    if td_landed_against is not None:
        updated.td_absorbed += max(td_landed_against, 0)
    if td_attempted_against is not None:
        updated.td_faced += max(td_attempted_against, 0)

    return updated


def build_fight_row(
    *,
    event: EventMeta,
    stub: FightStub,
    details: FightDetails,
    profile_1: FighterProfile,
    profile_2: FighterProfile,
    state_1: FighterState,
    state_2: FighterState,
) -> dict[str, Any]:
    winner_id, winner_name, outcome_label = winner_from_status(
        stub.fighter_1_id,
        stub.fighter_1_name,
        stub.fighter_1_status,
        stub.fighter_2_id,
        stub.fighter_2_name,
        stub.fighter_2_status,
    )
    fight_duration_seconds = compute_fight_duration_seconds(stub.round_ended, stub.time_ended)
    method_cat = method_category(stub.method)
    f1_age_days = age_days_at_fight(profile_1.dob, event.event_date)
    f2_age_days = age_days_at_fight(profile_2.dob, event.event_date)
    f1_days_since = days_since_last_fight(state_1.last_fight_date, event.event_date)
    f2_days_since = days_since_last_fight(state_2.last_fight_date, event.event_date)
    m1 = prefight_metrics_from_state(state_1)
    m2 = prefight_metrics_from_state(state_2)
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
        "result_method_category": method_cat,
        "fighter_1_id": stub.fighter_1_id,
        "fighter_1_name": stub.fighter_1_name,
        "fighter_1_dob": profile_1.dob.isoformat() if profile_1.dob else "",
        "fighter_1_age_days": f1_age_days,
        "fighter_1_height_cm": profile_1.height_cm,
        "fighter_1_reach_cm": profile_1.reach_cm,
        "fighter_1_stance": profile_1.stance,
        "fighter_1_wins_pre": state_1.wins,
        "fighter_1_losses_pre": state_1.losses,
        "fighter_1_draws_pre": state_1.draws,
        "fighter_1_no_contests_pre": state_1.no_contests,
        "fighter_1_total_fights_pre": state_1.total_fights,
        "fighter_1_win_streak_pre": state_1.win_streak,
        "fighter_1_days_since_last_fight": f1_days_since,
        "fighter_1_win_rate_pre": m1.win_rate,
        "fighter_1_finish_rate_pre": m1.finish_rate,
        "fighter_1_ko_wins_pre": state_1.ko_wins,
        "fighter_1_sub_wins_pre": state_1.sub_wins,
        "fighter_1_dec_wins_pre": state_1.dec_wins,
        "fighter_1_ko_losses_pre": state_1.ko_losses,
        "fighter_1_sub_losses_pre": state_1.sub_losses,
        "fighter_1_dec_losses_pre": state_1.dec_losses,
        "fighter_1_avg_fight_duration_sec_pre": m1.avg_fight_duration_sec,
        "fighter_1_avg_rounds_fought_pre": m1.avg_rounds_fought,
        "fighter_1_sig_str_landed_per_min_pre": m1.sig_str_landed_per_min,
        "fighter_1_sig_str_absorbed_per_min_pre": m1.sig_str_absorbed_per_min,
        "fighter_1_sig_str_accuracy_pre": m1.sig_str_accuracy,
        "fighter_1_sig_str_defense_pre": m1.sig_str_defense,
        "fighter_1_td_landed_per_15_pre": m1.td_landed_per_15,
        "fighter_1_td_absorbed_per_15_pre": m1.td_absorbed_per_15,
        "fighter_1_td_accuracy_pre": m1.td_accuracy,
        "fighter_1_td_defense_pre": m1.td_defense,
        "fighter_1_sub_attempts_per_15_pre": m1.sub_attempts_per_15,
        "fighter_1_knockdowns_per_15_pre": m1.knockdowns_per_15,
        "fighter_1_control_time_per_min_pre": m1.control_time_per_min,
        "fighter_2_id": stub.fighter_2_id,
        "fighter_2_name": stub.fighter_2_name,
        "fighter_2_dob": profile_2.dob.isoformat() if profile_2.dob else "",
        "fighter_2_age_days": f2_age_days,
        "fighter_2_height_cm": profile_2.height_cm,
        "fighter_2_reach_cm": profile_2.reach_cm,
        "fighter_2_stance": profile_2.stance,
        "fighter_2_wins_pre": state_2.wins,
        "fighter_2_losses_pre": state_2.losses,
        "fighter_2_draws_pre": state_2.draws,
        "fighter_2_no_contests_pre": state_2.no_contests,
        "fighter_2_total_fights_pre": state_2.total_fights,
        "fighter_2_win_streak_pre": state_2.win_streak,
        "fighter_2_days_since_last_fight": f2_days_since,
        "fighter_2_win_rate_pre": m2.win_rate,
        "fighter_2_finish_rate_pre": m2.finish_rate,
        "fighter_2_ko_wins_pre": state_2.ko_wins,
        "fighter_2_sub_wins_pre": state_2.sub_wins,
        "fighter_2_dec_wins_pre": state_2.dec_wins,
        "fighter_2_ko_losses_pre": state_2.ko_losses,
        "fighter_2_sub_losses_pre": state_2.sub_losses,
        "fighter_2_dec_losses_pre": state_2.dec_losses,
        "fighter_2_avg_fight_duration_sec_pre": m2.avg_fight_duration_sec,
        "fighter_2_avg_rounds_fought_pre": m2.avg_rounds_fought,
        "fighter_2_sig_str_landed_per_min_pre": m2.sig_str_landed_per_min,
        "fighter_2_sig_str_absorbed_per_min_pre": m2.sig_str_absorbed_per_min,
        "fighter_2_sig_str_accuracy_pre": m2.sig_str_accuracy,
        "fighter_2_sig_str_defense_pre": m2.sig_str_defense,
        "fighter_2_td_landed_per_15_pre": m2.td_landed_per_15,
        "fighter_2_td_absorbed_per_15_pre": m2.td_absorbed_per_15,
        "fighter_2_td_accuracy_pre": m2.td_accuracy,
        "fighter_2_td_defense_pre": m2.td_defense,
        "fighter_2_sub_attempts_per_15_pre": m2.sub_attempts_per_15,
        "fighter_2_knockdowns_per_15_pre": m2.knockdowns_per_15,
        "fighter_2_control_time_per_min_pre": m2.control_time_per_min,
        "age_days_diff_f1_minus_f2": numeric_diff(
            float(f1_age_days) if f1_age_days is not None else None,
            float(f2_age_days) if f2_age_days is not None else None,
        ),
        "height_cm_diff_f1_minus_f2": numeric_diff(profile_1.height_cm, profile_2.height_cm),
        "reach_cm_diff_f1_minus_f2": numeric_diff(profile_1.reach_cm, profile_2.reach_cm),
        "total_fights_diff_f1_minus_f2": numeric_diff(
            float(state_1.total_fights), float(state_2.total_fights)
        ),
        "wins_diff_f1_minus_f2": numeric_diff(float(state_1.wins), float(state_2.wins)),
        "losses_diff_f1_minus_f2": numeric_diff(float(state_1.losses), float(state_2.losses)),
        "win_streak_diff_f1_minus_f2": numeric_diff(
            float(state_1.win_streak), float(state_2.win_streak)
        ),
        "win_rate_diff_f1_minus_f2": numeric_diff(m1.win_rate, m2.win_rate),
        "finish_rate_diff_f1_minus_f2": numeric_diff(m1.finish_rate, m2.finish_rate),
        "days_since_last_fight_diff_f1_minus_f2": numeric_diff(
            float(f1_days_since) if f1_days_since is not None else None,
            float(f2_days_since) if f2_days_since is not None else None,
        ),
        "avg_fight_duration_sec_diff_f1_minus_f2": numeric_diff(
            m1.avg_fight_duration_sec, m2.avg_fight_duration_sec
        ),
        "avg_rounds_fought_diff_f1_minus_f2": numeric_diff(
            m1.avg_rounds_fought, m2.avg_rounds_fought
        ),
        "sig_str_landed_per_min_diff_f1_minus_f2": numeric_diff(
            m1.sig_str_landed_per_min, m2.sig_str_landed_per_min
        ),
        "sig_str_absorbed_per_min_diff_f1_minus_f2": numeric_diff(
            m1.sig_str_absorbed_per_min, m2.sig_str_absorbed_per_min
        ),
        "sig_str_accuracy_diff_f1_minus_f2": numeric_diff(
            m1.sig_str_accuracy, m2.sig_str_accuracy
        ),
        "sig_str_defense_diff_f1_minus_f2": numeric_diff(
            m1.sig_str_defense, m2.sig_str_defense
        ),
        "td_landed_per_15_diff_f1_minus_f2": numeric_diff(
            m1.td_landed_per_15, m2.td_landed_per_15
        ),
        "td_absorbed_per_15_diff_f1_minus_f2": numeric_diff(
            m1.td_absorbed_per_15, m2.td_absorbed_per_15
        ),
        "td_accuracy_diff_f1_minus_f2": numeric_diff(m1.td_accuracy, m2.td_accuracy),
        "td_defense_diff_f1_minus_f2": numeric_diff(m1.td_defense, m2.td_defense),
        "sub_attempts_per_15_diff_f1_minus_f2": numeric_diff(
            m1.sub_attempts_per_15, m2.sub_attempts_per_15
        ),
        "knockdowns_per_15_diff_f1_minus_f2": numeric_diff(
            m1.knockdowns_per_15, m2.knockdowns_per_15
        ),
        "control_time_per_min_diff_f1_minus_f2": numeric_diff(
            m1.control_time_per_min, m2.control_time_per_min
        ),
        "winner_fighter_id": winner_id,
        "winner_name": winner_name,
        "outcome_label": outcome_label,
        "scrape_timestamp_utc": utc_now_iso(),
    }
    return row


def process_event(
    *,
    store: CheckpointStore,
    client: HttpClient,
    event: EventMeta,
    max_fights_remaining: Optional[int],
    profile_cache: dict[str, FighterProfile],
    state_cache: dict[str, FighterState],
    fight_details_cache: dict[str, FightDetails],
    fetch_fight_details: bool,
    commit_every: int,
    stop_on_fight_error: bool,
) -> tuple[int, bool, bool]:
    """Process one event.

    Returns:
      (inserted_fights, event_completed, stopped_by_chunk_limit)
    """
    stubs = parse_event_fights(client, event)
    if not stubs:
        logging.warning("No fights found for event %s (%s)", event.event_name, event.event_url)
        return 0, True, False

    existing_fight_ids = store.existing_fight_ids_for_event(event.event_id)
    inserted = 0
    pending_writes = 0
    uncommitted_fights = 0
    uncommitted_fight_ids: list[str] = []
    uncommitted_fighter_ids: set[str] = set()
    event_completed = True
    stopped_by_limit = False
    for stub in stubs:
        if max_fights_remaining is not None and max_fights_remaining <= 0:
            stopped_by_limit = True
            event_completed = False
            break
        if stub.fight_id in existing_fight_ids:
            continue
        try:
            details: FightDetails
            if fetch_fight_details:
                details = fight_details_cache.get(stub.fight_id) or parse_fight_details(
                    client, stub.fight_url, stub.weight_class
                )
                fight_details_cache[stub.fight_id] = details
            else:
                inferred_title = 1 if "title" in stub.weight_class.lower() else 0
                details = FightDetails(
                    weight_class=stub.weight_class,
                    gender=infer_gender(stub.weight_class),
                    is_title_bout=inferred_title,
                    time_format="",
                    scheduled_rounds=5 if inferred_title else None,
                )

            profile_1 = get_or_fetch_profile(
                store,
                client,
                stub.fighter_1_id,
                stub.fighter_1_url,
                stub.fighter_1_name,
                profile_cache,
            )
            profile_2 = get_or_fetch_profile(
                store,
                client,
                stub.fighter_2_id,
                stub.fighter_2_url,
                stub.fighter_2_name,
                profile_cache,
            )
            state_1 = state_cache.get(stub.fighter_1_id) or store.get_fighter_state(
                stub.fighter_1_id
            )
            state_2 = state_cache.get(stub.fighter_2_id) or store.get_fighter_state(
                stub.fighter_2_id
            )
            state_cache[stub.fighter_1_id] = state_1
            state_cache[stub.fighter_2_id] = state_2

            row = build_fight_row(
                event=event,
                stub=stub,
                details=details,
                profile_1=profile_1,
                profile_2=profile_2,
                state_1=state_1,
                state_2=state_2,
            )

            result_method_category = method_category(stub.method)
            duration_seconds = compute_fight_duration_seconds(stub.round_ended, stub.time_ended)
            updated_state_1 = apply_result_to_state(
                state=state_1,
                status=stub.fighter_1_status,
                fight_date=event.event_date,
                result_method_category=result_method_category,
                round_ended=stub.round_ended,
                fight_duration_seconds=duration_seconds,
                knockdowns_for=stub.kd_1,
                sub_attempts_for=stub.sub_1,
                control_seconds_for=stub.ctrl_seconds_1,
                sig_str_landed_for=stub.sig_str_1_landed,
                sig_str_attempted_for=stub.sig_str_1_attempted,
                sig_str_landed_against=stub.sig_str_2_landed,
                sig_str_attempted_against=stub.sig_str_2_attempted,
                td_landed_for=stub.td_1_landed,
                td_attempted_for=stub.td_1_attempted,
                td_landed_against=stub.td_2_landed,
                td_attempted_against=stub.td_2_attempted,
            )
            updated_state_2 = apply_result_to_state(
                state=state_2,
                status=stub.fighter_2_status,
                fight_date=event.event_date,
                result_method_category=result_method_category,
                round_ended=stub.round_ended,
                fight_duration_seconds=duration_seconds,
                knockdowns_for=stub.kd_2,
                sub_attempts_for=stub.sub_2,
                control_seconds_for=stub.ctrl_seconds_2,
                sig_str_landed_for=stub.sig_str_2_landed,
                sig_str_attempted_for=stub.sig_str_2_attempted,
                sig_str_landed_against=stub.sig_str_1_landed,
                sig_str_attempted_against=stub.sig_str_1_attempted,
                td_landed_for=stub.td_2_landed,
                td_attempted_for=stub.td_2_attempted,
                td_landed_against=stub.td_1_landed,
                td_attempted_against=stub.td_1_attempted,
            )

            store.insert_fight(row)
            store.upsert_fighter_state(stub.fighter_1_id, updated_state_1)
            store.upsert_fighter_state(stub.fighter_2_id, updated_state_2)
            state_cache[stub.fighter_1_id] = updated_state_1
            state_cache[stub.fighter_2_id] = updated_state_2

            inserted += 1
            pending_writes += 1
            uncommitted_fights += 1
            uncommitted_fight_ids.append(stub.fight_id)
            uncommitted_fighter_ids.add(stub.fighter_1_id)
            uncommitted_fighter_ids.add(stub.fighter_2_id)
            existing_fight_ids.add(stub.fight_id)
            if max_fights_remaining is not None:
                max_fights_remaining -= 1
            if pending_writes >= commit_every:
                store.commit()
                pending_writes = 0
                uncommitted_fights = 0
                uncommitted_fight_ids.clear()
                uncommitted_fighter_ids.clear()
        except Exception as exc:
            store.rollback()
            if uncommitted_fights > 0:
                inserted = max(inserted - uncommitted_fights, 0)
                if max_fights_remaining is not None:
                    max_fights_remaining += uncommitted_fights
            for fight_id in uncommitted_fight_ids:
                existing_fight_ids.discard(fight_id)
            for fighter_id in uncommitted_fighter_ids:
                state_cache.pop(fighter_id, None)
            pending_writes = 0
            uncommitted_fights = 0
            uncommitted_fight_ids.clear()
            uncommitted_fighter_ids.clear()
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
        uncommitted_fights = 0
        uncommitted_fight_ids.clear()
        uncommitted_fighter_ids.clear()
    return inserted, event_completed, stopped_by_limit


def parse_date_filter(value: Optional[str]) -> Optional[dt.date]:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Use YYYY-MM-DD."
        ) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resumable UFC fight history scraper with SQLite checkpoints and CSV export."
        )
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/ufc_fights_rnn.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--checkpoint-db",
        type=Path,
        default=Path("data/checkpoints/ufc_fights_checkpoint.sqlite"),
        help="SQLite checkpoint path.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Max unprocessed events to scrape this run (chunk size).",
    )
    parser.add_argument(
        "--max-fights",
        type=int,
        default=None,
        help="Max new fights to scrape this run.",
    )
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
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.05,
        help="Delay after every HTTP request to reduce load.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per request.",
    )
    parser.add_argument(
        "--backoff-seconds",
        type=float,
        default=1.0,
        help="Base exponential backoff delay on retry.",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=DEFAULT_USER_AGENT,
        help="HTTP User-Agent header value.",
    )
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
        help=(
            "Commit checkpoint writes every N new fights. Lower is safer for abrupt disconnects; "
            "higher is faster."
        ),
    )
    parser.add_argument(
        "--skip-fight-details",
        action="store_true",
        help=(
            "Skip per-fight detail-page requests for faster scraping. "
            "Some context fields (time_format, scheduled_rounds) may be missing."
        ),
    )
    parser.add_argument(
        "--stop-on-fight-error",
        action="store_true",
        help=(
            "Stop processing the current event when a single fight fails to parse. "
            "Default behavior is to continue with remaining fights."
        ),
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip scraping and export current checkpoint DB to CSV.",
    )
    return parser


def filter_events(
    events: Iterable[EventMeta],
    *,
    processed_event_ids: set[str],
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
        if event.event_id in processed_event_ids:
            continue
        filtered.append(event)
        if max_events is not None and len(filtered) >= max_events:
            break
    return filtered


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    store = CheckpointStore(args.checkpoint_db)
    try:
        if args.export_only:
            rows = store.export_csv(args.output_csv)
            logging.info("Exported %s rows to %s", rows, args.output_csv)
            return 0

        commit_every = max(1, int(args.commit_every))

        if MISSING_SCRAPER_DEPS:
            raise SystemExit(
                "Missing dependencies. Run: pip install -r requirements-scraper.txt"
            )

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

        processed_event_ids = store.processed_event_ids()

        pending_events = filter_events(
            all_events,
            processed_event_ids=processed_event_ids,
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
            "Processing %s event(s). Existing fights in checkpoint DB: %s",
            len(pending_events),
            store.fights_count(),
        )

        total_new_fights = 0
        processed_events = 0
        max_fights_remaining = args.max_fights
        profile_cache: dict[str, FighterProfile] = {}
        state_cache: dict[str, FighterState] = {}
        fight_details_cache: dict[str, FightDetails] = {}

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
                profile_cache=profile_cache,
                state_cache=state_cache,
                fight_details_cache=fight_details_cache,
                fetch_fight_details=not args.skip_fight_details,
                commit_every=commit_every,
                stop_on_fight_error=args.stop_on_fight_error,
            )
            total_new_fights += inserted
            if max_fights_remaining is not None:
                max_fights_remaining -= inserted

            if event_completed:
                store.mark_event_processed(event)
                store.commit()
                processed_events += 1
                logging.info("Event completed. New fights added: %s", inserted)
            else:
                logging.warning(
                    "Event not fully completed, checkpoint retained at fight-level."
                )

            if stopped_by_limit:
                logging.info("Stopped early because --max-fights limit was reached.")
                break

        rows = store.export_csv(args.output_csv)
        logging.info(
            "Run complete. New fights: %s | Events completed this run: %s | Total CSV rows: %s",
            total_new_fights,
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
