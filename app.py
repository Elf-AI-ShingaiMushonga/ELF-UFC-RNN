#!/usr/bin/env python3
"""Flask app for UFC fight outcome prediction + bet tracking."""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path
from threading import RLock
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, int(os.cpu_count() or 1))))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores*",
    category=UserWarning,
)

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from web_predictor import FightPredictor


APP_ROOT = Path(__file__).resolve().parent
BETS_PATH = APP_ROOT / "data" / "bets_tracker.csv"
SCRAPER_SCRIPT = APP_ROOT / "scrape_ufc_fights.py"
SCRAPER_TIMEOUT_SECONDS = int(os.getenv("SCRAPER_TIMEOUT_SECONDS", "7200"))

BET_COLUMNS = [
    "bet_id",
    "created_at_utc",
    "settled_at_utc",
    "status",
    "result",
    "event_date",
    "fighter_1",
    "fighter_2",
    "pick",
    "actual_winner",
    "weight_class",
    "gender",
    "scheduled_rounds",
    "is_title_bout",
    "model",
    "model_label",
    "model_test_accuracy",
    "p_fighter_1",
    "p_fighter_2",
    "model_prob_pick",
    "odds_american",
    "implied_prob_at_bet",
    "model_edge",
    "stake",
    "pnl",
    "return_amount",
    "potential_profit",
    "sportsbook",
    "notes",
]

predictor = FightPredictor(
    APP_ROOT / "data" / "ufc_fights_rnn.csv",
    default_model=os.getenv("UFC_DEFAULT_MODEL", "accuracy_weighted_ensemble"),
    power_profile=os.getenv("UFC_POWER_PROFILE", "max_power"),
)
PREDICTOR_LOCK = RLock()
app = Flask(__name__)

RECOMMENDED_MIN_EDGE = 0.02
KELLY_FRACTION = 0.25
KELLY_CAP = 0.05


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if np.isnan(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def american_to_implied_prob(odds_american: int) -> float:
    if odds_american == 0:
        raise ValueError("American odds cannot be 0.")
    if odds_american > 0:
        return 100.0 / (odds_american + 100.0)
    abs_odds = abs(float(odds_american))
    return abs_odds / (abs_odds + 100.0)


def profit_from_american(stake: float, odds_american: int) -> float:
    if odds_american > 0:
        return stake * (odds_american / 100.0)
    return stake * (100.0 / abs(float(odds_american)))


def american_to_decimal(odds_american: int) -> float:
    if odds_american == 0:
        raise ValueError("American odds cannot be 0.")
    if odds_american > 0:
        return 1.0 + (odds_american / 100.0)
    return 1.0 + (100.0 / abs(float(odds_american)))


def parse_optional_american_odds(raw: str | None) -> int | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        odds = int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid American odds value: {text!r}") from exc
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    return odds


def parse_optional_positive_float(raw: str | None) -> float | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value: {text!r}") from exc
    if np.isnan(value) or value <= 0:
        raise ValueError("Value must be greater than 0.")
    return value


def build_bet_recommendation(
    prediction: dict[str, Any],
    odds_fighter_1: int | None,
    odds_fighter_2: int | None,
    bankroll: float | None,
) -> dict[str, Any]:
    base = {
        "available": False,
        "should_bet": False,
        "message": "Enter both fighter American odds to get a recommendation.",
        "recommended_pick": None,
        "recommended_odds": None,
        "recommended_edge": None,
        "recommended_expected_roi": None,
        "suggested_stake": None,
        "bankroll": bankroll,
        "sides": [],
    }
    if odds_fighter_1 is None or odds_fighter_2 is None:
        return base

    side_defs = [
        (prediction["fighter_1"], float(prediction["p_fighter_1"]), int(odds_fighter_1)),
        (prediction["fighter_2"], float(prediction["p_fighter_2"]), int(odds_fighter_2)),
    ]
    sides = []
    for fighter, prob, odds in side_defs:
        implied = float(american_to_implied_prob(odds))
        decimal_odds = float(american_to_decimal(odds))
        edge = float(prob - implied)
        expected_roi = float(prob * decimal_odds - 1.0)
        b = max(decimal_odds - 1.0, 1e-9)
        full_kelly = max(float((b * prob - (1.0 - prob)) / b), 0.0)
        suggested_stake = None
        if bankroll and bankroll > 0:
            suggested_fraction = min(full_kelly * KELLY_FRACTION, KELLY_CAP)
            suggested_stake = float(bankroll * suggested_fraction)
        sides.append(
            {
                "fighter": fighter,
                "prob": prob,
                "odds_american": odds,
                "implied_prob": implied,
                "edge": edge,
                "expected_roi": expected_roi,
                "full_kelly_fraction": full_kelly,
                "suggested_stake": suggested_stake,
            }
        )

    best = sorted(sides, key=lambda r: (r["expected_roi"], r["edge"]), reverse=True)[0]
    should_bet = best["expected_roi"] > 0 and best["edge"] >= RECOMMENDED_MIN_EDGE
    if should_bet:
        message = (
            f"Recommended: {best['fighter']} at {best['odds_american']:+d} "
            f"(edge {best['edge'] * 100:.2f}%, expected ROI {best['expected_roi'] * 100:.2f}%)."
        )
    else:
        message = (
            "No clear value bet at current odds (requires positive EV and "
            f"at least {RECOMMENDED_MIN_EDGE * 100:.0f}% edge)."
        )

    return {
        "available": True,
        "should_bet": bool(should_bet),
        "message": message,
        "recommended_pick": best["fighter"] if should_bet else None,
        "recommended_odds": int(best["odds_american"]) if should_bet else None,
        "recommended_edge": float(best["edge"]) if should_bet else None,
        "recommended_expected_roi": float(best["expected_roi"]) if should_bet else None,
        "suggested_stake": best["suggested_stake"] if should_bet else None,
        "bankroll": bankroll,
        "sides": sides,
    }


def run_scraper_update() -> str:
    if not SCRAPER_SCRIPT.exists():
        raise FileNotFoundError(f"Scraper script not found: {SCRAPER_SCRIPT}")
    cmd = [
        sys.executable,
        str(SCRAPER_SCRIPT),
        "--output-csv",
        str(APP_ROOT / "data" / "ufc_fights_rnn.csv"),
        "--checkpoint-db",
        str(APP_ROOT / "data" / "checkpoints" / "ufc_fights_checkpoint.sqlite"),
        "--log-level",
        "INFO",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(APP_ROOT),
        capture_output=True,
        text=True,
        timeout=SCRAPER_TIMEOUT_SECONDS,
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        tail = "\n".join((stdout + "\n" + stderr).strip().splitlines()[-12:])
        raise RuntimeError(f"Scraper failed (exit {result.returncode}).\n{tail}")
    lines = stdout.splitlines()
    if not lines:
        return "Scrape completed."
    return lines[-1]


def _ensure_bet_store() -> None:
    BETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not BETS_PATH.exists():
        pd.DataFrame(columns=BET_COLUMNS).to_csv(BETS_PATH, index=False)


def load_bets() -> pd.DataFrame:
    _ensure_bet_store()
    df = pd.read_csv(BETS_PATH)
    for col in BET_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[BET_COLUMNS].copy()

    numeric_cols = [
        "bet_id",
        "model_test_accuracy",
        "p_fighter_1",
        "p_fighter_2",
        "model_prob_pick",
        "odds_american",
        "implied_prob_at_bet",
        "model_edge",
        "stake",
        "pnl",
        "return_amount",
        "potential_profit",
        "scheduled_rounds",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    text_cols = [c for c in BET_COLUMNS if c not in numeric_cols]
    for col in text_cols:
        df[col] = df[col].astype("object")
    return df


def save_bets(df: pd.DataFrame) -> None:
    out = df.copy()
    for col in BET_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[BET_COLUMNS]
    out.to_csv(BETS_PATH, index=False)


def records_for_template(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out = df.copy()
    out = out.where(pd.notna(out), None)
    return out.to_dict(orient="records")


def next_bet_id(df: pd.DataFrame) -> int:
    if df.empty or df["bet_id"].dropna().empty:
        return 1
    return int(df["bet_id"].max()) + 1


def compute_tracker_summary(open_bets: pd.DataFrame, settled_bets: pd.DataFrame) -> dict[str, Any]:
    settled_stake = float(settled_bets["stake"].fillna(0).sum()) if not settled_bets.empty else 0.0
    settled_pnl = float(settled_bets["pnl"].fillna(0).sum()) if not settled_bets.empty else 0.0
    open_stake = float(open_bets["stake"].fillna(0).sum()) if not open_bets.empty else 0.0
    roi = (settled_pnl / settled_stake) if settled_stake > 0 else 0.0

    wins = int((settled_bets["result"] == "win").sum()) if not settled_bets.empty else 0
    losses = int((settled_bets["result"] == "loss").sum()) if not settled_bets.empty else 0
    pushes = int((settled_bets["result"] == "push").sum()) if not settled_bets.empty else 0
    graded = wins + losses
    hit_rate = (wins / graded) if graded > 0 else 0.0

    return {
        "open_bets": int(len(open_bets)),
        "settled_bets": int(len(settled_bets)),
        "open_stake": open_stake,
        "settled_stake": settled_stake,
        "settled_pnl": settled_pnl,
        "roi": roi,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": hit_rate,
    }


def model_performance_table(settled_bets: pd.DataFrame) -> list[dict[str, Any]]:
    if settled_bets.empty:
        return []

    rows: list[dict[str, Any]] = []
    for (model, model_label), g in settled_bets.groupby(["model", "model_label"], dropna=False):
        g = g.copy()
        wins = int((g["result"] == "win").sum())
        losses = int((g["result"] == "loss").sum())
        pushes = int((g["result"] == "push").sum())
        stake = float(g["stake"].fillna(0).sum())
        pnl = float(g["pnl"].fillna(0).sum())
        roi = (pnl / stake) if stake > 0 else 0.0

        graded = g[g["result"].isin(["win", "loss"])].copy()
        hit_rate = float((graded["result"] == "win").mean()) if not graded.empty else float("nan")

        probs = np.clip(graded["model_prob_pick"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        actual = (graded["result"] == "win").astype(int).to_numpy(dtype=int)
        if len(actual) > 0:
            log_loss_val = float(-np.mean(actual * np.log(probs) + (1 - actual) * np.log(1 - probs)))
            brier = float(np.mean((probs - actual) ** 2))
        else:
            log_loss_val = float("nan")
            brier = float("nan")

        rows.append(
            {
                "model": str(model),
                "model_label": str(model_label),
                "bets": int(len(g)),
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "hit_rate": hit_rate,
                "stake": stake,
                "pnl": pnl,
                "roi": roi,
                "avg_edge": float(g["model_edge"].fillna(0).mean()),
                "avg_confidence": float(np.abs(g["p_fighter_1"] - g["p_fighter_2"]).fillna(0).mean()),
                "log_loss": log_loss_val,
                "brier": brier,
            }
        )
    rows.sort(key=lambda r: (r["roi"], r["pnl"]), reverse=True)
    return rows


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    prediction: dict[str, Any] | None = None
    predictions: list[dict[str, Any]] | None = None
    error: str | None = None
    notice: str | None = None
    system_status = predictor.model_cache_status()

    form_values = {
        "fighter_1": "",
        "fighter_2": "",
        "event_date": str(pd.Timestamp.today().date()),
        "weight_class": predictor.context_defaults["weight_class"],
        "gender": predictor.context_defaults["gender"],
        "scheduled_rounds": predictor.context_defaults["scheduled_rounds"],
        "odds_fighter_1": "",
        "odds_fighter_2": "",
        "bankroll": "",
        "is_title_bout": False,
        "model": predictor.default_model,
        "predict_all": False,
    }

    if request.method == "POST":
        action = request.form.get("action", "predict").strip().lower()
        bets_df = load_bets()

        if action == "predict":
            form_values["fighter_1"] = request.form.get("fighter_1", "").strip()
            form_values["fighter_2"] = request.form.get("fighter_2", "").strip()
            form_values["event_date"] = request.form.get("event_date", form_values["event_date"]).strip()
            form_values["weight_class"] = request.form.get("weight_class", form_values["weight_class"]).strip()
            form_values["gender"] = request.form.get("gender", form_values["gender"]).strip()
            form_values["scheduled_rounds"] = request.form.get(
                "scheduled_rounds", str(form_values["scheduled_rounds"])
            ).strip()
            form_values["odds_fighter_1"] = request.form.get("odds_fighter_1", "").strip()
            form_values["odds_fighter_2"] = request.form.get("odds_fighter_2", "").strip()
            form_values["bankroll"] = request.form.get("bankroll", "").strip()
            form_values["is_title_bout"] = request.form.get("is_title_bout") == "on"
            form_values["model"] = request.form.get("model", predictor.default_model).strip()
            form_values["predict_all"] = request.form.get("predict_all") == "on"

            try:
                rounds = int(form_values["scheduled_rounds"])
                odds_f1 = parse_optional_american_odds(form_values["odds_fighter_1"])
                odds_f2 = parse_optional_american_odds(form_values["odds_fighter_2"])
                bankroll = parse_optional_positive_float(form_values["bankroll"])
                if form_values["predict_all"]:
                    with PREDICTOR_LOCK:
                        predictions = predictor.predict_matchup_all(
                            fighter_1_name=form_values["fighter_1"],
                            fighter_2_name=form_values["fighter_2"],
                            event_date=form_values["event_date"] or None,
                            weight_class=form_values["weight_class"] or None,
                            gender=form_values["gender"] or None,
                            scheduled_rounds=rounds,
                            is_title_bout=bool(form_values["is_title_bout"]),
                        )
                    for pred in predictions:
                        pred["recommendation"] = build_bet_recommendation(pred, odds_f1, odds_f2, bankroll)
                else:
                    with PREDICTOR_LOCK:
                        prediction = predictor.predict_matchup(
                            fighter_1_name=form_values["fighter_1"],
                            fighter_2_name=form_values["fighter_2"],
                            event_date=form_values["event_date"] or None,
                            weight_class=form_values["weight_class"] or None,
                            gender=form_values["gender"] or None,
                            scheduled_rounds=rounds,
                            is_title_bout=bool(form_values["is_title_bout"]),
                            model_name=form_values["model"],
                        )
                    prediction["recommendation"] = build_bet_recommendation(
                        prediction,
                        odds_f1,
                        odds_f2,
                        bankroll,
                    )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

        elif action == "update_data":
            try:
                scraper_summary = run_scraper_update()
                with PREDICTOR_LOCK:
                    system_status = predictor.reload_data()
                notice = (
                    "Data update complete. Models were not retrained. "
                    f"{scraper_summary}"
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

        elif action == "retrain_models":
            try:
                with PREDICTOR_LOCK:
                    system_status = predictor.retrain_models()
                notice = (
                    "Model retrain complete. "
                    f"Rows used: {system_status.get('data_rows', 'n/a')}."
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

        elif action == "add_bet":
            try:
                fighter_1 = request.form.get("fighter_1", "").strip()
                fighter_2 = request.form.get("fighter_2", "").strip()
                event_date = request.form.get("event_date", "").strip() or None
                weight_class = request.form.get("weight_class", "").strip() or None
                gender = request.form.get("gender", "").strip() or None
                scheduled_rounds = int(request.form.get("scheduled_rounds", "3"))
                is_title_bout = request.form.get("is_title_bout", "false").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                model_name = request.form.get("model", predictor.default_model).strip().lower()
                pick = request.form.get("pick", "").strip()
                odds_american = int(request.form.get("odds_american", "0").strip())
                stake = float(request.form.get("stake", "0").strip())
                sportsbook = request.form.get("sportsbook", "").strip()
                notes = request.form.get("notes", "").strip()

                if model_name == "all":
                    raise ValueError("Choose one model for a logged bet.")
                if stake <= 0:
                    raise ValueError("Stake must be greater than 0.")
                if odds_american == 0:
                    raise ValueError("American odds cannot be 0.")

                with PREDICTOR_LOCK:
                    pred = predictor.predict_matchup(
                        fighter_1_name=fighter_1,
                        fighter_2_name=fighter_2,
                        event_date=event_date,
                        weight_class=weight_class,
                        gender=gender,
                        scheduled_rounds=scheduled_rounds,
                        is_title_bout=is_title_bout,
                        model_name=model_name,
                    )
                if pick not in {pred["fighter_1"], pred["fighter_2"]}:
                    raise ValueError("Pick must match one of the two fighters.")

                p_pick = float(pred["p_fighter_1"] if pick == pred["fighter_1"] else pred["p_fighter_2"])
                implied = float(american_to_implied_prob(odds_american))
                edge = float(p_pick - implied)
                potential_profit = float(profit_from_american(stake, odds_american))

                new_row = {
                    "bet_id": next_bet_id(bets_df),
                    "created_at_utc": pd.Timestamp.utcnow().isoformat(),
                    "settled_at_utc": "",
                    "status": "open",
                    "result": "",
                    "event_date": pred["event_date"],
                    "fighter_1": pred["fighter_1"],
                    "fighter_2": pred["fighter_2"],
                    "pick": pick,
                    "actual_winner": "",
                    "weight_class": pred["weight_class"],
                    "gender": pred["gender"],
                    "scheduled_rounds": pred["scheduled_rounds"],
                    "is_title_bout": bool(pred["is_title_bout"]),
                    "model": pred["model"],
                    "model_label": pred["model_label"],
                    "model_test_accuracy": pred["model_test_accuracy"],
                    "p_fighter_1": pred["p_fighter_1"],
                    "p_fighter_2": pred["p_fighter_2"],
                    "model_prob_pick": p_pick,
                    "odds_american": odds_american,
                    "implied_prob_at_bet": implied,
                    "model_edge": edge,
                    "stake": stake,
                    "pnl": np.nan,
                    "return_amount": np.nan,
                    "potential_profit": potential_profit,
                    "sportsbook": sportsbook,
                    "notes": notes,
                }
                bets_df = pd.concat([bets_df, pd.DataFrame([new_row])], ignore_index=True)
                save_bets(bets_df)
                notice = (
                    f"Bet logged: {pick} at {odds_american:+d}, "
                    f"stake ${stake:.2f}, model edge {edge * 100:.2f}%."
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

        elif action == "settle_bet":
            try:
                bet_id = int(request.form.get("bet_id", "0").strip())
                actual_winner = request.form.get("actual_winner", "").strip()
                mask = bets_df["bet_id"] == bet_id
                if not mask.any():
                    raise ValueError(f"Bet id {bet_id} not found.")
                idx = bets_df.index[mask][0]
                if str(bets_df.at[idx, "status"]) == "settled":
                    raise ValueError(f"Bet id {bet_id} is already settled.")

                pick = str(bets_df.at[idx, "pick"])
                fighter_1 = str(bets_df.at[idx, "fighter_1"])
                fighter_2 = str(bets_df.at[idx, "fighter_2"])
                stake = _safe_float(bets_df.at[idx, "stake"], 0.0)
                odds = int(_safe_float(bets_df.at[idx, "odds_american"], 0.0))

                if actual_winner.lower() == "push":
                    result = "push"
                    pnl = 0.0
                    ret = stake
                elif actual_winner in {fighter_1, fighter_2}:
                    if pick == actual_winner:
                        result = "win"
                        pnl = float(profit_from_american(stake, odds))
                        ret = stake + pnl
                    else:
                        result = "loss"
                        pnl = -stake
                        ret = 0.0
                else:
                    raise ValueError("Settle winner must be Fighter 1, Fighter 2, or Push.")

                bets_df.at[idx, "status"] = "settled"
                bets_df.at[idx, "result"] = result
                bets_df.at[idx, "actual_winner"] = actual_winner
                bets_df.at[idx, "settled_at_utc"] = pd.Timestamp.utcnow().isoformat()
                bets_df.at[idx, "pnl"] = pnl
                bets_df.at[idx, "return_amount"] = ret
                save_bets(bets_df)
                notice = f"Bet {bet_id} settled as {result.upper()} (PnL: ${pnl:.2f})."
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

    bets_df = load_bets()
    open_bets = bets_df[bets_df["status"] == "open"].copy()
    settled_bets = bets_df[bets_df["status"] == "settled"].copy()

    open_bets = open_bets.sort_values(["created_at_utc", "bet_id"], ascending=[False, False])
    settled_bets = settled_bets.sort_values(["settled_at_utc", "bet_id"], ascending=[False, False])

    summary = compute_tracker_summary(open_bets, settled_bets)
    model_perf = model_performance_table(settled_bets)
    with PREDICTOR_LOCK:
        system_status = predictor.model_cache_status()

    return render_template(
        "index.html",
        fighter_names=predictor.fighter_names,
        weight_classes=predictor.weight_classes,
        genders=predictor.genders,
        model_options=predictor.model_options,
        form_values=form_values,
        prediction=prediction,
        predictions=predictions,
        notice=notice,
        error=error,
        system_status=system_status,
        summary=summary,
        model_perf=model_perf,
        open_bets=records_for_template(open_bets),
        settled_bets=records_for_template(settled_bets.head(120)),
    )


@app.route("/api/predict", methods=["POST"])
def predict_api() -> Any:
    payload = request.get_json(silent=True) or {}
    try:
        model_name = str(payload.get("model", predictor.default_model)).strip().lower()
        predict_all_raw = payload.get("predict_all", False)
        predict_all = bool(predict_all_raw)
        if isinstance(predict_all_raw, str):
            predict_all = predict_all_raw.strip().lower() in {"1", "true", "yes", "on"}
        if model_name == "all":
            predict_all = True
        odds_f1 = parse_optional_american_odds(str(payload.get("odds_fighter_1", "")).strip())
        odds_f2 = parse_optional_american_odds(str(payload.get("odds_fighter_2", "")).strip())
        bankroll = parse_optional_positive_float(str(payload.get("bankroll", "")).strip())

        common_kwargs = {
            "fighter_1_name": str(payload.get("fighter_1", "")),
            "fighter_2_name": str(payload.get("fighter_2", "")),
            "event_date": str(payload.get("event_date", "")).strip() or None,
            "weight_class": str(payload.get("weight_class", "")).strip() or None,
            "gender": str(payload.get("gender", "")).strip() or None,
            "scheduled_rounds": int(payload.get("scheduled_rounds", predictor.context_defaults["scheduled_rounds"])),
            "is_title_bout": bool(payload.get("is_title_bout", False)),
        }

        if predict_all:
            with PREDICTOR_LOCK:
                predictions = predictor.predict_matchup_all(**common_kwargs)
            for pred in predictions:
                pred["recommendation"] = build_bet_recommendation(pred, odds_f1, odds_f2, bankroll)
            return jsonify({"ok": True, "predictions": predictions})

        with PREDICTOR_LOCK:
            prediction = predictor.predict_matchup(model_name=model_name, **common_kwargs)
        prediction["recommendation"] = build_bet_recommendation(prediction, odds_f1, odds_f2, bankroll)
        return jsonify({"ok": True, "prediction": prediction})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/healthz", methods=["GET"])
def healthz() -> Any:
    return jsonify({"ok": True})


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
