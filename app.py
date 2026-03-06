#!/usr/bin/env python3
"""Flask app for training LSTM-from-sequences from a web UI."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock, Thread
from typing import Any

from flask import Flask, jsonify, render_template, request

APP_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = APP_ROOT / "scripts" / "train_lstm_from_sequences.py"
SCRAPER_SCRIPT = APP_ROOT / "scripts" / "scrape_ufc_fight_details.py"
SEQUENCE_SCRIPT = APP_ROOT / "scripts" / "build_fight_history_sequences.py"
AUDIT_SCRIPT = APP_ROOT / "scripts" / "audit_lstm_pipeline_data.py"
RAW_FIGHTS_CSV = APP_ROOT / "data" / "ufc_fight_details_lstm.csv"
SEQUENCE_CSV = APP_ROOT / "data" / "ufc_lstm_sequences.csv"
SCRAPER_CHECKPOINT_DB = APP_ROOT / "data" / "checkpoints" / "ufc_fight_details_checkpoint.sqlite"

DEFAULT_PARAMS: dict[str, Any] = {
    "input_csv": "data/ufc_lstm_sequences.csv",
    "model_path": "champion_lstm_model.pth",
    "scaler_path": "data/model_cache/lstm_sequence_scalers.pkl",
    "metrics_path": "data/model_cache/lstm_sequence_metrics.json",
    "epochs": 120,
    "patience": 20,
    "batch_size": 256,
    "hidden_size": 96,
    "num_layers": 1,
    "dropout": 0.35,
    "lr": 0.0005,
    "weight_decay": 0.0001,
    "grad_clip": 1.0,
    "bidirectional": True,
    "use_cross_attention": True,
    "attention_heads": 4,
    "attention_dropout": 0.10,
    "static_recency_mode": "ema",
    "ema_alpha": 0.65,
    "val_fraction": 0.15,
    "test_fraction": 0.15,
    "max_fights": "",
    "seed": 42,
    "device": "auto",
    "num_workers": 0,
    "log_level": "INFO",
}


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def parse_optional_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    return int(text)


def resolve_output_path(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (APP_ROOT / candidate).resolve()


def parse_train_params(payload: dict[str, Any]) -> dict[str, Any]:
    params = dict(DEFAULT_PARAMS)
    params.update({k: v for k, v in payload.items() if v is not None})

    parsed: dict[str, Any] = {
        "input_csv": str(params["input_csv"]).strip(),
        "model_path": str(params["model_path"]).strip(),
        "scaler_path": str(params["scaler_path"]).strip(),
        "metrics_path": str(params["metrics_path"]).strip(),
        "epochs": int(params["epochs"]),
        "patience": int(params["patience"]),
        "batch_size": int(params["batch_size"]),
        "hidden_size": int(params["hidden_size"]),
        "num_layers": int(params["num_layers"]),
        "dropout": float(params["dropout"]),
        "lr": float(params["lr"]),
        "weight_decay": float(params["weight_decay"]),
        "grad_clip": float(params["grad_clip"]),
        "bidirectional": parse_bool(params.get("bidirectional"), default=True),
        "use_cross_attention": parse_bool(params.get("use_cross_attention"), default=True),
        "attention_heads": int(params["attention_heads"]),
        "attention_dropout": float(params["attention_dropout"]),
        "static_recency_mode": str(params["static_recency_mode"]).strip().lower(),
        "ema_alpha": float(params["ema_alpha"]),
        "val_fraction": float(params["val_fraction"]),
        "test_fraction": float(params["test_fraction"]),
        "max_fights": parse_optional_int(params.get("max_fights")),
        "seed": int(params["seed"]),
        "device": str(params["device"]).strip().lower(),
        "num_workers": int(params["num_workers"]),
        "log_level": str(params["log_level"]).strip().upper(),
    }

    if not parsed["input_csv"]:
        raise ValueError("Input CSV path is required.")
    if parsed["epochs"] <= 0:
        raise ValueError("Epochs must be > 0.")
    if parsed["patience"] <= 0:
        raise ValueError("Patience must be > 0.")
    if parsed["batch_size"] <= 0:
        raise ValueError("Batch size must be > 0.")
    if parsed["hidden_size"] <= 0:
        raise ValueError("Hidden size must be > 0.")
    if parsed["num_layers"] <= 0:
        raise ValueError("Num layers must be > 0.")
    if not (0.0 <= parsed["dropout"] < 1.0):
        raise ValueError("Dropout must be in [0, 1).")
    if parsed["lr"] <= 0:
        raise ValueError("Learning rate must be > 0.")
    if parsed["weight_decay"] < 0:
        raise ValueError("Weight decay must be >= 0.")
    if parsed["grad_clip"] <= 0:
        raise ValueError("Grad clip must be > 0.")
    if parsed["attention_heads"] <= 0:
        raise ValueError("Attention heads must be > 0.")
    if not (0.0 <= parsed["attention_dropout"] < 1.0):
        raise ValueError("Attention dropout must be in [0, 1).")
    if parsed["static_recency_mode"] not in {"ema", "mean"}:
        raise ValueError("Static recency mode must be 'ema' or 'mean'.")
    if not (0.0 <= parsed["ema_alpha"] <= 1.0):
        raise ValueError("EMA alpha must be in [0, 1].")
    if not (0.01 <= parsed["val_fraction"] < 0.49):
        raise ValueError("Validation fraction must be between 0.01 and 0.49.")
    if not (0.01 <= parsed["test_fraction"] < 0.49):
        raise ValueError("Test fraction must be between 0.01 and 0.49.")
    if parsed["val_fraction"] + parsed["test_fraction"] >= 0.8:
        raise ValueError("Validation + test fractions are too large.")
    if parsed["max_fights"] is not None and parsed["max_fights"] <= 0:
        raise ValueError("Max fights must be empty or > 0.")
    if parsed["num_workers"] < 0:
        raise ValueError("Num workers must be >= 0.")
    if parsed["device"] not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("Device must be one of auto/cpu/cuda/mps.")
    if parsed["log_level"] not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("Log level must be DEBUG/INFO/WARNING/ERROR.")

    return parsed


def build_train_command(params: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, "-u", str(TRAIN_SCRIPT)]
    cmd += ["--input-csv", params["input_csv"]]
    cmd += ["--model-path", params["model_path"]]
    cmd += ["--scaler-path", params["scaler_path"]]
    cmd += ["--metrics-path", params["metrics_path"]]
    cmd += ["--epochs", str(params["epochs"])]
    cmd += ["--patience", str(params["patience"])]
    cmd += ["--batch-size", str(params["batch_size"])]
    cmd += ["--hidden-size", str(params["hidden_size"])]
    cmd += ["--num-layers", str(params["num_layers"])]
    cmd += ["--dropout", str(params["dropout"])]
    cmd += ["--lr", str(params["lr"])]
    cmd += ["--weight-decay", str(params["weight_decay"])]
    cmd += ["--grad-clip", str(params["grad_clip"])]
    cmd += ["--attention-heads", str(params["attention_heads"])]
    cmd += ["--attention-dropout", str(params["attention_dropout"])]
    cmd += ["--static-recency-mode", params["static_recency_mode"]]
    cmd += ["--ema-alpha", str(params["ema_alpha"])]
    cmd += ["--val-fraction", str(params["val_fraction"])]
    cmd += ["--test-fraction", str(params["test_fraction"])]
    if params["max_fights"] is not None:
        cmd += ["--max-fights", str(params["max_fights"])]
    cmd += ["--seed", str(params["seed"])]
    cmd += ["--device", params["device"]]
    cmd += ["--num-workers", str(params["num_workers"])]
    cmd += ["--log-level", params["log_level"]]

    cmd.append("--bidirectional" if params["bidirectional"] else "--no-bidirectional")
    cmd.append("--use-cross-attention" if params["use_cross_attention"] else "--no-cross-attention")
    return cmd


@dataclass
class TrainingRun:
    run_id: int
    started_at_utc: str
    params: dict[str, Any]
    command: list[str]
    metrics_path_abs: Path
    process: subprocess.Popen[str] | None = None
    running: bool = True
    finished_at_utc: str | None = None
    return_code: int | None = None
    logs: list[str] = field(default_factory=list)
    stop_requested: bool = False
    metrics: dict[str, Any] | None = None
    error: str | None = None


class TrainingManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._run_counter = 0
        self._current: TrainingRun | None = None
        self._max_log_lines = 8000

    def start(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._current and self._current.running:
                raise RuntimeError("A training run is already in progress.")
            if not TRAIN_SCRIPT.exists():
                raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")

            self._run_counter += 1
            cmd = build_train_command(params)
            proc = subprocess.Popen(
                cmd,
                cwd=str(APP_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            run = TrainingRun(
                run_id=self._run_counter,
                started_at_utc=utc_now_iso(),
                params=params,
                command=cmd,
                metrics_path_abs=resolve_output_path(params["metrics_path"]),
                process=proc,
                running=True,
            )
            run.logs.append(f"[{run.started_at_utc}] Started run #{run.run_id}")
            run.logs.append(f"Command: {' '.join(cmd)}")
            self._current = run

            worker = Thread(target=self._consume_output, args=(run,), daemon=True)
            worker.start()
            return self.snapshot()

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._current and self._current.running)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._current or not self._current.running:
                raise RuntimeError("No active training run to stop.")
            run = self._current
            run.stop_requested = True
            if run.process and run.process.poll() is None:
                run.process.terminate()
                run.logs.append(f"[{utc_now_iso()}] Stop requested. Sent terminate signal.")
            return self.snapshot()

    def snapshot(self, tail_lines: int = 500) -> dict[str, Any]:
        with self._lock:
            if self._current is None:
                return {
                    "has_run": False,
                    "state": "idle",
                    "running": False,
                    "log_tail": "",
                }
            run = self._current
            if run.running:
                state = "running"
            elif run.return_code == 0:
                state = "succeeded"
            elif run.stop_requested:
                state = "stopped"
            else:
                state = "failed"

            tail = "\n".join(run.logs[-max(1, tail_lines) :])
            return {
                "has_run": True,
                "state": state,
                "running": run.running,
                "run_id": run.run_id,
                "started_at_utc": run.started_at_utc,
                "finished_at_utc": run.finished_at_utc,
                "return_code": run.return_code,
                "stop_requested": run.stop_requested,
                "error": run.error,
                "params": run.params,
                "command": run.command,
                "metrics_path": str(run.metrics_path_abs),
                "metrics": run.metrics,
                "log_tail": tail,
            }

    def _append_log(self, run: TrainingRun, line: str) -> None:
        run.logs.append(line.rstrip("\n"))
        if len(run.logs) > self._max_log_lines:
            del run.logs[:1200]

    def _consume_output(self, run: TrainingRun) -> None:
        return_code = -1
        try:
            proc = run.process
            if proc is None:
                raise RuntimeError("Missing process handle for run.")
            if proc.stdout is not None:
                for line in proc.stdout:
                    with self._lock:
                        if self._current is run:
                            self._append_log(run, line)
            return_code = proc.wait()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                run.error = str(exc)
                self._append_log(run, f"[{utc_now_iso()}] Internal monitor error: {exc}")
            return_code = -1
        finally:
            with self._lock:
                run.running = False
                run.return_code = return_code
                run.finished_at_utc = utc_now_iso()
                run.process = None
                self._append_log(run, f"[{run.finished_at_utc}] Run completed with exit code {return_code}.")
                if run.metrics_path_abs.exists():
                    try:
                        with run.metrics_path_abs.open("r", encoding="utf-8") as f:
                            run.metrics = json.load(f)
                    except Exception as exc:  # noqa: BLE001
                        run.error = str(exc)
                        self._append_log(run, f"[{utc_now_iso()}] Could not parse metrics file: {exc}")


@dataclass(frozen=True)
class PipelineAction:
    key: str
    label: str
    command: list[str]
    description: str


PIPELINE_ACTIONS: dict[str, PipelineAction] = {
    "update_data": PipelineAction(
        key="update_data",
        label="Update Data",
        command=[sys.executable, "-u", str(SCRAPER_SCRIPT)],
        description="Continue scraper from checkpoint and append/update latest fight data.",
    ),
    "reset_data": PipelineAction(
        key="reset_data",
        label="Reset Data + Rescrape",
        command=[
            sys.executable,
            "-u",
            str(SCRAPER_SCRIPT),
            "--refresh-processed-events",
            "--refresh-existing-fights",
        ],
        description="Delete fight data/checkpoint and scrape a fresh full dataset.",
    ),
    "build_sequences": PipelineAction(
        key="build_sequences",
        label="Build Sequences",
        command=[sys.executable, "-u", str(SEQUENCE_SCRIPT)],
        description="Rebuild fighter-history sequence CSV from raw fight details.",
    ),
    "audit_data": PipelineAction(
        key="audit_data",
        label="Audit Data",
        command=[sys.executable, "-u", str(AUDIT_SCRIPT)],
        description="Run data integrity audit across raw and sequence datasets.",
    ),
}


@dataclass
class PipelineRun:
    run_id: int
    action_key: str
    action_label: str
    started_at_utc: str
    command: list[str]
    process: subprocess.Popen[str] | None = None
    running: bool = True
    finished_at_utc: str | None = None
    return_code: int | None = None
    logs: list[str] = field(default_factory=list)
    stop_requested: bool = False
    error: str | None = None


class PipelineManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._run_counter = 0
        self._current: PipelineRun | None = None
        self._max_log_lines = 8000

    def _cleanup_for_reset(self, run: PipelineRun) -> None:
        targets = [
            RAW_FIGHTS_CSV,
            SEQUENCE_CSV,
            SCRAPER_CHECKPOINT_DB,
            SCRAPER_CHECKPOINT_DB.with_name(SCRAPER_CHECKPOINT_DB.name + "-wal"),
            SCRAPER_CHECKPOINT_DB.with_name(SCRAPER_CHECKPOINT_DB.name + "-shm"),
        ]
        run.logs.append(f"[{utc_now_iso()}] Reset requested. Deleting existing data artifacts...")
        for path in targets:
            try:
                path.unlink()
                run.logs.append(f"Deleted: {path.relative_to(APP_ROOT)}")
            except FileNotFoundError:
                run.logs.append(f"Skip (not found): {path.relative_to(APP_ROOT)}")
            except Exception as exc:  # noqa: BLE001
                run.logs.append(f"Warning: could not delete {path.relative_to(APP_ROOT)}: {exc}")

    def start(self, action_key: str) -> dict[str, Any]:
        action = PIPELINE_ACTIONS.get(action_key)
        if action is None:
            raise ValueError(
                f"Unknown action '{action_key}'. Valid actions: {', '.join(sorted(PIPELINE_ACTIONS))}"
            )
        with self._lock:
            if self._current and self._current.running:
                raise RuntimeError("A data pipeline job is already in progress.")

            script_path = Path(action.command[2]) if len(action.command) >= 3 else None
            if script_path is not None and not script_path.exists():
                raise FileNotFoundError(f"Pipeline script not found: {script_path}")

            self._run_counter += 1
            run = PipelineRun(
                run_id=self._run_counter,
                action_key=action.key,
                action_label=action.label,
                started_at_utc=utc_now_iso(),
                command=action.command,
                running=True,
            )
            run.logs.append(f"[{run.started_at_utc}] Started job #{run.run_id}: {run.action_label}")
            run.logs.append(f"Description: {action.description}")
            if action.key == "reset_data":
                self._cleanup_for_reset(run)

            proc = subprocess.Popen(
                action.command,
                cwd=str(APP_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            run.process = proc
            run.logs.append(f"Command: {' '.join(action.command)}")
            self._current = run

            worker = Thread(target=self._consume_output, args=(run,), daemon=True)
            worker.start()
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._current or not self._current.running:
                raise RuntimeError("No active data pipeline job to stop.")
            run = self._current
            run.stop_requested = True
            if run.process and run.process.poll() is None:
                run.process.terminate()
                run.logs.append(f"[{utc_now_iso()}] Stop requested. Sent terminate signal.")
            return self.snapshot()

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._current and self._current.running)

    def snapshot(self, tail_lines: int = 500) -> dict[str, Any]:
        with self._lock:
            if self._current is None:
                return {
                    "has_run": False,
                    "state": "idle",
                    "running": False,
                    "log_tail": "",
                }
            run = self._current
            if run.running:
                state = "running"
            elif run.return_code == 0:
                state = "succeeded"
            elif run.stop_requested:
                state = "stopped"
            else:
                state = "failed"

            tail = "\n".join(run.logs[-max(1, tail_lines) :])
            return {
                "has_run": True,
                "state": state,
                "running": run.running,
                "run_id": run.run_id,
                "action_key": run.action_key,
                "action_label": run.action_label,
                "started_at_utc": run.started_at_utc,
                "finished_at_utc": run.finished_at_utc,
                "return_code": run.return_code,
                "stop_requested": run.stop_requested,
                "error": run.error,
                "command": run.command,
                "log_tail": tail,
            }

    def _append_log(self, run: PipelineRun, line: str) -> None:
        run.logs.append(line.rstrip("\n"))
        if len(run.logs) > self._max_log_lines:
            del run.logs[:1200]

    def _consume_output(self, run: PipelineRun) -> None:
        return_code = -1
        try:
            proc = run.process
            if proc is None:
                raise RuntimeError("Missing process handle for pipeline job.")
            if proc.stdout is not None:
                for line in proc.stdout:
                    with self._lock:
                        if self._current is run:
                            self._append_log(run, line)
            return_code = proc.wait()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                run.error = str(exc)
                self._append_log(run, f"[{utc_now_iso()}] Internal monitor error: {exc}")
            return_code = -1
        finally:
            with self._lock:
                run.running = False
                run.return_code = return_code
                run.finished_at_utc = utc_now_iso()
                run.process = None
                self._append_log(
                    run,
                    f"[{run.finished_at_utc}] Job completed with exit code {return_code}.",
                )


app = Flask(__name__)
trainer = TrainingManager()
pipeline = PipelineManager()


@app.route("/", methods=["GET"])
def index() -> str:
    return render_template(
        "index.html",
        defaults=DEFAULT_PARAMS,
        script_path=str(TRAIN_SCRIPT.relative_to(APP_ROOT)),
        status=trainer.snapshot(tail_lines=600),
        pipeline_status=pipeline.snapshot(tail_lines=600),
    )


@app.route("/api/train/start", methods=["POST"])
def start_training() -> Any:
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict()
    try:
        if pipeline.is_running():
            raise RuntimeError("Cannot start training while a data pipeline job is running.")
        params = parse_train_params(payload or {})
        status = trainer.start(params)
        return jsonify({"ok": True, "status": status})
    except Exception as exc:  # noqa: BLE001
        code = 409 if isinstance(exc, RuntimeError) else 400
        return jsonify({"ok": False, "error": str(exc)}), code


@app.route("/api/train/stop", methods=["POST"])
def stop_training() -> Any:
    try:
        status = trainer.stop()
        return jsonify({"ok": True, "status": status})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 409


@app.route("/api/train/status", methods=["GET"])
def training_status() -> Any:
    try:
        tail_lines = int(request.args.get("tail", "500"))
    except ValueError:
        tail_lines = 500
    tail_lines = min(max(tail_lines, 50), 3000)
    return jsonify({"ok": True, "status": trainer.snapshot(tail_lines=tail_lines)})


@app.route("/api/pipeline/start", methods=["POST"])
def start_pipeline() -> Any:
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict()
    action = str((payload or {}).get("action", "")).strip().lower()
    try:
        if trainer.is_running():
            raise RuntimeError("Cannot run data pipeline while training is active.")
        status = pipeline.start(action)
        return jsonify({"ok": True, "status": status})
    except Exception as exc:  # noqa: BLE001
        code = 409 if isinstance(exc, RuntimeError) else 400
        return jsonify({"ok": False, "error": str(exc)}), code


@app.route("/api/pipeline/stop", methods=["POST"])
def stop_pipeline() -> Any:
    try:
        status = pipeline.stop()
        return jsonify({"ok": True, "status": status})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 409


@app.route("/api/pipeline/status", methods=["GET"])
def pipeline_status() -> Any:
    try:
        tail_lines = int(request.args.get("tail", "500"))
    except ValueError:
        tail_lines = 500
    tail_lines = min(max(tail_lines, 50), 3000)
    return jsonify({"ok": True, "status": pipeline.snapshot(tail_lines=tail_lines)})


@app.route("/healthz", methods=["GET"])
def healthz() -> Any:
    return jsonify({"ok": True, "service": "ufc-lstm-pipeline-ui"})


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
