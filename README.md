# UFC Elf

UFC LSTM training project with:
- a Flask training console (`app.py`) for `scripts/train_lstm_from_sequences.py`
- data scraping and sequence/model pipelines under `scripts/`

## Quick Start (Web App)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

App default: `http://0.0.0.0:8000`.

The web UI now exposes:
- data pipeline controls: `Update Data`, `Reset Data + Rescrape`, `Build Sequences`, `Audit Data`
- sequence-model hyperparameters and training jobs with live logs

## Data + LSTM Pipeline (Current)

Run from repo root:

```bash
source venv/bin/activate
python scripts/scrape_ufc_fight_details.py --refresh-processed-events --refresh-existing-fights
python scripts/build_fight_history_sequences.py
python scripts/audit_lstm_pipeline_data.py
python scripts/train_lstm_from_sequences.py
```

Notes:
- Scraper output: `data/ufc_fight_details_lstm.csv`
- Sequence output: `data/ufc_lstm_sequences.csv`
- Trained artifacts: `data/model_cache/lstm_fight_details/`

## Script Guide

See `scripts/README.md` for a clean inventory of scripts and recommended entrypoints.

## Environment Variables

- `PORT` (default: `8000`)
- `FLASK_HOST` (default: `0.0.0.0`)
- `FLASK_DEBUG` (default: `0`)

## Deployment

EC2 deployment docs: `deploy/DEPLOY_EC2.md`.
