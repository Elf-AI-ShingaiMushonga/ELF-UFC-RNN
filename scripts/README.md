# Scripts Inventory

This folder contains data pipelines and training entrypoints. Use this guide to pick the right script.

## Recommended (Current)

- `scrape_ufc_fight_details.py`  
  Scrapes detailed per-fight UFC stats into `data/ufc_fight_details_lstm.csv`.

- `build_fight_history_sequences.py`  
  Converts raw fight details into rolling fighter-history sequences at `data/ufc_lstm_sequences.csv`.

- `audit_lstm_pipeline_data.py`  
  Validates raw + sequence datasets for duplicates, impossible stats, and coverage mismatches.

- `train_lstm_from_fight_details.py`  
  End-to-end LSTM training pipeline (cleaning, feature engineering, split, train, save artifacts).

- `train_lstm_from_sequences.py`  
  Trains a Siamese LSTM from `data/ufc_lstm_sequences.csv`.

- `train_lstm_xgboost_ensemble.py`  
  Trains a momentum-only Siamese LSTM, then trains XGBoost on static pre-fight features + momentum score.

## Classical / Earlier Training Scripts

- `build_prefight_tabular_dataset.py`: prepares `data/ufc_fights_cleaned.csv` from `data/ufc_fights_rnn.csv`.
- `train_classical_tabular_models.py`: RandomForest/SVM baseline on engineered diff features.
- `train_siamese_tabular_attention.py`: tabular Siamese attention network on cleaned pre-fight aggregates.

## Study / Web Predictor Support

- `siamese_study_pipeline.py`: shared utilities and study code used by `web_predictor.py`.

## Typical Workflow

From project root:

```bash
source venv/bin/activate
python scripts/scrape_ufc_fight_details.py --refresh-processed-events --refresh-existing-fights
python scripts/build_fight_history_sequences.py
python scripts/audit_lstm_pipeline_data.py
python scripts/train_lstm_from_sequences.py
# Optional: sequence + static ensemble
python scripts/train_lstm_xgboost_ensemble.py
```
