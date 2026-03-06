PYTHON ?= python

.PHONY: run-app scrape-lstm build-lstm-seq train-lstm pipeline-lstm

run-app:
	. venv/bin/activate && $(PYTHON) app.py

scrape-lstm:
	. venv/bin/activate && $(PYTHON) scripts/scrape_ufc_fight_details.py

build-lstm-seq:
	. venv/bin/activate && $(PYTHON) scripts/build_fight_history_sequences.py

train-lstm:
	. venv/bin/activate && $(PYTHON) scripts/train_lstm_from_fight_details.py

pipeline-lstm: scrape-lstm build-lstm-seq train-lstm
