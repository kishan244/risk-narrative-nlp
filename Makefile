install:
	pip install -r requirements-dev.txt

lint:
	ruff check .

test:
	python -m pytest -q

train:
	python -m riskradar train

build:
	python -m riskradar build

app:
	streamlit run app.py

.PHONY: install lint test train build app
