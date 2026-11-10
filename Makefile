.PHONY: all data features train valid test clean

all: data features train

data:
	python -m src.ingest

features:
	python -m src.features

train:
	python -m src.train

# Feature trials: fit on 2011, score H1 2012 only. Never touches test rows.
valid:
	python -m src.train --valid-only

test:
	python -m pytest tests/ -q

clean:
	rm -f data/processed/*.parquet reports/metrics.json reports/metrics_valid.json reports/*.png
