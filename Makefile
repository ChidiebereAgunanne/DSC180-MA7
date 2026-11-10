.PHONY: all data features train test clean

all: data features train

data:
	python -m src.ingest

features:
	python -m src.features

train:
	python -m src.train

test:
	python -m pytest tests/ -q

clean:
	rm -f data/processed/*.parquet reports/metrics.json reports/*.png
