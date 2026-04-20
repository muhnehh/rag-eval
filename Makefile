.PHONY: install ingest eval generate-data finetune benchmark push test all

install:
	pip install -e ".[dev,finetune]"

ingest:
	rageval ingest --docs-dir data/raw/

eval:
	rageval eval --golden data/golden/golden_qa.json

generate-data:
	python finetune/generate_data.py

finetune:
	python finetune/train.py

benchmark:
	python finetune/evaluate.py

push:
	python finetune/push_to_hub.py

test:
	pytest tests/ -v

all: generate-data finetune benchmark push
