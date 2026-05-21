.PHONY: eval benchmark test lint

MODEL   ?= model.p
DATA    ?= data.pickle
OUT_DIR ?= .
N       ?= 1000
WARMUP  ?= 100

eval:
	python eval.py --model $(MODEL) --data $(DATA) --out-dir $(OUT_DIR)

benchmark:
	python benchmark.py --model $(MODEL) --n $(N) --warmup $(WARMUP)

test:
	pytest tests/ -v

lint:
	ruff check src/ eval.py benchmark.py tests/
