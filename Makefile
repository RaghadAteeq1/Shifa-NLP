PY ?= python
# raise WORKERS to 2-4 on a multi-core server if p99 under load is above budget
WORKERS ?= 1
.PHONY: quickstart all data baseline dialect audit train ner index export serve load bench bench-batch eval final figures report test clean

# Fast path, no GPU and no downloads: data + TF-IDF baseline + dialect ID, then serve (baseline fallback).
quickstart: data baseline dialect serve

# Full pipeline (train on a GPU, e.g. Colab T4, then copy artifacts/ back).
all: data baseline dialect audit train ner index export bench bench-batch eval

data:       ; $(PY) -m scripts.make_data
baseline:   ; $(PY) -m scripts.train_baseline
dialect:    ; $(PY) -m scripts.train_dialect
audit:      ; $(PY) -m scripts.tokenizer_audit
train:      ; $(PY) -m scripts.train_topic
ner:        ; $(PY) -m scripts.train_ner
index:      ; $(PY) -m scripts.build_index
export:     ; $(PY) -m scripts.export_onnx
bench:      ; $(PY) -m scripts.benchmark
bench-batch:; $(PY) -m scripts.bench_batch
eval:       ; $(PY) -m scripts.evaluate --split val
serve:      ; $(PY) -m uvicorn shifa.serve:app --host 0.0.0.0 --port 8000 --workers $(WORKERS)
load:       ; $(PY) -m scripts.load_test
figures:    ; $(PY) -m scripts.make_figures
report:     ; $(PY) -m scripts.make_figures && $(PY) -m scripts.make_benchmarks_md
test:       ; $(PY) -m pytest -q

# The ONE time the frozen test split is opened. Every access is logged to reports/test_access.log.
final:
	SHIFA_FINAL=1 $(PY) -m scripts.evaluate --split test
	$(MAKE) report

# Keeps the hand-tagged error taxonomy (hours of reading) and the frozen-test access log (R7 evidence).
KEEP = errors_tagged.csv test_access.log
clean:
	rm -rf artifacts/*
	find reports -mindepth 1 $(foreach f,$(KEEP),! -name $(f)) -delete 2>/dev/null || true
