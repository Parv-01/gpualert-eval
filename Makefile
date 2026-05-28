# Reproduce the whole evaluation end to end.
#
#   make corpus   build the labelled corpus (injected fixtures + wild set)
#   make eval     run experiments 1-5 against the gpualert classifier
#   make all      corpus + eval
#   make clean    drop generated corpus/results (keeps the wild set + references)
#
# Point GPUALERT_SRC at a checkout of the main gpualert package if it isn't
# already importable, e.g.:
#   GPUALERT_SRC=../gpualert make all

PY ?= python3

.PHONY: all corpus eval clean test

all: corpus eval

corpus:
	$(PY) corpus/build.py

eval:
	$(PY) eval/run_all.py

test:
	$(PY) -m pytest -q tests

clean:
	rm -f corpus/labels.jsonl
	rm -rf corpus/injected
	rm -f results/*.csv results/*.png results/*.md
