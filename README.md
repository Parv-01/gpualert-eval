# gpualert-eval

Evaluation code and dataset for gpualert (https://github.com/Parv-01/gpualert).

```
corpus/     labelled fault logs (labels.jsonl) and reference captures
inject/     fault injectors and the on-node capture script
baselines/  exit-code, traceback and grep classifiers
eval/       experiments 1-5 and metrics
results/    generated tables and figures
```

```
pip install -r requirements.txt
pip install -e ../gpualert
python eval/run_all.py
```
