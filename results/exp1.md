# Experiment 1 — classification quality

Corpus: 474 samples (360 injected, 24 wild), 15 classes.

| classifier | macro-F1 | 95% CI | micro-F1 | accuracy |
|---|---|---|---|---|
| gpualert | 0.998 | [0.993, 1.000] | 0.998 | 0.998 |
| grep | 0.830 | [0.822, 0.838] | 0.859 | 0.859 |
| traceback | 0.559 | [0.552, 0.565] | 0.601 | 0.601 |
| exitcode | 0.133 | [0.133, 0.133] | 0.131 | 0.131 |

See `exp1_per_class.csv` for the per-class breakdown and `exp1_confusion.png` for where gpualert's errors land (the assertion/runtime rows that fall into the generic `traceback` column are the priority-ordering limitation discussed in the paper).
