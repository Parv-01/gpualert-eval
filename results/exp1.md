# Experiment 1 — classification quality

Corpus: 510 samples (480 injected, 30 wild), 15 classes.

| classifier | macro-F1 | 95% CI | micro-F1 | accuracy |
|---|---|---|---|---|
| gpualert | 0.905 | [0.884, 0.924] | 0.910 | 0.910 |
| grep | 0.830 | [0.822, 0.838] | 0.861 | 0.861 |
| traceback | 0.558 | [0.551, 0.565] | 0.596 | 0.596 |
| exitcode | 0.133 | [0.133, 0.133] | 0.133 | 0.133 |

See `exp1_per_class.csv` for the per-class breakdown and `exp1_confusion.png` for where gpualert's errors land (the assertion/runtime rows that fall into the generic `traceback` column are the priority-ordering limitation discussed in the paper).
