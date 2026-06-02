# Experiment 1 — classification quality

Corpus: 478 samples (420 injected, 28 wild), 15 classes.

| classifier | macro-F1 | 95% CI | micro-F1 | accuracy |
|---|---|---|---|---|
| gpualert | 0.859 | [0.842, 0.876] | 0.872 | 0.872 |
| grep | 0.765 | [0.753, 0.779] | 0.797 | 0.797 |
| traceback | 0.496 | [0.487, 0.507] | 0.536 | 0.536 |
| exitcode | 0.133 | [0.133, 0.133] | 0.130 | 0.130 |

See `exp1_per_class.csv` for the per-class breakdown and `exp1_confusion.png` for where gpualert's errors land (the assertion/runtime rows that fall into the generic `traceback` column are the priority-ordering limitation discussed in the paper).
