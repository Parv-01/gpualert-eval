from __future__ import annotations

import numpy as np

from eval.classes import ALL_PRED_LABELS, CLASSES

def confusion(y_true, y_pred, labels=None) -> np.ndarray:
    rows = CLASSES
    cols = labels or ALL_PRED_LABELS
    ri = {c: i for i, c in enumerate(rows)}
    ci = {c: i for i, c in enumerate(cols)}
    m = np.zeros((len(rows), len(cols)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in ri and p in ci:
            m[ri[t], ci[p]] += 1
    return m

def per_class_prf(y_true, y_pred) -> dict:
    res = {}
    for c in CLASSES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        support = sum(1 for t in y_true if t == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        res[c] = {"precision": prec, "recall": rec, "f1": f1,
                  "support": support, "tp": tp, "fp": fp, "fn": fn}
    return res

def macro_f1(y_true, y_pred) -> float:
    prf = per_class_prf(y_true, y_pred)
    f1s = [prf[c]["f1"] for c in CLASSES]
    return float(np.mean(f1s)) if f1s else 0.0

def micro_f1(y_true, y_pred) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == p and t in CLASSES)
    n = sum(1 for t in y_true if t in CLASSES)
    return tp / n if n else 0.0

def accuracy(y_true, y_pred) -> float:
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
