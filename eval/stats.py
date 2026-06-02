from __future__ import annotations

import math
from typing import Sequence

import numpy as np

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

def bootstrap_ci(values: Sequence[float], n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    n = arr.size
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = arr[idx].mean()
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return (float(lo), float(hi))

def bootstrap_metric_ci(per_sample_correct, statistic, n_boot=1000,
                        alpha=0.05, seed=0):
    n = len(per_sample_correct)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = statistic(idx)
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return float(lo), float(hi)

def mcnemar(both: int, only_a: int, only_b: int, neither: int) -> dict:
    n = only_a + only_b
    if n == 0:
        return {"discordant": 0, "p_value": 1.0, "stat": 0.0}

    k = min(only_a, only_b)
    p = 0.0
    for i in range(0, k + 1):
        p += math.comb(n, i) * (0.5 ** n)
    p_value = min(1.0, 2 * p)

    stat = (abs(only_a - only_b) - 1) ** 2 / n if n > 0 else 0.0
    return {"discordant": n, "p_value": p_value, "stat": stat}

def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    row1, row2 = a + b, c + d
    col1 = a + c

    def logcomb(n_, k_):
        return (math.lgamma(n_ + 1) - math.lgamma(k_ + 1)
                - math.lgamma(n_ - k_ + 1))

    def prob(a_):
        b_ = row1 - a_
        c_ = col1 - a_
        d_ = row2 - c_
        if min(a_, b_, c_, d_) < 0:
            return 0.0
        return math.exp(logcomb(row1, a_) + logcomb(row2, c_) - logcomb(n, col1))

    p_obs = prob(a)
    lo = max(0, col1 - row2)
    hi = min(col1, row1)
    total = 0.0
    for a_ in range(lo, hi + 1):
        pa = prob(a_)
        if pa <= p_obs + 1e-12:
            total += pa
    return min(1.0, total)

def welch_t(a: Sequence[float], b: Sequence[float]) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = a.size, b.size
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return {"t": 0.0, "p_value": 1.0, "mean_diff": float(ma - mb)}
    t = (ma - mb) / se

    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return {"t": float(t), "p_value": float(p), "mean_diff": float(ma - mb)}

def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("label lists must be the same length")
    n = len(labels_a)
    if n == 0:
        return float("nan")
    cats = sorted(set(labels_a) | set(labels_b))
    po = sum(1 for x, y in zip(labels_a, labels_b) if x == y) / n
    pa = {c: labels_a.count(c) / n for c in cats}
    pb = {c: labels_b.count(c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)
