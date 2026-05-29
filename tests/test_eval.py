"""A few pinned checks. Run with: python -m pytest -q tests

These aren't exhaustive -- they guard the bits most likely to break silently:
the stats helpers (against hand-computed values), the label mapping staying in
sync with the gpualert classifier, and each baseline doing the obvious thing.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import exitcode_baseline, grep_baseline, traceback_baseline
from eval import metrics
from eval.classes import CLASSES, LABEL_TO_MODE
from eval.stats import fisher_exact_2x2, mcnemar, wilson_interval


def test_fifteen_classes():
    assert len(CLASSES) == 15
    assert len(set(CLASSES)) == 15


def test_label_mapping_matches_gpualert():
    # every gpualert label maps to a known class, and all 15 are covered.
    assert set(LABEL_TO_MODE.values()) == set(CLASSES)


def test_wilson_known():
    lo, hi = wilson_interval(20, 20)
    assert hi == 1.0
    assert 0.8 < lo < 1.0


def test_fisher_symmetric_table():
    # a perfectly even 2x2 table is not significant.
    p = fisher_exact_2x2(5, 5, 5, 5)
    assert p > 0.9


def test_fisher_separated_table():
    p = fisher_exact_2x2(20, 0, 0, 20)
    assert p < 1e-3


def test_mcnemar_lopsided():
    out = mcnemar(both=10, only_a=12, only_b=1, neither=5)
    assert out["discordant"] == 13
    assert out["p_value"] < 0.05


def test_exitcode_baseline():
    assert exitcode_baseline.classify({"true_exit_code": -11}) == "segfault"
    assert exitcode_baseline.classify({"true_exit_code": -9}) == "oom_killer"
    assert exitcode_baseline.classify({"true_exit_code": 1}) == "generic"
    assert exitcode_baseline.classify({"true_exit_code": 0}) == "generic"


def test_grep_baseline():
    s = {"log": "torch ... CUDA out of memory. Tried to allocate"}
    assert grep_baseline.classify(s) == "cuda_oom"


def test_traceback_baseline_reads_exception():
    s = {"log": "Traceback (most recent call last):\nFileNotFoundError: x"}
    assert traceback_baseline.classify(s) == "file_not_found"
    s2 = {"log": "AssertionError: nope"}
    assert traceback_baseline.classify(s2) == "assertion"


def test_macro_f1_perfect():
    # macro-F1 averages over all 15 classes, so a perfect score needs every
    # class represented and predicted correctly.
    yt = list(CLASSES)
    assert metrics.macro_f1(yt, yt) == 1.0


def test_macro_f1_partial_coverage():
    # only 3 of 15 classes present and correct -> 3/15 = 0.2 macro-F1.
    yt = ["cuda_oom", "nccl", "segfault"]
    assert abs(metrics.macro_f1(yt, yt) - 0.2) < 1e-9
