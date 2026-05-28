"""Registry of the systems under test."""

from __future__ import annotations

from baselines import exitcode_baseline, grep_baseline, traceback_baseline
from eval import gpualert_adapter


class Classifier:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def predict(self, samples):
        return [self.fn(s) for s in samples]


def all_classifiers() -> list[Classifier]:
    return [
        Classifier(gpualert_adapter.NAME, gpualert_adapter.classify),
        Classifier(traceback_baseline.NAME, traceback_baseline.classify),
        Classifier(grep_baseline.NAME, grep_baseline.classify),
        Classifier(exitcode_baseline.NAME, exitcode_baseline.classify),
    ]
