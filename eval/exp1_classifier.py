"""Experiment 1 -- failure classification quality (the centerpiece).

Hypothesis: gpualert's ordered rule classifier identifies the failure mode more
accurately than exit-code, traceback-parsing, or naive-grep baselines.

Design:
  IV  = classifier (gpualert | traceback | grep | exitcode)
  DV  = per-class and macro/micro precision/recall/F1 over the 15 modes
  Data= the labelled corpus (injected >= 30/class, plus a held-out wild set)
  Stats= 1000x bootstrap 95% CIs on macro-F1; McNemar (exact) gpualert vs grep
         on the paired per-sample correctness.

Outputs (results/):
  exp1_per_class.csv     per-class P/R/F1/support for every classifier
  exp1_summary.csv       macro/micro F1 + accuracy + bootstrap CI per classifier
  exp1_confusion.csv     gpualert 15x16 confusion counts
  exp1_confusion.png     the same as a heatmap
  exp1_mcnemar.txt       paired test gpualert vs grep
  exp1.md                a written summary of the run
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from eval import metrics
from eval.classes import ALL_PRED_LABELS, CLASSES, PRETTY
from eval.classifiers import all_classifiers
from eval.dataset import load
from eval.stats import bootstrap_metric_ci, mcnemar

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _bootstrap_macro_f1_ci(y_true, y_pred, seed=0):
    yt = np.array(y_true, dtype=object)
    yp = np.array(y_pred, dtype=object)

    def stat(idx):
        return metrics.macro_f1(yt[idx].tolist(), yp[idx].tolist())

    return bootstrap_metric_ci(y_true, stat, n_boot=1000, seed=seed)


def run(samples=None) -> dict:
    RESULTS.mkdir(exist_ok=True)
    samples = samples if samples is not None else load()
    y_true = [s["mode"] for s in samples]
    clfs = all_classifiers()

    preds = {c.name: c.predict(samples) for c in clfs}

    # --- per-class table ---
    per_class_rows = []
    for name in preds:
        prf = metrics.per_class_prf(y_true, preds[name])
        for c in CLASSES:
            r = prf[c]
            per_class_rows.append({
                "classifier": name, "class": c,
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "f1": round(r["f1"], 4),
                "support": r["support"],
            })
    with (RESULTS / "exp1_per_class.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()))
        w.writeheader()
        w.writerows(per_class_rows)

    # --- summary table with bootstrap CI on macro-F1 ---
    summary = []
    for name in preds:
        mf = metrics.macro_f1(y_true, preds[name])
        mi = metrics.micro_f1(y_true, preds[name])
        acc = metrics.accuracy(y_true, preds[name])
        lo, hi = _bootstrap_macro_f1_ci(y_true, preds[name])
        summary.append({
            "classifier": name,
            "macro_f1": round(mf, 4),
            "macro_f1_ci_lo": round(lo, 4),
            "macro_f1_ci_hi": round(hi, 4),
            "micro_f1": round(mi, 4),
            "accuracy": round(acc, 4),
        })
    summary.sort(key=lambda r: r["macro_f1"], reverse=True)
    with (RESULTS / "exp1_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # --- confusion matrix for gpualert ---
    cm = metrics.confusion(y_true, preds["gpualert"])
    with (RESULTS / "exp1_confusion.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + [PRETTY[c] for c in ALL_PRED_LABELS])
        for i, c in enumerate(CLASSES):
            w.writerow([PRETTY[c]] + cm[i].tolist())
    _plot_confusion(cm)

    # --- McNemar gpualert vs grep ---
    a = preds["gpualert"]
    b = preds["grep"]
    both = sum(1 for t, x, y in zip(y_true, a, b) if x == t and y == t)
    only_a = sum(1 for t, x, y in zip(y_true, a, b) if x == t and y != t)
    only_b = sum(1 for t, x, y in zip(y_true, a, b) if x != t and y == t)
    neither = sum(1 for t, x, y in zip(y_true, a, b) if x != t and y != t)
    mc = mcnemar(both, only_a, only_b, neither)
    (RESULTS / "exp1_mcnemar.txt").write_text(
        "McNemar, gpualert vs grep (paired per-sample correctness)\n"
        f"  both correct      : {both}\n"
        f"  only gpualert     : {only_a}\n"
        f"  only grep         : {only_b}\n"
        f"  neither           : {neither}\n"
        f"  discordant pairs  : {mc['discordant']}\n"
        f"  exact p-value     : {mc['p_value']:.3e}\n"
    )

    _write_markdown(summary, samples)
    return {"summary": summary, "mcnemar": mc}


def _plot_confusion(cm: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # row-normalise for colour, annotate with counts
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.where(row_sums == 0, 1, row_sums))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(ALL_PRED_LABELS)))
    ax.set_xticklabels([PRETTY[c] for c in ALL_PRED_LABELS], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels([PRETTY[c] for c in CLASSES], fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("gpualert confusion matrix (row-normalised)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=7, color="black" if norm[i, j] < 0.6 else "white")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp1_confusion.png", dpi=140)
    plt.close(fig)


def _write_markdown(summary, samples) -> None:
    n_inj = sum(1 for s in samples if s["source"] == "injected")
    n_wild = sum(1 for s in samples if s["source"] == "wild")
    lines = ["# Experiment 1 — classification quality", "",
             f"Corpus: {len(samples)} samples ({n_inj} injected, {n_wild} wild), "
             f"{len(CLASSES)} classes.", "",
             "| classifier | macro-F1 | 95% CI | micro-F1 | accuracy |",
             "|---|---|---|---|---|"]
    for r in summary:
        lines.append(
            f"| {r['classifier']} | {r['macro_f1']:.3f} | "
            f"[{r['macro_f1_ci_lo']:.3f}, {r['macro_f1_ci_hi']:.3f}] | "
            f"{r['micro_f1']:.3f} | {r['accuracy']:.3f} |")
    lines += ["", "See `exp1_per_class.csv` for the per-class breakdown and "
              "`exp1_confusion.png` for where gpualert's errors land "
              "(the assertion/runtime rows that fall into the generic "
              "`traceback` column are the priority-ordering limitation discussed "
              "in the paper).", ""]
    (RESULTS / "exp1.md").write_text("\n".join(lines))


if __name__ == "__main__":
    out = run()
    for r in out["summary"]:
        print(f"{r['classifier']:>10}  macroF1={r['macro_f1']:.3f} "
              f"[{r['macro_f1_ci_lo']:.3f},{r['macro_f1_ci_hi']:.3f}]  "
              f"acc={r['accuracy']:.3f}")
