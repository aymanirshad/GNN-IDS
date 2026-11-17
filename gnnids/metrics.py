"""Detection and operational metrics (threshold-free and at a validation-chosen threshold)."""
import numpy as np


def best_thr(y, s, grid=None):
    """Threshold maximising macro-F1 on VALIDATION data (never on test).
    The candidate grid also contains validation-score quantiles, so a saturated model (scores piled near 0 or 1)
    still gets a usable threshold."""
    from sklearn.metrics import f1_score
    s = np.asarray(s, dtype=float)
    if grid is None:
        grid = np.unique(np.r_[np.linspace(0.02, 0.98, 49), np.quantile(s, np.linspace(0.5, 0.9995, 120))])
    return float(grid[int(np.argmax([f1_score(y, s >= t, average="macro") for t in grid]))])


def op_thr(y, s, fpr=0.01):
    """Operational threshold: the score below which (1 - fpr) of the BENIGN validation flows lie.
    Fixes the false-alarm budget instead of maximising F1, which is unstable when the validation attack mix
    differs from the test attack mix (zero-day shift)."""
    b = np.asarray(s)[np.asarray(y) == 0]
    # nextafter: flag only scores STRICTLY above the quantile (otherwise a model whose benign scores are all 0 flags everything)
    return float(np.nextafter(np.quantile(b, 1 - fpr), np.inf)) if len(b) else 0.5


def metrics(y, s, thr):
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                                 matthews_corrcoef, roc_auc_score, roc_curve)
    y = np.asarray(y).astype(int)
    p = (s >= thr).astype(int)
    two = len(np.unique(y)) == 2
    out = dict(f1_macro=float(f1_score(y, p, average="macro")), mcc=float(matthews_corrcoef(y, p)),
               accuracy=float(accuracy_score(y, p)), balanced_accuracy=float(balanced_accuracy_score(y, p)),
               recall=float(p[y == 1].mean()) if (y == 1).any() else 0.0,
               fpr=float(p[y == 0].mean()) if (y == 0).any() else 0.0,
               n=int(len(y)), n_attack=int(y.sum()), prevalence=float(y.mean()))
    if two:
        fpr, tpr, _ = roc_curve(y, s)
        fpr_at = lambda t: float(fpr[min(np.searchsorted(tpr, t, side="left"), len(fpr) - 1)])
        out.update(pr_auc=float(average_precision_score(y, s)), roc_auc=float(roc_auc_score(y, s)),
                   tpr_at_1pct_fpr=float(np.interp(0.01, fpr, tpr)),
                   tpr_at_0p1pct_fpr=float(np.interp(0.001, fpr, tpr)),
                   fpr_at_95tpr=fpr_at(0.95), fpr_at_99tpr=fpr_at(0.99))
    else:  # degenerate subset (only one class): threshold-free metrics undefined
        out.update(pr_auc=float("nan"), roc_auc=float("nan"), tpr_at_1pct_fpr=float("nan"),
                   tpr_at_0p1pct_fpr=float("nan"), fpr_at_95tpr=float("nan"), fpr_at_99tpr=float("nan"))
    return out


def per_family(fam, s, thr, names):
    """Recall per attack family at the given threshold."""
    out = {}
    for c, nm in enumerate(names):
        m = fam == c
        if nm != "Benign" and m.any():
            out[nm] = dict(n=int(m.sum()), recall=float((s[m] >= thr).mean()))
    return out


def bootstrap_ci(y, s, fn, n_boot=200, seed=0, alpha=0.05):
    """Percentile bootstrap CI of fn(y, s) over test flows. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(fn(y[i], s[i]))
    return float(fn(y, s)), float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
