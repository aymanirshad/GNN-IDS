"""Publication-quality figures and LaTeX/CSV tables for the paper. Every figure is saved as PNG (300 dpi) and
PDF (vector) so it can go straight into LaTeX/Word. Figures are skipped (with a message) if their data is missing."""
import glob
import json
import os
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .config import MODEL_NAMES
from .results_io import load_results, load_scores

# Okabe-Ito colour-blind-safe palette
COL = {"xgb": "#0072B2", "lgbm": "#56B4E9", "rf": "#009E73", "topo": "#8C8C8C", "iponly": "#2B2B2B",
       "tegs_nomem": "#E69F00", "tegs_nomem_adv": "#CC79A7", "tegs": "#D55E00", "tegs_adv": "#8E1B1B",
       "tegs_nomem_nf": "#F0E442", "tegs_nomem_nf_adv": "#B8A800", "tegs_nf": "#F28E8E", "tegs_nf_adv": "#7A3E9D"}
ORDER = ["iponly", "rf", "lgbm", "xgb", "topo", "tegs_nomem", "tegs_nomem_nf", "tegs_nomem_adv", "tegs_nomem_nf_adv", "tegs", "tegs_nf", "tegs_adv", "tegs_nf_adv"]
METRICS = ["pr_auc", "roc_auc", "f1_macro", "mcc", "accuracy", "balanced_accuracy", "tpr_at_1pct_fpr", "fpr_at_95tpr", "recall", "fpr"]
MLAB = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "f1_macro": "Macro-F1", "mcc": "MCC", "accuracy": "Accuracy", "balanced_accuracy": "Balanced acc.",
        "tpr_at_1pct_fpr": "TPR @ 1% FPR", "fpr_at_95tpr": "FPR @ 95% TPR", "recall": "Recall", "fpr": "FPR"}
DSN = {"UNSW": "NF-UNSW-NB15-v3", "UNSWb": "NF-UNSW-NB15-v3 (blocked split)", "CIC": "NF-CSE-CIC-IDS2018-v3 (chronological)",
       "CICb": "NF-CSE-CIC-IDS2018-v3 (blocked split)", "CICc": "NF-CSE-CIC-IDS2018-v3 (class-chronological)"}


def lab(m):
    return MODEL_NAMES.get(m, m)


def dn(d):
    return DSN.get(d, d)


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"], "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
        "legend.frameon": False, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.axisbelow": True, "pdf.fonttype": 42, "axes.titlepad": 6,
        "figure.constrained_layout.use": True})


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  figure:", name)


def panel_label(ax, i):
    ax.text(0.015, 0.975, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.85), zorder=10)


def ordered(models):
    return [m for m in ORDER if m in set(models)] + sorted(set(models) - set(ORDER))


# ------------------------------------------------------------------------------------------ data frames
def build_df(R):
    rows = []
    for r in R:
        if r.get("tag") or r.get("holdout"):
            continue
        t, c = r["test"], r.get("test_clean") or {}
        tr = r.get("transfer") or {}
        trm = tr.get("metrics") or {}
        lat = r.get("latency") or {}
        us = lat.get("mean_us_per_event", r.get("sec_per_flow_infer", np.nan) * 1e6 if "sec_per_flow_infer" in r else np.nan)
        rows.append(dict(dataset=r["dataset"], model=r["model_base"], seed=r["seed"],
                         **{k: t.get(k, np.nan) for k in METRICS}, **{"c_" + k: c.get(k, np.nan) for k in METRICS},
                         target=tr.get("target"), tr_pr_auc=trm.get("pr_auc", np.nan), tr_f1=trm.get("f1_macro", np.nan),
                         tr_roc=trm.get("roc_auc", np.nan), tr_prev=trm.get("prevalence", np.nan),
                         prevalence=t.get("prevalence", np.nan), us_event=us, p95_ms=lat.get("p95_ms_per_batch", np.nan),
                         op_recall=(r.get("test_op") or {}).get("recall", np.nan), op_fpr=(r.get("test_op") or {}).get("fpr", np.nan),
                         op_f1=(r.get("test_op") or {}).get("f1_macro", np.nan), op_acc=(r.get("test_op") or {}).get("accuracy", np.nan)))
    return pd.DataFrame(rows)


def ms(df, cols, by=("dataset", "model")):
    g = df.groupby(list(by))[cols]
    return g.mean(), g.std().fillna(0.0), g.count()


# ------------------------------------------------------------------------------------------ figures
def fig_dataset_overview(data_dir, out):
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    A = [(os.path.basename(f)[:-5], json.load(open(f))["audit"]) for f in files]
    if not A:
        return print("  skipped dataset overview (no data/*.json)")
    fig, axs = plt.subplots(len(A), 3, figsize=(12.5, 3.4 * len(A)), squeeze=False,
                            gridspec_kw=dict(width_ratios=[1, 1.5, 1], wspace=.35, hspace=.55))
    for i, (name, a) in enumerate(A):
        ax = axs[i, 0]
        rate = np.array(a["attack_rate_by_split"]) * 100
        bars = ax.bar(["Train\n(60%)", "Validation\n(20%)", "Test\n(20%)"], rate, color=["#4C78A8", "#72B7B2", "#E45756"], width=.6)
        for b, r, n in zip(bars, rate, a["split_sizes"]):
            ax.text(b.get_x() + b.get_width() / 2, r, f"{r:.1f}%\nn={n:,}", ha="center", va="bottom", fontsize=7.5)
        ax.set_ylim(0, max(rate) * 1.35); ax.set_ylabel("Attack flows (%)")
        ax.set_title(f"{dn(name)}\nclass balance per {a.get('split_mode', 'time-ordered')} split"); panel_label(ax, 0)
        ax = axs[i, 1]
        fams = {k: v for k, v in a["family_counts"].items() if k != "Benign"}
        keys = sorted(fams, key=fams.get)
        byspl = a.get("family_counts_by_split") or {f: [a["family_counts"][f]] * 3 for f in keys}
        if "family_counts_by_split" not in a:
            ax.text(.5, .5, "re-run prep/resplit to see\nper-split family counts", transform=ax.transAxes, ha="center", fontsize=8, color="#999")
        ys = np.arange(len(keys))
        for k_, (c_, nm) in enumerate(zip(["#4C78A8", "#72B7B2", "#E45756"], ["train", "val", "test"])):
            ax.scatter([max(byspl[f][k_], .8) for f in keys], ys, color=c_, s=30, label=nm, zorder=3)
        ax.hlines(ys, [max(min(byspl[f]), .8) for f in keys], [max(max(byspl[f]), .8) for f in keys], color="#bbb", lw=1, zorder=1)
        ax.set_yticks(ys); ax.set_yticklabels(keys); ax.set_xscale("log"); ax.set_xlabel("Attack flows (log scale)")
        ax.set_title("Attack families across splits\n(point at the left edge = absent from that split)")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=3, handletextpad=.2, columnspacing=.8); panel_label(ax, 1)
        ax = axs[i, 2]
        v = [a.get("val_rows_with_identical_features_in_train", 0.0) * 100,
             a["test_rows_with_identical_features_in_train"] * 100]
        bs = ax.bar(["Validation", "Test"], v, color=["#72B7B2", "#E45756"], width=.55)
        for b, x in zip(bs, v):
            ax.text(b.get_x() + b.get_width() / 2, x, f"{x:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(max(v) * 1.3, 5)); ax.set_ylabel("Rows (%)")
        ax.set_title("Leakage: flows whose exact feature\nvector also occurs in training"); panel_label(ax, 2)
    save(fig, out, "fig01_dataset_overview")


def fig_main(df, out, datasets):
    cols = ["pr_auc", "f1_macro", "tpr_at_1pct_fpr"]
    M, S, _ = ms(df, cols + ["c_" + c for c in cols])
    fig, axs = plt.subplots(len(datasets), 3, figsize=(11.5, 0.36 * len(df.model.unique()) * len(datasets) + 1.1 * len(datasets) + .5),
                            squeeze=False, sharey="row")
    for i, d in enumerate(datasets):
        models = ordered(df[df.dataset == d].model)
        for j, c in enumerate(cols):
            ax = axs[i, j]
            y = np.arange(len(models))[::-1]
            for yy, m in zip(y, models):
                mu, sd, cl = M.loc[(d, m), c], S.loc[(d, m), c], M.loc[(d, m), "c_" + c]
                ax.barh(yy, mu, xerr=sd, color=COL.get(m, "#888"), height=.68, error_kw=dict(lw=1, capsize=2),
                        hatch="////" if m == "iponly" else None, edgecolor="white" if m == "iponly" else "none")
                ax.text(1.03, yy, f"{mu:.3f}", va="center", ha="left", fontsize=7.5, color="#222")
                if not np.isnan(cl):
                    ax.scatter([cl], [yy], marker="D", s=22, color="white", edgecolor="black", zorder=4, linewidths=.8)
            ax.set_yticks(y); ax.set_yticklabels([lab(m) for m in models]); ax.set_xlim(0, 1.0); ax.set_xticks([0, .25, .5, .75, 1])
            ax.set_xlabel(MLAB[c])
            if i == 0:
                ax.set_title(MLAB[c])
            if j == 0:
                ax.set_ylabel(dn(d), fontweight="bold", fontsize=10)
            panel_label(ax, i * 3 + j)
    fig.legend(handles=[Line2D([], [], marker="D", color="white", markeredgecolor="black", ls="", label="leakage-cleaned test set"),
                        Patch(facecolor="#888", label="full test set, mean \u00b1 std over seeds")],
               loc="outside lower center", ncol=2)
    fig.suptitle("In-dataset detection on a time-ordered split", fontweight="bold")
    save(fig, out, "fig02_main_comparison")


def fig_leakage_effect(df, out, datasets):
    cols = ["pr_auc", "tpr_at_1pct_fpr"]
    M, _, _ = ms(df, cols + ["c_" + c for c in cols])
    if M[["c_" + c for c in cols]].isna().all().all():
        return print("  skipped leakage-effect (no clean-test results yet)")
    fig, axs = plt.subplots(len(datasets), 2, figsize=(9.5, 3.0 * len(datasets) + .3), squeeze=False, sharey="row")
    for i, d in enumerate(datasets):
        models = ordered(df[df.dataset == d].model)
        for j, c in enumerate(cols):
            ax = axs[i, j]; y = np.arange(len(models))[::-1]
            for yy, m in zip(y, models):
                a_, b_ = M.loc[(d, m), c], M.loc[(d, m), "c_" + c]
                ax.plot([a_, b_], [yy, yy], color="#bbb", lw=2, zorder=1)
                ax.scatter([a_], [yy], color=COL.get(m, "#888"), s=45, zorder=3)
                ax.scatter([b_], [yy], facecolor="white", edgecolor=COL.get(m, "#888"), s=45, zorder=3, linewidths=1.5)
            ax.set_yticks(y); ax.set_yticklabels([lab(m) for m in models]); ax.set_xlabel(MLAB[c])
            if i == 0:
                ax.set_title(MLAB[c])
            if j == 0:
                ax.set_ylabel(dn(d), fontweight="bold")
            panel_label(ax, i * 2 + j)
    fig.legend(handles=[Line2D([], [], marker="o", color="#555", ls="", label="full test set"),
                        Line2D([], [], marker="o", markerfacecolor="white", color="#555", ls="", label="leakage-cleaned (duplicates removed)")],
               loc="outside lower center", ncol=2)
    fig.suptitle("How much of the score is duplicate leakage?", fontweight="bold")
    save(fig, out, "fig03_leakage_effect")


def _first_scores(results_dir, R, dataset, model):
    cand = sorted((r["seed"], r) for r in R if r["dataset"] == dataset and r["model_base"] == model
                  and not r.get("tag") and not r.get("holdout"))
    for _, r in cand:
        name = f"{r['dataset']}_{model}_s{r['seed']}"
        s = load_scores(results_dir, name)
        if s is not None:
            return s
    return None


def fig_curves(results_dir, R, out, datasets):
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, axs = plt.subplots(len(datasets), 2, figsize=(10, 4.2 * len(datasets)), squeeze=False)
    got = False
    for i, d in enumerate(datasets):
        for m in ordered([r["model_base"] for r in R if r["dataset"] == d]):
            sc = _first_scores(results_dir, R, d, m)
            if sc is None or len(np.unique(sc["y"])) < 2:
                continue
            got = True
            s = sc["s"].astype(float); y = sc["y"]
            f, t, _ = roc_curve(y, s); p, r_, _ = precision_recall_curve(y, s)
            ls = "--" if m == "iponly" else "-"
            axs[i, 0].plot(np.maximum(f, 1e-6), t, color=COL.get(m, "#888"), ls=ls, lw=1.6, label=lab(m))
            axs[i, 1].plot(r_, p, color=COL.get(m, "#888"), ls=ls, lw=1.6)
        ax = axs[i, 0]; ax.set_xscale("log"); ax.set_xlim(1e-5, 1); ax.set_ylim(0, 1.02)
        ax.axvline(.01, color="#999", lw=.8, ls=":"); ax.text(.0105, .03, "1% FPR", fontsize=7, color="#666")
        ax.set_xlabel("False-positive rate (log scale)"); ax.set_ylabel("True-positive rate")
        ax.set_title(f"{dn(d)}: ROC"); panel_label(ax, 2 * i)
        ax = axs[i, 1]; ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02); ax.set_xlim(0, 1)
        ax.set_title(f"{dn(d)}: precision\u2013recall"); panel_label(ax, 2 * i + 1)
        axs[i, 0].legend(loc="lower right")
    if not got:
        return plt.close(fig) or print("  skipped curves (no score files)")
    save(fig, out, "fig04_roc_pr_curves")


def fig_score_hist(results_dir, R, out, datasets, models=("xgb", "tegs_nomem", "tegs", "tegs_adv")):
    rows = []
    for d in datasets:
        for m in models:
            sc = _first_scores(results_dir, R, d, m)
            if sc is not None:
                rows.append((d, m, sc))
    if not rows:
        return print("  skipped score histograms")
    nm = len(models)
    fig, axs = plt.subplots(len(datasets), nm, figsize=(2.9 * nm, 2.6 * len(datasets)), squeeze=False, sharey="row")
    for (d, m, sc) in rows:
        ax = axs[datasets.index(d), list(models).index(m)]
        s = sc["s"].astype(float); y = sc["y"]; bins = np.linspace(0, 1, 41)
        ax.hist(s[y == 0], bins, color="#4C78A8", alpha=.75, label="benign")
        ax.hist(s[y == 1], bins, color="#E45756", alpha=.75, label="attack")
        ax.axvline(float(sc["thr"]), color="black", ls="--", lw=1); ax.set_yscale("log")
        ax.set_title(lab(m) if datasets.index(d) == 0 else ""); ax.set_xlabel("predicted attack probability")
        if list(models).index(m) == 0:
            ax.set_ylabel(f"{dn(d)}\nflows (log)")
    axs[0, 0].legend(loc="upper center")
    fig.suptitle("Score distributions on the test set (dashed = validation-chosen threshold)", fontweight="bold")
    save(fig, out, "fig05_score_distributions")


def fig_family_heatmap(R, out, datasets):
    n = len(datasets)
    fig, axs = plt.subplots(1, n, figsize=(7.2 * n, 4.6), squeeze=False)
    for i, d in enumerate(datasets):
        rs = [r for r in R if r["dataset"] == d and not r.get("tag") and not r.get("holdout") and r.get("per_family")]
        if not rs:
            continue
        models = ordered([r["model_base"] for r in rs])
        fams = sorted({f for r in rs for f in r["per_family"]}, key=lambda f: -max(r["per_family"].get(f, {"n": 0})["n"] for r in rs))
        Z = np.full((len(models), len(fams)), np.nan)
        for a_, m in enumerate(models):
            for b_, f in enumerate(fams):
                v = [r["per_family"][f]["recall"] for r in rs if r["model_base"] == m and f in r["per_family"]]
                if v:
                    Z[a_, b_] = np.mean(v)
        ax = axs[0, i]
        im = ax.imshow(Z, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        for a_ in range(Z.shape[0]):
            for b_ in range(Z.shape[1]):
                if not np.isnan(Z[a_, b_]):
                    ax.text(b_, a_, f"{Z[a_, b_]:.2f}", ha="center", va="center", fontsize=7.5)
        nn = {f: max(r["per_family"][f]["n"] for r in rs if f in r["per_family"]) for f in fams}
        ax.set_xticks(range(len(fams))); ax.set_xticklabels([f"{f}\n(n={nn[f]:,})" for f in fams], rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(models))); ax.set_yticklabels([lab(m) for m in models])
        ax.set_title(f"{dn(d)}: recall per attack family"); panel_label(ax, i)
        plt.colorbar(im, ax=ax, fraction=.035, pad=.02, label="recall")
    save(fig, out, "fig06_family_recall_heatmap")


def fig_robustness(R, out, datasets):
    rs = [r for r in R if r.get("robust") and not r.get("holdout") and not r.get("tag")]
    if not rs:
        return print("  skipped robustness (no --robust runs)")
    ds = [d for d in datasets if any(r["dataset"] == d for r in rs)]
    fig, axs = plt.subplots(len(ds), 3, figsize=(12.5, 3.6 * len(ds)), squeeze=False, gridspec_kw=dict(width_ratios=[1.3, 1.3, .8]))
    for i, d in enumerate(ds):
        models = ordered([r["model_base"] for r in rs if r["dataset"] == d])
        for j, (fam, ttl) in enumerate([("inject_src", "Edge injection from attackers\n(training family)"),
                                        ("inject_dst", "Edge injection into victims\n(HELD-OUT family)")]):
            ax = axs[i, j]
            for m in models:
                recs = [(x["k"], x.get("detection_rate_op", x["detection_rate"]), r["seed"]) for r in rs if r["dataset"] == d and r["model_base"] == m
                        for x in r["robust"] if x["family"] == fam]
                df = pd.DataFrame(recs, columns=["k", "v", "seed"]).groupby("k").v.agg(["mean", "std"]).fillna(0)
                xs = np.arange(len(df))
                ax.plot(xs, df["mean"], marker="o", ms=4, lw=1.8, color=COL.get(m, "#888"), label=lab(m))
                ax.fill_between(xs, df["mean"] - df["std"], df["mean"] + df["std"], color=COL.get(m, "#888"), alpha=.18, lw=0)
            ax.set_xticks(xs); ax.set_xticklabels(df.index.astype(int)); ax.set_ylim(-.02, 1.03)
            ax.set_xlabel("fake edges added per attacker/victim host"); ax.set_ylabel("detection rate (recall)")
            ax.set_title(f"{dn(d)}\n{ttl}"); panel_label(ax, 3 * i + j)
        ax = axs[i, 2]
        v = []
        for m in models:
            vals = [x.get("detection_rate_op", x["detection_rate"]) for r in rs if r["dataset"] == d and r["model_base"] == m for x in r["robust"] if x["family"] == "drop_benign"]
            v.append((m, np.mean(vals) if vals else np.nan, np.std(vals) if vals else 0))
        ax.bar(range(len(v)), [x[1] for x in v], yerr=[x[2] for x in v], color=[COL.get(x[0], "#888") for x in v], capsize=2)
        ax.set_xticks(range(len(v))); ax.set_xticklabels([lab(x[0]) for x in v], fontsize=6.5, rotation=30, ha="right")
        ax.set_ylim(0, 1.05); ax.set_title("50% of benign flows\ndeleted (held-out)"); ax.set_ylabel("detection rate"); panel_label(ax, 3 * i + 2)
        if i == 0:
            axs[i, 0].legend(loc="lower left")
    fig.suptitle("Structural robustness at the threshold that gives 1% false alarms on clean validation data", fontweight="bold")
    save(fig, out, "fig07_structural_robustness")


def fig_window(R, out, datasets):
    rs = [r for r in R if r["model_base"].startswith("tegs") and not r.get("holdout")
          and (r.get("tag", "") == "" or str(r.get("tag", "")).startswith("_d"))]
    rows = [dict(dataset=r["dataset"], model=r["model_base"], delta=r["delta"], seed=r["seed"], pr_auc=r["test"].get("pr_auc", np.nan),
                 f1=r["test"]["f1_macro"], p95=(r.get("latency") or {}).get("p95_ms_per_batch", np.nan)) for r in rs]
    df = pd.DataFrame(rows)
    if df.empty or df.groupby(["dataset", "model"]).delta.nunique().max() < 2:
        return print("  skipped window sensitivity (run --delta sweeps with --tag _d30 ...)")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for k, (c, t) in enumerate([("pr_auc", "PR-AUC"), ("f1", "Macro-F1"), ("p95", "p95 latency per window (ms)")]):
        for (d, m), g in df.groupby(["dataset", "model"]):
            if g.delta.nunique() < 2:
                continue
            a = g.groupby("delta")[c].agg(["mean", "std"]).fillna(0)
            ls = "-" if d == datasets[0] else "--"
            axs[k].errorbar(a.index, a["mean"], yerr=a["std"], marker="o", ms=4, capsize=2, lw=1.6, ls=ls,
                            color=COL.get(m, "#888"), label=f"{lab(m)} / {dn(d)}")
        axs[k].set_xscale("log"); axs[k].set_xticks(sorted(df.delta.unique())); axs[k].set_xticklabels(sorted(df.delta.unique()))
        axs[k].set_xlabel("window length \u0394 (s)"); axs[k].set_ylabel(t); axs[k].set_title(t); panel_label(axs[k], k)
    axs[0].legend(fontsize=6.5)
    fig.suptitle("Sensitivity to the snapshot window", fontweight="bold")
    save(fig, out, "fig08_window_sensitivity")


def fig_latency(df, R, out):
    d = df.dropna(subset=["us_event"])
    sweep = [(r, x) for r in R for x in (r.get("latency_sweep") or []) if not r.get("tag") and not r.get("holdout")]
    ncol = 3 if sweep else 1
    fig, axs = plt.subplots(1, ncol, figsize=(4.4 * ncol + 1.5, 3.8), squeeze=False)
    ax = axs[0, 0]
    if not d.empty:
        a = d.groupby("model").us_event.mean().reindex(ordered(d.model.unique())).dropna()
        ax.barh([lab(m) for m in a.index][::-1], a.values[::-1], color=[COL.get(m, "#888") for m in a.index][::-1])
        ax.set_xscale("log"); ax.set_xlabel("inference cost per flow (\u00b5s, log scale)")
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_title("Batch inference cost"); panel_label(ax, 0)
    if sweep:
        sw = pd.DataFrame([dict(model=r["model_base"], seed=r["seed"], **x) for r, x in sweep])
        for k, (c, t) in enumerate([("p95_ms", "p95 latency per micro-batch (ms)"), ("events_per_s", "throughput (events / s)")]):
            ax = axs[0, k + 1]
            for (m, dev), g in sw.groupby(["model", "device"]):
                a = g.groupby("cap")[c].agg(["mean", "std"]).fillna(0)
                ax.errorbar(a.index, a["mean"], yerr=a["std"], marker="o" if dev == "cpu" else "s", ms=4, capsize=2,
                            ls="-" if dev == "cpu" else "--", color=COL.get(m, "#888"), label=f"{lab(m)} [{dev.upper()}]")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("micro-batch size (events)"); ax.set_ylabel(t)
            ax.set_title(t); panel_label(ax, k + 1)
        axs[0, 1].legend(fontsize=6.5)
    fig.suptitle("Streaming cost (RQ3)", fontweight="bold")
    save(fig, out, "fig09_latency_throughput")


def fig_transfer(df, out):
    t = df.dropna(subset=["target"])
    if t.empty:
        return print("  skipped transfer (no --target runs)")
    pairs = sorted(t.groupby(["dataset", "target"]).size().index.tolist())
    fig, axs = plt.subplots(len(pairs), 2, figsize=(10.5, 3.1 * len(pairs)), squeeze=False)
    for i, (d, tg) in enumerate(pairs):
        g = t[(t.dataset == d) & (t.target == tg)]
        models = ordered(g.model.unique())
        for j, (ci, ct, ttl) in enumerate([("pr_auc", "tr_pr_auc", "PR-AUC"), ("f1_macro", "tr_f1", "Macro-F1 (source threshold)")]):
            ax = axs[i, j]; w = .38; x = np.arange(len(models))
            a = g.groupby("model")[ci].agg(["mean", "std"]).reindex(models).fillna(0)
            b = g.groupby("model")[ct].agg(["mean", "std"]).reindex(models).fillna(0)
            ax.bar(x - w / 2, a["mean"], w, yerr=a["std"], color=[COL.get(m, "#888") for m in models], capsize=2)
            ax.bar(x + w / 2, b["mean"], w, yerr=b["std"], color=[COL.get(m, "#888") for m in models], hatch="////",
                   edgecolor="white", alpha=.75, capsize=2)
            if ci == "pr_auc" and not g.tr_prev.isna().all():
                ax.axhline(g.tr_prev.mean(), color="black", ls=":", lw=1)
                ax.text(-.45, g.tr_prev.mean() + .015, "chance level (target prevalence)", ha="left", fontsize=7)
            ax.set_xticks(x); ax.set_xticklabels([lab(m).replace(" (", "\n(").replace(" + ", "\n+ ") for m in models], fontsize=6.8)
            ax.set_ylim(0, 1.05); ax.set_ylabel(ttl)
            ax.set_title(f"{dn(d)} \u2192 {dn(tg)}: {ttl.split(' (')[0]}"); panel_label(ax, 2 * i + j)
    fig.legend(handles=[Patch(facecolor="#777", label="in-dataset test"), Patch(facecolor="#777", hatch="////", edgecolor="white", alpha=.75, label="cross-dataset transfer")],
               loc="outside lower center", ncol=2)
    fig.suptitle("Cross-dataset transfer (train on one network, test on another)", fontweight="bold")
    save(fig, out, "fig10_cross_dataset_transfer")


def fig_training(R, out, datasets):
    rs = [r for r in R if r.get("history") and not r.get("tag") and not r.get("holdout")]
    if not rs:
        return print("  skipped training curves (no history; rerun tegs with v0.2)")
    ds = [d for d in datasets if any(r["dataset"] == d for r in rs)]
    fig, axs = plt.subplots(len(ds), 2, figsize=(9.5, 3.0 * len(ds)), squeeze=False)
    for i, d in enumerate(ds):
        for m in ordered([r["model_base"] for r in rs if r["dataset"] == d]):
            for r in [x for x in rs if x["dataset"] == d and x["model_base"] == m]:
                e = [h["epoch"] + 1 for h in r["history"]]
                axs[i, 0].plot(e, [h["loss"] for h in r["history"]], color=COL.get(m, "#888"), alpha=.55, lw=1.2)
                axs[i, 1].plot(e, [h["val_pr_auc"] for h in r["history"]], color=COL.get(m, "#888"), alpha=.55, lw=1.2)
        axs[i, 0].set_yscale("log"); axs[i, 0].set_ylabel("training loss (log)"); axs[i, 1].set_ylabel("validation PR-AUC")
        axs[i, 0].set_title(f"{dn(d)}: training loss"); axs[i, 1].set_title(f"{dn(d)}: validation PR-AUC")
        for k in range(2):
            axs[i, k].set_xlabel("epoch"); panel_label(axs[i, k], 2 * i + k)
    axs[0, 0].legend(handles=[Line2D([], [], color=COL.get(m, "#888"), label=lab(m)) for m in ordered({r["model_base"] for r in rs})], fontsize=7)
    save(fig, out, "fig11_training_curves")


def fig_ablation(df, R, out, datasets):
    var = [m for m in ["tegs_nomem", "tegs_nomem_adv", "tegs", "tegs_adv"] if m in set(df.model)]
    if len(var) < 2:
        return print("  skipped ablation (need >= 2 TE-GS variants)")
    fig, axs = plt.subplots(len(datasets), 3, figsize=(11.5, 3.0 * len(datasets)), squeeze=False)
    for i, d in enumerate(datasets):
        for j, (c, ttl) in enumerate([("pr_auc", "PR-AUC"), ("tpr_at_1pct_fpr", "TPR @ 1% FPR")]):
            ax = axs[i, j]; g = df[df.dataset == d]
            a = g.groupby("model")[c].agg(["mean", "std"]).reindex(var).fillna(0)
            ax.bar(range(len(var)), a["mean"], yerr=a["std"], color=[COL[m] for m in var], capsize=3)
            ax.set_xticks(range(len(var))); ax.set_xticklabels([lab(m).replace(" (", "\n(").replace(" + ", "\n+ ") for m in var], fontsize=6.8)
            lo = max(0, float(a["mean"].min()) - .08); ax.set_ylim(lo, 1.005); ax.set_title(f"{dn(d)}: {ttl}"); panel_label(ax, 3 * i + j)
        ax = axs[i, 2]; vals = []
        for m in var:
            v = [x.get("detection_rate_op", x["detection_rate"]) for r in R if r["dataset"] == d and r["model_base"] == m and not r.get("tag") and not r.get("holdout")
                 for x in (r.get("robust") or []) if x["family"] == "inject_dst" and x["k"] == 10]
            vals.append((np.mean(v) if v else np.nan, np.std(v) if v else 0))
        ax.bar(range(len(var)), [v[0] for v in vals], yerr=[v[1] for v in vals], color=[COL[m] for m in var], capsize=3)
        ax.set_xticks(range(len(var))); ax.set_xticklabels([lab(m).replace(" (", "\n(").replace(" + ", "\n+ ") for m in var], fontsize=6.8)
        ax.set_ylim(0, 1.05); ax.set_title(f"{dn(d)}: detection under\nheld-out injection (k=10)"); panel_label(ax, 3 * i + 2)
    fig.suptitle("Ablation: what do memory and adversarial training buy?", fontweight="bold")
    save(fig, out, "fig12_ablation")


def fig_confusion(results_dir, R, out, datasets, models=("xgb", "tegs")):
    items = [(d, m, _first_scores(results_dir, R, d, m)) for d in datasets for m in models]
    items = [x for x in items if x[2] is not None]
    if not items:
        return print("  skipped confusion matrices")
    fig, axs = plt.subplots(len(datasets), len(models), figsize=(3.6 * len(models), 3.3 * len(datasets)), squeeze=False)
    for d, m, sc in items:
        ax = axs[datasets.index(d), list(models).index(m)]
        y, p = sc["y"], (sc["s"].astype(float) >= float(sc["thr"])).astype(int)
        cm = np.array([[np.sum((y == a) & (p == b)) for b in (0, 1)] for a in (0, 1)], float)
        nrm = cm / cm.sum(1, keepdims=True).clip(1)
        ax.imshow(nrm, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        for a in range(2):
            for b in range(2):
                ax.text(b, a, f"{nrm[a, b]:.3f}\n({int(cm[a, b]):,})", ha="center", va="center", fontsize=8, color="white" if nrm[a, b] > .5 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred. benign", "pred. attack"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["benign", "attack"])
        ax.set_title(f"{lab(m)}\n{dn(d)}", fontsize=9)
    save(fig, out, "fig13_confusion_matrices")


def fig_lofo(R, out, datasets):
    rs = [r for r in R if r.get("holdout") and r.get("per_family")]
    if not rs:
        return print("  skipped leave-one-family-out (no --holdout runs)")
    ds = [d for d in datasets if any(r["dataset"] == d for r in rs)]
    fig, axs = plt.subplots(1, len(ds), figsize=(6.2 * len(ds), 3.8), squeeze=False)
    for i, d in enumerate(ds):
        g = [r for r in rs if r["dataset"] == d]
        fams = sorted({r["holdout"] for r in g}); models = ordered({r["model_base"] for r in g}); w = .8 / len(models)
        for k, m in enumerate(models):
            v = [np.mean([r["per_family"].get(f, {"recall": np.nan})["recall"] for r in g if r["model_base"] == m and r["holdout"] == f] or [np.nan]) for f in fams]
            axs[0, i].bar(np.arange(len(fams)) + k * w - .4 + w / 2, v, w, color=COL.get(m, "#888"), label=lab(m))
        axs[0, i].set_xticks(range(len(fams))); axs[0, i].set_xticklabels(fams, rotation=25, ha="right")
        axs[0, i].set_ylim(0, 1.05); axs[0, i].set_ylabel("recall on the withheld family (zero-day)")
        axs[0, i].set_title(f"{dn(d)}: leave-one-attack-family-out"); panel_label(axs[0, i], i)
    axs[0, 0].legend(fontsize=6.5, ncol=2)
    save(fig, out, "fig14_leave_one_family_out")


def fig_operating_point(df, out, datasets):
    d0 = df.dropna(subset=["op_recall"])
    if d0.empty:
        return print("  skipped operating point (rerun with v0.3 to get test_op)")
    fig, axs = plt.subplots(len(datasets), 3, figsize=(12.5, 0.3 * len(d0.model.unique()) * len(datasets) + 2.2 * len(datasets)), squeeze=False, sharey="row")
    for i, d in enumerate(datasets):
        g = d0[d0.dataset == d]
        models = ordered(g.model.unique()); y = np.arange(len(models))[::-1]
        M = g.groupby("model")[["op_recall", "op_fpr", "op_acc"]].agg(["mean", "std"]).fillna(0)
        for j, (c, ttl) in enumerate([("op_recall", "Attack recall (TPR)"), ("op_fpr", "False-positive rate (budget 1%)"), ("op_acc", "Accuracy")]):
            ax = axs[i, j]
            for yy, m in zip(y, models):
                ax.barh(yy, M.loc[m, (c, "mean")], xerr=M.loc[m, (c, "std")], color=COL.get(m, "#888"), height=.68, error_kw=dict(lw=1, capsize=2),
                        hatch="////" if m == "iponly" else None, edgecolor="white" if m == "iponly" else "none")
            if c == "op_fpr":
                ax.axvline(.01, color="black", ls="--", lw=1); ax.set_xscale("symlog", linthresh=1e-3)
            else:
                ax.set_xlim(0, 1.0)
            ax.set_yticks(y); ax.set_yticklabels([lab(m) for m in models]); ax.set_xlabel(ttl)
            if i == 0:
                ax.set_title(ttl)
            if j == 0:
                ax.set_ylabel(dn(d), fontweight="bold")
            panel_label(ax, 3 * i + j)
    fig.suptitle("Operating point: threshold = 99th percentile of benign VALIDATION scores (target FPR 1%)", fontweight="bold")
    save(fig, out, "fig15_operating_point")


def fig_shortcuts(results_dir, out, top=12):
    files = sorted(glob.glob(os.path.join(results_dir, "shortcuts", "*.csv")))
    if not files:
        return print("  skipped shortcut features (run: python -m gnnids shortcuts --data data/UNSW)")
    fig, axs = plt.subplots(1, len(files), figsize=(6.2 * len(files), 4.2), squeeze=False)
    for i, f in enumerate(files):
        d = pd.read_csv(f).head(top).iloc[::-1]
        ax = axs[0, i]
        ax.barh(d.feature, d.roc_auc, color=["#D55E00" if v > .95 else "#0072B2" for v in d.roc_auc])
        ax.axvline(.5, color="black", ls=":", lw=1); ax.set_xlim(.45, 1.0)
        for yy, v in enumerate(d.roc_auc):
            ax.text(min(v + .004, .985), yy, f"{v:.3f}", va="center", fontsize=7.5)
        ax.set_xlabel("test ROC-AUC of a depth-3 tree using this ONE feature")
        ax.set_title(f"{dn(os.path.basename(f)[:-4])}: single-feature shortcuts"); panel_label(ax, i)
    save(fig, out, "fig16_shortcut_features")


def fig_sanity(results_dir, out):
    files = sorted(glob.glob(os.path.join(results_dir, "sanity", "*.csv")))
    if not files:
        return print("  skipped sanity checks (run: python -m gnnids sanity --data data/UNSW)")
    fig, axs = plt.subplots(1, len(files), figsize=(6.4 * len(files), 3.6), squeeze=False)
    for i, f in enumerate(files):
        d = pd.read_csv(f).iloc[::-1]
        ax = axs[0, i]
        cols = ["#999999" if "random" in s else "#D55E00" if "shuffled" in s else "#0072B2" for s in d.setting]
        ax.barh(d.setting, d.roc_auc, color=cols, xerr=d["roc_auc_std"].fillna(0).values if "roc_auc_std" in d else None, error_kw=dict(lw=1, capsize=3))
        for yy, v in enumerate(d.roc_auc):
            ax.text(min(v + .01, .985), yy, f"{v:.3f}", va="center", fontsize=8)
        ax.axvline(.5, color="black", ls=":", lw=1); ax.set_xlim(.4, 1.02)
        ax.set_xlabel("test ROC-AUC"); ax.set_title(f"{dn(os.path.basename(f)[:-4])}: is the score real?"); panel_label(ax, i)
    save(fig, out, "fig17_sanity_checks")


def fig_cold(R, out, datasets):
    rs = [r for r in R if r.get("test_cold") and not r.get("tag") and not r.get("holdout")]
    if not rs:
        return print("  skipped warm-vs-cold memory (needs runs from v0.4)")
    ds = [d for d in datasets if any(r["dataset"] == d for r in rs)]
    fig, axs = plt.subplots(len(ds), 2, figsize=(10.5, 3.2 * len(ds)), squeeze=False)
    for i, d in enumerate(ds):
        models = ordered({r["model_base"] for r in rs if r["dataset"] == d})
        for j, (k, ttl) in enumerate([("pr_auc", "PR-AUC"), ("tpr_at_1pct_fpr", "TPR @ 1% FPR")]):
            ax = axs[i, j]; w = .38; x = np.arange(len(models))
            for off, key, hatch in [(-w / 2, "test", None), (w / 2, "test_cold", "////")]:
                v = [np.nanmean([r[key][k] for r in rs if r["dataset"] == d and r["model_base"] == m]) for m in models]
                ax.bar(x + off, v, w, color=[COL.get(m, "#888") for m in models], hatch=hatch, edgecolor="white" if hatch else "none", alpha=1 if not hatch else .75)
            ax.set_xticks(x); ax.set_xticklabels([lab(m).replace(" (", "\n(").replace(" + ", "\n+ ") for m in models], fontsize=6.8)
            ax.set_ylim(0, 1.05); ax.set_ylabel(ttl); ax.set_title(f"{dn(d)}: {ttl}"); panel_label(ax, 2 * i + j)
    fig.legend(handles=[Patch(facecolor="#777", label="warm memory (continues from training)"), Patch(facecolor="#777", hatch="////", edgecolor="white", alpha=.75, label="cold start (all host memories zero)")],
               loc="outside lower center", ncol=2)
    fig.suptitle("Does the graph model rely on host identities memorised in training?", fontweight="bold")
    save(fig, out, "fig18_warm_vs_cold_memory")


# ------------------------------------------------------------------------------------------ tables
def make_tables(df, out):
    os.makedirs(out, exist_ok=True)
    cols = ["accuracy", "balanced_accuracy", "pr_auc", "roc_auc", "f1_macro", "mcc", "tpr_at_1pct_fpr", "fpr_at_95tpr"]
    cols = [c for c in cols if c in df.columns and not df[c].isna().all()]
    M, S, C = ms(df, cols + ["c_" + c for c in ["pr_auc", "tpr_at_1pct_fpr", "f1_macro"]])
    full = pd.concat({"mean": M, "std": S, "n_seeds": C[cols[:1]].set_axis(["n"], axis=1)}, axis=1)
    full.to_csv(os.path.join(out, "summary.csv"))
    for d in sorted(df.dataset.unique()):
        models = ordered(df[df.dataset == d].model)
        best = {c: (M.loc[d][c].min() if c.startswith("fpr") else M.loc[d][c].max()) for c in cols}
        L = [r"\begin{table}[t]", r"\centering", r"\caption{In-dataset results on %s (mean $\pm$ std over seeds, time-ordered split).}" % dn(d).replace("_", r"\_"),
             r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", "Model & " + " & ".join(MLAB[c] for c in cols) + r" \\", r"\midrule"]
        for m in models:
            cells = []
            for c in cols:
                s = f"{M.loc[(d, m), c]:.4f}" + r" {\scriptsize $\pm$ " + f"{S.loc[(d, m), c]:.4f}" + "}"
                if abs(M.loc[(d, m), c] - best[c]) < 1e-12:
                    s = r"\textbf{" + f"{M.loc[(d, m), c]:.4f}" + "}" + r" {\scriptsize $\pm$ " + f"{S.loc[(d, m), c]:.4f}" + "}"
                cells.append(s)
            L.append(lab(m).replace("&", r"\&") + " & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(os.path.join(out, f"table_{d}.tex"), "w").write("\n".join(L))
    print("  tables: summary.csv, table_<dataset>.tex")


def make_report(results_dir, data_dir, out, primary=None):
    set_style()
    R = load_results(results_dir)
    if not R:
        raise SystemExit(f"no results found in {results_dir}")
    df = build_df(R)
    datasets = primary or sorted(df.dataset.unique())
    print(f"{len(R)} runs, datasets {datasets}, writing to {out}")
    steps = [("dataset overview", lambda: fig_dataset_overview(data_dir, out)),
             ("main comparison", lambda: fig_main(df, out, datasets)),
             ("leakage effect", lambda: fig_leakage_effect(df, out, datasets)),
             ("ROC/PR curves", lambda: fig_curves(results_dir, R, out, datasets)),
             ("score histograms", lambda: fig_score_hist(results_dir, R, out, datasets)),
             ("family heatmap", lambda: fig_family_heatmap(R, out, datasets)),
             ("robustness", lambda: fig_robustness(R, out, datasets)),
             ("window sensitivity", lambda: fig_window(R, out, datasets)),
             ("latency", lambda: fig_latency(df, R, out)),
             ("transfer", lambda: fig_transfer(df, out)),
             ("training curves", lambda: fig_training(R, out, datasets)),
             ("ablation", lambda: fig_ablation(df, R, out, datasets)),
             ("confusion", lambda: fig_confusion(results_dir, R, out, datasets)),
             ("leave-one-family-out", lambda: fig_lofo(R, out, datasets)),
             ("operating point", lambda: fig_operating_point(df, out, datasets)),
             ("shortcut features", lambda: fig_shortcuts(results_dir, out)),
             ("sanity checks", lambda: fig_sanity(results_dir, out)),
             ("warm vs cold memory", lambda: fig_cold(R, out, datasets)),
             ("tables", lambda: make_tables(df, out))]
    for name, fn in steps:
        try:
            fn()
        except Exception:
            print(f"  !! {name} failed:\n{traceback.format_exc(limit=3)}")
