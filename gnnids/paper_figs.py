"""Paper-ready figures: compact (7.2 in = double column, or 3.5 in = single column), >= 8 pt text at final size,
short model names, one message per figure, dot plots instead of walls of bars. Saved as PNG (300 dpi) and PDF."""
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

from .plots import build_df
from .results_io import load_results, load_scores

NAME = {"xgb": "XGBoost", "lgbm": "LightGBM", "rf": "Random Forest", "topo": "XGB + topology stats", "iponly": "IP-identity only",
        "tegs_nomem": "E-GS static", "tegs_nomem_nf": "E-GS static + deg", "tegs_nomem_adv": "E-GS static + adv",
        "tegs_nomem_nf_adv": "E-GS static + deg + adv", "tegs": "TE-GS", "tegs_nf": "TE-GS + deg",
        "tegs_adv": "TE-GS + adv", "tegs_nf_adv": "TE-GS + deg + adv"}
ORDER = ["iponly", "rf", "lgbm", "xgb", "topo", "tegs_nomem", "tegs_nomem_nf", "tegs_nomem_adv", "tegs_nomem_nf_adv",
         "tegs", "tegs_nf", "tegs_adv", "tegs_nf_adv"]
# tabular = blues/greys, static graph = oranges, temporal graph = reds/purples
C = {"iponly": "#000000", "rf": "#2a9d8f", "lgbm": "#6baed6", "xgb": "#1f4e79", "topo": "#8c8c8c",
     "tegs_nomem": "#f2b134", "tegs_nomem_nf": "#e07b00", "tegs_nomem_adv": "#c9a227", "tegs_nomem_nf_adv": "#9c4f00",
     "tegs": "#e76f51", "tegs_nf": "#c1121f", "tegs_adv": "#9d4edd", "tegs_nf_adv": "#5a189a"}
DS = {"UNSW": "UNSW-NB15", "CIC": "CIC-2018 (chronological, zero-day)", "CICc": "CIC-2018 (class-chronological)", "CICb": "CIC-2018 (blocked)"}
W = 7.2
SHORT = {"UNSW": "UNSW-NB15", "CICc": "CIC-2018 (class-chron.)", "CIC": "CIC-2018 (zero-day)", "CICb": "CIC-2018 (blocked)"}


def ds_short(d):
    return SHORT.get(d, d)


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
        "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .8, "axes.grid": True,
        "grid.alpha": .25, "grid.linewidth": .5, "axes.axisbelow": True, "legend.frameon": False,
        "pdf.fonttype": 42, "figure.constrained_layout.use": True, "lines.linewidth": 1.6, "lines.markersize": 5})


def nm(m):
    return NAME.get(m, m)


def order(ms):
    s = set(ms)
    return [m for m in ORDER if m in s] + sorted(s - set(ORDER))


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print("  paper figure:", name)


def ttl(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left")


def legend_below(fig, handles, ncol=3):
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol)


def mean_std(df, ds, m, col):
    v = df[(df.dataset == ds) & (df.model == m)][col].dropna()
    return (v.mean(), v.std() if len(v) > 1 else 0.0, len(v)) if len(v) else (np.nan, 0.0, 0)


# ----------------------------------------------------------------------------- 1. why the scores are saturated
def p_saturation(results_dir, out, datasets=("UNSW", "CICc")):
    ds = [d for d in datasets if os.path.exists(os.path.join(results_dir, "shortcuts", d + ".csv"))]
    if not ds:
        return print("  skipped saturation figure (run shortcuts and sanity first)")
    fig, axs = plt.subplots(len(ds), 2, figsize=(W, 2.7 * len(ds)), squeeze=False, gridspec_kw=dict(width_ratios=[1.15, 1]))
    lab = {"all features": "All 47 features", "labels shuffled (must be ~0.5)": "Labels shuffled", "drop top-1 features": "Drop top-1 feature",
           "drop top-3 features": "Drop top-3", "drop top-5 features": "Drop top-5", "drop top-10 features": "Drop top-10",
           "random scorer (floor)": "Random scorer"}
    for i, d in enumerate(ds):
        s = pd.read_csv(os.path.join(results_dir, "shortcuts", d + ".csv")).head(7).iloc[::-1]
        ax = axs[i, 0]
        ax.barh(s.feature.str.replace("_", " ").str.title().str.replace("Ttl", "TTL").str.replace("Pkt", "Pkt").str.replace("Tcp", "TCP").str.replace("Ip ", "IP "),
                s.roc_auc, color=["#c1121f" if v > .99 else "#1f4e79" for v in s.roc_auc], height=.7)
        for y_, v in enumerate(s.roc_auc):
            ax.text(v + .003, y_, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlim(.5, 1.06); ax.axvline(.5, color="k", lw=.8)
        ax.set_xlabel("test ROC-AUC using ONE feature (depth-3 tree)")
        ttl(ax, chr(97 + 2 * i), f"{DS.get(d, d)}: single-feature shortcuts")
        f2 = os.path.join(results_dir, "sanity", d + ".csv")
        ax = axs[i, 1]
        if os.path.exists(f2):
            t = pd.read_csv(f2)
            t = t[~t.setting.str.startswith("random")].iloc[::-1]
            colr = ["#e07b00" if "shuffled" in x else "#1f4e79" for x in t.setting]
            err = t["roc_auc_std"].fillna(0).values if "roc_auc_std" in t else None
            ax.barh([lab.get(x, x) for x in t.setting], t.roc_auc, color=colr, height=.7, xerr=err, error_kw=dict(lw=1, capsize=2))
            for y_, v in enumerate(t.roc_auc):
                ax.text(1.03, y_, f"{v:.3f}", va="center", fontsize=8.5)
            ax.set_xlim(.3, 1.14); ax.set_xticks([.4, .6, .8, 1.0]); ax.axvline(.5, color="k", ls=":", lw=1)
            ax.set_xlabel("test ROC-AUC (XGBoost)")
        ttl(ax, chr(98 + 2 * i), "Sanity checks")
    save(fig, out, "P1_saturation_shortcuts")


# ----------------------------------------------------------------------------- 2. main comparison
def p_main(df, out, datasets=("UNSW", "CICc", "CIC")):
    ds = [d for d in datasets if d in set(df.dataset)]
    cols = [("pr_auc", "PR-AUC"), ("tpr_at_1pct_fpr", "TPR at 1% FPR")]
    ns = [len(order(df[df.dataset == d].model.unique())) for d in ds]
    fig, axs = plt.subplots(len(ds), 2, figsize=(W, .21 * sum(ns) + .75 * len(ds) + .6), squeeze=False,
                            gridspec_kw=dict(height_ratios=ns), sharex="col")
    for i, d in enumerate(ds):
        ms = order(df[df.dataset == d].model.unique())
        y = np.arange(len(ms))[::-1]
        for j, (c, t) in enumerate(cols):
            ax = axs[i, j]
            for yy, m in zip(y, ms):
                mu, sd, n = mean_std(df, d, m, c)
                ax.errorbar(mu, yy, xerr=sd if n > 1 else None, fmt="o", color=C.get(m, "#555"), ms=6.5, capsize=2, lw=1.2,
                            mfc="white" if m == "iponly" else C.get(m, "#555"), mew=1.5, clip_on=False, zorder=5)
                ax.text(1.04, yy, f"{mu:.3f}", va="center", fontsize=8, color="#222")
            ax.set_yticks(y); ax.set_yticklabels([nm(m) for m in ms] if j == 0 else [])
            ax.set_xlim(0, 1.0); ax.set_ylim(-.7, len(ms) - .3)
            ax.grid(axis="y", alpha=0)
            if i == 0:
                ax.set_title(t, loc="center", pad=24)
            if i == len(ds) - 1:
                ax.set_xlabel(t)
        axs[i, 0].text(0, 1.03, DS.get(d, d), transform=axs[i, 0].transAxes, ha="left", va="bottom", fontsize=9.5, fontweight="bold")
    legend_below(fig, [Line2D([], [], marker="o", color="#1f4e79", ls="", label="tabular baselines"),
                       Line2D([], [], marker="o", color="#e07b00", ls="", label="static E-GraphSAGE (E-GS)"),
                       Line2D([], [], marker="o", color="#c1121f", ls="", label="temporal TE-GS"),
                       Line2D([], [], marker="o", mfc="white", color="k", ls="", label="IP-only probe")], ncol=4)
    save(fig, out, "P2_main_comparison")


# ----------------------------------------------------------------------------- 3. per-family recall
def _family_recall(results_dir, r, names):
    """Recall per family at the 1%-FPR operating threshold (from saved scores); falls back to the F1 threshold."""
    sc = load_scores(results_dir, f"{r['dataset']}_{r['model_base']}_s{r['seed']}")
    if sc is None or "thr_op" not in sc:
        return {f: v for f, v in (r.get("per_family") or {}).items()}
    out = {}
    for c, nme in enumerate(names):
        m = sc["fam"] == c
        if nme != "Benign" and m.any():
            out[nme] = dict(n=int(m.sum()), recall=float((sc["s"][m].astype(float) > float(sc["thr_op"])).mean()))
    return out


def p_family(results_dir, data_dir, R, out, datasets=("CICc", "CIC"), models=("iponly", "xgb", "topo", "tegs_nomem_nf", "tegs", "tegs_nf", "tegs_nf_adv"), min_n=100):
    ds = [d for d in datasets if any(r["dataset"] == d and r.get("per_family") for r in R) and os.path.exists(os.path.join(data_dir, d + ".json"))]
    if not ds:
        return
    fig, axs = plt.subplots(1, len(ds), figsize=(W, 3.0), squeeze=False)
    short = {"DDOS_attack-HOIC": "DDoS-HOIC", "DDoS_attacks-LOIC-HTTP": "DDoS-LOIC", "DoS_attacks-GoldenEye": "GoldenEye",
             "DoS_attacks-Hulk": "Hulk", "FTP-BruteForce": "FTP-BF", "SSH-Bruteforce": "SSH-BF", "Infilteration": "Infiltr.",
             "Brute_Force_-Web": "Web-BF", "Reconnaissance": "Recon.", "Exploits": "Exploits", "Fuzzers": "Fuzzers"}
    for i, d in enumerate(ds):
        names = json.load(open(os.path.join(data_dir, d + ".json")))["audit"]["families"]
        rs = [r for r in R if r["dataset"] == d and not r.get("tag") and not r.get("holdout") and r.get("per_family")]
        rec = {id(r): _family_recall(results_dir, r, names) for r in rs}
        ms = [m for m in models if any(r["model_base"] == m for r in rs)]
        fams = sorted({f for r in rs for f, v in rec[id(r)].items() if v["n"] >= min_n}, key=lambda f: -max(rec[id(r)].get(f, {"n": 0})["n"] for r in rs))
        Z = np.array([[np.nanmean([rec[id(r)][f]["recall"] for r in rs if r["model_base"] == m and f in rec[id(r)]] or [np.nan]) for f in fams] for m in ms])
        ax = axs[0, i]
        im = ax.imshow(Z, cmap="Blues", vmin=0, vmax=1, aspect="auto"); ax.grid(False)
        for a in range(Z.shape[0]):
            for b in range(Z.shape[1]):
                if not np.isnan(Z[a, b]):
                    ax.text(b, a, f"{Z[a, b]:.2f}", ha="center", va="center", fontsize=8, color="white" if Z[a, b] > .6 else "black")
        ax.set_xticks(range(len(fams))); ax.set_xticklabels([short.get(f, f) for f in fams], rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(ms))); ax.set_yticklabels([nm(m) for m in ms])
        ttl(ax, chr(97 + i), f"{DS.get(d, d)}")
    cb = fig.colorbar(im, ax=axs.ravel().tolist(), fraction=.025, pad=.02); cb.set_label("recall at the 1%-FPR threshold")
    save(fig, out, "P3_per_family_recall")


# ----------------------------------------------------------------------------- 4. zero-day (leave-one-family-out)
def p_zeroday(results_dir, data_dir, R, out):
    from sklearn.metrics import roc_auc_score
    rows = []
    for r in R:
        if not r.get("holdout"):
            continue
        name = f"{r['dataset']}_{r['model_base']}{r.get('tag', '')}_lofo-{r['holdout']}_s{r['seed']}"
        sc = load_scores(results_dir, name)
        meta = os.path.join(data_dir, r["dataset"] + ".json")
        if sc is None or not os.path.exists(meta):
            continue
        names = json.load(open(meta))["audit"]["families"]
        h = names.index(r["holdout"])
        m = (sc["fam"] == h) | (sc["fam"] == 0)
        if (sc["fam"] == h).sum() == 0:
            continue
        rows.append(dict(dataset=r["dataset"], model=r["model_base"], family=r["holdout"], seed=r["seed"],
                         auc=roc_auc_score((sc["fam"][m] == h).astype(int), sc["s"][m].astype(float))))
    if not rows:
        return print("  skipped zero-day figure (no leave-one-family-out score files)")
    df = pd.DataFrame(rows)
    ds = sorted(df.dataset.unique())
    fig, axs = plt.subplots(1, len(ds), figsize=(W, 2.9), squeeze=False, sharey=True)
    for i, d in enumerate(ds):
        g = df[df.dataset == d]
        fams = sorted(g.family.unique()); ms = order(g.model.unique())
        for k, m in enumerate(ms):
            v = [g[(g.family == f) & (g.model == m)].auc.mean() for f in fams]
            axs[0, i].scatter(range(len(fams)), v, s=55, color=C.get(m, "#555"), label=nm(m), zorder=3, marker="o" if "tegs" in m else "s")
        axs[0, i].set_xticks(range(len(fams))); axs[0, i].set_xticklabels([f.replace("DDOS_attack-", "").replace("Infilteration", "Infiltr.") for f in fams], rotation=20, ha="right")
        axs[0, i].axhline(.5, color="k", ls=":", lw=1); axs[0, i].set_ylim(.35, 1.03)
        axs[0, i].set_xlim(-.5, len(fams) - .5)
        ttl(axs[0, i], chr(97 + i), DS.get(d, d))
    axs[0, 0].set_ylabel("ROC-AUC: withheld family vs benign")
    legend_below(fig, [Line2D([], [], marker="o" if "tegs" in m else "s", color=C.get(m, "#555"), ls="", label=nm(m)) for m in order(df.model.unique())], ncol=4)
    save(fig, out, "P4_zero_day_leave_one_family_out")


# ----------------------------------------------------------------------------- 5. transfer
def p_transfer(df, out, pairs=(("UNSW", "CICc"), ("CICc", "UNSW"))):
    t = df.dropna(subset=["target"])
    pairs = [p for p in pairs if not t[(t.dataset == p[0]) & (t.target == p[1])].empty]
    if not pairs:
        return print("  skipped transfer figure")
    allm = sorted({m for s_, g_ in pairs for m in t[(t.dataset == s_) & (t.target == g_)].model.unique()})
    ms = order(allm)
    y = np.arange(len(ms))[::-1]
    fig, axs = plt.subplots(1, len(pairs), figsize=(W, .27 * len(ms) + 1.6), squeeze=False, sharey=True)
    for i, (s_, g_) in enumerate(pairs):
        g = t[(t.dataset == s_) & (t.target == g_)]
        for yy, m in zip(y, ms):
            gm = g[g.model == m]
            if gm.empty:
                continue
            a_, b_ = gm.pr_auc.mean(), gm.tr_pr_auc.mean()
            axs[0, i].plot([a_, b_], [yy, yy], color="#bbb", lw=2, zorder=1)
            axs[0, i].scatter([a_], [yy], s=45, color=C.get(m, "#555"), zorder=3, clip_on=False)
            axs[0, i].scatter([b_], [yy], s=55, facecolor="white", edgecolor=C.get(m, "#555"), linewidths=1.6, zorder=3)
            axs[0, i].text(1.05, yy, f"{b_:.2f}", va="center", fontsize=8.5)
        prev = g.tr_prev.mean()
        if not np.isnan(prev):
            axs[0, i].axvline(prev, color="k", ls=":", lw=1)
            axs[0, i].text(prev + .01, len(ms) - .35, "chance", fontsize=8, va="top")
        axs[0, i].set_yticks(y); axs[0, i].set_yticklabels([nm(m) for m in ms])
        axs[0, i].set_xlim(0, 1.0); axs[0, i].set_ylim(-.7, len(ms) - .3); axs[0, i].grid(axis="y", alpha=0)
        axs[0, i].set_xlabel("PR-AUC"); ttl(axs[0, i], chr(97 + i), f"{ds_short(s_)} \u2192 {ds_short(g_)}")
    legend_below(fig, [Line2D([], [], marker="o", color="#555", ls="", label="in-dataset test"),
                       Line2D([], [], marker="o", mfc="white", color="#555", ls="", mew=1.6, label="after transfer (number = transfer PR-AUC)")], ncol=2)
    save(fig, out, "P5_cross_dataset_transfer")


# ----------------------------------------------------------------------------- 6. structural robustness
def p_robust(R, out, models=("tegs_nomem_nf", "tegs_nomem_nf_adv", "tegs_nf", "tegs_nf_adv", "tegs")):
    # models that failed to train (in-dataset PR-AUC < 0.5) are left out: their detection rate says nothing about robustness
    rs = [r for r in R if r.get("robust") and not r.get("tag") and not r.get("holdout") and r["model_base"] in models and r["test"].get("pr_auc", 0) >= .5]
    if not rs:
        return print("  skipped robustness figure")
    ds = [d for d in ["UNSW", "CICc"] if any(r["dataset"] == d for r in rs)]
    fig, axs = plt.subplots(len(ds), 2, figsize=(W, 2.6 * len(ds)), squeeze=False, sharey=True)
    for i, d in enumerate(ds):
        for j, (fam, t) in enumerate([("inject_src", "attacker-side injection (seen)"), ("inject_dst", "victim-side injection (held-out)")]):
            ax = axs[i, j]
            for m in order({r["model_base"] for r in rs if r["dataset"] == d}):
                recs = [(x["k"], x.get("detection_rate_op", x["detection_rate"])) for r in rs if r["dataset"] == d and r["model_base"] == m for x in r["robust"] if x["family"] == fam]
                g = pd.DataFrame(recs, columns=["k", "v"]).groupby("k").v.agg(["mean", "std"]).fillna(0)
                ax.errorbar(range(len(g)), g["mean"], yerr=g["std"], marker="o", color=C.get(m, "#555"), capsize=2, label=nm(m), ls="--" if "adv" not in m and "nf" not in m and "nomem" in m else "-")
            ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index.astype(int)); ax.set_ylim(-.03, 1.05)
            ax.set_xlabel("fake edges per attacker/victim host"); ax.set_ylabel("detection rate at 1% FPR threshold" if j == 0 else "")
            ttl(ax, chr(97 + 2 * i + j), f"{ds_short(d)}: {t}")
    h, l = axs[0, 0].get_legend_handles_labels()
    legend_below(fig, h, ncol=3)
    save(fig, out, "P6_structural_robustness")


# ----------------------------------------------------------------------------- 7. warm vs cold memory
def p_cold(R, out):
    rs = [r for r in R if r.get("test_cold") and not r.get("tag") and not r.get("holdout")]
    if not rs:
        return print("  skipped warm-vs-cold figure")
    ds = [d for d in ["UNSW", "CICc"] if any(r["dataset"] == d for r in rs)]
    fig, axs = plt.subplots(1, len(ds), figsize=(W, 3.0), squeeze=False, sharey=True)
    for i, d in enumerate(ds):
        ms = [m for m in order({r["model_base"] for r in rs if r["dataset"] == d}) if m in ("tegs", "tegs_nf", "tegs_adv", "tegs_nf_adv")]
        y = np.arange(len(ms))[::-1]
        for yy, m in zip(y, ms):
            a = np.mean([r["test"]["pr_auc"] for r in rs if r["dataset"] == d and r["model_base"] == m])
            b = np.mean([r["test_cold"]["pr_auc"] for r in rs if r["dataset"] == d and r["model_base"] == m])
            axs[0, i].plot([a, b], [yy, yy], color="#bbb", lw=2.2, zorder=1)
            axs[0, i].scatter([a], [yy], s=55, color=C.get(m, "#555"), zorder=3, clip_on=False); axs[0, i].scatter([b], [yy], s=60, facecolor="white", edgecolor=C.get(m, "#555"), linewidths=1.8, zorder=3)
            axs[0, i].text(min(a, b) - .02, yy, f"{b:.2f}", va="center", ha="right", fontsize=8)
        axs[0, i].set_yticks(y); axs[0, i].set_yticklabels([nm(m) for m in ms]); axs[0, i].set_xlim(0, 1.05); axs[0, i].grid(axis="y", alpha=0)
        axs[0, i].set_xlabel("test PR-AUC"); ttl(axs[0, i], chr(97 + i), ds_short(d))
    legend_below(fig, [Line2D([], [], marker="o", color="#555", ls="", label="warm memory (continues from training)"),
                       Line2D([], [], marker="o", mfc="white", color="#555", ls="", mew=1.8, label="cold start (all host memories zero)")], ncol=2)
    save(fig, out, "P7_warm_vs_cold_memory")


# ----------------------------------------------------------------------------- 8. window sensitivity (shows the instability)
def p_window(R, out, model="tegs_nf"):
    rs = [r for r in R if r["model_base"] == model and not r.get("holdout") and (r.get("tag", "") == "" or str(r.get("tag")).startswith("_d"))]
    df = pd.DataFrame([dict(dataset=r["dataset"], delta=r["delta"], seed=r["seed"], pr_auc=r["test"].get("pr_auc", np.nan)) for r in rs])
    if df.empty or df.groupby("dataset").delta.nunique().max() < 2:
        return print("  skipped window figure")
    ds = [d for d in ["UNSW", "CICc"] if d in set(df.dataset)]
    fig, axs = plt.subplots(1, len(ds), figsize=(W, 2.8), squeeze=False)
    for i, d in enumerate(ds):
        g = df[df.dataset == d]
        xs = sorted(g.delta.unique())
        for k, x in enumerate(xs):
            v = g[g.delta == x].pr_auc.values
            axs[0, i].scatter(np.full(len(v), k) + np.linspace(-.12, .12, len(v)), v, s=30, color="#c1121f", alpha=.8, zorder=3)
            axs[0, i].plot([k - .22, k + .22], [v.mean()] * 2, color="k", lw=2, zorder=4)
        axs[0, i].set_xticks(range(len(xs))); axs[0, i].set_xticklabels([f"{int(x)} s" for x in xs])
        axs[0, i].set_ylabel("test PR-AUC" if i == 0 else ""); axs[0, i].set_xlabel("snapshot window \u0394")
        axs[0, i].set_ylim(-.02, 1.05) if d != "UNSW" else axs[0, i].set_ylim(.995, 1.0005)
        ttl(axs[0, i], chr(97 + i), f"{ds_short(d)}  (dots = seeds, bar = mean)")
    save(fig, out, "P8_window_sensitivity")


# ----------------------------------------------------------------------------- 9. latency
def p_latency(R, out):
    sw = [(r, x) for r in R for x in (r.get("latency_sweep") or []) if not r.get("tag") and not r.get("holdout")]
    if not sw:
        return print("  skipped latency figure")
    df = pd.DataFrame([dict(dataset=r["dataset"], seed=r["seed"], **x) for r, x in sw])
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.8))
    for k, (c, t) in enumerate([("p95_ms", "p95 latency per micro-batch (ms)"), ("events_per_s", "throughput (events / s)")]):
        for (d, dev), g in df.groupby(["dataset", "device"]):
            a = g.groupby("cap")[c].mean()
            axs[k].plot(a.index, a.values, marker="o" if dev == "cpu" else "s", ls="-" if d == "UNSW" else "--", color="#1f4e79" if dev == "cpu" else "#c1121f",
                        label=f"{dev.upper()}, {ds_short(d)}")
        axs[k].set_xscale("log"); axs[k].set_yscale("log"); axs[k].set_xlabel("micro-batch size (events)"); ttl(axs[k], chr(97 + k), t)
    axs[0].legend(ncol=1, fontsize=8)
    save(fig, out, "P9_latency_throughput")


# ----------------------------------------------------------------------------- 10. dataset overview
def p_overview(data_dir, out, datasets=("UNSW", "CIC", "CICc")):
    A = [(d, json.load(open(os.path.join(data_dir, d + ".json")))["audit"]) for d in datasets if os.path.exists(os.path.join(data_dir, d + ".json"))]
    if not A:
        return
    fig, axs = plt.subplots(len(A), 2, figsize=(W, 2.5 * len(A)), squeeze=False, gridspec_kw=dict(width_ratios=[.7, 1.5]))
    cs = ["#1f4e79", "#6baed6", "#c1121f"]
    short = {"DDOS_attack-HOIC": "DDoS-HOIC", "DDoS_attacks-LOIC-HTTP": "DDoS-LOIC", "DoS_attacks-GoldenEye": "GoldenEye", "DoS_attacks-Hulk": "Hulk",
             "FTP-BruteForce": "FTP-BF", "SSH-Bruteforce": "SSH-BF", "Infilteration": "Infiltr.", "Brute_Force_-Web": "Web-BF", "Brute_Force_-XSS": "XSS-BF", "SQL_Injection": "SQL inj."}
    for i, (d, a) in enumerate(A):
        ax = axs[i, 0]
        rate = np.array(a["attack_rate_by_split"]) * 100
        b = ax.bar(["Train", "Val", "Test"], rate, color=cs, width=.6)
        for bb, r_ in zip(b, rate):
            ax.text(bb.get_x() + bb.get_width() / 2, r_ + .2, f"{r_:.1f}%", ha="center", fontsize=8.5)
        ax.set_ylim(0, max(rate) * 1.25); ax.set_ylabel("attack flows (%)"); ttl(ax, chr(97 + 2 * i), DS.get(d, d))
        ax = axs[i, 1]
        by = a.get("family_counts_by_split")
        if not by:
            continue
        fams = [f for f in by if f != "Benign"]; fams = sorted(fams, key=lambda f: sum(by[f]))
        for k in range(3):
            ax.scatter([max(by[f][k], .7) for f in fams], range(len(fams)), color=cs[k], s=28, zorder=3, label=["train", "val", "test"][k])
        ax.hlines(range(len(fams)), [max(min(by[f]), .7) for f in fams], [max(max(by[f]), .7) for f in fams], color="#ccc", lw=1, zorder=1)
        ax.set_xscale("log"); ax.set_yticks(range(len(fams))); ax.set_yticklabels([short.get(f, f) for f in fams], fontsize=8)
        ax.set_xlabel("attack flows (log; left edge = none)"); ttl(ax, chr(98 + 2 * i), "attack families per split")
        if i == 0:
            ax.legend(ncol=3, loc="lower right", fontsize=8)
    save(fig, out, "P10_dataset_overview")


# ----------------------------------------------------------------------------- 11. training stability
def p_training(R, out, dataset="CICc"):
    rs = [r for r in R if r["dataset"] == dataset and r.get("history") and not r.get("tag") and not r.get("holdout")]
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(W, 3.0))
    for r in rs:
        m = r["model_base"]
        ax.plot([h["epoch"] + 1 for h in r["history"]], [h["val_pr_auc"] for h in r["history"]], color=C.get(m, "#555"), alpha=.85, label=nm(m))
    ax.set_xlabel("epoch"); ax.set_ylabel("validation PR-AUC"); ax.set_ylim(0, 1.02)
    ax.legend(ncol=2, fontsize=8, loc="center right"); ttl(ax, "a", f"{DS.get(dataset, dataset)}: validation PR-AUC during training")
    save(fig, out, "P11_training_stability")


# ----------------------------------------------------------------------------- 12. factorial ablation
def p_ablation(R, out, datasets=("UNSW", "CICc", "CIC")):
    cells = {(0, 0): "tegs_nomem", (0, 1): "tegs_nomem_nf", (1, 0): "tegs", (1, 1): "tegs_nf"}   # (memory, degree features)
    rs = [r for r in R if not r.get("tag") and not r.get("holdout")]
    ds = [d for d in datasets if any(r["dataset"] == d and r["model_base"] in cells.values() for r in rs)]
    if not ds:
        return print("  skipped ablation figure")
    fig, axs = plt.subplots(1, len(ds), figsize=(W, 2.7), squeeze=False)
    for i, d in enumerate(ds):
        Z = np.full((2, 2), np.nan)
        lab = [["", ""], ["", ""]]
        for (a, b), m in cells.items():
            v = [r["test"]["pr_auc"] for r in rs if r["dataset"] == d and r["model_base"] == m]
            if v:
                Z[a, b] = np.mean(v)
                lab[a][b] = f"{np.mean(v):.3f}" + (f"\n\u00b1 {np.std(v, ddof=1):.3f}" if len(v) > 1 else "") + f"\n(n={len(v)})"
        ax = axs[0, i]
        ax.imshow(Z, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        for a in range(2):
            for b in range(2):
                ax.text(b, a, lab[a][b] or "n/a", ha="center", va="center", fontsize=8.5, color="white" if Z[a, b] > .6 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["no degree\nfeatures", "degree\nfeatures"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["static\n(no memory)", "GRU\nmemory"] if i == 0 else ["", ""])
        ttl(ax, chr(97 + i), ds_short(d))
    save(fig, out, "P12_ablation_memory_degree_features")


# ----------------------------------------------------------------------------- 13. precision-recall curves
def p_curves(results_dir, R, out, datasets=("CICc", "CIC"), models=("iponly", "xgb", "topo", "tegs_nomem_nf", "tegs_nf")):
    from sklearn.metrics import precision_recall_curve
    ds = [d for d in datasets if any(r["dataset"] == d for r in R)]
    if not ds:
        return
    fig, axs = plt.subplots(1, len(ds), figsize=(W, 3.0), squeeze=False)
    got = False
    for i, d in enumerate(ds):
        for m in models:
            # models that did not converge (PR-AUC < 0.5) produce degenerate curves and are left out
            cand = sorted(r["seed"] for r in R if r["dataset"] == d and r["model_base"] == m and not r.get("tag") and not r.get("holdout") and r["test"].get("pr_auc", 0) >= .5)
            sc = load_scores(results_dir, f"{d}_{m}_s{cand[0]}") if cand else None
            if sc is None or len(np.unique(sc["y"])) < 2:
                continue
            p, r_, _ = precision_recall_curve(sc["y"], sc["s"].astype(float))
            axs[0, i].plot(r_, p, color=C.get(m, "#555"), label=nm(m), ls="--" if m == "iponly" else "-"); got = True
        axs[0, i].set_xlim(0, 1); axs[0, i].set_ylim(0, 1.02); axs[0, i].set_xlabel("recall"); axs[0, i].set_ylabel("precision" if i == 0 else "")
        ttl(axs[0, i], chr(97 + i), DS.get(d, d))
    if not got:
        return plt.close(fig)
    h, l = axs[0, 0].get_legend_handles_labels()
    legend_below(fig, h, ncol=5)
    save(fig, out, "P13_precision_recall_curves")


def make_paper(results_dir, data_dir, out):
    style()
    R = load_results(results_dir)
    df = build_df(R)
    steps = [("saturation", lambda: p_saturation(results_dir, out)), ("main", lambda: p_main(df, out)), ("family", lambda: p_family(results_dir, data_dir, R, out)),
             ("zero-day", lambda: p_zeroday(results_dir, data_dir, R, out)), ("transfer", lambda: p_transfer(df, out)),
             ("robustness", lambda: p_robust(R, out)), ("cold", lambda: p_cold(R, out)), ("window", lambda: p_window(R, out)),
             ("latency", lambda: p_latency(R, out)), ("ablation", lambda: p_ablation(R, out)), ("curves", lambda: p_curves(results_dir, R, out)), ("overview", lambda: p_overview(data_dir, out)), ("training", lambda: p_training(R, out))]
    for name, fn in steps:
        try:
            fn()
        except Exception:
            print(f"  !! {name} failed:\n{traceback.format_exc(limit=3)}")
