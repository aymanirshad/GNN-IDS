"""Compact per-dataset summary: (1) metrics per model over seeds, (2) recall per attack family,
(3) paired bootstrap of one model (typically the graph model) against every other model on identical test flows."""
import json
import os

import numpy as np
import pandas as pd

from .results_io import load_results, load_scores


def _family_recall(results, r, names):
    """Recall per family at the operating threshold (1% false alarms on validation); falls back to the F1 threshold."""
    sc = load_scores(results, f"{r['dataset']}_{r['model_base']}_s{r['seed']}")
    if names is None or sc is None or "thr_op" not in sc:
        return {f: v["recall"] for f, v in (r.get("per_family") or {}).items()}
    return {nme: float((sc["s"][sc["fam"] == c].astype(float) > float(sc["thr_op"])).mean())
            for c, nme in enumerate(names) if nme != "Benign" and (sc["fam"] == c).any()}


def run_summarize(results, dataset, graph, out="report", data_dir="data", n_boot=200, seed=0):
    from sklearn.metrics import average_precision_score
    R = [r for r in load_results(results) if r["dataset"] == dataset and not r.get("tag") and not r.get("holdout")]
    if not R:
        raise SystemExit(f"no results for dataset {dataset!r} in {results}")
    os.makedirs(out, exist_ok=True)
    rows = [dict(model=r["model_base"], seed=r["seed"], pr_auc=r["test"]["pr_auc"], tpr_1pct_fpr=r["test"]["tpr_at_1pct_fpr"],
                 accuracy=r["test"].get("accuracy", np.nan), balanced_accuracy=r["test"].get("balanced_accuracy", np.nan),
                 f1_macro=r["test"]["f1_macro"]) for r in R]
    df = pd.DataFrame(rows)
    g = df.groupby("model").agg(n_seeds=("seed", "count"), pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                                pr_auc_min=("pr_auc", "min"), tpr1_mean=("tpr_1pct_fpr", "mean"), acc_mean=("accuracy", "mean"),
                                bal_acc_mean=("balanced_accuracy", "mean"), f1_mean=("f1_macro", "mean")
                                ).sort_values("pr_auc_mean", ascending=False).round(4)
    print(f"\n{dataset}: metrics over seeds (std = sample standard deviation)")
    print(g.to_string())
    g.to_csv(os.path.join(out, f"{dataset}_seeds.csv"))

    meta = os.path.join(data_dir, dataset + ".json")
    names = json.load(open(meta))["audit"]["families"] if os.path.exists(meta) else None
    fam = [dict(model=r["model_base"], family=f, recall=v) for r in R for f, v in _family_recall(results, r, names).items()]
    if fam:
        fr = pd.DataFrame(fam).groupby(["model", "family"]).recall.mean().unstack().round(2)
        print(f"\n{dataset}: recall per attack family")
        print(fr.to_string())
        fr.to_csv(os.path.join(out, f"{dataset}_family_recall.csv"))

    def sc(model):
        return load_scores(results, f"{dataset}_{model}_s0")
    G = sc(graph)
    if G is None:
        print(f"\nno seed-0 scores for {graph!r}: paired bootstrap skipped")
        return g
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(G["y"]), len(G["y"])) for _ in range(n_boot)]
    res = []
    for m in g.index:
        Z = sc(m)
        if m == graph or Z is None or not np.array_equal(Z["y"], G["y"]):
            continue
        d = [average_precision_score(G["y"][i], G["s"][i].astype(float)) - average_precision_score(G["y"][i], Z["s"][i].astype(float)) for i in idx]
        res.append(dict(model=graph, versus=m, delta_pr_auc=np.mean(d), ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5)))
    b = pd.DataFrame(res).round(4)
    print(f"\n{dataset}: paired bootstrap on identical test flows (seed 0), difference in PR-AUC, {graph} minus other")
    print(b.to_string(index=False))
    b.to_csv(os.path.join(out, f"{dataset}_paired_bootstrap.csv"), index=False)
    return g
