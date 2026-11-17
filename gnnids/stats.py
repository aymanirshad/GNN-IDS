"""Paired significance tests on IDENTICAL test flows (all models score the same test rows in the same order).
Includes the node-feature variants of the graph model."""
import glob
import json
import os

import numpy as np
import pandas as pd

PAIRS = [("tegs_nomem_nf", "xgb"), ("tegs_nomem_nf", "lgbm"), ("tegs_nomem_nf", "topo"), ("tegs_nomem_nf", "iponly"), ("tegs_nf_adv", "xgb"), ("tegs_nf_adv", "topo"), ("tegs_nf_adv", "iponly"), ("tegs_nf_adv", "tegs_nomem"), ("tegs_nomem", "topo"), ("tegs", "xgb"), ("tegs", "lgbm"), ("tegs", "topo"), ("tegs", "tegs_nomem"), ("tegs_nomem", "xgb"),
         ("tegs_adv", "tegs"), ("topo", "xgb"), ("iponly", "xgb")]


def _bootstrap(y, a, b, n=200, seed=0):
    from sklearn.metrics import average_precision_score as ap
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(ap(y[i], a[i]) - ap(y[i], b[i]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def run_stats(results_dir, out):
    rows = []
    for ds in sorted({os.path.basename(f).split("_")[0] for f in glob.glob(os.path.join(results_dir, "scores", "*.npz"))}):
        for a, b in PAIRS:
            pa = os.path.join(results_dir, "scores", f"{ds}_{a}_s0.npz")
            pb = os.path.join(results_dir, "scores", f"{ds}_{b}_s0.npz")
            if not (os.path.exists(pa) and os.path.exists(pb)):
                continue
            za, zb = np.load(pa), np.load(pb)
            if len(za["y"]) != len(zb["y"]) or not np.array_equal(za["y"], zb["y"]):
                continue
            d, lo, hi = _bootstrap(za["y"], za["s"].astype(float), zb["s"].astype(float))
            per_seed = []
            for s in range(10):
                ja, jb = (os.path.join(results_dir, f"{ds}_{m}_s{s}.json") for m in (a, b))
                if os.path.exists(ja) and os.path.exists(jb):
                    per_seed.append(json.load(open(ja))["test"]["pr_auc"] - json.load(open(jb))["test"]["pr_auc"])
            rows.append(dict(dataset=ds, model_a=a, model_b=b, d_pr_auc_seed0=d, ci_lo=lo, ci_hi=hi, ci_excludes_0=bool(lo > 0 or hi < 0),
                             seeds=len(per_seed), mean_d_pr_auc_over_seeds=float(np.mean(per_seed)) if per_seed else np.nan,
                             a_better_in_all_seeds=bool(per_seed and all(x > 0 for x in per_seed))))
    df = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "stats.csv"), index=False)
    print(df.round(4).to_string(index=False) if len(df) else "no comparable pairs found (run seed 0 of both models)")
