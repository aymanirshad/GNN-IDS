"""Tabular and topology-statistics baselines with an equal tuning budget (random search on VALIDATION data).

Models:  xgb | lgbm | rf          flow features only (the strong non-graph reference, RQ1)
         topo                     flow features + causal in-window host counts (IOTTRUST-style, simplified)
         iponly                   source/destination host id only (shortcut probe: how much does the layout alone explain?)
"""
import os
import time

import numpy as np
import pandas as pd

from .data import causal_stats, load
from .metrics import best_thr, metrics, op_thr, per_family
from .results_io import result_exists, run_name, save_result, save_scores


def make_model(name, p, seed, pos_w):
    if name in ("xgb", "topo"):
        import xgboost as xgb
        return xgb.XGBClassifier(**p, tree_method="hist", random_state=seed, n_jobs=-1,
                                 scale_pos_weight=pos_w, eval_metric="logloss")
    if name in ("lgbm", "iponly"):
        import lightgbm as lgb
        return lgb.LGBMClassifier(**p, random_state=seed, n_jobs=-1, scale_pos_weight=pos_w, verbose=-1)
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**p, random_state=seed, n_jobs=-1, class_weight="balanced_subsample")


def sample_params(name, rng):
    if name in ("xgb", "topo"):
        return dict(n_estimators=int(rng.choice([150, 300, 500])), max_depth=int(rng.integers(4, 11)),
                    learning_rate=float(rng.choice([0.05, 0.1, 0.2])), subsample=float(rng.uniform(.7, 1)),
                    colsample_bytree=float(rng.uniform(.6, 1)))
    if name == "lgbm":
        return dict(n_estimators=int(rng.choice([150, 300, 500])), num_leaves=int(rng.choice([31, 63, 127, 255])),
                    learning_rate=float(rng.choice([0.05, 0.1, 0.2])), subsample=float(rng.uniform(.7, 1)),
                    subsample_freq=1, colsample_bytree=float(rng.uniform(.6, 1)))
    if name == "iponly":
        return dict(n_estimators=200, num_leaves=63, learning_rate=0.1)
    return dict(n_estimators=int(rng.choice([100, 200])), max_depth=[None, 20, 30][int(rng.integers(0, 3))],
                min_samples_leaf=int(rng.integers(1, 4)))


def run_baselines(a):
    from sklearn.metrics import average_precision_score
    D = load(a.data)
    ds = os.path.basename(a.data)
    sp, y, names = D["split"], D["y"], D["names"]
    if a.holdout and a.holdout not in names:
        raise SystemExit(f"--holdout {a.holdout!r} not in families {names}")
    hf = names.index(a.holdout) if a.holdout else None
    extra = causal_stats(D["src"], D["dst"], D["ts"], (a.delta if a.delta > 0 else 60) * 1000) if "topo" in a.models else None   # slow groupby: only when needed
    tgt = load(a.target, D["mean"], D["std"]) if a.target else None
    for name in a.models:
        F = (np.hstack([D["X"], extra]) if name == "topo"
             else np.stack([D["src"], D["dst"]], 1) if name == "iponly" else D["X"])
        asdf = (lambda M: pd.DataFrame(M, columns=["s", "d"]).astype("category")) if name == "iponly" else (lambda M: M)
        for seed in a.seeds:
            if not a.force and result_exists(a.results, run_name(ds, name, a.tag, a.holdout, seed)):
                print("skip (done):", run_name(ds, name, a.tag, a.holdout, seed)); continue
            rng = np.random.default_rng(seed)
            tr, va, te = [np.flatnonzero(sp == k) for k in range(3)]
            if hf is not None:  # leave-one-attack-family-out: the family never appears in train/val
                tr, va = tr[D["fam"][tr] != hf], va[D["fam"][va] != hf]
            if name == "rf" and len(tr) > a.rf_cap:
                tr = np.sort(rng.choice(tr, a.rf_cap, replace=False))
            pos_w = float(min((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), 20))
            best, t0 = None, time.time()
            for _ in range(1 if name == "iponly" else a.trials):
                p = sample_params(name, rng)
                m = make_model(name, p, seed, pos_w)
                m.fit(asdf(F[tr]), y[tr])
                pv = m.predict_proba(asdf(F[va]))[:, 1]
                sc = average_precision_score(y[va], pv)
                if best is None or sc > best[0]:
                    best = (sc, p, m, pv)
            _, p, m, pv = best
            thr, thr_o = best_thr(y[va], pv), op_thr(y[va], pv)
            pt = m.predict_proba(asdf(F[te]))[:, 1]
            clean = ~D["dup"][te]
            n_lat = min(len(te), 20000)
            t1 = time.time()
            m.predict_proba(asdf(F[te][:n_lat]))
            res = dict(dataset=ds, model=name, model_base=name, tag=a.tag, seed=seed, delta=a.delta, holdout=a.holdout,
                       params=p, thr=thr, thr_op=thr_o, val=metrics(y[va], pv, thr), test=metrics(y[te], pt, thr),
                       test_op=metrics(y[te], pt, thr_o),
                       test_clean=metrics(y[te][clean], pt[clean], thr) if clean.sum() > 0 else None,
                       per_family=per_family(D["fam"][te], pt, thr, names),
                       seconds_fit=time.time() - t0, sec_per_flow_infer=(time.time() - t1) / n_lat)
            if tgt is not None and name not in ("topo", "iponly"):
                tt = np.flatnonzero(tgt["split"] == 2)
                mm = metrics(tgt["y"][tt], m.predict_proba(tgt["X"][tt])[:, 1], thr)
                res["transfer"] = dict(target=os.path.basename(a.target), metrics=mm,
                                       transfer_drop_f1=res["test"]["f1_macro"] - mm["f1_macro"])
            nm = run_name(ds, name, a.tag, a.holdout, seed)
            save_result(a.results, nm, res)
            if not a.no_scores:
                save_scores(a.results, nm, y[te], pt, D["fam"][te], clean, thr, thr_o)
