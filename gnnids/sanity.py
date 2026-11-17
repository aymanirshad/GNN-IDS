"""Sanity checks for suspiciously high scores. Three questions:
 1. Is the pipeline able to score high on NOISE?  -> train on label-shuffled data; test AUC must be ~0.5.
 2. Does the score depend on a few columns?        -> drop the k most important features and retrain.
 3. How high is the floor?                          -> prevalence = PR-AUC of a random scorer."""
import os

import numpy as np
import pandas as pd

from .data import load


def run_sanity(data, results="results", max_train=200_000, seed=0, drop_ks=(1, 3, 5, 10), n_shuffle=10):
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, roc_auc_score
    D = load(data)
    ds = os.path.basename(data)
    X, y, sp, feats = D["X"], D["y"], D["split"], D["meta"]["features"]
    rng = np.random.default_rng(seed)
    tr, te = np.flatnonzero(sp == 0), np.flatnonzero(sp == 2)
    if len(tr) > max_train:
        tr = np.sort(rng.choice(tr, max_train, replace=False))

    def fit(cols, ytr, n_est=150):
        m = xgb.XGBClassifier(n_estimators=n_est, max_depth=6, learning_rate=.1, tree_method="hist", n_jobs=-1,
                              random_state=seed, eval_metric="logloss").fit(X[tr][:, cols], ytr)
        p = m.predict_proba(X[te][:, cols])[:, 1]
        return m, float(roc_auc_score(y[te], p)), float(average_precision_score(y[te], p))

    allc = list(range(X.shape[1]))
    m, a, p = fit(allc, y[tr])
    rows = [dict(setting="all features", roc_auc=a, pr_auc=p, n_features=len(allc))]
    imp = np.argsort(-m.feature_importances_)
    # ONE shuffle is a noisy check when test attacks form a few tight clusters (a random model can then land far from 0.5 by luck),
    # so repeat it and report mean +- std: a real leak shows as a mean clearly above 0.5, luck as a large std around 0.5.
    sh = np.array([fit(allc, rng.permutation(y[tr]), n_est=60)[1:] for _ in range(n_shuffle)])
    rows.append(dict(setting="labels shuffled (must be ~0.5)", roc_auc=float(sh[:, 0].mean()), pr_auc=float(sh[:, 1].mean()),
                     roc_auc_std=float(sh[:, 0].std()), n_features=len(allc), dropped=f"mean of {n_shuffle} shuffles"))
    for k in drop_ks:
        if k >= len(allc):
            continue
        keep = [c for c in allc if c not in set(imp[:k])]
        _, a, p = fit(keep, y[tr])
        rows.append(dict(setting=f"drop top-{k} features", roc_auc=a, pr_auc=p, n_features=len(keep),
                         dropped=", ".join(feats[i] for i in imp[:k])))
    rows.append(dict(setting="random scorer (floor)", roc_auc=.5, pr_auc=float(y[te].mean()), n_features=0))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(results, "sanity"), exist_ok=True)
    df.to_csv(os.path.join(results, "sanity", ds + ".csv"), index=False)
    print(df.to_string(index=False))
    return df
