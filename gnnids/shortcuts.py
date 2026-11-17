"""Shortcut diagnostic: how well does ONE flow feature (a depth-3 decision tree on that feature alone) separate
attack from benign on the test split? If a single column already gives ROC-AUC ~ 1, the dataset is saturated by an
artefact of the testbed (e.g. a TTL or port pattern of the attack tool) and headline scores say little about the model."""
import os

import numpy as np
import pandas as pd

from .data import load


def run_shortcuts(data, results="results", max_train=300_000, seed=0):
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.tree import DecisionTreeClassifier
    D = load(data)
    ds = os.path.basename(data)
    feats = D["meta"]["features"]
    sp, y, X = D["split"], D["y"], D["X"]
    rng = np.random.default_rng(seed)
    tr = np.flatnonzero(sp == 0)
    if len(tr) > max_train:
        tr = rng.choice(tr, max_train, replace=False)
    te = np.flatnonzero(sp == 2)
    rows = []
    for j, f in enumerate(feats):
        t = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=seed).fit(X[tr][:, [j]], y[tr])
        p = t.predict_proba(X[te][:, [j]])[:, 1]
        rows.append(dict(feature=f, roc_auc=float(roc_auc_score(y[te], p)), pr_auc=float(average_precision_score(y[te], p))))
    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    os.makedirs(os.path.join(results, "shortcuts"), exist_ok=True)
    out = os.path.join(results, "shortcuts", ds + ".csv")
    df.to_csv(out, index=False)
    print(df.head(10).to_string(index=False))
    print("saved", out)
    return df
