"""Equal-budget hyper-parameter search for TE-GS (the baselines already get a random search on VALIDATION data).
Each trial trains one model with a random configuration; the best configuration is the one with the highest
VALIDATION PR-AUC. Test data is never used. Run on the BLOCKED dataset (a validation set with the same attack mix)."""
import copy
import json
import os

import numpy as np

from .train import run_tegs


def run_tune(a):
    rng = np.random.default_rng(a.tune_seed)
    base = os.path.basename(a.data)
    best, rows = None, []
    for i in range(a.trials):
        b = copy.copy(a)
        b.hidden = int(rng.choice([32, 64, 128])); b.layers = int(rng.choice([1, 2, 3]))
        b.lr = float(rng.choice([1e-3, 2e-3, 5e-3])); b.lam = float(rng.choice([.5, 1.0, 2.0]))
        b.seeds = [a.tune_seed]; b.tag = f"_tune{i}"; b.robust = False; b.latency_caps = []
        b.target = None; b.no_scores = True; b.holdout = None; b.force = True
        print(f"--- trial {i}: hidden={b.hidden} layers={b.layers} lr={b.lr} lam={b.lam}")
        run_tegs(b)
        f = [x for x in os.listdir(a.results) if x.startswith(f"{base}_") and f"_tune{i}_" in x][0]
        r = json.load(open(os.path.join(a.results, f)))
        sc = r["val"]["pr_auc"]
        rows.append(dict(trial=i, hidden=b.hidden, layers=b.layers, lr=b.lr, lam=b.lam, val_pr_auc=sc))
        if best is None or sc > best["val_pr_auc"]:
            best = rows[-1]
    flags = f"--hidden {best['hidden']} --layers {best['layers']} --lr {best['lr']} --lam {best['lam']}"
    out = os.path.join(a.results, f"best_{base}{'_nomem' if a.nomem else ''}{'_nf' if a.node_feats else ''}{'_adv' if a.adv else ''}.json")
    json.dump(dict(best=best, trials=rows, flags=flags), open(out, "w"), indent=1)
    print("BEST (by validation PR-AUC):", best, "\nuse:", flags)
