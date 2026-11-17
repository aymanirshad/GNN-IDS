"""Saving and loading experiment results (one JSON per run + one compressed score file per run)."""
import glob
import json
import os

import numpy as np


def run_name(dataset, model, tag, holdout, seed):
    return f"{dataset}_{model}{tag}{'_lofo-' + holdout if holdout else ''}_s{seed}"


def result_exists(results_dir, name):
    return os.path.exists(os.path.join(results_dir, name + ".json"))


def save_result(results_dir, name, res):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, name + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    t = res.get("test", {})
    print(name, {k: round(t[k], 4) for k in ("f1_macro", "pr_auc", "tpr_at_1pct_fpr") if k in t})


def save_scores(results_dir, name, y, s, fam, clean, thr, thr_op=None):
    d = os.path.join(results_dir, "scores")
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, name + ".npz"), y=np.asarray(y, np.int8), s=np.asarray(s, np.float16),
                        fam=np.asarray(fam, np.int16), clean=np.asarray(clean, bool), thr=float(thr),
                        thr_op=float(thr if thr_op is None else thr_op))


def normalise(r):
    """Bring results written by older versions to the current schema."""
    a = r.get("args") or {}
    if "tag" not in r:
        r["tag"] = a.get("tag", "")
    if "model_base" not in r:
        if r["model"].startswith("tegs") and a:
            r["model_base"] = "tegs" + ("_nomem" if a.get("nomem") else "") + ("_nf" if a.get("node_feats") else "") + ("_adv" if a.get("adv") else "")
        else:
            r["model_base"] = r["model"]
    r.setdefault("delta", a.get("delta", 60))
    r.setdefault("holdout", a.get("holdout"))
    return r


def load_results(results_dir):
    return [normalise(json.load(open(f))) for f in sorted(glob.glob(os.path.join(results_dir, "*.json")))]


def load_scores(results_dir, name):
    p = os.path.join(results_dir, "scores", name + ".npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}
