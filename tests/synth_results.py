"""Fabricates a realistic-looking results folder (NOT real results) to test the report code and preview figures."""
import json
import os

import numpy as np

from gnnids.results_io import save_scores

FAMS = {"UNSW": ["Benign", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms"],
        "CIC": ["Benign", "Bot", "Brute_Force_-Web", "DDOS_attack-HOIC", "DoS_attacks-Hulk", "FTP-BruteForce", "Infilteration", "SQL_Injection", "SSH-Bruteforce"]}


def make(root, seeds=(0, 1, 2)):
    rng = np.random.default_rng(0)
    res, data = os.path.join(root, "results"), os.path.join(root, "data")
    os.makedirs(res, exist_ok=True); os.makedirs(data, exist_ok=True)
    quality = {"UNSW": dict(iponly=.999, rf=.97, lgbm=.975, xgb=.978, topo=.98, tegs_nomem=.972, tegs=.974, tegs_adv=.969),
               "CIC": dict(iponly=.35, rf=.55, lgbm=.62, xgb=.6, topo=.63, tegs_nomem=.58, tegs=.64, tegs_adv=.66)}
    for d, fam in FAMS.items():
        nb = 3000
        json.dump(dict(features=[], audit=dict(
            rows_raw=100, rows_after_dedup=90, rows_used=80, split_sizes=[60000, 20000, 20000],
            attack_rate_by_split=[.02, .07, .067] if d == "UNSW" else [.178, .014, .061],
            test_rows_with_identical_features_in_train=.32 if d == "UNSW" else .13,
            val_rows_with_identical_features_in_train=.30 if d == "UNSW" else .12, families=fam,
            family_counts={f: int(10 ** rng.uniform(1.3, 4)) for f in fam},
            family_counts_by_split={f: [int(10 ** rng.uniform(0, 3.5)) * (rng.random() > .2) for _ in range(3)] for f in fam})),
            open(os.path.join(data, d + ".json"), "w"))
        for m, q in quality[d].items():
            for s in seeds:
                qq = float(np.clip(q + rng.normal(0, .004), .05, .9999))
                y = (rng.random(nb) < .07).astype(int)
                sc = np.clip(np.where(y == 1, rng.beta(8 * qq + .5, 1.2, nb), rng.beta(1.2, 8 * qq + .5, nb)), 0, 1)
                clean = rng.random(nb) > .3
                f = np.where(y == 1, rng.integers(1, len(fam), nb), 0)
                thr = .5
                nm = f"{d}_{m}_s{s}"
                save_scores(res, nm, y, sc, f, clean, thr)
                met = lambda k: dict(pr_auc=qq, roc_auc=min(qq + .01, 1), f1_macro=qq - .01, mcc=qq - .02, tpr_at_1pct_fpr=min(qq + .005, 1),
                                     fpr_at_95tpr=max(.001, 1 - qq), recall=qq, fpr=.01, prevalence=.07, n=nb, n_attack=int(y.sum()), accuracy=.95 + .04 * qq, balanced_accuracy=.9 + .09 * qq)
                r = dict(dataset=d, model=m, model_base=m, tag="", seed=s, delta=60, holdout=None, thr=thr, thr_op=.6, test=met(0),
                         test_op=dict(recall=float(np.clip(qq - .05, 0, 1)), fpr=float(.01 * rng.uniform(.5, 2.5)), f1_macro=qq - .02, accuracy=.97),
                         test_clean={k: v - .12 * (1 - qq) * (d == "UNSW" and m == "iponly") - (.08 if d == "CIC" else .03) * (1 - qq) for k, v in met(0).items()},
                         per_family={ff: dict(n=int(10 ** rng.uniform(1.3, 4)), recall=float(np.clip(qq + rng.normal(0, .15), 0, 1))) for ff in fam[1:]})
                if m.startswith("tegs"):
                    r["test_cold"] = {k: (v - .15 * (not m.endswith("nomem")) * (1 - qq) if isinstance(v, float) else v) for k, v in met(0).items()}
                    r["history"] = [dict(epoch=e, loss=float(.05 * np.exp(-e * .7) + rng.random() * .002), val_pr_auc=float(qq - .1 * np.exp(-e)), val_pr_auc_attacked=None) for e in range(8)]
                    r["latency"] = dict(p50_ms_per_batch=2, p95_ms_per_batch=2.5 + rng.random(), p99_ms_per_batch=4, mean_us_per_event=25 + rng.random() * 5, flows_per_sec=4e4, batch_cap=20000)
                    r["latency_sweep"] = [dict(device=dv, cap=c, p50_ms=.6 + (c ** .5) * .05 * (1 if dv == "cpu" else .5), p95_ms=.9 + (c ** .5) * .07 * (1 if dv == "cpu" else .5),
                                               p99_ms=1.5, events_per_s=c / ((.6 + c ** .5 * .05) * 1e-3) / (1 if dv == "cpu" else .6)) for dv in ("cpu", "cuda") for c in (1, 32, 256)]
                    adv = m.endswith("adv")
                    r["robust"] = [dict(family=f_, k=k, detection_rate=float(np.clip((1 if adv else 1) * np.exp(-k / (60 if adv else 25)) * qq + rng.normal(0, .02), 0, 1)), pr_auc=.5, fpr=.01, detection_rate_op=float(np.clip(np.exp(-k / (60 if adv else 25)) * qq + rng.normal(0, .02), 0, 1)))
                                   for f_ in ("inject_src", "inject_dst") for k in (0, 1, 5, 10, 100)] + [dict(family="drop_benign", k=0, detection_rate=float(qq - .05 * (not adv)), pr_auc=.5, fpr=.01)]
                else:
                    r["sec_per_flow_infer"] = {"rf": 4e-5, "lgbm": 2e-6, "xgb": 1.5e-6, "topo": 1.8e-6, "iponly": 1e-6}[m]
                if m in ("xgb", "tegs", "lgbm", "tegs_adv"):
                    r["transfer"] = dict(target="CIC" if d == "UNSW" else "UNSW", metrics=dict(pr_auc=.2 + rng.random() * .1, f1_macro=.45 + rng.random() * .1, prevalence=.06), transfer_drop_f1=.4)
                json.dump(r, open(os.path.join(res, nm + ".json"), "w"))
        for m in ("xgb", "tegs"):
            for fm in fam[1:4]:
                json.dump(dict(dataset=d, model=m, model_base=m, tag="", seed=0, delta=60, holdout=fm, thr=.5, test=dict(f1_macro=.8, pr_auc=.7, prevalence=.07),
                               per_family={fm: dict(n=100, recall=float(rng.uniform(.05, .9)))}), open(os.path.join(res, f"{d}_{m}_lofo-{fm}_s0.json"), "w"))
        for dl in (30, 120, 300):
            for s in seeds:
                json.dump(dict(dataset=d, model="tegs_d%d" % dl, model_base="tegs", tag="_d%d" % dl, seed=s, delta=dl, holdout=None, thr=.5,
                               test=dict(f1_macro=.95 + rng.normal(0, .01), pr_auc=.96 + rng.normal(0, .01), prevalence=.07),
                               latency=dict(p95_ms_per_batch=2 + dl / 60 + rng.random())), open(os.path.join(res, f"{d}_tegs_d{dl}_s{s}.json"), "w"))
    import pandas as pd
    os.makedirs(os.path.join(res, 'shortcuts'), exist_ok=True)
    for d in FAMS:
        pd.DataFrame(dict(feature=[f'FEATURE_{i}' for i in range(20)], roc_auc=sorted(np.r_[[.999, .97, .92] if d == 'UNSW' else [.7, .66, .62], rng.uniform(.5, .6, 17)], reverse=True),
                          pr_auc=0.5)).to_csv(os.path.join(res, 'shortcuts', d + '.csv'), index=False)
    os.makedirs(os.path.join(res, 'sanity'), exist_ok=True)
    for d in FAMS:
        pd.DataFrame(dict(setting=['all features', 'labels shuffled (must be ~0.5)', 'drop top-1 features', 'drop top-5 features', 'random scorer (floor)'],
                          roc_auc=[.999, .5, .998, .99, .5], pr_auc=[.99, .07, .98, .9, .07], n_features=[47, 47, 46, 42, 0])).to_csv(os.path.join(res, 'sanity', d + '.csv'), index=False)
    return res, data
