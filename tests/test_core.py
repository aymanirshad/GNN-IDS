import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from gnnids.attacks import attack, make_batches
from gnnids.metrics import best_thr, bootstrap_ci, metrics, op_thr, per_family

ROOT = os.path.dirname(os.path.dirname(__file__))


def test_make_batches_partitions_everything():
    rng = np.random.default_rng(0)
    ts = np.sort(rng.integers(0, 600_000, 1000))
    b = make_batches(ts, 60_000, 50)
    assert np.array_equal(np.concatenate(b), np.arange(1000))
    assert max(len(x) for x in b) <= 50
    # no batch spans two windows
    w = (ts - ts[0]) // 60_000
    assert all(len(set(w[x])) == 1 for x in b)


@pytest.mark.parametrize("fam", ["inject_src", "inject_dst", "drop_benign"])
def test_attack_keeps_real_edges_and_flags_fakes(fam):
    rng = np.random.default_rng(1)
    src, dst = rng.integers(0, 20, 300), rng.integers(0, 20, 300)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    y = (rng.random(300) < .2).astype(np.float32)
    s, d, x, yy, real = attack(src, dst, X, y, fam, 5, rng)
    assert len(s) == len(d) == len(x) == len(yy) == len(real)
    if fam != "drop_benign":
        assert real[:300].all() and not real[300:].any() and (yy[300:] == 0).all()
        assert np.array_equal(yy[:300], y)
    else:
        assert (yy == 1).sum() == (y == 1).sum() and len(yy) <= 300


def test_attack_budget_zero_is_identity():
    rng = np.random.default_rng(2)
    src, dst = rng.integers(0, 9, 50), rng.integers(0, 9, 50)
    X = np.zeros((50, 2), np.float32); y = np.zeros(50, np.float32); y[:5] = 1
    s, d, x, yy, real = attack(src, dst, X, y, "inject_src", 0, rng)
    assert len(s) == 50 and real.all()


def test_metrics_perfect_random_and_degenerate():
    y = np.array([0] * 90 + [1] * 10)
    m = metrics(y, np.r_[np.zeros(90), np.ones(10)], .5)
    assert m["f1_macro"] == 1 and m["pr_auc"] == 1 and m["roc_auc"] == 1 and m["tpr_at_1pct_fpr"] == 1
    rng = np.random.default_rng(0)
    yr = (rng.random(20000) < .1).astype(int)
    mr = metrics(yr, rng.random(20000), .5)
    assert abs(mr["roc_auc"] - .5) < .03 and abs(mr["pr_auc"] - .1) < .03
    d = metrics(np.zeros(10, int), rng.random(10), .5)  # one class only must not crash
    assert np.isnan(d["pr_auc"])


def test_best_thr_uses_given_scores():
    y = np.array([0] * 50 + [1] * 50); s = np.r_[np.full(50, .2), np.full(50, .8)]
    assert .2 < best_thr(y, s) <= .8


def test_per_family_and_bootstrap():
    fam = np.array([0, 1, 1, 2, 2, 2]); s = np.array([.1, .9, .2, .8, .9, .7])
    pf = per_family(fam, s, .5, ["Benign", "A", "B"])
    assert pf["A"]["recall"] == .5 and pf["B"]["recall"] == 1.0 and "Benign" not in pf
    rng = np.random.default_rng(0); y = (rng.random(500) < .3).astype(int); sc = y * .6 + rng.random(500) * .5
    from sklearn.metrics import roc_auc_score
    p, lo, hi = bootstrap_ci(y, sc, roc_auc_score, n_boot=100)
    assert lo <= p <= hi


def make_csv(path, n=6000):
    rng = np.random.default_rng(0)
    hosts = [f"10.0.0.{i}" for i in range(50)]
    ts = np.sort(rng.integers(1_600_000_000_000, 1_600_000_000_000 + 4 * 3600 * 1000, n))
    atk = rng.random(n) < .1
    df = pd.DataFrame({"IPV4_SRC_ADDR": rng.choice(hosts, n), "L4_SRC_PORT": 1, "IPV4_DST_ADDR": rng.choice(hosts, n), "L4_DST_PORT": 80,
                       "F0": rng.gamma(2, 100, n) * np.where(atk, 3, 1), "F1": rng.gamma(2, 5000, n), "F2": rng.integers(0, 3, n),
                       "FLOW_START_MILLISECONDS": ts, "FLOW_END_MILLISECONDS": ts + 10, "Label": atk.astype(int),
                       "Attack": np.where(atk, rng.choice(["DoS", "Scan"], n), "Benign")})
    df.loc[10:200, ["F0", "F1", "F2"]] = df.loc[0, ["F0", "F1", "F2"]].values   # inject duplicates to exercise the audit
    df.to_csv(path, index=False)


def test_prep_is_chronological_purged_and_train_only_scaled(tmp_path):
    from gnnids.data import load, prep
    make_csv(tmp_path / "x.csv")
    a = prep(str(tmp_path / "x.csv"), str(tmp_path / "d" / "X"), max_rows=100000, purge_ms=60_000)
    D = load(str(tmp_path / "d" / "X"))
    ts, sp = D["ts"], D["split"]
    assert np.all(np.diff(ts) >= 0)                                   # time-ordered
    assert ts[sp == 0].max() < ts[sp == 1].min() < ts[sp == 1].max() < ts[sp == 2].min()
    assert ts[sp == 1].min() - ts[sp == 0].max() >= 0 and ts[sp == 2].min() - ts[sp == 1].max() >= 0
    assert abs(D["X"][sp == 0].mean()) < 1e-3 and abs(D["X"][sp == 0].std(0) - 1).max() < 1e-2   # scaler fitted on train only
    assert D["dup"].dtype == bool and not D["dup"][sp == 0].any()
    assert a["split_sizes"][0] > a["split_sizes"][1]
    assert set(D["names"]) == {"Benign", "DoS", "Scan"}


def test_report_generates_all_figures(tmp_path):
    sys.path.insert(0, os.path.dirname(__file__))
    from synth_results import make
    from gnnids.plots import make_report
    res, data = make(str(tmp_path))
    out = str(tmp_path / "report")
    make_report(res, data, out)
    expected = ["fig01_dataset_overview", "fig02_main_comparison", "fig03_leakage_effect", "fig04_roc_pr_curves", "fig05_score_distributions",
                "fig06_family_recall_heatmap", "fig07_structural_robustness", "fig08_window_sensitivity", "fig09_latency_throughput",
                "fig10_cross_dataset_transfer", "fig11_training_curves", "fig12_ablation", "fig13_confusion_matrices", "fig14_leave_one_family_out", "fig15_operating_point", "fig16_shortcut_features", "fig17_sanity_checks", "fig18_warm_vs_cold_memory"]
    for e in expected:
        assert os.path.exists(os.path.join(out, e + ".png")), e
        assert os.path.exists(os.path.join(out, e + ".pdf")), e
    assert os.path.exists(os.path.join(out, "summary.csv")) and os.path.exists(os.path.join(out, "table_UNSW.tex"))


def test_op_thr_hits_fpr_budget_and_accuracy_present():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < .05).astype(int)
    s = np.where(y == 1, rng.beta(5, 2, 20000), rng.beta(2, 5, 20000))
    thr = op_thr(y, s, .01)
    m = metrics(y, s, thr)
    assert abs(m["fpr"] - .01) < .003
    assert 0 <= m["accuracy"] <= 1 and 0 <= m["balanced_accuracy"] <= 1


def test_blocked_split_every_split_spans_time_and_is_purged():
    from gnnids.data import assign_split
    ts = np.arange(0, 100_000, 5) * 1000          # 20000 rows
    sp = assign_split(ts, "blocked", purge_ms=60_000, n_blocks=20)
    for k in range(3):
        t = ts[sp == k]
        assert t.max() - t.min() > .6 * (ts.max() - ts.min())     # each split covers most of the period
    # no train row within purge_ms of a val/test row
    for k in (1, 2):
        a, b = ts[sp == 0], ts[sp == k]
        assert np.min(np.abs(a[:, None][::50] - b[None, :][:, ::50])) >= 60_000 - 1
    assert (sp == -1).sum() > 0 and abs((sp == 0).mean() - .6) < .06


def test_resplit_and_shortcuts_roundtrip(tmp_path):
    from gnnids.data import load, prep, resplit
    from gnnids.shortcuts import run_shortcuts
    make_csv(tmp_path / "x.csv")
    prep(str(tmp_path / "x.csv"), str(tmp_path / "d" / "X"), max_rows=100000, purge_ms=60_000)
    a = resplit(str(tmp_path / "d" / "X"), str(tmp_path / "d" / "Xb"), "blocked", 60_000, 10)
    D = load(str(tmp_path / "d" / "Xb"))
    assert a["split_mode"] == "blocked" and set(np.unique(D["split"])) == {0, 1, 2}
    assert abs(D["X"][D["split"] == 0].mean()) < 1e-3          # scaler re-fitted on the NEW train split
    df = run_shortcuts(str(tmp_path / "d" / "Xb"), str(tmp_path / "res"))
    assert df.iloc[0].feature in ("F0", "F1", "F2") and df.iloc[0].roc_auc > .7   # F0 is the planted signal


def test_sanity_label_shuffle_is_chance_and_real_signal_is_found(tmp_path):
    from gnnids.data import prep
    from gnnids.sanity import run_sanity
    make_csv(tmp_path / "x.csv", n=9000)
    prep(str(tmp_path / "x.csv"), str(tmp_path / "d" / "X"), max_rows=100000, purge_ms=60_000)
    df = run_sanity(str(tmp_path / "d" / "X"), str(tmp_path / "res"), drop_ks=(1,), n_shuffle=3)
    r = dict(zip(df.setting, df.roc_auc))
    assert r["all features"] > .8
    assert abs(r["labels shuffled (must be ~0.5)"] - .5) < .1


def test_classchrono_every_family_in_every_split_and_time_ordered_inside_class():
    from gnnids.data import assign_split
    rng = np.random.default_rng(0)
    n = 20000
    ts = np.sort(rng.integers(0, 10**9, n))
    fam = np.zeros(n, int)
    fam[(ts > 2e8) & (ts < 2.5e8)] = 1                 # a family that exists only in one short burst
    sp = assign_split(ts, "classchrono", purge_ms=1000, fam=fam)
    for c in (0, 1):
        for k in range(3):
            assert ((fam == c) & (sp == k)).sum() > 0
        assert ts[(fam == c) & (sp == 0)].max() <= ts[(fam == c) & (sp == 1)].min() <= ts[(fam == c) & (sp == 2)].min()


def test_classchrono_survives_bursty_family_and_small_family_goes_to_train():
    """Regression test for the real-data failure: a family whose rows all lie inside a few seconds (a burst) must
    still reach validation and test even with a large purge_ms."""
    from gnnids.data import assign_split
    ts = np.r_[np.arange(0, 100_000_000, 5000), 50_000_000 + np.arange(400) * 10, 70_000_000 + np.arange(10)]
    order = np.argsort(ts, kind="stable"); ts = ts[order]
    fam = np.r_[np.zeros(20000, int), np.ones(400, int), np.full(10, 2)][order]
    sp = assign_split(ts, "classchrono", purge_ms=300_000, fam=fam)
    for k in range(3):
        assert ((fam == 1) & (sp == k)).sum() > 50            # burst family present in all splits
    assert ((fam == 2) & (sp == 0)).sum() == 10               # <30 rows: train only


def test_best_thr_handles_saturated_scores_and_op_thr_not_degenerate():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < .1).astype(int)
    s = np.where(y == 1, .995 + rng.random(5000) * .004, .98 + rng.random(5000) * .0099)   # everything piled near 1
    thr = best_thr(y, s)
    assert metrics(y, s, thr)["f1_macro"] > .6
    s0 = np.where(y == 1, .9, 0.0)                     # benign scores all exactly 0
    assert metrics(y, s0, op_thr(y, s0))["fpr"] == 0.0


def test_event_batches_and_clip(tmp_path):
    ts = np.arange(1000) * 7
    b = make_batches(ts, 0, 64)
    assert np.array_equal(np.concatenate(b), np.arange(1000)) and max(len(x) for x in b) == 64
    from gnnids.data import load, prep
    make_csv(tmp_path / "x.csv")
    prep(str(tmp_path / "x.csv"), str(tmp_path / "d" / "X"), max_rows=100000, purge_ms=60_000)
    D = load(str(tmp_path / "d" / "X"), clip=3.0)
    assert D["X"].max() <= 3.0 and D["X"].min() >= -3.0


def test_stats_paired_bootstrap(tmp_path):
    from gnnids.results_io import save_scores
    from gnnids.stats import run_stats
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < .2).astype(int)
    good = np.clip(y * .6 + rng.random(4000) * .4, 0, 1); bad = rng.random(4000)
    for m, s in [("tegs", good), ("xgb", bad)]:
        save_scores(str(tmp_path), f"D_{m}_s0", y, s, y, np.ones(4000, bool), .5)
        json.dump(dict(test=dict(pr_auc=float(.9 if m == "tegs" else .2))), open(tmp_path / f"D_{m}_s0.json", "w"))
    run_stats(str(tmp_path), str(tmp_path / "out"))
    df = pd.read_csv(tmp_path / "out" / "stats.csv")
    assert len(df) == 1 and bool(df.ci_excludes_0.iloc[0]) and df.d_pr_auc_seed0.iloc[0] > 0


def test_paper_figures_generate(tmp_path):
    sys.path.insert(0, os.path.dirname(__file__))
    from synth_results import make
    from gnnids.paper_figs import make_paper
    res, data = make(str(tmp_path))
    out = str(tmp_path / "paper")
    make_paper(res, data, out)
    for name in ["P1_saturation_shortcuts", "P2_main_comparison", "P7_warm_vs_cold_memory", "P9_latency_throughput", "P10_dataset_overview", "P12_ablation_memory_degree_features"]:
        assert os.path.exists(os.path.join(out, name + ".png")), name
        assert os.path.exists(os.path.join(out, name + ".pdf")), name


def test_summarize_table_and_paired_bootstrap(tmp_path):
    from gnnids.results_io import save_scores
    from gnnids.summarize import run_summarize
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < .2).astype(int)
    scores = {"tegs_nomem_nf": np.clip(y * .7 + rng.random(3000) * .3, 0, 1), "xgb": rng.random(3000)}
    for m, s in scores.items():
        for seed in (0, 1):
            save_scores(str(tmp_path), f"D_{m}_s{seed}", y, s, y, np.ones(3000, bool), .5)
            json.dump(dict(dataset="D", model=m, model_base=m, tag="", seed=seed, holdout=None,
                           test=dict(pr_auc=float(.99 if m != "xgb" else .3), tpr_at_1pct_fpr=.9, f1_macro=.9)),
                      open(tmp_path / f"D_{m}_s{seed}.json", "w"))
    g = run_summarize(str(tmp_path), "D", "tegs_nomem_nf", str(tmp_path / "out"), data_dir=str(tmp_path), n_boot=30)
    assert list(g.index)[0] == "tegs_nomem_nf" and int(g.loc["xgb", "n_seeds"]) == 2
    b = pd.read_csv(tmp_path / "out" / "D_paired_bootstrap.csv")
    assert len(b) == 1 and b.delta_pr_auc.iloc[0] > .3 and b.ci_low.iloc[0] > 0
