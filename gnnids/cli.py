import argparse
import warnings


def _tegs_args(q):
    q.add_argument("--data", required=True); q.add_argument("--target"); q.add_argument("--results", default="results")
    q.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    q.add_argument("--delta", type=int, default=60); q.add_argument("--cap", type=int, default=20000)
    q.add_argument("--hidden", type=int, default=64); q.add_argument("--layers", type=int, default=2)
    q.add_argument("--lr", type=float, default=2e-3); q.add_argument("--epochs", type=int, default=20)
    q.add_argument("--patience", type=int, default=6); q.add_argument("--device", default=None)
    q.add_argument("--nomem", action="store_true", help="static E-GraphSAGE-style ablation (no GRU memory)")
    q.add_argument("--no_shuffle", action="store_true", help="do not shuffle training batches of static (--nomem) models")
    q.add_argument("--node_feats", action="store_true", help="add in/out-degree of each host in the window as node features")
    q.add_argument("--no_edge", action="store_true", help="ablation: hide flow (edge) features, keep only graph structure and degree features")
    q.add_argument("--adv", action="store_true", help="structural adversarial training")
    q.add_argument("--adv_prob", type=float, default=0.5); q.add_argument("--lam", type=float, default=1.0)
    q.add_argument("--adv_ks", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    q.add_argument("--sel_k", type=int, default=5, help="budget of the attacked validation stream used for checkpoint selection")
    q.add_argument("--adv_real_only", dest="adv_real_only", action="store_true", default=True)
    q.add_argument("--adv_mem", action="store_true", help="adversarial pass also writes its node states into memory (train/test consistency)")
    q.add_argument("--force", action="store_true", help="recompute runs that already have a result file")
    q.add_argument("--adv_all_edges", dest="adv_real_only", action="store_false",
                   help="v0.1 behaviour: adversarial loss over real AND fake edges")


def build_parser():
    p = argparse.ArgumentParser(prog="gnnids", description="GNN-IDS Phase 3 benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("prep", help="chronological split, scaling, leakage audit")
    q.add_argument("--csv", required=True); q.add_argument("--out", required=True)
    q.add_argument("--max_rows", type=int, default=1_000_000); q.add_argument("--blocks", type=int, default=20)
    q.add_argument("--purge_ms", type=int, default=300_000)
    q.add_argument("--chunksize", type=int, default=0); q.add_argument("--keep_per_chunk", type=int, default=50_000)
    q.add_argument("--split", choices=["chrono", "blocked"], default="chrono")

    q = sub.add_parser("resplit", help="re-assign train/val/test of a prepared dataset (no raw CSV needed)")
    q.add_argument("--data", required=True); q.add_argument("--out", required=True)
    q.add_argument("--mode", choices=["chrono", "blocked", "classchrono"], default="classchrono")
    q.add_argument("--purge_ms", type=int, default=300_000); q.add_argument("--n_blocks", type=int, default=20)

    q = sub.add_parser("shortcuts", help="single-feature shortcut diagnostic")
    q.add_argument("--data", required=True); q.add_argument("--results", default="results")

    q = sub.add_parser("sanity", help="label-shuffle and feature-drop checks for suspiciously high scores")
    q.add_argument("--data", required=True); q.add_argument("--results", default="results"); q.add_argument("--n_shuffle", type=int, default=10)

    q = sub.add_parser("baselines", help="XGBoost / LightGBM / RF / topology / IP-only")
    q.add_argument("--data", required=True); q.add_argument("--target"); q.add_argument("--results", default="results")
    q.add_argument("--models", nargs="+", default=["xgb", "lgbm", "rf", "topo", "iponly"])
    q.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    q.add_argument("--trials", type=int, default=6); q.add_argument("--delta", type=int, default=60)
    q.add_argument("--rf_cap", type=int, default=400_000); q.add_argument("--holdout", default=None,
                   help="leave-one-attack-family-out: name of the family withheld from training")
    q.add_argument("--tag", default=""); q.add_argument("--no_scores", action="store_true")
    q.add_argument("--force", action="store_true", help="recompute runs that already have a result file")

    q = sub.add_parser("tegs", help="TE-GS and ablations")
    _tegs_args(q)
    q.add_argument("--robust", action="store_true"); q.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 5, 10, 100])
    q.add_argument("--latency_caps", type=int, nargs="*", default=[], help="micro-batch sizes for the latency sweep, e.g. 1 32 256")
    q.add_argument("--lat_events", type=int, default=5000)
    q.add_argument("--holdout", default=None); q.add_argument("--tag", default=""); q.add_argument("--no_scores", action="store_true")

    q = sub.add_parser("tune", help="equal-budget random search for TE-GS on validation PR-AUC")
    _tegs_args(q); q.set_defaults(results="tuning")
    q.add_argument("--trials", type=int, default=6); q.add_argument("--tune_seed", type=int, default=0)
    q.add_argument("--lat_events", type=int, default=5000); q.add_argument("--budgets", type=int, nargs="+", default=[0])

    q = sub.add_parser("stats", help="paired bootstrap between models on identical test flows")
    q.add_argument("--results", default="results"); q.add_argument("--out", default="report")

    q = sub.add_parser("summarize", help="metrics over seeds, per-family recall and paired bootstrap for one dataset")
    q.add_argument("--results", default="results"); q.add_argument("--dataset", required=True)
    q.add_argument("--graph", default="tegs_nomem_nf", help="model compared against all others in the paired bootstrap")
    q.add_argument("--data_dir", default="data"); q.add_argument("--out", default="report")

    q = sub.add_parser("paper", help="compact, paper-ready figures (PNG + PDF)")
    q.add_argument("--results", default="results"); q.add_argument("--data_dir", default="data"); q.add_argument("--out", default="report/paper")

    q = sub.add_parser("report", help="tables (CSV + LaTeX) and publication figures")
    q.add_argument("--results", default="results"); q.add_argument("--data_dir", default="data")
    q.add_argument("--out", default="report")
    q.add_argument("--primary", nargs="*", default=None, help="datasets to feature in the main figures (default: all)")
    return p


def main(argv=None):
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    a = build_parser().parse_args(argv)
    if a.cmd == "prep":
        from .data import prep
        prep(a.csv, a.out, a.max_rows, a.blocks, a.purge_ms, a.chunksize, a.keep_per_chunk, a.split)
    elif a.cmd == "resplit":
        from .data import resplit
        resplit(a.data, a.out, a.mode, a.purge_ms, a.n_blocks)
    elif a.cmd == "sanity":
        from .sanity import run_sanity
        run_sanity(a.data, a.results, n_shuffle=a.n_shuffle)
    elif a.cmd == "shortcuts":
        from .shortcuts import run_shortcuts
        run_shortcuts(a.data, a.results)
    elif a.cmd == "baselines":
        from .baselines import run_baselines
        run_baselines(a)
    elif a.cmd == "summarize":
        from .summarize import run_summarize
        run_summarize(a.results, a.dataset, a.graph, a.out, a.data_dir)
    elif a.cmd == "paper":
        from .paper_figs import make_paper
        make_paper(a.results, a.data_dir, a.out)
    elif a.cmd == "stats":
        from .stats import run_stats
        run_stats(a.results, a.out)
    elif a.cmd == "tune":
        from .tune import run_tune
        a.holdout, a.tag, a.no_scores, a.robust, a.latency_caps = None, "", True, False, []
        run_tune(a)
    elif a.cmd == "tegs":
        from .train import run_tegs
        run_tegs(a)
    else:
        from .plots import make_report
        make_report(a.results, a.data_dir, a.out, a.primary)
