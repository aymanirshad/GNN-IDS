"""Training / evaluation of TE-GS and its ablations, robustness sweep, latency sweep and transfer."""
import copy
import os
import time

import numpy as np

from .attacks import attack, make_batches
from .data import load
from .metrics import best_thr, metrics, op_thr, per_family
from .results_io import result_exists, run_name, save_result, save_scores


def run_tegs(a):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score
    from .model import TEGS

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    D = load(a.data)
    ds = os.path.basename(a.data)
    names = D["names"]
    X, y, S, T = D["X"], D["y"], D["src"], D["dst"]
    N = int(max(S.max(), T.max()) + 1)
    idx = [np.flatnonzero(D["split"] == k) for k in range(3)]
    if a.holdout and a.holdout not in names:
        raise SystemExit(f"--holdout {a.holdout!r} not in families {names}")
    hf = names.index(a.holdout) if a.holdout else None
    if hf is not None:  # leave-one-attack-family-out
        idx[0], idx[1] = [i[D["fam"][i] != hf] for i in idx[:2]]
    delta = a.delta * 1000
    B = [[ix[b] for b in make_batches(D["ts"][ix], delta, a.cap)] for ix in idx]
    alpha = float(1 - y[idx[0]].mean())
    print(f"{len(B[0])} training batches per epoch, {len(B[1])} validation, {len(B[2])} test (cap={a.cap}); few batches = few optimizer steps = under-training")
    base = "tegs" + ("_nomem" if a.nomem else "") + ("_nf" if a.node_feats else "") + ("_adv" if a.adv else "")

    def focal(lg, yy):  # focal cross-entropy (gamma=2), class-balanced alpha
        bce = F.binary_cross_entropy_with_logits(lg, yy, reduction="none")
        w = torch.where(yy == 1, alpha, 1 - alpha)
        return (w * (1 - torch.exp(-bce)) ** 2 * bce).mean()

    def tens(device, *arrs):
        return [torch.from_numpy(np.ascontiguousarray(v)).to(device) for v in arrs]

    def run(model, Dd, batches, mem, device, fam=None, k=0, rng=None, timing=None):
        sc, yy = [], []
        model.eval()
        with torch.no_grad():
            for b in batches:
                s_, d_, x_, y_ = Dd["src"][b], Dd["dst"][b], Dd["X"][b], Dd["y"][b].astype(np.float32)
                real = np.ones(len(y_), bool)
                if fam:
                    s_, d_, x_, y_, real = attack(s_, d_, x_, y_, fam, k, rng)
                st, dt, xt = tens(device, s_, d_, x_)
                t0 = time.perf_counter()
                lg, ids, u = model(xt, st, dt, mem)
                if device == "cuda":
                    torch.cuda.synchronize()
                if timing is not None:
                    timing.append((time.perf_counter() - t0, len(s_)))
                if model.use_mem:
                    mem[ids] = u
                sc.append(torch.sigmoid(lg).cpu().numpy()[real])
                yy.append(y_[real])
        return np.concatenate(sc), np.concatenate(yy).astype(int)

    def latency(model, device):
        """Per-micro-batch latency for several micro-batch sizes (the streaming experiment, RQ3)."""
        m2 = copy.deepcopy(model).to(device)
        out = []
        sub = idx[2][:a.lat_events]
        for cap in a.latency_caps:
            bs = [sub[b] for b in make_batches(D["ts"][sub], delta, cap)]
            tm = []
            run(m2, D, bs, torch.zeros(N, a.hidden, device=device), device, timing=tm)
            t = np.array([x for x, _ in tm])
            out.append(dict(device=device, cap=cap, p50_ms=float(np.percentile(t, 50) * 1e3),
                            p95_ms=float(np.percentile(t, 95) * 1e3), p99_ms=float(np.percentile(t, 99) * 1e3),
                            events_per_s=float(sum(n for _, n in tm) / t.sum())))
        return out

    for seed in a.seeds:
        if not a.force and result_exists(a.results, run_name(ds, base + a.tag, "", a.holdout, seed)):
            print("skip (done):", run_name(ds, base + a.tag, "", a.holdout, seed)); continue
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = TEGS(X.shape[1], a.hidden, a.layers, use_mem=not a.nomem, node_feats=a.node_feats, no_edge=getattr(a, 'no_edge', False)).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
        best, bad, hist, t_train = (-1, None, None), 0, [], time.time()
        for ep in range(a.epochs):
            model.train()
            mem = torch.zeros(N, a.hidden, device=dev)
            tot = torch.zeros((), device=dev)
            # static models keep no state between windows, so batch ORDER is free: shuffle it. (Contiguous time slices are often one
            # single class, e.g. a whole DDoS burst; a long run of single-class batches stops static models from learning.)
            # Memory models need chronological order and are not shuffled.
            order = rng.permutation(len(B[0])) if (a.nomem and not a.no_shuffle) else range(len(B[0]))
            for bi in order:
                b = B[0][bi]
                s_, d_, x_, y_ = S[b], T[b], X[b], y[b].astype(np.float32)
                st, dt, xt, yt = tens(dev, s_, d_, x_, y_)
                lg, ids, u = model(xt, st, dt, mem)
                loss = focal(lg, yt)
                if a.adv and rng.random() < a.adv_prob:   # structural adversarial sample (training family only)
                    s2, d2, x2, y2, r2 = attack(s_, d_, x_, y_, "inject_src", int(rng.choice(a.adv_ks)), rng)
                    s2, d2, x2, y2, r2 = tens(dev, s2, d2, x2, y2, r2)
                    la, ids2, u2 = model(x2, s2, d2, mem)
                    if a.adv_mem:  # memory absorbs the injected edges, as it will at test time under attack
                        ids, u = ids2, u2
                    # loss on REAL edges only: fake benign edges are numerous and would drown the signal
                    loss = loss + a.lam * (focal(la[r2], y2[r2]) if a.adv_real_only else focal(la, y2))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += loss.detach()
                if model.use_mem:
                    with torch.no_grad():
                        mem[ids] = u.detach()
            sv, yv = run(model, D, B[1], mem.clone(), dev)
            sc_clean = average_precision_score(yv, sv)
            sc, sc_adv = sc_clean, None
            if a.adv:  # checkpoint selection also looks at an attacked validation stream
                sa, ya = run(model, D, B[1], mem.clone(), dev, fam="inject_src", k=a.sel_k, rng=np.random.default_rng(123))
                sc_adv = average_precision_score(ya, sa)
                sc = 0.5 * sc_clean + 0.5 * sc_adv
            hist.append(dict(epoch=ep, loss=float(tot.item() / max(len(B[0]), 1)), val_pr_auc=float(sc_clean),
                             val_pr_auc_attacked=None if sc_adv is None else float(sc_adv)))
            print(f"seed {seed} epoch {ep} loss {hist[-1]['loss']:.4f} val PR-AUC {sc_clean:.4f}"
                  + ("" if sc_adv is None else f" (attacked {sc_adv:.4f})"))
            if sc > best[0]:
                best, bad = (sc, copy.deepcopy(model.state_dict()), mem.clone()), 0
            else:
                bad += 1
                if bad >= a.patience:
                    break
        model.load_state_dict(best[1])
        t_train = time.time() - t_train
        mem_v = best[2].clone()
        sv, yv = run(model, D, B[1], mem_v, dev)                 # val continues the train memory
        thr, thr_o = best_thr(yv, sv), op_thr(yv, sv)
        mem_t0 = mem_v.clone()
        timing = []
        st_, yt_ = run(model, D, B[2], mem_t0.clone(), dev, timing=timing)
        ft_ = D["fam"][idx[2]]
        # cold start: same test stream, but ALL host memories start at zero -> the model cannot rely on host identities learned in training
        sc_cold, yc_cold = run(model, D, B[2], torch.zeros(N, a.hidden, device=dev), dev)
        clean = ~D["dup"][idx[2]]
        bt = np.array([t for t, _ in timing])
        res = dict(dataset=ds, model=base + a.tag, model_base=base, tag=a.tag, seed=seed, delta=a.delta,
                   holdout=a.holdout, args=vars(a), thr=thr, thr_op=thr_o, device=dev, history=hist,
                   val=metrics(yv, sv, thr), test=metrics(yt_, st_, thr), test_op=metrics(yt_, st_, thr_o), test_cold=metrics(yc_cold, sc_cold, thr), test_cold_op=metrics(yc_cold, sc_cold, thr_o),
                   test_clean=metrics(yt_[clean], st_[clean], thr) if clean.sum() > 0 else None,
                   per_family=per_family(ft_, st_, thr, names), seconds_train=t_train,
                   latency=dict(p50_ms_per_batch=float(np.percentile(bt, 50) * 1e3),
                                p95_ms_per_batch=float(np.percentile(bt, 95) * 1e3),
                                p99_ms_per_batch=float(np.percentile(bt, 99) * 1e3),
                                mean_us_per_event=float(bt.sum() / sum(n for _, n in timing) * 1e6),   # total time / total events
                                flows_per_sec=float(sum(n for _, n in timing) / bt.sum()), batch_cap=a.cap))
        if a.robust:
            rob = []
            for fam in ["inject_src", "inject_dst", "drop_benign"]:
                for k in ([0] if fam == "drop_benign" else a.budgets):
                    s2, y2 = run(model, D, B[2], mem_t0.clone(), dev, fam=fam, k=k, rng=np.random.default_rng(seed + 1))
                    mm = metrics(y2, s2, thr)
                    rob.append(dict(family=fam, k=k, detection_rate=mm["recall"], pr_auc=mm["pr_auc"], fpr=mm["fpr"],
                                    detection_rate_op=float((s2[y2 == 1] >= thr_o).mean()) if (y2 == 1).any() else float("nan")))
            res["robust"] = rob
        if a.latency_caps:
            res["latency_sweep"] = latency(model, "cpu") + (latency(model, "cuda") if dev == "cuda" else [])
        if a.target:
            Dt = load(a.target, D["mean"], D["std"])
            it = np.flatnonzero(Dt["split"] == 2)
            Bt = [it[b] for b in make_batches(Dt["ts"][it], delta, a.cap)]
            Nt = int(max(Dt["src"].max(), Dt["dst"].max()) + 1)
            s3, y3 = run(model, Dt, Bt, torch.zeros(Nt, a.hidden, device=dev), dev)
            mm = metrics(y3, s3, thr)
            res["transfer"] = dict(target=os.path.basename(a.target), metrics=mm,
                                   transfer_drop_f1=res["test"]["f1_macro"] - mm["f1_macro"])
        nm = run_name(ds, base + a.tag, "", a.holdout, seed)
        save_result(a.results, nm, res)
        if not a.no_scores:
            save_scores(a.results, nm, yt_, st_, ft_, clean, thr, thr_o)
