"""Pre-processing: split assignment (chronological or blocked) with purge gap, train-only scaling, leakage audit.

Design rules (they are what makes the benchmark honest):
  * rows are ordered by flow start time
  * big files are sub-sampled in CONTIGUOUS time slices, never by random rows
  * scalers are fitted on the training split only
  * IP addresses and ports build the graph but are NEVER model features
  * every val/test row whose feature vector also occurs in the training split is flagged (`dup`) so results
    can be reported on the full test set AND on the leakage-cleaned test set
"""
import json
import os

import numpy as np
import pandas as pd

from .config import ATTACK, DST, LABEL, NOT_FEATURES, SRC, TS


def row_hash(X):
    return pd.util.hash_pandas_object(pd.DataFrame(X), index=False).values


def dup_mask(X, split):
    """True for val/test rows whose exact feature vector appears in the training split."""
    h = row_hash(X)
    return (split != 0) & np.isin(h, h[split == 0])


def causal_stats(src, dst, ts, delta):
    """Counts of EARLIER flows in the same window (never looks ahead): src out-count, dst in-count, pair count."""
    df = pd.DataFrame({"w": (ts - ts.min()) // delta, "s": src, "d": dst})
    o = df.groupby(["w", "s"]).cumcount().values
    i = df.groupby(["w", "d"]).cumcount().values
    p = df.groupby(["w", "s", "d"]).cumcount().values
    return np.log1p(np.stack([o, i, p], 1)).astype(np.float32)


def assign_split(ts, mode="chrono", purge_ms=300_000, n_blocks=20, fam=None):
    """Return split ids (0 train, 1 val, 2 test, -1 purged) for time-sorted timestamps.
    chrono : first 60% train / next 20% val / last 20% test (strict temporal drift: later periods can contain
             attack families never seen before -> a zero-day style test)
    blocked: the stream is cut into `n_blocks` contiguous segments assigned cyclically (3 train : 1 val : 1 test);
             every split sees the whole period; a purge gap is dropped around each role change so that no
             window straddles two splits."""
    n = len(ts)
    if mode == "classchrono":
        # chronological 60/20/20 INSIDE every class (benign and each attack family), by RANK within the class:
        # every family with >= 30 rows appears in train, validation and test, and time order is kept inside the family.
        # A purge of 1% of the class rows (>= 1 row) is dropped at each boundary. (A time-based purge deleted whole
        # validation/test portions of bursty families such as Bot and FTP-BruteForce on the real data.)
        # Families with < 30 rows are too small to evaluate and are put in train only.
        # Caveat for the paper: class-stratified, NOT globally time-ordered; absolute time ranges of the splits overlap.
        sp = np.full(n, -1, np.int8)
        for c in np.unique(fam):
            ix = np.flatnonzero(fam == c)
            m = len(ix)
            if m < 30:
                sp[ix] = 0
                continue
            o = ix[np.argsort(ts[ix], kind="stable")]
            a, b, g = int(.6 * m), int(.8 * m), max(1, int(.01 * m))
            sp[o[:a]] = 0
            sp[o[a + g:b]] = 1
            sp[o[b + g:]] = 2
        return sp
    if mode == "chrono":
        t1, t2 = ts[int(.6 * n)], ts[int(.8 * n)]
        sp = np.full(n, -1)
        sp[ts < t1] = 0
        sp[(ts >= t1 + purge_ms) & (ts < t2)] = 1
        sp[ts >= t2 + purge_ms] = 2
        return sp
    if mode != "blocked":
        raise ValueError(f"unknown split mode {mode!r}")
    seg = np.minimum((np.arange(n) * n_blocks) // n, n_blocks - 1)
    role = np.array([2 if j % 5 == 4 else 1 if j % 5 == 3 else 0 for j in range(n_blocks)])
    sp = role[seg].copy()
    starts = np.array([np.flatnonzero(seg == j)[0] for j in range(n_blocks)])
    for j in range(1, n_blocks):
        if role[j] != role[j - 1]:
            b = ts[starts[j]]
            sp[(ts > b - purge_ms) & (ts < b + purge_ms)] = -1
    return sp


def make_audit(y, fam, split, families, dup, purge_ms, raw=(0, 0), mode="chrono"):
    return dict(rows_raw=raw[0], rows_after_dedup=raw[1], rows_used=int(len(y)), split_mode=mode,
                split_sizes=[int((split == k).sum()) for k in range(3)],
                attack_rate_by_split=[float(y[split == k].mean()) for k in range(3)],
                test_rows_with_identical_features_in_train=float(dup[split == 2].mean()),
                val_rows_with_identical_features_in_train=float(dup[split == 1].mean()),
                purge_ms=purge_ms, families=families,
                family_counts={f: int((fam == i).sum()) for i, f in enumerate(families)},
                family_counts_by_split={f: [int(((fam == i) & (split == k)).sum()) for k in range(3)]
                                        for i, f in enumerate(families)})


def read_csv(path, chunksize=0, keep_per_chunk=50_000):
    if not chunksize:
        return pd.read_csv(path)
    parts, seen = [], 0
    for ch in pd.read_csv(path, chunksize=chunksize):  # keep a contiguous slice of every chunk (fits in RAM)
        seen += len(ch)
        parts.append(ch.iloc[:keep_per_chunk])
    df = pd.concat(parts, ignore_index=True)
    print(f"read {seen} rows, kept {len(df)}")
    return df


def _save(out, D, split, mean, std, meta):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out + ".npz", **D, split=split.astype(np.int8), mean=mean.astype(np.float32), std=std.astype(np.float32))
    json.dump(meta, open(out + ".json", "w"), indent=1)


def _show(audit):
    print(json.dumps({k: v for k, v in audit.items() if k != "family_counts_by_split"}, indent=1))


def prep(csv, out, max_rows=1_000_000, blocks=20, purge_ms=300_000, chunksize=0, keep_per_chunk=50_000,
         split_mode="chrono"):
    df = read_csv(csv, chunksize, keep_per_chunk)
    miss = [c for c in [SRC, DST, TS, LABEL, ATTACK] if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing columns {miss}. Found: {list(df.columns)}. Edit gnnids/config.py")
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    df = df.sort_values(TS, kind="stable").reset_index(drop=True)
    n1 = len(df)
    if len(df) > max_rows:  # evenly spread contiguous blocks keep temporal structure
        bs = max_rows // blocks
        starts = np.linspace(0, len(df) - bs, blocks).astype(int)
        df = pd.concat([df.iloc[s:s + bs] for s in starts]).reset_index(drop=True)
    ts = df[TS].values.astype(np.int64)
    split = assign_split(ts, split_mode, purge_ms)
    keep = split >= 0
    df, ts, split = df[keep].reset_index(drop=True), ts[keep], split[keep]
    feats = [c for c in df.columns if c not in NOT_FEATURES and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].values.astype(np.float64)
    tr = split == 0
    big = X[tr].max(0) > 100                       # heavy-tailed columns -> signed log1p
    X[:, big] = np.sign(X[:, big]) * np.log1p(np.abs(X[:, big]))
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    all_hosts, _ = pd.factorize(pd.concat([df[SRC], df[DST]], ignore_index=True))
    src, dst = all_hosts[:len(df)], all_hosts[len(df):]
    families = ["Benign"] + sorted(set(df[ATTACK]) - {"Benign"})
    fam = df[ATTACK].map({f: i for i, f in enumerate(families)}).values
    y = df[LABEL].values.astype(np.int8)
    dup = dup_mask(X, split)
    audit = make_audit(y, fam, split, families, dup, purge_ms, raw=(n0, n1), mode=split_mode)
    D = dict(X=X.astype(np.float32), y=y, fam=fam.astype(np.int16), src=src, dst=dst, ts=ts)
    _save(out, D, split, mean, std, dict(features=feats, log_mask=big.tolist(), n_hosts=int(all_hosts.max() + 1), audit=audit))
    _show(audit)
    return audit


def resplit(src_path, out_path, mode="blocked", purge_ms=300_000, n_blocks=20):
    """Re-assign train/val/test of an already prepared dataset (no raw CSV needed) and re-fit the scaler."""
    z = np.load(src_path + ".npz")
    meta = json.load(open(src_path + ".json"))
    D = {k: z[k] for k in ["X", "y", "fam", "src", "dst", "ts"]}
    sp = assign_split(D["ts"], mode, purge_ms, n_blocks, fam=D["fam"])
    k = sp >= 0
    D = {n: v[k] for n, v in D.items()}
    sp = sp[k]
    mean, std = D["X"][sp == 0].mean(0), D["X"][sp == 0].std(0) + 1e-6
    old = meta["audit"]
    audit = make_audit(D["y"], D["fam"], sp, old["families"], dup_mask(D["X"], sp), purge_ms,
                       raw=(old.get("rows_raw", 0), old.get("rows_after_dedup", 0)), mode=mode)
    _save(out_path, D, sp, mean, std, dict(meta, audit=audit))
    _show(audit)
    for f, c in audit["family_counts_by_split"].items():
        if f != "Benign" and c[0] > 0 and (c[1] == 0 or c[2] == 0):
            print(f"WARNING: family {f} has train/val/test counts {c}: it cannot be evaluated in every split")
    return audit


def load(path, mean=None, std=None, clip=10.0):
    """Load a prepared dataset. Pass (mean, std) of the SOURCE dataset to standardise a transfer target."""
    z = np.load(path + ".npz")
    meta = json.load(open(path + ".json"))
    m = z["mean"] if mean is None else mean
    s = z["std"] if std is None else std
    D = {k: z[k] for k in ["y", "fam", "src", "dst", "ts", "split"]}
    D["X"] = np.clip((z["X"] - m) / s, -clip, clip).astype(np.float32)   # clip z-scores: heavy-tailed NetFlow counts can blow up a GNN
    D["mean"], D["std"], D["meta"] = z["mean"], z["std"], meta
    D["dup"] = dup_mask(D["X"], D["split"])  # recomputed here so old prepared files keep working
    D["names"] = meta["audit"]["families"]
    return D
