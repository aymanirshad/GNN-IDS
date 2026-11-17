"""Structural perturbations of one time window, and window/micro-batch construction (numpy only).

Families (Venturi et al. 2024 / Galli et al. 2025 motivate edge/node injection against GNN-NIDS):
  inject_src  : every attacker host adds k benign-looking flows to random hosts        (TRAINING family)
  inject_dst  : k benign-looking flows from random hosts INTO every victim host        (HELD-OUT family)
  drop_benign : half of the benign flows vanish (e.g. sensor loss / evasion by silence) (HELD-OUT family)
This is a simplified problem-space model: fake edges copy the features of real benign flows of the window.
"""
import numpy as np


def attack(src, dst, X, y, family, k, rng, cap=20000):
    n = len(src)
    if family == "drop_benign":
        keep = (y == 1) | (rng.random(n) > 0.5)
        return src[keep], dst[keep], X[keep], y[keep], np.ones(int(keep.sum()), bool)
    ben, att = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if k == 0 or len(ben) == 0 or len(att) == 0:
        return src, dst, X, y, np.ones(n, bool)
    hosts = np.unique(np.concatenate([src, dst]))
    anchors = np.unique(src[att] if family == "inject_src" else dst[att])
    a = np.repeat(anchors, k)
    if len(a) > cap:
        a = rng.choice(a, cap, replace=False)
    peers = rng.choice(hosts, len(a))
    fx = X[rng.choice(ben, len(a))]
    ns, nd = (a, peers) if family == "inject_src" else (peers, a)
    return (np.concatenate([src, ns]), np.concatenate([dst, nd]), np.vstack([X, fx]),
            np.concatenate([y, np.zeros(len(a), y.dtype)]),
            np.concatenate([np.ones(n, bool), np.zeros(len(a), bool)]))


def make_batches(ts, delta, cap):
    """Split a time-sorted index range into windows of `delta` ms, then into micro-batches of <= cap events.
    The concatenation of all batches is exactly arange(len(ts)) (nothing lost, order kept)."""
    if delta <= 0:  # pure event stream: consecutive micro-batches of `cap` events, no window boundaries
        return [np.arange(i, min(i + cap, len(ts))) for i in range(0, len(ts), cap)]
    w = (ts - ts[0]) // delta
    out = []
    for seg in np.split(np.arange(len(ts)), np.flatnonzero(np.diff(w)) + 1):
        out += [seg[i:i + cap] for i in range(0, len(seg), cap)]
    return out
