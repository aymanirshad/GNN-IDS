"""TE-GS: temporal, edge-aware GraphSAGE with a GRU node memory (plain PyTorch, no graph library needed).

Lineage / what is borrowed:
  * edge-aware message passing m_{u->v} = f([h_u ; e_uv]) with mean aggregation  -- E-GraphSAGE (Lo et al., NOMS 2022)
  * per-host recurrent memory updated by a GRU cell after every window            -- temporal-GNN practice (e.g. TGN, Tanvir 2026)
  * edge classifier on [h_src ; h_dst ; e]                                         -- E-GraphSAGE
Set use_mem=False for the static E-GraphSAGE-style ablation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_agg(msg, idx, n):
    out = torch.zeros(n, msg.size(1), device=msg.device).index_add_(0, idx, msg)
    cnt = torch.zeros(n, device=msg.device).index_add_(0, idx, torch.ones(len(idx), device=msg.device)).clamp(min=1)
    return out / cnt[:, None]


class TEGS(nn.Module):
    def __init__(s, fe, h=64, L=2, use_mem=True, drop=0.1, node_feats=False, no_edge=False):
        super().__init__()
        s.h, s.L, s.use_mem, s.node_feats, s.no_edge = h, L, use_mem, node_feats, no_edge
        if node_feats:  # in/out degree of the host inside the current window (the 'causal structural counts' of Phase 2)
            s.nodef = nn.Linear(2, h)
        s.emb = nn.Sequential(nn.Linear(fe, h), nn.ReLU())
        s.msg = nn.ModuleList([nn.Linear(2 * h, h) for _ in range(L)])
        s.app = nn.ModuleList([nn.Linear(2 * h, h) for _ in range(L)])
        s.gru = nn.GRUCell(h, h)
        s.head = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, 1))

    def forward(s, x, src, dst, mem):
        ids, inv = torch.unique(torch.cat([src, dst]), return_inverse=True)
        E, n = len(src), len(ids)
        ls, ld = inv[:E], inv[E:]
        e = s.emb(x)
        if s.no_edge:  # ablation: hide the flow features, keep only the graph structure (and degree features if enabled)
            e = torch.zeros_like(e)
        old = mem[ids]
        hv = old if s.use_mem else torch.zeros(n, s.h, device=x.device)
        if s.node_feats:
            one = torch.ones(E, device=x.device)
            dout = torch.zeros(n, device=x.device).index_add_(0, ls, one)
            din = torch.zeros(n, device=x.device).index_add_(0, ld, one)
            hv = hv + F.relu(s.nodef(torch.log1p(torch.stack([din, dout], 1))))
        tgt = torch.cat([ld, ls])                      # messages flow in both directions
        for l in range(s.L):
            m = F.relu(s.msg[l](torch.cat([torch.cat([hv[ls], hv[ld]]), torch.cat([e, e])], 1)))
            hv = F.relu(s.app[l](torch.cat([hv, mean_agg(m, tgt, n)], 1)))
        u = s.gru(hv, old) if s.use_mem else hv
        return s.head(torch.cat([u[ls], u[ld], e], 1)).squeeze(1), ids, u
