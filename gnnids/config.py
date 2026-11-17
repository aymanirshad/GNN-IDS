"""Column names of the NetFlow-v3 datasets and shared constants.
If a CSV uses other column names, change them here (nowhere else)."""
SRC, DST, TS = "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "FLOW_START_MILLISECONDS"
LABEL, ATTACK = "Label", "Attack"
NOT_FEATURES = [SRC, DST, "L4_SRC_PORT", "L4_DST_PORT", TS, "FLOW_END_MILLISECONDS", LABEL, ATTACK]
DEFAULT_DELTA_S = 60

# human-readable model names used in every figure/table
MODEL_NAMES = {
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "rf": "Random Forest",
    "topo": "XGBoost + topology stats",
    "iponly": "IP-identity only (shortcut probe)",
    "tegs_nomem": "E-GraphSAGE (static)",
    "tegs_nomem_adv": "E-GraphSAGE + adv. training",
    "tegs": "TE-GS (GRU memory)",
    "tegs_adv": "TE-GS + adv. training",
    "tegs_nf": "TE-GS + degree features",
    "tegs_nf_adv": "TE-GS + degree feat. + adv.",
    "tegs_nomem_nf": "E-GraphSAGE + degree features",
    "tegs_nomem_nf_adv": "E-GraphSAGE + degree feat. + adv.",
}
