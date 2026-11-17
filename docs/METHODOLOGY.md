# Methodology

## Datasets

| Dataset | Flows used | Train / validation / test | Attack rate (train / val / test) | Hosts | Test flows with feature vector also in train |
|---|---|---|---|---|---|
| NF-UNSW-NB15-v3 (`UNSW`, chronological) | 472,395 | 284,727 / 93,426 / 94,242 | 2.1% / 6.9% / 6.7% | 39 | 32.4% |
| NF-CSE-CIC-IDS2018-v3 (`CIC`, chronological) | 968,418 | 589,408 / 189,972 / 189,038 | 17.8% / 1.4% / 6.1% | 31,356 | 13.0% |
| NF-CSE-CIC-IDS2018-v3 (`CICc`, class-chronological) | 949,060 | 581,057 / 183,999 / 184,004 | 12.3% in every split | 31,356 | 16.2% (validation: 21.3%) |

Each flow has 47 numeric NetFlow-v3 features. IP addresses, ports, timestamps and labels are never used as features. Flows are read as a contiguous slice of every CSV chunk (`--keep_per_chunk` rows from each `--chunksize` rows), exact duplicate rows are removed, and the result is sorted by flow start time. The prepared datasets are therefore a sub-sample of the original captures: window-level host degrees are lower than in the full capture and should be read as relative quantities.

## Data preparation
* Splits (`gnnids/data.py`):
  * `chrono`: first 60% train, next 20% validation, last 20% test, with a 5-minute purge gap. Later periods contain attack families absent from training (zero-day setting). On CIC-2018 the attack families are concentrated in time: training contains DDoS-HOIC, DDoS-LOIC-HTTP, DoS-GoldenEye, DoS-Hulk, FTP-BruteForce, SSH-Bruteforce, Brute-Force-Web, Brute-Force-XSS and SQL-Injection; validation contains only Infiltration (2,590 flows, 1.4% of the period); the test period contains Bot (9,961 flows, 86% of its attacks) and Infiltration (1,575 flows). Neither Bot nor Infiltration occurs in training.
  * `classchrono`: chronological 60/20/20 inside every class (rank based, 1% purge at each boundary). Every family with at least 30 flows appears in all splits; `Brute_Force_-XSS` (2 flows) and `SQL_Injection` (20 flows) are placed in training only. The split is class-stratified, not globally time-ordered, and removes natural drift between splits.
  * `blocked`: contiguous segments assigned cyclically. It is implemented but not used, because attack families concentrated in time can fall into a single split.
* Features: columns whose training maximum exceeds 100 receive a signed `log1p`; all features are then standardised with training statistics only and clipped to +-10 standard deviations. The log-transform columns are selected per dataset (see Limitations in `RESULTS.md` for the consequence for transfer experiments).
* Every validation/test flow whose feature vector also occurs in training is flagged; results are available on the full test set and on the duplicate-free test set.

## Models
* **Tabular baselines**: XGBoost and LightGBM with random search on validation PR-AUC (equal budget, 3 trials). *XGBoost + topology statistics* adds causal in-window host counts (number of earlier flows of the source, of the destination and of the pair). *IP-identity only* uses the source and destination host identifiers as categorical features and measures the host-layout shortcut. Random Forest is implemented (`--models rf`) but is not part of the final run.
* **E-GraphSAGE (static)**: edge-aware message passing, `m_{u->v} = f([h_u ; e_uv])`, mean aggregation in both directions, edge classifier on `[h_src ; h_dst ; e]` (`gnnids/model.py`). Hidden size 128, two layers, full-neighbourhood mean aggregation within a window (no neighbour sampling).
* **TE-GS**: the same encoder with a GRU memory per host, updated after every window.
* **Degree features** (`--node_feats`): in- and out-degree of each host inside the current window.
* **Adversarial training** (`--adv`): edges injected from attacker hosts (1, 5, 10, 20 or 50 per attacker host) are added to training windows; the loss is computed on real edges. Checkpoints are selected on a clean and an attacked validation stream.
* Training uses focal loss (gamma 2, class-balanced alpha), Adam with learning rate 0.002, at most 20 epochs with patience 6 on validation PR-AUC, windows of 60 s split into micro-batches of at most 2,048 flows. Static models shuffle the order of training batches; memory models keep chronological order.

## Evaluation
* Metrics: PR-AUC, ROC-AUC, macro-F1, MCC, accuracy, balanced accuracy, TPR at 1% and 0.1% FPR, FPR at 95% and 99% TPR, recall per attack family.
* Two thresholds are reported: macro-F1 optimal on validation, and an operating point at 1% false alarms on validation benign flows.
* Cold-memory test: the test stream is replayed with all host memories reset to zero.
* Structural robustness (3 seeds, budgets 0, 1, 5, 10 and 100 fake edges): attacker-side edge injection (the training family), victim-side edge injection and deletion of half of the benign flows (both held out from training). Fake edges copy the features of real benign flows from the same window. Detection rate is measured at the 1%-FPR operating point.
* Zero-day: leave-one-attack-family-out (one seed); the withheld family is excluded from training and validation and scored against benign flows. Transfer: train on one dataset, test on the other with source-dataset standardisation.
* Streaming: p50/p95/p99 forward-pass latency and throughput for micro-batches of 1, 32 and 256 events on CPU and a T4 GPU, over the first 5,000 test events.
* Significance: paired bootstrap on identical test flows (`stats`, `summarize`).

## Diagnostics
* `shortcuts`: ROC-AUC of a depth-3 tree on a single feature.
* `sanity`: training on shuffled labels must give ROC-AUC near 0.5; dropping the most important features shows whether the score depends on a few columns.

## Differences from the Phase 2 design
| Phase 2 | Implemented |
|---|---|
| 43 NetFlow features | 47 numeric features (NetFlow-v3 files contain 53 columns; timestamps, addresses, ports and labels excluded). |
| UNSW-NB15, CSE-CIC-IDS2018, ToN-IoT, BoT-IoT, CIC-IoT2023 | UNSW-NB15 and CSE-CIC-IDS2018 only. |
| Chronological 60/20/20 split | Same, plus the class-chronological split `CICc`, added because the chronological CIC-2018 validation period contains only Infiltration attacks (1.4% of flows) and the test period is dominated by Bot. |
| 10 seeds per model | 3 seeds; leave-one-family-out uses 1 seed. |
| McNemar, Wilcoxon, Friedman, Holm correction | Paired bootstrap on identical test flows; no multiple-comparison correction. |
| Baselines: XGBoost, LightGBM, Random Forest, MLP, topology statistics, GCN, GAT, GraphSAGE, SAGE-Transformer, windowed GCN + GRU, hardened E-GraphSAGE (Galli) | XGBoost, LightGBM, XGBoost + topology statistics, IP-identity probe, static E-GraphSAGE, TE-GS. |
| Node features: degree, distinct peers, flow rate | In- and out-degree within the window. |
| Neighbour sampling with fixed fan-out | Full-neighbourhood mean aggregation within a window; micro-batch cap of 2,048 flows. |
| Adversarial training: endpoint swap, edge injection, bounded feature perturbation | Attacker-side edge injection only. |
| Attack budgets 0 to 1000; edge insertion, deletion and node insertion; surrogate-model attacker | Budgets 0, 1, 5, 10, 100; attacker-side and victim-side edge injection and benign-flow deletion; no node insertion, no surrogate attacker. |
| Window length 30/60/120/300 s | Same for TE-GS + degree features. Encoder depth 1, 2 and 3 was varied for static + degree features. Neighbour fan-out was not varied (no neighbour sampling). |
| Ablation without edge attributes and topology-only | Topology-only run for static + degree features (`--no_edge --tag _noedge`: flow features hidden, degree features kept). The edge-attributes-only counterpart is the tabular baseline. |
| GNNExplainer, time to detection, false-positive audit by host role, UNSW-NB15 label audit | Not performed. |
