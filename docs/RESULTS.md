# Results

All results come from one complete run of `notebooks/GNN_IDS_Phase3.ipynb` (Google Colab, T4 GPU). Values are means with sample standard deviations over seeds; `n` is the number of seeds. Unless stated otherwise, baselines and graph models use seeds 0, 1 and 2. Per-run files are in `results/`, aggregated tables in `report/`. Reproduce with `python -m gnnids summarize`, `stats`, `report` and `paper`. Dataset sizes and split composition are listed in [METHODOLOGY.md](METHODOLOGY.md).

Model names: *static* = E-GraphSAGE without memory; *TE-GS* = E-GraphSAGE with GRU node memory; *+ deg* = in-/out-degree node features; *+ adv* = adversarial training.

## 1. Benchmark saturation

| Dataset | Strongest single feature (ROC-AUC) | All features | Top-10 features dropped | Labels shuffled | IP-identity probe (PR-AUC) |
|---|---|---|---|---|---|
| NF-UNSW-NB15-v3 | `MIN_TTL`, `MAX_TTL`: 1.000 | 1.000 | 1.000 | 0.41 ± 0.06 | 1.000 ± 0.000 |
| NF-CSE-CIC-IDS2018-v3 (class-chronological) | `LONGEST_FLOW_PKT`: 0.979 | 0.991 | 0.990 | 0.47 ± 0.08 | 0.999 ± 0.000 |

On NF-UNSW-NB15-v3 every model reaches PR-AUC ≥ 0.999 and TPR@1%FPR ≈ 1.000 (accuracy 0.9999 against 0.933 for the majority class; recall 1.00 for every attack family). The prepared data contain only 39 distinct hosts, and a model that sees nothing but the source and destination host identifiers separates the classes perfectly. In-distribution scores on this dataset and on the class-chronological CIC-2018 split therefore cannot rank models. Shuffling the training labels gives ROC-AUC near 0.5, so the high scores do not come from a leak in the training pipeline; they reflect testbed artefacts (TTL values, packet lengths, fixed host roles). Figure: `P1_saturation_shortcuts.png`.

Duplicates: 32.4% (UNSW), 13.0% (CIC chronological) and 16.2% (CICc) of test flows have a feature vector that also occurs in training. On NF-UNSW-NB15-v3 the scores on the duplicate-free test set equal the full-test scores. On CICc tabular and static models change by about 0.011 PR-AUC (XGBoost 0.980 → 0.969; static + deg 0.981 → 0.970) and the memory models by +0.024 to +0.027 (TE-GS + deg 0.921 → 0.945; TE-GS 0.890 → 0.917). On the CIC-2018 chronological split the choice matters more: XGBoost 0.513 → 0.724, static + deg 0.878 → 0.881, TE-GS + deg 0.660 → 0.713 (mean PR-AUC, full versus duplicate-free test set).

## 2. CIC-2018, chronological split (zero-day), 3 seeds

The attack families are concentrated in time. Training contains nine DoS, DDoS, brute-force and injection families; validation contains only Infiltration; the test period contains Bot (86% of test attacks, never seen in training) and Infiltration (14%). The result is therefore a statement about Bot and Infiltration.

| Model | PR-AUC | TPR@1%FPR | Macro-F1 | Bot recall | Infiltration recall |
|---|---|---|---|---|---|
| Static + deg | **0.878 ± 0.002** | 0.864 | 0.957 | 1.00 | 0.00 |
| Static | 0.856 ± 0.017 | 0.865 | 0.930 | 1.00 | 0.01 |
| XGBoost + topology statistics | 0.768 ± 0.096 | 0.745 | 0.755 | 0.66 | 0.01 |
| TE-GS + deg | 0.660 ± 0.378 | 0.571 | 0.777 | 0.66 | 0.01 |
| XGBoost | 0.513 ± 0.200 | 0.292 | 0.490 | 0.33 | 0.01 |
| LightGBM | 0.467 ± 0.254 | 0.252 | 0.616 | 0.27 | 0.01 |
| IP-identity only | 0.042 ± 0.002 | 0.002 | 0.484 | 0.04 | 0.03 |

Bot and Infiltration recall are taken at the operating point of 1% false alarms on validation benign flows (mean over seeds, `report/CIC_family_recall.csv`); macro-F1 uses the validation-F1 threshold.

Seed values: static + deg 0.876 / 0.877 / 0.879; XGBoost 0.679 / 0.568 / 0.290; LightGBM 0.760 / 0.341 / 0.300; XGBoost + topology 0.810 / 0.836 / 0.658; TE-GS + deg 0.223 / 0.877 / 0.880. In the first TE-GS + deg run (seed 0) training did not converge (validation PR-AUC 0.014, equal to the validation prevalence); the other two seeds match the static model.

Paired bootstrap on identical test flows (seed 0; PR-AUC difference, static + deg minus other): XGBoost +0.198 [0.192, 0.205]; LightGBM +0.256 [0.248, 0.264]; XGBoost + topology +0.067 [0.060, 0.073]; static without degree features +0.033 [0.027, 0.039].

The static graph models detect Bot at the 1%-false-alarm operating point in every seed (observed false-alarm rate 0.0 to 0.6%); the tabular models do so in at most one seed. Degree features add about 0.02 PR-AUC and reduce seed variance; the larger difference is between the static graph encoders (0.856 and 0.878) and the tabular models. No model detects Infiltration. Figures: `P2`, `P3`, `P13`.

Threshold behaviour. Thresholds are chosen on validation data, which here contains only Infiltration and therefore carries no information about Bot. At the macro-F1 threshold XGBoost predicts almost everything benign (balanced accuracy 0.5006, Bot recall 0.33 or less). At the operating point of 1% false alarms on validation benign flows, XGBoost recovers overall attack recall 0.858 in seed 0 but 0.012 and 0.002 in seeds 1 and 2; the static + deg model keeps 0.86 in all three seeds.

## 3. CIC-2018, class-chronological split (in-distribution), 3 seeds

| Model | PR-AUC | TPR@1%FPR |
|---|---|---|
| IP-identity only | 0.999 ± 0.000 | 0.999 |
| XGBoost + topology statistics | 0.981 ± 0.000 | 0.967 |
| Static + deg | 0.981 ± 0.003 | 0.966 |
| Static | 0.980 ± 0.001 | 0.965 |
| XGBoost | 0.980 ± 0.000 | 0.966 |
| LightGBM | 0.980 ± 0.000 | 0.966 |
| Static + deg + adv | 0.980 ± 0.002 | 0.965 |
| TE-GS + deg | 0.921 ± 0.048 | 0.866 |
| TE-GS | 0.890 ± 0.056 | 0.793 |
| TE-GS + deg + adv | 0.527 ± 0.314 | 0.394 |

Static graph models and tabular models are indistinguishable (differences below 0.001 PR-AUC; the paired bootstrap on seed 0 favours XGBoost by 0.001). Memory models are lower and vary by seed, and adversarial training makes the memory model unstable (seeds 0.205 / 0.832 / 0.544). The IP-identity probe is the highest-scoring model, which confirms that host identity alone suffices on this split. The weakest families for the tabular and static models are Infiltration (recall 0.01 to 0.06) and Brute-Force-Web (13 test flows, recall 0.15 to 0.26). `Brute_Force_-XSS` and `SQL_Injection` occur in training only.

## 4. Recurrent memory and host identity (3 seeds)

PR-AUC with memory continued from training (warm) and with all host memories reset at test time (cold):

| Dataset | Model | Warm | Cold |
|---|---|---|---|
| NF-UNSW-NB15-v3 | TE-GS | 0.999 ± 0.000 | 0.999 ± 0.001 |
| NF-UNSW-NB15-v3 | TE-GS + deg | 1.000 ± 0.000 | 0.999 ± 0.000 |
| CIC-2018 (class-chronological) | TE-GS | 0.890 ± 0.056 | 0.843 ± 0.101 |
| CIC-2018 (class-chronological) | TE-GS + deg | 0.921 ± 0.048 | 0.918 ± 0.054 |

Cold-start scores are close to warm scores. The largest drop is one seed of TE-GS on CIC-2018 (0.834 → 0.743). The cold-start test therefore gives no evidence that the memory carries a large host-identity shortcut in these runs; in the leave-one-family-out experiment (section 7) the memory models do generalise worse to the withheld Bot family than the static models. Figure: `P7`.

## 5. Ablation (test PR-AUC, 3 seeds)

| Variant | UNSW-NB15 | CIC-2018 class-chronological | CIC-2018 zero-day |
|---|---|---|---|
| Static | 0.999 | 0.980 | 0.856 |
| Static + deg | 0.999 | 0.981 | 0.878 |
| Static + deg + adv | 0.999 | 0.980 | not run |
| TE-GS | 0.999 | 0.890 | not run |
| TE-GS + deg | 1.000 | 0.921 | 0.660 (seeds 1 and 2: 0.878) |
| TE-GS + deg + adv | 0.999 | 0.527 | not run |

Window length is in section 9. Flow-feature and depth ablations, static + deg (test PR-AUC, mean ± sd over 3 seeds; `--no_edge` hides the flow features so that only graph structure and degree features remain):

| Variant | UNSW-NB15 | CIC-2018 class-chronological | CIC-2018 zero-day |
|---|---|---|---|
| Full, 2 layers | 0.999 ± 0.000 | 0.981 ± 0.003 | 0.878 ± 0.002 |
| Topology only (`--no_edge`) | 0.910 ± 0.033 | 0.976 ± 0.002 | 0.905 ± 0.000 |
| 1 layer | 0.999 ± 0.000 | 0.979 ± 0.003 | 0.878 ± 0.000 |
| 3 layers | 0.999 ± 0.000 | 0.976 ± 0.002 | 0.877 ± 0.002 |

Without flow features the model reaches 0.905 on the zero-day split (per seed 0.905 / 0.905 / 0.906), slightly above the 0.878 with them, so on that split graph structure alone carries the detection of the unseen Bot family. Where test attacks resemble training attacks the flow features are needed (UNSW-NB15 0.910 against 0.999). Depth has no effect beyond the seed spread. The result is three seeds on one split and concerns Bot. The neighbour fan-out sweep was not run because the encoder has no neighbour sampling. Figures: `P12`, `P11` (the new runs are not part of the figures).

## 6. Structural robustness (3 seeds)

Detection rate at the 1%-FPR operating point after injecting fake benign-looking edges. Attacker-side injection (*seen*) is the training family of the adversarial models; victim-side injection (*held-out*) is not used in training. Values are listed per seed.

**NF-UNSW-NB15-v3**

| Model | 10 edges, seen | 100 edges, seen | 100 edges, held-out |
|---|---|---|---|
| Static | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Static + deg | 1.00 / 0.999 / 1.00 | 0.999 / 0.881 / 0.999 | 0.999 / 0.808 / 0.998 |
| Static + deg + adv | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 0.999 | 1.00 / 1.00 / 0.999 |
| TE-GS | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 0.542 | 1.00 / 1.00 / 0.407 |
| TE-GS + deg | 1.00 / 1.00 / 0.999 | 0.991 / 0.992 / 0.380 | 0.987 / 0.990 / 0.291 |
| TE-GS + deg + adv | 1.00 / 1.00 / 1.00 | 1.00 / 0.999 / 1.00 | 1.00 / 0.998 / 1.00 |

Without injection every model detects 1.00 of the attacks, and deleting half of the benign flows has no effect. Adversarial training keeps detection at 0.998 or higher at 100 injected edges in all seeds, including the held-out victim-side family. The models without adversarial training degrade in some seeds, and TE-GS with degree features falls to 0.29 to 0.38 in one seed.

**CIC-2018 (class-chronological)**, detection rate at 0 and 100 edges (seeds equal unless noted): static 0.965 → 0.881 (attacker-side), 0.965 → 0.915–0.958 (victim-side); static + deg 0.965 → 0.881 and 0.90–0.96; static + deg + adv 0.965 → 0.881 and 0.95–0.96. The memory models are unaffected by injection (identical detection at 0 and 100 edges), at their own, lower clean detection rate (TE-GS 0.61–0.85; TE-GS + deg 0.74–0.96). On this split adversarial training does not improve attacker-side robustness. Figure: `P6`. The attack is a simplified problem-space model (fake edges copy real benign flows of the window); no feature-space attack and no surrogate-model attacker is evaluated.

## 7. Zero-day by leave-one-family-out (CIC-2018 class-chronological, 1 seed)

ROC-AUC of the withheld family against benign flows (recall at 1% false alarms on benign test flows in brackets):

| Withheld family | TE-GS + deg | Static + deg | XGBoost | LightGBM |
|---|---|---|---|---|
| Bot | 0.464 (0.003) | 1.000 (1.000) | 0.609 (0.000) | 0.728 (0.000) |
| DDoS-HOIC | 0.999 (0.988) | 1.000 (1.000) | 0.814 (0.000) | 0.859 (0.043) |
| Infiltration | 0.384 (0.003) | 0.287 (0.004) | 0.318 (0.010) | 0.501 (0.009) |

The static + deg model separates Bot and DDoS-HOIC from benign flows when they are withheld; the memory model does so for HOIC but not for Bot; the tabular models are weaker on all three. Infiltration is not recovered by any model. On NF-UNSW-NB15-v3 every model scores 1.00 on the withheld families Exploits, Fuzzers and Reconnaissance (figure `P4`). Results are from one seed.

## 8. Transfer

PR-AUC on the target test set after training on the source and standardising with source statistics, per seed (0 / 1 / 2). Chance equals the target prevalence.

| Source → target | Static | Static + deg | TE-GS + deg | XGBoost | LightGBM | Chance |
|---|---|---|---|---|---|---|
| CIC-2018 chronological → UNSW-NB15 | 0.348 / 0.204 / 0.554 | 0.926 / 0.249 / 0.546 (mean 0.574 ± 0.339) | 0.073 / 0.059 / 0.064 | 0.047 / 0.073 / 0.046 | 0.126 / 0.097 / 0.134 | 0.067 |
| UNSW-NB15 → CIC-2018 class-chronological | 0.376 / 0.630 / 0.283 | 0.479 / 0.186 / 0.193 (mean 0.286) | 0.421 / 0.828 / 0.556 (mean 0.602) | 0.162 / 0.186 / 0.248 | 0.243 / 0.333 / 0.240 | 0.123 |
| CIC-2018 class-chronological → UNSW-NB15 | 0.785 / 0.879 / 0.123 | 0.275 / 0.526 / 0.747 (mean 0.516) | 0.056 / 0.052 / 0.047 | 0.043 / 0.078 / 0.046 | 0.055 / 0.073 / 0.069 | 0.067 |

Static graph models transfer above chance in all three directions, tabular models and TE-GS with degree features stay near chance from CIC-2018 to UNSW-NB15, and every result varies strongly between seeds. Figure `P5` shows the two class-chronological directions only.

**Sensitivity to the log-transform columns.** Columns whose training maximum exceeds 100 receive a signed `log1p`, and this selection is made per dataset. Six of the 47 columns differ between UNSW and CIC (`PROTOCOL`, `TCP_FLAGS`, `CLIENT_TCP_FLAGS`, `SERVER_TCP_FLAGS`, `RETRANSMITTED_IN_PKTS`, `FTP_COMMAND_RET_CODE`), so in a transfer experiment these columns reach the model with a different transformation than in training. The CIC-2018 chronological → UNSW-NB15 direction was therefore repeated on CPU with the prepared datasets rebuilt with one shared mask (all 47 columns log-transformed), and, for comparison, with the original datasets (3 seeds; one tuning trial for the tabular models):

| Model | Notebook run (GPU, per-dataset masks) | CPU re-run, per-dataset masks | CPU re-run, shared mask |
|---|---|---|---|
| Static + deg | 0.926 / 0.249 / 0.546 (mean 0.574) | 0.671 / 0.331 / 0.535 (mean 0.512 ± 0.171) | 0.958 / 0.825 / 0.864 (mean 0.882 ± 0.069) |
| XGBoost | 0.047 / 0.073 / 0.046 (mean 0.055) | 0.046 / 0.088 / 0.045 (mean 0.060) | 0.042 / 0.041 / 0.041 (mean 0.041) |
| LightGBM | 0.126 / 0.097 / 0.134 (mean 0.119) | 0.163 / 0.168 / 0.103 (mean 0.145) | 0.054 / 0.055 / 0.045 (mean 0.051) |

The ranking is unchanged: tabular models stay near chance (0.067) and the static graph model with degree features is far above. With consistent preprocessing the graph model transfers higher and more stably, so the per-dataset masks lower and destabilise transfer. The other transfer directions and the figure `P5` were not re-run with a shared mask.

## 9. Window length and stability (TE-GS + deg, CIC-2018 class-chronological, 3 seeds)

| Window | PR-AUC per seed | Mean |
|---|---|---|
| 30 s | 0.742 / 0.862 / 0.901 | 0.835 |
| 60 s (default) | 0.944 / 0.953 / 0.866 | 0.921 |
| 120 s | 0.952 / 0.962 / 0.928 | 0.947 |
| 300 s | 0.959 / 0.971 / 0.926 | 0.952 |

Short windows lower the score and increase the spread. On NF-UNSW-NB15-v3 all windows give PR-AUC ≥ 0.998. Figure: `P8`.

## 10. Streaming cost (forward pass only)

TE-GS + deg + adv, first 5,000 test events, range over three seeds and two datasets (UNSW, CICc). Feature extraction and graph construction are not included.

| Device | Micro-batch | p50 (ms) | p95 (ms) | Events/s |
|---|---|---|---|---|
| CPU | 1 | 0.82 – 0.95 | 1.16 – 1.39 | 1,004 – 1,146 |
| CPU | 32 | 1.35 – 9.23 | 1.60 – 10.73 | 3,458 – 22,807 |
| CPU | 256 | 2.58 – 61.2 | 2.96 – 67.8 | 4,074 – 82,197 |
| GPU (T4) | 1 | 1.54 – 1.72 | 2.34 – 2.53 | 541 – 607 |
| GPU (T4) | 32 | 1.59 – 2.43 | 1.82 – 3.68 | 12,475 – 19,044 |
| GPU (T4) | 256 | 1.53 – 2.71 | 1.69 – 3.05 | 71,551 – 135,120 |

A single event takes about 1.2 to 1.4 ms (p95) on CPU, roughly 1,000 events per second; the GPU is slower for single events and faster for large micro-batches. CPU timings for larger micro-batches vary by more than a factor of 20 between runs on the shared Colab host. Detection quality at micro-batch size 1 was not evaluated. Figure: `P9`.

## 11. Comparison with published results

Published figures come from the papers reviewed in Phases 1 and 2. They use different datasets, feature sets, splits and metrics, so the comparison is qualitative.

| Reference | Reported | This work |
|---|---|---|
| Yilmaz et al. 2026 (GraphSAGE, UNSW-NB15) | F1 0.9721 ± 0.0006 | Macro-F1 0.9995 for static E-GraphSAGE and 0.9997 for XGBoost on NF-UNSW-NB15-v3 (chronological). The higher scores reflect the saturation of this dataset version, not a better model. |
| El Azhar et al. 2026 (Graph-Fusion, Bot-IoT, ToN-IoT) | F1 100.00% and 98.35% | Saturation is confirmed on UNSW-NB15: a single TTL column and an IP-identity probe both reach ROC-AUC 1.000. |
| IOTTRUST 2026 (topology statistics, UNSW-NB15) | +0.09 accuracy points over flow-only | XGBoost + topology versus XGBoost: +0.0001 PR-AUC on UNSW-NB15 and +0.0007 on CICc; +0.255 on the CIC-2018 zero-day split (standard deviations 0.10 and 0.20). |
| Mohammad et al. 2026 (GNN versus DNN, CICDDoS2019) | GNN 0.86 against DNN 0.97 | Static E-GraphSAGE matches XGBoost on CICc (0.980 against 0.980) and exceeds it on the zero-day split (0.878 against 0.513). |
| González-Ramos et al. 2026 (hybrid CNN-LSTM-Transformer, official UNSW-NB15 temporal split) | Accuracy 94.20%, F1 95.69% | Accuracy 0.9999 on the chronological NF-UNSW-NB15-v3 split. Different feature set; not comparable. |
| Galli et al. 2025 (adversarial training of E-GraphSAGE, node injection) | Baseline degrades beyond about 100 injected nodes; hardened model better but not perfect | Edge injection on UNSW-NB15: models without adversarial training degrade at 100 edges in some seeds (0.29 to 0.88); with adversarial training detection stays at 0.998 or higher. On CIC-2018 adversarial training does not help against attacker-side injection. |
| GIDS-Eval 2026 (snapshot window) | Window alone changes average precision by 38.3% on average | TE-GS + deg on CICc: PR-AUC 0.835 at 30 s, 0.952 at 300 s (14% relative); no effect on UNSW-NB15. |

## 12. Research questions

* **RQ1 (graph versus tuned tabular model).** In-distribution the answer is no: NF-UNSW-NB15-v3 is saturated and on CICc static graph models and tuned tabular models differ by less than 0.001 PR-AUC. On the chronological CIC-2018 split the static graph models reach PR-AUC 0.856 to 0.878 against 0.467 to 0.513 for XGBoost and LightGBM, because they detect the unseen Bot family at 1% false alarms in all seeds. XGBoost with topology statistics closes part of the gap (0.768). The temporal memory does not help: TE-GS + deg matches the static model in two seeds and fails in one.
* **RQ2 (transfer).** Static graph models with degree features reach PR-AUC 0.51 to 0.88 after transfer from the chronological CIC-2018 split to UNSW-NB15 (chance 0.067; 0.88 with a shared log-transform), while XGBoost and LightGBM remain at 0.04 to 0.15. Transfer from UNSW-NB15 to CICc is weaker (static + deg 0.29, TE-GS + deg 0.60, tabular 0.20 to 0.27) and from CICc to UNSW-NB15 it ranges from 0.12 to 0.88 for static models and is at chance for TE-GS and tabular models. Seed variance is large and the magnitude depends on the consistency of the preprocessing between datasets (section 8).
* **RQ3 (latency).** A forward pass for one event takes about 1.2 to 1.4 ms (p95) on CPU, so incremental inference fits a latency budget of a few milliseconds per event. Detection quality at micro-batch size 1 was not measured.
* **RQ4 (adversarial robustness).** On NF-UNSW-NB15-v3 adversarial training trained on attacker-side injection holds against victim-side injection, a family not seen in training, at 100 fake edges in all three seeds. A surrogate-model attacker and budgets above 100 were not evaluated. On CIC-2018 the benefit does not appear.

## Limitations

* Two datasets, each a sub-sample (a contiguous slice of every CSV chunk); window degrees are lower than in the full captures. NF-UNSW-NB15-v3 contains 39 hosts.
* The chronological CIC-2018 split has a validation period with a single attack family (Infiltration) that no model detects, so checkpoint selection and thresholds are not informed by the families present in the test period. The zero-day result concerns Bot only.
* The class-chronological split is class-stratified rather than globally time-ordered and removes natural drift between splits.
* Duplicate feature vectors between training and test (13 to 32% of test flows) are reported separately; scores on CIC-2018 chronological change by up to 0.21 PR-AUC when they are removed.
* Three seeds, and one seed for leave-one-family-out. TE-GS + deg on the zero-day split has one non-converged seed. Tabular baselines on the zero-day split vary strongly between seeds.
* GPU runs are not bit-reproducible; rerunning gives slightly different numbers.
* The structural attack is a simplified problem-space model; no feature-space attack and no surrogate-model attacker is evaluated.
* No model detects Infiltration; Brute-Force-Web has 13 test flows.
* Several experiments of the Phase 2 design were not run; see the table at the end of [METHODOLOGY.md](METHODOLOGY.md). The topology-only and depth ablations were added after the main run.
* The log-transform columns are selected per dataset, so transfer experiments mix transformations in six columns (section 8). The results are reported as executed, together with a sensitivity re-run for one direction.
