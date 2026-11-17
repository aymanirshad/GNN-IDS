# Changelog

## 0.6
* New option `--no_edge` for `tegs`: hides the flow (edge) features inside the model so that only graph structure and degree features remain (topology-only ablation). Default behaviour is unchanged.
* Added topology-only (`--tag _noedge`) and encoder-depth (`--layers 1`, `--layers 3`, `--tag _L1`, `_L3`) ablations for static + degree features; results in `docs/RESULTS.md`.

## 0.5
* Documentation aligned with the final Phase 3 run: `docs/RESULTS.md`, `docs/METHODOLOGY.md` and the findings table of `README.md` were rewritten from the per-run results (3 seeds; leave-one-family-out 1 seed).
* Added dataset composition, duplicate audit, deviations from the Phase 2 design, comparison with published results and the answers to the four research questions.
* Documented known limitations: validation period of the chronological CIC-2018 split contains only Infiltration, per-dataset log-transform columns in the transfer experiments, sub-sampled captures, simplified structural attack.

## 0.4
* Class-chronological split: rank-based assignment inside each class, so bursty families keep validation and test flows.
* Static models shuffle training batches; memory models stay chronological.
* New commands: `summarize`, `paper`, `stats`, `sanity`, `shortcuts`, `resplit`, `tune`.
* Cold-memory evaluation, operating-point threshold (1% false alarms), accuracy and balanced accuracy.
* Skip finished runs (`--force` to recompute), z-score clipping, saturation-robust threshold search.
* Total time per training batch reduced with smaller batches (`--cap`).

## 0.1 - 0.3
* Initial pipeline: preparation, baselines, TE-GS, structural attacks, robustness, latency and transfer experiments.
