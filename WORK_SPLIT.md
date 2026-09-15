# Work split

Three people. Each owns a track end to end: experiments, charts, and the
matching part of the report and slides, so anyone can answer questions on
their own section.

The dataset is not in the repo. Everyone first downloads CIC-IDS2017, runs
`python 01_prepare_data.py`, and confirms `data/cicids2017.parquet` appears.
See the README.

## Jonathan (JONATHANJOHN95001) — Kernel PCA and lead

- **Task:** Kernel PCA dimensionality reduction, the assignment's main task.
- **Scripts:** `03_kernel_pca.py`.
- **Experiments:** the 5 kernels x 3 test sizes x 3 component counts grid for
  all seven classifiers; repeat runs on other samples (`--seed 1`, `--seed 2`)
  to show the results are stable; tune `gamma` for the rbf, poly and sigmoid
  kernels (`--gammas`).
- **Charts:** 07, 08, 09 (kernel-by-classifier, best-vs-baseline, components).
- **Report:** abstract, introduction, Kernel PCA method and results.
- **Also:** review and merge the others' pull requests; assemble the final
  report and slides.

## Subangi (SUBANGIVIGNESH30) — classifier comparison and class imbalance

- **Task:** compare the classifiers and deal with the class imbalance.
- **Scripts:** `02_experiment.py`, `05_imbalance.py`.
- **Experiments:** every classifier at test sizes 0.2 / 0.4 / 0.6 on the full
  dataset (`python 02_experiment.py --all`); then class weights, undersampling
  and SMOTE on the top few models. Use `--sample` on a weaker laptop; the full
  run takes a few hours and about 5 GB of memory.
- **Charts:** 03, 04, 05, 06 (accuracy vs macro F1, per-class F1, test size,
  throughput) and 11, 12 (imbalance).
- **Report:** method (pipeline, why macro F1), classifier comparison, the
  imbalance study.

## Varshini (varshiniT221) — the networking side

- **Task:** the ACN part. What an IDS is, what the flow features mean, and how
  each attack shows up in them.
- **Scripts:** `04_feature_importance.py`.
- **Work:** run the feature analysis; explain each of the attacks at the
  protocol level (see `docs/networking_background.md`, already drafted as a
  starting point); write the deployment discussion (NetFlow / IPFIX, detection
  delay, throughput, encryption).
- **Charts:** 01, 02 (class sizes, duplicates) and 10 (feature importance).
- **Report:** background, the dataset and cleaning, feature analysis,
  deployment.

## How we avoid clashes on GitHub

- Each person works on their own branch and opens a pull request for Jonathan
  to review and merge. Do not push straight to `main`.
- Each person owns their own files, so two people rarely touch the same one.
- If you run the classifier grid, write to your own results file with
  `--out results_<name>.csv` so `results/results.csv` is not a merge conflict.
- The report text lives in a shared Google Doc, not here. Only the charts,
  code and results tables are in the repo.

## Order

1. Everyone: accept the GitHub invite, get the data, run steps 1 and 2 once.
2. Experiments, in parallel.
3. Charts and your report section.
4. Merge, build the slides, rehearse. Sir can ask anyone about any part.
