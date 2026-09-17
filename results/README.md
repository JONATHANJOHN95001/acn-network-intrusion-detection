# What is in results/

Every file here is generated. Nothing is edited by hand, so anything can be
rebuilt by rerunning the script that made it.

## Classifier comparison

| File | What it holds |
|---|---|
| `results.csv` | One row per run: model, split, test size, accuracy, macro and weighted F1, fit and prediction time, throughput, plus per-class recall and F1 |
| `reports/<model>_<split>[_s<sample>].txt` | The full per-class classification report and confusion matrix for that run |
| `reports/<model>_<split>[_s<sample>]_cm.npy` | The same confusion matrix as an array, for plotting |

`results.csv` mixes three run protocols; the `sample` column says which:

| `sample` | Rows used | Why it exists |
|---|---|---|
| `full` | all 2,574,059 flows | the most faithful run, test size 0.2 only |
| `400000` | 400k stratified sample | the headline table, all three test sizes, strictly comparable |
| `100000` | 100k stratified sample | the only scale where KNN and RBF-SVM finish, so all sixteen models appear |

Rebuild: `python 02_experiment.py --all`. Runs already present are skipped, so
an interrupted run can simply be started again.

## Everything else

| Folder | Contents | Rebuild with |
|---|---|---|
| `01_data_audit.txt` | the cleaning report: duplicates, dead columns, class distribution | `python 01_prepare_data.py` |
| `kernel_pca/` | the kernel and component grid, one CSV and one Excel workbook per run, including the repeat samples (`_seed1`, `_seed2`) and the gamma sweep (`_gamma`) | `python 03_kernel_pca.py --classifiers all` |
| `features/` | permutation importance, redundant feature pairs, and the per-attack signatures | `python 04_feature_importance.py` |
| `imbalance/` | macro F1 and per-class recall for each imbalance strategy | `python 05_imbalance.py` |
| `paper/` | the two checks on common assumptions: the shortcut-feature ablation and the duplicate-leakage measurement | `python 09_paper_experiments.py`, `python 10_leakage.py` |
| `charts/` | the twelve report figures | `python 06_charts.py` |
| `ACN_IDS_Report.docx`, `ACN_IDS_Slides.pptx` | the write-up and the deck, built from the files above | `python 07_report.py`, `python 08_slides.py` |

Run logs (`run*.log`) are deliberately not committed: they contain absolute
paths from whichever machine produced them.

## Reading the numbers

Report **macro F1**, not accuracy. Benign traffic is 83% of the data, so a
model that ignores every rare attack still scores above 0.9 on accuracy. Macro
F1 weights all fifteen classes equally, which is why it separates models that
accuracy cannot tell apart.
