# Network Intrusion Detection on CIC-IDS2017

Advanced Computer Networks course project. Compares machine-learning classifiers
for detecting network attacks from bidirectional flow records, with attention to
the rare attack classes that headline accuracy hides.

## Dataset

[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) from the Canadian
Institute for Cybersecurity: five days of captured traffic (3 to 7 July 2017),
converted to flows by CICFlowMeter. Each row is one flow with 78 numeric features
(packet lengths, inter-arrival times, TCP flag counts, window sizes) and a label.

| Capture | Attacks |
|---|---|
| Monday | none (benign baseline) |
| Tuesday | FTP-Patator, SSH-Patator |
| Wednesday | DoS Hulk, GoldenEye, slowloris, Slowhttptest, Heartbleed |
| Thursday AM | Web Attack: Brute Force, XSS, SQL Injection |
| Thursday PM | Infiltration |
| Friday AM | Bot |
| Friday PM | DDoS, PortScan |

### Getting the data

The data is not in this repository (1.1 GB; GitHub caps files at 100 MB).

1. Download `MachineLearningCSV.zip` from the dataset page above.
2. Extract it into the project folder. `01_prepare_data.py` finds the eight CSVs either
   next to the scripts or in the `MachineLearningCVE/` folder the zip creates.

Both the official file names and renamed, day-numbered copies work:

| Day | Official name | Renamed copy |
|---|---|---|
| Monday | `Monday-WorkingHours.pcap_ISCX.csv` | `01_Monday-WorkingHours-Benign.csv` |
| Tuesday | `Tuesday-WorkingHours.pcap_ISCX.csv` | `02_Tuesday-WorkingHours-BruteForce.csv` |
| Wednesday | `Wednesday-workingHours.pcap_ISCX.csv` | `03_Wednesday-WorkingHours-DoS.csv` |
| Thursday AM | `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | `04_Thursday-WorkingHours-Morning-WebAttacks.csv` |
| Thursday PM | `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | `05_Thursday-WorkingHours-Afternoon-Infiltration.csv` |
| Friday AM | `Friday-WorkingHours-Morning.pcap_ISCX.csv` | `06_Friday-WorkingHours-Morning-Bot.csv` |
| Friday PM | `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | `07_Friday-WorkingHours-Afternoon-PortScan.csv` |
| Friday PM | `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | `08_Friday-WorkingHours-Afternoon-DDoS.csv` |

To confirm all eight are found before the full run, or to read them from another folder:

```bash
python 01_prepare_data.py --check
python 01_prepare_data.py --data-dir path/to/MachineLearningCVE
```

## What the audit found

`01_prepare_data.py` checks the raw files before any model sees them:

- **9.07% of rows are exact duplicates** (256,684 of 2,830,743). Left in, they end up
  on both sides of the train/test split, so part of the test set is memorised rather
  than predicted. PortScan alone loses 43% of its rows to deduplication. We removed
  them, and then measured whether it matters: it barely does (see below), but the
  deduplicated set is still the honest one to report.
- `Fwd Header Length` appears twice with identical values.
- 8 features are constant across the whole dataset and carry no information.
- Only `Flow Bytes/s` and `Flow Packets/s` contain missing or infinite values.

After cleaning: **2,574,059 flows, 69 features, 15 classes**, with a
**195,291 : 1** ratio between the largest class (BENIGN) and the smallest (Heartbleed, 11 flows).

## Setup

Tested on Python 3.14.

```bash
pip install -r requirements.txt
```

## Running

```bash
# 1. clean the CSVs into data/cicids2017.parquet and write results/01_data_audit.txt
python 01_prepare_data.py

# 2. list available models
python 02_experiment.py --list

# 3. run models across train/test splits (test fraction)
python 02_experiment.py --models decision_tree,random_forest --splits 0.4,0.3,0.2

# every model that is feasible on the full dataset
python 02_experiment.py --all --splits 0.4

# KNN and RBF-SVM scale superlinearly, so run them on a stratified subsample
python 02_experiment.py --models knn,svm --sample 100000
```

Results accumulate in `results/results.csv`. Each run also writes a full
classification report and confusion matrix to `results/reports/`.

## Method

Every model runs through the same pipeline, so comparisons are like for like:

- median imputation, fitted on the training split only
- standard scaling for scale-sensitive models (linear models, SVM, MLP, Naive Bayes, LDA/QDA); tree models skip it
- stratified train/test split, so every class appears in both halves
- **macro** and weighted precision, recall and F1, plus per-class recall
- training time, prediction time and throughput in flows per second

### Why macro F1

Weighted averages scale each class by its size, so 2.1 million benign flows
drown out 11 Heartbleed flows. A decision tree on the 60/40 split shows the gap:

| Metric | Score |
|---|---|
| Accuracy | 0.9982 |
| Weighted F1 | 0.9982 |
| **Macro F1** | **0.8958** |

The per-class report shows where it fails: XSS at F1 0.45 (confused with web
brute force, which has a near-identical flow profile), Infiltration at recall 0.57.

## Classifier comparison

Every model, on a 400,000-flow stratified sample at a 60/40 split. The full
2.5M-row dataset was also run at test size 0.2 and gives the same ranking;
the sample is used for the headline table so that all three test sizes are
strictly comparable.

| Model | Accuracy | Macro F1 | Fit | Flows/s |
|---|---|---|---|---|
| Decision Tree | 0.9979 | 0.8851 | 19 s | 1,437,230 |
| XGBoost | 0.9984 | 0.8752 | 72 s | 85,748 |
| Random Forest | 0.9980 | 0.8697 | 49 s | 129,916 |
| CatBoost | 0.9985 | 0.8671 | 645 s | 424,167 |
| MLP | 0.9955 | 0.7401 | 99 s | 677,802 |
| Logistic Regression | 0.9747 | 0.6260 | 23 s | 1,081,860 |
| Hist. Gradient Boosting | 0.9859 | 0.6159 | 13 s | 109,611 |
| Naive Bayes | 0.7091 | 0.4759 | 4 s | 170,698 |
| LDA | 0.9080 | 0.4545 | 5 s | 961,398 |
| SGD | 0.9616 | 0.3575 | 10 s | 992,414 |
| QDA | 0.3664 | 0.3108 | 4 s | 47,671 |
| LightGBM | 0.7527 | 0.1240 | 32 s | 48,919 |

- **The plain Decision Tree wins on macro F1** and is the fastest to train and
  to predict with. For an IDS that has to keep up with a live link, that
  combination beats the boosted ensembles, which score no better here.
- **Accuracy cannot rank these models.** The top four sit within 0.0006 of each
  other on accuracy but differ by 0.018 on macro F1, and LightGBM reaches 0.75
  accuracy at 0.12 macro F1.
- **LightGBM collapses with default settings** on all three splits: it stops
  predicting the rare classes at all. Class weighting repairs it completely
  (see below), so the default result is a warning about defaults, not about the
  algorithm.
- Two models were dropped from the sweep on measured cost: AdaBoost (macro F1
  0.165) and linear SVM, which needed 42.7 minutes for one split to reach
  0.579, a score logistic regression nearly matches in 79 seconds. Both remain
  runnable with `--models`.

## Handling the class imbalance

Each strategy is applied to the training split only; the test split keeps the
real class mix, so the scores stay honest.

| Model | No correction | Class weights | Undersample | SMOTE | Under+SMOTE |
|---|---|---|---|---|---|
| Decision Tree | 0.8726 | 0.9044 | 0.8444 | 0.8393 | 0.8446 |
| Random Forest | 0.8630 | 0.8801 | 0.8822 | 0.8671 | 0.8744 |
| LightGBM | 0.1153 | 0.9101 | 0.1435 | 0.9142 | 0.9046 |

**Class weighting is the best and the cheapest fix**: one parameter, no
resampling and no extra training time. It takes Heartbleed recall to 1.00 and,
for the decision tree, Infiltration to 1.00. SMOTE is better only on XSS
(0.51 vs 0.35 for random forest), so the right choice depends on which attack
matters most. SQL injection and XSS stay hard under every strategy, which
matches the feature analysis: their attack lives in the HTTP payload that flow
features cannot see.

## Does the Kernel PCA result hold?

The grid was repeated on three independent samples (`--seed 0/1/2`). The gain
from the best Kernel PCA setting over using all 69 features:

| Classifier | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| SVM (RBF) | +0.073 | +0.158 | +0.064 | **+0.098** |
| KNN | +0.025 | +0.005 | +0.026 | **+0.019** |
| Random Forest | -0.056 | -0.059 | -0.030 | -0.048 |
| Decision Tree | -0.034 | -0.059 | -0.086 | -0.059 |
| Naive Bayes | -0.085 | -0.090 | -0.076 | -0.084 |
| SGD | -0.124 | -0.139 | -0.136 | -0.133 |
| Logistic Regression | -0.152 | -0.173 | -0.142 | -0.156 |

The sign is the same on all three samples for every classifier, so the
conclusion is reproducible: Kernel PCA helps only the two distance-based
classifiers.

### Tuning gamma changes the kernel ranking

The nonlinear kernels were partly handicapped by scikit-learn's default
`gamma`. Averaged macro F1 over the tuning grid:

| Kernel | gamma 0.001 | 0.005 | default | 0.05 | 0.1 |
|---|---|---|---|---|---|
| rbf | **0.512** | 0.482 | 0.452 | 0.430 | 0.408 |
| poly | **0.473** | 0.455 | 0.406 | 0.352 | 0.341 |
| sigmoid | 0.514 | 0.517 | 0.540 | 0.567 | **0.570** |

rbf and poly want a much smaller gamma than the default, sigmoid a larger one.
So "the nonlinear kernels are worse" holds for the defaults but overstates the
gap once gamma is tuned.

## Does the duplicate leakage actually inflate the scores?

A common claim about CIC-IDS2017 is that leaving the duplicate rows in inflates
published results. We tested it rather than assuming it (`10_leakage.py`): keep
the duplicates, split at random, then find which test rows appear verbatim in the
training split and score those rows separately from the genuinely unseen ones.

| Model | Test rows also in training | Accuracy on those rows | Accuracy on unseen rows | Gap |
|---|---|---|---|---|
| Decision Tree | 13.2% | 0.9994 | 0.9983 | +0.0011 |
| Random Forest | 13.2% | 0.9998 | 0.9984 | +0.0014 |

**The inflation is negligible: about +0.0002 on the reported accuracy.** Even with
13.2% of the test set memorisable, the models do essentially as well on rows they
have never seen, because accuracy on unseen rows is already 0.998 and there is no
headroom left for memorisation to add anything.

Two caveats worth stating. First, this is an accuracy result on an easy dataset; it
does not mean duplicate leakage is harmless in general. Second, macro F1 is actually
*lower* with the duplicates kept (0.867 vs 0.896 for the decision tree), because the
duplicates are concentrated in a few classes and skew the balance. Removing them
remains the right thing to do, just not for the reason usually given.

### And do the models lean on "shortcut" features?

Permutation importance ranks Destination Port and the two initial TCP window sizes
highest. Those describe the service and the sending machine's TCP stack rather than
the attack, so we retrained without them (`09_paper_experiments.py`):

| Model | All 69 features | No shortcut features | Change |
|---|---|---|---|
| Decision Tree | 0.8851 | 0.8434 | -0.042 |
| XGBoost | 0.8752 | 0.8373 | -0.038 |
| Random Forest | 0.8697 | 0.8460 | -0.024 |
| Logistic Regression | 0.6260 | 0.6056 | -0.020 |

The models lose only 2 to 4 points of macro F1, so the results are **not** an artefact
of those features. The more useful lesson is about importance scores themselves: the
feature ranked most important can be removed at almost no cost, because the dataset
holds 28 highly correlated feature pairs (8 of them identical) and the signal simply
travels another route.

## Kernel PCA dimensionality reduction

`03_kernel_pca.py` reduces the 69 features with Kernel PCA, then classifies
the reduced data.

| Setting | Values |
|---|---|
| Test size | 0.2, 0.4, 0.6 |
| Kernel | linear, poly, rbf, sigmoid, cosine |
| Components | 5, 10, 15, plus a baseline with no reduction (all 69 features) |
| Classifier | Logistic Regression, SGD, Decision Tree, Random Forest, Naive Bayes, KNN, SVM (RBF) |

That is 7 classifiers x 3 test sizes x (5 kernels x 3 component counts + baseline) = **336 runs**.

Kernel PCA builds an n x n kernel matrix, so on all 2.5M flows it would need
about 18 TB of memory. It runs on one fixed stratified sample of 10,336 flows,
which is then split at each test size. Every class keeps at least 50 rows (or
all of its rows, if it has fewer), otherwise Heartbleed, with 11 flows in the
whole dataset, would vanish from the sample.

Each reduction is computed once per test size and shared by all seven
classifiers, so they are compared on identical features and the same split.

```bash
python 03_kernel_pca.py --classifiers all                # the full 336-run grid
python 03_kernel_pca.py                                  # Logistic Regression only
python 03_kernel_pca.py --classifiers lr,dt,rf --components 10 --sample 20000
```

Results go to `results/kernel_pca/` as CSV and as an Excel workbook with sheets for
accuracy and macro F1 (rows: classifier, components, kernel; columns: test size),
the best setting per classifier, a kernel x classifier summary, and every run.

### Best Kernel PCA setting per classifier (macro F1)

Compared with the same classifier on all 69 features at the same test size.

| Classifier | No reduction | Best Kernel PCA | Setting (kernel, components, test size) | Change |
|---|---|---|---|---|
| Random Forest | 0.864 | 0.811 | linear, 10, 0.4 | -0.053 |
| Decision Tree | 0.829 | 0.795 | linear, 10, 0.2 | -0.034 |
| KNN | 0.742 | **0.767** | sigmoid, 15, 0.2 | **+0.025** |
| Naive Bayes | 0.619 | 0.581 | sigmoid, 15, 0.4 | -0.038 |
| SVM (RBF) | 0.511 | **0.585** | sigmoid, 15, 0.2 | **+0.073** |
| Logistic Regression | 0.645 | 0.493 | linear, 15, 0.2 | -0.152 |
| SGD | 0.519 | 0.397 | linear, 10, 0.4 | -0.122 |

### Which kernel suits which classifier (macro F1, averaged over test sizes and component counts)

| Classifier | linear | poly | rbf | sigmoid | cosine |
|---|---|---|---|---|---|
| Random Forest | **0.790** | 0.678 | 0.748 | 0.772 | 0.776 |
| Decision Tree | **0.751** | 0.625 | 0.700 | 0.721 | 0.744 |
| KNN | 0.727 | 0.633 | 0.684 | **0.743** | 0.722 |
| Naive Bayes | 0.398 | 0.249 | 0.371 | **0.512** | 0.476 |
| SVM (RBF) | 0.406 | 0.141 | 0.231 | **0.480** | 0.330 |
| Logistic Regression | **0.397** | 0.201 | 0.180 | 0.198 | 0.209 |
| SGD | **0.328** | 0.177 | 0.174 | 0.188 | 0.196 |

### Findings

- **Random Forest is the strongest classifier**, with or without reduction. With linear
  Kernel PCA it keeps 0.811 macro F1 and 0.980 accuracy from 10 components, a 7x smaller
  feature set than the original 69.
- **Kernel PCA helps only the distance-based classifiers.** KNN gains 0.025 and SVM gains
  0.073 with the sigmoid kernel at 15 components. Both depend on distances between flows,
  and in the full 69-feature space many correlated flow statistics (packet-length and
  inter-arrival summaries of the same traffic) weigh on those distances. A compact
  projection is a plausible reason they improve.
- **Linear models lose the most** (Logistic Regression -0.152, SGD -0.122). With a handful
  of components they cannot separate the rare attack classes.
- **The best kernel depends on the classifier.** Linear is best for Logistic Regression,
  SGD and both tree models; sigmoid is best for Naive Bayes, KNN and SVM. **Poly is the
  worst kernel for all seven.**
- **More components help every classifier**: each one improves from 5 to 10 to 15.
- **Test size matters little** for most classifiers. KNN and SVM lose the most when
  training data shrinks to 40% (test size 0.6).
- Accuracy stays between 0.88 and 0.98 for most settings while macro F1 ranges from
  0.14 to 0.87, so macro F1 is the measure that separates the settings.
- The sigmoid kernel is not positive semi-definite, so its kernel matrix has negative
  eigenvalues. The `arpack` eigensolver is used because it keeps the largest positive ones.

Caveats: one sample and one split per test size, so differences of about 0.01 are within
noise; kernel parameters (`gamma`, `degree`, `coef0`) and classifier settings are defaults;
and the per-class minimum makes rare attacks far more common in the sample than in real traffic.

## Layout

```
01_prepare_data.py       raw CSVs -> cleaned Parquet + audit report
02_experiment.py         model registry, pipeline, metrics, experiment runner
03_kernel_pca.py         Kernel PCA dimensionality reduction + classification grid
04_feature_importance.py which flow features identify each attack
05_imbalance.py          class-weight / undersample / SMOTE study
06_charts.py             renders the report figures from the results files
07_report.py             assembles results/ACN_IDS_Report.docx from the results
08_slides.py             assembles results/ACN_IDS_Slides.pptx from the results
common.py                shared stratified sampler
plot_style.py            shared chart style
SGDC_60_40.py            original single-model baseline
docs/                    networking background (what the features and attacks mean)
results/                 audit report, results tables, per-run reports, charts/
data/                    generated Parquet (not committed)
WORK_SPLIT.md            who owns which part
```

## Feature analysis and class imbalance

```bash
# which features separate each attack from benign traffic
python 04_feature_importance.py

# does class weighting / undersampling / SMOTE recover the rare attacks?
python 05_imbalance.py                 # full dataset (slow); --sample 300000 for a quick look

# render every report chart from whatever results exist so far
python 06_charts.py
```

`docs/networking_background.md` explains the flow features and every attack in
protocol terms, and reads the numbers straight from `results/features/`.

## Roadmap

- [x] Data audit and cleaning
- [x] Experiment framework
- [x] Kernel PCA grid with Logistic Regression (5 kernels x 3 test sizes x 3 component counts)
- [x] Kernel PCA grid with all 7 classifiers (336 runs)
- [x] Full algorithm grid across 80/20, 60/40, 40/60 splits
- [x] Comparison plots and confusion-matrix heatmaps
- [x] Class-imbalance study: class weights, undersampling, SMOTE on the training split only
- [x] Feature importance explained in TCP and flow terms
- [x] Write-up (results/ACN_IDS_Report.docx, results/ACN_IDS_Slides.pptx)
