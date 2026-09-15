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
  on both sides of the train/test split and inflate every score. PortScan alone
  loses 43% of its rows to deduplication.
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
01_prepare_data.py     raw CSVs -> cleaned Parquet + audit report
02_experiment.py       model registry, pipeline, metrics, experiment runner
03_kernel_pca.py       Kernel PCA dimensionality reduction + classification grid
SGDC_60_40.py          original single-model baseline
results/               audit report, results tables, per-run reports
data/                  generated Parquet (not committed)
```

## Roadmap

- [x] Data audit and cleaning
- [x] Experiment framework
- [x] Kernel PCA grid with Logistic Regression (5 kernels x 3 test sizes x 3 component counts)
- [x] Kernel PCA grid with all 7 classifiers (336 runs)
- [ ] Full algorithm grid across 80/20, 60/40, 40/60 splits
- [ ] Comparison plots and confusion-matrix heatmaps
- [ ] Class-imbalance study: class weights, undersampling, SMOTE on the training split only
- [ ] Feature importance explained in TCP and flow terms
- [ ] Write-up
