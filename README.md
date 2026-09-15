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
2. Extract the eight `*.pcap_ISCX.csv` files into the project root, next to the scripts.

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
| Classifier | Logistic Regression (others selectable) |

Kernel PCA builds an n x n kernel matrix, so on all 2.5M flows it would need
about 18 TB of memory. It runs on one fixed stratified sample of about 10,000
flows, which is then split at each test size. Every class keeps at least 50
rows (or all of its rows, if it has fewer), otherwise Heartbleed, with 11
flows in the whole dataset, would vanish from the sample.

```bash
python 03_kernel_pca.py                                  # the full grid above
python 03_kernel_pca.py --classifiers lr,dt,rf --sample 20000
```

Results go to `results/kernel_pca/` as CSV and as an Excel workbook with
accuracy and macro F1 sheets (rows: classifier, components, kernel; columns: test size).

### Results with Logistic Regression (macro F1, test size 0.2)

| Kernel | 5 components | 10 components | 15 components |
|---|---|---|---|
| none (all 69 features) | 0.645 | 0.645 | 0.645 |
| linear | 0.279 | 0.450 | 0.493 |
| sigmoid | 0.156 | 0.206 | 0.297 |
| poly | 0.154 | 0.142 | 0.277 |
| cosine | 0.194 | 0.221 | 0.229 |
| rbf | 0.157 | 0.201 | 0.214 |

- No Kernel PCA setting beats the unreduced baseline (0.956 accuracy, 0.645 macro F1).
  The leading components mostly describe benign traffic, and the detail that separates
  rare attacks is discarded.
- The linear kernel is best at every component count, and more components help almost everywhere.
- Test size changes scores by only 0.01 to 0.04.
- Poly with 5 components reaches 0.808 accuracy, which is the share of benign flows in the
  sample (0.8075): it labels everything benign. Its macro F1 of 0.15 exposes this.
- The sigmoid kernel is not positive semi-definite, so its kernel matrix has negative
  eigenvalues. The `arpack` eigensolver is used because it keeps the largest positive ones.

Caveats: kernel parameters (`gamma`, `degree`, `coef0`) are left at their defaults, and the
per-class minimum makes rare attacks far more common in the sample than in real traffic.

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
- [ ] Kernel PCA grid with the remaining classifiers
- [ ] Full algorithm grid across 80/20, 60/40, 40/60 splits
- [ ] Comparison plots and confusion-matrix heatmaps
- [ ] Class-imbalance study: class weights, undersampling, SMOTE on the training split only
- [ ] Feature importance explained in TCP and flow terms
- [ ] Write-up
