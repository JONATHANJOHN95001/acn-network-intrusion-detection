"""
Step 3: Kernel PCA dimensionality reduction, then classification.

Grid from the assignment sheet:
    test size   0.2, 0.4, 0.6
    kernel      linear, poly, rbf, sigmoid, cosine
    classifier  LR (more can be added below)
    components  all 69 features (the no-reduction baseline, "72C" on the sheet), 5, 10, 15

Kernel PCA builds an n x n kernel matrix, so it cannot run on all 2.5M
flows (that matrix would need ~18 TB). It runs on one fixed stratified
sample, which is then split at each test size, exactly like the
"1000 rows -> 200 / 400 / 600" example.

Every step (impute, scale, Kernel PCA, classifier) is fitted on the
training split only.

Usage:
    python 03_kernel_pca.py                                   # LR, 5 kernels, 3 tests, baseline + 5/10/15
    python 03_kernel_pca.py --components 10
    python 03_kernel_pca.py --classifiers lr,dt,rf --sample 20000
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
PARQUET = HERE / "data" / "cicids2017.parquet"
RESULTS = HERE / "results" / "kernel_pca"
SEED = 0

KERNELS = ["linear", "poly", "rbf", "sigmoid", "cosine"]

# Short codes match the sheet ("LR"). On a sample this size, even SVM and
# KNN are cheap, so every classifier from the PDF is usable here.
CLASSIFIERS = {
    "lr":  ("LR",  lambda: LogisticRegression(max_iter=2000, random_state=SEED)),
    "sgd": ("SGD", lambda: SGDClassifier(random_state=SEED)),
    "dt":  ("DT",  lambda: DecisionTreeClassifier(criterion="entropy", random_state=SEED)),
    "rf":  ("RF",  lambda: RandomForestClassifier(n_estimators=200, criterion="entropy",
                                                  n_jobs=-1, random_state=SEED)),
    "nb":  ("NB",  lambda: GaussianNB()),
    "knn": ("KNN", lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    "svm": ("SVM", lambda: SVC(kernel="rbf", random_state=SEED)),
}


def load_sample(n, floor):
    """Proportional per-class sample, but keep at least `floor` rows of every
    class (or all of it, if the class is smaller). Without the floor,
    Heartbleed (11 rows in 2.5M) would round down to zero.

    The rows are chosen from the Label column alone, then the Parquet file is
    streamed in batches keeping only those rows. Loading all 2.5M rows at
    once needs about 1 GB, which a laptop with a browser open may not have."""
    import pyarrow.parquet as pq

    labels = pd.read_parquet(PARQUET, columns=["Label"])["Label"]
    frac = n / len(labels)
    rng = np.random.default_rng(SEED)
    keep = []
    for idx in labels.groupby(labels).indices.values():
        k = max(int(round(len(idx) * frac)), min(floor, len(idx)))
        keep.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    del labels

    parts, start = [], 0
    for batch in pq.ParquetFile(PARQUET).iter_batches(batch_size=100_000):
        end = start + batch.num_rows
        lo, hi = np.searchsorted(keep, [start, end])
        if hi > lo:
            parts.append(batch.take(keep[lo:hi] - start).to_pandas())
        start = end
    df = pd.concat(parts, ignore_index=True)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def run(clf_key, kernel, n_comp, test_size, X, y):
    """n_comp = 0 skips Kernel PCA: the classifier sees all features."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y)
    code, make = CLASSIFIERS[clf_key]
    steps = [("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]
    if n_comp:
        # arpack keeps the largest positive eigenvalues. The randomized solver
        # ranks them by magnitude, so with the sigmoid kernel (which is not
        # positive semi-definite) it picks large negative ones and fails.
        steps.append(("kpca", KernelPCA(n_components=n_comp, kernel=kernel,
                                        eigen_solver="arpack", random_state=SEED,
                                        n_jobs=-1)))
    steps.append(("clf", make()))
    pipe = Pipeline(steps)

    t0 = time.time()
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    return {
        "test_size": test_size,
        "kernel": kernel,
        "classifier": code,
        "components": n_comp,
        "n_train": len(ytr),
        "n_test": len(yte),
        "accuracy": round(accuracy_score(yte, pred), 4),
        "macro_f1": round(f1_score(yte, pred, average="macro", zero_division=0), 4),
        "seconds": round(time.time() - t0, 1),
    }


def write_excel(df, path):
    """Laid out like the table on the sheet: rows = classifier, components,
    kernel; columns = test size. Components 'all' is the no-reduction baseline."""
    order = {k: i for i, k in enumerate(["none"] + KERNELS)}
    d = (df.assign(_k=df["kernel"].map(order))
           .sort_values(["classifier", "components", "_k", "test_size"]))
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for metric in ("accuracy", "macro_f1"):
            piv = d.pivot_table(index=["classifier", "components", "kernel"],
                                columns="test_size", values=metric, sort=False)
            piv.columns = [f"test {c}" for c in piv.columns]
            piv = piv.rename(index={0: "all"}, level="components")
            piv.to_excel(xl, sheet_name=metric)
        # the long list from the bottom of the sheet, one row per experiment
        d.drop(columns="_k").to_excel(xl, sheet_name="all runs", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifiers", default="lr")
    ap.add_argument("--kernels", default=",".join(KERNELS))
    ap.add_argument("--tests", default="0.2,0.4,0.6")
    ap.add_argument("--components", default="0,5,10,15",
                    help="0 = no Kernel PCA (all features, the baseline)")
    ap.add_argument("--sample", type=int, default=10000)
    ap.add_argument("--floor", type=int, default=50,
                    help="minimum rows kept per class in the sample")
    a = ap.parse_args()

    clfs = [c.strip() for c in a.classifiers.split(",")]
    kernels = [k.strip() for k in a.kernels.split(",")]
    tests = [float(t) for t in a.tests.split(",")]
    comps = [int(c) for c in a.components.split(",")]
    bad = [c for c in clfs if c not in CLASSIFIERS] + [k for k in kernels if k not in KERNELS]
    if bad:
        raise SystemExit(f"unknown: {bad}")

    df = load_sample(a.sample, a.floor)
    X = df.drop(columns=["Label", "Day"])
    y = LabelEncoder().fit_transform(df["Label"])
    print(f"sample: {len(df):,} rows x {X.shape[1]} features, {df['Label'].nunique()} classes")

    reduced = [c for c in comps if c]
    total = len(clfs) * len(tests) * (len(kernels) * len(reduced) + (0 in comps))
    print(f"runs:   {total}  ({len(clfs)} classifier x {len(tests)} test sizes x "
          f"({len(kernels)} kernels x {len(reduced)} component counts"
          f"{' + baseline' if 0 in comps else ''}))\n")
    print(f"{'#':>3}  {'test':>4}  {'kernel':<8}{'clf':<5}{'comp':>4}  "
          f"{'acc':>7}  {'macroF1':>7}  {'sec':>6}")

    rows, i = [], 0
    for clf in clfs:
        for comp in comps:
            for kernel in (kernels if comp else ["none"]):
                for ts in tests:
                    i += 1
                    shown = comp or "all"
                    try:
                        r = run(clf, kernel, comp, ts, X, y)
                        rows.append(r)
                        print(f"{i:>3}  {ts:>4}  {kernel:<8}{r['classifier']:<5}{shown:>4}  "
                              f"{r['accuracy']:>7.4f}  {r['macro_f1']:>7.4f}  {r['seconds']:>6}")
                    except Exception as e:
                        print(f"{i:>3}  {ts:>4}  {kernel:<8}{clf:<5}{shown:>4}  "
                              f"FAILED {type(e).__name__}: {e}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    tag = f"{'_'.join(clfs)}_s{a.sample}"
    out.to_csv(RESULTS / f"kpca_{tag}.csv", index=False)
    write_excel(out, RESULTS / f"kpca_{tag}.xlsx")
    print(f"\nwrote {RESULTS / f'kpca_{tag}.xlsx'}")


if __name__ == "__main__":
    main()
