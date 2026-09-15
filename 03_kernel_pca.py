"""
Step 3: Kernel PCA dimensionality reduction, then classification.

Grid from the assignment sheet:
    test size   0.2, 0.4, 0.6
    kernel      linear, poly, rbf, sigmoid, cosine
    components  all 69 features (the no-reduction baseline, "72C" on the sheet), 5, 10, 15
    classifier  LR, SGD, DT, RF, NB, KNN, SVM

Kernel PCA builds an n x n kernel matrix, so it cannot run on all 2.5M
flows (that matrix would need ~18 TB). It runs on one fixed stratified
sample, which is then split at each test size, exactly like the
"1000 rows -> 200 / 400 / 600" example.

Each (test size, kernel, components) reduction is computed once and shared
by every classifier, so all classifiers see identical features and the same
train/test split. Imputation, scaling and Kernel PCA are fitted on the
training split only.

Usage:
    python 03_kernel_pca.py                                   # LR only
    python 03_kernel_pca.py --classifiers all                 # all 7 classifiers
    python 03_kernel_pca.py --classifiers lr,dt,rf --components 10 --sample 20000
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


def reduce(kernel, n_comp, Xtr, Xte):
    """Fit impute -> scale -> Kernel PCA on the training split, transform both.
    n_comp = 0 skips Kernel PCA: the classifiers see all features."""
    steps = [("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]
    if n_comp:
        # arpack keeps the largest positive eigenvalues. The randomized solver
        # ranks them by magnitude, so with the sigmoid kernel (which is not
        # positive semi-definite) it picks large negative ones and fails.
        steps.append(("kpca", KernelPCA(n_components=n_comp, kernel=kernel,
                                        eigen_solver="arpack", random_state=SEED,
                                        n_jobs=-1)))
    pre = Pipeline(steps)
    return pre.fit_transform(Xtr), pre.transform(Xte)


def classify(clf_key, Ztr, ytr, Zte, yte):
    code, make = CLASSIFIERS[clf_key]
    t0 = time.time()
    pred = make().fit(Ztr, ytr).predict(Zte)
    return code, {
        "accuracy": round(accuracy_score(yte, pred), 4),
        "macro_f1": round(f1_score(yte, pred, average="macro", zero_division=0), 4),
        "clf_seconds": round(time.time() - t0, 1),
    }


def write_excel(df, path, clf_order):
    """Laid out like the table on the sheet: rows = classifier, components,
    kernel; columns = test size. Components 'all' is the no-reduction baseline."""
    korder = {k: i for i, k in enumerate(["none"] + KERNELS)}
    corder = {c: i for i, c in enumerate(clf_order)}
    d = (df.assign(_k=df["kernel"].map(korder), _c=df["classifier"].map(corder))
           .sort_values(["_c", "components", "_k", "test_size"]))
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for metric in ("accuracy", "macro_f1"):
            piv = d.pivot_table(index=["classifier", "components", "kernel"],
                                columns="test_size", values=metric, sort=False)
            piv.columns = [f"test {c}" for c in piv.columns]
            piv = piv.rename(index={0: "all"}, level="components")
            piv.to_excel(xl, sheet_name=metric)

        # best Kernel PCA setting per classifier, next to its no-reduction baseline
        best = []
        for clf, g in d.groupby("classifier", sort=False):
            base = g[g["components"] == 0]
            red = g[g["components"] > 0]
            top = red.loc[red["macro_f1"].idxmax()]
            best.append({
                "classifier": clf,
                "baseline macro_f1 (mean of 3 tests)": round(base["macro_f1"].mean(), 4),
                "best kernel": top["kernel"],
                "best components": int(top["components"]),
                "best test size": top["test_size"],
                "best macro_f1": top["macro_f1"],
                "best accuracy": top["accuracy"],
            })
        pd.DataFrame(best).to_excel(xl, sheet_name="best per classifier", index=False)

        # which kernel suits which classifier: macro F1 averaged over tests and components
        rank = (d[d["components"] > 0]
                .pivot_table(index="classifier", columns="kernel", values="macro_f1",
                             aggfunc="mean", sort=False)
                .reindex(columns=KERNELS).round(4))
        rank.to_excel(xl, sheet_name="kernel x classifier")

        # the long list from the bottom of the sheet, one row per experiment
        d.drop(columns=["_k", "_c"]).to_excel(xl, sheet_name="all runs", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifiers", default="lr", help="comma list, or 'all'")
    ap.add_argument("--kernels", default=",".join(KERNELS))
    ap.add_argument("--tests", default="0.2,0.4,0.6")
    ap.add_argument("--components", default="0,5,10,15",
                    help="0 = no Kernel PCA (all features, the baseline)")
    ap.add_argument("--sample", type=int, default=10000)
    ap.add_argument("--floor", type=int, default=50,
                    help="minimum rows kept per class in the sample")
    a = ap.parse_args()

    clfs = (list(CLASSIFIERS) if a.classifiers == "all"
            else [c.strip() for c in a.classifiers.split(",")])
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
    n_red = len(tests) * (len(kernels) * len(reduced) + (0 in comps))
    print(f"reductions: {n_red}   classifiers: {len(clfs)}   runs: {n_red * len(clfs)}\n")
    print(f"{'#':>4}  {'test':>4}  {'kernel':<8}{'comp':>4}  {'clf':<5}"
          f"{'acc':>7}  {'macroF1':>7}  {'sec':>5}")

    rows, i = [], 0
    for ts in tests:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=ts, random_state=SEED, stratify=y)
        for comp in comps:
            for kernel in (kernels if comp else ["none"]):
                shown = comp or "all"
                t0 = time.time()
                try:
                    Ztr, Zte = reduce(kernel, comp, Xtr, Xte)
                except Exception as e:
                    i += len(clfs)
                    print(f"      {ts:>4}  {kernel:<8}{shown:>4}  "
                          f"REDUCTION FAILED {type(e).__name__}: {e}")
                    continue
                red_s = round(time.time() - t0, 1)
                for clf in clfs:
                    i += 1
                    try:
                        code, r = classify(clf, Ztr, ytr, Zte, yte)
                    except Exception as e:
                        print(f"{i:>4}  {ts:>4}  {kernel:<8}{shown:>4}  {clf:<5}"
                              f"FAILED {type(e).__name__}: {e}")
                        continue
                    rows.append({"test_size": ts, "kernel": kernel, "classifier": code,
                                 "components": comp, "n_train": len(ytr), "n_test": len(yte),
                                 **r, "kpca_seconds": red_s})
                    print(f"{i:>4}  {ts:>4}  {kernel:<8}{shown:>4}  {code:<5}"
                          f"{r['accuracy']:>7.4f}  {r['macro_f1']:>7.4f}  {r['clf_seconds']:>5}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    tag = f"{'all' if a.classifiers == 'all' else '_'.join(clfs)}_s{a.sample}"
    out.to_csv(RESULTS / f"kpca_{tag}.csv", index=False)
    write_excel(out, RESULTS / f"kpca_{tag}.xlsx", [CLASSIFIERS[c][0] for c in clfs])
    print(f"\nwrote results/kernel_pca/kpca_{tag}.xlsx")


if __name__ == "__main__":
    main()
