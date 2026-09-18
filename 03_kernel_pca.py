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

Each (test size, kernel, components, gamma) reduction is computed once and
shared by every classifier, so all classifiers see identical features and
the same train/test split. Imputation, scaling and Kernel PCA are fitted on
the training split only.

Usage:
    python 03_kernel_pca.py                                   # LR only
    python 03_kernel_pca.py --classifiers all                 # all 7 classifiers
    python 03_kernel_pca.py --classifiers all --seed 1        # same grid, different sample
    python 03_kernel_pca.py --classifiers all --kernels rbf,poly,sigmoid \
        --components 10,15 --gammas 0.001,0.005,default,0.05,0.1      # tune gamma
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import load_sample

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
RESULTS = HERE / "results" / "kernel_pca"
SEED = 0

# Half the cores at most: saturating all of them for hours destabilised
# this machine (video memory manager bugcheck).
N_JOBS = max(1, (os.cpu_count() or 2) // 2 - 2)

KERNELS = ["linear", "poly", "rbf", "sigmoid", "cosine"]
GAMMA_KERNELS = ("poly", "rbf", "sigmoid")   # linear and cosine have no gamma

# Short codes match the sheet ("LR"). On a sample this size, even SVM and
# KNN are cheap, so every classifier from the PDF is usable here.
CLASSIFIERS = {
    "lr":  ("LR",  lambda: LogisticRegression(max_iter=2000, random_state=SEED)),
    "sgd": ("SGD", lambda: SGDClassifier(random_state=SEED)),
    "dt":  ("DT",  lambda: DecisionTreeClassifier(criterion="entropy", random_state=SEED)),
    "rf":  ("RF",  lambda: RandomForestClassifier(n_estimators=200, criterion="entropy",
                                                  n_jobs=N_JOBS, random_state=SEED)),
    "nb":  ("NB",  lambda: GaussianNB()),
    "knn": ("KNN", lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=N_JOBS)),
    "svm": ("SVM", lambda: SVC(kernel="rbf", random_state=SEED)),
}


def gamma_label(kernel, gamma):
    if kernel not in GAMMA_KERNELS:
        return "-"
    return "default" if gamma is None else f"{gamma:g}"


def reduce(kernel, n_comp, gamma, Xtr, Xte):
    """Fit impute -> scale -> Kernel PCA on the training split, transform both.
    n_comp = 0 skips Kernel PCA: the classifiers see all features.
    gamma = None uses scikit-learn's default, 1 / number of features."""
    steps = [("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]
    if n_comp:
        # arpack keeps the largest positive eigenvalues. The randomized solver
        # ranks them by magnitude, so with the sigmoid kernel (which is not
        # positive semi-definite) it picks large negative ones and fails.
        steps.append(("kpca", KernelPCA(n_components=n_comp, kernel=kernel, gamma=gamma,
                                        eigen_solver="arpack", random_state=SEED,
                                        n_jobs=N_JOBS)))
    pre = Pipeline(steps)
    return pre.fit_transform(Xtr), pre.transform(Xte)


def classify(clf_key, Ztr, ytr, Zte, yte):
    code, make = CLASSIFIERS[clf_key]
    t0 = time.time()
    pred = make().fit(Ztr, ytr).predict(Zte)
    return code, {
        "accuracy": round(accuracy_score(yte, pred), 4),
        # macro averages weight all fifteen classes equally, so a model that
        # ignores the rare attacks cannot hide behind the benign majority
        "macro_precision": round(precision_score(yte, pred, average="macro",
                                                 zero_division=0), 4),
        "macro_recall": round(recall_score(yte, pred, average="macro",
                                           zero_division=0), 4),
        "macro_f1": round(f1_score(yte, pred, average="macro", zero_division=0), 4),
        "clf_seconds": round(time.time() - t0, 1),
    }


def write_excel(df, path, clf_order):
    """Laid out like the table on the sheet: rows = classifier, components,
    kernel (and gamma, when several were tried); columns = test size.
    Components 'all' is the no-reduction baseline."""
    korder = {k: i for i, k in enumerate(["none"] + KERNELS)}
    corder = {c: i for i, c in enumerate(clf_order)}
    d = (df.assign(_k=df["kernel"].map(korder), _c=df["classifier"].map(corder),
                   _g=df["gamma"].fillna(-1))
           .sort_values(["_c", "components", "_k", "_g", "test_size"]))
    tuned = d.loc[d["gamma_label"] != "-", "gamma_label"].nunique() > 1
    index = ["classifier", "components", "kernel"] + (["gamma_label"] if tuned else [])

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            if metric not in d.columns:      # older result files predate these
                continue
            piv = d.pivot_table(index=index, columns="test_size", values=metric, sort=False)
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
                "baseline macro_f1 (mean of tests)": (round(base["macro_f1"].mean(), 4)
                                                      if len(base) else None),
                "best kernel": top["kernel"],
                "best gamma": top["gamma_label"],
                "best components": int(top["components"]),
                "best test size": top["test_size"],
                "best macro_f1": top["macro_f1"],
                "best accuracy": top["accuracy"],
            })
        pd.DataFrame(best).to_excel(xl, sheet_name="best per classifier", index=False)

        # which kernel suits which classifier: macro F1 averaged over everything else
        rank = (d[d["components"] > 0]
                .pivot_table(index="classifier", columns="kernel", values="macro_f1",
                             aggfunc="mean", sort=False)
                .reindex(columns=[k for k in KERNELS if k in set(d["kernel"])]).round(4))
        rank.to_excel(xl, sheet_name="kernel x classifier")

        if tuned:
            gd = d[d["gamma_label"] != "-"]
            cols = list(gd.drop_duplicates("gamma_label").sort_values("_g")["gamma_label"])
            geff = (gd.pivot_table(index=["classifier", "kernel"], columns="gamma_label",
                                   values="macro_f1", aggfunc="mean", sort=False)
                      .reindex(columns=cols).round(4))
            geff.to_excel(xl, sheet_name="gamma effect")

        # the long list from the bottom of the sheet, one row per experiment
        d.drop(columns=["_k", "_c", "_g"]).to_excel(xl, sheet_name="all runs", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifiers", default="lr", help="comma list, or 'all'")
    ap.add_argument("--kernels", default=",".join(KERNELS))
    ap.add_argument("--tests", default="0.2,0.4,0.6")
    ap.add_argument("--components", default="0,5,10,15",
                    help="0 = no Kernel PCA (all features, the baseline)")
    ap.add_argument("--gammas", default="default",
                    help="comma list for poly/rbf/sigmoid; 'default' = 1 / number of features")
    ap.add_argument("--sample", type=int, default=10000)
    ap.add_argument("--floor", type=int, default=50,
                    help="minimum rows kept per class in the sample")
    ap.add_argument("--seed", type=int, default=0,
                    help="which random sample to draw (the split itself stays fixed)")
    a = ap.parse_args()

    clfs = (list(CLASSIFIERS) if a.classifiers == "all"
            else [c.strip() for c in a.classifiers.split(",")])
    kernels = [k.strip() for k in a.kernels.split(",")]
    tests = [float(t) for t in a.tests.split(",")]
    comps = [int(c) for c in a.components.split(",")]
    gammas = [None if g.strip() == "default" else float(g) for g in a.gammas.split(",")]
    bad = [c for c in clfs if c not in CLASSIFIERS] + [k for k in kernels if k not in KERNELS]
    if bad:
        raise SystemExit(f"unknown: {bad}")

    df = load_sample(a.sample, a.floor, a.seed)
    X = df.drop(columns=["Label", "Day"])
    y = LabelEncoder().fit_transform(df["Label"])
    default_gamma = 1 / X.shape[1]
    print(f"sample: {len(df):,} rows x {X.shape[1]} features, {df['Label'].nunique()} classes"
          f"  (seed {a.seed})")

    plan = [(comp, kernel, g)
            for comp in comps
            for kernel in (kernels if comp else ["none"])
            for g in (gammas if comp and kernel in GAMMA_KERNELS else [None])]
    n_red = len(tests) * len(plan)
    print(f"reductions: {n_red}   classifiers: {len(clfs)}   runs: {n_red * len(clfs)}\n")
    print(f"{'#':>4}  {'test':>4}  {'kernel':<8}{'gamma':<8}{'comp':>4}  {'clf':<5}"
          f"{'acc':>7}  {'macroF1':>7}  {'sec':>5}")

    rows, i = [], 0
    for ts in tests:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=ts, random_state=SEED, stratify=y)
        for comp, kernel, g in plan:
            shown, glab = comp or "all", gamma_label(kernel, g)
            t0 = time.time()
            try:
                Ztr, Zte = reduce(kernel, comp, g, Xtr, Xte)
            except Exception as e:
                i += len(clfs)
                print(f"      {ts:>4}  {kernel:<8}{glab:<8}{shown:>4}  "
                      f"REDUCTION FAILED {type(e).__name__}: {e}")
                continue
            red_s = round(time.time() - t0, 1)
            for clf in clfs:
                i += 1
                try:
                    code, r = classify(clf, Ztr, ytr, Zte, yte)
                except Exception as e:
                    print(f"{i:>4}  {ts:>4}  {kernel:<8}{glab:<8}{shown:>4}  {clf:<5}"
                          f"FAILED {type(e).__name__}: {e}")
                    continue
                gval = (default_gamma if g is None else g) if kernel in GAMMA_KERNELS else np.nan
                rows.append({"test_size": ts, "kernel": kernel, "gamma_label": glab,
                             "gamma": gval, "classifier": code, "components": comp,
                             "sample_seed": a.seed, "n_train": len(ytr), "n_test": len(yte),
                             **r, "kpca_seconds": red_s})
                print(f"{i:>4}  {ts:>4}  {kernel:<8}{glab:<8}{shown:>4}  {code:<5}"
                      f"{r['accuracy']:>7.4f}  {r['macro_f1']:>7.4f}  {r['clf_seconds']:>5}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    tag = f"{'all' if a.classifiers == 'all' else '_'.join(clfs)}_s{a.sample}"
    if a.seed:
        tag += f"_seed{a.seed}"
    if a.gammas != "default":
        tag += "_gamma"
    out.to_csv(RESULTS / f"kpca_{tag}.csv", index=False)
    write_excel(out, RESULTS / f"kpca_{tag}.xlsx", [CLASSIFIERS[c][0] for c in clfs])
    print(f"\nwrote results/kernel_pca/kpca_{tag}.xlsx")


if __name__ == "__main__":
    main()
