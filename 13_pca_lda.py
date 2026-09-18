"""
Step 13: the other two dimensionality reduction methods from the code sheet.

The 75 scripts in runs/ all use Kernel PCA. The sheet covers three methods,
so this runs the remaining two over the same grid and the same sample, which
makes the three directly comparable:

    2 methods (PCA, LDA) x 5 classifiers x 3 test sizes = 30 runs

Everything else is held identical to the 75 runs: same 10,000-flow stratified
sample, same seed, same mean imputation and scaling fitted on the training
split only, same macro-averaged metrics.

One difference is forced by the method itself. LDA projects onto at most
n_classes - 1 directions, so with 15 classes it can return 14 components, not
15. That is a property of LDA, not a setting, and it is recorded in the output
rather than worked around.

    python 13_pca_lda.py                 run the grid, write the sheet
    python 13_pca_lda.py --sheet-only    rebuild the sheet from the CSV
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from common import load_sample

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
RESULTS = HERE / "results"
CSV = RESULTS / "pca_lda.csv"
XLSX = RESULTS / "pca_lda.xlsx"

SEED = 0
N_JOBS = max(1, (os.cpu_count() or 2) // 2)
N_COMPONENTS = 15          # same as the 75 runs; LDA caps itself below this
SAMPLE = 10_000
FLOOR = 50
TESTS = [0.2, 0.4, 0.6]

# See 11_make_75_scripts.py: the rarest class supplies fewer training rows than
# there are components, so QDA needs shrinkage, and the rank test needs a
# tolerance matched to the scale of the components rather than sklearn's
# absolute default of 1e-4.
RANK_TOL = 1e-12


def qda_shrinkage(X, y):
    """Smallest shrinkage that lets every class covariance be estimated."""
    last = None
    for s in (0.1, 0.25, 0.5, 0.8, 0.95):
        try:
            QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=s,
                                          tol=RANK_TOL).fit(X, y)
        except Exception as exc:
            last = exc
            continue
        return s
    raise SystemExit(f"QDA cannot be estimated at any shrinkage: {last}")


CLASSIFIERS = {
    "LR": lambda Xtr, ytr: LogisticRegression(max_iter=2000, random_state=SEED),
    "RF": lambda Xtr, ytr: RandomForestClassifier(n_estimators=200, criterion="entropy",
                                                  n_jobs=N_JOBS, random_state=SEED),
    "KNN": lambda Xtr, ytr: KNeighborsClassifier(n_neighbors=5, n_jobs=N_JOBS),
    "SVM": lambda Xtr, ytr: SVC(kernel="rbf", probability=True, random_state=SEED),
    "QDA": lambda Xtr, ytr: QuadraticDiscriminantAnalysis(
        solver="eigen", tol=RANK_TOL, shrinkage=qda_shrinkage(Xtr, ytr)),
}


def reduce(method, Xtr, Xte, ytr):
    """Fit the reduction on the training split only, then apply it to both."""
    if method == "PCA":
        dr = PCA(n_components=N_COMPONENTS, random_state=SEED)
        return dr.fit_transform(Xtr), dr.transform(Xte)
    # LDA is supervised, so it sees the training labels. It can return at most
    # one direction fewer than there are classes.
    k = min(N_COMPONENTS, len(set(ytr)) - 1)
    dr = LinearDiscriminantAnalysis(n_components=k)
    return dr.fit_transform(Xtr, ytr), dr.transform(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-only", action="store_true")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    if not a.sheet_only:
        df = load_sample(SAMPLE, FLOOR, SEED)
        y = LabelEncoder().fit_transform(df["Label"])
        X = df.drop(columns=["Label", "Day"])
        print(f"{len(df):,} flows, {X.shape[1]} features, {len(set(y))} classes\n")

        rows = []
        for test in TESTS:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=test, random_state=SEED, stratify=y)
            imputer = SimpleImputer(strategy="mean").fit(Xtr)
            scaler = StandardScaler().fit(imputer.transform(Xtr))
            Atr = scaler.transform(imputer.transform(Xtr))
            Ate = scaler.transform(imputer.transform(Xte))

            for method in ("PCA", "LDA"):
                Ztr, Zte = reduce(method, Atr, Ate, ytr)
                for clf_tag, make in CLASSIFIERS.items():
                    t0 = time.time()
                    pred = make(Ztr, ytr).fit(Ztr, ytr).predict(Zte)
                    row = {
                        "Test": test, "DR": method, "Clas": clf_tag,
                        "Acc": round(accuracy_score(yte, pred), 4),
                        "Pre": round(precision_score(yte, pred, average="macro",
                                                     zero_division=0), 4),
                        "Rec": round(recall_score(yte, pred, average="macro",
                                                  zero_division=0), 4),
                        "F1": round(f1_score(yte, pred, average="macro",
                                             zero_division=0), 4),
                        "components": Ztr.shape[1],
                        "seconds": round(time.time() - t0, 1),
                    }
                    rows.append(row)
                    print(f"  test {test}  {method:<4} {clf_tag:<4} "
                          f"acc {row['Acc']:.4f}  F1 {row['F1']:.4f}  "
                          f"({row['components']} components, {row['seconds']:.0f}s)",
                          flush=True)
        pd.DataFrame(rows).to_csv(CSV, index=False)

    d = pd.read_csv(CSV)
    with pd.ExcelWriter(XLSX, engine="openpyxl") as xl:
        d[["Test", "DR", "Clas", "Acc", "Pre", "Rec", "F1"]].to_excel(
            xl, sheet_name="PCA and LDA", index=False)
        d.to_excel(xl, sheet_name="full detail", index=False)
        (d.pivot_table(index="Clas", columns="DR", values="F1", aggfunc="mean")
          .round(4).to_excel(xl, sheet_name="method x classifier"))

        # the same table for Kernel PCA, so all three methods sit side by side
        kpca = RESULTS / "75_runs.csv"
        if kpca.exists():
            k = pd.read_csv(kpca)
            k["DR"] = "Kernel PCA (" + k["DR"] + ")"
            both = pd.concat([d[["Test", "DR", "Clas", "F1"]],
                              k[["Test", "DR", "Clas", "F1"]]], ignore_index=True)
            (both.pivot_table(index="Clas", columns="DR", values="F1", aggfunc="mean")
                 .round(4).to_excel(xl, sheet_name="all DR methods"))
    print(f"\nwrote {XLSX.name}: {len(d)} rows")


if __name__ == "__main__":
    main()
