"""
Test_0.4_DR_COS_QDA

Test size 0.4  |  Kernel PCA (cosine)  |  Quadratic Discriminant Analysis  (reference sheet #17)

Note: the rarest class has fewer training rows than there are components, so the class covariance needs shrinkage before QDA can be fitted.

Run on its own:

    python Test_0.4_DR_COS_QDA.py

Writes Test_0.4_DR_COS_QDA.jpg (confusion matrix) and appends one row to
results/75_runs.csv.
"""

import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_sample                      # noqa: E402

warnings.filterwarnings("ignore")

# ----- this run -------------------------------------------------------------
TEST_SIZE = 0.4
KERNEL = "cosine"
CLASSIFIER = "QDA"
N_COMPONENTS = 15
# ----------------------------------------------------------------------------

SEED = 0
N_JOBS = max(1, (os.cpu_count() or 2) // 2)
STEM = "Test_0.4_DR_COS_QDA"
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


# QDA estimates a separate covariance matrix per class, so the rarest class
# has to supply more rows than there are components. Heartbleed supplies about
# five against 15 components, so its covariance is rank deficient. Shrinkage
# pulls each class covariance towards a shared spherical one, which makes it
# estimable; RANK_TOL is the threshold sklearn uses to decide whether that
# worked, and its default of 1e-4 is far above the 1e-7 scale of these
# components, so it would reject a covariance that is in fact positive definite.
RANK_TOL = 1e-12


def qda_shrinkage(X, y):
    """Smallest shrinkage that lets every class covariance be estimated.

    Less shrinkage keeps the model closer to true QDA, so the values are
    tried in order and the first that fits is used.
    """
    last = None
    for s in (0.1, 0.25, 0.5, 0.8, 0.95):
        try:
            QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=s,
                                          tol=RANK_TOL).fit(X, y)
        except Exception as exc:
            last = exc
            continue
        print("QDA shrinkage " + str(s))
        return s
    raise SystemExit("QDA cannot be estimated at any shrinkage: " + str(last))


def main():
    t0 = time.time()
    df = load_sample(10000, 50, SEED)
    y = LabelEncoder().fit_transform(df["Label"])
    X = df.drop(columns=["Label", "Day"])
    classes = sorted(df["Label"].unique())
    print(f"{len(df):,} flows, {X.shape[1]} features, {len(classes)} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    # mean imputation and scaling, fitted on the training split only
    imputer = SimpleImputer(strategy="mean").fit(X_train)
    scaler = StandardScaler().fit(imputer.transform(X_train))
    X_train = scaler.transform(imputer.transform(X_train))
    X_test = scaler.transform(imputer.transform(X_test))

    # dimensionality reduction: Kernel PCA, fitted on the training split only.
    # arpack keeps the largest positive eigenvalues; the randomized solver ranks
    # them by magnitude and fails on the sigmoid kernel, which is not PSD.
    kpca = KernelPCA(n_components=N_COMPONENTS, kernel=KERNEL,
                     eigen_solver="arpack", random_state=SEED, n_jobs=N_JOBS)
    X_train = kpca.fit_transform(X_train)
    X_test = kpca.transform(X_test)
    print(f"Kernel PCA ({KERNEL}): {X_train.shape[1]} components")

    classifier = QuadraticDiscriminantAnalysis(
        solver="eigen", tol=RANK_TOL,
        shrinkage=qda_shrinkage(X_train, y_train))
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    pre_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {pre:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # confusion matrix image, named to match this script
    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=1))
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Test {TEST_SIZE}  |  Kernel PCA {KERNEL}  |  Quadratic Discriminant Analysis\n"
                 f"accuracy {acc:.4f}   macro F1 {f1:.4f}", fontsize=10)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > cm.max() / 3 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, label="flows (log scale)")
    fig.tight_layout()
    fig.savefig(HERE / f"{STEM}.jpg", dpi=150)
    plt.close(fig)

    # one row appended to the shared results table
    RESULTS.mkdir(exist_ok=True)
    row = {"Test": TEST_SIZE, "DR": "COS", "Clas": CLASSIFIER,
           "Acc": round(acc, 4), "Pre": round(pre, 4), "Rec": round(rec, 4),
           "F1": round(f1, 4),
           "Pre_weighted": round(pre_w, 4), "Rec_weighted": round(rec_w, 4),
           "F1_weighted": round(f1_w, 4),
           "n_components": N_COMPONENTS, "seconds": round(time.time() - t0, 1),
           "script": STEM}
    out = RESULTS / "75_runs.csv"
    old = pd.read_csv(out).to_dict("records") if out.exists() else []
    old = [r for r in old if r.get("script") != STEM]
    pd.DataFrame(old + [row]).to_csv(out, index=False)
    print(f"saved {STEM}.jpg and appended to results/75_runs.csv "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
