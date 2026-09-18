"""
Step 11: generate the 75 standalone run scripts.

The assignment asks for one script per combination:

    5 kernels x 3 test sizes  = 15 per classifier
    x 5 classifiers           = 75 runs

Each generated script is self-contained and runnable on its own. It applies
Kernel PCA with one kernel at one test size, trains one classifier, prints
accuracy, precision, recall and F1, saves its confusion matrix as a .jpg, and
appends its row to results/75_runs.csv.

Naming follows the required format:   Test_0.2_DR_RBF_DT.py / .jpg

The five algorithms are the ones assigned to this student in the reference
sheet: 1, 5, 7, 15, 17. Two of them need a note:

  #1  Linear Regression and #5 Elastic Net are REGRESSORS. They predict a
      continuous number, so accuracy, precision, recall, F1 and a confusion
      matrix cannot be computed from them at all. Two other numbered entries
      are taken instead, both genuine classifiers and neither belonging to the
      other student (whose set is 4, 6, 14, 16, 18):
        #1 Linear Regression -> #2  Logistic Regression
        #5 Elastic Net       -> #13 k-Nearest Neighbors
      The five then cover five different families: linear, tree ensemble,
      kernel, discriminant and instance-based.
  #17 QDA needs two settings the others do not. Heartbleed has only eleven
      flows in the whole dataset, so a split leaves about five for training,
      far fewer than the 15 components: its class covariance is rank
      deficient and plain QDA refuses to fit. Shrinkage makes it estimable,
      and the rank tolerance has to be lowered as well, because sklearn
      tests rank against an absolute 1e-4 while these eigenvalues are 1e-7.

    python 11_make_75_scripts.py          -> writes runs/*.py
    python 12_run_75.py                   -> runs them all, builds the Excel
"""

from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "runs"

N_COMPONENTS = 15          # fixed: the 75 runs vary kernel, test size and classifier
SAMPLE = 10_000            # Kernel PCA builds an n x n kernel matrix, so it needs a sample
FLOOR = 50                 # keep at least this many rows of every class in the sample

TESTS = [0.2, 0.4, 0.6]

KERNELS = {               # tag -> sklearn kernel name
    "LIN": "linear",
    "POLY": "poly",
    "RBF": "rbf",
    "SIG": "sigmoid",
    "COS": "cosine",
}

# tag -> (sheet number, display name, import line, constructor, note)
CLASSIFIERS = {
    "LR": (2, "Logistic Regression",
           "from sklearn.linear_model import LogisticRegression",
           "LogisticRegression(max_iter=2000, random_state=SEED)",
           "taken in place of sheet #1 Linear Regression, which is a regressor "
           "and cannot classify"),
    "KNN": (13, "k-Nearest Neighbors",
            "from sklearn.neighbors import KNeighborsClassifier",
            "KNeighborsClassifier(n_neighbors=5, n_jobs=N_JOBS)",
            "taken in place of sheet #5 Elastic Net, which is a regressor and "
            "cannot classify"),
    "RF": (7, "Random Forest",
           "from sklearn.ensemble import RandomForestClassifier",
           'RandomForestClassifier(n_estimators=200, criterion="entropy",\n'
           "                                    n_jobs=N_JOBS, random_state=SEED)",
           ""),
    "SVM": (15, "Support Vector Machine",
            "from sklearn.svm import SVC",
            'SVC(kernel="rbf", probability=True, random_state=SEED)',
            ""),
    "QDA": (17, "Quadratic Discriminant Analysis",
            "from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis",
            'QuadraticDiscriminantAnalysis(\n'
            "        solver=\"eigen\", tol=RANK_TOL,\n"
            "        shrinkage=qda_shrinkage(X_train, y_train))",
            "the rarest class has fewer training rows than there are components, "
            "so the class covariance needs shrinkage before QDA can be fitted"),
}

# tag -> extra code placed in the generated script, above main()
HELPERS = {
    "QDA": '''# QDA estimates a separate covariance matrix per class, so the rarest class
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


''',
}

TEMPLATE = '''"""
{stem}

Test size {test}  |  Kernel PCA ({kernel})  |  {clf_name}  (reference sheet #{num})

{note_block}Run on its own:

    python {stem}.py

Writes {stem}.jpg (confusion matrix) and appends one row to
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
{clf_import}

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_sample                      # noqa: E402

warnings.filterwarnings("ignore")

# ----- this run -------------------------------------------------------------
TEST_SIZE = {test}
KERNEL = "{kernel}"
CLASSIFIER = "{clf_tag}"
N_COMPONENTS = {ncomp}
# ----------------------------------------------------------------------------

SEED = 0
N_JOBS = max(1, (os.cpu_count() or 2) // 2)
STEM = "{stem}"
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


{helper}def main():
    t0 = time.time()
    df = load_sample({sample}, {floor}, SEED)
    y = LabelEncoder().fit_transform(df["Label"])
    X = df.drop(columns=["Label", "Day"])
    classes = sorted(df["Label"].unique())
    print(f"{{len(df):,}} flows, {{X.shape[1]}} features, {{len(classes)}} classes")

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
    print(f"Kernel PCA ({{KERNEL}}): {{X_train.shape[1]}} components")

    classifier = {clf_ctor}
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    pre_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy  : {{acc:.4f}}")
    print(f"Precision : {{pre:.4f}}")
    print(f"Recall    : {{rec:.4f}}")
    print(f"F1 Score  : {{f1:.4f}}")

    # confusion matrix image, named to match this script
    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=1))
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Test {{TEST_SIZE}}  |  Kernel PCA {{KERNEL}}  |  {clf_name}\\n"
                 f"accuracy {{acc:.4f}}   macro F1 {{f1:.4f}}", fontsize=10)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > cm.max() / 3 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, label="flows (log scale)")
    fig.tight_layout()
    fig.savefig(HERE / f"{{STEM}}.jpg", dpi=150)
    plt.close(fig)

    # one row appended to the shared results table
    RESULTS.mkdir(exist_ok=True)
    row = {{"Test": TEST_SIZE, "DR": "{kernel_tag}", "Clas": CLASSIFIER,
           "Acc": round(acc, 4), "Pre": round(pre, 4), "Rec": round(rec, 4),
           "F1": round(f1, 4),
           "Pre_weighted": round(pre_w, 4), "Rec_weighted": round(rec_w, 4),
           "F1_weighted": round(f1_w, 4),
           "n_components": N_COMPONENTS, "seconds": round(time.time() - t0, 1),
           "script": STEM}}
    out = RESULTS / "75_runs.csv"
    old = pd.read_csv(out).to_dict("records") if out.exists() else []
    old = [r for r in old if r.get("script") != STEM]
    pd.DataFrame(old + [row]).to_csv(out, index=False)
    print(f"saved {{STEM}}.jpg and appended to results/75_runs.csv "
          f"({{time.time() - t0:.0f}}s)")


if __name__ == "__main__":
    main()
'''


def main():
    OUT.mkdir(exist_ok=True)
    made = 0
    for clf_tag, (num, clf_name, clf_import, clf_ctor, note) in CLASSIFIERS.items():
        for kernel_tag, kernel in KERNELS.items():
            for test in TESTS:
                stem = f"Test_{test}_DR_{kernel_tag}_{clf_tag}"
                note_block = f"Note: {note}.\n\n" if note else ""
                (OUT / f"{stem}.py").write_text(
                    TEMPLATE.format(stem=stem, test=test, kernel=kernel,
                                    kernel_tag=kernel_tag, clf_tag=clf_tag,
                                    clf_name=clf_name, num=num,
                                    clf_import=clf_import, clf_ctor=clf_ctor,
                                    note_block=note_block, ncomp=N_COMPONENTS,
                                    helper=HELPERS.get(clf_tag, ""),
                                    sample=SAMPLE, floor=FLOOR),
                    encoding="utf-8")
                made += 1
    print(f"wrote {made} scripts to runs/")
    print(f"  {len(CLASSIFIERS)} classifiers x {len(KERNELS)} kernels x {len(TESTS)} test sizes")
    print(f"  fixed at {N_COMPONENTS} components, {SAMPLE:,}-flow stratified sample")


if __name__ == "__main__":
    main()
