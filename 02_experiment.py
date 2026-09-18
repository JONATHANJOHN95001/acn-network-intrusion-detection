"""
Step 2: experiment framework.

One place that defines the models, the preprocessing, the split and the
metrics, so every algorithm is compared on identical footing.

Usage:
    python 02_experiment.py --list
    python 02_experiment.py --all                       # every full-scale model, test sizes 0.2 / 0.4 / 0.6
    python 02_experiment.py --models decision_tree,random_forest --splits 0.4
    python 02_experiment.py --models knn,svm --sample 100000 --out results_knn_svm.csv

Runs already in the results file are skipped, so an interrupted run can simply
be started again. --fresh reruns them.
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
PARQUET = HERE / "data" / "cicids2017.parquet"
RESULTS = HERE / "results"
SEED = 0

# Use at most half the cores. Saturating every core for hours, together with
# heavy memory use, destabilised this machine (video memory manager bugcheck),
# so headroom here is a stability requirement, not a nicety.
N_JOBS = max(1, (os.cpu_count() or 2) // 2)


def _xgb():
    from xgboost import XGBClassifier

    return XGBClassifier(tree_method="hist", n_jobs=N_JOBS, random_state=SEED, verbosity=0)


def _lgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_jobs=N_JOBS, random_state=SEED, verbose=-1)


def _cat():
    from catboost import CatBoostClassifier

    # 300 rounds instead of the default 1000: on 2M rows the default takes
    # about an hour by itself, for little gain.
    return CatBoostClassifier(iterations=300, random_seed=SEED, thread_count=N_JOBS,
                              verbose=0, allow_writing_files=False)


# Ordered roughly fastest to slowest, so partial results arrive early.
# scale=True  -> a StandardScaler is inserted ahead of the estimator.
# heavy=True  -> superlinear in n; only sensible on a subsample.
MODELS = {
    "naive_bayes": dict(
        scale=True, heavy=False,
        make=lambda: GaussianNB()),
    "lda": dict(
        scale=True, heavy=False,
        make=lambda: LinearDiscriminantAnalysis()),
    "qda": dict(
        scale=True, heavy=False,
        # Fixed shrinkage, not "auto": the data holds 8 pairs of identical
        # features, so a class covariance is exactly singular and Ledoit-Wolf
        # picks too little shrinkage. The eigen solver also avoids svd's
        # "samples must exceed features" check, which Heartbleed (7 training
        # rows, 69 features) fails.
        make=lambda: QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=0.5)),
    "sgd": dict(
        scale=True, heavy=False,
        make=lambda: SGDClassifier(loss="hinge", penalty="l2", alpha=1e-4,
                                   max_iter=1000, tol=1e-3, n_jobs=N_JOBS,
                                   random_state=SEED)),
    "decision_tree": dict(
        scale=False, heavy=False,
        make=lambda: DecisionTreeClassifier(criterion="entropy", random_state=SEED)),
    "lightgbm": dict(scale=False, heavy=False, make=_lgbm),
    "xgboost": dict(scale=False, heavy=False, make=_xgb),
    "random_forest": dict(
        scale=False, heavy=False,
        make=lambda: RandomForestClassifier(n_estimators=100, criterion="entropy",
                                            n_jobs=N_JOBS, random_state=SEED)),
    "hist_gb": dict(
        scale=False, heavy=False,
        make=lambda: HistGradientBoostingClassifier(random_state=SEED)),
    "catboost": dict(scale=False, heavy=False, make=_cat),
    "adaboost": dict(
        scale=False, heavy=False,
        make=lambda: AdaBoostClassifier(n_estimators=100, random_state=SEED)),
    "mlp": dict(
        scale=True, heavy=False,
        make=lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60,
                                   early_stopping=True, random_state=42)),
    "linear_svm": dict(
        scale=True, heavy=False,
        # bounded: on the full dataset the default never converged and hung the grid
        make=lambda: LinearSVC(dual="auto", max_iter=2000, random_state=SEED)),
    "logistic": dict(
        scale=True, heavy=False,
        make=lambda: LogisticRegression(max_iter=1000, random_state=SEED)),
    "knn": dict(
        scale=True, heavy=True,
        make=lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=N_JOBS)),
    "svm": dict(
        scale=True, heavy=True,
        make=lambda: SVC(kernel="rbf", random_state=SEED)),
}


def split_tag(test_size):
    train = round((1 - test_size) * 100)
    return f"{train}/{round(test_size * 100)}", f"{train}_{round(test_size * 100)}"


def required_free_gb(sample):
    """How much free RAM this run actually needs. The full dataset is the heavy
    case; a small sample needs far less, so a single fixed threshold turned away
    jobs that were never going to strain the machine."""
    if sample is None:
        return 3.0
    if sample > 500_000:
        return 2.5
    if sample > 150_000:
        return 2.0
    return 1.5


def memory_guard(min_free_gb=3.0):
    """Refuse to start when free RAM is already low. Running heavy jobs on top of
    a nearly full machine crashed it (shared-memory iGPU + no headroom)."""
    try:
        import ctypes

        class S(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = S(); st.dwLength = ctypes.sizeof(S)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        free = st.ullAvailPhys / 2**30
    except Exception:
        return None
    print(f"  free RAM: {free:.1f} GB")
    if free < min_free_gb:
        raise SystemExit(f"Only {free:.1f} GB RAM free; this run needs {min_free_gb} GB. "
                         "Close some programs (a browser is usually the big one) and retry.")
    return free


def load(sample=None, floor=50):
    """The full dataset, or a stratified sample that keeps at least `floor`
    rows of every class (see common.load_sample)."""
    from common import load_sample

    t = time.time()
    df = load_sample(sample, floor, SEED)
    y = df["Label"].to_numpy()
    X = df.drop(columns=["Label", "Day"])
    print(f"  loaded {X.shape[0]:,} x {X.shape[1]} in {time.time() - t:.1f}s")
    return X, y


def build(cfg):
    steps = [("impute", SimpleImputer(strategy="mean"))]
    if cfg["scale"]:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", cfg["make"]()))
    return Pipeline(steps)


def run_one(name, X, y_enc, classes, test_size, tag_suffix=""):
    cfg = MODELS[name]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_enc, test_size=test_size, random_state=SEED, stratify=y_enc
    )

    pipe = build(cfg)
    t0 = time.time()
    pipe.fit(Xtr, ytr)
    fit_s = time.time() - t0

    t0 = time.time()
    pred = pipe.predict(Xte)
    pred_s = time.time() - t0

    split, split_file = split_tag(test_size)
    rec = recall_score(yte, pred, average=None, labels=range(len(classes)), zero_division=0)
    f1c = f1_score(yte, pred, average=None, labels=range(len(classes)), zero_division=0)
    row = {
        "model": name,
        "split": split,
        "test_size": test_size,
        "n_train": len(ytr),
        "n_test": len(yte),
        "accuracy": accuracy_score(yte, pred),
        "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(yte, pred, average="weighted", zero_division=0),
        "macro_prec": precision_score(yte, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(yte, pred, average="macro", zero_division=0),
        "fit_seconds": round(fit_s, 2),
        "pred_seconds": round(pred_s, 2),
        "flows_per_sec": int(len(yte) / pred_s) if pred_s > 0 else 0,
    }
    for c, r in zip(classes, rec):
        row[f"recall::{c}"] = r
    for c, f in zip(classes, f1c):
        row[f"f1::{c}"] = f

    tag = f"{name}_{split_file}{tag_suffix}"
    (RESULTS / "reports").mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(yte, pred, labels=range(len(classes)))
    (RESULTS / "reports" / f"{tag}.txt").write_text(
        f"{name}  split {split}\n\n"
        + classification_report(yte, pred, target_names=classes, digits=4, zero_division=0)
        + "\n\nConfusion matrix (rows = true, cols = predicted)\n"
        + str(cm),
        encoding="utf-8",
    )
    np.save(RESULTS / "reports" / f"{tag}_cm.npy", cm)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--splits", default="0.2,0.4,0.6", help="test sizes")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--fresh", action="store_true",
                    help="rerun model/split pairs already in the results file")
    a = ap.parse_args()

    if a.list:
        print(f"{'model':<16}{'scaled':<9}{'full-scale ok':<15}")
        for k, v in MODELS.items():
            ok = "no (subsample)" if v["heavy"] else "yes"
            print(f"{k:<16}{str(v['scale']):<9}{ok:<15}")
        return

    if a.all:
        # adaboost and linear_svm are excluded from the sweep on measured cost:
        # adaboost scored macro F1 0.165, and linear_svm needed 42.7 min for a
        # split to reach 0.579, which logistic regression nearly matches in 79 s.
        # Both stay available with --models.
        skip = {"adaboost", "linear_svm"}
        names = [n for n, v in MODELS.items() if not v["heavy"] and n not in skip]
    else:
        names = [m.strip() for m in a.models.split(",") if m.strip()]
    bad = [n for n in names if n not in MODELS]
    if bad:
        raise SystemExit(f"unknown model(s): {bad}")

    splits = [float(s) for s in a.splits.split(",")]
    memory_guard(required_free_gb(a.sample))
    X, y = load(a.sample)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(le.classes_)
    RESULTS.mkdir(exist_ok=True)
    suffix = f"_s{a.sample}" if a.sample else ""

    out_path = RESULTS / a.out
    rows = pd.read_csv(out_path).to_dict("records") if out_path.exists() else []
    sample_key = str(a.sample or "full")
    done = set() if a.fresh else {(r["model"], r["split"]) for r in rows
                                  if str(r.get("sample")) == sample_key}

    print(f"  {len(names)} models x {len(splits)} test sizes, n_jobs={N_JOBS}")
    for ts in splits:
        for n in names:
            if (n, split_tag(ts)[0]) in done:
                print(f"\n>>> {n}  split {split_tag(ts)[0]}: already done, skipped", flush=True)
                continue
            print(f"\n>>> {n}  split {split_tag(ts)[0]}", flush=True)
            try:
                r = run_one(n, X, y_enc, classes, ts, suffix)
                r["sample"] = a.sample or "full"
                rows = [
                    x for x in rows
                    if not (x["model"] == r["model"]
                            and x["split"] == r["split"]
                            and str(x.get("sample")) == str(r["sample"]))
                ]
                rows.append(r)
                print(f"    acc {r['accuracy']:.4f}  macroF1 {r['macro_f1']:.4f}  "
                      f"fit {r['fit_seconds']}s  pred {r['pred_seconds']}s  "
                      f"{r['flows_per_sec']:,} flows/s", flush=True)
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"\nwrote results/{a.out}")


if __name__ == "__main__":
    main()
