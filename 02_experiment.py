"""
Step 2: experiment framework.

One place that defines the models, the preprocessing, the split and the
metrics, so every algorithm is compared on identical footing.

Usage:
    python 02_experiment.py --list
    python 02_experiment.py --models decision_tree --splits 0.4
    python 02_experiment.py --models decision_tree,random_forest --splits 0.4,0.3,0.2
    python 02_experiment.py --all --splits 0.4
    python 02_experiment.py --models svm,knn --sample 100000     # subsample first
"""

import argparse
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


def _xgb():
    from xgboost import XGBClassifier

    return XGBClassifier(tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0)


def _lgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_jobs=-1, random_state=SEED, verbose=-1)


def _cat():
    from catboost import CatBoostClassifier

    return CatBoostClassifier(random_seed=SEED, verbose=0, allow_writing_files=False)


# scale=True  -> a StandardScaler is inserted ahead of the estimator.
# heavy=True  -> superlinear in n; only sensible on a subsample.
MODELS = {
    "logistic": dict(
        scale=True, heavy=False,
        make=lambda: LogisticRegression(max_iter=1000, n_jobs=-1, random_state=SEED)),
    "sgd": dict(
        scale=True, heavy=False,
        make=lambda: SGDClassifier(loss="hinge", penalty="l2", alpha=1e-4,
                                   max_iter=1000, tol=1e-3, n_jobs=-1, random_state=SEED)),
    "linear_svm": dict(
        scale=True, heavy=False,
        make=lambda: LinearSVC(dual="auto", random_state=SEED)),
    "naive_bayes": dict(
        scale=True, heavy=False,
        make=lambda: GaussianNB()),
    "lda": dict(
        scale=True, heavy=False,
        make=lambda: LinearDiscriminantAnalysis()),
    "qda": dict(
        scale=True, heavy=False,
        make=lambda: QuadraticDiscriminantAnalysis(reg_param=0.01)),
    "mlp": dict(
        scale=True, heavy=False,
        make=lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60,
                                   early_stopping=True, random_state=42)),
    "decision_tree": dict(
        scale=False, heavy=False,
        make=lambda: DecisionTreeClassifier(criterion="entropy", random_state=SEED)),
    "random_forest": dict(
        scale=False, heavy=False,
        make=lambda: RandomForestClassifier(n_estimators=100, criterion="entropy",
                                            n_jobs=-1, random_state=SEED)),
    "adaboost": dict(
        scale=False, heavy=False,
        make=lambda: AdaBoostClassifier(n_estimators=100, random_state=SEED)),
    "hist_gb": dict(
        scale=False, heavy=False,
        make=lambda: HistGradientBoostingClassifier(random_state=SEED)),
    "xgboost": dict(scale=False, heavy=False, make=_xgb),
    "lightgbm": dict(scale=False, heavy=False, make=_lgbm),
    "catboost": dict(scale=False, heavy=False, make=_cat),
    "knn": dict(
        scale=True, heavy=True,
        make=lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    "svm": dict(
        scale=True, heavy=True,
        make=lambda: SVC(kernel="rbf", random_state=SEED)),
}


def load(sample=None):
    t = time.time()
    df = pd.read_parquet(PARQUET)
    if sample and sample < len(df):
        # stratified subsample that keeps every class present
        frac = sample / len(df)
        df = (
            df.groupby("Label", group_keys=False)
            .apply(lambda g: g.sample(max(2, int(round(len(g) * frac))), random_state=SEED))
            .reset_index(drop=True)
        )
    y = df["Label"].to_numpy()
    X = df.drop(columns=["Label", "Day"])
    print(f"  loaded {X.shape[0]:,} x {X.shape[1]} in {time.time() - t:.1f}s")
    return X, y


def build(cfg):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if cfg["scale"]:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", cfg["make"]()))
    return Pipeline(steps)


def run_one(name, X, y_enc, classes, test_size):
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

    rec = recall_score(yte, pred, average=None, labels=range(len(classes)), zero_division=0)
    row = {
        "model": name,
        "split": f"{int((1 - test_size) * 100)}/{int(test_size * 100)}",
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

    tag = f"{name}_{int((1 - test_size) * 100)}_{int(test_size * 100)}"
    (RESULTS / "reports").mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(yte, pred, labels=range(len(classes)))
    (RESULTS / "reports" / f"{tag}.txt").write_text(
        f"{name}  split {row['split']}\n\n"
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
    ap.add_argument("--splits", default="0.4")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default="results.csv")
    a = ap.parse_args()

    if a.list:
        print(f"{'model':<16}{'scaled':<9}{'full-scale ok':<15}")
        for k, v in MODELS.items():
            ok = "no (subsample)" if v["heavy"] else "yes"
            print(f"{k:<16}{str(v['scale']):<9}{ok:<15}")
        return

    if a.all:
        names = [n for n, v in MODELS.items() if not v["heavy"]]
    else:
        names = [m.strip() for m in a.models.split(",") if m.strip()]
    bad = [n for n in names if n not in MODELS]
    if bad:
        raise SystemExit(f"unknown model(s): {bad}")

    splits = [float(s) for s in a.splits.split(",")]
    X, y = load(a.sample)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(le.classes_)
    RESULTS.mkdir(exist_ok=True)

    out_path = RESULTS / a.out
    rows = pd.read_csv(out_path).to_dict("records") if out_path.exists() else []

    for ts in splits:
        for n in names:
            print(f"\n>>> {n}  split {int((1 - ts) * 100)}/{int(ts * 100)}")
            try:
                r = run_one(n, X, y_enc, classes, ts)
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
                      f"{r['flows_per_sec']:,} flows/s")
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")
            pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
