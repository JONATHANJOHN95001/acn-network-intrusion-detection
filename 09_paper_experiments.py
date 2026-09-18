"""
Step 9: the two experiments that test whether the headline scores mean what
they appear to mean.

A. Shortcut features.
   Permutation importance says the models lean hardest on Destination Port
   and the two initial TCP window sizes. Those are properties of the service
   and of the sending machine's TCP stack, not of the attack: in this testbed
   the attacks come from a few Linux machines and most benign traffic from
   Windows clients, so a model can score well by recognising the machine.
   We retrain with those features removed and measure what is left.

B. Duplicate rows.
   9.07% of the raw rows are exact duplicates. Split at random, copies land
   on both sides, so the model is scored on rows it memorised. Most published
   work on this dataset does not remove them. We run both ways and measure
   how much the duplicates inflate the result.

Everything else is held fixed: same sample size, same split, same seed, same
pipeline. Output: results/paper/.

    python 09_paper_experiments.py
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import load_sample

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
OUT = HERE / "results" / "paper"
CLEAN_PARQUET = HERE / "data" / "cicids2017.parquet"
DUP_PARQUET = HERE / "data" / "cicids2017_withdups.parquet"
SEED = 0
N_JOBS = max(1, (os.cpu_count() or 2) // 2)
SAMPLE = 400_000
TEST_SIZE = 0.4

PORT = ["Destination Port"]
WINDOW = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]

FEATURE_SETS = {
    "all 69 features": [],
    "no destination port": PORT,
    "no TCP window size": WINDOW,
    "no shortcut features": PORT + WINDOW,
}

RARE = ["Heartbleed", "Web Attack - Sql Injection", "Infiltration",
        "Web Attack - XSS", "Web Attack - Brute Force", "Bot"]


def _xgb():
    from xgboost import XGBClassifier

    return XGBClassifier(tree_method="hist", n_jobs=N_JOBS, random_state=SEED, verbosity=0)


MODELS = {
    "decision_tree": (False, lambda: DecisionTreeClassifier(criterion="entropy",
                                                            random_state=SEED)),
    "random_forest": (False, lambda: RandomForestClassifier(n_estimators=100,
                                                            criterion="entropy",
                                                            n_jobs=N_JOBS, random_state=SEED)),
    "xgboost": (False, _xgb),
    "logistic": (True, lambda: LogisticRegression(max_iter=1000, random_state=SEED)),
}


def evaluate(X, y, classes, scale, make):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE,
                                          random_state=SEED, stratify=y)
    steps = [("impute", SimpleImputer(strategy="mean"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", make()))
    pipe = Pipeline(steps)
    t0 = time.time()
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    rec = recall_score(yte, pred, average=None, labels=range(len(classes)), zero_division=0)
    return {
        "accuracy": accuracy_score(yte, pred),
        "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(yte, pred, average="weighted", zero_division=0),
        "fit_seconds": round(time.time() - t0, 1),
        **{f"recall::{c}": r for c, r in zip(classes, rec)},
    }


def experiment_a():
    """Retrain without the shortcut features."""
    print("=" * 70)
    print("A. SHORTCUT FEATURES")
    print("=" * 70)
    df = load_sample(SAMPLE, 50, SEED, parquet=CLEAN_PARQUET)
    le = LabelEncoder()
    y = le.fit_transform(df["Label"])
    classes = list(le.classes_)
    X_full = df.drop(columns=["Label", "Day"])
    del df
    print(f"{len(X_full):,} flows, {X_full.shape[1]} features, {len(classes)} classes\n")

    rows = []
    for label, drop in FEATURE_SETS.items():
        X = X_full.drop(columns=[c for c in drop if c in X_full.columns])
        for name, (scale, make) in MODELS.items():
            r = evaluate(X, y, classes, scale, make)
            rows.append({"feature_set": label, "n_features": X.shape[1],
                         "model": name, **r})
            print(f"  {label:<22}{name:<16} macroF1 {r['macro_f1']:.4f}  "
                  f"acc {r['accuracy']:.4f}  ({X.shape[1]} features)", flush=True)
    return pd.DataFrame(rows), classes


def experiment_b():
    """Same pipeline, duplicates kept versus removed."""
    print()
    print("=" * 70)
    print("B. DUPLICATE ROWS")
    print("=" * 70)
    rows = []
    for label, src in (("duplicates removed", CLEAN_PARQUET),
                       ("duplicates kept", DUP_PARQUET)):
        if not src.exists():
            print(f"  {label}: {src.name} missing, skipped")
            continue
        df = load_sample(SAMPLE, 50, SEED, parquet=src)
        le = LabelEncoder()
        y = le.fit_transform(df["Label"])
        classes = list(le.classes_)
        X = df.drop(columns=["Label", "Day"])
        del df
        dup_share = 0.0
        if label == "duplicates kept":
            dup_share = float(X.duplicated().mean())
        print(f"\n  [{label}] {len(X):,} flows"
              + (f", {dup_share:.1%} of them duplicates" if dup_share else ""))
        for name, (scale, make) in MODELS.items():
            r = evaluate(X, y, classes, scale, make)
            rows.append({"condition": label, "model": name, **r})
            print(f"    {name:<16} macroF1 {r['macro_f1']:.4f}  acc {r['accuracy']:.4f}",
                  flush=True)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    a, classes = experiment_a()
    a.to_csv(OUT / "ablation.csv", index=False)
    b = experiment_b()
    b.to_csv(OUT / "leakage.csv", index=False)

    lines = ["A. SHORTCUT FEATURES", "",
             "Macro F1 by feature set (400k sample, 60/40 split):", ""]
    pa = a.pivot_table(index="model", columns="feature_set", values="macro_f1", sort=False)
    pa = pa[list(FEATURE_SETS)]
    lines.append(pa.round(4).to_string())
    lines += ["", "Change when the shortcut features are removed:", ""]
    delta = (pa["no shortcut features"] - pa["all 69 features"]).round(4)
    lines.append(delta.to_string())
    lines += ["", "Accuracy by feature set:", ""]
    lines.append(a.pivot_table(index="model", columns="feature_set", values="accuracy",
                               sort=False)[list(FEATURE_SETS)].round(4).to_string())

    if len(b):
        lines += ["", "", "B. DUPLICATE ROWS", "",
                  "Same pipeline, duplicates kept versus removed:", ""]
        pb = b.pivot_table(index="model", columns="condition", values="macro_f1", sort=False)
        if pb.shape[1] == 2:
            pb = pb[["duplicates removed", "duplicates kept"]]
            pb["inflation"] = (pb["duplicates kept"] - pb["duplicates removed"]).round(4)
        lines.append(pb.round(4).to_string())
        lines += ["", "Accuracy:", ""]
        pbacc = b.pivot_table(index="model", columns="condition", values="accuracy", sort=False)
        lines.append(pbacc.round(4).to_string())

    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\ndone in {time.time() - t0:.0f}s, wrote results/paper/")


if __name__ == "__main__":
    main()
