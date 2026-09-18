"""
Step 5: class-imbalance study.

The same models are trained with different ways of handling the imbalance.
Only the TRAINING split is ever changed; the test split keeps the real
class mix, so the scores stay honest.

    none           train as is
    class_weight   weight each class by the inverse of its frequency
    undersample    keep at most 200,000 BENIGN training rows
    smote          add synthetic rows so every class has at least 5,000
                   training rows (SMOTE interpolates between neighbours)
    under+smote    both

Models: Decision Tree, Random Forest and LightGBM. All three are fast on the
full dataset and accept class weights.

Usage:
    python 05_imbalance.py                                  # full dataset, 60/40
    python 05_imbalance.py --sample 300000                  # quicker check
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from common import load_sample

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
OUT = HERE / "results" / "imbalance"
SEED = 0
N_JOBS = max(1, (os.cpu_count() or 2) // 2 - 2)   # leave real headroom
BENIGN_KEEP = 200_000
SMOTE_TO = 5_000
STRATEGIES = ["none", "class_weight", "undersample", "smote", "under+smote"]
RARE = ["Heartbleed", "Web Attack - Sql Injection", "Infiltration", "Web Attack - XSS",
        "Web Attack - Brute Force", "Bot"]


def _lgbm(balanced):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_jobs=N_JOBS, random_state=SEED, verbose=-1,
                          class_weight="balanced" if balanced else None)


MODELS = {
    "decision_tree": lambda bal: DecisionTreeClassifier(
        criterion="entropy", random_state=SEED, class_weight="balanced" if bal else None),
    "random_forest": lambda bal: RandomForestClassifier(
        n_estimators=100, criterion="entropy", n_jobs=N_JOBS, random_state=SEED,
        class_weight="balanced" if bal else None),
    "lightgbm": _lgbm,
}


def resample(strategy, X, y, benign):
    """Apply the strategy to the training split only."""
    if strategy in ("undersample", "under+smote") and np.sum(y == benign) > BENIGN_KEEP:
        X, y = RandomUnderSampler(sampling_strategy={benign: BENIGN_KEEP},
                                  random_state=SEED).fit_resample(X, y)
    if strategy in ("smote", "under+smote"):
        counts = np.bincount(y)
        target = {c: SMOTE_TO for c, n in enumerate(counts) if 0 < n < SMOTE_TO}
        if target:
            k = min(5, min(counts[c] for c in target) - 1)
            X, y = SMOTE(sampling_strategy=target, k_neighbors=k,
                         random_state=SEED).fit_resample(X, y)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="default: full dataset")
    ap.add_argument("--floor", type=int, default=2000)
    ap.add_argument("--test-size", type=float, default=0.4)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",")]
    strategies = [s.strip() for s in a.strategies.split(",")]

    t0 = time.time()
    df = load_sample(a.sample, a.floor, SEED)
    le = LabelEncoder()
    y = le.fit_transform(df["Label"])
    classes = list(le.classes_)
    benign = classes.index("BENIGN")
    X = df.drop(columns=["Label", "Day"])
    del df
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=a.test_size, random_state=SEED,
                                          stratify=y)
    del X
    imputer = SimpleImputer(strategy="mean").fit(Xtr)
    Xtr = imputer.transform(Xtr).astype(np.float32)
    Xte = imputer.transform(Xte).astype(np.float32)
    print(f"train {len(ytr):,} / test {len(yte):,} rows, {len(classes)} classes, "
          f"loaded in {time.time() - t0:.0f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_s{a.sample}" if a.sample else ""
    out_path = OUT / f"imbalance{tag}.csv"
    rows = []
    for strat in strategies:
        Xs, ys = resample(strat, Xtr, ytr, benign)
        print(f"\n[{strat}] training rows: {len(ys):,}")
        for name in models:
            clf = MODELS[name](strat == "class_weight")
            t1 = time.time()
            clf.fit(Xs, ys)
            fit_s = time.time() - t1
            pred = clf.predict(Xte)
            rec = recall_score(yte, pred, average=None, labels=range(len(classes)), zero_division=0)
            f1c = f1_score(yte, pred, average=None, labels=range(len(classes)), zero_division=0)
            row = {"model": name, "strategy": strat, "n_train": len(ys),
                   "accuracy": accuracy_score(yte, pred),
                   "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
                   "weighted_f1": f1_score(yte, pred, average="weighted", zero_division=0),
                   "fit_seconds": round(fit_s, 1)}
            row.update({f"recall::{c}": r for c, r in zip(classes, rec)})
            row.update({f"f1::{c}": f for c, f in zip(classes, f1c)})
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            rare = "  ".join(f"{c.split(' - ')[-1][:11]} {row[f'recall::{c}']:.2f}" for c in RARE)
            print(f"  {name:<14} macroF1 {row['macro_f1']:.4f}  acc {row['accuracy']:.4f}  "
                  f"fit {fit_s:5.0f}s | recall: {rare}", flush=True)

    res = pd.DataFrame(rows)
    lines = ["Macro F1 by model and strategy (test split keeps the real class mix):", ""]
    lines.append(res.pivot_table(index="model", columns="strategy", values="macro_f1",
                                 sort=False).round(4).to_string())
    for c in RARE:
        lines += ["", f"Recall on {c}:"]
        lines.append(res.pivot_table(index="model", columns="strategy", values=f"recall::{c}",
                                     sort=False).round(3).to_string())
    (OUT / f"summary{tag}.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\ndone in {time.time() - t0:.0f}s, wrote results/imbalance/")


if __name__ == "__main__":
    main()
