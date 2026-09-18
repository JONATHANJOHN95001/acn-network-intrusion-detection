"""
Step 10: how much do the duplicate rows actually inflate the score?

The earlier attempt compared two different datasets, which confounded the
class balance and diluted the duplicates by sampling. This measures leakage
directly instead:

  1. take the FULL dataset with duplicates kept, as published work on
     CIC-IDS2017 usually does, and split it at random
  2. find which test rows appear verbatim in the training split (same feature
     vector and same label). Those are rows the model can simply memorise
  3. train, then score the leaked test rows and the genuinely unseen test rows
     separately

If the model scores far better on the leaked rows, the overall figure is
inflated, and by roughly the leaked share times that gap. The deduplicated
pipeline is run alongside as the honest reference.

Two fast, strong models are enough to make the point.

    python 10_leakage.py
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
OUT = HERE / "results" / "paper"
CLEAN = HERE / "data" / "cicids2017.parquet"
WITHDUPS = HERE / "data" / "cicids2017_withdups.parquet"
SEED = 0
N_JOBS = max(1, (os.cpu_count() or 2) // 2)
TEST_SIZE = 0.4

MODELS = {
    "decision_tree": lambda: DecisionTreeClassifier(criterion="entropy", random_state=SEED),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, criterion="entropy",
                                                    n_jobs=N_JOBS, random_state=SEED),
}


def run(parquet, label):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    t0 = time.time()
    df = pd.read_parquet(parquet)
    le = LabelEncoder()
    y = le.fit_transform(df["Label"])
    classes = list(le.classes_)

    # A row is "the same record" if its features and its label match. Day is
    # metadata the model never sees, so it is excluded from the comparison.
    h = pd.util.hash_pandas_object(df.drop(columns=["Day"]), index=False).to_numpy()
    X = df.drop(columns=["Label", "Day"]).to_numpy(dtype=np.float32)
    del df
    print(f"{len(y):,} flows, {X.shape[1]} features, loaded in {time.time() - t0:.0f}s")

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    leaked = np.isin(h[te], np.unique(h[tr]))
    share = leaked.mean()
    print(f"test rows that also appear in training: {leaked.sum():,} of {len(te):,} "
          f"({share:.1%})")

    imp = SimpleImputer(strategy="mean").fit(X[tr])
    Xtr, Xte = imp.transform(X[tr]), imp.transform(X[te])
    ytr, yte = y[tr], y[te]
    del X

    rows = []
    for name, make in MODELS.items():
        t1 = time.time()
        pred = make().fit(Xtr, ytr).predict(Xte)
        fit_s = time.time() - t1
        overall_acc = accuracy_score(yte, pred)
        overall_f1 = f1_score(yte, pred, average="macro", zero_division=0)
        acc_leaked = accuracy_score(yte[leaked], pred[leaked]) if leaked.any() else np.nan
        acc_unseen = accuracy_score(yte[~leaked], pred[~leaked]) if (~leaked).any() else np.nan
        f1_unseen = (f1_score(yte[~leaked], pred[~leaked], average="macro",
                              labels=range(len(classes)), zero_division=0)
                     if (~leaked).any() else np.nan)
        rows.append({"condition": label, "model": name,
                     "leaked_share": round(share, 4),
                     "accuracy_overall": overall_acc, "macro_f1_overall": overall_f1,
                     "accuracy_on_leaked_rows": acc_leaked,
                     "accuracy_on_unseen_rows": acc_unseen,
                     "macro_f1_on_unseen_rows": f1_unseen,
                     "fit_seconds": round(fit_s, 1)})
        print(f"  {name:<15} overall acc {overall_acc:.4f} / macroF1 {overall_f1:.4f}"
              f" | leaked rows acc {acc_leaked:.4f} | unseen rows acc {acc_unseen:.4f}"
              f"  ({fit_s:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    if WITHDUPS.exists():
        frames.append(run(WITHDUPS, "duplicates kept (as most published work does)"))
    else:
        print("run: python 01_prepare_data.py --keep-duplicates")
    frames.append(run(CLEAN, "duplicates removed (this project)"))
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(OUT / "leakage_full.csv", index=False)

    lines = ["Duplicate-row leakage, full dataset, 60/40 split", ""]
    for cond, g in res.groupby("condition", sort=False):
        lines += [cond, f"  test rows also present in training: {g['leaked_share'].iloc[0]:.1%}"]
        for _, r in g.iterrows():
            lines.append(
                f"  {r['model']:<15} overall acc {r['accuracy_overall']:.4f}  "
                f"macroF1 {r['macro_f1_overall']:.4f}   "
                f"acc on leaked {r['accuracy_on_leaked_rows']:.4f}  "
                f"acc on unseen {r['accuracy_on_unseen_rows']:.4f}")
        lines.append("")
    dup = res[res["condition"].str.startswith("duplicates kept")]
    if len(dup):
        lines += ["Memorisation gap (accuracy on leaked rows minus accuracy on unseen rows):"]
        for _, r in dup.iterrows():
            gap = r["accuracy_on_leaked_rows"] - r["accuracy_on_unseen_rows"]
            infl = gap * r["leaked_share"]
            lines.append(f"  {r['model']:<15} gap {gap:+.4f}   "
                         f"contribution to the reported accuracy {infl:+.4f}")
    (OUT / "leakage_full_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print("wrote results/paper/leakage_full.csv")


if __name__ == "__main__":
    main()
