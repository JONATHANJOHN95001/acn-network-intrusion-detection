"""
Step 4: which flow features identify each attack.

1. Redundant features: pairs of columns that are identical or almost
   perfectly correlated (CICFlowMeter reports some quantities twice).
2. Global importance: a Random Forest is trained on a stratified sample,
   then each feature in turn is shuffled on held-out data; the drop in
   macro F1 is that feature's importance (permutation importance). The
   forest's own impurity-based importance is reported alongside.
3. Attack signatures: for each attack, the features that on their own best
   separate it from benign traffic, with the typical (median) values side
   by side. Durations and inter-arrival times are in microseconds.

Usage:
    python 04_feature_importance.py
    python 04_feature_importance.py --sample 500000 --repeats 5
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from common import load_sample

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
OUT = HERE / "results" / "features"
SEED = 0
N_JOBS = max(1, (os.cpu_count() or 2) // 2 - 2)   # leave real headroom
BENIGN = "BENIGN"


def redundant_pairs(X, threshold=0.99):
    """Feature pairs with |Pearson r| >= threshold, flagging exact copies."""
    corr = X.corr().abs()
    cols = list(X.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if r >= threshold:
                rows.append({"feature_a": a, "feature_b": b, "abs_corr": round(float(r), 4),
                             "identical": bool(X[a].equals(X[b]))})
    out = pd.DataFrame(rows, columns=["feature_a", "feature_b", "abs_corr", "identical"])
    return out.sort_values(["identical", "abs_corr"], ascending=False)


def capped(y, cap, seed):
    """Row positions keeping at most `cap` rows per class (all rows of smaller classes)."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        where = np.flatnonzero(y == c)
        idx.append(where if len(where) <= cap else rng.choice(where, cap, replace=False))
    return np.sort(np.concatenate(idx))


def signatures(X, y, top=5, benign_cap=20000):
    """For each attack, the features that best separate it from benign traffic
    on their own. separation = |AUC - 0.5| * 2 for that single feature
    (0 = no separation, 1 = perfect)."""
    rng = np.random.default_rng(SEED)
    ben = np.flatnonzero(y == BENIGN)
    if len(ben) > benign_cap:
        ben = rng.choice(ben, benign_cap, replace=False)
    fill = X.iloc[ben].median()
    Xb = X.iloc[ben].fillna(fill)
    rows = []
    for c in sorted(set(y) - {BENIGN}):
        Xa = X[y == c].fillna(fill)
        target = np.r_[np.zeros(len(Xb)), np.ones(len(Xa))]
        scores = []
        for f in X.columns:
            v = np.r_[Xb[f].to_numpy(), Xa[f].to_numpy()]
            if np.ptp(v) == 0:
                continue
            auc = roc_auc_score(target, v)
            scores.append((abs(auc - 0.5) * 2, f, auc))
        scores.sort(reverse=True)
        for rank, (sep, f, auc) in enumerate(scores[:top], 1):
            rows.append({"attack": c, "rank": rank, "feature": f,
                         "separation": round(sep, 3),
                         "attack_is": "higher" if auc > 0.5 else "lower",
                         "median_attack": float(Xa[f].median()),
                         "median_benign": float(Xb[f].median()),
                         "n_attack": len(Xa)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300000)
    ap.add_argument("--floor", type=int, default=2000,
                    help="minimum rows kept per class in the sample")
    ap.add_argument("--repeats", type=int, default=3,
                    help="shuffles per feature for permutation importance")
    a = ap.parse_args()

    t0 = time.time()
    df = load_sample(a.sample, a.floor, SEED)
    y = df["Label"].to_numpy()
    X = df.drop(columns=["Label", "Day"])
    del df
    print(f"sample: {len(X):,} rows x {X.shape[1]} features, {len(set(y))} classes")
    OUT.mkdir(parents=True, exist_ok=True)

    red = redundant_pairs(X)
    red.to_csv(OUT / "redundant_features.csv", index=False)
    print(f"redundant pairs (|r| >= 0.99): {len(red)}, identical copies: {int(red['identical'].sum())}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    imputer = SimpleImputer(strategy="mean").fit(Xtr)
    Xtr = pd.DataFrame(imputer.transform(Xtr), columns=X.columns)
    Xte = pd.DataFrame(imputer.transform(Xte), columns=X.columns)
    rf = RandomForestClassifier(n_estimators=100, criterion="entropy", n_jobs=N_JOBS,
                                random_state=SEED).fit(Xtr, ytr)
    base = f1_score(yte, rf.predict(Xte), average="macro")
    print(f"random forest macro F1 on held-out data: {base:.4f}")

    sub = capped(yte, 5000, SEED)
    t1 = time.time()
    perm = permutation_importance(rf, Xte.iloc[sub], yte[sub], scoring="f1_macro",
                                  n_repeats=a.repeats, random_state=SEED)
    print(f"permutation importance on {len(sub):,} held-out rows: {time.time() - t1:.0f}s")
    imp = (pd.DataFrame({"feature": X.columns,
                         "permutation_mean": perm.importances_mean,
                         "permutation_std": perm.importances_std,
                         "impurity": rf.feature_importances_})
             .sort_values("permutation_mean", ascending=False))
    imp.to_csv(OUT / "importance.csv", index=False)

    sig = signatures(X, y)
    sig.to_csv(OUT / "signatures.csv", index=False)

    lines = [f"Random Forest, {len(X):,}-flow stratified sample, 70/30 split",
             f"held-out macro F1: {base:.4f}", "",
             "Top 15 features by permutation importance (drop in macro F1 when shuffled):"]
    for _, r in imp.head(15).iterrows():
        lines.append(f"  {r['feature']:<30} {r['permutation_mean']:.4f} +/- {r['permutation_std']:.4f}"
                     f"   impurity {r['impurity']:.4f}")
    lines += ["", "Identical feature pairs:"]
    lines += [f"  {r['feature_a']}  ==  {r['feature_b']}"
              for _, r in red[red["identical"]].iterrows()] or ["  none"]
    lines += ["", "Attack signatures (top 3 single-feature separators vs benign):"]
    for c, g in sig.groupby("attack"):
        lines.append(f"  {c}  ({int(g['n_attack'].iloc[0]):,} flows)")
        for _, r in g.head(3).iterrows():
            lines.append(f"      {r['feature']:<30} sep {r['separation']:.2f}  {r['attack_is']:<6}"
                         f"  median {r['median_attack']:,.1f} vs benign {r['median_benign']:,.1f}")
    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\ndone in {time.time() - t0:.0f}s, wrote results/features/")


if __name__ == "__main__":
    main()
