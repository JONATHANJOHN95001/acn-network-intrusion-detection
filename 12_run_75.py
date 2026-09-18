"""
Step 12: run the 75 scripts and build the results sheet.

Each script in runs/ is executed on its own, exactly as it would be by hand,
and appends its row to results/75_runs.csv. This then writes the Excel sheet
in the required layout:

    Test | DR | Clas | Acc | Pre | Rec | F1

Precision, recall and F1 are macro-averaged, which weights all fifteen classes
equally. The weighted versions are kept in extra columns, since a model can
score well on the weighted average while detecting no rare attack at all.

    python 12_run_75.py                 run every script that has no result yet
    python 12_run_75.py --fresh         run all 75 again
    python 12_run_75.py --sheet-only    just rebuild the sheet from the CSV
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RUNS = HERE / "runs"
RESULTS = HERE / "results"
CSV = RESULTS / "75_runs.csv"
XLSX = RESULTS / "75_runs.xlsx"

KERNEL_ORDER = ["LIN", "POLY", "RBF", "SIG", "COS"]
CLF_ORDER = ["LR", "RF", "KNN", "SVM", "QDA"]
CLF_FULL = {"LR": "Logistic Regression (sheet #2)",
            "RF": "Random Forest (sheet #7)",
            "KNN": "k-Nearest Neighbors (sheet #13)",
            "SVM": "Support Vector Machine (sheet #15)",
            "QDA": "Quadratic Discriminant Analysis (sheet #17)"}


def build_sheet():
    if not CSV.exists():
        print("no results yet")
        return None
    d = pd.read_csv(CSV)
    d["_k"] = d["DR"].map({k: i for i, k in enumerate(KERNEL_ORDER)})
    d["_c"] = d["Clas"].map({c: i for i, c in enumerate(CLF_ORDER)})
    d = d.sort_values(["_c", "_k", "Test"]).drop(columns=["_k", "_c"])

    main = d[["Test", "DR", "Clas", "Acc", "Pre", "Rec", "F1"]]
    with pd.ExcelWriter(XLSX, engine="openpyxl") as xl:
        main.to_excel(xl, sheet_name="75 runs", index=False)
        d.to_excel(xl, sheet_name="full detail", index=False)
        # best setting per classifier, by macro F1
        best = (d.sort_values("F1", ascending=False)
                 .groupby("Clas", sort=False)
                 .head(1)[["Clas", "Test", "DR", "Acc", "Pre", "Rec", "F1"]])
        best["Classifier"] = best["Clas"].map(CLF_FULL)
        best.to_excel(xl, sheet_name="best per classifier", index=False)
        # kernel by classifier, macro F1 averaged over the three test sizes
        (d.pivot_table(index="Clas", columns="DR", values="F1", aggfunc="mean")
          .reindex(index=CLF_ORDER, columns=KERNEL_ORDER).round(4)
          .to_excel(xl, sheet_name="kernel x classifier"))
    print(f"\nwrote {XLSX.name}: {len(main)} rows")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="rerun scripts that already have a result")
    ap.add_argument("--sheet-only", action="store_true", help="only rebuild the Excel sheet")
    a = ap.parse_args()

    if a.sheet_only:
        build_sheet()
        return

    scripts = sorted(RUNS.glob("Test_*.py"))
    if not scripts:
        raise SystemExit("no scripts in runs/; run 11_make_75_scripts.py first")

    done = set()
    if CSV.exists() and not a.fresh:
        done = set(pd.read_csv(CSV)["script"])

    todo = [s for s in scripts if s.stem not in done]
    print(f"{len(scripts)} scripts, {len(done)} already done, {len(todo)} to run\n")

    t0 = time.time()
    failed = []
    for i, s in enumerate(todo, 1):
        t1 = time.time()
        r = subprocess.run([sys.executable, str(s)], capture_output=True, text=True)
        if r.returncode != 0:
            failed.append(s.stem)
            last = (r.stderr or "").strip().splitlines()
            print(f"  [{i:>2}/{len(todo)}] {s.stem:<28} FAILED: "
                  f"{last[-1][:90] if last else 'no output'}", flush=True)
            continue
        line = [x for x in r.stdout.splitlines() if x.startswith(("Accuracy", "F1 Score"))]
        vals = "  ".join(line).replace("Accuracy  : ", "acc ").replace("F1 Score  : ", "F1 ")
        print(f"  [{i:>2}/{len(todo)}] {s.stem:<28} {vals}   ({time.time() - t1:.0f}s)",
              flush=True)

    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min")
    if failed:
        print(f"{len(failed)} failed: {failed}")
    d = build_sheet()
    if d is not None:
        imgs = len(list(RUNS.glob("*.jpg")))
        print(f"images: {imgs} of {len(scripts)}")


if __name__ == "__main__":
    main()
