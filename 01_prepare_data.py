"""
Step 1: CIC-IDS2017 preparation and audit.

Reads the eight raw CICFlowMeter CSVs, cleans them, and writes a single
Parquet file plus a human-readable audit report.

Cleaning applied, in order:
  1. strip whitespace from column names
  2. drop the duplicated 'Fwd Header Length.1' column (byte-identical to its twin)
  3. coerce every feature to numeric, turning +/-inf into NaN
  4. normalise the Label text (the raw files use a non-UTF-8 dash)
  5. tag each row with the capture day it came from
  6. drop zero-variance columns, measured across the whole dataset
  7. drop exact duplicate rows, which otherwise leak between train and test
  8. downcast features to float32

Where the CSVs can be:
  - in the project folder, or in a MachineLearningCVE/ subfolder (the folder
    name inside the official download), or anywhere given with --data-dir
  - under their official names (Monday-WorkingHours.pcap_ISCX.csv ...) or the
    renamed, day-numbered names (01_Monday-WorkingHours-Benign.csv ...)

Usage:
    python 01_prepare_data.py                     # find the CSVs, clean, write Parquet
    python 01_prepare_data.py --check             # only show which file was found for each day
    python 01_prepare_data.py --data-dir D:/cicids2017
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
OUT_PARQUET = HERE / "data" / "cicids2017.parquet"
OUT_REPORT = HERE / "results" / "01_data_audit.txt"

# (official name, renamed name). The order of the days fixes the row order,
# which decides which copy of a duplicate row is kept, so keep it stable.
FILES = {
    "Monday":      ("Monday-WorkingHours.pcap_ISCX.csv",
                    "01_Monday-WorkingHours-Benign.csv"),
    "Tuesday":     ("Tuesday-WorkingHours.pcap_ISCX.csv",
                    "02_Tuesday-WorkingHours-BruteForce.csv"),
    "Wednesday":   ("Wednesday-workingHours.pcap_ISCX.csv",
                    "03_Wednesday-WorkingHours-DoS.csv"),
    "Thursday-AM": ("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                    "04_Thursday-WorkingHours-Morning-WebAttacks.csv"),
    "Thursday-PM": ("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
                    "05_Thursday-WorkingHours-Afternoon-Infiltration.csv"),
    "Friday-AM":   ("Friday-WorkingHours-Morning.pcap_ISCX.csv",
                    "06_Friday-WorkingHours-Morning-Bot.csv"),
    "Friday-PM-DDoS":     ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
                           "08_Friday-WorkingHours-Afternoon-DDoS.csv"),
    "Friday-PM-PortScan": ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                           "07_Friday-WorkingHours-Afternoon-PortScan.csv"),
}

log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def find_csvs(data_dir=None):
    """Return {day: path}, trying every folder and both names for each day."""
    dirs = ([Path(data_dir)] if data_dir else []) + [HERE, HERE / "MachineLearningCVE"]
    found, missing = {}, []
    for day, names in FILES.items():
        hit = next((d / n for d in dirs for n in names if (d / n).is_file()), None)
        if hit:
            found[day] = hit
        else:
            missing.append(day)
    if missing:
        lines = [f"Missing CSV for: {', '.join(missing)}", "Looked in:"]
        lines += [f"  {d}" for d in dirs]
        lines += ["for either of these names:"]
        lines += [f"  {FILES[day][0]}  or  {FILES[day][1]}" for day in missing]
        raise SystemExit("\n".join(lines))
    return found


def clean_label(s: pd.Series) -> pd.Series:
    # The raw Web Attack labels contain byte 0x96 (an en dash in cp1252),
    # which latin-1 decodes to a control character. Normalise to plain ASCII.
    return (
        s.astype("string")
        .str.replace(r"[^\x20-\x7E]+", "-", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="folder holding the eight CSVs (searched first)")
    ap.add_argument("--check", action="store_true",
                    help="only report which file was found for each day")
    a = ap.parse_args()

    paths = find_csvs(a.data_dir)
    if a.check:
        for day, p in paths.items():
            print(f"{day:<20} {p.name}   ({p.stat().st_size/1e6:,.0f} MB)")
        print(f"\nall {len(paths)} files found")
        return

    t0 = time.time()
    log("=" * 72)
    log("CIC-IDS2017 PREPARATION AND AUDIT")
    log("=" * 72)

    frames = []
    log("\n[1] Reading raw CSVs")
    log(f"    {'day':<20} {'rows':>10} {'cols':>6} {'seconds':>8}")

    for day, path in paths.items():
        t = time.time()
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)
        df.columns = df.columns.str.strip()

        # 2. the duplicated header. pandas renames the second occurrence to '.1'
        dupe_cols = [c for c in df.columns if c.endswith(".1")]
        df = df.drop(columns=dupe_cols)

        label = clean_label(df["Label"])
        feats = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce")

        # 3. infinities become NaN so they can be imputed later
        feats = feats.replace([np.inf, -np.inf], np.nan)

        out = feats.astype("float32")
        out["Label"] = label
        out["Day"] = day
        frames.append(out)
        log(f"    {day:<20} {len(df):>10,} {df.shape[1]:>6} {time.time()-t:>8.1f}")

    data = pd.concat(frames, ignore_index=True)
    del frames
    raw_rows = len(data)
    log(f"\n    combined: {raw_rows:,} rows x {data.shape[1]} cols")

    feature_cols = [c for c in data.columns if c not in ("Label", "Day")]

    # 6. zero-variance columns, measured on the full dataset
    log("\n[2] Zero-variance features (dropped)")
    nun = data[feature_cols].nunique(dropna=False)
    constant = list(nun[nun <= 1].index)
    for c in constant:
        log(f"    - {c}")
    log(f"    {len(constant)} dropped, {len(feature_cols) - len(constant)} features remain")
    data = data.drop(columns=constant)
    feature_cols = [c for c in data.columns if c not in ("Label", "Day")]

    # missing values
    log("\n[3] Missing / infinite values (left as NaN for the imputer)")
    na = data[feature_cols].isna().sum()
    na = na[na > 0]
    if len(na) == 0:
        log("    none")
    for c, n in na.items():
        log(f"    {c:<28} {n:>8,}  ({n/raw_rows*100:.4f}%)")

    # 7. exact duplicates, the leakage source
    log("\n[4] Exact duplicate rows (dropped: they leak across the split)")
    dup_mask = data.duplicated()
    n_dup = int(dup_mask.sum())
    log(f"    {n_dup:,} of {raw_rows:,}  ({n_dup/raw_rows*100:.2f}%)")
    dup_by_label = data.loc[dup_mask, "Label"].value_counts()
    for lab, n in dup_by_label.items():
        log(f"       {lab:<32} {n:>9,}")
    data = data[~dup_mask].reset_index(drop=True)

    log("\n[5] Final class distribution")
    log(f"    {'label':<32} {'rows':>10} {'share':>9}")
    vc = data["Label"].value_counts()
    for lab, n in vc.items():
        log(f"    {lab:<32} {n:>10,} {n/len(data)*100:>8.4f}%")
    log(f"    {'TOTAL':<32} {len(data):>10,}")
    log(f"\n    classes: {len(vc)}   imbalance ratio: {vc.max()/vc.min():,.0f} : 1")

    log("\n[6] Writing Parquet")
    OUT_PARQUET.parent.mkdir(exist_ok=True)
    data.to_parquet(OUT_PARQUET, index=False, compression="snappy")
    mb = OUT_PARQUET.stat().st_size / 1e6
    csv_mb = sum(p.stat().st_size for p in paths.values()) / 1e6
    log(f"    {OUT_PARQUET.name}: {mb:,.0f} MB  (from {csv_mb:,.0f} MB of CSV)")
    log(f"    shape: {data.shape[0]:,} rows x {len(feature_cols)} features + Label + Day")
    log(f"\nDone in {time.time()-t0:.0f}s")

    OUT_REPORT.parent.mkdir(exist_ok=True)
    OUT_REPORT.write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
