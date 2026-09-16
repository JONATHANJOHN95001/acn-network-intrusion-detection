"""
Shared helpers for the experiment scripts.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PARQUET = HERE / "data" / "cicids2017.parquet"


def load_sample(n, floor=50, seed=0):
    """Stratified sample of about n flows: each class in proportion to its
    size, but at least `floor` rows of every class (or all of it, if the
    class is smaller). Without the floor, Heartbleed (11 rows in 2.5M) would
    round down to zero. n=None returns the full dataset.

    The rows are chosen from the Label column alone, then the Parquet file is
    streamed in batches keeping only those rows. Loading all 2.5M rows at
    once needs about 1 GB, which a laptop with a browser open may not have."""
    import pyarrow.parquet as pq

    if n is None:
        return pd.read_parquet(PARQUET)

    labels = pd.read_parquet(PARQUET, columns=["Label"])["Label"]
    frac = n / len(labels)
    rng = np.random.default_rng(seed)
    keep = []
    for idx in labels.groupby(labels).indices.values():
        k = max(int(round(len(idx) * frac)), min(floor, len(idx)))
        keep.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    del labels

    parts, start = [], 0
    for batch in pq.ParquetFile(PARQUET).iter_batches(batch_size=100_000):
        end = start + batch.num_rows
        lo, hi = np.searchsorted(keep, [start, end])
        if hi > lo:
            parts.append(batch.take(keep[lo:hi] - start).to_pandas())
        start = end
    df = pd.concat(parts, ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def pick_grid_protocol(results_csv):
    """Results may hold runs from more than one protocol (the full dataset and a
    sample). Return (rows, label) for whichever is most complete, preferring the
    full dataset on a tie, so every output describes the same set of runs."""
    import pandas as pd

    if not results_csv.exists():
        return None, ""
    df = pd.read_csv(results_csv)
    best, best_n, label = None, 0, ""
    for key, g in df.groupby(df["sample"].astype(str)):
        if len(g) > best_n or (len(g) == best_n and key == "full"):
            best, best_n, label = g, len(g), key
    if best is None:
        return None, ""
    return best, ("Full dataset" if label == "full"
                  else f"{int(label):,}-flow stratified sample")
