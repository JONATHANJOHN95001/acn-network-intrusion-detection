"""
Step 6: report charts.

Builds every chart whose data exists and skips the rest, so it can be rerun
as experiments finish. Output: results/charts/*.png at 200 dpi, sized for
the text width of an A4 page. Every value shown here is also in the CSV and
Excel results, which serve as the table version of each chart.

Usage:
    python 06_charts.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

import plot_style as ps

HERE = Path(__file__).parent
RES = HERE / "results"
PARQUET = HERE / "data" / "cicids2017.parquet"
W = 6.5                                   # inches, text width of an A4 page
KERNELS = ["linear", "poly", "rbf", "sigmoid", "cosine"]

MODEL_NAMES = {
    "naive_bayes": "Naive Bayes", "lda": "LDA", "qda": "QDA", "sgd": "SGD",
    "decision_tree": "Decision Tree", "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "random_forest": "Random Forest", "hist_gb": "Hist. Gradient Boosting",
    "catboost": "CatBoost", "adaboost": "AdaBoost", "mlp": "MLP (neural network)",
    "linear_svm": "Linear SVM", "logistic": "Logistic Regression", "knn": "KNN",
    "svm": "SVM (RBF)",
}
KPCA_NAMES = {"LR": "Logistic Regression", "SGD": "SGD", "DT": "Decision Tree",
              "RF": "Random Forest", "NB": "Naive Bayes", "KNN": "KNN", "SVM": "SVM (RBF)"}
SHORT = {"Web Attack - Brute Force": "Web brute force", "Web Attack - XSS": "Web XSS",
         "Web Attack - Sql Injection": "Web SQL injection"}
STRATEGY_NAMES = {"none": "No correction", "class_weight": "Class weights",
                  "undersample": "Undersample benign", "smote": "SMOTE",
                  "under+smote": "Undersample + SMOTE"}


def short(c):
    return SHORT.get(c, c)


def compact(v, _=None):
    for div, suffix in ((1e6, "M"), (1e3, "k")):
        if v >= div:
            s = f"{v / div:.1f}".rstrip("0").rstrip(".")
            return f"{s}{suffix}"
    return f"{v:,.0f}"


def label_at_tip(ax, text, v, y, dx=4, **kw):
    ax.annotate(text, (v, y), xytext=(dx, 0), textcoords="offset points",
                ha="left" if dx >= 0 else "right", va="center", fontsize=8, color=ps.INK2, **kw)


def grid_protocol():
    from common import pick_grid_protocol

    return pick_grid_protocol(RES / "results.csv")


def full_grid(split="60/40"):
    df, _ = grid_protocol()
    if df is None:
        return None
    df = df[df["split"] == split]
    return df if len(df) else None


def grid_label():
    return grid_protocol()[1] or "grid"


def class_counts():
    return pd.read_parquet(PARQUET, columns=["Label"])["Label"].value_counts()


# ---------------------------------------------------------------- dataset

def chart_class_distribution():
    counts = class_counts()
    n = len(counts)
    fig, ax = plt.subplots(figsize=(W, 0.25 * n + 1.35))
    top = ps.titles(fig, "Class sizes after cleaning",
                    f"Flows per class, log scale. Benign traffic outnumbers the rarest attack "
                    f"{round(counts.max() / counts.min()):,} to 1.")
    fig.subplots_adjust(left=0.22, right=0.95, top=top, bottom=0.1)
    ax.set_xscale("log")
    ax.set_xlim(1, 3e7)
    ax.set_ylim(n - 0.5, -0.5)
    for i, v in enumerate(counts.values):
        ps.bar(ax, 1, v, i, ps.BLUE)
        label_at_tip(ax, f"{v:,}", v, i)
    ax.set_yticks(range(n), [short(c) for c in counts.index])
    ax.xaxis.set_major_formatter(FuncFormatter(compact))
    ax.set_xlabel("Flows (log scale)")
    ps.quiet_axes(ax, "x")
    return ps.save(fig, "01_class_distribution")


def chart_duplicates():
    txt = (RES / "01_data_audit.txt").read_text(encoding="utf-8")
    sec4 = txt.split("[4]")[1].split("[5]")[0]
    sec5 = txt.split("[5]")[1]
    overall = re.search(r"\(([\d.]+%)\)", sec4).group(1)
    dups = {m.group(1).strip(): int(m.group(2).replace(",", ""))
            for m in re.finditer(r"^ {7}(\S.*?)\s{2,}([\d,]+)\s*$", sec4, re.M)}
    final = {m.group(1).strip(): int(m.group(2).replace(",", ""))
             for m in re.finditer(r"^ {4}(\S.*?)\s{2,}([\d,]+)\s+[\d.]+%\s*$", sec5, re.M)}
    share = pd.Series({c: dups.get(c, 0) / (final[c] + dups.get(c, 0)) for c in final})
    share = share.sort_values(ascending=False)
    n = len(share)
    fig, ax = plt.subplots(figsize=(W, 0.25 * n + 1.45))
    top = ps.titles(fig, "Rows removed as exact duplicates, by class",
                    f"{overall} of all rows were exact copies. Left in, copies land on both sides "
                    "of the\ntrain/test split, so the model is tested on rows it has already seen.")
    fig.subplots_adjust(left=0.22, right=0.95, top=top, bottom=0.1)
    ax.set_xlim(0, max(share.max() * 1.18, 0.05))
    ax.set_ylim(n - 0.5, -0.5)
    for i, v in enumerate(share.values):
        if v > 0:
            ps.bar(ax, 0, v, i, ps.BLUE)
        label_at_tip(ax, f"{v:.1%}" if v >= 0.001 else ("<0.1%" if v > 0 else "0%"), v, i)
    ax.set_yticks(range(n), [short(c) for c in share.index])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Share of the class's rows that were duplicates")
    ps.quiet_axes(ax, "x")
    return ps.save(fig, "02_duplicates_removed")


# ---------------------------------------------------------------- classifier comparison

def chart_accuracy_vs_macro_f1():
    df = full_grid()
    if df is None:
        return None
    df = df.sort_values("macro_f1", ascending=False)
    n = len(df)
    fig, ax = plt.subplots(figsize=(W, 0.3 * n + 1.6))
    top = ps.titles(fig, "Accuracy hides the rare attacks",
                    "Full dataset, 60/40 split. Accuracy is above 0.9 for most models; macro F1 "
                    "averages\nall 15 classes equally, so missed rare attacks pull it down.")
    fig.subplots_adjust(left=0.27, right=0.97, top=top - 0.04, bottom=0.08)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(n - 0.5, -0.5)
    for i, (acc, mf) in enumerate(zip(df["accuracy"], df["macro_f1"])):
        ax.plot([mf, acc], [i, i], color=ps.GRAY, lw=2 * ps.PX, solid_capstyle="round", zorder=2)
        if mf >= 0.1:
            label_at_tip(ax, f"{mf:.2f}", mf, i, dx=-7)
        else:
            label_at_tip(ax, f"{mf:.2f}", mf, i, dx=7)
    ax.scatter(df["accuracy"], range(n), s=64, color=ps.BLUE, edgecolors=ps.SURFACE,
               linewidths=1.5, zorder=3, label="Accuracy")
    ax.scatter(df["macro_f1"], range(n), s=64, color=ps.ORANGE, edgecolors=ps.SURFACE,
               linewidths=1.5, zorder=3, label="Macro F1 (labelled)")
    ax.set_yticks(range(n), [MODEL_NAMES.get(m, m) for m in df["model"]])
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2, handletextpad=0.2,
              columnspacing=1.5, borderaxespad=0.3)
    ps.quiet_axes(ax, "x")
    return ps.save(fig, "03_accuracy_vs_macro_f1")


def _heatmap(ax, M, row_labels, col_labels, fmt="{:.2f}", fontsize=6.8, bold_row_max=False):
    mesh = ax.pcolormesh(M, cmap=ps.SEQ, vmin=0, vmax=1, edgecolors=ps.SURFACE,
                         linewidth=1.5)
    ax.set_xlim(0, M.shape[1])
    ax.set_ylim(M.shape[0], 0)
    for r in range(M.shape[0]):
        rmax = np.nanmax(M[r]) if bold_row_max else None
        for c in range(M.shape[1]):
            v = M[r, c]
            if np.isnan(v):
                continue
            ax.text(c + 0.5, r + 0.5, fmt.format(v), ha="center", va="center",
                    fontsize=fontsize, color=ps.text_on(ps.SEQ(v)),
                    fontweight="bold" if bold_row_max and v == rmax else "normal")
    ax.set_xticks(np.arange(M.shape[1]) + 0.5, col_labels)
    ax.set_yticks(np.arange(M.shape[0]) + 0.5, row_labels)
    for s in ax.spines.values():
        s.set_visible(False)
    return mesh


def _colorbar(fig, mesh, ax, label):
    cb = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.015, aspect=30)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, length=0, colors=ps.INK2)
    cb.set_label(label, fontsize=8, color=ps.INK2)
    return cb


def chart_per_class_f1():
    df = full_grid()
    if df is None:
        return None
    counts = class_counts()
    df = df.sort_values("macro_f1", ascending=False)
    cols = [c for c in counts.index if f"f1::{c}" in df.columns]
    M = df[[f"f1::{c}" for c in cols]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(W, 0.3 * len(df) + 2.6))
    top = ps.titles(fig, "F1 score per model and class",
                    "Full dataset, 60/40 split. Classes run from most to fewest flows, left to "
                    "right; the rare\nattacks on the right are where models differ.")
    fig.subplots_adjust(left=0.24, right=0.93, top=top, bottom=0.2)
    mesh = _heatmap(ax, M, [MODEL_NAMES.get(m, m) for m in df["model"]],
                    [short(c) for c in cols], fontsize=6.3)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    _colorbar(fig, mesh, ax, "F1")
    return ps.save(fig, "04_per_class_f1")


def chart_test_size():
    f = RES / "results.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df[df["sample"].astype(str) == "full"]
    if df["split"].nunique() < 2:
        return None
    piv = df.pivot_table(index="model", columns="test_size", values="macro_f1")
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    tests = list(piv.columns)
    n = len(piv)
    fig, ax = plt.subplots(figsize=(W, 0.3 * n + 1.6))
    top = ps.titles(fig, "Test size changes little",
                    f"Macro F1 ({grid_label().lower()}) at each test size. Most models barely move "
                    "between\n20%, 40% and 60% of the data held out for testing.")
    fig.subplots_adjust(left=0.27, right=0.97, top=top - 0.04, bottom=0.08)
    ax.set_ylim(n - 0.5, -0.5)
    for i, (_, row) in enumerate(piv.iterrows()):
        ax.plot([row.min(), row.max()], [i, i], color=ps.GRAY, lw=2 * ps.PX,
                solid_capstyle="round", zorder=2)
    colors = ps.ORDINAL3[:len(tests)]
    for t, col in zip(tests, colors):
        ax.scatter(piv[t], range(n), s=56, color=col, edgecolors=ps.SURFACE, linewidths=1.5,
                   zorder=3, label=f"Test size {t:g}")
    ax.set_yticks(range(n), [MODEL_NAMES.get(m, m) for m in piv.index])
    ax.set_xlim(0, 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlabel("Macro F1")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=3, handletextpad=0.2,
              columnspacing=1.5, borderaxespad=0.3)
    ps.quiet_axes(ax, "x")
    return ps.save(fig, "05_test_size")


def chart_throughput():
    df = full_grid()
    if df is None:
        return None
    df = df.sort_values("flows_per_sec", ascending=False)
    n = len(df)
    fig, ax = plt.subplots(figsize=(W, 0.27 * n + 1.45))
    top = ps.titles(fig, "How many flows each model can classify per second",
                    "Prediction speed on one laptop, log scale. An IDS on a busy link must keep "
                    "up with\nthe flow rate, so speed matters as much as accuracy.")
    fig.subplots_adjust(left=0.27, right=0.93, top=top, bottom=0.1)
    ax.set_xscale("log")
    ax.set_xlim(1e3, df["flows_per_sec"].max() * 12)
    ax.set_ylim(n - 0.5, -0.5)
    for i, v in enumerate(df["flows_per_sec"]):
        ps.bar(ax, 1e3, v, i, ps.BLUE)
        label_at_tip(ax, compact(v), v, i)
    ax.set_yticks(range(n), [MODEL_NAMES.get(m, m) for m in df["model"]])
    ax.xaxis.set_major_formatter(FuncFormatter(compact))
    ax.set_xlabel("Flows classified per second (log scale)")
    ps.quiet_axes(ax, "x")
    return ps.save(fig, "06_throughput")


# ---------------------------------------------------------------- kernel PCA

def _kpca():
    f = RES / "kernel_pca" / "kpca_all_s10000.csv"
    return pd.read_csv(f) if f.exists() else None


def chart_kpca_kernels():
    d = _kpca()
    if d is None:
        return None
    red = d[d["components"] > 0]
    piv = red.pivot_table(index="classifier", columns="kernel", values="macro_f1",
                          aggfunc="mean").reindex(columns=KERNELS)
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(W, 0.36 * len(piv) + 1.75))
    top = ps.titles(fig, "Which Kernel PCA kernel suits which classifier",
                    "Macro F1 averaged over test sizes 0.2 / 0.4 / 0.6 and 5 / 10 / 15 "
                    "components.\nBold marks the best kernel for each classifier.")
    fig.subplots_adjust(left=0.24, right=0.9, top=top, bottom=0.08)
    mesh = _heatmap(ax, piv.to_numpy(), [KPCA_NAMES.get(c, c) for c in piv.index],
                    KERNELS, fmt="{:.3f}", fontsize=8, bold_row_max=True)
    ax.xaxis.tick_top()
    _colorbar(fig, mesh, ax, "Macro F1")
    return ps.save(fig, "07_kpca_kernel_by_classifier")


def chart_kpca_best_vs_none():
    d = _kpca()
    if d is None:
        return None
    rows = []
    for clf, g in d.groupby("classifier"):
        best = g[g["components"] > 0].sort_values("macro_f1", ascending=False).iloc[0]
        base = g[(g["components"] == 0) & (g["test_size"] == best["test_size"])].iloc[0]
        rows.append((clf, base["macro_f1"], best["macro_f1"],
                     f"{best['kernel']}, {int(best['components'])} comp."))
    rows.sort(key=lambda r: r[2] - r[1], reverse=True)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(W, 0.38 * n + 1.7))
    top = ps.titles(fig, "Kernel PCA helps only KNN and SVM",
                    "Macro F1 with all 69 features versus the best Kernel PCA setting, at the "
                    "same test size.")
    fig.subplots_adjust(left=0.24, right=0.8, top=top - 0.05, bottom=0.08)
    ax.set_xlim(0, 1)
    ax.set_ylim(n - 0.5, -0.5)
    for i, (_, base, best, setting) in enumerate(rows):
        ax.plot([base, best], [i, i], color=ps.GRAY, lw=2 * ps.PX, solid_capstyle="round",
                zorder=2)
        delta = best - base
        ax.annotate(f"{delta:+.3f}   {setting}", (1, i), xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8,
                    color=ps.INK if delta > 0 else ps.INK2,
                    annotation_clip=False, **(ps.SEMIBOLD if delta > 0 else {}))
    ax.scatter([r[1] for r in rows], range(n), s=64, color=ps.PAIR[0], edgecolors=ps.SURFACE,
               linewidths=1.5, zorder=3, label="All 69 features (no reduction)")
    ax.scatter([r[2] for r in rows], range(n), s=64, color=ps.PAIR[1], edgecolors=ps.SURFACE,
               linewidths=1.5, zorder=3, label="Best Kernel PCA setting")
    ax.set_yticks(range(n), [KPCA_NAMES.get(r[0], r[0]) for r in rows])
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlabel("Macro F1")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2, handletextpad=0.2,
              columnspacing=1.5, borderaxespad=0.3)
    ps.quiet_axes(ax, "x")
    return ps.save(fig, "08_kpca_best_vs_no_reduction")


def chart_kpca_components():
    d = _kpca()
    if d is None:
        return None
    levels = [5, 10, 15, 0]
    names = ["5 components", "10 components", "15 components", "All 69 features"]
    means = d.pivot_table(index="classifier", columns="components", values="macro_f1",
                          aggfunc="mean")[levels]
    means = means.loc[means[0].sort_values(ascending=False).index]
    n = len(means)
    fig, ax = plt.subplots(figsize=(W, 3.6))
    top = ps.titles(fig, "More components help every classifier",
                    "Macro F1 averaged over kernels and test sizes; the darkest bar is the "
                    "unreduced baseline.")
    fig.subplots_adjust(left=0.08, right=0.98, top=top - 0.07, bottom=0.12)
    thick, gap = 9, 2
    for i, (_, row) in enumerate(means.iterrows()):
        for k, (lvl, col) in enumerate(zip(levels, ps.ORDINAL4)):
            off = (k - (len(levels) - 1) / 2) * (thick + gap)
            ps.bar(ax, 0, row[lvl], i, col, thick_px=thick, offset_px=off, horizontal=False)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(n), [KPCA_NAMES.get(c, c).replace(" ", "\n", 1) for c in means.index])
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_ylabel("Macro F1")
    handles = [Line2D([], [], marker="s", linestyle="", markersize=7, color=c) for c in ps.ORDINAL4]
    ax.legend(handles, names, loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=4,
              handletextpad=0.2, columnspacing=1.4, borderaxespad=0.3)
    ps.quiet_axes(ax, "y")
    return ps.save(fig, "09_kpca_components")


# ---------------------------------------------------------------- features and imbalance

def chart_feature_importance():
    f = RES / "features" / "importance.csv"
    if not f.exists():
        return None
    imp = pd.read_csv(f).head(15)
    n = len(imp)
    fig, ax = plt.subplots(figsize=(W, 0.27 * n + 1.5))
    top = ps.titles(fig, "The flow features the model relies on most",
                    "Drop in macro F1 when each feature is shuffled (Random Forest, held-out "
                    "data).\nThin lines show the spread over repeated shuffles.")
    fig.subplots_adjust(left=0.3, right=0.95, top=top, bottom=0.1)
    hi = (imp["permutation_mean"] + imp["permutation_std"]).max()
    ax.set_xlim(0, hi * 1.2)
    ax.set_ylim(n - 0.5, -0.5)
    for i, (_, r) in enumerate(imp.iterrows()):
        ps.bar(ax, 0, max(r["permutation_mean"], 0), i, ps.BLUE)
        lo_, hi_ = r["permutation_mean"] - r["permutation_std"], r["permutation_mean"] + r["permutation_std"]
        ax.plot([lo_, hi_], [i, i], color=ps.INK2, lw=ps.PX, zorder=4)
        label_at_tip(ax, f"{r['permutation_mean']:.3f}", hi_, i)
    ax.set_yticks(range(n), imp["feature"])
    ax.set_xlabel("Drop in macro F1 when shuffled")
    ps.quiet_axes(ax, "x")
    return ps.save(fig, "10_feature_importance")


def _imbalance():
    d = RES / "imbalance"
    f = d / "imbalance.csv"
    if not f.exists():                      # fall back to the sample run
        cands = sorted(d.glob("imbalance_s*.csv"), key=lambda p: p.stat().st_size)
        if not cands:
            return None
        f = cands[-1]
    return pd.read_csv(f)


def chart_imbalance():
    d = _imbalance()
    if d is None:
        return None
    models = list(dict.fromkeys(d["model"]))
    strategies = list(dict.fromkeys(d["strategy"]))
    fig, axes = plt.subplots(1, len(models), figsize=(W, 0.3 * len(strategies) + 1.9),
                             sharey=True)
    top = ps.titles(fig, "Which imbalance fix helps each model",
                    "Macro F1 on the untouched test split (full dataset, 60/40). The best "
                    "strategy per model is highlighted.")
    fig.subplots_adjust(left=0.2, right=0.97, top=top - 0.06, bottom=0.12, wspace=0.18)
    for ax, m in zip(np.atleast_1d(axes), models):
        g = d[d["model"] == m].set_index("strategy").reindex(strategies)
        best = g["macro_f1"].idxmax()
        ax.set_xlim(0, 1.18)
        ax.set_ylim(len(strategies) - 0.5, -0.5)
        for i, s in enumerate(strategies):
            v = g.loc[s, "macro_f1"]
            ps.bar(ax, 0, v, i, ps.BLUE if s == best else ps.GRAY, thick_px=14)
            label_at_tip(ax, f"{v:.3f}", v, i, **(ps.SEMIBOLD if s == best else {}))
        ax.set_title(MODEL_NAMES.get(m, m), fontsize=9, color=ps.INK, loc="left", pad=4)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ps.quiet_axes(ax, "x")
    np.atleast_1d(axes)[0].set_yticks(range(len(strategies)),
                                      [STRATEGY_NAMES.get(s, s) for s in strategies])
    return ps.save(fig, "11_imbalance_macro_f1")


def chart_imbalance_rare_recall():
    d = _imbalance()
    if d is None:
        return None
    rare = ["Heartbleed", "Web Attack - Sql Injection", "Infiltration", "Web Attack - XSS",
            "Web Attack - Brute Force", "Bot"]
    d = d.copy()
    d["row"] = [f"{MODEL_NAMES.get(m, m)}  /  {STRATEGY_NAMES.get(s, s)}"
                for m, s in zip(d["model"], d["strategy"])]
    M = d[[f"recall::{c}" for c in rare]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(W, 0.27 * len(d) + 1.9))
    top = ps.titles(fig, "Recall on the six rarest attacks",
                    "Share of each rare attack that was detected, per model and imbalance "
                    "strategy.")
    fig.subplots_adjust(left=0.38, right=0.92, top=top - 0.02, bottom=0.05)
    mesh = _heatmap(ax, M, d["row"], [short(c) for c in rare], fontsize=7.2)
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=25, ha="left", rotation_mode="anchor")
    for y in range(len(set(d["strategy"])), len(d), len(set(d["strategy"]))):
        ax.axhline(y, color=ps.SURFACE, lw=4)
    _colorbar(fig, mesh, ax, "Recall")
    return ps.save(fig, "12_imbalance_rare_recall")


def main():
    ps.setup()
    charts = [chart_class_distribution, chart_duplicates, chart_accuracy_vs_macro_f1,
              chart_per_class_f1, chart_test_size, chart_throughput, chart_kpca_kernels,
              chart_kpca_best_vs_none, chart_kpca_components, chart_feature_importance,
              chart_imbalance, chart_imbalance_rare_recall]
    for fn in charts:
        try:
            path = fn()
        except Exception as e:  # keep going; report which chart broke
            print(f"  FAILED  {fn.__name__}: {type(e).__name__}: {e}")
            continue
        print(f"  {'wrote  ' + path.name if path else 'skipped ' + fn.__name__ + ' (no data yet)'}")


if __name__ == "__main__":
    main()
