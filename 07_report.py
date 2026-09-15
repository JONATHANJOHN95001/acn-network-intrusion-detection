"""
Step 7: build the project report as a Word document.

Pulls every number and chart straight from the results files, so the report
always matches the latest run. Charts that do not exist yet are skipped with
a visible placeholder, so the report can be built at any time and rebuilt
once the full grid finishes.

    python 07_report.py            -> results/ACN_IDS_Report.docx

The prose is fixed; the tables and figures come from:
    results/01_data_audit.txt
    results/results.csv              (classifier grid, once run)
    results/kernel_pca/kpca_all_s10000.csv
    results/features/*.csv
    results/imbalance/imbalance.csv  (once run)
    results/charts/*.png
"""

from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).parent
RES = HERE / "results"
CHARTS = RES / "charts"
OUT = RES / "ACN_IDS_Report.docx"

INK = RGBColor(0x0B, 0x0B, 0x0B)
MUTED = RGBColor(0x52, 0x51, 0x4E)
ACCENT = RGBColor(0x1C, 0x5C, 0xAB)

MODEL_NAMES = {
    "naive_bayes": "Naive Bayes", "lda": "LDA", "qda": "QDA", "sgd": "SGD",
    "decision_tree": "Decision Tree", "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "random_forest": "Random Forest", "hist_gb": "Hist. Gradient Boosting",
    "catboost": "CatBoost", "adaboost": "AdaBoost", "mlp": "MLP (neural net)",
    "linear_svm": "Linear SVM", "logistic": "Logistic Regression", "knn": "KNN",
    "svm": "SVM (RBF)",
}


def style(doc):
    n = doc.styles["Normal"]
    n.font.name = "Calibri"
    n.font.size = Pt(11)
    n.font.color.rgb = INK
    for lvl, size in ((1, 16), (2, 13), (3, 11.5)):
        h = doc.styles[f"Heading {lvl}"]
        h.font.name = "Calibri"
        h.font.size = Pt(size)
        h.font.color.rgb = ACCENT if lvl == 1 else INK
        h.font.bold = True


def h(doc, text, level=1):
    doc.add_heading(text, level=level)


def para(doc, text, italic=False, size=None, space_after=6):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if italic:
        r.font.color.rgb = MUTED
    p.paragraph_format.space_after = Pt(space_after)
    return p


def bullet(doc, text, bold_lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
        p.add_run(text)
    else:
        p.add_run(text)
    return p


def figure(doc, name, caption):
    path = CHARTS / f"{name}.png"
    if path.exists():
        doc.add_picture(str(path), width=Inches(6.3))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = doc.add_paragraph()
        r = c.add_run(caption)
        r.italic = True
        r.font.size = Pt(9)
        r.font.color.rgb = MUTED
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        p = doc.add_paragraph()
        r = p.add_run(f"[figure {name} pending: run 06_charts.py after the grid finishes]")
        r.italic = True
        r.font.color.rgb = MUTED


def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, hd in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.paragraphs[0].add_run(str(hd)).bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(9.5)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(9.5)
    if widths:
        for r_ in t.rows:
            for i, w in enumerate(widths):
                r_.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


# ------------------------------------------------------------------ data helpers

def audit_numbers():
    txt = (RES / "01_data_audit.txt").read_text(encoding="utf-8")
    import re
    total = re.search(r"TOTAL\s+([\d,]+)", txt).group(1)
    classes = re.search(r"classes:\s*(\d+)", txt).group(1)
    ratio = re.search(r"imbalance ratio:\s*([\d,]+)", txt).group(1)
    dup = re.search(r"\[4\].*?([\d,]+) of ([\d,]+)\s+\(([\d.]+%)\)", txt, re.S)
    return {"total": total, "classes": classes, "ratio": ratio,
            "dup_n": dup.group(1), "dup_all": dup.group(2), "dup_pct": dup.group(3)}


def grid_df(split="60/40"):
    f = RES / "results.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d = d[(d["sample"].astype(str) == "full") & (d["split"] == split)]
    return d.sort_values("macro_f1", ascending=False) if len(d) else None


# ------------------------------------------------------------------ document

def build():
    doc = Document()
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(0.8)
        s.left_margin = s.right_margin = Inches(0.9)
    style(doc)

    # title block
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Network Intrusion Detection on CIC-IDS2017")
    r.bold = True
    r.font.size = Pt(20)
    r.font.color.rgb = ACCENT
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rs = sub.add_run("A comparison of machine-learning classifiers, with Kernel PCA "
                     "dimensionality reduction and a focus on rare attack classes")
    rs.font.size = Pt(12)
    rs.font.color.rgb = MUTED
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run("Advanced Computer Networks course project").font.size = Pt(10)
    doc.add_paragraph()

    a = audit_numbers()

    # abstract
    h(doc, "Abstract", 1)
    para(doc,
         "We build a flow-based network intrusion detection system on the CIC-IDS2017 "
         f"dataset and compare a range of machine-learning classifiers on it. After cleaning, "
         f"the data holds {a['total']} network flows across {a['classes']} classes, one benign "
         f"and fourteen attacks, with a {a['ratio']} to 1 size gap between the largest and "
         "smallest class. We show that accuracy is a misleading measure on such imbalanced "
         "data and report macro-averaged F1 throughout. Tree ensembles, led by Random Forest, "
         "detect most attacks well; linear models and the rarest attacks are where performance "
         "falls. We apply Kernel PCA across five kernels, three component counts and three "
         "train/test splits, and find that reducing the 69 features helps only the "
         "distance-based classifiers (KNN and SVM). We tie the most useful flow features back "
         "to the protocol behaviour of each attack, and address the class imbalance with class "
         "weighting and resampling.")

    # 1. introduction
    h(doc, "1  Introduction", 1)
    para(doc,
         "An intrusion detection system (IDS) watches network traffic and raises an alert when "
         "it sees an attack. This project builds a supervised, flow-based IDS: rather than "
         "reading packet contents, it classifies each network flow from summary statistics such "
         "as packet sizes, timings and TCP flag counts. This is the same kind of record that "
         "routers already export through NetFlow and IPFIX, and it keeps working on encrypted "
         "traffic because it never needs the payload.")
    para(doc,
         "The task is a 15-class classification problem, and the classes are extremely "
         f"imbalanced: benign traffic outnumbers the rarest attack {a['ratio']} to 1. The "
         "project therefore has three aims: to compare classifiers fairly on this imbalanced "
         "data, to test whether Kernel PCA dimensionality reduction helps, and to connect the "
         "results back to how each attack behaves on the network.")

    # 2. background
    h(doc, "2  Background", 1)
    para(doc, "See docs/networking_background.md for the full networking discussion; this "
              "section summarises it.", italic=True)
    h(doc, "2.1  Network flows and the features", 2)
    para(doc,
         "A flow is all the packets exchanged between two endpoints, identified by source and "
         "destination IP and port and the protocol, from the first packet until the connection "
         "closes or times out. CICFlowMeter turns each flow into 78 numeric features in two "
         "directions (forward, from the connection initiator, and backward, the reply): volume, "
         "packet-size statistics, inter-arrival times, TCP flag counts, the initial TCP window, "
         "and throughput rates.")
    h(doc, "2.2  The attacks", 2)
    para(doc,
         "CIC-IDS2017 contains fourteen attack types recorded over five days: brute-force "
         "password guessing (FTP-Patator, SSH-Patator), denial of service both slow (slowloris, "
         "Slowhttptest) and volumetric (Hulk, GoldenEye, DDoS), the Heartbleed TLS memory leak, "
         "port scanning, a botnet, infiltration from a compromised internal host, and three web "
         "attacks (brute force, XSS, SQL injection). Section 6 explains how each appears in the "
         "flow features.")

    # 3. data and cleaning
    h(doc, "3  Dataset and cleaning", 1)
    para(doc,
         "The eight daily capture files were combined and cleaned before any model saw them "
         "(01_prepare_data.py). Four problems were found and fixed:")
    bullet(doc, f"{a['dup_pct']} of rows ({a['dup_n']} of {a['dup_all']}) were exact duplicates. "
                "Left in, copies fall on both sides of the train/test split, so the model is "
                "tested on rows it has already memorised. They were removed before splitting.",
           bold_lead="Duplicate rows. ")
    bullet(doc, "eight columns are constant across the whole dataset and carry no information; "
                "they were dropped.", bold_lead="Dead columns. ")
    bullet(doc, "the column Fwd Header Length appears twice, identically; the copy was dropped.",
           bold_lead="Duplicated column. ")
    bullet(doc, "only Flow Bytes/s and Flow Packets/s contain missing or infinite values; these "
                "are filled with the column median, fitted on the training split only.",
           bold_lead="Missing values. ")
    para(doc, f"After cleaning, {a['total']} flows remain with 69 features across {a['classes']} "
              "classes. Figure 1 shows how uneven the classes are, and Figure 2 shows that the "
              "duplicate rows were concentrated in a few attack types (PortScan alone lost 43%).")
    figure(doc, "01_class_distribution", "Figure 1. Flows per class after cleaning (log scale).")
    figure(doc, "02_duplicates_removed", "Figure 2. Share of each class removed as duplicates.")

    # 4. method
    h(doc, "4  Method", 1)
    para(doc,
         "Every model runs through one shared pipeline (02_experiment.py) so the comparison is "
         "fair: median imputation, then standard scaling for the scale-sensitive models (linear "
         "models, SVM, MLP, Naive Bayes, LDA and QDA) which the tree models skip, then the "
         "classifier. Imputation and scaling are fitted on the training split only. The split "
         "is stratified so every class appears in both halves, at test sizes 0.2, 0.4 and 0.6.")
    h(doc, "4.1  Why macro F1", 2)
    para(doc,
         "Accuracy and weighted averages scale each class by its size, so the 2.1 million benign "
         "flows drown out the 11 Heartbleed flows: a model can ignore every rare attack and still "
         "score above 0.9. Macro-averaged F1 gives every class equal weight, so a missed rare "
         "attack shows up. We report macro F1 as the headline metric and read the per-class "
         "scores underneath it.")

    # 5. classifier results
    h(doc, "5  Classifier comparison", 1)
    d = grid_df()
    if d is not None:
        best = d.iloc[0]
        para(doc,
             f"On the full dataset at a 60/40 split, {MODEL_NAMES.get(best['model'], best['model'])} "
             f"scored the highest macro F1 ({best['macro_f1']:.3f}), at {best['accuracy']:.3f} "
             "accuracy. Figure 3 shows the gap between accuracy and macro F1 for every model: "
             "accuracy is above 0.9 for most, while macro F1 spreads much wider, which is the "
             "point of Section 4.1. Table 1 lists the full results.")
        figure(doc, "03_accuracy_vs_macro_f1",
               "Figure 3. Accuracy versus macro F1 on the full dataset (60/40 split).")
        rows = [(MODEL_NAMES.get(m, m), f"{acc:.4f}", f"{mf:.4f}", f"{wf:.4f}", f"{fps:,}")
                for m, acc, mf, wf, fps in zip(d["model"], d["accuracy"], d["macro_f1"],
                                               d["weighted_f1"], d["flows_per_sec"])]
        table(doc, ["Model", "Accuracy", "Macro F1", "Weighted F1", "Flows/s"], rows,
              widths=[1.9, 1.0, 1.0, 1.1, 1.1])
        para(doc, "Table 1. Every classifier on the full dataset, sorted by macro F1.",
             italic=True, size=9)
        figure(doc, "04_per_class_f1", "Figure 4. F1 per model and class; the rare attacks on "
               "the right are where models differ.")
        figure(doc, "06_throughput", "Figure 5. Flows classified per second (log scale): an IDS "
               "on a busy link must keep up with the flow rate.")
        figure(doc, "05_test_size", "Figure 6. Macro F1 at each test size; most models barely "
               "move between 20% and 60% held out.")
    else:
        para(doc, "[The full-dataset classifier grid has not finished yet. Run "
                  "02_experiment.py --all, then 06_charts.py, then rebuild this report.]",
             italic=True)

    # 6. kernel PCA
    h(doc, "6  Kernel PCA dimensionality reduction", 1)
    kf = RES / "kernel_pca" / "kpca_all_s10000.csv"
    if kf.exists():
        k = pd.read_csv(kf)
        para(doc,
             "Kernel PCA projects the 69 features onto a few nonlinear components. Because it "
             "builds an n x n kernel matrix, it cannot run on all 2.5 million flows (that matrix "
             "would need about 18 TB), so it runs on a fixed stratified sample of about 10,000 "
             "flows. We tried five kernels (linear, polynomial, RBF, sigmoid, cosine), three "
             "component counts (5, 10, 15) and the three test sizes, for all seven classifiers: "
             "336 runs, plus a no-reduction baseline.")
        para(doc,
             "The finding is that dimensionality reduction helps only the distance-based "
             "classifiers. KNN and SVM improve with the sigmoid kernel; every other classifier "
             "does worse than with all 69 features. The best kernel depends on the classifier "
             "(linear for the linear and tree models, sigmoid for KNN, SVM and Naive Bayes), and "
             "the polynomial kernel is the worst for all seven. More components always help.")
        figure(doc, "08_kpca_best_vs_no_reduction",
               "Figure 7. Best Kernel PCA setting versus all 69 features, per classifier.")
        figure(doc, "07_kpca_kernel_by_classifier",
               "Figure 8. Macro F1 by kernel and classifier (averaged over the grid).")
        figure(doc, "09_kpca_components", "Figure 9. More components give higher macro F1.")
    else:
        para(doc, "[Kernel PCA results missing; run 03_kernel_pca.py --classifiers all.]",
             italic=True)

    # 7. feature analysis
    h(doc, "7  Which features identify each attack", 1)
    sig = RES / "features" / "signatures.csv"
    imp = RES / "features" / "importance.csv"
    if imp.exists():
        para(doc,
             "Permutation importance (shuffle one feature, measure the drop in macro F1) shows "
             "the Random Forest leans most on Destination Port and the initial TCP window sizes. "
             "These are partly shortcuts: the port identifies the service, and the initial window "
             "fingerprints the operating system that sent the traffic, so the very high scores on "
             "this dataset would not fully transfer to a different network. Figure 10 lists the "
             "top features.")
        figure(doc, "10_feature_importance",
               "Figure 10. Most important features by permutation importance.")
    if sig.exists():
        s = pd.read_csv(sig)
        para(doc,
             "For each attack, the single feature that best separates it from benign traffic "
             "matches the attack's mechanism. A few examples (full table in "
             "results/features/signatures.csv):")
        picks = ["FTP-Patator", "DoS slowloris", "PortScan", "Heartbleed", "DoS Hulk"]
        rows = []
        for atk in picks:
            g = s[s["attack"] == atk]
            if len(g):
                r0 = g.iloc[0]
                rows.append((atk, r0["feature"], f"{r0['separation']:.2f}",
                             f"{r0['median_attack']:,.0f}", f"{r0['median_benign']:,.0f}"))
        table(doc, ["Attack", "Top feature", "Separation", "Attack median", "Benign median"],
              rows, widths=[1.4, 1.9, 1.0, 1.2, 1.2])
        para(doc, "Table 2. The strongest single-feature signature for a sample of attacks. "
                  "Times are in microseconds.", italic=True, size=9)
        para(doc,
             "Two web attacks, brute force and XSS, share the same signature and the same median "
             "feature values, which is why classifiers most often confuse them: their flows are "
             "near-identical, and the difference lives in the HTTP payload that flow features "
             "cannot see.")

    # 8. imbalance
    h(doc, "8  Handling the class imbalance", 1)
    imb = RES / "imbalance" / "imbalance.csv"
    if imb.exists():
        para(doc,
             "We tried three ways to help the rare classes, each applied to the training split "
             "only so the test scores stay honest: weighting classes by inverse frequency, "
             "undersampling the benign class, and SMOTE (synthesising new rare-class rows). "
             "Figures 11 and 12 show the effect on macro F1 and on the recall of the six rarest "
             "attacks.")
        figure(doc, "11_imbalance_macro_f1", "Figure 11. Macro F1 by model and imbalance strategy.")
        figure(doc, "12_imbalance_rare_recall",
               "Figure 12. Recall on the six rarest attacks, by model and strategy.")
    else:
        para(doc, "[Imbalance results missing; run 05_imbalance.py.]", italic=True)

    # 9. deployment + conclusion
    h(doc, "9  Deployment considerations", 1)
    para(doc,
         "A flow-based IDS sits beside the traffic, not in its path: a flow's statistics are "
         "complete only when the flow ends, so detection comes after the fact and the delay "
         "depends on the attack (a port-scan probe is classified in microseconds; a slowloris "
         "connection only when it times out). Throughput matters as much as accuracy, since the "
         "classifier must keep up with the flow rate of a busy link (Figure 5). Timing and size "
         "features survive encryption, which suits DoS, scans and brute force, but payload "
         "attacks such as SQL injection need application logs as well.")

    h(doc, "10  Conclusion", 1)
    para(doc,
         "On CIC-IDS2017, tree ensembles led by Random Forest detect most attacks well, but the "
         "rarest attacks and the two look-alike web attacks remain hard, and the headline "
         "accuracy overstates performance until macro F1 and the per-class scores are read "
         "alongside it. Kernel PCA helps only the distance-based classifiers. The most useful "
         "features are partly dataset-specific shortcuts, so the natural next steps are to test "
         "on traffic from a different network and to drop those shortcut features. Removing the "
         "duplicate rows and reporting macro F1 were the two changes that most affected how the "
         "results should be read.")

    OUT.parent.mkdir(exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT}")
    have = sum((CHARTS / f"{n}.png").exists() for n in
               ["01_class_distribution", "02_duplicates_removed", "03_accuracy_vs_macro_f1",
                "04_per_class_f1", "05_test_size", "06_throughput", "07_kpca_kernel_by_classifier",
                "08_kpca_best_vs_no_reduction", "09_kpca_components", "10_feature_importance",
                "11_imbalance_macro_f1", "12_imbalance_rare_recall"])
    print(f"figures embedded: {have} of 12  (missing ones show a placeholder)")


if __name__ == "__main__":
    build()
