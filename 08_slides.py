"""
Step 8: build the presentation slides.

Same idea as the report: text is fixed, numbers and charts come from the
results files, and a chart that does not exist yet leaves a labelled gap.
16:9 slides.

    python 08_slides.py            -> results/ACN_IDS_Slides.pptx
"""

from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).parent
RES = HERE / "results"
CHARTS = RES / "charts"
OUT = RES / "ACN_IDS_Slides.pptx"

INK = RGBColor(0x0B, 0x0B, 0x0B)
MUTED = RGBColor(0x52, 0x51, 0x4E)
ACCENT = RGBColor(0x1C, 0x5C, 0xAB)
SURFACE = RGBColor(0xFC, 0xFC, 0xFB)
PAGE = RGBColor(0xFF, 0xFF, 0xFF)

MODEL_NAMES = {
    "naive_bayes": "Naive Bayes", "lda": "LDA", "qda": "QDA", "sgd": "SGD",
    "decision_tree": "Decision Tree", "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "random_forest": "Random Forest", "hist_gb": "Hist. GB", "catboost": "CatBoost",
    "adaboost": "AdaBoost", "mlp": "MLP", "linear_svm": "Linear SVM",
    "logistic": "Logistic Regression", "knn": "KNN", "svm": "SVM (RBF)",
}

EMU = 914400
W, H = int(13.333 * EMU), int(7.5 * EMU)


def _bg(slide):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAGE


def _text(slide, left, top, width, height, text, size, color=INK, bold=False,
          align=PP_ALIGN.LEFT, italic=False, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        r.font.name = "Calibri"
    return box


def _bar(slide):
    """accent side bar for a content slide title area"""
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.18), H)
    s.fill.solid(); s.fill.fore_color.rgb = ACCENT
    s.line.fill.background()
    return s


def title_slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(s)
    from pptx.enum.shapes import MSO_SHAPE
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.4), W, Inches(2.7))
    band.fill.solid(); band.fill.fore_color.rgb = ACCENT; band.line.fill.background()
    _text(s, 0.9, 2.7, 11.5, 1.4, "Network Intrusion Detection\non CIC-IDS2017",
          34, PAGE, bold=True)
    _text(s, 0.9, 4.35, 11.5, 0.6,
          "Comparing classifiers, Kernel PCA, and the rare-attack problem",
          16, RGBColor(0xDD, 0xE8, 0xF7))
    _text(s, 0.9, 6.5, 11.5, 0.5, "Advanced Computer Networks course project", 13, MUTED)
    return s


def content(prs, title):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(s); _bar(s)
    _text(s, 0.55, 0.35, 12.2, 0.9, title, 26, ACCENT, bold=True)
    return s


def bullets(slide, items, left=0.7, top=1.5, width=7.0, size=17, gap=True):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(5.2))
    tf = box.text_frame; tf.word_wrap = True
    first = True
    for it in items:
        lead, rest = it if isinstance(it, tuple) else (None, it)
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(10 if gap else 4)
        r = p.add_run(); r.text = "•  "
        r.font.size = Pt(size); r.font.color.rgb = ACCENT; r.font.bold = True
        if lead:
            rl = p.add_run(); rl.text = lead
            rl.font.size = Pt(size); rl.font.bold = True; rl.font.color.rgb = INK
            rl.font.name = "Calibri"
        rr = p.add_run(); rr.text = rest
        rr.font.size = Pt(size); rr.font.color.rgb = INK; rr.font.name = "Calibri"
    return box


def picture(slide, name, left=7.7, top=1.4, width=5.2):
    path = CHARTS / f"{name}.png"
    if path.exists():
        slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width))
        return True
    _text(slide, left, top + 1.5, width, 0.8, f"[{name}\npending]", 12, MUTED,
          align=PP_ALIGN.CENTER, italic=True)
    return False


def big_picture(slide, name, left=3.3, top=1.35, width=9.4):
    path = CHARTS / f"{name}.png"
    if path.exists():
        slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width))
    else:
        _text(slide, left, top + 2, width, 0.8, f"[{name} pending]", 14, MUTED,
              align=PP_ALIGN.CENTER, italic=True)


def grid_top():
    from common import pick_grid_protocol

    d, _ = pick_grid_protocol(RES / "results.csv")
    if d is None:
        return None
    d = d[d["split"] == "60/40"]
    return d.sort_values("macro_f1", ascending=False) if len(d) else None


def build():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H

    title_slide(prs)

    # 1 problem
    s = content(prs, "The problem")
    bullets(s, [
        ("Intrusion detection. ", "classify each network flow as benign or one of 14 attacks."),
        ("Flow-based. ", "uses summary statistics (sizes, timings, TCP flags), not packet "
         "contents, so it works on encrypted traffic. Same idea as NetFlow / IPFIX."),
        ("The catch: imbalance. ", "benign traffic outnumbers the rarest attack 195,291 to 1."),
        ("Three aims. ", "compare classifiers fairly, test Kernel PCA, and explain the "
         "results in networking terms."),
    ], width=11.5)

    # 2 dataset
    s = content(prs, "The dataset, after cleaning")
    bullets(s, [
        ("2,574,059 flows, ", "69 features, 15 classes (CIC-IDS2017, five days of traffic)."),
        ("Removed 9% duplicate rows ", "that would otherwise leak across the train/test split."),
        ("Dropped 8 dead columns ", "and one duplicated column."),
        ("Rarest attacks: ", "Heartbleed 11 flows, SQL injection 21, Infiltration 36."),
    ], width=6.6)
    picture(s, "01_class_distribution", left=7.4, top=1.35, width=5.5)

    # 3 method / macro F1
    s = content(prs, "Why accuracy is not enough")
    bullets(s, [
        ("One shared pipeline. ", "impute, scale (for scale-sensitive models), classify; "
         "fitted on the training split only."),
        ("Accuracy hides rare attacks. ", "a model can ignore every rare attack and still "
         "score above 0.9, because benign traffic is 83% of the data."),
        ("We report macro F1. ", "it weights every class equally, so a missed rare attack "
         "shows up. Read with the per-class scores."),
    ], width=11.5)

    # 4 classifier results
    s = content(prs, "Classifier comparison")
    d = grid_top()
    if d is not None:
        best = d.iloc[0]
        bullets(s, [
            (f"{MODEL_NAMES.get(best['model'], best['model'])} wins. ",
             f"macro F1 {best['macro_f1']:.3f}, accuracy {best['accuracy']:.3f}."),
            ("Tree ensembles lead; ", "linear models trail."),
            ("Accuracy above 0.9 for most, ", "but macro F1 spreads far wider."),
        ], left=0.7, top=1.4, width=5.4)
        picture(s, "03_accuracy_vs_macro_f1", left=6.3, top=1.3, width=6.6)
    else:
        big_picture(s, "03_accuracy_vs_macro_f1")

    # 5 kernel PCA
    s = content(prs, "Kernel PCA: helps only KNN and SVM")
    bullets(s, [
        ("5 kernels x 3 sizes x 3 counts, ", "7 classifiers = 336 runs on a 10k sample."),
        ("Reduction helps only KNN (+0.025) and SVM (+0.073), ", "both with the sigmoid kernel."),
        ("Every other classifier ", "does worse than with all 69 features."),
        ("Poly is the worst kernel; ", "more components always help."),
    ], left=0.7, top=1.4, width=5.6)
    picture(s, "08_kpca_best_vs_no_reduction", left=6.4, top=1.5, width=6.5)

    # 6 features
    s = content(prs, "What the features say about each attack")
    bullets(s, [
        ("PortScan: ", "tiny flows, zero payload, thousands of probes per second."),
        ("Slow DoS: ", "packets seconds apart, not milliseconds."),
        ("Heartbleed: ", "huge server reply packets separate all 11 flows perfectly."),
        ("Web brute force vs XSS: ", "identical flow signature, so they get confused."),
        ("Caution: ", "top features (port, TCP window) are partly dataset shortcuts."),
    ], left=0.7, top=1.4, width=6.0)
    picture(s, "10_feature_importance", left=6.7, top=1.35, width=6.2)

    # 7 imbalance
    s = content(prs, "Fixing the class imbalance")
    bullets(s, [
        ("Applied to training only, ", "so the test scores stay honest."),
        ("Class weights, undersampling, SMOTE.", ""),
        ("Recovers rare-attack recall ", "that the plain models miss entirely."),
    ], left=0.7, top=1.5, width=5.2)
    picture(s, "12_imbalance_rare_recall", left=6.1, top=1.3, width=6.8)

    # 8 deployment + conclusion
    s = content(prs, "Deployment and takeaways")
    bullets(s, [
        ("Runs beside the traffic, ", "not inline: a verdict comes only when the flow ends."),
        ("Speed matters: ", "the classifier must keep up with the link's flow rate."),
        ("Encryption is fine; ", "payload attacks (SQL injection, XSS) need application logs too."),
        ("Takeaway: ", "tree ensembles win; report macro F1, not accuracy; Kernel PCA is not "
         "a free win; the rare and look-alike attacks stay hard."),
    ], width=11.8)

    # thanks
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(s)
    from pptx.enum.shapes import MSO_SHAPE
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(3.0), W, Inches(1.6))
    band.fill.solid(); band.fill.fore_color.rgb = ACCENT; band.line.fill.background()
    _text(s, 0.9, 3.25, 11.5, 1.0, "Thank you", 32, PAGE, bold=True)
    _text(s, 0.9, 5.0, 11.5, 0.5,
          "github.com/JONATHANJOHN95001/acn-network-intrusion-detection", 14, MUTED)

    OUT.parent.mkdir(exist_ok=True)
    prs.save(OUT)
    print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    build()
