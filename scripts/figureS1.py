"""Supplementary Figure S1: study selection and data-extraction verification."""
from matplotlib.patches import FancyBboxPatch
import data as D
from plot import Page, PAGE_W, SMALL, TINY, HEAD

GREEN, ROSE, AMBER, INK = "#E3F1EA", "#F6E3E3", "#FBF0D9", "#1A1A1A"
EG, ER, EA = "#2E7D5B", "#B5585B", "#B7862A"


def box(ax, x, y, w, h, lines, fc, ec, bold_first=False, fs=SMALL, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=0.6))
    tx = x + w / 2 if ha == "center" else x + 1.6
    n = len(lines)
    for i, s in enumerate(lines):
        ax.text(tx, y + h / 2 + (i - (n - 1) / 2) * 2.5, s, fontsize=fs, ha=ha, va="center", color=INK,
                fontweight="bold" if (bold_first and i == 0) else "normal")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#555555", mutation_scale=6, shrinkA=0, shrinkB=0))


def stage(ax, x, y0, y1, label):
    ax.add_patch(FancyBboxPatch((x, y0), 4.5, y1 - y0, boxstyle="round,pad=0,rounding_size=1", fc="#EDEDED",
                                ec="none"))
    ax.text(x + 2.25, (y0 + y1) / 2, label, rotation=90, fontsize=SMALL, fontweight="bold", ha="center", va="center")


# selection and audit counts are from the screening and verification records
def main():
    pg = Page(h=96)
    ax = pg.canvas(0, 0, PAGE_W, 96)
    pg.letter(0, 0, "a")
    pg.text(4.2, 0.5, "Study selection (PRISMA 2020)", fontsize=HEAD, fontweight="bold")
    X, W, XE, WE = 7.5, 40.0, 52.0, 47.0
    stage(ax, 0, 7, 21, "Identification")
    stage(ax, 0, 23, 67, "Screening")
    stage(ax, 0, 69, 82, "Included")
    box(ax, X, 7, W, 14, ["Records from the curated MMC", "human-microbiome corpus*", "n = 13,092"], GREEN, EG)
    box(ax, XE, 9, WE, 10, ["Duplicates removed: n = 0", "(source already de-duplicated)"], ROSE, ER)
    box(ax, X, 25, W, 10, ["Records screened (title, abstract)", "n = 13,092"], GREEN, EG)
    box(ax, XE, 23, WE, 14, ["Excluded at title/abstract: n = 12,910", "no confirmed lung cancer; no lung-cancer",
                              "finding; no prokaryotic sequencing data;", "review, protocol or commentary"], ROSE, ER,
        bold_first=True)
    box(ax, X, 43, W, 10, ["Full texts assessed for eligibility", "n = 182"], GREEN, EG)
    box(ax, XE, 40, WE, 16, ["Excluded at full text: n = 35", "out of scope, n = 33 (incl. 1 non-English)",
                              "full text not available, n = 1", "irreconcilable internal data conflict, n = 1"],
        ROSE, ER, bold_first=True)
    box(ax, X, 59, W, 8, ["Studies eligible: n = 147"], GREEN, EG)
    box(ax, XE, 58.5, WE, 9, ["Excluded after extraction: n = 3", "sampled before any cancer diagnosis"], AMBER, EA,
        bold_first=True)
    box(ax, X, 72, W, 10, ["Studies included in the review", f"n = {len(D.papers())}"], GREEN, EG, bold_first=True)
    cx = X + W / 2
    for y0, y1 in [(21, 25), (35, 43), (53, 59), (67, 72)]:
        arrow(ax, cx, y0, cx, y1)
    for y in [14, 30, 48, 63]:
        arrow(ax, X + W, y, XE, y)
    ax.text(0, 86, "* Assembled from a systematic PubMed search (2012–2025) and screened in Rayyan by the",
            fontsize=TINY, va="top")
    ax.text(0, 89, "Microbiome Metadata Crisis consortium (Degregori et al.); search strings in Supplementary Methods S1.",
            fontsize=TINY, va="top")
    X2 = 106
    pg.letter(X2 - 4, 0, "b")
    pg.text(X2 + 0.2, 0.5, "Data-extraction verification", fontsize=HEAD, fontweight="bold")
    stage(ax, X2 - 4, 7, 21, "Extraction")
    stage(ax, X2 - 4, 23, 67, "Verification")
    stage(ax, X2 - 4, 69, 82, "Verified")
    WB = PAGE_W - X2 - 2
    box(ax, X2 + 2, 7, WB, 14, [f"{len(D.papers())} studies extracted into a 63-field", "codebook by Claude Opus 4.8, 5 and 5.5",
                                "under prespecified rules"], GREEN, EG)
    box(ax, X2 + 2, 25, WB, 18, ["Pass 1: random 48-paper sample,", "3 reviewers, cell by cell vs full text",
                                 "65 wrong cells confirmed; 13 of 48 papers", "fully correct (≈ 95–96% accurate)"],
        AMBER, EA)
    box(ax, X2 + 2, 47, WB, 20, ["Pass 2: fresh 48-paper sample (14 shared)", "48 flags: 37 confirmed and fixed,",
                                 "5 already correct, 6 rejected;", "23 papers edited (≈ 97–98% accurate;",
                                 "1.9% per-field error)"], AMBER, EA)
    box(ax, X2 + 2, 72, WB, 10, ["Verified dataset: " + f"{len(D.findings()):,} taxon-level", f"findings and {len(D.diversity()):,} diversity results"],
        GREEN, EG, bold_first=False)
    bx = X2 + 2 + WB / 2
    for y0, y1 in [(21, 25), (43, 47), (67, 72)]:
        arrow(ax, bx, y0, bx, y1)
    pg.save("FigureS1")


if __name__ == "__main__":
    main()
