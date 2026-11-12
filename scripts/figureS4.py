"""Supplementary Figure S4: consistency of diversity metrics by body site and comparison."""
import pandas as pd
from matplotlib.patches import Rectangle
import data as D
from plot import (Page, PAGE_W, TXT, SMALL, TINY, BLACK, VERM, BLUE, PURPLE, ORANGE, SKY, NA_GREY, GREEN, YELLOW,
                  text_on, swatch)

MIN_P = 3
NO_POOL = {"other", "body site comparison"}
CTX_ORDER = ["cancer vs healthy", "cancer vs benign", "tumor vs paired normal", "stage comparison", "metastasis",
             "subtype comparison", "therapy response", "pre vs post treatment", "survival", "recurrence", "smoking",
             "comorbidity", "demographic"]
ALPHA_METRICS = ["Shannon", "Simpson", "Inverse Simpson", "Evenness", "Observed richness", "Chao1", "ACE",
                 "Faith's PD", "Fisher's alpha"]
BETA_METRICS = ["Bray-Curtis", "Weighted UniFrac", "Unweighted UniFrac", "UniFrac, weighting not stated", "Jaccard",
                "Jensen-Shannon", "Aitchison", "Euclidean", "ThetaYC"]
ALPHA = [("higher", "Higher in the unfavourable group", VERM), ("lower", "Lower in the unfavourable group", BLUE),
         ("inconsistent", "Mixed direction", PURPLE), ("no side", "Different, no favourable side", ORANGE),
         ("undirected", "Different, direction not stated", SKY), ("ns", "No significant difference", NA_GREY)]
BETA = [("sig", "Significant separation", GREEN), ("ns", "No significant separation", NA_GREY),
        ("visual", "Plot only, not tested", YELLOW)]
CTX_HEAD = {"cancer vs healthy": "Cancer vs\nhealthy", "cancer vs benign": "Cancer vs\nbenign",
            "tumor vs paired normal": "Tumour vs\npaired\nnormal", "therapy response": "Non-resp.\nvs resp.",
            "pre vs post treatment": "Pre vs\npost\ntreatment", "survival": "Shorter vs\nlonger\nsurvival",
            "metastasis": "Metast.\nvs non-\nmetast.", "subtype comparison": "Subtype",
            "comorbidity": "Comorbid.\nvs none", "stage comparison": "Stage",
            "recurrence": "Recurrence", "smoking": "Smoking", "demographic": "Demographic"}
SITE_HEAD = {"gut": "Gut", "lower-respiratory-tract": "Lower respiratory tract", "lung-tissue": "Lung tissue",
             "oral": "Oral"}


def compute():
    v = D.diversity_votes()
    v = v[v.outcome.notna() & v.metric.isin(ALPHA_METRICS + BETA_METRICS) & ~v.comparison.isin(NO_POOL)]
    pm = v.groupby(["site", "comparison", "metric_type", "metric"]).key.nunique()
    keep = pm[pm >= MIN_P].reset_index()[["site", "comparison"]].drop_duplicates()
    keep["so"], keep["co"] = keep.site.map(D.DIV_SITES.index), keep.comparison.map(CTX_ORDER.index)
    cols = list(keep.sort_values(["so", "co"])[["site", "comparison"]].itertuples(index=False, name=None))
    allv = v.merge(keep[["site", "comparison"]])
    blocks = [(kind, head, [m for m in ms if ((allv.metric_type == kind) & (allv.metric == m)).any()], cats)
              for kind, head, ms, cats in [("alpha", "Alpha diversity", ALPHA_METRICS, ALPHA),
                                           ("beta", "Beta diversity", BETA_METRICS, BETA)]]
    return cols, allv, blocks


def main():
    cols, allv, blocks = compute()
    npap = allv.groupby(["site", "comparison"]).key.nunique()
    W = PAGE_W
    LW_, SG = 25.5, 1.4
    CW = (W - LW_ - SG * (len({c[0] for c in cols}) - 1)) / len(cols)
    BW = CW - 1.0
    RH, HB = 2.9, 3.4
    xs, x, prev = [], LW_, None
    for s, _ in cols:
        if prev is not None and s != prev:
            x += SG
        xs.append(x); x += CW; prev = s
    SITE_Y, COMP_Y, PAP_Y = 2.5, 7.4, 11.6
    Y0 = 13.6
    body_h = sum(HB + len(ms) * RH for _, _, ms, _ in blocks)
    LEG = Y0 + body_h + 3.5
    H = LEG + 10.5
    pg = Page(h=H)
    ax = pg.canvas(0, 0, W, H)
    for s in dict.fromkeys(c[0] for c in cols):
        idx = [j for j, c in enumerate(cols) if c[0] == s]
        xa, xb = xs[idx[0]] + 0.3, xs[idx[-1]] + CW - 0.3
        ax.text((xa + xb) / 2, SITE_Y, SITE_HEAD[s], fontsize=TXT, fontweight="bold", ha="center", va="bottom")
        ax.plot([xa, xb], [SITE_Y + 0.4] * 2, color=BLACK, lw=0.6)
    ax.text(LW_ - 1.0, PAP_Y, "Papers", fontsize=SMALL, ha="right", va="center")
    for j, (s, c) in enumerate(cols):
        ax.text(xs[j] + CW / 2, COMP_Y, CTX_HEAD[c], fontsize=TINY, ha="center", va="center", linespacing=1.0)
        ax.text(xs[j] + CW / 2, PAP_Y, str(int(npap[(s, c)])), fontsize=SMALL, fontweight="bold", ha="center",
                va="center")
    present = set()
    y = Y0
    for kind, head, metrics, cats in blocks:
        ax.plot([0, W], [y, y], color=BLACK, lw=0.7)
        ax.text(0, y + HB / 2, head, fontsize=TXT, fontweight="bold", va="center")
        ax.plot([0, W], [y + HB, y + HB], color=BLACK, lw=0.35)
        y += HB
        for m in metrics:
            ax.text(1.5, y + RH / 2, m, fontsize=SMALL, va="center")
            for j, (s, c) in enumerate(cols):
                ax.add_patch(Rectangle((xs[j], y), CW, RH, fc="white", ec="#C8C8C8", lw=0.25))
                g = allv[(allv.site == s) & (allv.comparison == c) & (allv.metric_type == kind) & (allv.metric == m)]
                if g.empty:
                    continue
                k = g.outcome.value_counts()
                n = int(k.sum())
                cx = xs[j] + 0.5
                for o, _, col in cats:
                    v = int(k.get(o, 0))
                    if not v:
                        continue
                    present.add((kind, o))
                    ww = BW * v / n
                    ax.add_patch(Rectangle((cx, y + 0.35), ww, RH - 0.7, fc=col, ec="white", lw=0.25))
                    if ww >= 1.5:
                        ax.text(cx + ww / 2, y + RH / 2, str(v), fontsize=3.8, fontweight="bold", ha="center",
                                va="center", color=text_on(col))
                    cx += ww
                if n >= MIN_P:
                    ax.add_patch(Rectangle((xs[j], y), CW, RH, fill=False, ec=BLACK, lw=0.7, zorder=5))
            y += RH
    ax.plot([0, W], [y, y], color=BLACK, lw=0.5)
    ly = LEG
    for lab, kind, cats in [("Alpha", "alpha", ALPHA), ("Beta", "beta", BETA)]:
        ax.text(0, ly, lab, fontsize=SMALL, fontweight="bold", va="center")
        xk = 8.0
        for o, name, col in cats:
            if (kind, o) in present:
                xk = swatch(ax, xk, ly, col, name, fs=SMALL, w=2.2, h=1.6)
        ly += 3.0
    ax.add_patch(Rectangle((8.0, ly - 0.8), 2.2, 1.6, fill=False, ec=BLACK, lw=0.7))
    ax.text(11.1, ly, f"at least {MIN_P} papers in the cell. Bars are shares of papers; numbers are papers "
            "(printed where the segment is wide enough).", fontsize=SMALL, va="center")
    pg.save("FigureS4")
    print(f"  Figure S4: {len(cols)} body-site x comparison pairs, {allv.key.nunique()} studies")


if __name__ == "__main__":
    main()
