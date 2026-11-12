"""Supplementary Figure S3: corpus characteristics."""
import numpy as np
import data as D
from plot import (Page, SMALL, HEAD, BLACK, BLUE, SKY, VERM, NA_GREY, LIGHTGRID, SITE_LABEL, SITE_COLOR, hbar)

HEAD_H = 4.2
INTERV_LABEL = {"none-observational": "Observational (no intervention)", "immunotherapy": "Immunotherapy",
                "other-combination": "Other / combination", "surgery": "Surgery", "radiotherapy": "Radiotherapy",
                "chemotherapy": "Chemotherapy"}
TIMING_LABEL = {"single-timepoint": "Single timepoint", "pre-post-paired": "Paired pre / post",
                "single-timepoint, pre-post-paired": "Single timepoint + paired subset",
                "longitudinal-multi-timepoint": "Longitudinal, multiple timepoints"}
MODEL_ORDER = ["covariate-adjusted", "balanced-population", "unadjusted", "not-applicable"]
MODEL_LABEL = {"covariate-adjusted": "Covariate-adjusted", "balanced-population": "Balanced population",
               "unadjusted": "Unadjusted", "not-applicable": "Not applicable"}
MODEL_COLOR = {"covariate-adjusted": BLUE, "balanced-population": SKY, "unadjusted": VERM, "not-applicable": NA_GREY}


def counts(s, labels, colors=None):
    c = s.value_counts()
    return [(labels[k], int(c[k]), (colors or {}).get(k)) for k in labels if k in c.index]


def panel_head(pg, x, y, letter, title):
    if letter:
        pg.letter(x, y, letter)
    pg.text(x + (4.2 if letter else 0), y + 0.5, title, fontsize=HEAD, fontweight="bold")


def cohort_size(pg, x, y, w, h, m):
    p = m.explode("sites").rename(columns={"sites": "site"})
    sites = [s for s in D.SITE_ORDER if s in set(p.site)]
    med_all = float(m.n_total.median())
    ax = pg.axes(x + 11, y + HEAD_H + 1, w - 12, h - HEAD_H - 11)
    rng = np.random.default_rng(7)
    ax.axhline(med_all, color=BLACK, ls=(0, (4, 3)), lw=0.6, zorder=2)
    for xi, s in enumerate(sites):
        v = p.loc[p.site == s, "n_total"].to_numpy()
        ax.scatter(xi + rng.uniform(-0.22, 0.22, len(v)), v, s=7, color=SITE_COLOR[s], alpha=0.85,
                   edgecolor=BLACK, linewidth=0.25, zorder=3)
        md = float(np.median(v))
        ax.hlines(md, xi - 0.32, xi + 0.32, color=BLACK, lw=1.2, zorder=4)
        ax.text(xi + 0.36, md, f"{md:,g}", ha="left", va="center", fontsize=SMALL, fontweight="bold", zorder=5,
                bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none"))
    ax.set_yscale("log"); ax.set_ylim(5, 6000); ax.minorticks_off()
    ax.set_yticks([10, 30, 100, 300, 1000, 3000]); ax.set_yticklabels(["10", "30", "100", "300", "1,000", "3,000"])
    ax.set_xlim(-0.6, len(sites) - 0.3)
    ax.set_xticks(range(len(sites)))
    ax.set_xticklabels([f"{SITE_LABEL[s]}\n(n = {int((p.site == s).sum())})" for s in sites])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Participants per paper (log scale)")
    ax.yaxis.grid(True, color=LIGHTGRID, lw=0.4); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(len(sites) - 0.35, med_all * 1.08, f"all papers, median {med_all:g}", ha="right", va="bottom",
            fontsize=SMALL)


def main():
    m = D.cohort()
    iv = counts(m.intervention_category, INTERV_LABEL)
    tm = counts(m.sample_timing, TIMING_LABEL)
    md = counts(m.model, dict(zip(MODEL_ORDER, [MODEL_LABEL[k] for k in MODEL_ORDER])), MODEL_COLOR)
    mm_per = 24.0 / max(n for _, n, _ in iv + tm + md)
    pg = Page(h=92)
    panel_head(pg, 0, 0, "a", "Intervention")
    hbar(pg.canvas(0, HEAD_H, 58, 22), 0, 0.5, iv, mm_per, 33.0)
    panel_head(pg, 59, 0, None, "Sample timing")
    hbar(pg.canvas(59, HEAD_H, 64, 22), 0, 0.5, tm, mm_per, 35.5)
    panel_head(pg, 125, 0, "b", "How confounders were handled")
    hbar(pg.canvas(125, HEAD_H, 58, 15), 0, 0.5, md, mm_per, 27.0, pct_of=len(m))
    panel_head(pg, 0, 28, "c", "Cohort size by body site")
    cohort_size(pg, 0, 28, 183, 64, m)
    pg.save("FigureS3")


if __name__ == "__main__":
    main()
