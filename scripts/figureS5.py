"""Supplementary Figure S5: microbiome classifiers."""
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import data as D
from plot import (Page, PAGE_W, TXT, SMALL, TINY, HEAD, BLACK, GREEN, BLUE, SKY, VERM, SITE_LABEL, SITE_COLOR,
                  hbar, swatch, text_on, text_w, tree_rows, leaf_info, place_rows, draw_tree_axis, scale_bar,
                  label_width)

SITE_ORDER = ["gut", "lower-respiratory-tract", "lung-tissue", "oral", "blood-plasma-serum", "pleural-fluid", "other"]
CTX_ORDER = ["cancer vs healthy", "cancer vs benign", "tumor vs paired normal", "subtype comparison",
             "therapy response", "survival", "recurrence", "metastasis", "comorbidity", "other"]
CTX_LABEL = {"cancer vs healthy": "Cancer vs healthy", "cancer vs benign": "Cancer vs benign",
             "tumor vs paired normal": "Tumor vs paired normal", "subtype comparison": "Subtype comparison",
             "therapy response": "Therapy response", "survival": "Survival", "recurrence": "Recurrence",
             "metastasis": "Metastasis", "comorbidity": "Comorbidity", "other": "Other"}
SITE_SHORT = {"gut": "Gut", "lower-respiratory-tract": "Lower resp. tract", "lung-tissue": "Lung", "oral": "Oral",
              "blood-plasma-serum": "Blood", "pleural-fluid": "Pleural"}
RANKS = ["phylum", "class", "order", "family", "genus", "species", "clade"]
VAL = [("External cohort", GREEN), ("Internal resampling or hold-out", BLUE),
       ("Internal only, method not stated", SKY), ("No validation", VERM)]
MIN_RECUR = 2
HEAD_H = 4.2
GOLD = "#E69F00"
HGRID = "#9A9A9A"
PAL = ['#08254f', '#123e73', '#1c5999', '#2672bd', '#448ed6', '#74abe1', '#a6c9ec', '#d6e6f7', '#c8c8c8',
       '#fddad3', '#f7ada4', '#f17e73', '#eb5043', '#d53127', '#b1211e', '#8b1015', '#67000d']


def classifier_votes():
    v = D.votes()
    c = v[v.finding_type == "classifier"].copy()
    f = D.findings()
    c = D.valence_of(c, f[f.finding_type == "classifier"])
    assert c.clinical_valence.isin(["unfavorable", "not-applicable"]).all()
    c, _ = D.attach_tree(c)
    c["gg2_name"] = c.tree_leaf.astype(str)
    c["row_rank"] = np.where(c.is_phylum_row, "phylum", c.rank_disp)
    c["node"] = c.tree_node_id.astype(int)
    c["ctx"] = c.comparison_context.str.split(r"\s*,\s*")
    c = c.explode("ctx")
    c["site"] = c.body_site_resolved.str.split(r"\s*,\s*")
    c = c.explode("site")
    c["dirv"] = np.where(c.clinical_valence.eq("unfavorable") & c.direction_vote.isin(["up", "down"]),
                         c.direction_vote, "none")
    lv = D.papers().set_index("key").validation_level
    assert lv[sorted(set(c.key))].isin([k for k, _ in VAL]).all(), "classifier study without a validation level"
    c["validation_level"] = c.key.map(lv)
    return c


def matrix_data(c):
    direction = lambda s: (lambda d: "none" if not d else ("mixed" if len(d) > 1 else d.pop()))(set(s) - {"none"})
    cell = (c.groupby(["key", "row_id", "site", "ctx"])
            .agg(vote=("dirv", direction), validation_level=("validation_level", "first")).reset_index())
    meta = c.groupby("row_id").agg(row_rank=("row_rank", "first"), phylum=("phylum", "first"),
                                   gg2_name=("gg2_name", "first"), node=("node", "first"),
                                   leaf_label=("tree_leaf_label", "first"), papers=("key", "nunique"))
    lab = D.figure_labels([n for r, n in zip(meta.index, meta.leaf_label) if not r.startswith("P|")])
    meta["label"] = [r[2:] if r.startswith("P|") else lab[n] for r, n in zip(meta.index, meta.leaf_label)]
    strata = c.groupby(["site", "ctx"]).key.nunique().rename("papers").reset_index()
    cols = strata[strata.papers >= 2].copy()
    cols["so"], cols["co"] = cols.site.map(SITE_ORDER.index), cols.ctx.map(CTX_ORDER.index)
    cols = cols.sort_values(["so", "co"]).reset_index(drop=True)
    shown = cell.merge(cols[["site", "ctx"]], on=["site", "ctx"])
    recur = meta[meta.papers >= MIN_RECUR]
    sel = recur[recur.index.isin(shown.row_id)]
    return cell, cols, shown[shown.row_id.isin(sel.index)], sel


def landscape(pg, y, c):
    site = c.groupby("site").key.nunique().sort_values(ascending=False)
    comp = c.groupby("ctx").key.nunique().sort_values(ascending=False, kind="stable")
    rank = c.groupby("taxonomic_rank").key.nunique().reindex(RANKS)
    mm_per = 22.0 / max(site.max(), comp.max(), rank.max())
    pg.letter(0, y, "a")
    for x, name in [(4.2, "Body site"), (62, "Comparison"), (124, "Rank of the reported feature")]:
        pg.text(x, y + 0.5, name, fontsize=TXT, fontweight="bold")
    hbar(pg.canvas(0, y + HEAD_H, 60, 34), 0, 0.3, [(SITE_LABEL[s], int(n), SITE_COLOR[s]) for s, n in site.items()],
         mm_per, 27.0, row=3.0, bar=1.9)
    hbar(pg.canvas(62, y + HEAD_H, 60, 34), 0, 0.3, [(CTX_LABEL[k], int(n), None) for k, n in comp.items()],
         mm_per, 26.0, row=3.0, bar=1.9)
    hbar(pg.canvas(124, y + HEAD_H, 59, 34), 0, 0.3, [(k.capitalize(), int(n), None) for k, n in rank.items()],
         mm_per, 12.0, row=3.0, bar=1.9)


def validation(pg, x, y, w, c):
    lv = c.drop_duplicates("key").set_index("key").validation_level
    rows = [(f"All classifier papers (n = {len(lv)})", [int((lv == k).sum()) for k, _ in VAL])]
    for s in SITE_ORDER:
        keys = sorted(set(c[c.site == s].key))
        if len(keys) >= 2:
            rows.append((f"{SITE_SHORT[s]} (n = {len(keys)})", [int((lv[keys] == k).sum()) for k, _ in VAL]))
    pg.letter(x, y, "c")
    pg.text(x + 4.2, y + 0.5, "How the classifier models were validated", fontsize=HEAD, fontweight="bold")
    ax = pg.canvas(x, y + HEAD_H + 1, w, 60)
    lab_w, rh, bh = 25.0, 3.6, 2.6
    bw = w - lab_w - 3.5
    for i, (lab, vals) in enumerate(rows):
        yc = rh * (i + 0.5)
        ax.text(lab_w - 1.0, yc, lab, fontsize=SMALL, ha="right", va="center",
                fontweight="bold" if i == 0 else "normal")
        tot, cx = sum(vals), lab_w
        for v, (_, col) in zip(vals, VAL):
            if v:
                ww = bw * v / tot
                ax.add_patch(Rectangle((cx, yc - bh / 2), ww, bh, fc=col, ec="white", lw=0.3))
                ax.text(cx + ww / 2, yc, str(v), fontsize=TINY, fontweight="bold", ha="center", va="center",
                        color=text_on(col))
                cx += ww
    yb = rh * len(rows) + 0.4
    ax.plot([lab_w, lab_w + bw], [yb, yb], color=BLACK, lw=0.4)
    for f in (0, 0.25, 0.5, 0.75, 1.0):
        ax.plot([lab_w + bw * f] * 2, [yb, yb + 0.7], color=BLACK, lw=0.4)
        ax.text(lab_w + bw * f, yb + 1.0, f"{f:.0%}", fontsize=TINY, ha="center", va="top")
    ax.text(lab_w + bw / 2, yb + 3.6, "Share of classifier papers", fontsize=SMALL, ha="center", va="top")
    ly = yb + 8.4
    for i, (lab, col) in enumerate(VAL):
        swatch(ax, 2.0, ly + i * 2.8, col, lab, fs=SMALL)


def matrix(pg, x0, y0, wmax, c, md):
    cell, cols, shown, sel = md
    taxa = sel.reset_index()
    taxa["row_id"] = np.where(taxa.row_rank.eq("phylum"), "P|" + taxa.phylum, taxa.gg2_name)
    _, taxa["gg2_axis_order"] = leaf_info(taxa.gg2_name)
    tr, rows = tree_rows(taxa[["row_id", "label", "row_rank", "phylum", "node", "gg2_axis_order"]])
    papers = dict(zip(taxa.row_id, taxa.papers))
    RH_, PG = 2.75, 0.8
    ys, grid_h = place_rows(rows, RH_, PG)
    TREE_W = 13.0
    LAB_X = TREE_W + 1.0
    RANK_X = LAB_X + label_width(rows) + 7.2
    X0 = RANK_X + 1.2
    CW, GAP, TOTW = 5.9, 1.1, 6.0
    xs, xx, prev = [], X0, None
    for col in cols.itertuples():
        if prev is not None and col.site != prev:
            xx += GAP
        xs.append(xx); xx += CW; prev = col.site
    XT = xx + GAP
    right = XT + TOTW
    assert right <= wmax, (right, wmax)
    HDR = 26.0
    ax = pg.canvas(x0, y0, right, HDR + grid_h + 34)
    pg.letter(x0, y0, "b")
    pg.text(x0 + 4.2, y0 + 0.5, "Recurrent classifier taxa by body site and comparison", fontsize=HEAD,
            fontweight="bold")
    yN, yL, yS = HDR - 1.5, HDR - 3.2, HDR - 19.5
    ax.text(RANK_X, yN, "Papers", fontsize=TINY, ha="right", va="center")
    ax.text(RANK_X, yL, "Comparison", fontsize=TINY, ha="right", va="bottom")
    for j, col in enumerate(cols.itertuples()):
        xc = xs[j] + CW / 2
        ax.text(xc, yN, str(col.papers), fontsize=TINY, ha="center", va="center")
        ax.text(xc, yL, CTX_LABEL[col.ctx], fontsize=TINY, rotation=90, ha="center", va="bottom")
    ax.text(XT + TOTW / 2, yN, str(cell.key.nunique()), fontsize=TINY, fontweight="bold", ha="center", va="center")
    ax.text(XT + TOTW / 2, yL, "All sites and\ncomparisons", fontsize=TINY, fontweight="bold", rotation=90,
            ha="center", va="bottom", linespacing=1.0)
    for s, g in cols.groupby("site", sort=False):
        xa, xb = xs[g.index.min()] + 0.2, xs[g.index.max()] + CW - 0.2
        ax.plot([xa, xb], [yS, yS], color=BLACK, lw=0.6)
        ax.text((xa + xb) / 2, yS - 0.6, SITE_SHORT[s] if len(g) > 1 else SITE_SHORT[s].replace(" ", "\n"),
                fontsize=TINY, fontweight="bold", ha="center", va="bottom", linespacing=1.0)
    unit = draw_tree_axis(ax, tr, rows, ys, 0, TREE_W, LAB_X, RANK_X, right, RH_, y0=HDR)
    for i, r in enumerate(rows):
        y = HDR + ys[i] - RH_ / 2
        for j, col in enumerate(cols.itertuples()):
            g = shown[(shown.row_id == r["rid"]) & (shown.site == col.site) & (shown.ctx == col.ctx)]
            if g.empty:
                ax.add_patch(Rectangle((xs[j], y), CW, RH_, fc="white", ec=HGRID, lw=0.25, zorder=2))
                continue
            up, dn = int((g.vote == "up").sum()), int((g.vote == "down").sum())
            none, n = int(g.vote.isin(["none", "mixed"]).sum()), g.key.nunique()
            fc = PAL[int(np.clip(up - dn, -8, 8)) + 8]
            ax.add_patch(Rectangle((xs[j], y), CW, RH_, fc=fc, ec=HGRID, lw=0.25, zorder=2))
            if (g.validation_level == "External cohort").any():
                ax.add_patch(Rectangle((xs[j] + 0.35, y + 0.3), CW - 0.7, RH_ - 0.6, fill=False, ec=GOLD, lw=0.8,
                                       zorder=4))
            if n >= 2:
                ax.add_patch(Rectangle((xs[j], y), CW, RH_, fill=False, ec=BLACK, lw=0.9, zorder=5))
            ax.text(xs[j] + CW / 2, y + RH_ / 2, f"{up}/{dn}/{none}", fontsize=3.8, ha="center", va="center",
                    color=text_on(fc), zorder=6)
        n = papers.get(r["rid"])
        if n is not None:
            ax.text(XT + TOTW / 2, y + RH_ / 2, str(int(n)), fontsize=TINY, fontweight="bold", ha="center",
                    va="center")
    yb = HDR + grid_h + 3.5
    scale_bar(ax, 0, yb, unit)
    yb += 4.0
    ax.text(0, yb, "Cell colour = net papers (taxon higher − lower in the unfavourable group); "
            "cell numbers = papers higher / lower / without a direction.", fontsize=SMALL, va="center")
    yb += 3.6
    sw = 3.2
    for i, k in enumerate(range(-3, 4)):
        ax.add_patch(Rectangle((i * sw, yb - 0.9), sw, 1.8, fc=PAL[k + 8], ec="none"))
        ax.text((i + 0.5) * sw, yb + 1.3, ("≤" if k == -3 else "≥+" if k == 3 else "+" if k > 0 else "") + str(k),
                fontsize=TINY, ha="center", va="top")
    ax.add_patch(Rectangle((0, yb - 0.9), 7 * sw, 1.8, fill=False, ec=BLACK, lw=0.3))
    xk = 7 * sw + 2.5
    ax.add_patch(Rectangle((xk, yb - 0.9), sw, 1.8, fc="white", ec=HGRID, lw=0.3))
    ax.text(xk + sw + 0.9, yb, "Not selected", fontsize=SMALL, va="center")
    yb += 5.0
    xk = 0
    for lab, colr, inset, lw in [("2 or more papers in the cell", BLACK, 0.0, 0.9),
                                 ("at least one paper validated externally", GOLD, 0.35, 0.8)]:
        ax.add_patch(Rectangle((xk, yb - 0.9), sw, 1.8, fc="white", ec=HGRID, lw=0.3))
        ax.add_patch(Rectangle((xk + inset, yb - 0.9 + inset * 0.6), sw - 2 * inset, 1.8 - 1.2 * inset, fill=False,
                               ec=colr, lw=lw))
        ax.text(xk + sw + 0.9, yb, lab, fontsize=SMALL, va="center")
        xk += sw + 0.9 + text_w(lab, SMALL) + 4.0
    return HDR + grid_h + 25.0, right


def main():
    c = classifier_votes()
    md = matrix_data(c)
    t = md[3]
    nph = t.phylum.nunique()
    yb = 40.0
    pg = Page(h=yb + 26.0 + (len(t) + nph - (t.row_rank == "phylum").sum()) * 2.75 + (nph - 1) * 0.8 + 25.0)
    landscape(pg, 0, c)
    h_b, w_b = matrix(pg, 0, yb, 128.0, c, md)
    validation(pg, w_b + 3.0, yb, PAGE_W - w_b - 3.0, c)
    pg.save("FigureS5")
    print(f"  Figure S5: {c.key.nunique()} classifier studies, {len(t)} recurrent taxa shown")


if __name__ == "__main__":
    main()
