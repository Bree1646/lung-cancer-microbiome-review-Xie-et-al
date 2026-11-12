"""Figure 3: differential-abundance consensus by comparison and body site."""
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import data as D
from plot import (Page, PAGE_W, TXT, SMALL, TINY, BLACK, LIGHTGRID, UP, DOWN, LIGHT_UP, LIGHT_DOWN, LOW_AGREE,
                  swatch, tree_rows, place_rows, draw_tree_axis, scale_bar, label_width)

MIN_PAPERS = 3
# comparison -> body sites, in the order drawn
GROUPS = [("cancer vs healthy", "Cancer vs healthy", ["gut", "lower-respiratory-tract", "oral"]),
          ("cancer vs benign", "Cancer vs benign", ["lower-respiratory-tract"]),
          ("tumor vs paired normal", "Tumor vs paired normal", ["lung-tissue", "lower-respiratory-tract"])]
SITE_SHORT = {"gut": "Gut", "lower-respiratory-tract": "Lower resp. tract", "lung-tissue": "Lung tissue",
              "oral": "Oral"}
RH, PGAP = 2.6, 0.9
BARF = 0.62
MIXED = "#777777"


def paper_votes(x):
    rec = []
    for (key, rid), g in x.groupby(["key", "row_id"]):
        eff = set(g.effect)
        rec.append({"key": key, "row_id": rid,
                    "vote": "mixed" if len(eff) > 1 else ("up" if eff == {"enriched"} else "down"),
                    "tested": bool(g.evidentiary_tier.astype(str).str.lower().str.contains("tested").any())})
    return pd.DataFrame(rec)


def summarise(pv):
    out = {}
    for rid, g in pv.groupby("row_id"):
        up, dn, mx = g[g.vote == "up"], g[g.vote == "down"], g[g.vote == "mixed"]
        out[rid] = {"papers_total": len(g), "papers_increased": len(up), "papers_decreased": len(dn),
                    "papers_mixed": len(mx), "tested_increased": int(up.tested.sum()),
                    "descriptive_increased": int((~up.tested).sum()), "tested_decreased": int(dn.tested.sum()),
                    "descriptive_decreased": int((~dn.tested).sum()),
                    "agreement": max(len(up), len(dn)) / len(g)}
    return out


def compute():
    d = D.da_votes()
    panels = []
    for ctx, glabel, sites in GROUPS:
        for s in sites:
            x = d[(d.ctx == ctx) & (d.site == s)]
            sm = summarise(paper_votes(x))
            panels.append({"ctx": ctx, "group": glabel, "site": s, "n": x.key.nunique(), "sm": sm,
                           "q": {r for r, v in sm.items() if v["papers_total"] >= MIN_PAPERS}})
    ids = sorted(set().union(*(p["q"] for p in panels)))
    meta = d.groupby("row_id").agg(label=("tree_leaf_label", "first"), rank=("rank_disp", "first"),
                                   node=("tree_node_id", "first"), phylum=("phylum", "first"),
                                   order=("gg2_axis_order", "min"))
    own = d.groupby("row_id").tree_node_id.agg(lambda s: sorted(set(int(v) for v in s)))
    taxa = [r for r in ids if not r.startswith("P|")]
    lab = D.figure_labels([meta.label[r] for r in taxa])
    sel = []
    for p in sorted({meta.phylum[r] for r in ids}):
        nodes = [int(meta.node[r]) for r in taxa if meta.phylum[r] == p] + list(own.get("P|" + p, []))
        sel.append({"row_id": "P|" + p, "label": p, "row_rank": "phylum", "phylum": p,
                    "node": D.tree().mrca(nodes), "gg2_axis_order": -1})
    for r in taxa:
        m = meta.loc[r]
        sel.append({"row_id": r, "label": lab[m.label], "row_rank": m["rank"], "phylum": m.phylum,
                    "node": int(m.node), "gg2_axis_order": m.order})
    tr, rows = tree_rows(pd.DataFrame(sel))
    return panels, tr, rows


def main():
    panels, tr, rows = compute()
    ys, grid_h = place_rows(rows, RH, PGAP)

    W = PAGE_W
    TREE_X, TREE_W = 0.0, 15.0
    LAB_X = TREE_X + TREE_W + 1.2
    RANK_X = LAB_X + label_width(rows) + 7.5
    X0 = RANK_X + 1.8
    GAP = 3.2
    pan = []
    for p in panels:
        c = pd.DataFrame({r: p["sm"][r] for r in p["q"]}).T
        mx = int(max(c.papers_increased.max(), c.papers_decreased.max()))
        pan.append(dict(st=p["group"], site=p["site"], n=p["n"], cells=c, half=mx + 1.2))
    MIN_W, avail = 13.0, W - X0 - GAP * (len(pan) - 1) - 2.0
    data_half = [p["half"] for p in pan]
    lo, hi = 0.01, 10.0
    for _ in range(80):
        upm = (lo + hi) / 2
        if sum(max(2 * h * upm, MIN_W) for h in data_half) > avail:
            hi = upm
        else:
            lo = upm
    upm = lo
    for p, h in zip(pan, [max(h, MIN_W / (2 * upm)) for h in data_half]):
        p["lim"] = int(np.floor(p["half"] - 1.2 + 1e-9))
        p["half"] = h
    x = X0
    for p in pan:
        p["x0"], p["w"] = x, 2 * p["half"] * upm
        p["xc"] = x + p["half"] * upm
        x += p["w"] + GAP

    HEAD = 17.0
    LEG_Y = HEAD + grid_h + 11.5
    H = LEG_Y + 12.0
    pg = Page(h=H)
    ax = pg.canvas(0, 0, W, H)

    ax.text(RANK_X, HEAD - 12.2, "Study type", fontsize=SMALL, ha="right", va="center")
    ax.text(RANK_X, HEAD - 7.2, "Body site", fontsize=SMALL, ha="right", va="center")
    for st in dict.fromkeys(p["st"] for p in pan):
        g = [p for p in pan if p["st"] == st]
        xa, xb = g[0]["x0"], g[-1]["x0"] + g[-1]["w"]
        ax.text((xa + xb) / 2, HEAD - 13.2, st, fontsize=TXT, fontweight="bold", ha="center", va="bottom")
        ax.plot([xa, xb], [HEAD - 12.2] * 2, color=BLACK, lw=0.6)
    for p in pan:
        ax.text(p["xc"], HEAD - 7.2, SITE_SHORT[p["site"]], fontsize=SMALL, fontweight="bold", ha="center",
                va="center")
        ax.text(p["xc"], HEAD - 3.6, f"n = {p['n']} papers", fontsize=TINY, ha="center", va="center")

    unit = draw_tree_axis(ax, tr, rows, ys, TREE_X, TREE_W, LAB_X, RANK_X, W, RH, y0=HEAD)
    scale_bar(ax, TREE_X, HEAD + grid_h + 4.5, unit)
    xm = (X0 + W) / 2
    ax.text(xm - 1.0, HEAD + grid_h + 4.6, "← papers: depleted in cancer or tumour", fontsize=SMALL, ha="right", va="top")
    ax.text(xm + 1.0, HEAD + grid_h + 4.6, "enriched in cancer or tumour →", fontsize=SMALL, ha="left", va="top")

    bh = RH * BARF
    for p in pan:
        c = p["cells"]
        top, bot = HEAD, HEAD + grid_h
        lim = p["lim"]
        step = 4 if lim > 6 else 2
        for t in range(-lim, lim + 1):
            if t and t % step == 0:
                xx = p["xc"] + t * upm
                ax.plot([xx, xx], [top, bot], color=LIGHTGRID, lw=0.35, zorder=0)
        for i, r in enumerate(rows):
            if r["rid"] not in c.index:
                continue
            e = c.loc[r["rid"]]
            y = HEAD + ys[i]
            if e.agreement < 0.70:
                ax.add_patch(Rectangle((p["x0"], y - RH / 2), p["w"], RH, fc=LOW_AGREE, ec="none", zorder=0.5))
            for start, length, fc, ec in [(0, e.tested_increased, UP, "none"),
                                          (e.tested_increased, e.descriptive_increased, LIGHT_UP, UP),
                                          (0, -e.tested_decreased, DOWN, "none"),
                                          (-e.tested_decreased, -e.descriptive_decreased, LIGHT_DOWN, DOWN)]:
                if length:
                    ax.add_patch(Rectangle((p["xc"] + start * upm, y - bh / 2), length * upm, bh, fc=fc, ec=ec,
                                           lw=0.3, zorder=3))
            if e.papers_increased:
                ax.text(p["xc"] + e.papers_increased * upm + 0.6, y, str(int(e.papers_increased)), fontsize=TINY,
                        fontweight="bold", va="center", ha="left", zorder=5)
            if e.papers_decreased:
                ax.text(p["xc"] - e.papers_decreased * upm - 0.6, y, str(int(e.papers_decreased)), fontsize=TINY,
                        fontweight="bold", va="center", ha="right", zorder=5)
            if e.papers_mixed:
                ax.plot([p["xc"]], [y], marker="D", ms=1.8, mfc=MIXED, mec="white", mew=0.3, zorder=6)
        ax.plot([p["xc"]] * 2, [top, bot], color=BLACK, lw=0.6, zorder=4)
        ax.plot([p["x0"], p["x0"] + p["w"]], [bot] * 2, color=BLACK, lw=0.5)
        for t in range(-lim, lim + 1):
            if t % step == 0:
                xx = p["xc"] + t * upm
                ax.plot([xx, xx], [bot, bot + 0.8], color=BLACK, lw=0.4)
                ax.text(xx, bot + 1.1, str(abs(t)), fontsize=TINY, ha="center", va="top")

    x = 0.0
    for fc, ec, lab in [(UP, "none", "Enriched in cancer or tumour, tested"),
                        (LIGHT_UP, UP, "Enriched, descriptive"),
                        (DOWN, "none", "Depleted in cancer or tumour, tested"),
                        (LIGHT_DOWN, DOWN, "Depleted, descriptive"),
                        (LOW_AGREE, "#BBBBBB", "Agreement below 0.70")]:
        x = swatch(ax, x, LEG_Y, fc, lab, fs=SMALL, ec=ec, lw=0.3)
    ax.plot([x + 1.2], [LEG_Y], marker="D", ms=2.2, mfc=MIXED, mec="white", mew=0.3)
    ax.text(x + 2.6, LEG_Y, "Paper reporting both directions", fontsize=SMALL, va="center")
    ax.text(0, LEG_Y + 3.6, "Bars = papers, one vote per paper and taxon in each panel, on the same scale in "
            "every panel. A taxon is drawn in a panel only when at least 3 of that panel's papers report it.",
            fontsize=SMALL, va="center")
    pg.save("Figure3")


if __name__ == "__main__":
    main()
