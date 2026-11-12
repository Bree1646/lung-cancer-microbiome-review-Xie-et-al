"""Figure 2: differential abundance of reported taxa in favourable and unfavourable groups."""
import math
import numpy as np
from matplotlib.patches import Rectangle, Wedge
from matplotlib.collections import PatchCollection, LineCollection
import data as D
from plot import (Page, PAGE_W, SMALL, TINY, BLACK, ITALIC, PT_MM, text_w, swatch, scale_bar, phylum_label)
from site_icons import draw_icon

PAL = ['#08254f', '#123e73', '#1c5999', '#2672bd', '#448ed6', '#74abe1', '#a6c9ec', '#d6e6f7', '#c8c8c8',
       '#fddad3', '#f7ada4', '#f17e73', '#eb5043', '#d53127', '#b1211e', '#8b1015', '#67000d']
A_SHORT = {"gut": "Gut", "lower-respiratory-tract": "LRT", "lung-tissue": "Lung", "oral": "Oral",
           "blood-plasma-serum": "Blood", "pleural-fluid": "Pleural", "other": "Other"}
FS_TIP = 3.9
SITE_GREY = "#7A7A7A"
NET_NOTE = "Net papers = papers enriched − papers depleted in the unfavourable group"
LABS = ["≤−8"] + [str(i).replace("-", "−") for i in range(-7, 0)] + ["0"] + [f"+{i}" for i in range(1, 8)] + ["≥+8"]


def colour(u, v):
    return "white" if (u == 0 and v == 0) else PAL[int(np.clip(u - v, -8, 8)) + 8]


def compute():
    d = D.da_votes()
    rows, layout = D.valence_rows(d)
    sn = d.groupby("site").key.nunique().sort_values(ascending=False, kind="stable")
    cols = list(sn.items())
    idx = {r["row_id"]: i for i, r in enumerate(rows)}
    up = np.zeros((len(rows), len(cols)), int)
    dn = np.zeros_like(up)
    for j, (s, _) in enumerate(cols):
        for (r, eff), g in d[d.site == s].groupby(["row_id", "effect"]):
            if r in idx:
                (up if eff == "enriched" else dn)[idx[r], j] = g.key.nunique()
    keep = np.array([r["kind"] == "taxon" or r["n_papers"] >= D.MIN_PAPERS_VALENCE for r in rows])
    up[~keep] = 0
    dn[~keep] = 0
    return rows, layout, cols, up, dn


def tip_text(r):
    return r["label"].split(" (")[0] if r["kind"] == "phylum" else r["label"]


def main():
    rows, (root, kids), cols, up, dn = compute()
    tree = D.tree()
    pcol = D.phylum_colours()
    phy = [r["phylum"] for r in rows]
    sets = D.tree_phyla(root, kids, dict(enumerate(phy)))
    colour_of = lambda nd: pcol[next(iter(sets[nd]))] if len(sets[nd]) == 1 else BLACK
    base = tree.length[root]
    maxd = max(tree.length[r["node"]] - base for r in rows)

    n, m = len(rows), len(cols)
    W = PAGE_W
    GAP = 46.0
    step = (360.0 - GAP) / n
    ang = lambda k: 90.0 - GAP / 2 - (k + 0.5) * step
    R_BAND = W / 2 - 24.0 - 0.8
    R_BAND0 = R_BAND - 1.4
    R_TREE = 27.0
    R_A0 = R_TREE + 1.0
    DR = (R_BAND0 - 0.5 - R_A0) / m
    rin = [R_A0 + j * DR for j in range(m)]
    CX, CY = W / 2, W / 2 + 1.0
    KEY_Y = 2 * CY + 0.5
    H = KEY_Y + 24.0
    pg = Page(h=H)
    ax = pg.canvas(0, 0, W, H)
    ax.set_aspect("equal")
    P = lambda rad, deg: (CX + rad * math.cos(math.radians(deg)), CY - rad * math.sin(math.radians(deg)))
    wedge = lambda r0, r1, a0, a1: Wedge((CX, CY), r1, -a1, -a0, width=r1 - r0)

    patches, fcs, band = [], [], []
    for k in range(n):
        a0, a1 = ang(k) - step / 2, ang(k) + step / 2
        for j, r0 in enumerate(rin):
            patches.append(wedge(r0, r0 + DR, a0, a1))
            fcs.append(colour(up[k, j], dn[k, j]))
        band.append(wedge(R_BAND0, R_BAND, a0, a1))
    ax.add_collection(PatchCollection(patches, facecolors=fcs, edgecolors="#D5D5D5", linewidths=0.12, zorder=2))
    ax.add_collection(PatchCollection(band, facecolors=[pcol[p] for p in phy], edgecolors="none", zorder=3))
    for k, r in enumerate(rows):
        a = ang(k)
        x, y = P(R_BAND + 0.6, a)
        flip = 90 < a % 360 < 270
        ax.text(x, y, tip_text(r), fontsize=FS_TIP, rotation=(a + 180) if flip else a, rotation_mode="anchor",
                ha="right" if flip else "left", va="center",
                fontweight="bold" if r["kind"] == "phylum" else "normal",
                fontstyle="italic" if r["rank"] in ITALIC else "normal")

    r0 = 1.0
    unit = (R_TREE - 0.4 - r0) / maxd

    def arc(dd, a0, a1):
        mm = max(2, int(abs(a1 - a0) / 0.5) + 2)
        return [P(r0 + dd * unit, t) for t in np.linspace(a0, a1, mm)]
    segs, scol, ldep = D.draw_tree_lines(tree, root, kids, {k: ang(k) for k in range(n)}, colour_of,
                                         lambda dd, a: P(r0 + dd * unit, a), arc)
    ax.add_collection(LineCollection(segs, colors=scol, linewidths=0.35, zorder=4, capstyle="round"))
    lead = [[P(r0 + ldep[k] * unit, ang(k)), P(R_TREE + 0.8, ang(k))] for k in range(n)]
    ax.add_collection(LineCollection(lead, colors=[pcol[p] for p in phy], linewidths=0.18,
                                     linestyles=[(0, (1, 1.6))], zorder=3))

    ICON = 2.2
    for j, (site, n_art) in enumerate(cols):
        y = CY - (rin[j] + DR / 2)
        lab = f"{A_SHORT[site]} ({n_art})"
        w = ICON + 0.4 + text_w(lab, SMALL, "bold")
        x = CX - w / 2
        draw_icon(ax, site, x + ICON / 2, y, ICON, SITE_GREY, lw=ICON * 0.075 / PT_MM, flip=True)
        ax.text(x + ICON + 0.4, y, lab, fontsize=SMALL, fontweight="bold", ha="left", va="center")
    a_lo, a_hi = ang(n - 1) - step / 2, ang(0) + step / 2
    for j in range(m):
        for a, sgn in ((a_hi, 1), (a_lo, -1)):
            aa = a + sgn * math.degrees(0.6 / (rin[j] + DR / 2))
            ax.plot(*zip(P(rin[j] + 0.25, aa), P(rin[j] + DR - 0.25, aa)), color=SITE_GREY, lw=2.2,
                    solid_capstyle="butt", zorder=5)

    y = KEY_Y
    ax.text(0, y, NET_NOTE, fontsize=SMALL, va="top")
    y += 3.0
    sw = 4.6
    for i, c in enumerate(PAL):
        ax.add_patch(Rectangle((i * sw, y), sw, 1.8, fc=c, ec="none"))
        ax.text((i + 0.5) * sw, y + 2.2, LABS[i], fontsize=TINY, ha="center", va="top")
    ax.add_patch(Rectangle((0, y), 17 * sw, 1.8, fill=False, ec=BLACK, lw=0.3))
    ax.add_patch(Rectangle((17 * sw + 3, y), sw * 0.6, 1.8, fc="white", ec="#AAAAAA", lw=0.3))
    ax.text(17 * sw + 3 + sw * 0.6 + 1.0, y + 0.9, "not reported; grey = equal", fontsize=SMALL, va="center")
    scale_bar(ax, 17 * sw + 3, y + 5.0, unit)
    y += 7.0
    phyla = list(dict.fromkeys(phy))
    colw = W / 4
    for i, p in enumerate(phyla):
        swatch(ax, (i % 4) * colw, y + (i // 4) * 2.6, pcol[p], phylum_label(p), fs=TINY, w=2.0, h=1.5)
    pg.save("Figure2")
    n_taxa = sum(r["kind"] == "taxon" for r in rows)
    print(f"  Figure 2: {n_taxa} taxa in >= 2 studies, {len(phyla)} phylum rows")


if __name__ == "__main__":
    main()
