"""Supplementary Figure S2: agreement between studies sampled in China and elsewhere."""
from functools import lru_cache
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import data as D
from plot import (Page, PAGE_W, TXT, HEAD, LW, BLACK, GRID, GREEN, ORANGE, NA_GREY, SITE_LABEL, SITE_COLOR,
                  text_w, text_on, swatch)
from figure3 import GROUPS

NPERM, SEED = 5000, 2026
MIN_PAPERS = 3
FIG3_PANELS = [(s, ctx) for ctx, _, sites in GROUPS for s in sites]


def paper_votes(d):
    x = d.assign(vote=d.effect.map({"enriched": "up", "depleted": "down"}))
    v = (x.groupby(["site", "ctx", "row_id", "key", "group"]).vote
         .agg(lambda s: "mixed" if s.nunique() > 1 or "mixed" in set(s) else s.iloc[0]).reset_index())
    v["s"] = v.vote.map({"up": 1, "down": -1, "mixed": 0})
    return v


def cell_net(v):
    n = v.groupby(["site", "ctx", "row_id", "group"]).s.sum().unstack()
    k = v.groupby(["site", "ctx", "row_id", "group"]).key.nunique().unstack()
    both = n.notna().all(axis=1)
    return n[both], k[both]


def agreement(v):
    n, _ = cell_net(v)
    nz = n[(n["China"] != 0) & (n["Rest of world"] != 0)]
    return float((np.sign(nz["China"]) == np.sign(nz["Rest of world"])).mean()), len(nz), len(n)


@lru_cache(None)
def compute():
    p = D.papers().set_index("key")
    group = p.country_sampled.map(lambda c: ("China" if c == "China" else "Rest of world") if isinstance(c, str) else None)
    d = D.da_votes().copy()
    d["group"] = d.key.map(group)
    v = paper_votes(d[d.group.notna()])
    obs, n_nz, n_cells = agreement(v)
    rng = np.random.default_rng(SEED)
    lab = v[["site", "ctx", "key", "group"]].drop_duplicates()
    perm = []
    for _ in range(NPERM):
        l2 = lab.copy()
        l2["group"] = l2.groupby(["site", "ctx"]).group.transform(lambda s: rng.permutation(s.values))
        perm.append(agreement(v.drop(columns="group").merge(l2, on=["site", "ctx", "key"]))[0])
    perm = np.array(perm)
    n, k = cell_net(v)
    cells = n.rename(columns={"China": "net_china", "Rest of world": "net_rest"}).join(
        k.rename(columns={"China": "papers_china", "Rest of world": "papers_rest"})).reset_index()

    va = paper_votes(d.assign(group=d.group.fillna("multi-country")))

    def maj(g):
        u, dn = (g.vote == "up").sum(), (g.vote == "down").sum()
        return "higher" if u > dn else "lower" if dn > u else ("tie" if len(g) else "no papers")
    rec = []
    for site, ctx in FIG3_PANELS:
        x = va[(va.site == site) & (va.ctx == ctx)]
        tot = x.groupby("row_id").key.nunique()
        for rid in tot[tot >= MIN_PAPERS].index:
            g = x[x.row_id == rid]
            nr = g[g.group != "China"]
            a, wo = maj(g), maj(nr)
            status = ("no non-Chinese papers" if nr.empty else "holds" if wo == a else
                      "flips" if {wo, a} == {"higher", "lower"} else "becomes a tie" if wo == "tie" else "resolves a tie")
            rec.append({"body_site": site, "comparison": ctx, "row_id": rid, "status": status})
    stats = dict(papers_china=v[v.group == "China"].key.nunique(),
                 papers_rest=v[v.group == "Rest of world"].key.nunique(),
                 cells_both=n_cells, cells_nonzero_both=n_nz, agreement=obs,
                 perm_median=float(np.median(perm)), perm_lo=float(np.percentile(perm, 2.5)),
                 perm_hi=float(np.percentile(perm, 97.5)), p_one_sided=((perm <= obs).sum() + 1) / (NPERM + 1))
    return cells, pd.DataFrame(rec), stats


SAME, OPP = "#E8F5EF", "#B2182B"
SITES = ["gut", "lower-respiratory-tract", "lung-tissue", "oral", "pleural-fluid"]
size = lambda n: 2.0 + 2.2 * n
S3_CTX = {"cancer vs healthy": "cancer vs healthy", "cancer vs benign": "cancer vs benign",
          "tumor vs paired normal": "tumour vs paired normal"}
S3_ORDER = [("gut", "cancer vs healthy"), ("lower-respiratory-tract", "cancer vs healthy"),
            ("oral", "cancer vs healthy"), ("lower-respiratory-tract", "cancer vs benign"),
            ("lower-respiratory-tract", "tumor vs paired normal"), ("lung-tissue", "tumor vs paired normal")]
S3_STATUS = [("holds", "Holds", GREEN), ("becomes a tie", "Becomes a tie", NA_GREY),
             ("flips", "Flips", "#B2182B"), ("resolves a tie", "Resolves a tie", ORANGE),
             ("no non-Chinese papers", "No non-Chinese papers", "#FFFFFF")]


def without_china(ax, W, y0, yb, t):
    n_rows = len(S3_ORDER)
    ROW = (yb - y0) / n_rows
    BARH = ROW * 0.48
    mx = int(t.groupby(["body_site", "comparison"]).size().max())
    upm = (W - 2) / (mx + 1)
    for i, (s, c) in enumerate(S3_ORDER):
        g = t[(t.body_site == s) & (t.comparison == c)]
        yc = y0 + ROW * i + ROW * 0.62
        ax.text(0.8, yc - BARH / 2 - 0.9, f"{SITE_LABEL[s]}", ha="left", va="bottom", fontsize=TXT, fontweight="bold")
        ax.text(0.8 + text_w(SITE_LABEL[s], TXT, "bold") + 1.0, yc - BARH / 2 - 0.9, S3_CTX[c], ha="left",
                va="bottom", fontsize=TXT, color="#555555")
        x = 0.0
        for key, _, col in S3_STATUS:
            n = int((g.status == key).sum())
            if not n:
                continue
            ax.add_patch(Rectangle((x, yc - BARH / 2), n * upm, BARH, fc=col,
                                   ec=GRID if col == "#FFFFFF" else "none", lw=0.4))
            ax.text(x + n * upm / 2, yc, str(n), ha="center", va="center", fontsize=TXT, fontweight="bold",
                    color=text_on(col))
            x += n * upm
    ax.plot([0, 0], [y0, yb], color=BLACK, lw=LW)
    ax.plot([0, mx * upm + 0.5 * upm], [yb, yb], color=BLACK, lw=LW)
    for k in range(0, mx + 1, 5):
        ax.plot([k * upm] * 2, [yb, yb + 0.7], color=BLACK, lw=LW)
        ax.text(k * upm, yb + 1.2, str(k), ha="center", va="top", fontsize=TXT)
    ax.text(mx * upm / 2, yb + 4.2, "Figure 3 taxa (reported by ≥ 3 papers)", ha="center", va="top", fontsize=TXT)
    lx, ly = W * 0.6, y0 + ROW * 2.25
    for _, lab, col in S3_STATUS:
        swatch(ax, lx, ly, col, lab, w=3.0, h=2.2, ec=GRID, lw=0.4)
        ly += 3.6


def main():
    cells, t, st = compute()
    H = 78.0
    pg = Page(h=H)
    pg.letter(0, 0, "a")
    pg.text(4.2, 0.5, "Net papers per taxon, China vs rest of world", fontsize=HEAD, fontweight="bold")
    a = pg.axes(12, 8, 62, 62)
    lim = int(max(cells.net_china.abs().max(), cells.net_rest.abs().max())) + 1
    a.fill_between([0, lim], 0, lim, color=SAME, lw=0, zorder=0)
    a.fill_between([-lim, 0], -lim, 0, color=SAME, lw=0, zorder=0)
    a.axhline(0, color=BLACK, lw=LW, zorder=1); a.axvline(0, color=BLACK, lw=LW, zorder=1)
    rng = np.random.default_rng(7)
    for s in SITES:
        g = cells[cells.site == s]
        if g.empty:
            continue
        jx, jy = rng.uniform(-0.22, 0.22, len(g)), rng.uniform(-0.22, 0.22, len(g))
        a.scatter(g.net_china + jx, g.net_rest + jy, s=size(g.papers_china + g.papers_rest), color=SITE_COLOR[s],
                  edgecolor="white", lw=0.3, alpha=0.9, zorder=3)
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    tk = range(-(lim // 2) * 2, (lim // 2) * 2 + 1, 2)
    a.set_xticks(tk); a.set_yticks(tk)
    a.set_xlabel("China: net papers (higher − lower in the unfavourable group)")
    a.set_ylabel("Rest of world: net papers")
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.text(lim * 0.95, lim * 0.95, "same direction", ha="right", va="top", fontsize=TXT, color="#00694A",
           fontweight="bold")
    a.text(-lim * 0.95, lim * 0.95, "opposite", ha="left", va="top", fontsize=TXT, color=OPP, fontweight="bold")
    lg = pg.canvas(78, 8, 24, 62)
    lg.text(0, 0, "Body site", fontsize=TXT, fontweight="bold", va="top")
    y = 4.0
    for s in SITES:
        if (cells.site == s).any():
            lg.scatter([1.2], [y], s=size(4), color=SITE_COLOR[s], edgecolor="white", lw=0.3, clip_on=False)
            lg.text(3.0, y, SITE_LABEL[s], fontsize=TXT, va="center")
            y += 3.2
    y += 2.0
    lg.text(0, y, "Papers", fontsize=TXT, fontweight="bold", va="top")
    y += 4.0
    for n in (2, 6, 10):
        lg.scatter([1.2], [y], s=size(n), color=NA_GREY, edgecolor="white", lw=0.3, clip_on=False)
        lg.text(3.0, y, str(n), fontsize=TXT, va="center")
        y += 3.6
    X2 = 106
    pg.letter(X2 - 4, 0, "b")
    pg.text(X2 + 0.2, 0.5, "Figure 3 majority directions without China", fontsize=HEAD, fontweight="bold")
    without_china(pg.canvas(X2, 0, PAGE_W - X2 - 1, H), PAGE_W - X2 - 1, 8.0, 70.0, t)
    pg.save("FigureS2")


if __name__ == "__main__":
    main()
    c, t, s = compute()
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()})
    print(len(t), t.status.value_counts().to_dict())
