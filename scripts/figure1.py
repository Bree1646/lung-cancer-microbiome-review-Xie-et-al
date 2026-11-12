"""Figure 1: the shape of the lung cancer microbiome literature."""
import re
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.colors import LogNorm, LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle, Wedge
from matplotlib.ticker import FixedLocator, NullLocator
import data as D
from plot import (Page, SMALL, HEAD, LW, BLACK, BAR, BLUE, ORANGE, PURPLE, GREEN, PT_MM, SITE_LABEL, SITE_COLOR,
                  SITE_BY_LABEL, text_w, text_on)
from site_icons import draw_icon

HEAD_H = 4.2
METHOD_COLOR = {"16S rRNA": BLUE, "Shotgun metagenomic": ORANGE, "More than one method": PURPLE,
                "Long-read nanopore": BAR}
MAP_CMAP = LinearSegmentedColormap.from_list(
    "blues_trunc", [matplotlib.colormaps["Blues"](x) for x in np.linspace(0.30, 1.0, 256)])
DESIGN_LABEL = {"disease-vs-healthy": "Cancer vs healthy", "disease-vs-benign": "Cancer vs benign",
                "within-patient-paired": "Within-patient paired (tumor/normal)",
                "therapy-responder-vs-nonresponder": "Therapy responder vs non-responder",
                "single-arm-descriptive": "Single-arm descriptive",
                "disease-subtype-stratified": "Cancer subtype stratified",
                "severity-stratified": "Severity / stage stratified",
                "disease-vs-disease": "Cancer vs other disease", "pre-post-treatment": "Pre / post treatment",
                "comorbidity-stratified": "Comorbidity / exposure stratified"}
ICD_DX = {"0": "2C25.0 Adenocarcinoma", "1": "2C25.1 Small cell", "2": "2C25.2 Squamous cell", "3": "2C25.3 Large cell",
          "4": "2C25.4 Neuroendocrine", "Y": "2C25.Y Other specified", "Z": "2C25.Z Unspecified"}
ICD_STAGE = {"XS1G": "XS1G Stage I", "XS4P": "XS4P Stage II", "XS6H": "XS6H Stage III", "XS9R": "XS9R Stage IV"}
CONTEXT_LABELS = {
    "cancer vs healthy": "Cancer versus healthy controls", "cancer vs benign": "Cancer versus benign lung disease",
    "tumor vs paired normal": "Tumor versus paired adjacent-normal tissue",
    "stage comparison": "Stage comparison", "subtype comparison": "Subtype comparison",
    "metastasis": "Metastasis", "recurrence": "Recurrence", "survival": "Survival",
    "therapy response": "Therapy response", "pre vs post treatment": "Pre versus post treatment",
    "smoking": "Smoking", "comorbidity": "Comorbidity", "demographic": "Demographic",
    "body site comparison": "Body-site comparison", "other": "Other",
}
G_SITES = {"gut": "Gut", "lower-respiratory-tract": "Lower respiratory tract", "lung-tissue": "Lung tissue",
           "oral": "Oral", "blood-plasma-serum": "Blood, plasma or serum", "pleural-fluid": "Pleural fluid",
           "other": "Other"}
ALL = "All body sites"


def years_data(m):
    t = pd.crosstab(m.year.astype(int), m.assay_class)
    yrs = list(range(t.index.min(), t.index.max() + 1))
    t = t.reindex(index=yrs, columns=[c for c in METHOD_COLOR if c in t.columns], fill_value=0)
    t.insert(0, "papers", t.sum(axis=1))
    return t.rename_axis("year").reset_index()


def impact_data(m):
    from scipy.stats import gaussian_kde
    vals = pd.to_numeric(m.Impact_factor, errors="coerce").dropna()
    kde = gaussian_kde(np.log10(vals.values))
    gx = np.linspace(np.log10(0.5), np.log10(100.0), 600)
    return vals, 10 ** gx, kde(gx), dict(median=float(vals.median()), q1=float(vals.quantile(0.25)),
                                         q3=float(vals.quantile(0.75)))


def sites_data(m):
    c = m.explode("sites").sites.value_counts()
    return [(SITE_LABEL[s], int(c[s]), round(100 * int(c[s]) / len(m), 1)) for s in D.SITE_ORDER if s in c.index]


def design_data(m):
    c = m.designs.explode().value_counts()
    return [(DESIGN_LABEL.get(k, k), int(c[k])) for k in c.index if k != "in-vitro-validation"]


def icd_data(m):
    dx = m.ICD11_diagnosis.astype(str).map(lambda s: sorted(set(re.findall(r"2C25\.?([0-9YZ])", s))))
    stg = m.ICD11_stage.astype(str).map(lambda s: sorted(set(re.findall(r"XS[0-9A-Z]{2}", s))))
    cd, cs = dx.explode().dropna().value_counts(), stg.explode().dropna().value_counts()
    return ([(ICD_DX[k], int(cd[k])) for k in cd.index],
            [(ICD_STAGE[k], int(cs[k])) for k in ICD_STAGE if k in cs.index])


def grid_data():
    v = D.votes()
    v = v[v.finding_type == "differential abundance"]
    d = v.assign(ctx=v.comparison_context.str.split(r"\s*,\s*")).explode("ctx")
    d = d.assign(site=d.body_site_resolved.str.split(r"\s*,\s*")).explode("site")
    d["comparison"], d["body_site"] = d.ctx.map(CONTEXT_LABELS), d.site.map(G_SITES)
    sites = [G_SITES[s] for s in G_SITES if G_SITES[s] in set(d.body_site)]
    e = d[d.direction_vote.isin(["up", "down", "mixed"]) & ~d.is_cross_site_comparison.astype(bool)]
    rows = []
    for c, g in e.groupby("comparison"):
        for s, h in list(g.groupby("body_site")) + [(ALL, g)]:
            rows.append({"comparison": c, "body_site": s, "papers": h.key.nunique()})
    c = pd.DataFrame(rows)
    p = c.pivot(index="comparison", columns="body_site", values="papers").reindex(columns=sites + [ALL])
    p = p.fillna(0).astype(int)
    order = c[c.body_site == ALL].set_index("comparison").papers.sort_values(ascending=False, kind="stable").index
    p = p.reindex(order).reset_index()
    collapse = ["Blood, plasma or serum", "Pleural fluid", "Other"]
    assert ((p[collapse] > 0).sum(axis=1) <= 1).all()
    p.insert(p.columns.get_loc("Blood, plasma or serum"), "Other sites", p[collapse].sum(axis=1))
    return p.drop(columns=collapse)


def panel_head(pg, x, y, letter, title):
    pg.letter(x, y, letter)
    pg.text(x + 4.2, y + 0.5, title, fontsize=HEAD, fontweight="bold")


def years(pg, x, y, w, h, m):
    t = years_data(m)
    ax = pg.axes(x + 8, y + HEAD_H + 1, w - 9, h - HEAD_H - 8)
    bottom = np.zeros(len(t))
    for c in METHOD_COLOR:
        ax.bar(t.year, t[c], bottom=bottom, color=METHOD_COLOR[c], width=0.72, lw=0,
               label=f"{c} ({int(t[c].sum())})")
        bottom += t[c].to_numpy()
    for yr, v in zip(t.year, t.papers):
        if v:
            ax.text(yr, v + 0.6, str(v), ha="center", va="bottom", fontsize=SMALL, fontweight="bold")
    ax.set_xticks(t.year)
    ax.set_xticklabels([str(v) for v in t.year], rotation=90)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_ylim(0, t.papers.max() * 1.12)
    ax.set_ylabel("Papers")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(t.year.min() - 0.6, t.year.max() + 0.6)
    ax.legend(loc="upper left", handlelength=1.0, handleheight=0.9, borderaxespad=0.1, labelspacing=0.3,
              fontsize=SMALL)


def world():
    import geopandas as gpd
    import pyogrio
    from pathlib import Path
    shp = Path(pyogrio.__file__).parent / "tests" / "fixtures" / "naturalearth_lowres" / "naturalearth_lowres.shp"
    w = gpd.read_file(shp)
    for name, code in {"Taiwan": "TWN", "France": "FRA", "Norway": "NOR", "Cyprus": "CYP"}.items():
        w.loc[w["name"].eq(name), "iso_a3"] = code
    return w


def country_map(pg, x, y, w, h, m):
    c = m.sampled_iso.dropna().value_counts()
    wd = world()
    assert set(c.index) <= set(wd.iso_a3)
    wd["papers"] = wd.iso_a3.map(c)
    norm = LogNorm(vmin=1, vmax=c.max())
    map_h = h - HEAD_H - 9
    ax = pg.axes(x, y + HEAD_H, w, map_h)
    wd.plot(ax=ax, color="#EEEEEE", edgecolor="white", linewidth=0.25)
    wd.dropna(subset=["papers"]).plot(ax=ax, column="papers", cmap=MAP_CMAP, norm=norm,
                                      edgecolor="#4D4D4D", linewidth=0.25)
    ax.set_xlim(-170, 180); ax.set_ylim(-58, 85); ax.set_aspect("auto"); ax.axis("off")
    cax = pg.axes(x + w * 0.3, y + HEAD_H + map_h + 1.2, w * 0.4, 1.8)
    sm = matplotlib.cm.ScalarMappable(cmap=MAP_CMAP, norm=norm)
    cb = pg.fig.colorbar(sm, cax=cax, orientation="horizontal")
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100) if t <= c.max()] + [int(c.max())]
    cb.set_ticks(ticks); cb.set_ticklabels([str(t) for t in ticks])
    cb.ax.xaxis.set_minor_locator(NullLocator())
    cb.outline.set_linewidth(LW); cb.ax.tick_params(labelsize=SMALL, length=1.5, pad=1)
    cb.set_label("Papers (log scale)", fontsize=SMALL, labelpad=1)


def impact(pg, x, y, w, h, m):
    vals, xx, yy, s = impact_data(m)
    ax = pg.axes(x + 7, y + HEAD_H + 1, w - 8, h - HEAD_H - 8)
    ax.fill_between(xx, 0, yy, color="#C6E3F7", lw=0)
    iqr = (xx >= s["q1"]) & (xx <= s["q3"])
    ax.fill_between(xx[iqr], 0, yy[iqr], color="#7FBADF", lw=0, label=f"Middle half ({s['q1']:.1f}–{s['q3']:.1f})")
    ax.plot(xx, yy, color=BLUE, lw=0.8)
    ym = np.interp(s["median"], xx, yy)
    ax.plot([s["median"]] * 2, [0, ym], color=BLACK, lw=0.6, ls=(0, (3, 2)))
    ax.text(s["median"] * 1.15, ym * 1.02, f"Median {s['median']:.1f}", fontsize=SMALL, fontweight="bold",
            va="bottom", ha="left")
    top = yy.max() * 1.18
    rug = -top * 0.07
    for v in vals:
        ax.plot([v, v], [rug * 0.25, rug * 1.25], color=BLACK, lw=0.3)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 100); ax.set_ylim(rug * 1.6, top)
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 50, 100]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("Journal impact factor (log scale)")
    ax.set_ylabel("Density")
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_bounds(0, top)
    ax.axhline(0, color=BLACK, lw=LW)
    ax.legend(loc="upper left", fontsize=SMALL, handlelength=1.0, borderaxespad=0.1)


def hbar_axes(pg, x, y, w, h, labels, values, lab_w, xmax=None, step=None, xlabel="Papers"):
    ax = pg.axes(x + lab_w, y, w - lab_w - 4.0, h)
    n = len(values)
    yy = np.arange(n)
    ax.barh(yy, values, height=0.68, color=BAR, lw=0)
    xmax = xmax or max(values) * 1.15
    for yi, v in zip(yy, values):
        ax.text(v + xmax * 0.012, yi, str(v), va="center", ha="left", fontsize=SMALL, fontweight="bold")
    ax.set_yticks(yy); ax.set_yticklabels(labels)
    ax.set_ylim(n - 0.45, -0.55)
    ax.set_xlim(0, xmax)
    if step:
        ax.set_xticks(np.arange(0, xmax + 1e-9, step))
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(xlabel, labelpad=1.0)
    return ax


def design(pg, x, y, w, h, m, step=10):
    t = design_data(m)
    hbar_axes(pg, x, y + HEAD_H + 1, w, h - HEAD_H - 7, [l for l, _ in t], [n for _, n in t], 36.0, xmax=55, step=step)


def icd11(pg, x, y, w, h, m, lab_w=27.0):
    dg, st = icd_data(m)
    gap, sub = 8.0, 3.4
    avail = h - HEAD_H - 2 * sub - gap - 7
    row = avail / (len(dg) + len(st))
    yy = y + HEAD_H
    for t, head_ in ((dg, "Diagnosis code"), (st, "Stage extension")):
        pg.text(x, yy, head_, fontsize=SMALL, fontweight="bold")
        hh = row * len(t)
        hbar_axes(pg, x, yy + sub, w, hh, [l for l, _ in t], [n for _, n in t], lab_w,
                  xmax=115, step=25, xlabel="Papers" if t is st else "")
        yy += sub + hh + gap


PIE_NAME = {"Lower respiratory tract": "Lower respiratory\ntract", "Blood / plasma / serum": "Blood / plasma /\nserum"}


def body_sites(pg, x, y, w, h, m, d):
    t = sites_data(m)
    ax = pg.canvas(x, y + HEAD_H, w, h - HEAD_H)
    R = d / 2
    cx, cy = w / 2, (h - HEAD_H) / 2 + 0.5
    tot = sum(n for _, n, _ in t)
    LH, GAP, PAD = 2.05, 4.6, 0.9
    RAD = 1.6
    a0 = 90.0
    L = []
    for site, n, pct in t:
        c = SITE_COLOR[SITE_BY_LABEL[site]]
        a1 = a0 - 360.0 * n / tot
        ax.add_patch(Wedge((cx, cy), R, -a0, -a1, fc=c, ec="white", lw=0.6))
        txt = f"{PIE_NAME.get(site, site)}\n{n} ({pct:.0f}%)"
        L.append(dict(m=(a0 + a1) / 2, th=(a0 + a1) / 2, txt=txt,
                      tw=max(text_w(s_, SMALL) for s_ in txt.split("\n")), hh=(txt.count("\n") + 1) * LH))
        a0 = a1

    def box(e):
        th = np.deg2rad(e["th"])
        px, py = cx + (R + GAP) * np.cos(th), cy - (R + GAP) * np.sin(th)
        return px + np.cos(th) * e["tw"] / 2, py - np.sin(th) * e["hh"] / 2

    def overlap(e, f):
        (ex, ey), (fx, fy) = box(e), box(f)
        return (abs(ex - fx) < (e["tw"] + f["tw"]) / 2 + PAD) and (abs(ey - fy) < (e["hh"] + f["hh"]) / 2 + PAD)

    for _ in range(3000):
        moved = False
        for i in range(len(L)):
            e, f = L[i], L[(i + 1) % len(L)]
            if overlap(e, f):
                e["th"] += 0.4; f["th"] -= 0.4; moved = True
        if not moved:
            break
    assert not moved, "pie labels still overlap"
    for e in L:
        bx, by = box(e)
        mid, th = np.deg2rad(e["m"]), np.deg2rad(e["th"])
        ex, ey = cx + (R + RAD) * np.cos(mid), cy - (R + RAD) * np.sin(mid)
        qx, qy = cx + (R + GAP - 0.9) * np.cos(th), cy - (R + GAP - 0.9) * np.sin(th)
        ax.plot([cx + R * np.cos(mid), ex, qx], [cy - R * np.sin(mid), ey, qy], color=BAR, lw=0.6,
                solid_capstyle="round", solid_joinstyle="round")
        ax.text(bx, by, e["txt"], ha="center", va="center", fontsize=SMALL, linespacing=1.0)
    ax.set_aspect("equal")


SHORT_CTX = {"Cancer versus healthy controls": "Cancer vs healthy controls",
             "Cancer versus benign lung disease": "Cancer vs benign lung disease",
             "Tumor versus paired adjacent-normal tissue": "Tumor vs paired normal tissue",
             "Pre versus post treatment": "Pre vs post treatment"}
STARS = {("Cancer versus healthy controls", "Gut"), ("Cancer versus healthy controls", "Lower respiratory tract"),
         ("Cancer versus healthy controls", "Oral"), ("Cancer versus benign lung disease", "Lower respiratory tract"),
         ("Tumor versus paired adjacent-normal tissue", "Lung tissue"),
         ("Tumor versus paired adjacent-normal tissue", "Lower respiratory tract")}
SITE_HEAD = {"Gut": "Gut", "Lower respiratory tract": "Lower\nresp. tract", "Lung tissue": "Lung\ntissue",
             "Oral": "Oral", "Other sites": "Other (blood,\npleural, other)"}


def mix(c, t):
    r = np.array(to_rgb(c))
    return tuple(1 - t * (1 - r))


def grid(pg, x, y, w, h):
    p = grid_data()
    sites = [c for c in p.columns if c not in ("comparison", ALL)]
    ax = pg.canvas(x, y + HEAD_H, w, h - HEAD_H)
    lab_w, tw, head_h, leg_h = 33.0, 17.0, 8.6, 7.0
    rh = (h - HEAD_H - head_h - leg_h) / len(p)
    cw = (w - lab_w - tw - 2) / len(sites)
    vmax = int(p[sites].to_numpy().max())
    for j, c in enumerate(sites):
        k = SITE_BY_LABEL.get(c, "other")
        xc = lab_w + cw * (j + 0.5)
        draw_icon(ax, k, xc, 1.6, 2.6, SITE_COLOR[k], lw=2.6 * 0.075 / PT_MM, flip=True)
        ax.text(xc, 3.6, SITE_HEAD[c], ha="center", va="top", fontsize=SMALL, linespacing=0.95)
    for i, r in enumerate(p.itertuples(index=False)):
        ax.text(0, head_h + rh * (i + 0.5), SHORT_CTX.get(r.comparison, r.comparison), va="center", fontsize=6.0)
    gap = 0.35
    for i, r in enumerate(p.itertuples(index=False)):
        for j, c in enumerate(sites):
            v = int(r[j + 1])
            k = SITE_BY_LABEL.get(c, "other")
            x0, y0 = lab_w + cw * j + gap, head_h + rh * i + gap
            star = (r.comparison, c) in STARS
            fc = mix(SITE_COLOR[k], 0.15 + 0.85 * (v / vmax) ** 0.7) if v else "#F6F6F6"
            ax.add_patch(Rectangle((x0, y0), cw - 2 * gap, rh - 2 * gap, fc=fc,
                                   ec=BLACK if star else "none", lw=0.7 if star else 0))
            if v:
                hexc = "#%02x%02x%02x" % tuple(int(255 * q) for q in fc)
                ax.text(x0 + cw / 2 - gap, y0 + rh / 2 - gap, str(v), ha="center", va="center", fontsize=SMALL,
                        fontweight="bold", color=text_on(hexc))
    x0, tot = w - tw, p[ALL].to_numpy()
    k = (tw - 5.0) / tot.max()
    ax.text(x0, 2.2, "All sites\n(papers)", ha="left", va="top", fontsize=SMALL, linespacing=0.95)
    for i, v in enumerate(tot):
        yc = head_h + rh * (i + 0.5)
        ax.add_patch(Rectangle((x0, yc - rh * 0.33), v * k, rh * 0.66, fc="#9E9E9E", ec="none"))
        ax.text(x0 + v * k + 0.7, yc, str(v), va="center", fontsize=SMALL, fontweight="bold")
    ax.plot([x0, x0], [head_h, head_h + rh * len(tot)], color=BLACK, lw=LW)
    yb = head_h + len(p) * rh + 2.6
    note = "(darker = more papers, in each site's colour)"
    sw, g1, g2, sep = 3.0, 0.6, 1.8, 4.0
    scale = [("Papers", None)] + [(str(v), v) for v in (1, 5, 10, 25)] + [(note, None)]
    w_scale = sum((sw + g1 if v else 0) + text_w(t, SMALL) + g2 for t, v in scale) - g2
    w_fig3 = sw + g1 + text_w("Figure 3 panel", SMALL)
    rows = [[*scale, None, ("Figure 3 panel", "fig3")]] if w_scale + sep + w_fig3 <= w else \
        [scale, [("Figure 3 panel", "fig3")]]
    for kk, row in enumerate(rows):
        width = sum(sep if it is None else (sw + g1 if it[1] else 0) + text_w(it[0], SMALL) + g2 for it in row) - g2
        xx, yy = (w - width) / 2, yb + kk * 2.6
        for it in row:
            if it is None:
                xx += sep - g2
                continue
            t, v = it
            if v == "fig3":
                ax.add_patch(Rectangle((xx, yy), sw, 1.6, fc="white", ec=BLACK, lw=0.7))
                xx += sw + g1
            elif v:
                ax.add_patch(Rectangle((xx, yy), sw, 1.6, fc=mix(GREEN, 0.15 + 0.85 * (v / vmax) ** 0.7), ec="none"))
                xx += sw + g1
            ax.text(xx, yy + 0.8, t, va="center", fontsize=SMALL)
            xx += text_w(t, SMALL) + g2


PANELS = [
    ("Papers per year and sequencing method", years, 0, 0, 86, 52, {}),
    ("Country where the cohort was sampled", country_map, 90, 0, 93, 52, {}),
    ("Journal impact factor", impact, 0, 56, 64, 62, {}),
    ("Body sites", body_sites, 68, 56, 52, 62, dict(d=28)),
    ("Study designs", design, 124, 56, 59, 58, dict(step=25)),
    ("Lung cancer ICD-11 codes", icd11, 0, 122, 62, 73, {}),
    ("Papers per comparison and body site", grid, 66, 122, 117, 73, {}),
]


def main():
    m = D.cohort()
    pg = Page(h=196)
    for L, (title, fn, x, y, w, h, kw) in zip("abcdefg", PANELS):
        panel_head(pg, x, y, L, title)
        if fn is grid:
            fn(pg, x, y, w, h, **kw)
        else:
            fn(pg, x, y, w, h, m, **kw)
    pg.save("Figure1")


if __name__ == "__main__":
    main()
