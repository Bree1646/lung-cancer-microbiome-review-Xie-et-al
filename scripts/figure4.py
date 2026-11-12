"""Figure 4: immunotherapy."""
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import data as D
from plot import (Page, PAGE_W, TXT, SMALL, TINY, HEAD, BLACK, LW, LIGHTGRID, UP, DOWN, LIGHT_UP, LIGHT_DOWN,
                  LOW_AGREE, swatch, tree_rows, place_rows, draw_tree_axis, scale_bar, label_width, leaf_info)

RESP_DEFS = ["Objective response", "Durable/clinical benefit", "Major pathological response",
             "Progression-free survival", "Overall survival"]
T_RNR = "Responders vs non-responders"
TYPES = [T_RNR, "Hyperprogression", "Toxicity (irAE)", "On-treatment change", "Concomitant medication",
         "Microbiome- or biomarker-stratified"]
REGIMENS = ["ICI monotherapy cohort", "ICI combined therapy cohort", "Mixed cohort", "Not reported"]
SITE_LABEL = {"gut": "Gut", "lower-respiratory-tract": "Lower respiratory tract", "lung-tissue": "Lung tissue",
              "other": "Other (metastatic lung tumors)"}
TIMING = {"single-timepoint": "Single timepoint", "pre-post-paired": "Paired pre/post",
          "single-timepoint, pre-post-paired": "Single timepoint + paired subset",
          "longitudinal-multi-timepoint": "Longitudinal, multiple timepoints"}
MIN_PAPERS = 2


def ici_votes():
    f = D.findings()
    sel = f[f.ici_comparison_type.notna()]
    extra = ["ici_comparison_type", "ici_responder_definition"]
    v = pd.concat([D.build_votes(g.drop(columns=extra)).assign(ctype=t)
                   for t, g in sel.groupby("ici_comparison_type", sort=False)], ignore_index=True)
    K = ["key", "taxon", "taxonomic_rank", "body_site_resolved", "comparison_context", "direction_to", "ctype"]
    a = (sel.rename(columns={"body_site_finding": "body_site_resolved", "ici_comparison_type": "ctype"})
         .groupby(K, dropna=False)
         .agg(clinical_valence=("clinical_valence", lambda s: "|".join(sorted(set(s)))),
              endpoints=("ici_responder_definition",
                         lambda s: sorted({e for x in s.dropna() for e in x.split("; ")}, key=RESP_DEFS.index)))
         .reset_index())
    n = len(v)
    v = v.merge(a, on=K, how="left")
    assert len(v) == n and not v.clinical_valence.str.contains("|", regex=False).any()
    v["tested"] = v.evidentiary_tier.astype(str).str.lower().str.contains("tested")
    v["regimen"] = v.key.map(D.papers().set_index("key").ici_regimen)
    v["site"] = v.body_site_resolved.str.split(r"\s*,\s*")
    v = v.explode("site")
    d, _ = D.attach_tree(v)
    d["row_rank"] = np.where(d.is_phylum_row, "phylum", d.rank_disp)
    d["node"] = d.tree_node_id.astype(int)
    d["leaf_label"] = d.tree_leaf_label
    return v, d


def landscape_counts(V):
    p = D.papers().set_index("key")
    land = []
    for k, g in V.groupby("key"):
        land.append({"regimen": g.regimen.iloc[0], "sample_timing": TIMING[p.loc[k, "sample_timing"]],
                     "body_sites": "; ".join(SITE_LABEL[s] for s in sorted(set(g.site))),
                     "comparison_types": "; ".join(sorted({t for ts in g.ctype for t in ts.split("; ") if t in TYPES},
                                                          key=TYPES.index)),
                     "responder_definitions": "; ".join(sorted({e for es in g.endpoints for e in es},
                                                               key=RESP_DEFS.index))})
    land = pd.DataFrame(land)
    multi = lambda col, order=None: (lambda s: [(l, int(s[l])) for l in
                                                (order or s.sort_values(ascending=False, kind="stable").index)
                                                if l in s])(land[col].str.split("; ").explode().value_counts())
    return [("Regimen", multi("regimen", REGIMENS)), ("Body site", multi("body_sites")),
            ("Sample timing", multi("sample_timing")), ("Comparison type", multi("comparison_types", TYPES)),
            ("Responder definition", multi("responder_definitions", RESP_DEFS))]


def responder_summary(d):
    x = d[d.ctype.eq(T_RNR) & d.direction_vote.isin(["up", "down"]) & d.clinical_valence.eq("unfavorable")
          & d.site.eq("gut")].copy()
    x["vote"] = x.direction_vote
    rec = []
    for (key, r), g in x.groupby(["key", "row_id"]):
        dirs = set(g.vote)
        rec.append({"key": key, "row_id": r, "vote": "mixed" if len(dirs) > 1 else next(iter(dirs)),
                    "tested": bool(g.tested.any()),
                    "endpoints": "; ".join(sorted({e for es in g.endpoints for e in es}, key=RESP_DEFS.index)),
                    "regimen": g.regimen.iloc[0]})
    pv = pd.DataFrame(rec)
    meta = x.groupby("row_id").agg(row_rank=("row_rank", "first"), phylum=("phylum", "first"),
                                   gg2_axis_order=("gg2_axis_order", "min"), leaf_label=("leaf_label", "first"))
    rows = []
    for r, g in pv.groupby("row_id"):
        up, dn, mx = g[g.vote == "up"], g[g.vote == "down"], g[g.vote == "mixed"]
        ep = {e for s in g.endpoints for e in s.split("; ") if e}
        rows.append({"row_id": r, "papers_total": len(g), "papers_up": len(up), "papers_down": len(dn),
                     "papers_mixed": len(mx), "tested_up": int(up.tested.sum()), "descriptive_up": int((~up.tested).sum()),
                     "tested_down": int(dn.tested.sum()), "descriptive_down": int((~dn.tested).sum()),
                     "agreement": max(len(up), len(dn)) / len(g), "endpoint_families": len(ep),
                     "regimen_groups": g.regimen.nunique()})
    sm = pd.DataFrame(rows).join(meta, on="row_id")
    sm["gg2_name"] = np.where(sm.row_id.str.startswith("P|"), "p__" + sm.phylum, sm.row_id)
    s = sm[sm.papers_total >= MIN_PAPERS].sort_values("gg2_axis_order").copy()
    lab = D.figure_labels(list(s.loc[~s.row_id.str.startswith("P|"), "leaf_label"]))
    s["label"] = [p if r.startswith("P|") else lab[l] for r, p, l in zip(s.row_id, s.phylum, s.leaf_label)]
    return s.reset_index(drop=True), x.key.nunique()


LAND_H = 46.0
RH, PGAP, BARF = 2.6, 0.9, 0.62
MIXED = "#777777"
RAMPS = {
    "Regimen": ["#3B2C7A", "#5A4BA0", "#7F70C0", "#A497D6"],
    "Body site": ["#00573F", "#1F7A5C", "#46A07F", "#78C0A3"],
    "Sample timing": ["#7A4200", "#A25F0E", "#C98128", "#E2A458"],
    "Comparison type": ["#5E1A3D", "#7A2852", "#963A68", "#B0507F", "#C76C97", "#D98DAF"],
    "Responder definition": ["#0B4F6C", "#1E6B8E", "#3888AB", "#5BA5C4", "#83C0D9"],
}
SHORT = {"Other (metastatic lung tumors)": "Other (metastatic tumors)",
         "Single timepoint + paired subset": "Single + paired subset",
         "Longitudinal, multiple timepoints": "Longitudinal (≥3 timepoints)",
         "Microbiome- or biomarker-stratified": "Microbiome/biomarker-stratified"}


def landscape(pg, y, panels, n_studies, n_response):
    d = {p: [(SHORT.get(k, k), n, c) for (k, n), c in zip(items, RAMPS[p])] for p, items in panels}
    ax = pg.canvas(0, y, PAGE_W, LAND_H)
    mm_per = 18.0 / 25
    row, bar = 3.0, 2.0
    cells = [("Regimen", 4.2, 0, 32), ("Body site", 64, 0, 33), ("Sample timing", 124, 0, 36),
             ("Comparison type", 4.2, 22, 40), ("Responder definition", 64, 22, 33)]
    for p, x, yy, lw in cells:
        ax.text(x, yy + 0.5, p, fontsize=TXT, fontweight="bold", va="top")
        y0, x0 = yy + 4.6, x + lw
        for i, (lab, n, c) in enumerate(d[p]):
            yc = y0 + row * (i + 0.5)
            ax.text(x0 - 1.0, yc, lab, ha="right", va="center", fontsize=SMALL)
            ax.add_patch(Rectangle((x0, yc - bar / 2), 18.0, bar, fc="#F4F4F4", ec="none"))
            ax.add_patch(Rectangle((x0, yc - bar / 2), max(n * mm_per, 0.3), bar, fc=c, ec="none"))
            ax.text(x0 + 18.8, yc, f"{n}", ha="left", va="center", fontsize=SMALL, fontweight="bold")
        ax.plot([x0, x0], [y0, y0 + row * len(d[p])], color=BLACK, lw=LW)
    ax.text(124, 22.5, f"{n_studies} immunotherapy studies", fontsize=TXT, fontweight="bold", va="top")
    ax.text(124, 27.1, "Bars share one scale (track = 25 papers).\nA study can contribute several comparison\n"
            f"types; responder definition counts the {n_response}\nresponse studies only.", fontsize=SMALL,
            va="top", linespacing=1.35, color="#444444")


def responders(pg, y0, h_avail, s, n_gut):
    s = s.copy()
    s["row_id"] = np.where(s.row_rank.eq("phylum"), "P|" + s.phylum, s.gg2_name)
    nodes, _ = leaf_info(s.gg2_name, strict=False)
    s["node"] = nodes
    for i in s.index[s.node.isna()]:
        assert s.row_rank[i] == "phylum", s.label[i]
        s.loc[i, "node"] = D.tree().mrca(s.node[(s.phylum == s.phylum[i]) & s.node.notna()].astype(int))
    s["node"] = s.node.astype(int)
    tr, rows = tree_rows(s[["row_id", "label", "row_rank", "phylum", "node", "gg2_axis_order"]])
    d = s.set_index("row_id")
    ys, grid_h = place_rows(rows, RH, PGAP)

    TREE_X, TREE_W = 0.0, 15.0
    LAB_X = TREE_X + TREE_W + 1.2
    RANK_X = LAB_X + label_width(rows) + 7.5
    maxbar = int(d[["papers_up", "papers_down"]].to_numpy().max()) + 1
    upm = 2.3
    XC = RANK_X + 2.0 + maxbar * upm
    XR = XC + maxbar * upm
    cols = [("Total", "papers_total"), ("Agree", "agreement"), ("Responder\ndefinitions", "endpoint_families"),
            ("Regimen\ngroups", "regimen_groups")]
    col_x = [XR + 4.0 + 8.5 * j for j in range(len(cols))]
    RULE_TO = col_x[-1] + 4.5
    HD = 9.0
    ax = pg.canvas(0, y0, PAGE_W, h_avail)
    for (hd, _), cx in zip(cols, col_x):
        ax.text(cx, HD - 1.2, hd, fontsize=TINY, fontweight="bold", ha="center", va="bottom", linespacing=0.95)
    ax.text(XC, HD - 5.5, f"Gut, responders vs non-responders ({n_gut} papers)", fontsize=SMALL, fontweight="bold",
            ha="center", va="bottom")
    unit = draw_tree_axis(ax, tr, rows, ys, TREE_X, TREE_W, LAB_X, RANK_X, RULE_TO, RH, y0=HD)
    top, bot = HD, HD + grid_h
    for t in range(-maxbar + 1, maxbar):
        if t and t % 2 == 0:
            ax.plot([XC + t * upm] * 2, [top, bot], color=LIGHTGRID, lw=0.35, zorder=0)
    bh = RH * BARF
    for i, r in enumerate(rows):
        y = HD + ys[i]
        if r["rid"] not in d.index:
            continue
        e = d.loc[r["rid"]]
        if e.agreement < 0.70:
            ax.add_patch(Rectangle((XC - maxbar * upm, y - RH / 2), 2 * maxbar * upm, RH, fc=LOW_AGREE, ec="none",
                                   zorder=0.5))
        for start, length, fc, ec in [(0, e.tested_up, UP, "none"), (e.tested_up, e.descriptive_up, LIGHT_UP, UP),
                                      (0, -e.tested_down, DOWN, "none"),
                                      (-e.tested_down, -e.descriptive_down, LIGHT_DOWN, DOWN)]:
            if length:
                ax.add_patch(Rectangle((XC + start * upm, y - bh / 2), length * upm, bh, fc=fc, ec=ec, lw=0.3,
                                       zorder=3))
        if e.papers_up:
            ax.text(XC + e.papers_up * upm + 0.6, y, str(int(e.papers_up)), fontsize=TINY, fontweight="bold",
                    va="center")
        if e.papers_down:
            ax.text(XC - e.papers_down * upm - 0.6, y, str(int(e.papers_down)), fontsize=TINY, fontweight="bold",
                    va="center", ha="right")
        if e.papers_mixed:
            ax.plot([XC], [y], marker="D", ms=1.8, mfc=MIXED, mec="white", mew=0.3, zorder=6)
        for (hd, col), cx in zip(cols, col_x):
            v = e[col]
            ax.text(cx, y, f"{v:.2f}" if col == "agreement" else str(int(v)), fontsize=TINY,
                    fontweight="bold" if col == "papers_total" else "normal", ha="center", va="center")
    ax.plot([XC] * 2, [top, bot], color=BLACK, lw=0.6, zorder=4)
    ax.plot([XC - maxbar * upm, XR], [bot] * 2, color=BLACK, lw=0.5)
    for t in range(-maxbar + 1, maxbar):
        if t % 2 == 0:
            ax.plot([XC + t * upm] * 2, [bot, bot + 0.8], color=BLACK, lw=0.4)
            ax.text(XC + t * upm, bot + 1.1, str(abs(t)), fontsize=TINY, ha="center", va="top")
    ax.text(XC - 0.8, bot + 4.0, "← higher in responders", fontsize=SMALL, ha="right", va="top")
    ax.text(XC + 0.8, bot + 4.0, "higher in non-responders →", fontsize=SMALL, ha="left", va="top")
    scale_bar(ax, TREE_X, bot + 4.6, unit)
    lx, ly = RULE_TO + 4.0, HD + 2.0
    for fc, ec, lab in [(UP, "none", "Higher in non-responders, tested"),
                        (LIGHT_UP, UP, "Higher in non-responders, descriptive"),
                        (DOWN, "none", "Higher in responders, tested"),
                        (LIGHT_DOWN, DOWN, "Higher in responders, descriptive"),
                        (LOW_AGREE, "#BBBBBB", "Agreement below 0.70")]:
        swatch(ax, lx, ly, fc, lab, fs=SMALL, ec=ec, lw=0.3)
        ly += 3.0
    ax.text(lx, ly + 1.5, "Total = contributing papers;\nAgree = majority-direction share;\n"
            "the last two columns count distinct\nresponder definitions and regimen\ngroups behind each taxon.",
            fontsize=TINY, va="top", linespacing=1.25)


def main():
    V, d = ici_votes()
    panels = landscape_counts(V)
    s, n_gut = responder_summary(d)
    n_votes = V.drop_duplicates(["key", "taxon", "taxonomic_rank", "body_site_resolved", "comparison_context",
                                 "direction_to"]).shape[0]
    n_resp = V[V.endpoints.str.len() > 0].key.nunique()
    nph = s.phylum.nunique()
    nrow = len(s) - int(s.row_rank.eq("phylum").sum()) + nph
    grid_h = nrow * RH + (nph - 1) * PGAP
    yb = LAND_H + 1.0
    pg = Page(h=yb + 2 + 9.0 + grid_h + 10.5)
    pg.letter(0, 0, "a")
    pg.text(4.2, 0.5, "", fontsize=HEAD, fontweight="bold")
    landscape(pg, 0, panels, V.key.nunique(), n_resp)
    pg.letter(0, yb, "b")
    pg.text(4.2, yb + 0.5, "Immunotherapy outcome: responders versus non-responders", fontsize=HEAD, fontweight="bold")
    responders(pg, yb + 2, pg.h - yb - 2, s, n_gut)
    pg.save("Figure4")
    print(f"  Figure 4: {V.key.nunique()} studies, {n_votes} votes; panel b {n_gut} studies, {len(s)} taxa")


if __name__ == "__main__":
    main()
