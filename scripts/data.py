"""Loads the four data workbooks and derives the votes and the Greengenes2 placement."""
from functools import lru_cache
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

SHORT = {"assay": "assay", "smoking_status": "smoking_status", "body_site": "body_site",
         "sample_timing": "sample_timing", "study design": "study_design",
         "intervention_category": "intervention_category", "statistical_model_std": "statistical_model_std"}


@lru_cache(None)
def papers():
    p = pd.read_excel(ROOT / "papers.xlsx")
    return p.rename(columns={c: SHORT[c.split(" (")[0]] for c in p.columns if c.split(" (")[0] in SHORT})


@lru_cache(None)
def findings():
    return pd.read_excel(ROOT / "findings.xlsx", sheet_name="findings")


@lru_cache(None)
def fb_ratio():
    return pd.read_excel(ROOT / "findings.xlsx", sheet_name="FB_ratio")


@lru_cache(None)
def diversity():
    return pd.read_excel(ROOT / "diversity_findings.xlsx")


@lru_cache(None)
def gg2_taxa():
    return pd.read_excel(ROOT / "gg2_placement.xlsx", sheet_name="taxa")


@lru_cache(None)
def gg2_nodes():
    return pd.read_excel(ROOT / "gg2_placement.xlsx", sheet_name="tree_nodes")


GK = ["key", "taxon", "taxonomic_rank", "body_site_finding", "comparison_context", "finding_type", "direction_to"]
# confirmed shared cohorts: a finding reported by both papers counts once (earlier paper kept)
PAIRS = [("rayyan-527939734", "rayyan-527951274"), ("rayyan-527948782", "rayyan-527947475"),
         ("rayyan-527947892", "rayyan-527945618"), ("rayyan-527948782", "rayyan-527947021"),
         ("rayyan-527946195", "rayyan-527947076")]
DUP_KEY = ["taxon", "taxonomic_rank", "body_site_resolved", "comparison_context", "finding_type", "direction_vote"]


def _join(s):
    u = [str(x) for x in dict.fromkeys(s.dropna()) if str(x) != ""]
    return "; ".join(u) if u else None


def _vote(dirs):
    d = list(dict.fromkeys(dirs))
    if len(d) == 1:
        return d[0]
    dd = [x for x in d if x != "non-directional"]
    return dd[0] if len(dd) == 1 else "mixed"


def build_votes(f):
    f = f.copy()
    for c in GK:
        f[c] = f[c].fillna("")
    f["_dt"] = f["direction_to"].replace("", None)
    corr = f.finding_type.isin(["correlation", "co-occurrence"])
    f.loc[corr, "_dt"] = None
    f["_sign"] = ""
    f.loc[corr, "_sign"] = f.loc[corr, "direction"].fillna("")
    parts = []
    for _, g in f.groupby(GK[:-1] + ["_sign"], sort=False):
        g = g.copy()
        if g.finding_type.iloc[0] in ("correlation", "co-occurrence"):
            g["direction_to"] = _join(g["direction_to"]) or ""
        else:
            u = list(dict.fromkeys(g._dt.dropna()))
            g["direction_to"] = (u[0] if u else "") if len(u) <= 1 else g._dt.fillna(u[0])
        parts.append(g)
    f = pd.concat(parts)
    out = []
    for k, g in f.groupby(GK + ["_sign"], sort=False):
        r = dict(zip(GK, k))
        r["direction_vote"] = _vote(g.direction.fillna("non-directional"))
        r["n_source_rows"] = len(g)
        r["is_cross_site_comparison"] = g.is_cross_site_comparison.any()
        s = sorted(set(map(str, g.evidentiary_tier.dropna())))
        r["evidentiary_tier"] = ";".join(s) if s else None
        r["is_classifier_panel"] = g.is_classifier_panel.any()
        r["correlate_with"] = _join(g.correlate_with)
        out.append(r)
    v = pd.DataFrame(out).rename(columns={"body_site_finding": "body_site_resolved"})
    for c in GK:
        c2 = "body_site_resolved" if c == "body_site_finding" else c
        v[c2] = v[c2].replace("", None)
    return v


@lru_cache(None)
def votes():
    v = build_votes(findings())
    drop = set()
    for a, b in PAIRS:
        m = v[v.key == a].merge(v[v.key == b], on=DUP_KEY)
        drop |= {(b, *r) for r in m[DUP_KEY].itertuples(index=False)}
    sig = list(zip(v.key, *[v[c] for c in DUP_KEY]))
    return v[[s not in drop for s in sig]].reset_index(drop=True)


def valence_of(v, f):
    f = f.rename(columns={"body_site_finding": "body_site_resolved"})
    K = ["key", "taxon", "taxonomic_rank", "body_site_resolved", "comparison_context", "direction_to"]
    val = f.groupby(K, dropna=False).clinical_valence.agg(lambda s: "|".join(sorted(set(s)))).reset_index()
    n = len(v)
    v = v.merge(val, on=K, how="left")
    assert len(v) == n and v.clinical_valence.notna().all() and not v.clinical_valence.str.contains("|", regex=False).any()
    return v


CLASSIC = {
    "Bacillota": "Firmicutes", "Pseudomonadota": "Proteobacteria", "Bacteroidota": "Bacteroidetes",
    "Actinomycetota": "Actinobacteria", "Verrucomicrobiota": "Verrucomicrobia",
    "Fusobacteriota": "Fusobacteria", "Synergistota": "Synergistetes", "Acidobacteriota": "Acidobacteria",
    "Cyanobacteriota": "Cyanobacteria", "Methanobacteriota": "Euryarchaeota",
    "Spirochaetota": "Spirochaetes", "Chloroflexota": "Chloroflexi", "Deinococcota": "Deinococcus-Thermus",
    "Planctomycetota": "Planctomycetes", "Chlamydiota": "Chlamydiae",
    "Gemmatimonadota": "Gemmatimonadetes", "Nitrospirota": "Nitrospirae",
    "Armatimonadota": "Armatimonadetes",
}
BLOCK_LABEL = {"Deltaproteobacteria": "Deltaproteobacteria (several GG2 phyla)",
               "Euryarchaeota": "Euryarchaeota (several GG2 phyla)"}
PHY_PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#882255", "#999933",
           "#AA4499", "#44AA99", "#661100", "#117733", "#B8860B", "#332288", "#6699CC", "#888888",
           "#A6761D", "#7F3C8D", "#1B9E77", "#E7298A"]


def collapse(name):
    return re.sub(r"(_[A-Z]{1,2})?(_\d+)?$", "", str(name))


def phylum_label(p):
    if p in BLOCK_LABEL:
        return BLOCK_LABEL[p]
    return f"{p} ({CLASSIC[p]})" if p in CLASSIC else p


def figure_labels(names):
    clean = lambda n: re.sub(r"_(?:[A-Z]{1,2}(?:_\d+)?|\d+)(?=\s|$)", "", n)
    letter = lambda n: re.sub(r"_\d+(?=\s|$)", "", n)
    out = {n: clean(n) for n in names}
    for f in (letter, lambda n: n):
        seen = {}
        for n, l in out.items():
            seen.setdefault(l, []).append(n)
        for grp in seen.values():
            if len(grp) > 1:
                for n in grp:
                    out[n] = f(n)
    return out


def attach_tree(d):
    t = gg2_taxa().assign(is_red=lambda x: x.problem.eq("Placed only at its domain"))
    keep = ["taxon_name_corpus", "FINAL_rank", "is_red", "gg2_node", "gg2_rank", "gg2_phylum", "gg2_domain",
            "tree_leaf", "tree_leaf_label", "problem", "tree_node_id", "root_to_node_length", "gg2_axis_order",
            "gg2_genus"]
    n = len(d)
    d = d.merge(t[keep], left_on=["taxon", "taxonomic_rank"], right_on=["taxon_name_corpus", "FINAL_rank"], how="left")
    assert len(d) == n and d.tree_leaf.notna().all(), "taxa missing from gg2_placement.xlsx"
    genome = d.problem.eq("Species found via a GG2 reference genome")
    d.loc[genome, "tree_leaf_label"] = d.loc[genome, "taxon"]
    red = d[d.is_red]
    d = d[~d.is_red].copy()
    leaf = d.tree_leaf.astype(str)
    is_block = d.gg2_rank.eq("block")
    ph = d.gg2_phylum.map(lambda p: collapse(p) if pd.notna(p) else None)
    ph = np.where(is_block & leaf.str.startswith("BLOCK p__Bacillota"), "Bacillota", ph)
    ph = np.where(is_block & ~leaf.str.startswith("BLOCK p__Bacillota"), "Deltaproteobacteria", ph)
    ph = np.where(pd.isna(ph) & leaf.eq("x__Euryarchaeota"), "Euryarchaeota", ph)
    d["phylum"] = ph
    assert d.phylum.notna().all()
    d["is_phylum_row"] = d.gg2_rank.isin(["phylum", "block"]) | leaf.eq("x__Euryarchaeota")
    d["row_id"] = np.where(d.is_phylum_row, "P|" + d.phylum, leaf)
    d["rank_disp"] = np.where(leaf.str.startswith("x__"), d.taxonomic_rank, d.gg2_rank)
    return d, red


class GG2Tree:

    def __init__(self):
        n = gg2_nodes()
        self.parent = {int(r.node_id): (None if pd.isna(r.parent_node_id) else int(r.parent_node_id))
                       for r in n.itertuples()}
        self.length = dict(zip(n.node_id.astype(int), n.root_to_node_length.astype(float)))

    def path(self, node):
        out = []
        while node is not None:
            out.append(node)
            node = self.parent[node]
        return out[::-1]

    def mrca(self, nodes):
        paths = [self.path(int(x)) for x in nodes]
        k = 0
        while all(len(p) > k for p in paths) and len({p[k] for p in paths}) == 1:
            k += 1
        return paths[0][k - 1]

    def layout(self, leaves):
        root = self.mrca([nd for _, nd, _ in leaves])
        ch, at = {}, {}
        for lid, nd, sk in leaves:
            at.setdefault(nd, []).append((sk, lid))
            p = self.path(nd)
            p = p[p.index(root):]
            for a, b in zip(p, p[1:]):
                ch.setdefault(a, set()).add(b)
        size = {}

        def count(nd):
            size[nd] = len(at.get(nd, [])) + sum(count(c) for c in ch.get(nd, ()))
            return size[nd]
        count(root)
        order, kids = [], {}

        def walk(nd):
            kids[nd] = ([("L", lid) for _, lid in sorted(at.get(nd, []))]
                        + sorted(ch.get(nd, ()), key=lambda c: (size[c], c)))
            for c in kids[nd]:
                if isinstance(c, tuple):
                    order.append(c[1])
                else:
                    walk(c)
        walk(root)
        return root, order, kids


@lru_cache(None)
def tree():
    return GG2Tree()


def tree_phyla(root, kids, leaf_phylum):
    out = {}

    def f(nd):
        s = set()
        for c in kids[nd]:
            s |= {leaf_phylum[c[1]]} if isinstance(c, tuple) else f(c)
        out[nd] = s
        return s
    f(root)
    return out


def draw_tree_lines(tr, root, kids, tip_pos, colour_of, to_xy, arc_pts):
    segs, cols, leaf_depth = [], [], {}
    base = tr.length[root]
    pos = {}

    def place(nd):
        ps = [tip_pos[c[1]] if isinstance(c, tuple) else place(c) for c in kids[nd]]
        pos[nd] = (min(ps) + max(ps)) / 2
        return pos[nd]
    place(root)

    def draw(nd):
        dn = tr.length[nd] - base
        cps = []
        for c in kids[nd]:
            if isinstance(c, tuple):
                leaf_depth[c[1]] = dn
                cps.append(tip_pos[c[1]])
            else:
                dc = tr.length[c] - base
                segs.append([to_xy(dn, pos[c]), to_xy(dc, pos[c])])
                cols.append(colour_of(c))
                cps.append(pos[c])
                draw(c)
        if len(cps) > 1:
            segs.append(arc_pts(dn, min(cps), max(cps)))
            cols.append(colour_of(nd))
    draw(root)
    return segs, cols, leaf_depth


@lru_cache(None)
def da_votes():
    v = votes()
    v = v[v.finding_type == "differential abundance"].copy()
    f = findings()
    v = valence_of(v, f[f.finding_type == "differential abundance"])
    d = v[v.direction_vote.isin(["up", "down", "mixed"]) & v.clinical_valence.isin(["favorable", "unfavorable"])].copy()
    d = d[~d.is_cross_site_comparison.astype(bool)].copy()
    d["direction"] = d.direction_vote.map({"up": ["up"], "down": ["down"], "mixed": ["up", "down"]})
    d = d.explode("direction")
    sign = d.direction.map({"up": 1, "down": -1}) * d.clinical_valence.map({"unfavorable": 1, "favorable": -1})
    d["effect"] = np.where(sign > 0, "enriched", "depleted")
    d, _ = attach_tree(d)
    d["ctx"] = d.comparison_context.str.split(r"\s*,\s*")
    d = d.explode("ctx")
    multi = d.comparison_context.str.contains(",")
    d = d[~(multi & d.ctx.eq("subtype comparison"))].copy()
    d["site"] = d.body_site_resolved.str.split(r"\s*,\s*")
    return d.explode("site")


MIN_PAPERS_VALENCE = 2


def valence_rows(d):
    tr = tree()
    g = d.groupby("row_id").agg(n_papers=("key", "nunique"), order=("gg2_axis_order", "min"),
                                label=("tree_leaf_label", "first"), rank=("rank_disp", "first"),
                                node=("tree_node_id", "first"),
                                phylum=("phylum", "first"), is_phylum_row=("is_phylum_row", "first")).reset_index()
    own_nodes = d.groupby("row_id").tree_node_id.agg(lambda s: sorted(set(int(x) for x in s)))
    g = g[g.n_papers >= MIN_PAPERS_VALENCE].copy()
    rows = []
    for p, sub in g.groupby("phylum"):
        nodes = [int(x) for x in sub[~sub.is_phylum_row].node]
        prow = sub[sub.is_phylum_row]
        if len(prow):
            nodes += own_nodes[prow.row_id.iloc[0]]
        rows.append({"row_id": "P|" + p, "kind": "phylum", "label": phylum_label(p), "rank": "phylum",
                     "phylum": p, "n_papers": int(prow.n_papers.iloc[0]) if len(prow) else 0,
                     "node": tr.mrca(nodes), "order": -1})
        for r in sub[~sub.is_phylum_row].itertuples():
            rows.append({"row_id": r.row_id, "kind": "taxon", "label": r.label, "rank": r.rank, "phylum": p,
                         "n_papers": int(r.n_papers), "node": int(r.node), "order": r.order})
    lab = figure_labels([r["label"] for r in rows if r["kind"] == "taxon"])
    for r in rows:
        if r["kind"] == "taxon":
            r["label"] = lab[r["label"]]
    root, order, kids = tr.layout([(i, r["node"], (r["order"], r["label"])) for i, r in enumerate(rows)])
    rows = [rows[i] for i in order]
    remap = {o: k for k, o in enumerate(order)}
    kids = {nd: [("L", remap[c[1]]) if isinstance(c, tuple) else c for c in cs] for nd, cs in kids.items()}
    runs = [k for k, r in enumerate(rows) if k == 0 or r["phylum"] != rows[k - 1]["phylum"]]
    assert len(runs) == len({r["phylum"] for r in rows}) and all(rows[k]["kind"] == "phylum" for k in runs)
    return rows, (root, kids)


@lru_cache(None)
def phylum_colours():
    rows, _ = valence_rows(da_votes())
    n = pd.Series([r["phylum"] for r in rows if r["kind"] == "taxon"]).value_counts()
    first = list(dict.fromkeys(r["phylum"] for r in rows))
    ordered = sorted(first, key=lambda p: (-n.get(p, 0), first.index(p)))
    return {p: PHY_PAL[i] for i, p in enumerate(ordered)}


def pcol(p):
    c = phylum_colours()
    if p not in c:
        free = [x for x in PHY_PAL if x not in c.values()]
        c[p] = free[0] if free else "#000000"
    return c[p]


SITE_ORDER = ["gut", "lower-respiratory-tract", "lung-tissue", "oral", "blood-plasma-serum",
              "pleural-fluid", "urine", "other"]
ISO3 = {"China": "CHN", "USA": "USA", "Japan": "JPN", "Italy": "ITA", "Hungary": "HUN", "Thailand": "THA",
        "Russia": "RUS", "South Korea": "KOR", "Canada": "CAN", "Poland": "POL", "United Kingdom": "GBR",
        "Turkey": "TUR", "India": "IND", "Portugal": "PRT", "Switzerland": "CHE", "France": "FRA",
        "Netherlands": "NLD", "Germany": "DEU", "Spain": "ESP", "Israel": "ISR", "Cyprus": "CYP",
        "Lithuania": "LTU", "Croatia": "HRV"}
DESIGNS = ["disease-vs-healthy", "disease-vs-benign", "within-patient-paired", "comorbidity-stratified",
           "therapy-responder-vs-nonresponder", "pre-post-treatment", "single-arm-descriptive",
           "disease-vs-disease", "severity-stratified", "disease-subtype-stratified", "in-vitro-validation"]
ASSAY_LABEL = {"16S-rRNA": "16S rRNA", "shotgun-metagenomic": "Shotgun metagenomic",
               "whole-genome-sequencing": "Whole-genome sequencing", "long-read-nanopore": "Long-read nanopore"}


def tokens(v, sep=r"[;,]"):
    return [t.strip() for t in re.split(sep, str(v)) if t.strip()]


def _design_parent(t):
    if t.startswith("comorbidity/exposure-stratified"):
        return "comorbidity-stratified"
    hit = [d for d in DESIGNS if t.startswith(d)]
    assert hit, f"unmapped study design: {t}"
    return hit[0]


@lru_cache(None)
def cohort():
    m = papers().copy()
    f = findings()
    site_set = f.groupby("key").body_site_finding.agg(lambda s: {t for x in s.dropna() for t in tokens(x)})
    assert set(m.key) == set(f.key) and set().union(*site_set) <= set(SITE_ORDER)
    m["sites"] = m.key.map(lambda k: [t for t in SITE_ORDER if t in site_set[k]])
    m["n_total"] = pd.to_numeric(m.sample_size_total)
    m["sampled_iso"] = m.country_sampled.map(lambda c: ISO3[c] if isinstance(c, str) and c else None)
    m["designs"] = m.study_design.map(lambda s: sorted({_design_parent(t) for t in tokens(s, ";")}))
    m["assays"] = m.assay.map(lambda s: [a for a in ASSAY_LABEL if a in tokens(s)])
    assert (m.assays.map(len) == m.assay.map(lambda s: len(tokens(s)))).all(), "unknown assay"
    m["assay_class"] = m.assays.map(lambda a: ASSAY_LABEL[a[0]] if len(a) == 1 else "More than one method")
    m["model"] = m.statistical_model_std.replace({"longitudinal-repeated-measures": "covariate-adjusted"})
    return m


DIV_SITES = ["gut", "lower-respiratory-tract", "lung-tissue", "oral", "blood-plasma-serum", "pleural-fluid"]


def _div_outcome(r):
    t, c, v = r.metric_type, r.result_category, r.clinical_valence
    if t == "beta":
        return {"significant-difference": "sig", "no-significant-difference": "ns", "visual-only": "visual"}.get(c)
    if c == "no-significant-difference":
        return "ns"
    if c == "significant-difference":
        return "undirected"
    if c == "mixed":
        return "inconsistent"
    if c in ("increased", "decreased"):
        if v == "unfavorable":
            return "higher" if c == "increased" else "lower"
        if v == "favorable":
            return "lower" if c == "increased" else "higher"
        return "no side"
    return None


def _div_collapse(s):
    s = set(s.dropna())
    if not s:
        return None
    if s & {"higher", "lower", "inconsistent", "no side", "undirected"}:
        if "inconsistent" in s or {"higher", "lower"} <= s:
            return "inconsistent"
        for o in ["higher", "lower", "no side", "undirected"]:
            if o in s:
                return o
    if "sig" in s:
        return "sig"
    if "ns" in s:
        return "ns"
    return "visual" if "visual" in s else None


@lru_cache(None)
def diversity_votes():
    d = diversity().copy()
    d["metric"] = d.metric_name
    d["outcome"] = d.apply(_div_outcome, axis=1)
    rows = d.assign(comparison=d.comparison_context.str.split(r",\s*")).explode("comparison")
    rows["site"] = rows.body_site_finding.str.split(r",\s*")
    rows.loc[rows.is_cross_site_comparison.astype(bool), "site"] = rows.loc[
        rows.is_cross_site_comparison.astype(bool), "body_site_finding"].map(lambda s: ["cross-site"])
    rows = rows.explode("site")
    keep = rows.site.isin(DIV_SITES) & ~rows.comparison.str.startswith("multi-group")
    return (rows[keep].groupby(["key", "site", "comparison", "metric_type", "metric"])
            .agg(outcome=("outcome", _div_collapse)).reset_index())
