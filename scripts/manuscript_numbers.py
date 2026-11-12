"""Numbers in the text and legends, and Supplementary Tables S4 and S5."""
import numpy as np
import pandas as pd
import data as D
import figure3
import figure4
import figureS2
import figureS4
import figureS5

OUT = []


def put(section, item, value):
    OUT.append({"section": section, "item": item, "value": value})


def pct(a, b, nd=0):
    return f"{a} of {b} ({100 * a / b:.{nd}f}%)"


def corpus():
    m, f = D.cohort(), D.findings()
    N = len(m)
    put("Abstract/Results", "included studies", N)
    put("Abstract/Results", "participants (sum of sample_size_total)", f"{int(m.n_total.sum()):,}")
    put("Abstract/Results", "taxon-level findings", f"{len(f):,}")
    for t, n in f.finding_type.value_counts().items():
        put("Results", f"findings: {t}", f"{n:,}")
    put("Results", "phylum-ratio (Firmicutes/Bacteroidetes) findings", len(D.fb_ratio()))
    put("Abstract/Results", "votes after deduplication", f"{len(D.votes()):,}")
    yc = m.year.astype(int).value_counts()
    put("Results", "studies in 2020 / 2024 / 2025", f"{yc.get(2020, 0)} / {yc.get(2024, 0)} / {yc.get(2025, 0)}")
    put("Results", "publication years", f"{m.year.min()}-{m.year.max()}")
    for k, n in m.assay_class.value_counts().items():
        put("Results", f"sequencing method: {k}", n)
    ifs = pd.to_numeric(m.Impact_factor, errors="coerce").dropna()
    put("Results/Figure 1c", "journal impact factor: studies with a value", len(ifs))
    put("Results", "journal impact factor: median (IQR)",
        f"{ifs.median():.1f} ({ifs.quantile(.25):.1f}-{ifs.quantile(.75):.1f})")
    cs = m.sampled_iso.dropna()
    put("Results/Figure 1b", "studies with a single sampling country", len(cs))
    put("Results", "sampled in China", pct(int((cs == "CHN").sum()), len(cs)))
    put("Results", "countries", cs.nunique())
    sl = m.explode("sites").sites.value_counts()
    for s in D.SITE_ORDER:
        if s in sl:
            put("Results/Figure 1d", f"body site: {s}", f"{sl[s]} ({100 * sl[s] / N:.0f}%)")
    put("Results", "studies sampling more than one site", int((m.sites.map(len) > 1).sum()))
    n = m.n_total
    put("Results", "participants per study: median (IQR; range)",
        f"{n.median():g} ({n.quantile(.25):.0f}-{n.quantile(.75):.0f}; {n.min():.0f}-{n.max():,.0f})")
    put("Results/Table S4", "studies with fewer than 100 participants", pct(int((n < 100).sum()), N))
    sp = m.explode("sites").groupby("sites").n_total.median()
    for s in D.SITE_ORDER:
        if s in sp:
            put("Results", f"median participants: {s}", f"{sp[s]:g}")
    big = m[n > 1000].sort_values("n_total", ascending=False)
    put("Results", "studies with more than 1,000 participants", ", ".join(f"{v:,.0f}" for v in big.n_total))
    put("Results", "largest study after the two pan-cancer cohorts", f"{n.sort_values().iloc[-3]:,.0f}")
    flags = m.statistical_quality_flag_tags.fillna("")
    for tag in ["Small-sample-size", "No-multiplicity-correction", "Confounder-imbalanced-unadjusted"]:
        put("Results/Table S4", f"flag {tag}", pct(int(flags.str.contains(tag).sum()), N))
    for k, v in m.intervention_category.value_counts().items():
        put("Results/Figure S3a", f"intervention: {k}", v)
    for k, v in m.sample_timing.value_counts().items():
        put("Results/Figure S3a", f"sample timing: {k}", v)
    dz = m.designs.explode().value_counts()
    put("Results/Figure 1e", "design: cancer vs healthy (disease-vs-healthy)", dz["disease-vs-healthy"])
    for k, v in m.model.value_counts().items():
        put("Results/Figure S3b", f"confounder handling: {k}", pct(int(v), N))
    put("Table S4", "smoking status not reported", pct(int((m.smoking_status == "no").sum()), N))
    put("Figure 1f", "studies with a 2C25 diagnosis / stage code",
        f"{int(m.ICD11_diagnosis.notna().sum())} / {int(m.ICD11_stage.notna().sum())}")
    put("Figure S3c", "points (study x site)", len(m.explode("sites")))
    keys, groups = set(), []
    for a, b in D.PAIRS:
        keys |= {a, b}
        hit = [g for g in groups if a in g or b in g]
        merged = set().union({a, b}, *hit)
        groups = [g for g in groups if g not in hit] + [merged]
    put("Results", "confirmed shared-cohort studies / clusters", f"{len(keys)} / {len(groups)}")


def replication():
    v = D.votes()
    d = v[(v.finding_type == "differential abundance") & v.direction_vote.isin(["up", "down", "mixed"])]
    d = d[~d.is_cross_site_comparison.astype(bool)]
    d, _ = D.attach_tree(d)
    d["ctx"] = d.comparison_context.astype(str).str.split(r"\s*,\s*")
    d = d.explode("ctx")
    d = d[~(d.comparison_context.astype(str).str.contains(",") & d.ctx.eq("subtype comparison"))]
    d["site"] = d.body_site_resolved.astype(str).str.split(r"\s*,\s*")
    d = d.explode("site")
    pv = (d.groupby(["ctx", "site", "row_id", "key"]).direction_vote
          .agg(lambda s: s.iloc[0] if s.nunique() == 1 and s.iloc[0] != "mixed" else "mixed").reset_index())
    c = pv.groupby(["ctx", "site", "row_id"]).direction_vote.agg(
        papers="size", up=lambda s: int((s == "up").sum()), down=lambda s: int((s == "down").sum()),
        mixed=lambda s: int((s == "mixed").sum())).reset_index()
    c["agreement"] = c[["up", "down"]].max(axis=1) / c.papers
    N = len(c)
    two, three = c.papers >= 2, c.papers >= 3
    cons = three & (c.agreement >= 0.70)
    both = two & (c.up > 0) & (c.down > 0)
    never = (c.groupby("row_id").papers.max() == 1)
    put("Results", "eligible votes / studies / GG2 taxa",
        f"{len(d.drop_duplicates(['key', 'taxon', 'taxonomic_rank', 'ctx', 'site', 'direction_to'])):,} / "
        f"{d.key.nunique()} / {c.row_id.nunique()}")
    put("Abstract/Results", "taxon x body site x comparison cells", f"{N:,}")
    put("Results", "cells reported by one study", pct(int((c.papers == 1).sum()), N, 1))
    put("Results", "cells reported by two or more studies", pct(int(two.sum()), N, 1))
    put("Abstract/Results", "cells (2+ studies) with opposite directions", pct(int(both.sum()), int(two.sum()), 1))
    put("Results", "cells reported by three or more studies", pct(int(three.sum()), N, 1))
    put("Abstract/Results", "consensus cells (>= 3 studies, agreement >= 0.70)", pct(int(cons.sum()), N, 1))
    put("Results", "taxa never reported by a second study in the same cell", pct(int(never.sum()), len(never), 1))
    cc = c[cons]
    put("Results", "consensus cells: comparison contexts / context x body-site pairs",
        f"{cc.ctx.nunique()} / {len(cc.groupby(['ctx', 'site']))}")
    for (ctx, site), k in cc.groupby(["ctx", "site"]).size().sort_values(ascending=False).head(2).items():
        put("Results", f"consensus cells: {ctx}, {site}", k)
    s5 = [("All taxon × site × design cells", N), ("Reported by exactly one study", int((c.papers == 1).sum())),
          ("Reported by two or more studies", int(two.sum())), (" of which contain both directions", int(both.sum())),
          ("Reported by three or more studies", int(three.sum())),
          (" with agreement ≥ 0.70 (consensus)", int(cons.sum())),
          (" with agreement < 0.70", int((three & (c.agreement < 0.70)).sum()))]
    pd.DataFrame([{"": k, "Cells": f"{n:,}", f"% of {N:,}": f"{100 * n / N:.1f}%".replace("100.0%", "100%")}
                  for k, n in s5]).to_csv(D.OUT / "Table_S5.csv", index=False)
    return d


def figures_2_3():
    d = D.da_votes()
    rows, _ = D.valence_rows(d)
    put("Figure 2", "rows with >= 2 studies (taxa incl. phylum-level rows) / phylum rows",
        f"{sum(r['kind'] == 'taxon' or r['n_papers'] >= 2 for r in rows)} / {sum(r['kind'] == 'phylum' for r in rows)}")
    panels, _, rows = figure3.compute()
    label = {r["rid"]: r["label"] for r in rows}
    for p in panels:
        put("Figure 3", f"panel {p['ctx']}, {p['site']}: studies / taxa with >= 3 studies", f"{p['n']} / {len(p['q'])}")
        for rid in sorted(p["q"], key=lambda r: label[r]):
            v = p["sm"][rid]
            put("Results (Figure 3 cells)", f"{p['ctx']}, {p['site']}: {label[rid]}",
                f"enriched {v['papers_increased']}, depleted {v['papers_decreased']}, both {v['papers_mixed']} "
                f"of {v['papers_total']}")


def diversity():
    v = D.diversity_votes()
    a = v[(v.metric_type == "alpha") & v.outcome.notna()]
    b = v[(v.metric_type == "beta") & v.outcome.notna()]
    put("Results", "diversity results extracted", f"{len(D.diversity()):,}")
    put("Abstract/Results", "alpha-diversity votes / studies", f"{len(a)} / {a.key.nunique()}")
    put("Results", "beta-diversity votes / studies", f"{len(b)} / {b.key.nunique()}")
    ao = a.outcome.value_counts()
    put("Results", "alpha: lower / higher in the unfavourable group", f"{ao.get('lower', 0)} / {ao.get('higher', 0)}")
    put("Abstract/Results", "alpha: no significant difference", pct(int(ao.get("ns", 0)), len(a)))
    x = a[(a.site == "lower-respiratory-tract") & (a.comparison == "cancer vs benign")]
    s = x.groupby("key").outcome.agg(set)
    put("Results", "alpha, LRT cancer vs benign: studies higher / lower / all",
        f"{int(s.map(lambda o: 'higher' in o).sum())} / {int(s.map(lambda o: 'lower' in o).sum())} / {len(s)}")
    for o, n in b.outcome.value_counts().items():
        put("Results", f"beta: {o}", pct(int(n), len(b)))
    cols, allv, _ = figureS4.compute()
    put("Figure S4", "body site x comparison pairs / studies", f"{len(cols)} / {allv.key.nunique()}")


def classifiers():
    c = figureS5.classifier_votes()
    N = c.key.nunique()
    put("Results", "classifier studies", N)
    for k, n in c.groupby("ctx").key.nunique().sort_values(ascending=False, kind="stable").items():
        put("Results/Figure S5a", f"classifier comparison: {k}", n)
    for k, n in c.groupby("taxonomic_rank").key.nunique().items():
        put("Results/Figure S5a", f"classifier feature rank: {k}", n)
    put("Results/Figure S5b", "taxa selected by >= 2 classifier studies (shown)", len(figureS5.matrix_data(c)[3]))
    lv = c.drop_duplicates("key").validation_level.value_counts()
    for k, n in lv.items():
        put("Results/Table S4", f"validation: {k}", pct(int(n), N))
    p = D.cohort().set_index("key")
    auc = p.best_reported_auc.dropna()
    put("Results", "studies reporting an AUC / median best AUC", f"{len(auc)} / {auc.median():.2f}")
    put("Results", "AUC >= 0.90 / >= 0.95", f"{int((auc >= 0.90).sum())} / {int((auc >= 0.95).sum())}")
    small = p.loc[auc.index]
    put("Results", "AUC >= 0.90 with fewer than 100 participants",
        int(((small.best_reported_auc >= 0.90) & (small.n_total < 100)).sum()))
    top = small[(small.best_reported_auc >= 0.99)]
    for k, r in top.iterrows():
        put("Results", "highest AUC: value / participants / validation",
            f"{r.best_reported_auc} / {r.n_total:.0f} / {r.validation_level}")


def immunotherapy():
    V, d = figure4.ici_votes()
    n_votes = V.drop_duplicates(["key", "taxon", "taxonomic_rank", "body_site_resolved", "comparison_context",
                                 "direction_to"]).shape[0]
    put("Results/Figure 4a", "ICI studies / votes", f"{V.key.nunique()} / {n_votes}")
    p, f = D.papers(), D.findings()
    imm = set(p[p.intervention_category.str.contains("immunotherapy")].key) - set(V.key)
    no_da = imm - set(f[f.finding_type == "differential abundance"].key)
    put("Results", "immunotherapy studies without differential-abundance findings", len(no_da))
    for h, items in figure4.landscape_counts(V):
        for k, n in items:
            put("Results/Figure 4a", f"{h}: {k}", n)
    s, n_gut = figure4.responder_summary(d)
    put("Results/Figure 4b", "gut responder studies / taxa (>= 2 studies)", f"{n_gut} / {len(s)}")
    for r in s.itertuples():
        put("Results (Figure 4b)", r.label, f"higher in responders {r.papers_down}, in non-responders {r.papers_up}, "
                                             f"both {r.papers_mixed} of {r.papers_total}")
    x = d[d.ctype.eq(figure4.T_RNR) & d.direction_vote.isin(["up", "down"]) & d.clinical_valence.eq("unfavorable")
          & d.site.eq("gut")]
    for ph in ["Bacillota", "Actinomycetota"]:
        g = x[x.phylum == ph]
        put("Results", f"studies with any {ph} taxon higher in non-responders / responders",
            f"{g[g.direction_vote == 'up'].key.nunique()} / {g[g.direction_vote == 'down'].key.nunique()}")


def china():
    cells, t, s = figureS2.compute()
    put("Results/Figure S2a", "cells reported by both groups / with a non-zero net in both",
        f"{s['cells_both']} / {s['cells_nonzero_both']}")
    put("Results", "same direction (observed)", f"{100 * s['agreement']:.0f}%")
    put("Results", "permuted: median (95% range); one-sided P",
        f"{100 * s['perm_median']:.0f}% ({100 * s['perm_lo']:.0f}-{100 * s['perm_hi']:.0f}%); "
        f"P = {s['p_one_sided']:.4f}")
    put("Results/Figure S2b", "Figure 3 taxon-context entries", len(t))
    for k, n in t.status.value_counts().items():
        put("Results/Figure S2b", f"without China: {k}", n)


def table_s4(d_rep):
    m = D.cohort()
    N = len(m)
    flags = m.statistical_quality_flag_tags.fillna("")
    c = figureS5.classifier_votes()
    lv = c.drop_duplicates("key").validation_level
    rows = [("Fewer than 100 participants", int((m.n_total < 100).sum()), N),
            ("No multiplicity correction", int(flags.str.contains("No-multiplicity-correction").sum()), N),
            ("Smallest comparison arm below 15 participants", int(flags.str.contains("Small-sample-size").sum()), N),
            ("No adjustment for confounding", int((m.model == "unadjusted").sum()), N),
            ("Matched or balanced without an adjusted model", int((m.model == "balanced-population").sum()), N),
            ("Confounders entered into a model", int((m.model == "covariate-adjusted").sum()), N),
            ("Confounders measured, imbalanced and not adjusted for",
             int(flags.str.contains("Confounder-imbalanced-unadjusted").sum()), N),
            ("Smoking status not reported", int((m.smoking_status == "no").sum()), N),
            ("Single-timepoint sampling", int((m.sample_timing == "single-timepoint").sum()), N),
            ("Classifier studies validated in an external cohort", int((lv == "External cohort").sum()), len(lv))]
    pd.DataFrame([{"Feature": f, "Studies": (str(n) if b == N else f"{n} of {b}"), "%": f"{100 * n / b:.0f}%"}
                  for f, n, b in rows]).to_csv(D.OUT / "Table_S4.csv", index=False)


def main():
    corpus()
    d = replication()
    figures_2_3()
    diversity()
    classifiers()
    immunotherapy()
    china()
    table_s4(d)
    pd.DataFrame(OUT).to_csv(D.OUT / "manuscript_numbers.csv", index=False)
    print(f"wrote manuscript_numbers.csv ({len(OUT)} numbers), Table_S4.csv, Table_S5.csv")


if __name__ == "__main__":
    main()
