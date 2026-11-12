# The lung cancer microbiome: data and code

Data and code for the systematic review of 144 primary human studies of the microbiome in pathology-confirmed
lung cancer. Running `scripts/run_all.py` rebuilds every figure, Supplementary Tables S4 and S5, and the numbers
in the text from the four data workbooks below.

## Data

| File | Sheet | Rows | One row per |
|---|---|---|---|
| `papers.xlsx` | papers | 144 | included study: bibliographic fields, journal impact factor, sampling country, assay, ICD-11 codes, body site, sample timing, study design, intervention, immune-checkpoint-inhibitor (ICI) regimen, confounder handling, sample size, statistical-quality flags, classifier validation level, best reported AUC |
| `findings.xlsx` | findings | 4,711 | taxon-level finding: taxon and rank, finding type, direction, the group the direction refers to (`direction_to`) and its clinical valence, comparison context, evidentiary tier, body site; for ICI comparisons, the comparison type and responder definition |
| `findings.xlsx` | FB_ratio | 10 | Firmicutes/Bacteroidetes ratio finding |
| `diversity_findings.xlsx` | diversity_findings | 1,028 | alpha- or beta-diversity result: metric, outcome, `direction_to`, valence, body site, comparison context |
| `gg2_placement.xlsx` | taxa | 1,063 | taxon name and rank in `findings.xlsx`, placed on its closest-fit node of the Greengenes2 2024.09 phylogeny |
| `gg2_placement.xlsx` | tree_nodes | 1,538 | node of the pruned Greengenes2 tree (parent node and root-to-node branch length) |

Studies are linked across files by `key`. Controlled vocabularies are listed in the column headers of
`papers.xlsx`.

## Code

`scripts/` (Python 3.11; `pip install -r requirements.txt`; figures use the Arial font):

| Script | Output in `results/` |
|---|---|
| `data.py` | shared data layer: loads the workbooks and derives the vote table and the Greengenes2 placement |
| `plot.py`, `site_icons.py` | shared figure style, layout and phylogram helpers; body-site silhouettes |
| `figure1.py` … `figure4.py` | Figures 1-4 |
| `figureS1.py` … `figureS5.py` | Supplementary Figures S1-S5 |
| `manuscript_numbers.py` | `manuscript_numbers.csv` (numbers in the text and legends), `Table_S4.csv`, `Table_S5.csv` |
| `run_all.py` | everything above |

```
cd scripts
python3 run_all.py
```

Each figure is written as PNG (450 dpi), PDF and SVG. The China-versus-rest-of-world permutation test
(Supplementary Figure S2) uses a fixed seed, so results are identical on every run.
