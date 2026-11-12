"""Shared figure style, page layout and phylogram helpers."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from data import OUT, tree, tree_phyla, draw_tree_lines, pcol, phylum_label, gg2_taxa

MM = 1 / 25.4
PT_MM = 25.4 / 72
PAGE_W = 183.0
TXT, SMALL, TINY, HEAD, LETTER = 6.0, 5.0, 4.5, 6.5, 8.0
LW = 0.5
MARGIN = 3.0

BLACK, BAR, GRID, LIGHTGRID = "#000000", "#4D4D4D", "#BDBDBD", "#E6E6E6"
VERM, BLUE, SKY, GREEN, ORANGE, PURPLE, YELLOW, NA_GREY = \
    "#D55E00", "#0072B2", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#F0E442", "#C9C9C9"
UP, DOWN = VERM, BLUE
LIGHT_UP, LIGHT_DOWN = "#F4C3AA", "#A9D5EA"
LOW_AGREE = "#EFEFEF"

SITE_ORDER = ["gut", "lower-respiratory-tract", "lung-tissue", "oral", "blood-plasma-serum",
              "pleural-fluid", "urine", "other"]
SITE_LABEL = {"gut": "Gut", "lower-respiratory-tract": "Lower respiratory tract", "lung-tissue": "Lung tissue",
              "oral": "Oral", "blood-plasma-serum": "Blood / plasma / serum", "pleural-fluid": "Pleural fluid",
              "urine": "Urine", "other": "Other"}
SITE_COLOR = {"gut": GREEN, "lower-respiratory-tract": SKY, "lung-tissue": BLUE, "oral": ORANGE,
              "blood-plasma-serum": VERM, "pleural-fluid": PURPLE, "urine": YELLOW, "other": NA_GREY}
SITE_BY_LABEL = {**{v: k for k, v in SITE_LABEL.items()}, "Blood, plasma or serum": "blood-plasma-serum",
                 "Blood": "blood-plasma-serum"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": TXT, "pdf.fonttype": 42, "svg.fonttype": "none",
    "text.color": BLACK, "axes.labelcolor": BLACK, "axes.edgecolor": BLACK, "axes.linewidth": LW,
    "xtick.color": BLACK, "ytick.color": BLACK, "xtick.labelsize": TXT, "ytick.labelsize": TXT,
    "xtick.major.width": LW, "ytick.major.width": LW, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelsize": TXT, "axes.labelpad": 2.0,
    "legend.fontsize": TXT, "legend.frameon": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "mathtext.default": "regular",
    "lines.linewidth": 0.8, "patch.linewidth": LW,
})


def text_w(s, size, weight="normal", style="normal"):
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    tp = TextPath((0, 0), str(s), size=size, prop=FontProperties(family="Arial", weight=weight, style=style))
    return tp.get_extents().width * PT_MM


class Page:

    def __init__(self, h, w=PAGE_W):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w * MM, h * MM))

    def axes(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h], **kw)

    def canvas(self, x, y, w, h):
        ax = self.axes(x, y, w, h)
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
        ax.set_clip_on(False)
        return ax

    def letter(self, x, y, s):
        self.fig.text(x / self.w, 1 - y / self.h, s, fontsize=LETTER, fontweight="bold", va="top", ha="left")

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", TXT); kw.setdefault("va", "top"); kw.setdefault("ha", "left")
        self.fig.text(x / self.w, 1 - y / self.h, s, **kw)

    def content_bbox(self):
        from matplotlib.transforms import Bbox
        fig = self.fig
        r = fig.canvas.get_renderer()
        boxes = [t.get_window_extent(r) for t in fig.texts if t.get_visible() and t.get_text()]
        for ax in fig.axes:
            if ax.axison:
                boxes.append(ax.get_tightbbox(r))
                continue
            for a in ax.get_children():
                if a is ax.patch or not a.get_visible() or a in ax.spines.values():
                    continue
                if hasattr(a, "get_text") and not a.get_text():
                    continue
                try:
                    b = a.get_window_extent(r)
                except Exception:
                    continue
                if b is not None and np.isfinite(b.bounds).all() and b.width + b.height > 0:
                    boxes.append(b)
        bb = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
        m = MARGIN * MM
        return Bbox.from_extents(bb.x0 - m, bb.y0 - m, bb.x1 + m, bb.y1 + m)

    def save(self, stem):
        kw = dict(bbox_inches=self.content_bbox(), pad_inches=0)
        for ext in ("pdf", "svg", "png"):
            self.fig.savefig(OUT / f"{stem}.{ext}", **kw, **({"dpi": 450} if ext == "png" else {}))
        plt.close(self.fig)
        print("wrote", stem)


def hbar(ax, x, y, items, mm_per, lab_w, row=3.2, bar=2.0, color=BAR, values=True, fs=TXT, pct_of=None):
    x0 = x + lab_w
    for i, (lab, v, c) in enumerate(items):
        yc = y + row * (i + 0.5)
        ax.text(x0 - 1.0, yc, lab, ha="right", va="center", fontsize=fs)
        ax.add_patch(Rectangle((x0, yc - bar / 2), max(v * mm_per, 0.25), bar, fc=c or color, ec="none"))
        if values:
            s = f"{v:,}" if pct_of is None else f"{v} ({100 * v / pct_of:.0f}%)"
            ax.text(x0 + v * mm_per + 0.8, yc, s, ha="left", va="center", fontsize=fs, fontweight="bold")
    ax.plot([x0, x0], [y, y + row * len(items)], color=BLACK, lw=LW)
    return y + row * len(items)


def swatch(ax, x, y, c, label, w=2.4, h=1.8, fs=TXT, ec="none", lw=LW, hatch=None):
    ax.add_patch(Rectangle((x, y - h / 2), w, h, fc=c, ec=ec, lw=lw, hatch=hatch))
    ax.text(x + w + 0.9, y, label, va="center", ha="left", fontsize=fs)
    return x + w + 0.9 + text_w(label, fs) + 3.0


def legend_row(ax, x, y, entries, width, fs=TXT, row=2.8, **kw):
    cx, cy = x, y
    for c, lab in entries:
        need = 2.4 + 0.9 + text_w(lab, fs) + 3.0
        if cx > x and cx + need > x + width:
            cx, cy = x, cy + row
        cx = swatch(ax, cx, cy, c, lab, fs=fs, **kw)
    return cy + row


def text_on(hexcol):
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return BLACK if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "#FFFFFF"


ITALIC = {"genus", "species", "strain", "subspecies"}


class TreeRows(list):
    pass


def tree_order(items, keys):
    tr = tree()
    root, order, kids = tr.layout([(i, k[0], (0 if k[2] else 1, k[3])) for i, k in enumerate(keys)])
    out = TreeRows(items[i] for i in order)
    remap = {o: n for n, o in enumerate(order)}
    out.layout = (root, {nd: [("L", remap[c[1]]) if isinstance(c, tuple) else c for c in cs] for nd, cs in kids.items()})
    out.nodes = [keys[i][0] for i in order]
    out.phyla = [keys[i][1] for i in order]
    runs = [k for k, ph in enumerate(out.phyla) if k == 0 or ph != out.phyla[k - 1]]
    assert len(runs) == len(set(out.phyla)) and all(keys[order[k]][2] for k in runs)
    return out


def tree_rows(sel):
    tr = tree()
    items, keys = [], []
    for p, g in sel.groupby("phylum"):
        items.append(("header", phylum_label(p), "P|" + p, "phylum"))
        keys.append((tr.mrca([int(n) for n in g.node]), p, True, (-1, "")))
        for r in g.itertuples():
            if r.row_id.startswith("P|"):
                continue
            items.append(("taxon", r.label, r.row_id, r.row_rank))
            keys.append((int(r.node), p, False, (r.gg2_axis_order, r.label)))
    t = tree_order(items, keys)
    return t, [dict(kind=k, label=l, rid=r, rank=rk) for k, l, r, rk in t]


def leaf_info(names, strict=True):
    t = gg2_taxa().dropna(subset=["tree_leaf", "tree_node_id"])
    leaf = t.groupby("tree_leaf").agg(node=("tree_node_id", "first"), order=("gg2_axis_order", "min"))
    out = leaf.reindex(list(names))
    missing = [n for n, v in zip(names, out.node) if pd.isna(v)]
    if strict:
        assert not missing, f"not on the GG2 tree: {missing}"
    return [None if pd.isna(v) else int(v) for v in out.node], out.order.tolist()


def place_rows(rows, rh, pgap):
    y, ys = 0.0, []
    for i, r in enumerate(rows):
        if r["kind"] == "header" and i:
            y += pgap
        ys.append(y + rh / 2)
        y += rh
    return ys, y


def draw_tree(ax, rows, tip, x0, w, lead_x, lw=0.7, ms=1.8):
    tr = tree()
    root, kids = rows.layout
    base = tr.length[root]
    maxd = max(tr.length[n] - base for n in rows.nodes) or 1.0
    unit = (w - 0.05) / maxd
    sets = tree_phyla(root, kids, dict(enumerate(rows.phyla)))
    colour_of = lambda nd: pcol(next(iter(sets[nd]))) if len(sets[nd]) == 1 else BLACK
    segs, scol, ldep = draw_tree_lines(tr, root, kids, tip, colour_of,
                                       lambda dd, p: (x0 + dd * unit, p),
                                       lambda dd, a, b: [(x0 + dd * unit, a), (x0 + dd * unit, b)])
    ax.add_collection(LineCollection(segs, colors=scol, linewidths=lw, zorder=4, capstyle="projecting"))
    for i in range(len(rows)):
        c = pcol(rows.phyla[i])
        xx = x0 + ldep[i] * unit
        ax.plot([xx, lead_x], [tip[i], tip[i]], color=c, lw=lw * 0.64, ls=(0, (1, 1.6)), zorder=3, clip_on=False)
        ax.plot([xx], [tip[i]], marker="o", ms=ms, color=c, zorder=5, mec="none", clip_on=False)
    return unit


def draw_tree_axis(ax, tr, rows, ys, x_tree, tree_w, lab_x, rank_x, rule_to, rh, y0=0.0,
                   fs=SMALL, fs_rank=TINY, rank_for=None):
    tip = {i: y0 + y for i, y in enumerate(ys)}
    unit = draw_tree(ax, tr, tip, x_tree, tree_w, lab_x - 0.7, lw=0.4, ms=1.0)
    for i, r in enumerate(rows):
        y = y0 + ys[i]
        head = r["kind"] == "header"
        ax.text(lab_x, y, r["label"], fontsize=fs, va="center", ha="left",
                fontweight="bold" if head else "normal",
                fontstyle="italic" if (r["rank"] in ITALIC and not head) else "normal")
        rk = rank_for(r) if rank_for else ("Phylum" if head else r["rank"].capitalize())
        if rk:
            ax.text(rank_x, y, rk, fontsize=fs_rank, va="center", ha="right", fontweight="bold")
        if head:
            ax.plot([lab_x, rule_to], [y - rh / 2] * 2, color=BLACK, lw=0.7, zorder=8, solid_capstyle="butt")
            ax.plot([lab_x, rule_to], [y + rh / 2] * 2, color=BLACK, lw=0.35, zorder=8, solid_capstyle="butt")
    return unit


def scale_bar(ax, x, y, unit, length=50, fs=TINY):
    w = length * unit
    for xs, ys in (([x, x + w], [y, y]), ([x, x], [y - 0.5, y + 0.5]), ([x + w, x + w], [y - 0.5, y + 0.5])):
        ax.plot(xs, ys, color=BLACK, lw=0.5)
    ax.text(x + w + 0.8, y, f"{length} (Greengenes2 branch length)", fontsize=fs, va="center", ha="left")


def label_width(rows, fs=SMALL):
    return max(text_w(r["label"], fs, "bold") if r["kind"] == "header" else
               text_w(r["label"], fs, style="italic" if r["rank"] in ITALIC else "normal") for r in rows)
