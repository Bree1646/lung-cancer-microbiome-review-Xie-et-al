"""Body-site silhouettes drawn as vector paths."""
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle
from matplotlib.transforms import Affine2D

M, L, C, Z = Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CLOSEPOLY


def _lung(side):
    s = side
    v = [(0.12 * s, 0.70), (0.55 * s, 0.85), (0.92 * s, 0.10), (0.85 * s, -0.75),
         (0.55 * s, -0.95), (0.30 * s, -0.80), (0.12 * s, -0.60),
         (0.08 * s, -0.20), (0.10 * s, 0.30), (0.12 * s, 0.70), (0.12 * s, 0.70)]
    return Path(v, [M, C, C, C, C, C, C, C, C, C, Z])


def _drop(cx=0.0, cy=0.0, k=1.0):
    v = [(0, 1), (0.25, 0.55), (0.70, 0.05), (0.70, -0.35), (0.70, -0.80), (0.35, -1.0), (0, -1.0),
         (-0.35, -1.0), (-0.70, -0.80), (-0.70, -0.35), (-0.70, 0.05), (-0.25, 0.55), (0, 1), (0, 1)]
    v = [(cx + k * a, cy + k * b) for a, b in v]
    return Path(v, [M, C, C, C, C, C, C, C, C, C, C, C, C, Z])


def _tooth():
    v = [(-0.80, 0.55), (-0.85, 0.95), (-0.35, 1.00), (0.0, 0.82), (0.35, 1.00), (0.85, 0.95), (0.80, 0.55),
         (0.75, 0.10), (0.62, -0.10), (0.55, -0.45), (0.48, -0.95), (0.25, -1.00), (0.18, -0.65),
         (0.12, -0.30), (-0.12, -0.30), (-0.18, -0.65), (-0.25, -1.00), (-0.48, -0.95), (-0.55, -0.45),
         (-0.62, -0.10), (-0.75, 0.10), (-0.80, 0.55), (-0.80, 0.55)]
    return Path(v, [M] + [C] * 21 + [Z])


def _strokes(lines):
    v, c = [], []
    for ln in lines:
        v += ln; c += [M] + [L] * (len(ln) - 1)
    return Path(v, c)


def _gut():
    colon = [[(-0.25, -0.95), (-0.70, -0.85), (-0.75, 0.70), (0.70, 0.75), (0.75, -0.75), (0.25, -0.85),
              (0.25, -0.98)]]
    t = np.linspace(0, 1, 60)
    coil = [[(-0.40 + 0.80 * x, 0.36 - 0.34 * k + 0.07 * np.sin(2 * np.pi * 1.5 * x)) for x in t]
            for k in range(3)]
    for k in range(2):
        sgn = -1 if k % 2 == 0 else 1
        y0 = 0.36 - 0.34 * k
        coil.append([(sgn * (0.40 + 0.14 * np.sin(np.pi * a)), y0 - 0.34 * a) for a in np.linspace(0, 1, 16)])
    return _strokes(colon), _strokes(coil)


def _airway():
    tr = [(0, 1.0), (0, 0.25)]
    br = []
    def branch(x, y, ang, ln, d):
        x2, y2 = x + ln * np.cos(ang), y + ln * np.sin(ang)
        br.append([(x, y), (x2, y2)])
        if d:
            branch(x2, y2, ang + 0.45, ln * 0.62, d - 1)
            branch(x2, y2, ang - 0.45, ln * 0.62, d - 1)
    branch(0, 0.25, np.radians(-145), 0.62, 3)
    branch(0, 0.25, np.radians(-35), 0.62, 3)
    return _strokes([tr]), _strokes(br)


def _person():
    body = [(-0.65, -1.0), (-0.65, -0.25), (-0.60, 0.15), (-0.20, 0.25), (0.0, 0.25), (0.20, 0.25), (0.60, 0.15),
            (0.65, -0.25), (0.65, -1.0), (-0.65, -1.0)]
    return Path(body, [M, L, C, C, L, C, C, C, L, Z])


def draw_icon(ax, site, x, y, size, colour, zorder=7, lw=None, flip=False):
    tf = Affine2D().scale(size / 2, -size / 2 if flip else size / 2).translate(x, y) + ax.transData
    if lw is None:
        lw = max(0.8, size * 72 * 0.075)
    def fill(p, fc=colour, ec="none", lw_=0):
        ax.add_patch(PathPatch(p, facecolor=fc, edgecolor=ec, lw=lw_, transform=tf, zorder=zorder))
    def line(p, w=lw, col=colour):
        ax.add_patch(PathPatch(p, facecolor="none", edgecolor=col, lw=w, capstyle="round", joinstyle="round",
                               transform=tf, zorder=zorder))
    if site == "lung-tissue":
        fill(_lung(-1)); fill(_lung(1))
        line(_strokes([[(0, 1.0), (0, 0.55)], [(0, 0.55), (-0.22, 0.30)], [(0, 0.55), (0.22, 0.30)]]), w=lw * 1.1)
    elif site == "lower-respiratory-tract":
        tr, br = _airway()
        line(tr, w=lw * 1.9); line(br, w=lw * 1.2)
    elif site == "gut":
        colon, coil = _gut()
        line(colon, w=lw * 2.1); line(coil, w=lw * 0.9)
    elif site == "oral":
        fill(_tooth())
    elif site == "blood-plasma-serum":
        fill(_drop(k=0.95))
    elif site == "pleural-fluid":
        sh = Affine2D().scale(0.66).translate(0, 0.14)
        line(_lung(-1).transformed(sh), w=lw * 0.9); line(_lung(1).transformed(sh), w=lw * 0.9)
        ring = Path.unit_circle().transformed(Affine2D().scale(0.95, 0.98))
        line(ring, w=lw * 0.9)
        fill(_drop(0, -0.66, 0.22))
    else:
        ax.add_patch(Circle((0, 0.62), 0.33, facecolor=colour, edgecolor="none", transform=tf, zorder=zorder))
        fill(_person())
