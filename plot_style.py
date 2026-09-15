"""
Shared chart style for the report figures.

Colours come from one validated palette (checked for colour-blind
separation and contrast against the background): a single blue for
one-series charts, blue and orange for two series, and shades of blue for
ordered categories and heatmaps. Bars have a rounded data end and a square
baseline; grids and axes are thin and quiet so the data stands out.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.transforms import IdentityTransform  # noqa: E402

OUT = Path(__file__).parent / "results" / "charts"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GRAY = "#c3c2b7"          # de-emphasised marks
BLUE = "#2a78d6"          # series 1
ORANGE = "#eb6834"        # series 2
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]  # light to dark
ORDINAL3 = ["#86b6ef", "#2a78d6", "#184f95"]
ORDINAL4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
PAIR = ["#86b6ef", "#1c5cab"]   # before / after

PX = 0.75                       # one screen pixel, in points
SEMIBOLD = {"fontfamily": ["Segoe UI Semibold", "DejaVu Sans"]}
SEQ = LinearSegmentedColormap.from_list("seq", RAMP)


def setup():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 9,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": PX,
        "axes.labelcolor": INK2,
        "axes.labelsize": 9,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    })


class RoundedBar(Patch):
    """A bar with a rounded data end and a square baseline end.

    Geometry is built in display space at draw time, so the corners stay
    circular on any axis scale (including log) and at any output size.
    Sizes are in points; `offset` shifts the bar across its category axis."""

    K = 0.5523  # cubic Bezier constant for a quarter circle

    def __init__(self, ax, v0, v1, pos, thick, radius, horizontal=True, offset=0.0, **kw):
        super().__init__(**kw)
        self._ax, self._v0, self._v1, self._pos = ax, v0, v1, pos
        self._thick, self._radius, self._h, self._offset = thick, radius, horizontal, offset
        self.set_transform(IdentityTransform())

    def get_path(self):
        ax = self._ax
        pt = ax.figure.dpi / 72.0
        if self._h:
            (b, c), (e, _) = ax.transData.transform([(self._v0, self._pos), (self._v1, self._pos)])
        else:
            (c, b), (_, e) = ax.transData.transform([(self._pos, self._v0), (self._pos, self._v1)])
        c += self._offset * pt
        h = self._thick * pt / 2
        length = abs(e - b)
        if length < 0.5:
            return MPath([(0, 0), (0, 0)])
        s = 1 if e >= b else -1
        r = min(self._radius * pt, h, length)
        k = self.K * r
        # along = value axis, across = category axis; build as (along, across)
        pts = [(b, c - h), (e - s * r, c - h),
               (e - s * r + s * k, c - h), (e, c - h + r - k), (e, c - h + r),
               (e, c + h - r),
               (e, c + h - r + k), (e - s * r + s * k, c + h), (e - s * r, c + h),
               (b, c + h), (b, c - h)]
        codes = [MPath.MOVETO, MPath.LINETO,
                 MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.LINETO,
                 MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.LINETO, MPath.CLOSEPOLY]
        if not self._h:
            pts = [(y, x) for x, y in pts]
        return MPath(pts, codes)


def bar(ax, v0, v1, pos, color, thick_px=16, offset_px=0.0, horizontal=True,
        radius_px=4, zorder=3):
    p = RoundedBar(ax, v0, v1, pos, thick_px * PX, radius_px * PX, horizontal,
                   offset_px * PX, facecolor=color, edgecolor="none", zorder=zorder)
    ax.add_patch(p)
    return p


def quiet_axes(ax, value_axis="x"):
    """Hairline grid on the value axis only; baseline kept, other spines off."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if value_axis == "x":
        ax.spines["bottom"].set_visible(False)
        ax.grid(axis="x", color=GRID, linewidth=PX, linestyle="-", zorder=0)
    else:
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=PX, linestyle="-", zorder=0)
    ax.set_axisbelow(True)


def titles(fig, title, subtitle=None):
    """Title and subtitle at the top left. Returns the figure fraction just
    below them, for fig.subplots_adjust(top=...)."""
    h = fig.get_figheight() * 72          # figure height in points
    y = 1 - 4 / h
    fig.text(0.01, y, title, ha="left", va="top", fontsize=11.5, color=INK, **SEMIBOLD)
    y -= 17 / h
    if subtitle:
        fig.text(0.01, y, subtitle, ha="left", va="top", fontsize=9, color=INK2,
                 linespacing=1.35)
        y -= 13.5 / h * (subtitle.count("\n") + 1)
    return y - 16 / h


def note(fig, text, y=0.01):
    fig.text(0.012, y, text, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def _luminance(color):
    def lin(v):
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = to_rgb(color)
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def text_on(color):
    """White or ink, whichever has more contrast on this fill."""
    L = _luminance(color)
    return "white" if 1.05 / (L + 0.05) > (L + 0.05) / (_luminance(INK) + 0.05) else INK


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path
