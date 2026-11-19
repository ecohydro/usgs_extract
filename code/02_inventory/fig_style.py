"""
Shared style for all manuscript figures, following the journal's figure guidelines (Annette, 2026-11-19).

Colours for restored vs NWIS data = the Figure 3 scheme (chosen 2026-11-15; every pair separated under
normal + deuteranopia/protanopia/tritanopia simulation):
  NWIS        grey  #9E9E9E   NWIS data
  BOTH        gold  #DDAA33   restored data at an NWIS site / restored values also in NWIS
  RESTORED    blue  #0072B2   restored data (no NWIS counterpart)
Water-type colours (Figures 1-2) live in water_palette.py.

Journal rules applied here:
  - one typeface (Arial) in one size: everything 7 pt; panel letters lower-case bold in the same size;
    panel titles bold 7 pt (Annette: try bold titles at 7 pt)
  - thinnest lines >= 1 pt (LW_MIN). Exception kept by Annette: subwatershed (HUC12) and HUC8 boundary
    lines on the maps stay thin
  - labels in sentence case (label()); commas only in numbers of five or more digits (num())
  - maps carry a scale bar with its length written on the bar (scalebar())

Usage:
  import fig_style as S
  S.apply()                      # rcParams: Arial 7 pt, line widths >= 1 pt, bold titles
  S.letter(ax, 'a')              # panel letter
"""
import matplotlib.pyplot as plt
import numpy as np

NWIS, BOTH, RESTORED = '#9E9E9E', '#DDAA33', '#0072B2'
NWIS_DARK = '#6E6E6E'             # NWIS grey for thin outlines / hollow markers, where #9E9E9E is too faint
OUTLINE = '#4d4d4d'               # polygon outlines, state boundary
LAND = '#ececea'                  # background land fill on maps
GRID = '#e6e6e6'

WIDTH = 7.25                      # in, journal full width
SIZE = 7                          # the one font size
LETTER = TITLE = BODY = SMALL = SIZE
LW_MIN = 1.0                      # thinnest line, points
FONT = 'Arial'


def apply():
    plt.rcParams.update({
        'font.family': FONT, 'font.size': SIZE, 'axes.titlesize': SIZE, 'axes.titleweight': 'bold',
        'axes.labelsize': SIZE, 'xtick.labelsize': SIZE, 'ytick.labelsize': SIZE,
        'legend.fontsize': SIZE, 'legend.title_fontsize': SIZE, 'figure.titlesize': SIZE,
        'axes.linewidth': LW_MIN, 'grid.linewidth': LW_MIN, 'lines.linewidth': max(LW_MIN, 1.0),
        'patch.linewidth': LW_MIN, 'hatch.linewidth': LW_MIN,
        'xtick.major.width': LW_MIN, 'ytick.major.width': LW_MIN,
        'xtick.minor.width': LW_MIN, 'ytick.minor.width': LW_MIN,
        'lines.markeredgewidth': LW_MIN, 'legend.frameon': False,
    })


def letter(ax, s, x=-0.02, y=1.0, **kw):
    """Lower-case bold panel letter, same size as the rest of the figure (baseline of a loc='left' title)."""
    kw = {'ha': 'right', 'va': 'baseline', **kw}
    return ax.text(x, y, s.lower(), transform=ax.transAxes, fontsize=LETTER, fontweight='bold', **kw)


def label(name):
    """Sentence case for display: 'Stream Discharge' -> 'Stream discharge' (data keep their names).
    Acronyms written in capitals (NWIS, HUC) are left alone."""
    words = str(name).split(' ')
    return ' '.join([words[0]] + [w if (w.isupper() and len(w) > 1) else w.lower() for w in words[1:]])


def num(n):
    """Journal number style: commas only for five or more digits (1376, 23,055)."""
    n = int(round(n))
    return f'{n:,}' if abs(n) >= 10000 else str(n)


def num_formatter():
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, _: num(v) if float(v).is_integer() else f'{v:g}')


def scalebar(ax, km, lat, loc=(0.05, 0.05), color='black'):
    """Scale bar for a lon/lat axes (degrees; any aspect). Length written on the bar. `lat` = latitude
    used for the km -> degrees conversion (map centre)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = km / (111.32 * np.cos(np.radians(lat)))
    xs = x0 + loc[0] * (x1 - x0)
    ys = y0 + loc[1] * (y1 - y0)
    ax.plot([xs, xs + dx], [ys, ys], color=color, lw=2.0, solid_capstyle='butt', zorder=20)
    ax.text(xs + dx / 2, ys + 0.012 * (y1 - y0), f'{km:g} km', ha='center', va='bottom', fontsize=SIZE,
            color=color, zorder=20)
