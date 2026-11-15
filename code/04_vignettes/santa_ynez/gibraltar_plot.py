"""
Gibraltar Dam annual-mean discharge panel (import; do not copy). Draws from the files written by
gibraltar_annual.py: data/analysis/santa_ynez/gibraltar_annual.csv (plotted series) and
gibraltar_restored_monthly.csv (restored yearly values, for the overlap markers).
"""
from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[3] / 'data/analysis/santa_ynez'
GIBRALTAR_DAM, JUNCAL_DAM = 1920, 1930
NW, RS = '#4477AA', '#EE6677'
FLOOR = 0.001                      # zero-flow years are drawn on the bottom axis (log scale)


def _runs(s):
    """Split a year-sorted frame into runs of consecutive water years (no line across gaps)."""
    s = s.sort_values('wy')
    return [g for _, g in s.groupby((s.wy.diff() != 1).cumsum())]


def _draw(ax, s, color, ms, lw, open_marker=False):
    pos, zero = s[s.mean_cfs > 0], s[s.mean_cfs <= 0]
    if open_marker:
        ax.plot(pos.wy, pos.mean_cfs, 'o', mfc='white', mec=color, ms=ms, mew=0.7)
    else:
        for g in _runs(pos):
            ax.plot(g.wy, g.mean_cfs, '-o', color=color, ms=ms, lw=lw)
    ax.plot(zero.wy, [FLOOR] * len(zero), 'v', color=color, ms=ms + 0.5, clip_on=False)


def plot_gibraltar(ax, legend_kw=None):
    series = pd.read_csv(OUT / 'gibraltar_annual.csv')
    rest = pd.read_csv(OUT / 'gibraltar_restored_monthly.csv')
    for src in ['NWIS 11122000 (above dam)', 'NWIS 11123000 (below dam)']:
        s = series[series.src == src]
        _draw(ax, s[s.complete.astype(bool)], NW, 2.5, 0.9)
        _draw(ax, s[~s.complete.astype(bool)], NW, 2.5, 0.9, open_marker=True)
    _draw(ax, series[series.src == 'Restored'], RS, 3, 1.1)
    ov = rest[(rest.water_year <= 1918) | (rest.water_year >= 1934)].dropna(subset=['annual_cfs'])
    ax.plot(ov.water_year, ov.annual_cfs.clip(lower=FLOOR), 'x', color=RS, ms=3, mew=0.7)
    ax.axvline(GIBRALTAR_DAM - 0.5, color='0.3', lw=0.8, ls='--')
    ax.text(GIBRALTAR_DAM - 0.2, 1500, 'Gibraltar Dam\ncompleted 1920', fontsize=6, va='top', ha='left', color='0.3')
    ax.axvline(JUNCAL_DAM + 0.5, color='0.6', lw=0.6, ls=':')
    ax.text(JUNCAL_DAM + 0.7, 1500, 'Juncal Dam\n(upstream) 1930', fontsize=6, va='top', ha='left', color='0.45')
    ax.set_yscale('log')
    ax.set_ylim(FLOOR, 2000)
    ax.set_xlim(1902, 1980)
    ax.set_xlabel('Water year')
    ax.set_ylabel('Annual mean discharge (cfs)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.25, which='major')
    h = [Line2D([], [], color=NW, marker='o', ms=2.5, lw=0.9, label='NWIS daily values, annual mean'),
         Line2D([], [], color=NW, marker='o', ms=2.5, lw=0, mfc='white', label='NWIS, incomplete water year'),
         Line2D([], [], color=RS, marker='o', ms=3, lw=1.1, label='Restored, not in NWIS (1921-1933)'),
         Line2D([], [], color=RS, marker='x', ms=3, lw=0, label='Restored, also in NWIS'),
         Line2D([], [], color='0.4', marker='v', ms=3, lw=0, label='Zero flow for the year')]
    ax.legend(handles=h, **(legend_kw or dict(fontsize=6, frameon=False, loc='upper center',
                                              bbox_to_anchor=(0.5, -0.2), ncol=3)))
    return series[series.src == 'Restored']
