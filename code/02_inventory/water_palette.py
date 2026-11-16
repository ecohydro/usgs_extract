
# water_palette.py
# Fixed, colour-blind safe colours for water categories across all figures.
#
# 2026-11-15 (Annette chose "V1"): colours from Paul Tol (2021) and Okabe & Ito (2008), picked so
# stream discharge is blue, with water-like hues for the other water types. Groundwater and
# precipitation colours swapped 2026-11-15 (Annette; groundwater now sky blue, precipitation wine).
# Checked under normal, deuteranopia, protanopia and tritanopia simulation (colorspacious, severity
# 100): every pair differs by OKLab dE x100 >= 9.0 (same 7 colours after the swap;
# target >= 8). Replaces seaborn "colorblind" (worst pair dE 3.2).
# Grey is reserved for "Other (type undetermined)" so it never reads as a water type.

from typing import Dict, List

# Canonical order we will use everywhere (case-insensitive matching)
CATEGORY_ORDER: List[str] = [
    "Stream Discharge",
    "Groundwater",
    "Precipitation",
    "Irrigation",
    "Reservoir",
    "Springs",
    "Water Quality",
]

COLOR_MAP: Dict[str, str] = {
    "Stream Discharge": "#004488",   # deep blue (Tol high-contrast)
    "Groundwater":      "#66CCEE",   # sky blue (Tol bright)
    "Precipitation":    "#882255",   # wine-purple (Tol muted)
    "Irrigation":       "#CCBB44",   # olive-yellow (Tol bright)
    "Reservoir":        "#44AA99",   # teal (Tol muted)
    "Springs":          "#0072B2",   # blue (Okabe-Ito)
    "Water Quality":    "#CC3311",   # red (Tol vibrant)
}
OTHER_COLOR = "#BBBBBB"              # Other (type undetermined) / anything not a core water type

# Trial switch (2026-11-15): WATER_PALETTE=<Crameri categorical set, e.g. batlowS> uses that set's
# first 7 colours in CATEGORY_ORDER (Scientific colour maps v8, cmcrameri package). Default: V1 above.
import os as _os
_trial = _os.environ.get("WATER_PALETTE")
_hex = _os.environ.get("WATER_PALETTE_HEX")   # trial: 7 comma-separated hex colours in CATEGORY_ORDER
if _hex:
    COLOR_MAP = dict(zip(CATEGORY_ORDER, [h.strip() for h in _hex.split(",")]))
elif _trial:
    import matplotlib.colors as _mc
    import cmcrameri.cm as _cmc
    _cols = [_mc.to_hex(c) for c in getattr(_cmc, _trial).colors[:len(CATEGORY_ORDER)]]
    COLOR_MAP = dict(zip(CATEGORY_ORDER, _cols))


def normalize_name(name: str) -> str:
    return str(name).strip().lower()


def get_color_for(name: str) -> str:
    """Fixed colour for a category name (case-insensitive); grey for anything else."""
    for canon in CATEGORY_ORDER:
        if normalize_name(name) == normalize_name(canon):
            return COLOR_MAP[canon]
    return OTHER_COLOR
