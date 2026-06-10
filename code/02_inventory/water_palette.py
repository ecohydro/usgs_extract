
# water_palette.py
# Fixed, color-blind safe mapping for water categories across all plots.

from typing import Dict, List
import seaborn as sns

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

# Use seaborn's colorblind palette (Paul Tol) — 7 distinct colors
_BASE_COLORS = sns.color_palette("colorblind", 7)

# Build a name->color mapping using the canonical names above
COLOR_MAP: Dict[str, tuple] = {name: _BASE_COLORS[i] for i, name in enumerate(CATEGORY_ORDER)}

def normalize_name(name: str) -> str:
    return str(name).strip().lower()

def get_color_for(name: str):
    """Return the fixed color for a category name (case-insensitive).
    If the category is not in our canonical list, we append a new color
    from an extended colorblind palette (cycled) but keep it deterministic
    by sorted position.
    """
    # First try canonical names
    for canon in CATEGORY_ORDER:
        if normalize_name(name) == normalize_name(canon):
            return COLOR_MAP[canon]

    # Fallback: deterministic extension
    extended = sns.color_palette("colorblind", 12)
    # Position hash by name for determinism
    idx = abs(hash(normalize_name(name))) % len(extended)
    return extended[idx]
