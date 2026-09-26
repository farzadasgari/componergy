"""Print-safe (grayscale-readable) styling for generation-source and event plots."""

from __future__ import annotations

SOURCE_STYLE = {
    "Fossil": {"color": "#d55e00", "linestyle": "-", "marker": "o", "hatch": "//"},
    "Hydro": {"color": "#0072b2", "linestyle": "--", "marker": "s", "hatch": "\\\\"},
    "Solar": {"color": "#e69f00", "linestyle": (0, (3, 1, 1, 1)), "marker": "^", "hatch": "xx"},
    "Wind": {"color": "#009e73", "linestyle": ":", "marker": "D", "hatch": ".."},
    "Nuclear": {"color": "#cc79a7", "linestyle": "-.", "marker": "v", "hatch": "++"},
    "Other Renewable": {"color": "#999999", "linestyle": (0, (5, 1)), "marker": "P", "hatch": "oo"},
}

EVENT_STYLE = {
    "Heatwave": {"color": "#d55e00", "hatch": "///"},
    "Drought": {"color": "#0072b2", "hatch": "\\\\\\"},
    "Compound": {"color": "#cc79a7", "hatch": "xx"},
}
