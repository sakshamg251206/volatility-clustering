"""Shared figure style and table export."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from volclust.data import ROOT

FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"

# Colour per asset class (colour-blind-safe Okabe-Ito palette)
CLASS_COLORS = {
    "Equity index": "#0072B2", "Single stock": "#56B4E9", "FX": "#009E73",
    "Rates": "#CC79A7", "Commodity": "#E69F00", "Crypto": "#D55E00",
}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "legend.frameon": False, "savefig.bbox": "tight",
})


def save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Minimal DataFrame -> GitHub markdown table (avoids a tabulate dependency)."""
    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    rows = [[str(i)] + [fmt(v) for v in r] for i, r in zip(df.index, df.itertuples(index=False))]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def save_table(df: pd.DataFrame, name: str, floatfmt: str = "{:.3f}") -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv")
    (TAB / f"{name}.md").write_text(to_markdown(df, floatfmt) + "\n")
