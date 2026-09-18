"""Small plotting primitives used by stable model reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc


def draw_categorical_umap(axis, xy: np.ndarray, values: pd.Series, title: str) -> None:
    """Draw one categorical embedding panel with Scanpy-compatible palettes."""

    values = values.astype(object).where(values.notna(), "NA").astype(str)
    n_categories = values.nunique()
    if n_categories <= 20:
        palette = sc.pl.palettes.default_20
    elif n_categories <= 28:
        palette = sc.pl.palettes.default_28
    else:
        palette = sc.pl.palettes.default_102
    for index, category in enumerate(sorted(values.unique())):
        take = values.eq(category).to_numpy()
        axis.scatter(
            xy[take, 0], xy[take, 1], s=4,
            color=palette[index % len(palette)], linewidths=0,
            rasterized=True, label=category,
        )
    axis.set_title(title, fontsize=14)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.legend(
        frameon=False, fontsize=6.5, markerscale=3,
        ncol=2 if values.nunique() > 12 else 1,
        loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0.2,
    )
