"""Optional protein-information filters for SCANVI supervision.

CytoFuse remains label-free and its taxonomy is kept unchanged.  This module
only changes which cells are treated as observed labels by the downstream
SCANVI stage.  Protein decoder targets remain grouped by the original node.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def mark_low_information_clusters(
    cytofuse_run: Path,
    *,
    max_positive_markers: int = 2,
) -> dict[str, object]:
    """Mark low-positive-marker local clusters as unlabeled for SCANVI.

    The original node IDs and node kinds are preserved in ``*_before_filter``
    columns.  Protein target preparation can therefore still use the original
    node/batch groups, while training reads ``assigned_node_kind`` and sees
    these cells as ``unlabeled``.
    """

    if max_positive_markers < 0:
        raise ValueError("max_positive_markers must be non-negative")
    tables = Path(cytofuse_run) / "tables"
    assignments_path = tables / "cell_taxonomy_assignments.csv"
    if not assignments_path.exists():
        raise FileNotFoundError(assignments_path)
    assignments = pd.read_csv(assignments_path, low_memory=False)
    required = {"cell_barcode", "batch", "local_node_id"}
    missing = required - set(assignments.columns)
    if missing:
        raise KeyError(f"taxonomy assignments lack columns: {sorted(missing)}")

    marker_rows: list[pd.DataFrame] = []
    for path in sorted((Path(cytofuse_run) / "batch_models").glob(
        "*_cluster_marker_library_valid.csv"
    )):
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            continue
        columns = {"batch", "cluster", "n_positive_markers"}
        if not columns.issubset(frame.columns):
            continue
        marker_rows.append(frame[[
            "batch", "cluster", "n_positive_markers",
            *[x for x in (
                "n_negative_markers", "n_selected_markers",
                "mean_marker_confidence", "mean_marker_weight",
                "mean_marker_effect",
            ) if x in frame.columns],
        ]])
    if not marker_rows:
        raise FileNotFoundError(
            "no *_cluster_marker_library_valid.csv files found under "
            f"{Path(cytofuse_run) / 'batch_models'}"
        )
    marker_info = pd.concat(marker_rows, ignore_index=True)
    marker_info["local_node_id"] = (
        marker_info["batch"].astype(str) + "|" +
        marker_info["cluster"].astype(str)
    )
    marker_info = marker_info.drop_duplicates("local_node_id")
    assignments = assignments.copy()
    assignments["low_information_before_filter"] = False
    assignments["low_information_n_positive_markers"] = pd.NA
    assignments["low_information_filter_reason"] = ""
    assignments = assignments.merge(
        marker_info.drop(columns=["batch", "cluster"], errors="ignore"),
        on="local_node_id", how="left", suffixes=("", "_marker"),
    )
    positive = pd.to_numeric(
        assignments["n_positive_markers"], errors="coerce"
    )
    low = positive.le(max_positive_markers).fillna(False)
    assignments["low_information_before_filter"] = low.to_numpy()
    assignments["low_information_n_positive_markers"] = positive
    assignments["low_information_filter_reason"] = low.map(
        lambda value: (
            f"n_positive_markers<={max_positive_markers}" if value else ""
        )
    )

    # Keep an immutable audit copy.  The node itself remains available for
    # protein translation; only the classifier supervision kind is changed.
    for column in ("assigned_taxonomy_node_id", "assigned_node_kind",
                   "integrated_supervision_node_id", "integrated_node_kind"):
        if column in assignments.columns:
            assignments[f"{column}_before_low_information_filter"] = assignments[column]
    for column in ("assigned_node_kind", "integrated_node_kind"):
        if column in assignments.columns:
            assignments.loc[low, column] = "unlabeled"
    assignments.to_csv(assignments_path, index=False)

    cluster_summary = (
        assignments.loc[low]
        .groupby(["batch", "local_node_id"], as_index=False)
        .agg(
            n_cells=("cell_barcode", "size"),
            n_positive_markers=("n_positive_markers", "first"),
        )
    )
    cluster_summary["filter_reason"] = (
        f"n_positive_markers<={max_positive_markers}"
    )
    cluster_summary.to_csv(
        tables / "low_information_supervision_clusters.csv", index=False
    )
    summary = {
        "enabled": True,
        "max_positive_markers": int(max_positive_markers),
        "n_filtered_clusters": int(cluster_summary.shape[0]),
        "n_filtered_cells": int(low.sum()),
        "taxonomy_assignments": str(assignments_path),
        "cluster_summary": str(tables / "low_information_supervision_clusters.csv"),
        "protein_targets_preserved": True,
        "scanvi_classification_supervision": "unlabeled",
        "scanvi_parent_set_supervision": "neutral_mask",
    }
    (Path(cytofuse_run) / "low_information_supervision_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = ["mark_low_information_clusters"]
