"""Prepare cluster-aware, batch-translated protein reconstruction targets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .. import __version__
from ..config import TargetPreparationConfig
from .._utils.preprocessing import dense


def _finite_median(values: np.ndarray) -> np.ndarray:
    """Median over observed values; an entirely unobserved column stays NaN."""
    if np.isfinite(values).all():
        return np.median(values, axis=0)
    return np.ma.median(np.ma.masked_invalid(values), axis=0).filled(np.nan)


def prepare_translated_protein_targets(
    config: TargetPreparationConfig,
) -> dict[str, object]:
    """Create translated protein targets and a copied immutable taxonomy.

    Translation is additive and is estimated independently for every
    supervision-node, batch, and protein. Within-group scale is not changed.
    Nodes observed in only one batch use the median shift learned from
    replicated nodes in the same batch.
    """

    config.output_dir.mkdir(parents=True, exist_ok=True)
    source_tables = config.cytofuse_run / "tables"
    target_tables = config.output_dir / "taxonomy/tables"
    if target_tables.exists():
        shutil.rmtree(target_tables)
    shutil.copytree(source_tables, target_tables)

    data = ad.read_h5ad(config.rna_h5ad)
    data.obs_names = data.obs_names.astype(str)
    cells = pd.read_csv(target_tables / "cell_taxonomy_assignments.csv")
    cells["cell_barcode"] = cells.cell_barcode.astype(str)
    cells = cells.set_index("cell_barcode").reindex(data.obs_names)
    cells.index.name = "cell_barcode"
    if cells.batch.isna().any():
        raise ValueError("taxonomy does not cover all input cells")
    if "assigned_taxonomy_node_id" not in cells:
        cells["assigned_taxonomy_node_id"] = cells["aligned_leaf_id"]
    if "assigned_node_kind" not in cells:
        cells["assigned_node_kind"] = "leaf"
    cells.reset_index().to_csv(
        target_tables / "cell_taxonomy_assignments.csv", index=False
    )

    marker_reference = (
        ad.read_h5ad(config.marker_reference_h5ad)
        if config.marker_reference_h5ad is not None else None
    )
    if marker_reference is not None:
        reference_markers = pd.Index(marker_reference.var_names.astype(str))
        marker_source = str(config.marker_reference_h5ad)
    else:
        states = pd.read_csv(source_tables / "taxonomy_states.csv")
        leaf_states = states.loc[
            states["tree_node_id"].astype(str).str.startswith("ALIGNED_")
        ]
        positive = leaf_states.loc[
            leaf_states["state"].astype(str).eq("+"), "marker"
        ].astype(str)
        if positive.empty:
            positive = leaf_states.loc[
                leaf_states["state"].astype(str).isin(["+", "-"]), "marker"
            ].astype(str)
        reference_markers = pd.Index(sorted(positive.unique()))
        if reference_markers.empty:
            raise ValueError("CytoFuse taxonomy has no informative marker calls")
        marker_source = str(source_tables / "taxonomy_states.csv")
    feature = data.var["feature_types"].astype(str).eq("Antibody Capture").to_numpy()
    protein_names = pd.Index(
        data.var.loc[feature, "feature_name"].astype(str)
        if "feature_name" in data.var
        else data.var_names[feature].astype(str)
    )
    if protein_names.duplicated().any():
        raise ValueError("protein feature names are duplicated")
    selected_names = protein_names if config.all_proteins else reference_markers
    selected_columns = protein_names.get_indexer(selected_names)
    if np.any(selected_columns < 0):
        missing = selected_names[selected_columns < 0].tolist()
        raise KeyError(f"proteins absent from raw data: {missing}")

    counts = dense(data.X[:, feature]).astype(np.float32)
    observed = np.ones(counts.shape, dtype=bool)
    measurement_key = "protein_measurement_mask"
    if measurement_key in data.obsm:
        observed = dense(data.obsm[measurement_key]).astype(bool)
        if observed.shape == data.shape:
            observed = observed[:, feature]
        if observed.shape != counts.shape:
            raise ValueError("protein_measurement_mask must align with proteins or all variables")
        if not observed.any(axis=1).all():
            raise ValueError("every cell must have at least one measured protein")
    log_value = np.log1p(counts)
    if observed.all():
        all_protein_clr = log_value - log_value.mean(axis=1, keepdims=True)
    else:
        center = np.where(observed, log_value, 0).sum(axis=1, keepdims=True) / observed.sum(axis=1, keepdims=True)
        all_protein_clr = np.where(observed, log_value - center, 0).astype(np.float32)
    selected_observed = observed[:, selected_columns]
    original = all_protein_clr[:, selected_columns].copy()
    shifted = original.copy()
    groups = cells[config.group_key].astype(str).to_numpy()
    batches = cells.batch.astype(str).to_numpy()
    direct_shift: dict[tuple[str, str], np.ndarray] = {}
    group_batch_counts: list[dict[str, object]] = []
    for group in sorted(pd.unique(groups)):
        use_group = groups == group
        present = sorted(pd.unique(batches[use_group]))
        centers = {
            batch: _finite_median(np.where(selected_observed[use_group & (batches == batch)], original[use_group & (batches == batch)], np.nan))
            for batch in present
        }
        for batch in present:
            group_batch_counts.append({
                "node": group,
                "batch": batch,
                "n_cells": int(np.sum(use_group & (batches == batch))),
                "node_n_batches": len(present),
            })
        if len(present) < 2:
            continue
        target = _finite_median(np.stack([centers[x] for x in present]))
        for batch in present:
            direct_shift[(group, batch)] = target - centers[batch]

    fallback: dict[str, np.ndarray] = {}
    for batch in sorted(pd.unique(batches)):
        candidates = [
            delta for (_, current_batch), delta in direct_shift.items()
            if current_batch == batch
        ]
        fallback[batch] = (
            np.nan_to_num(_finite_median(np.stack(candidates)), nan=0.0)
            if candidates
            else np.zeros(len(selected_names), dtype=np.float32)
        )

    center_rows: list[dict[str, object]] = []
    for group in sorted(pd.unique(groups)):
        use_group = groups == group
        present = sorted(pd.unique(batches[use_group]))
        replicated = len(present) >= 2
        for batch in present:
            use = use_group & (batches == batch)
            before = _finite_median(np.where(selected_observed[use], original[use], np.nan))
            delta = direct_shift[(group, batch)] if replicated else fallback[batch]
            shifted[use] += np.where(selected_observed[use], np.nan_to_num(delta, nan=0.0), 0)
            after = _finite_median(np.where(selected_observed[use], shifted[use], np.nan))
            for protein_index, protein in enumerate(selected_names):
                center_rows.append({
                    "node": group,
                    "batch": batch,
                    "marker": protein,
                    "n_cells": int(use.sum()),
                    "node_replicated": replicated,
                    "shift_source": "node_batch" if replicated else "batch_fallback",
                    "center_before": float(before[protein_index]),
                    "translation": float(delta[protein_index]),
                    "center_after": float(after[protein_index]),
                })

    obs = data.obs.copy()
    obs["supervision_node"] = groups
    target_data = ad.AnnData(
        X=shifted.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(selected_names, name="protein")),
    )
    target_data.layers["original_clr"] = original.astype(np.float32)
    target_data.layers["normalized_mask"] = selected_observed
    for layer in ("assigned_peak_state", "state_supervision_mask"):
        if (
            not config.all_proteins
            and marker_reference is not None
            and layer in marker_reference.layers
        ):
            target_data.layers[layer] = np.asarray(marker_reference.layers[layer])
    target_data.uns["normalization"] = {
        "method": "all_protein_clr_then_supervision_node_batch_translation",
        "group_key": config.group_key,
        "within_group_scale_changed": False,
        "single_batch_node_policy": "replicated-node-derived batch fallback",
        "marker_source": marker_source,
        "feature_scope": "all_proteins" if config.all_proteins else "marker_subset",
        "direction_marker_names": reference_markers.tolist(),
        "citepool_baseline_version": __version__,
        "measurement_mask_key": measurement_key if measurement_key in data.obsm else None,
        "clr_center_scope": "observed_proteins",
    }
    target_path = config.output_dir / "cluster_translated_marker_targets.h5ad"
    target_data.write_h5ad(target_path, compression="gzip")
    pd.DataFrame(center_rows).to_csv(
        config.output_dir / "translation_centers.csv", index=False
    )
    group_batch = pd.DataFrame(group_batch_counts)
    group_batch.to_csv(config.output_dir / "node_batch_counts.csv", index=False)
    summary = {
        "citepool_baseline_version": __version__,
        "n_cells": int(data.n_obs),
        "n_batches": int(pd.unique(batches).size),
        "n_proteins": int(len(selected_names)),
        "n_direction_markers": int(len(reference_markers)),
        "feature_scope": "all_proteins" if config.all_proteins else "marker_subset",
        "n_supervision_nodes": int(pd.unique(groups).size),
        "n_internal_node_cells": int(
            cells.assigned_node_kind.astype(str).eq("internal").sum()
        ),
        "n_single_batch_node_cells": int(group_batch.loc[
            group_batch.node_n_batches.eq(1), "n_cells"
        ].sum()),
        "decoder_pairs": int(np.prod(target_data.shape)),
        "target": str(target_path),
        "taxonomy": str(config.output_dir / "taxonomy"),
    }
    (config.output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna-h5ad", type=Path, required=True)
    parser.add_argument("--cytofuse-run", type=Path, required=True)
    parser.add_argument(
        "--marker-reference-h5ad", type=Path, default=None,
        help=(
            "Optional frozen marker template. Without it, use markers with a "
            "positive call in the uncut CytoFuse aligned-leaf state table."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-key", default="assigned_taxonomy_node_id")
    parser.add_argument(
        "--all-proteins", action=argparse.BooleanOptionalAction, default=True,
        help="Translate every antibody feature instead of only reference markers.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = prepare_translated_protein_targets(TargetPreparationConfig(
        rna_h5ad=args.rna_h5ad,
        cytofuse_run=args.cytofuse_run,
        marker_reference_h5ad=args.marker_reference_h5ad,
        output_dir=args.output_dir,
        group_key=args.group_key,
        all_proteins=args.all_proteins,
    ))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
