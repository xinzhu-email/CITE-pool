#!/usr/bin/env python3
"""Internal protein-marker identity engine for CITEpool."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


HERE = Path(__file__).resolve().parent

from .methods.cluster import summarize_phenotypes
from .methods.paramshared import fit_phenotypes
from .tree import (
    StateThresholds,
    align_clusters,
    build_cell_tree_assignments,
    build_cluster_marker_states,
    consolidate_equivalent_local_clusters,
    build_resolution_aware_atlas,
    build_taxonomy,
    build_hierarchy_relation_graph,
)
DEFAULT_OUTPUT = HERE / "outputs" / "run"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="citepool identity",
        description="Fit protein clusters and build an ambiguity-safe taxonomy."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--batches",
        nargs="+",
        default=None,
        help="Optional batch subset. By default every batch in --batch-key is used.",
    )
    parser.add_argument("--batch-key", default="batch")
    protein_source = parser.add_mutually_exclusive_group()
    protein_source.add_argument(
        "--protein-layer",
        default=None,
        help="Use this AnnData layer instead of X for protein measurements.",
    )
    protein_source.add_argument(
        "--protein-obsm-key",
        default=None,
        help="Use this AnnData obsm matrix for protein measurements.",
    )
    parser.add_argument(
        "--protein-names-key",
        default=None,
        help="AnnData uns key containing marker names for an ndarray obsm matrix.",
    )
    parser.add_argument(
        "--protein-feature-type-key",
        default=None,
        help="Optional adata.var column used to select protein features from X or a layer.",
    )
    parser.add_argument(
        "--protein-feature-type-value",
        default="Antibody Capture",
        help="Value selected from --protein-feature-type-key.",
    )
    parser.add_argument(
        "--protein-mask-obsm-key",
        default=None,
        help="Optional per-cell protein measurement mask; batch-unmeasured markers are excluded.",
    )
    parser.add_argument(
        "--min-protein-measurement-fraction",
        type=float,
        default=0.90,
        help="Minimum within-batch observed fraction for a masked protein marker.",
    )
    parser.add_argument(
        "--normalization",
        choices=("clr", "none"),
        default="clr",
        help="Apply per-cell log1p CLR to nonnegative counts, or use preprocessed values.",
    )
    parser.add_argument(
        "--mosaic-panel-stabilization",
        choices=("auto", "off"),
        default="auto",
        help=(
            "In masked mosaic panels, apply CLR separately to shared and "
            "panel-specific marker blocks and stabilize structural clustering."
        ),
    )
    parser.add_argument(
        "--panel-specific-structural-budget",
        type=float,
        default=0.0,
        help=(
            "Maximum aggregate structural weight of panel-specific markers "
            "relative to the shared-marker block (default: 0; shared backbone)."
        ),
    )
    parser.add_argument(
        "--panel-specific-marker-max-weight",
        type=float,
        default=0.0,
        help=(
            "Maximum structural weight of any individual panel-specific marker; "
            "zero keeps the mosaic initialization on the shared-marker backbone."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--primary-readout",
        choices=("taxonomy", "atlas"),
        default="taxonomy",
        help=(
            "Primary cell label: atlas keeps resolution-aware aligned nodes as "
            "the main readout; taxonomy remains available as a parent-node audit."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Compute backend for phenotype EM; auto uses PyTorch CUDA when available.",
    )
    parser.add_argument("--force-fit", action="store_true")
    parser.add_argument("--max-cells-per-batch", type=int, default=None)
    parser.add_argument(
        "--initial-resolution",
        type=float,
        default=0.5,
        help="Leiden resolution used for batch-local phenotype initialization.",
    )
    parser.add_argument("--max-iter", type=int, default=35)
    parser.add_argument("--warmup-iter", type=int, default=4)
    parser.add_argument("--structural-interval", type=int, default=4)
    parser.add_argument(
        "--freeze-marker-parameters",
        action="store_true",
        help="Ablation: keep Q25/Q75 marker-state parameters fixed during EM.",
    )
    parser.add_argument("--min-component-cells", type=int, default=40)
    parser.add_argument("--min-marker-weight", type=float, default=0.25)
    parser.add_argument(
        "--min-configuration-bhattacharyya",
        type=float,
        default=1.0,
        help="Minimum final Bhattacharyya distance for a marker to emit +/- calls.",
    )
    parser.add_argument("--min-state-confidence", type=float, default=0.60)
    parser.add_argument("--min-state-separation", type=float, default=0.15)
    parser.add_argument("--min-cluster-fraction", type=float, default=0.60)
    parser.add_argument("--min-marker-effect", type=float, default=0.10)
    parser.add_argument("--min-cell-state-probability", type=float, default=0.60)
    parser.add_argument(
        "--trimodal-bic-gain",
        type=float,
        default=20.0,
        help=(
            "Minimum BIC improvement of a three-state marginal marker model "
            "over a two-state model before the middle state is exposed as '?'."
        ),
    )
    parser.add_argument(
        "--trimodal-min-bhattacharyya",
        type=float,
        default=0.25,
        help=(
            "Minimum adjacent-component Bhattacharyya distance for the "
            "three-state marker-call guard."
        ),
    )
    parser.add_argument(
        "--trimodal-min-component-weight",
        type=float,
        default=0.04,
        help="Minimum mixture weight for each component of a trimodal marker.",
    )
    parser.add_argument(
        "--trimodal-markers",
        nargs="+",
        default=["CD4"],
        help=(
            "Markers eligible for the post-hoc trimodal guard (default: CD4). "
            "Use ALL to screen every marker, or NONE to disable trimodal calls. "
            "A listed marker is activated only when the BIC, adjacent-peak "
            "separation, and component-weight criteria are all met."
        ),
    )
    parser.add_argument("--max-markers-per-cluster", type=int, default=40)
    parser.add_argument("--max-markers-per-state", type=int, default=20)
    parser.add_argument("--min-shared-observed", type=int, default=3)
    parser.add_argument("--min-alignment-agreement", type=float, default=0.80)
    parser.add_argument("--min-configuration-coverage", type=float, default=0.45)
    parser.add_argument("--min-leaf-equivalence-coverage", type=float, default=0.55)
    parser.add_argument(
        "--min-alignment-call-confidence",
        type=float,
        default=0.50,
        help=(
            "Downgrade less reliable local +/- calls to unknown for alignment; "
            "the original calls remain in the local configuration table."
        ),
    )
    parser.add_argument("--min-prototype-batches", type=int, default=2)
    parser.add_argument("--max-alignment-neighbors", type=int, default=None)
    parser.add_argument(
        "--max-batch-conflicts",
        type=int,
        default=2,
        help=(
            "Maximum weak +/- conflicts allowed when nearest configurations "
            "come from disjoint batches; strong conflicts are never merged."
        ),
    )
    parser.add_argument(
        "--small-component-max-cells",
        type=int,
        default=100,
        help="Absorb final aligned fragments at or below this size into their nearest large prototype.",
    )
    parser.add_argument(
        "--small-component-target-ratio",
        type=float,
        default=4.0,
        help="Required target-to-fragment cell-count ratio for final fragment absorption.",
    )
    parser.add_argument(
        "--cross-batch-small-component-max-cells",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--same-batch-small-component-target-ratio",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--keep-unmatched-small-components",
        action="store_true",
        help="Deprecated compatibility flag; unmatched fragments are kept by default.",
    )
    parser.add_argument(
        "--force-unmatched-small-components",
        action="store_true",
        help="Ablation: force unmatched small fragments into the nearest admissible prototype.",
    )
    parser.add_argument(
        "--prototype-absorb-max-batches",
        type=int,
        default=2,
        help="Maximum source batches eligible for prototype absorption (default: 2).",
    )
    parser.add_argument(
        "--prototype-absorb-max-cells",
        type=int,
        default=2500,
        help="Maximum cells for prototype absorption (default: 2500).",
    )
    parser.add_argument("--prototype-absorb-min-agreement", type=float, default=0.70)
    parser.add_argument("--prototype-absorb-min-coverage", type=float, default=0.35)
    oversplit_mode = parser.add_mutually_exclusive_group()
    oversplit_mode.add_argument(
        "--enable-oversplit-collapse",
        dest="enable_oversplit_collapse",
        action="store_true",
        help="Opt in to the post-hoc oversplit collapse heuristic.",
    )
    oversplit_mode.add_argument(
        "--disable-oversplit-collapse",
        dest="enable_oversplit_collapse",
        action="store_false",
        help="Compatibility alias; keep oversplit collapse disabled.",
    )
    parser.set_defaults(enable_oversplit_collapse=True)
    parser.add_argument("--oversplit-min-agreement", type=float, default=0.92)
    parser.add_argument("--oversplit-min-shared", type=int, default=5)
    parser.add_argument("--oversplit-min-coverage", type=float, default=0.60)
    parser.add_argument("--min-tree-shared-observed", type=int, default=1)
    parser.add_argument("--min-tree-weighted-shared", type=float, default=0.50)
    parser.add_argument(
        "--consensus-up-level",
        type=int,
        default=2,
        help=(
            "Legacy diagnostic only: move this many steps upward for "
            "cytofuse_consensus_id. The primary cytofuse_final_id is selected "
            "by the global configuration-evidence cut below."
        ),
    )
    parser.add_argument("--final-min-shared-observed", type=int, default=3)
    parser.add_argument("--final-min-common-positive", type=int, default=1)
    parser.add_argument("--final-max-conflicts", type=int, default=1)
    parser.add_argument("--final-max-conflict-fraction", type=float, default=0.10)
    complement_mode = parser.add_mutually_exclusive_group()
    complement_mode.add_argument(
        "--enable-batch-complementary-final-merge",
        dest="enable_batch_complementary_final_merge",
        action="store_true",
        help="Opt in to the final batch-complementary equivalence merge.",
    )
    complement_mode.add_argument(
        "--disable-batch-complementary-final-merge",
        dest="enable_batch_complementary_final_merge",
        action="store_false",
        help="Compatibility alias; keep complementary nodes as atlas relations.",
    )
    parser.set_defaults(enable_batch_complementary_final_merge=True)
    parser.add_argument("--batch-complement-max-conflicts", type=int, default=3)
    parser.add_argument(
        "--batch-complement-max-conflict-fraction",
        type=float,
        default=0.20,
    )
    structural_split_mode = parser.add_mutually_exclusive_group()
    structural_split_mode.add_argument(
        "--enable-structural-split",
        dest="enable_structural_split",
        action="store_true",
        help="Enable EM structural splitting after Leiden initialization.",
    )
    structural_split_mode.add_argument(
        "--disable-structural-split",
        dest="enable_structural_split",
        action="store_false",
        help="Ablation: disable EM structural splitting.",
    )
    parser.set_defaults(enable_structural_split=True)
    parser.add_argument("--minimum-state-specificity", type=float, default=0.10)
    parser.add_argument("--score-absolute-tolerance", type=float, default=0.05)
    parser.add_argument("--score-relative-tolerance", type=float, default=0.02)
    score_mode = parser.add_mutually_exclusive_group()
    score_mode.add_argument(
        "--weighted-scores",
        dest="use_weighted_scores",
        action="store_true",
        help="Use confidence/specificity-weighted alignment as an optional ablation.",
    )
    score_mode.add_argument(
        "--unweighted-scores",
        dest="use_weighted_scores",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(use_weighted_scores=False)
    parser.add_argument(
        "--max-tree-conflicts",
        type=int,
        default=None,
        help="Hard robustness gate; eligible pairs still use the requested positive/negative/conflict priority.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def materialize_input(input_path: Path, cache_dir: Path) -> Path:
    """Return an h5ad path, expanding gzip inputs inside the workflow directory."""

    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input file does not exist: {input_path}")
    if input_path.suffix != ".gz":
        return input_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / input_path.with_suffix("").name
    if output.exists() and output.stat().st_mtime >= input_path.stat().st_mtime:
        return output
    temporary = output.with_suffix(output.suffix + ".partial")
    with gzip.open(input_path, "rb") as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target)
    temporary.replace(output)
    return output


def resolve_batches(
    batch_values: pd.Series,
    requested: Sequence[str] | None,
) -> list[str]:
    """Resolve all or selected batches while preserving input order."""

    if batch_values.isna().any():
        raise ValueError("batch column contains missing values")
    values = batch_values.astype(str)
    if values.str.strip().eq("").any():
        raise ValueError("batch column contains empty labels")
    available = list(pd.unique(values))
    if not available:
        raise ValueError("batch column contains no batches")
    if requested is None:
        return available
    selected = [str(batch) for batch in requested]
    if len(selected) != len(set(selected)):
        raise ValueError("--batches contains duplicate labels")
    missing = [batch for batch in selected if batch not in set(available)]
    if missing:
        raise ValueError(f"unknown batches: {missing}")
    return selected


def make_batch_file_ids(batches: Sequence[str]) -> dict[str, str]:
    """Map arbitrary batch labels to unique, path-safe cache identifiers."""

    output: dict[str, str] = {}
    used: set[str] = set()
    for batch in batches:
        label = str(batch)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._-")
        slug = slug[:80] or "batch"
        candidate = slug
        if candidate != label or candidate in used:
            digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:8]
            candidate = f"{slug}_{digest}"
        while candidate in used:
            digest = hashlib.sha256(
                f"{label}:{len(used)}".encode("utf-8")
            ).hexdigest()[:8]
            candidate = f"{slug}_{digest}"
        output[label] = candidate
        used.add(candidate)
    return output


def protein_matrix_and_names(
    adata_batch,
    *,
    protein_layer: str | None = None,
    protein_obsm_key: str | None = None,
    protein_names_key: str | None = None,
    protein_feature_type_key: str | None = None,
    protein_feature_type_value: str = "Antibody Capture",
    protein_mask_obsm_key: str | None = None,
    min_protein_measurement_fraction: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a validated cells × proteins matrix and marker names."""

    if protein_obsm_key is not None:
        if protein_obsm_key not in adata_batch.obsm:
            raise KeyError(f"missing protein obsm key: {protein_obsm_key}")
        source = adata_batch.obsm[protein_obsm_key]
        if isinstance(source, pd.DataFrame):
            marker_names = np.asarray(source.columns, dtype=str)
            matrix = source.to_numpy()
        else:
            matrix = source
            if protein_names_key is None:
                raise ValueError(
                    "an ndarray protein obsm requires --protein-names-key "
                    "pointing to marker names in adata.uns"
                )
            if protein_names_key not in adata_batch.uns:
                raise KeyError(f"missing protein names uns key: {protein_names_key}")
            marker_names = np.asarray(
                adata_batch.uns[protein_names_key],
                dtype=str,
            )
    else:
        if protein_names_key is not None:
            raise ValueError(
                "--protein-names-key is only valid with --protein-obsm-key"
            )
        if protein_layer is not None:
            if protein_layer not in adata_batch.layers:
                raise KeyError(f"missing protein layer: {protein_layer}")
            matrix = adata_batch.layers[protein_layer]
        else:
            matrix = adata_batch.X
        marker_names = np.asarray(adata_batch.var_names, dtype=str)

        if protein_feature_type_key is not None:
            if protein_feature_type_key not in adata_batch.var:
                raise KeyError(
                    f"missing protein feature-type var key: {protein_feature_type_key}"
                )
            feature_mask = (
                adata_batch.var[protein_feature_type_key].astype(str).to_numpy()
                == str(protein_feature_type_value)
            )
            if not feature_mask.any():
                raise ValueError(
                    "protein feature-type selection retained no markers"
                )
            matrix = matrix[:, feature_mask]
            marker_names = marker_names[feature_mask]

    if sp.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != adata_batch.n_obs:
        raise ValueError("protein data must be a two-dimensional cells × markers matrix")
    if matrix.shape[1] != len(marker_names):
        raise ValueError("protein marker names do not match matrix columns")
    if len(marker_names) == 0 or len(set(marker_names)) != len(marker_names):
        raise ValueError("protein marker names must be non-empty and unique")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("protein matrix contains non-finite values")

    if protein_mask_obsm_key is not None:
        if protein_mask_obsm_key not in adata_batch.obsm:
            raise KeyError(f"missing protein measurement mask: {protein_mask_obsm_key}")
        observed = np.asarray(adata_batch.obsm[protein_mask_obsm_key], dtype=bool)
        if observed.shape[0] != matrix.shape[0]:
            raise ValueError("protein measurement mask has the wrong number of cells")
        if observed.shape[1] == adata_batch.n_vars and matrix.shape[1] != adata_batch.n_vars:
            if protein_feature_type_key is None:
                raise ValueError(
                    "a full-variable protein mask requires --protein-feature-type-key"
                )
            feature_mask = (
                adata_batch.var[protein_feature_type_key].astype(str).to_numpy()
                == str(protein_feature_type_value)
            )
            observed = observed[:, feature_mask]
        if observed.shape != matrix.shape:
            raise ValueError(
                "protein measurement mask must match either selected proteins or all variables"
            )
        minimum = float(min_protein_measurement_fraction)
        if not 0.0 < minimum <= 1.0:
            raise ValueError("min protein measurement fraction must be in (0, 1]")
        retained = observed.mean(axis=0) >= minimum
        if not retained.any():
            raise ValueError("protein measurement mask retained no batch markers")
        matrix = matrix[:, retained].copy()
        observed = observed[:, retained]
        marker_names = marker_names[retained]
        for column in range(matrix.shape[1]):
            missing = ~observed[:, column]
            if missing.any():
                measured = matrix[~missing, column]
                if measured.size == 0:
                    raise ValueError("retained protein marker has no observed values")
                matrix[missing, column] = np.median(measured)
    return matrix, marker_names


def protein_data(adata_batch, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    return protein_matrix_and_names(
        adata_batch,
        protein_layer=args.protein_layer,
        protein_obsm_key=args.protein_obsm_key,
        protein_names_key=args.protein_names_key,
        protein_feature_type_key=args.protein_feature_type_key,
        protein_feature_type_value=args.protein_feature_type_value,
        protein_mask_obsm_key=args.protein_mask_obsm_key,
        min_protein_measurement_fraction=args.min_protein_measurement_fraction,
    )


def normalize_proteins(
    matrix: np.ndarray,
    method: str,
    *,
    clr_groups: np.ndarray | None = None,
) -> np.ndarray:
    if method == "none":
        return np.asarray(matrix, dtype=np.float64)
    if method != "clr":
        raise ValueError(f"unknown protein normalization: {method}")
    if np.any(matrix < 0):
        raise ValueError(
            "CLR normalization requires nonnegative protein values; "
            "use --normalization none for already transformed data"
        )
    logged = np.log1p(matrix)
    if clr_groups is None:
        return logged - logged.mean(axis=1, keepdims=True)
    groups = np.asarray(clr_groups)
    if groups.shape != (matrix.shape[1],):
        raise ValueError("clr_groups must have one value per marker")
    normalized = np.empty_like(logged)
    for group in np.unique(groups):
        selected = groups == group
        normalized[:, selected] = (
            logged[:, selected] - logged[:, selected].mean(axis=1, keepdims=True)
        )
    return normalized


def marker_batch_prevalence(
    adata,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Return the fraction of selected batches measuring each protein marker."""

    if args.protein_mask_obsm_key is None:
        return {}
    if args.protein_obsm_key is not None:
        raise ValueError(
            "automatic mosaic stabilization currently requires var-aligned proteins"
        )
    if args.protein_mask_obsm_key not in adata.obsm:
        raise KeyError(f"missing protein measurement mask: {args.protein_mask_obsm_key}")
    marker_names = np.asarray(adata.var_names, dtype=str)
    feature_mask = np.ones(adata.n_vars, dtype=bool)
    if args.protein_feature_type_key is not None:
        feature_mask = (
            adata.var[args.protein_feature_type_key].astype(str).to_numpy()
            == str(args.protein_feature_type_value)
        )
        marker_names = marker_names[feature_mask]
    observed = np.asarray(adata.obsm[args.protein_mask_obsm_key], dtype=bool)
    if observed.shape[1] == adata.n_vars:
        observed = observed[:, feature_mask]
    if observed.shape[1] != len(marker_names):
        raise ValueError("protein measurement mask is not aligned to selected markers")
    batch_values = adata.obs[args.batch_key].astype(str).to_numpy()
    measured = []
    for batch in args.batches:
        rows = batch_values == str(batch)
        measured.append(
            observed[rows].mean(axis=0) >= float(args.min_protein_measurement_fraction)
        )
    prevalence = np.mean(np.vstack(measured), axis=0)
    return dict(zip(marker_names, prevalence.astype(float)))


def model_paths(model_dir: Path, batch: str) -> list[Path]:
    suffixes = [
        "cell_clusters.csv",
        "cluster_marker_library_valid.csv",
        "marker_params.csv",
        "protein_probabilities.npz",
    ]
    return [model_dir / f"{batch}_{suffix}" for suffix in suffixes]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batch_fit_fingerprint(
    args: argparse.Namespace,
    *,
    batch: str,
    batch_file_id: str,
    input_h5ad: Path,
) -> dict:
    stat = input_h5ad.stat()
    return {
        "model_version": "leiden_structural_em_v2",
        "batch": str(batch),
        "batch_file_id": str(batch_file_id),
        "input": str(input_h5ad.resolve()),
        "input_size": int(stat.st_size),
        "input_mtime_ns": int(stat.st_mtime_ns),
        "seed": int(args.seed),
        "device": str(args.device),
        "max_cells_per_batch": args.max_cells_per_batch,
        "initial_resolution": float(args.initial_resolution),
        "update_marker_parameters": not bool(args.freeze_marker_parameters),
        "protein_layer": args.protein_layer,
        "protein_obsm_key": args.protein_obsm_key,
        "protein_names_key": args.protein_names_key,
        "protein_feature_type_key": args.protein_feature_type_key,
        "protein_feature_type_value": args.protein_feature_type_value,
        "protein_mask_obsm_key": args.protein_mask_obsm_key,
        "min_protein_measurement_fraction": float(
            args.min_protein_measurement_fraction
        ),
        "normalization": args.normalization,
        "mosaic_panel_stabilization": args.mosaic_panel_stabilization,
        "panel_specific_structural_budget": float(
            args.panel_specific_structural_budget
        ),
        "panel_specific_marker_max_weight": float(
            args.panel_specific_marker_max_weight
        ),
        "marker_batch_prevalence": getattr(args, "marker_batch_prevalence", {}),
        "min_configuration_bhattacharyya": float(
            args.min_configuration_bhattacharyya
        ),
        "min_component_cells": int(args.min_component_cells),
        "min_component_fraction": 0.0005,
        "merge_threshold": 0.02,
        "redundant_merge_threshold": 0.025,
        "paramshared_source_sha256": file_sha256(
            HERE / "methods" / "paramshared.py"
        ),
        "cluster_source_sha256": file_sha256(
            HERE / "methods" / "cluster.py"
        ),
    }


def batch_fit_metadata_path(model_dir: Path, batch: str) -> Path:
    return model_dir / f"{batch}_fit_config.json"


def batch_fit_is_reusable(
    model_dir: Path,
    batch: str,
    fingerprint: dict,
) -> bool:
    paths = model_paths(model_dir, batch)
    metadata_path = batch_fit_metadata_path(model_dir, batch)
    if not all(path.exists() and path.stat().st_size > 0 for path in paths):
        return False
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        return False
    try:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(previous.get(key) == value for key, value in fingerprint.items())


def save_batch_fit_metadata(
    model_dir: Path,
    batch: str,
    fingerprint: dict,
) -> None:
    path = batch_fit_metadata_path(model_dir, batch)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(fingerprint, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def fit_one_batch(
    adata,
    batch: str,
    args: argparse.Namespace,
    model_dir: Path,
    batch_file_id: str,
) -> None:
    mask = np.asarray(adata.obs[args.batch_key].astype(str) == batch)
    indices = np.flatnonzero(mask)
    if args.max_cells_per_batch and len(indices) > args.max_cells_per_batch:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(
            rng.choice(indices, size=args.max_cells_per_batch, replace=False)
        )
    if args.protein_feature_type_key is not None:
        if args.protein_feature_type_key not in adata.var:
            raise KeyError(
                f"missing protein feature-type var key: {args.protein_feature_type_key}"
            )
        feature_mask = (
            adata.var[args.protein_feature_type_key].astype(str).to_numpy()
            == str(args.protein_feature_type_value)
        )
        batch_view = adata[indices, feature_mask]
    else:
        batch_view = adata[indices]
    batch_data = batch_view.to_memory()
    matrix, marker_names = protein_data(batch_data, args)
    prevalence_map = getattr(args, "marker_batch_prevalence", {})
    prevalence = np.asarray(
        [float(prevalence_map.get(marker, 1.0)) for marker in marker_names],
        dtype=float,
    )
    shared = np.isclose(prevalence, 1.0)
    stabilize = (
        args.mosaic_panel_stabilization == "auto"
        and args.normalization == "clr"
        and shared.any()
        and (~shared).any()
    )
    clr_groups = np.where(shared, 0, 1) if stabilize else None
    normalized = normalize_proteins(
        matrix,
        args.normalization,
        clr_groups=clr_groups,
    )
    structural_multipliers = np.ones(len(marker_names), dtype=float)
    if stabilize:
        budget = float(args.panel_specific_structural_budget)
        if not np.isfinite(budget) or budget < 0:
            raise ValueError("panel-specific structural budget must be non-negative")
        max_marker_weight = float(args.panel_specific_marker_max_weight)
        if not np.isfinite(max_marker_weight) or max_marker_weight < 0:
            raise ValueError(
                "panel-specific marker max weight must be finite and non-negative"
            )
        specific_count = int((~shared).sum())
        aggregate_multiplier = budget * int(shared.sum()) / max(specific_count, 1)
        multiplier = min(1.0, aggregate_multiplier, max_marker_weight)
        structural_multipliers[~shared] = multiplier
        logging.info(
            "%s panel stabilization: %d shared + %d specific markers; "
            "specific structural multiplier %.4f (aggregate %.4f, cap %.4f)",
            batch,
            int(shared.sum()),
            specific_count,
            multiplier,
            aggregate_multiplier,
            max_marker_weight,
        )
    result = fit_phenotypes(
        normalized,
        marker_names=marker_names,
        initial_resolution=args.initial_resolution,
        min_component_weight=0.025,
        min_component_weight_3=args.trimodal_min_component_weight,
        min_bic_gain=10.0,
        bic_gain_23=args.trimodal_bic_gain,
        trimodal_min_bhattacharyya=args.trimodal_min_bhattacharyya,
        trimodal_markers=args.trimodal_markers,
        min_bhattacharyya=0.08,
        jitter_noise_sd=0.0,
        jitter_mode="none",
        shrinkage=8.0,
        alpha_theta=0.2,
        alpha_pi=0.08,
        max_iter=args.max_iter,
        warmup_iter=args.warmup_iter,
        structural_interval=args.structural_interval,
        min_component_cells=min(args.min_component_cells, len(indices)),
        min_component_fraction=0.0005,
        merge_threshold=0.02,
        redundant_merge_threshold=0.025,
        min_marker_component_weight=0.01,
        min_marker_posterior_confidence=0.60,
        min_configuration_bhattacharyya=args.min_configuration_bhattacharyya,
        random_state=args.seed,
        device=args.device,
        update_marker_parameters=not args.freeze_marker_parameters,
        enable_structural_split=args.enable_structural_split,
        discrimination_reweight=False,
        structural_marker_weight_multipliers=structural_multipliers,
    )
    summarize_phenotypes(
        adata_batch=batch_data,
        result=result,
        marker_names=marker_names,
        batch_id=batch_file_id,
        subtype_key="subtype",
        minor_subset_key="minor_subset",
        batch_key=args.batch_key,
        save_csv=True,
        output_dir=model_dir,
    )
    for suffix in ("cell_clusters.csv", "cluster_marker_library_valid.csv"):
        path = model_dir / f"{batch_file_id}_{suffix}"
        frame = pd.read_csv(path, low_memory=False)
        frame["batch"] = str(batch)
        frame.to_csv(path, index=False)


def build_states_for_batch(
    model_dir: Path,
    batch: str,
    batch_file_id: str,
    thresholds: StateThresholds,
) -> pd.DataFrame:
    cells = pd.read_csv(
        model_dir / f"{batch_file_id}_cell_clusters.csv",
        low_memory=False,
    )
    params = pd.read_csv(
        model_dir / f"{batch_file_id}_marker_params.csv",
        low_memory=False,
    )
    probabilities = np.load(
        model_dir / f"{batch_file_id}_protein_probabilities.npz",
        allow_pickle=True,
    )
    call_n_states = probabilities.get(
        "call_n_states",
        params.get("n_states", pd.Series(2, index=params.index)).astype(int).to_numpy(),
    )
    return build_cluster_marker_states(
        batch_id=batch,
        cluster_labels=cells["protein_phenotype"].astype(str).to_numpy(),
        marker_names=probabilities["marker_names"].astype(str),
        low_prob=probabilities.get("call_low_prob", probabilities["low_prob"]),
        middle_prob=probabilities.get(
            "call_middle_prob", np.zeros_like(probabilities["low_prob"])
        ),
        high_prob=probabilities.get("call_high_prob", probabilities["high_prob"]),
        call_n_states=call_n_states,
        marker_params=params,
        thresholds=thresholds,
    )


def save_outputs(
    output_dir: Path,
    state_table: pd.DataFrame,
    alignment: dict[str, pd.DataFrame],
    taxonomy: dict,
    resolution_atlas: dict,
    cell_assignments: pd.DataFrame,
) -> None:
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    state_table.to_csv(tables / "cluster_marker_states_long.csv", index=False)
    wide = state_table.pivot(
        index=["batch", "cluster", "local_node_id", "n_cells"],
        columns="marker",
        values="state",
    ).reset_index()
    wide.to_csv(tables / "cluster_marker_states_wide.csv", index=False)
    for key, frame in alignment.items():
        frame.to_csv(tables / f"alignment_{key}.csv", index=False)
    for key in ["nodes", "edges", "merges", "states", "pair_audit"]:
        taxonomy[key].to_csv(tables / f"taxonomy_{key}.csv", index=False)
    for key in ["nodes", "edges", "relations", "assignments"]:
        resolution_atlas[key].to_csv(
            tables / f"resolution_atlas_{key}.csv",
            index=False,
        )
    hierarchy = build_hierarchy_relation_graph(taxonomy, resolution_atlas)
    hierarchy["nodes"].to_csv(tables / "hierarchy_nodes.csv", index=False)
    hierarchy["edges"].to_csv(tables / "hierarchy_edges.csv", index=False)
    cell_assignments.to_csv(tables / "cell_taxonomy_assignments.csv", index=False)
    if "cytofuse_final_id" in cell_assignments:
        public_columns = [
            column
            for column in (
                "cell_barcode",
                "batch",
                "assignment_confidence",
                "cytofuse_final_id",
                "cytofuse_atlas_id",
                "cytofuse_primary_id",
                "cytofuse_primary_readout",
                "cytofuse_final_resolution",
                "cytofuse_final_merge_reason",
                "cytofuse_final_rule",
            )
            if column in cell_assignments
        ]
        cell_assignments[public_columns].to_csv(
            tables / "final_cell_assignments.csv.gz",
            index=False,
            compression="gzip",
        )
        final_membership = (
            cell_assignments[
                [
                    "cytofuse_final_id",
                    "cytofuse_atlas_id",
                    "aligned_leaf_id",
                    "local_node_id",
                    "batch",
                    "protein_phenotype",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                ["cytofuse_final_id", "aligned_leaf_id", "batch", "local_node_id"]
            )
            .reset_index(drop=True)
        )
        final_membership.to_csv(tables / "final_membership.csv", index=False)
        final_nodes = (
            cell_assignments.groupby("cytofuse_final_id", sort=True)
            .agg(
                n_cells=("cell_barcode", "size"),
                n_batches=("batch", "nunique"),
                n_local_clusters=("local_node_id", "nunique"),
                n_aligned_leaves=("aligned_leaf_id", "nunique"),
            )
            .reset_index()
        )
        final_reasons = (
            cell_assignments.groupby("cytofuse_final_id")[
                "cytofuse_final_merge_reason"
            ]
            .apply(lambda values: ";".join(sorted(set(values.astype(str)))))
            .rename("merge_reasons")
            .reset_index()
        )
        final_nodes = final_nodes.merge(
            final_reasons,
            on="cytofuse_final_id",
            how="left",
        )
        final_nodes.to_csv(tables / "final_nodes.csv", index=False)
    if "cytofuse_consensus_id" in cell_assignments:
        consensus_membership = (
            cell_assignments[
                [
                    "cytofuse_consensus_id",
                    "aligned_leaf_id",
                    "local_node_id",
                    "batch",
                    "protein_phenotype",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                ["cytofuse_consensus_id", "aligned_leaf_id", "batch", "local_node_id"]
            )
            .reset_index(drop=True)
        )
        consensus_membership.to_csv(
            tables / "consensus_membership.csv",
            index=False,
        )
        consensus_nodes = (
            cell_assignments.groupby("cytofuse_consensus_id", sort=True)
            .agg(
                n_cells=("cell_barcode", "size"),
                n_batches=("batch", "nunique"),
                n_local_clusters=("local_node_id", "nunique"),
                n_aligned_leaves=("aligned_leaf_id", "nunique"),
            )
            .reset_index()
        )
        leaf_desc = (
            consensus_membership.groupby("cytofuse_consensus_id")["aligned_leaf_id"]
            .apply(lambda x: ";".join(sorted(set(map(str, x)))))
            .rename("aligned_leaf_descendants")
            .reset_index()
        )
        consensus_nodes = consensus_nodes.merge(
            leaf_desc,
            on="cytofuse_consensus_id",
            how="left",
        )
        consensus_nodes.to_csv(tables / "consensus_nodes.csv", index=False)
    with (tables / "taxonomy_forest.json").open("w", encoding="utf-8") as handle:
        json.dump(taxonomy["forest"], handle, indent=2, ensure_ascii=False)
    with (tables / "resolution_atlas_forest.json").open("w", encoding="utf-8") as handle:
        json.dump(resolution_atlas["forest"], handle, indent=2, ensure_ascii=False)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    model_dir = args.output / "batch_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    input_h5ad = materialize_input(args.input, args.output / "cache")
    logging.info("Reading %s", input_h5ad)
    # Materialize once before simultaneous batch-row and protein-feature
    # subsetting.  h5py permits only one fancy index on a backed dataset; the
    # CytoFuse workflow legitimately needs both.  Inputs here are routinely
    # small enough after protein selection, while this preserves the same data.
    adata = ad.read_h5ad(input_h5ad)
    if args.batch_key not in adata.obs:
        raise KeyError(f"missing batch column: {args.batch_key}")
    args.batches = resolve_batches(adata.obs[args.batch_key], args.batches)
    batch_file_ids = make_batch_file_ids(args.batches)
    logging.info(
        "Using %d batches from %s: %s",
        len(args.batches),
        args.batch_key,
        ", ".join(args.batches),
    )
    if args.panel_specific_structural_budget < 0:
        raise ValueError("panel-specific structural budget must be non-negative")
    args.marker_batch_prevalence = (
        marker_batch_prevalence(adata, args)
        if args.mosaic_panel_stabilization == "auto"
        else {}
    )
    if args.marker_batch_prevalence:
        n_shared = sum(
            np.isclose(value, 1.0)
            for value in args.marker_batch_prevalence.values()
        )
        logging.info(
            "Mosaic marker support: %d shared across all selected batches, "
            "%d panel-specific",
            int(n_shared),
            int(len(args.marker_batch_prevalence) - n_shared),
        )

    config = vars(args).copy()
    config.update(
        {
            "input": str(args.input),
            "output": str(args.output),
            "materialized_input": str(input_h5ad),
            "batch_file_ids": batch_file_ids,
            "readout_policy": {
                "primary_label": "cytofuse_final_id",
                "primary_is_coarse_main_readout": True,
                "finer_aligned_leaf_is_audit_only": True,
                "ground_truth_used_for_fitting_alignment_or_tree": False,
                "coarse_to_fine_relations_are_not_equivalence_merges": True,
            },
        }
    )
    with (args.output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)

    for batch in args.batches:
        batch_file_id = batch_file_ids[batch]
        fingerprint = batch_fit_fingerprint(
            args,
            batch=batch,
            batch_file_id=batch_file_id,
            input_h5ad=input_h5ad,
        )
        if (
            not args.force_fit
            and batch_fit_is_reusable(
                model_dir,
                batch_file_id,
                fingerprint,
            )
        ):
            logging.info("Reusing completed model for %s", batch)
            continue
        started = time.monotonic()
        n_cells = int((adata.obs[args.batch_key].astype(str) == batch).sum())
        logging.info("Fitting %s (%d cells)", batch, n_cells)
        fit_one_batch(
            adata,
            batch,
            args,
            model_dir,
            batch_file_id,
        )
        save_batch_fit_metadata(
            model_dir,
            batch_file_id,
            fingerprint,
        )
        logging.info("Finished %s in %.1f min", batch, (time.monotonic() - started) / 60)

    thresholds = StateThresholds(
        min_marker_weight=args.min_marker_weight,
        min_state_confidence=args.min_state_confidence,
        min_state_separation=args.min_state_separation,
        min_cluster_fraction=args.min_cluster_fraction,
        min_marker_effect=args.min_marker_effect,
        min_cell_state_probability=args.min_cell_state_probability,
        max_markers_per_cluster=args.max_markers_per_cluster,
        max_markers_per_state=args.max_markers_per_state,
    )
    state_table = pd.concat(
        [
            build_states_for_batch(
                model_dir,
                batch,
                batch_file_ids[batch],
                thresholds,
            )
            for batch in args.batches
        ],
        ignore_index=True,
    )
    state_table, local_consolidation = consolidate_equivalent_local_clusters(
        state_table,
        min_shared_observed=args.min_shared_observed,
        min_call_confidence=args.min_alignment_call_confidence,
    )
    alignment = align_clusters(
        state_table,
        min_shared_observed=args.min_shared_observed,
        minimum_state_specificity=args.minimum_state_specificity,
        score_absolute_tolerance=args.score_absolute_tolerance,
        score_relative_tolerance=args.score_relative_tolerance,
        use_weighted_scores=args.use_weighted_scores,
        min_alignment_agreement=args.min_alignment_agreement,
        min_configuration_coverage=args.min_configuration_coverage,
        min_leaf_equivalence_coverage=args.min_leaf_equivalence_coverage,
        min_alignment_call_confidence=args.min_alignment_call_confidence,
        min_prototype_batches=args.min_prototype_batches,
        max_candidate_neighbors=args.max_alignment_neighbors,
        max_batch_conflicts=args.max_batch_conflicts,
        allow_strong_conflicts=False,
        small_component_max_cells=args.small_component_max_cells,
        small_component_target_ratio=args.small_component_target_ratio,
        cross_batch_small_component_max_cells=(
            args.cross_batch_small_component_max_cells
        ),
        same_batch_small_component_target_ratio=(
            args.same_batch_small_component_target_ratio
        ),
        force_small_component_merge=args.force_unmatched_small_components,
        prototype_absorb_max_batches=args.prototype_absorb_max_batches,
        prototype_absorb_max_cells=args.prototype_absorb_max_cells,
        prototype_absorb_min_agreement=args.prototype_absorb_min_agreement,
        prototype_absorb_min_coverage=args.prototype_absorb_min_coverage,
        collapse_oversplit_components=args.enable_oversplit_collapse,
        oversplit_min_agreement=args.oversplit_min_agreement,
        oversplit_min_shared=args.oversplit_min_shared,
        oversplit_min_coverage=args.oversplit_min_coverage,
    )
    alignment["local_consolidation"] = local_consolidation
    resolution_atlas = build_resolution_aware_atlas(
        alignment["membership"],
        alignment["states"],
        local_state_table=state_table,
        min_shared_observed=args.min_shared_observed,
        minimum_state_specificity=args.minimum_state_specificity,
    )
    taxonomy = build_taxonomy(
        alignment["membership"],
        alignment["states"],
        min_tree_shared_observed=args.min_tree_shared_observed,
        min_tree_weighted_shared=args.min_tree_weighted_shared,
        max_tree_conflicts=args.max_tree_conflicts,
        minimum_state_specificity=args.minimum_state_specificity,
        score_absolute_tolerance=args.score_absolute_tolerance,
        score_relative_tolerance=args.score_relative_tolerance,
        use_weighted_scores=args.use_weighted_scores,
    )
    all_cells = pd.concat(
        [
            pd.read_csv(
                model_dir / f"{batch_file_ids[batch]}_cell_clusters.csv",
                low_memory=False,
            )
            for batch in args.batches
        ],
        ignore_index=True,
    )
    all_cells["protein_phenotype_original"] = all_cells["protein_phenotype"].astype(str)
    original_node = (
        all_cells["batch"].astype(str)
        + "|"
        + all_cells["protein_phenotype_original"]
    )
    consolidated_cluster = local_consolidation.set_index(
        "local_node_id_original"
    )["cluster"].astype(str)
    all_cells["protein_phenotype"] = original_node.map(consolidated_cluster).fillna(
        all_cells["protein_phenotype_original"]
    )
    cell_assignments = build_cell_tree_assignments(
        all_cells,
        alignment["membership"],
        taxonomy["edges"],
        tree_merges=taxonomy["merges"],
        atlas_assignments=resolution_atlas["assignments"],
        consensus_up_level=args.consensus_up_level,
        final_min_shared_observed=args.final_min_shared_observed,
        final_min_common_positive=args.final_min_common_positive,
        final_max_conflicts=args.final_max_conflicts,
        final_max_conflict_fraction=args.final_max_conflict_fraction,
        merge_batch_complementary_children=args.enable_batch_complementary_final_merge,
        batch_complement_max_conflicts=args.batch_complement_max_conflicts,
        batch_complement_max_conflict_fraction=(
            args.batch_complement_max_conflict_fraction
        ),
        primary_readout=args.primary_readout,
    )
    save_outputs(
        args.output,
        state_table,
        alignment,
        taxonomy,
        resolution_atlas,
        cell_assignments,
    )
    adata.file.close()
    logging.info(
        "Complete: %d local clusters -> %d aligned configurations -> %d roots",
        state_table["local_node_id"].nunique(),
        alignment["membership"]["aligned_node_id"].nunique(),
        len(taxonomy["forest"]["roots"]),
    )
    logging.info("Outputs: %s", args.output / "tables")


if __name__ == "__main__":
    main()
