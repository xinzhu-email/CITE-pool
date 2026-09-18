"""Public API for the CytoFuse implementation bundled with CITEpool.

Fitting, alignment, final evidence cut, and taxonomy construction all execute
from :mod:`citepool.identity._engine`; no sibling source checkout or
``sys.path`` modification is required.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from ...config import CytoFuseConfig
from . import CYTOFUSE_ENGINE_VERSION
from .integrated_tree import build_integrated_hierarchy


def _source_fingerprints() -> dict[str, str]:
    """Fingerprint the bundled algorithm sources used by this run."""

    root = Path(__file__).resolve().parent
    relative_paths = (
        "runner.py",
        "tree.py",
        "methods/cluster.py",
        "methods/paramshared.py",
    )
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in relative_paths
    }


def run_cytofuse(config: CytoFuseConfig) -> dict[str, object]:
    """Run current CytoFuse and construct the atlas-first CITEpool hierarchy."""

    from .runner import main as bundled_main

    argv = [
        "--input", str(Path(config.input_h5ad)),
        "--output", str(Path(config.output_dir)),
        "--batch-key", config.batch_key,
        "--normalization", config.normalization,
        "--initial-resolution", str(config.initial_resolution),
        "--seed", str(config.seed),
        "--device", config.device,
        "--primary-readout", config.primary_readout,
        # section1 h5ad stores RNA and ADT in one matrix.  The current
        # CytoFuse runner must be told explicitly which features are proteins;
        # otherwise all RNA genes would be interpreted as protein features.
        "--protein-feature-type-key", "feature_types",
        "--protein-feature-type-value", "Antibody Capture",
    ]
    if config.batches:
        argv.extend(["--batches", *map(str, config.batches)])
    if config.force_fit:
        argv.append("--force-fit")
    argv.extend(map(str, config.extra_args))
    bundled_main(argv)

    tables = Path(config.output_dir) / "tables"
    assignments_path = tables / "cell_taxonomy_assignments.csv"
    assignments = pd.read_csv(assignments_path, low_memory=False)
    if "cytofuse_final_id" not in assignments:
        raise KeyError("current CytoFuse output lacks cytofuse_final_id")
    # Baseline training uses these names to construct parent descendant sets.
    # The current final cut is the primary supervision node; aligned leaves
    # remain available as the finer audit label.
    primary_column = (
        "cytofuse_primary_id"
        if "cytofuse_primary_id" in assignments
        else "cytofuse_final_id"
    )
    assignments["assigned_taxonomy_node_id"] = assignments[primary_column].astype(str)
    # ``cytofuse_final_id`` is already the final global evidence cut.  A
    # TAXON_* id here means that its aligned children were merged and are no
    # longer final labels; it is therefore a terminal supervision label.  A
    # genuine parent/child supervision relation is introduced separately by
    # the resolution atlas in ``integrated_tree.py``.
    assignments["assigned_node_kind"] = "leaf"
    assignments.to_csv(assignments_path, index=False)
    integrated_summary = build_integrated_hierarchy(Path(config.output_dir))
    assignments = pd.read_csv(assignments_path, low_memory=False)
    config_path = Path(config.output_dir) / "run_config.json"
    summary_path = Path(config.output_dir) / "run_summary.json"
    summary = {
        "cytofuse_implementation": "citepool_baseline_bundled_cytofuse",
        "cytofuse_engine_version": CYTOFUSE_ENGINE_VERSION,
        "cytofuse_source_sha256": _source_fingerprints(),
        "input": str(Path(config.input_h5ad).resolve()),
        "output": str(Path(config.output_dir).resolve()),
        "initial_resolution": float(config.initial_resolution),
        "n_local_clusters": int(assignments["local_node_id"].nunique()),
        "n_aligned_leaves": int(assignments["aligned_leaf_id"].nunique()),
        "n_final_nodes": int(assignments["assigned_taxonomy_node_id"].nunique()),
        "n_taxonomy_roots": int(assignments["taxonomy_root"].nunique()),
        "known_labels_used_for_construction": False,
        "taxonomy_cut": "current_cytofuse_final_id",
        "assigned_node_key": "assigned_taxonomy_node_id",
        "primary_readout": config.primary_readout,
        "integrated_hierarchy": integrated_summary,
        # The integrated hierarchy below is an audit/visualization tree.
        # CITEpool training consumes the final taxonomy labels and explicit
        # resolution-atlas parent sets instead.
        "citepool_supervision_node_key": "assigned_taxonomy_node_id",
        "citepool_supervision_kind_key": "assigned_node_kind",
    }
    summary_path.write_text(
        __import__("json").dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    """CLI for ``python -m citepool_baseline cytofuse``.

    This invokes the same bundled implementation as the Python API.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--normalization", choices=("clr", "none"), default="clr")
    parser.add_argument("--initial-resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--primary-readout", choices=("taxonomy", "atlas"), default="taxonomy"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force-fit", action="store_true")
    parser.add_argument(
        "--cytofuse-arg", action="append", default=[],
        help="Append one bundled CytoFuse CLI token; repeat for each token.",
    )
    args = parser.parse_args()
    summary = run_cytofuse(CytoFuseConfig(
        input_h5ad=args.input,
        output_dir=args.output,
        batch_key=args.batch_key,
        normalization=args.normalization,
        initial_resolution=args.initial_resolution,
        seed=args.seed,
        device=args.device,
        force_fit=args.force_fit,
        primary_readout=args.primary_readout,
        extra_args=tuple(args.cytofuse_arg),
    ))
    print(__import__("json").dumps(summary, indent=2, ensure_ascii=False))


__all__ = ["run_cytofuse", "main", "build_integrated_hierarchy"]
