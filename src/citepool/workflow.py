"""End-to-end stable CITEpool workflow orchestration."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
from typing import Any

from . import __version__
from .config import (
    CytoFuseConfig,
    RNARefinementConfig,
    TargetPreparationConfig,
    TrainingConfig,
    WorkflowConfig,
)


def _configured_training(
    *,
    input_h5ad: Path,
    taxonomy_dir: Path,
    output_dir: Path,
    protein_target_h5ad: Path,
    seed: int,
    overrides: dict[str, object],
) -> TrainingConfig:
    valid = {item.name for item in fields(TrainingConfig)}
    fixed = {"rna_h5ad", "taxonomy_dir", "output_dir", "protein_target_h5ad"}
    unknown = set(overrides) - valid
    forbidden = set(overrides) & fixed
    if unknown or forbidden:
        raise ValueError(
            f"invalid training_overrides; unknown={sorted(unknown)}, "
            f"workflow-owned={sorted(forbidden)}"
        )
    values: dict[str, Any] = {
        "rna_h5ad": input_h5ad,
        "taxonomy_dir": taxonomy_dir,
        "output_dir": output_dir,
        "protein_target_h5ad": protein_target_h5ad,
        "seed": seed,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def run_workflow(config: WorkflowConfig) -> dict[str, object]:
    """Run CytoFuse, one complete CITEpool model, and optional refinement."""

    # Keep stage-specific heavy dependencies lazy so configuration and --help
    # remain usable before an optional stage's dependencies are installed.
    from .identity import infer_identity as run_cytofuse
    from .identity import prepare_protein_targets as prepare_translated_protein_targets
    from .representation import learn_representation as train_parent_set_scanvi

    config.output_dir.mkdir(parents=True, exist_ok=True)
    cytofuse_dir = config.output_dir / "cytofuse"
    prepared_dir = config.output_dir / "prepared"
    initial_dir = config.output_dir / "citepool_initial"

    cytofuse_summary = run_cytofuse(CytoFuseConfig(
        input_h5ad=config.input_h5ad,
        output_dir=cytofuse_dir,
        batch_key=config.batch_key,
        normalization="clr",
        initial_resolution=config.initial_resolution,
        seed=config.cytofuse_seed,
        device=config.device,
        primary_readout=config.primary_readout,
        extra_args=config.cytofuse_extra_args,
    ))
    low_information_summary = {
        "enabled": False,
        "n_filtered_clusters": 0,
        "n_filtered_cells": 0,
    }
    if config.unlabel_low_information_clusters:
        from .identity._supervision import mark_low_information_clusters

        low_information_summary = mark_low_information_clusters(
            cytofuse_dir,
            max_positive_markers=config.low_information_max_positive_markers,
        )
    target_summary = prepare_translated_protein_targets(TargetPreparationConfig(
        rna_h5ad=config.input_h5ad,
        cytofuse_run=cytofuse_dir,
        marker_reference_h5ad=config.marker_reference_h5ad,
        output_dir=prepared_dir,
        # Protein targets are part of training and therefore use the
        # training taxonomy. The integrated hierarchy is structural/audit
        # only and must not change target grouping.
        group_key="assigned_taxonomy_node_id",
        all_proteins=config.all_proteins,
    ))
    protein_target = Path(str(target_summary["target"]))
    initial_training = _configured_training(
        input_h5ad=config.input_h5ad,
        taxonomy_dir=prepared_dir / "taxonomy",
        output_dir=initial_dir,
        protein_target_h5ad=protein_target,
        seed=config.seed,
        overrides=config.training_overrides,
    )
    initial_summary = train_parent_set_scanvi(initial_training)

    refinement_summary = None
    refined_summary = None
    final_model_dir = initial_dir
    final_taxonomy_dir = prepared_dir / "taxonomy"
    if config.enable_rna_refinement:
        from .refinement import refine_identity as refine_taxonomy_with_rna

        refinement_dir = config.output_dir / "rna_refinement"
        refinement_values: dict[str, Any] = {
            "rna_h5ad": config.input_h5ad,
            "initial_model_h5ad": initial_dir / "official_scanvi_parent_set.h5ad",
            "taxonomy_dir": prepared_dir / "taxonomy",
            "output_dir": refinement_dir,
            "batch_key": config.batch_key,
            "parent_leaves": config.refinement_parent_leaves,
            "seed": config.seed,
        }
        valid = {item.name for item in fields(RNARefinementConfig)}
        fixed = {
            "rna_h5ad", "initial_model_h5ad", "taxonomy_dir", "output_dir",
            "batch_key", "parent_leaves",
        }
        unknown = set(config.refinement_overrides) - valid
        forbidden = set(config.refinement_overrides) & fixed
        if unknown or forbidden:
            raise ValueError(
                f"invalid refinement_overrides; unknown={sorted(unknown)}, "
                f"workflow-owned={sorted(forbidden)}"
            )
        refinement_values.update(config.refinement_overrides)
        refinement_summary = refine_taxonomy_with_rna(
            RNARefinementConfig(**refinement_values)
        )
        final_model_dir = config.output_dir / "citepool_refined"
        final_taxonomy_dir = refinement_dir
        refined_summary = train_parent_set_scanvi(_configured_training(
            input_h5ad=config.input_h5ad,
            taxonomy_dir=final_taxonomy_dir,
            output_dir=final_model_dir,
            protein_target_h5ad=protein_target,
            seed=config.seed,
            overrides=config.training_overrides,
        ))

    summary = {
        "citepool_baseline_version": __version__,
        "workflow": (
            "cytofuse -> structural_hierarchy_audit -> citepool -> "
            "rna_refinement -> citepool"
            if config.enable_rna_refinement
            else "cytofuse -> structural_hierarchy_audit -> citepool"
        ),
        "known_labels_used_for_cytofuse_or_tree": False,
        "taxonomy_policy": (
            "training_taxonomy_plus_explicit_resolution_atlas_parent_sets;"
            "integrated_hierarchy_audit_only"
        ),
        "rna_refinement_enabled": config.enable_rna_refinement,
        "cytofuse": cytofuse_summary,
        "low_information_supervision_filter": low_information_summary,
        "target_preparation": target_summary,
        "initial_citepool": initial_summary,
        "rna_refinement": refinement_summary,
        "refined_citepool": refined_summary,
        "final_model_dir": str(final_model_dir),
        "final_taxonomy_dir": str(final_taxonomy_dir),
    }
    (config.output_dir / "workflow_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    serializable_config = asdict(config)
    (config.output_dir / "workflow_config.json").write_text(
        json.dumps(serializable_config, indent=2, default=str, ensure_ascii=False)
        + "\n"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker-reference-h5ad", type=Path)
    parser.add_argument("--all-proteins", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--initial-resolution", type=float, default=1.0)
    parser.add_argument("--cytofuse-seed", type=int, default=0)
    parser.add_argument(
        "--unlabel-low-information-clusters", action="store_true",
        help=(
            "Keep low-positive-marker clusters for protein decoding but make "
            "their SCANVI labels unknown."
        ),
    )
    parser.add_argument(
        "--low-information-max-positive-markers", type=int, default=2,
        help="Maximum positive marker calls for the optional unlabeled filter.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--primary-readout", choices=("taxonomy", "atlas"), default="taxonomy",
        help="First-pass CytoFuse readout used as the RNA refinement parent set.",
    )
    parser.add_argument("--rna-refinement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refinement-parent-leaves", nargs="+")
    parser.add_argument("--scvi-epochs", type=int, default=100)
    parser.add_argument("--scanvi-epochs", type=int, default=50)
    parser.add_argument(
        "--rna-linear-classifier-ratio",
        type=float,
        default=0.0,
        help=(
            "Positive values enable the legacy latent linear classifier; "
            "the baseline default uses only the official SCANVI classifier."
        ),
    )
    parser.add_argument("--rna-gene-classifier-ratio", type=float, default=50.0)
    parser.add_argument("--rna-gene-classifier-l1-ratio", type=float, default=0.001)
    parser.add_argument("--rna-gene-classifier-top-n", type=int, default=15)
    parser.add_argument(
        "--disable-rna-gene-classifier", action="store_true",
        help="Disable the direct interpretable RNA-to-leaf linear classifier.",
    )
    parser.add_argument("--protein-ratio", type=float, default=1.0)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument(
        "--cytofuse-arg", action="append", default=[],
        help="Append one advanced CytoFuse CLI token; repeat for each token.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    devices: int | str = int(args.devices) if args.devices.isdigit() else args.devices
    summary = run_workflow(WorkflowConfig(
        input_h5ad=args.input,
        output_dir=args.output,
        marker_reference_h5ad=args.marker_reference_h5ad,
        all_proteins=args.all_proteins,
        batch_key=args.batch_key,
        initial_resolution=args.initial_resolution,
        cytofuse_seed=args.cytofuse_seed,
        unlabel_low_information_clusters=args.unlabel_low_information_clusters,
        low_information_max_positive_markers=args.low_information_max_positive_markers,
        seed=args.seed,
        device=args.device,
        primary_readout=args.primary_readout,
        enable_rna_refinement=args.rna_refinement,
        refinement_parent_leaves=(
            tuple(args.refinement_parent_leaves)
            if args.refinement_parent_leaves else None
        ),
        cytofuse_extra_args=tuple(args.cytofuse_arg),
        training_overrides={
            "scvi_epochs": args.scvi_epochs,
            "scanvi_epochs": args.scanvi_epochs,
            "rna_linear_classifier_ratio": args.rna_linear_classifier_ratio,
            "rna_gene_classifier_ratio": args.rna_gene_classifier_ratio,
            "rna_gene_classifier_l1_ratio": args.rna_gene_classifier_l1_ratio,
            "rna_gene_classifier_top_n": args.rna_gene_classifier_top_n,
            "enable_rna_gene_classifier": not args.disable_rna_gene_classifier,
            "protein_ratio": args.protein_ratio,
            "accelerator": args.accelerator,
            "devices": devices,
        },
    ))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
