"""Typed configuration objects for the stable CITEpool baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TargetPreparationConfig:
    """Inputs for cluster-aware, batch-translation protein targets."""

    rna_h5ad: Path
    cytofuse_run: Path
    marker_reference_h5ad: Path | None
    output_dir: Path
    group_key: str = "assigned_taxonomy_node_id"
    all_proteins: bool = True


@dataclass(frozen=True)
class ProteinIdentityConfig:
    """Inputs for label-free protein alignment and the uncut marker tree."""

    input_h5ad: Path
    output_dir: Path
    batch_key: str = "batch"
    batches: tuple[str, ...] | None = None
    normalization: str = "clr"
    initial_resolution: float = 1.0
    seed: int = 0
    device: str = "auto"
    force_fit: bool = False
    extra_args: tuple[str, ...] = ()
    primary_readout: str = "taxonomy"


@dataclass(frozen=True)
class RefinementConfig:
    """Recursive RNA-PCA refinement performed after a complete first model."""

    rna_h5ad: Path
    initial_model_h5ad: Path
    taxonomy_dir: Path
    output_dir: Path
    batch_key: str = "batch"
    parent_leaves: tuple[str, ...] | None = None
    separation_cutoff: float = 0.5
    n_hvg: int = 500
    n_pcs: int = 5
    min_node_cells: int = 50
    min_batch_cells: int = 100
    min_split_batches: int | None = None  # max(1, total dataset batches // 2)
    min_child_cells: int = 50
    small_fragment_max_cells: int = 50
    small_fragment_separation: float = 0.8
    dip_cutoff: float = 0.00495
    partition_cutoff: float = 0.2
    variance_cutoff: float = 0.3
    enable_2d: bool = True
    min_2d_states: int = 2
    max_2d_components: int = 4
    max_2d_balanced_cells: int = 1500
    two_d_posterior_cutoff: float = 0.6
    two_d_bic_gain_per_cell: float = 0.1
    # None uses the same BIC gain threshold as the 2-D candidate gate.
    one_d_bic_gain_per_cell: float | None = None
    two_d_variance_cutoff: float = 0.01
    two_d_max_center_distance: float = 4.0
    two_d_min_match_margin: float = 0.0
    max_depth: int = 8
    seed: int = 2026


@dataclass(frozen=True)
class WorkflowConfig:
    """Configuration for CytoFuse -> CITEpool -> optional RNA refinement."""

    input_h5ad: Path
    output_dir: Path
    marker_reference_h5ad: Path | None = None
    all_proteins: bool = True
    batch_key: str = "batch"
    initial_resolution: float = 1.0
    cytofuse_seed: int = 0
    seed: int = 2026
    device: str = "auto"
    primary_readout: str = "taxonomy"
    enable_rna_refinement: bool = True
    refinement_parent_leaves: tuple[str, ...] | None = None
    cytofuse_extra_args: tuple[str, ...] = ()
    # Optional protein-information filter.  Disabled by default so the
    # published baseline remains unchanged.  When enabled, affected cells
    # keep their protein-target node but are unlabeled for SCANVI.
    unlabel_low_information_clusters: bool = False
    low_information_max_positive_markers: int = 2
    training_overrides: dict[str, object] = field(default_factory=dict)
    refinement_overrides: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RepresentationConfig:
    """Inputs and fixed defaults for the parent-set SCANVI baseline."""

    rna_h5ad: Path
    taxonomy_dir: Path
    output_dir: Path
    protein_target_h5ad: Path | None = None
    n_hvg: int = 3000
    # ``batch_aware`` is the historical Scanpy/Seurat-v3 selection.  The
    # reference-only transfer ablation uses ``stable_per_batch`` to prioritize
    # genes that are independently variable in every reference batch.
    hvg_selection_method: str = "batch_aware"
    gene_min_cells: int = 20
    n_latent: int = 32
    scvi_epochs: int = 100
    scanvi_epochs: int = 50
    batch_size: int = 1024
    classification_ratio: float = 50.0
    # Optional per-leaf multipliers for the official SCANVI classifier loss.
    classification_leaf_weights: dict[str, float] = field(default_factory=dict)
    # Legacy auxiliary classifier on the learned SCANVI latent.  Disabled in
    # the baseline so the official SCANVI classifier is the only classifier
    # that changes the RNA embedding; values > 0 enable the old ablation.
    rna_linear_classifier_ratio: float = 0.0
    # Interpretable direct RNA -> leaf classifier.  Its input is standardized
    # log-normalized HVG expression, so its coefficients map back to genes.
    rna_gene_classifier_ratio: float = 50.0
    rna_gene_classifier_l1_ratio: float = 0.001
    rna_gene_classifier_top_n: int = 15
    enable_rna_gene_classifier: bool = True
    parent_set_ratio: float = 50.0
    protein_ratio: float = 1.0
    # Components inside the auxiliary protein objective.  The historical
    # baseline is recovered by point=1 and distribution=0.
    protein_point_loss: str = "smooth_l1"
    protein_point_loss_ratio: float = 1.0
    protein_distribution_loss_ratio: float = 0.0
    # Optional balancing for continuous protein regression.
    protein_variance_normalize: bool = False
    protein_celltype_sqrt_balance: bool = False
    protein_variance_epsilon: float = 1e-4
    # Optional explicit per-marker multipliers applied after variance scaling.
    protein_marker_weights: dict[str, float] = field(default_factory=dict)
    protein_hidden_dim: int = 256
    # When enabled, each protein gets its own latent -> hidden -> 1 head.
    protein_independent_heads: bool = False
    # Large full-query runs can export latent/protein predictions immediately;
    # UMAP is then run separately on the query-only latent matrix.
    compute_umap: bool = True
    supervision_node_key: str = "assigned_taxonomy_node_id"
    supervision_kind_key: str = "assigned_node_kind"
    # Use CytoFUSE's resolution-aware coarse-to-fine relations as additional
    # parent-set supervision.  This is deliberately separate from the main
    # final-label readout, so the taxonomy merge labels remain reproducible.
    use_resolution_atlas_parent_sets: bool = True
    # Disable cluster/atlas supervision while preserving the model label universe.
    use_label_supervision: bool = True
    seed: int = 2026
    accelerator: str = "gpu"
    devices: int | str = 1
    early_stopping: bool = True


@dataclass(frozen=True)
class BenchmarkConfig:
    """Inputs for the three-experiment section1 benchmark."""

    root: Path
    output_dir: Path
    direction_marker_root: Path | None = None
    data_root: Path = Path("data/section1")
    experiments: tuple[str, ...] = ("expr1", "expr2", "expr3")
    n_jobs: int = 8

# Historical names remain aliases for saved configurations and old scripts.
CytoFuseConfig = ProteinIdentityConfig
TrainingConfig = RepresentationConfig
RNARefinementConfig = RefinementConfig
