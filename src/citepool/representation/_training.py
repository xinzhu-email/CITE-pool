"""Stable parent-set SCANVI training with interpretable auxiliary heads."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scipy import sparse
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
)
import torch
import torch.nn.functional as F

from scvi.data.fields import ObsmField
from scvi.model import SCVI
from scvi import REGISTRY_KEYS
from scvi.train import SemiSupervisedTrainingPlan

from .. import __version__
from ..config import TrainingConfig
from ._gene_classifier import (
    RNAGeneLinearClassifier,
    compute_log_normalized_gene_stats,
    predict_gene_classifier,
    write_gene_weight_reports,
)
from .._utils.plotting import draw_categorical_umap


UNKNOWN = "Unknown"
PARENT_MASK_KEY = "parent_leaf_mask"
PROTEIN_TARGET_KEY = "protein_target"
PROTEIN_MASK_KEY = "protein_target_mask"
PROTEIN_LOSS_WEIGHT_KEY = "protein_loss_weight"
PARENT_SET_ACTIVE_KEY = "parent_set_active"


class IndependentProteinDecoder(torch.nn.Module):
    """One compact nonlinear regression head per protein."""

    def __init__(self, n_latent: int, hidden_dim: int, n_proteins: int):
        super().__init__()
        self.heads = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(n_latent, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_proteins)
        ])

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(latent) for head in self.heads], dim=-1)


class ParentSetTrainingPlan(SemiSupervisedTrainingPlan):
    """Official SCANVI plan plus parent-set and auxiliary protein losses."""

    def __init__(
        self,
        module,
        n_classes: int,
        *,
        parent_set_ratio: float = 50.0,
        parent_mask_key: str = PARENT_MASK_KEY,
        parent_set_active_key: str = PARENT_SET_ACTIVE_KEY,
        classification_ratio: float = 50.0,
        classification_leaf_weights: list[float] | None = None,
        protein_ratio: float = 0.15,
        protein_point_loss: str = "smooth_l1",
        protein_point_loss_ratio: float = 1.0,
        protein_distribution_loss_ratio: float = 0.0,
        protein_target_key: str = PROTEIN_TARGET_KEY,
        protein_mask_key: str = PROTEIN_MASK_KEY,
        protein_loss_weight_key: str = PROTEIN_LOSS_WEIGHT_KEY,
        rna_linear_classifier_ratio: float = 50.0,
        rna_gene_classifier_ratio: float = 50.0,
        rna_gene_classifier_l1_ratio: float = 0.001,
        **kwargs,
    ):
        super().__init__(module, n_classes, classification_ratio=classification_ratio, **kwargs)
        self.classification_ratio = float(classification_ratio)
        self.parent_set_ratio = float(parent_set_ratio)
        self.parent_mask_key = parent_mask_key
        self.parent_set_active_key = parent_set_active_key
        weights = (
            torch.ones(n_classes, dtype=torch.float32)
            if classification_leaf_weights is None
            else torch.as_tensor(classification_leaf_weights, dtype=torch.float32)
        )
        if len(weights) != n_classes or torch.any(~torch.isfinite(weights)) or torch.any(weights < 1):
            raise ValueError("classification_leaf_weights must contain one finite value >= 1 per class")
        self.register_buffer("classification_leaf_weights", weights)
        self.protein_ratio = float(protein_ratio)
        if protein_point_loss not in {"smooth_l1", "mse"}:
            raise ValueError(
                "protein_point_loss must be 'smooth_l1' or 'mse', got "
                f"{protein_point_loss!r}"
            )
        self.protein_point_loss = protein_point_loss
        self.protein_point_loss_ratio = float(protein_point_loss_ratio)
        self.protein_distribution_loss_ratio = float(
            protein_distribution_loss_ratio
        )
        self.protein_target_key = protein_target_key
        self.protein_mask_key = protein_mask_key
        self.protein_loss_weight_key = protein_loss_weight_key
        self.rna_linear_classifier_ratio = float(rna_linear_classifier_ratio)
        self.rna_gene_classifier_ratio = float(rna_gene_classifier_ratio)
        self.rna_gene_classifier_l1_ratio = float(rna_gene_classifier_l1_ratio)

    def training_step(self, batch, batch_idx):
        """Run the official loss and append the two stable auxiliary losses."""

        if isinstance(batch, dict):
            full_dataset = labelled_dataset = batch
        elif len(batch) == 2:
            full_dataset, labelled_dataset = batch[0], batch[1]
        else:
            full_dataset, labelled_dataset = batch, None

        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        loss_kwargs = {"labelled_tensors": labelled_dataset, **self.loss_kwargs}
        inference_outputs, _, loss_output = self.forward(
            full_dataset, loss_kwargs=loss_kwargs
        )
        loss = loss_output.loss

        # The official objective already includes the ordinary classification
        # loss. Add only (weight - 1) so the requested multiplier is exact.
        weighted_classifier_remainder = loss * 0.0
        if labelled_dataset is not None and torch.any(self.classification_leaf_weights > 1):
            labelled_inference_for_weight = self.module.inference(
                **self.module._get_inference_input(labelled_dataset)
            )
            weighted_labels = labelled_dataset[REGISTRY_KEYS.LABELS_KEY].reshape(-1).long()
            classifier_output = self.module.classifier(labelled_inference_for_weight["z"])
            if self.module.classifier.logits:
                log_probability = F.log_softmax(classifier_output, dim=-1)
            else:
                log_probability = torch.log(classifier_output.clamp_min(1e-8))
            per_cell_nll = F.nll_loss(log_probability, weighted_labels, reduction="none")
            remainder = self.classification_leaf_weights[weighted_labels] - 1.0
            weighted_classifier_remainder = (per_cell_nll * remainder).mean()
            loss = loss + self.classification_ratio * weighted_classifier_remainder

        linear_classifier_loss = loss * 0.0
        if hasattr(self.module, "rna_linear_classifier") and labelled_dataset is not None:
            labelled_inference = self.module.inference(
                **self.module._get_inference_input(labelled_dataset)
            )
            labels = labelled_dataset[REGISTRY_KEYS.LABELS_KEY].reshape(-1).long()
            linear_logits = self.module.rna_linear_classifier(labelled_inference["z"])
            linear_classifier_loss = F.cross_entropy(linear_logits, labels)
            loss = loss + self.rna_linear_classifier_ratio * linear_classifier_loss

        gene_classifier_loss = loss * 0.0
        if hasattr(self.module, "rna_gene_classifier") and labelled_dataset is not None:
            labels = labelled_dataset[REGISTRY_KEYS.LABELS_KEY].reshape(-1).long()
            gene_logits = self.module.rna_gene_classifier(
                labelled_dataset[REGISTRY_KEYS.X_KEY]
            )
            gene_classifier_loss = F.cross_entropy(gene_logits, labels)
            loss = loss + self.rna_gene_classifier_ratio * gene_classifier_loss

        allowed = full_dataset[self.parent_mask_key].bool()
        parent_set_active = full_dataset[self.parent_set_active_key].bool().reshape(-1)
        allowed_count = allowed.sum(dim=-1)
        # A true atlas parent with one surviving classifier descendant still
        # needs a one-class likelihood.  Terminal leaves are excluded via the
        # explicit parent-set activity flag and remain under the official
        # SCANVI classification loss.
        partial = (
            parent_set_active
            & (allowed_count >= 1)
            & (allowed_count < allowed.shape[-1])
        )
        if partial.any():
            classifier_output = self.module.classifier(inference_outputs["z"])
            if self.module.classifier.logits:
                log_probability = F.log_softmax(classifier_output, dim=-1)
            else:
                log_probability = torch.log(classifier_output.clamp_min(1e-8))
            allowed_log_probability = log_probability[partial].masked_fill(
                ~allowed[partial], -torch.inf
            )
            parent_set_loss = -torch.logsumexp(
                allowed_log_probability, dim=-1
            ).mean()
            loss = loss + self.parent_set_ratio * parent_set_loss
            if hasattr(self.module, "rna_linear_classifier"):
                linear_logits = self.module.rna_linear_classifier(inference_outputs["z"])
                linear_log_probability = F.log_softmax(linear_logits, dim=-1)
                linear_allowed = linear_log_probability[partial].masked_fill(
                    ~allowed[partial], -torch.inf
                )
                linear_parent_loss = -torch.logsumexp(
                    linear_allowed, dim=-1
                ).mean()
                loss = loss + self.rna_linear_classifier_ratio * linear_parent_loss
            else:
                linear_parent_loss = loss * 0.0
            if hasattr(self.module, "rna_gene_classifier"):
                gene_logits = self.module.rna_gene_classifier(
                    full_dataset[REGISTRY_KEYS.X_KEY]
                )
                gene_log_probability = F.log_softmax(gene_logits, dim=-1)
                gene_allowed = gene_log_probability[partial].masked_fill(
                    ~allowed[partial], -torch.inf
                )
                gene_parent_loss = -torch.logsumexp(
                    gene_allowed, dim=-1
                ).mean()
                loss = loss + self.rna_gene_classifier_ratio * gene_parent_loss
            else:
                gene_parent_loss = loss * 0.0
        else:
            parent_set_loss = loss * 0.0
            linear_parent_loss = loss * 0.0
            gene_parent_loss = loss * 0.0

        if hasattr(self.module, "rna_gene_classifier"):
            # Mean per-class L1 norm provides gentle sparsity while leaving
            # the official SCANVI encoder and classifier untouched.
            gene_l1_loss = self.module.rna_gene_classifier.linear.weight.abs().sum(
                dim=-1
            ).mean()
            loss = loss + self.rna_gene_classifier_l1_ratio * gene_l1_loss
        else:
            gene_l1_loss = loss * 0.0

        if hasattr(self.module, "protein_decoder"):
            protein_prediction = self.module.protein_decoder(inference_outputs["z"])
            protein_target = full_dataset[self.protein_target_key]
            protein_mask = full_dataset[self.protein_mask_key].bool()
            protein_loss_weight = full_dataset[self.protein_loss_weight_key]
            if self.protein_point_loss == "mse":
                raw_protein_loss = F.mse_loss(
                    protein_prediction, protein_target, reduction="none"
                )
            else:
                raw_protein_loss = F.smooth_l1_loss(
                    protein_prediction, protein_target, reduction="none"
                )
            effective_weight = protein_loss_weight * protein_mask
            protein_point_loss = (
                (raw_protein_loss * effective_weight).sum()
                / effective_weight.sum().clamp_min(1e-8)
            )
            # Empirical 1-Wasserstein distance between prediction and target.
            # It is evaluated separately inside every experimental batch so
            # technical offsets cannot be satisfied by moving one batch onto
            # another.  Sorting is differentiable with respect to its values;
            # no positive/negative threshold or parametric peak model is used.
            if self.protein_distribution_loss_ratio > 0:
                distribution_terms = []
                batch_codes = full_dataset[REGISTRY_KEYS.BATCH_KEY].reshape(-1)
                for batch_code in torch.unique(batch_codes):
                    in_batch = batch_codes == batch_code
                    for protein_index in range(protein_prediction.shape[1]):
                        valid = in_batch & protein_mask[:, protein_index]
                        if valid.sum() < 2:
                            continue
                        predicted_quantiles = torch.sort(
                            protein_prediction[valid, protein_index]
                        ).values
                        target_quantiles = torch.sort(
                            protein_target[valid, protein_index]
                        ).values
                        distribution_terms.append(
                            torch.abs(predicted_quantiles - target_quantiles).mean()
                        )
                if distribution_terms:
                    protein_distribution_loss = torch.stack(
                        distribution_terms
                    ).mean()
                else:
                    protein_distribution_loss = loss * 0.0
            else:
                protein_distribution_loss = loss * 0.0
            protein_loss = (
                self.protein_point_loss_ratio * protein_point_loss
                + self.protein_distribution_loss_ratio
                * protein_distribution_loss
            )
            loss = loss + self.protein_ratio * protein_loss
        else:
            protein_loss = loss * 0.0
            protein_point_loss = loss * 0.0
            protein_distribution_loss = loss * 0.0

        batch_size = loss_output.n_obs_minibatch
        self.log(
            "train_loss", loss, on_epoch=True, batch_size=batch_size, prog_bar=True
        )
        self.log(
            "train_parent_set_loss", parent_set_loss,
            on_epoch=True, batch_size=batch_size,
        )
        self.log(
            "train_weighted_classifier_remainder", weighted_classifier_remainder,
            on_epoch=True, batch_size=batch_size,
        )
        self.log(
            "train_rna_linear_classifier_loss", linear_classifier_loss + linear_parent_loss,
            on_epoch=True, batch_size=batch_size,
        )
        self.log(
            "train_rna_gene_classifier_loss", gene_classifier_loss + gene_parent_loss,
            on_epoch=True, batch_size=batch_size,
        )
        self.log(
            "train_rna_gene_classifier_l1", gene_l1_loss,
            on_epoch=True, batch_size=batch_size,
        )
        self.log(
            "train_protein_loss", protein_loss,
            on_epoch=True, batch_size=batch_size,
        )
        self.log(
            "train_protein_point_loss", protein_point_loss,
            on_epoch=True, batch_size=batch_size,
        )
        self.log(
            "train_protein_distribution_loss", protein_distribution_loss,
            on_epoch=True, batch_size=batch_size,
        )
        self.compute_and_log_metrics(loss_output, self.train_metrics, "train")
        return loss


def taxonomy_masks(
    table_dir: Path,
    cells: pd.DataFrame,
    supervision_node_key: str,
    supervision_kind_key: str,
    *,
    use_resolution_atlas_parent_sets: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Build masks from training supervision, never from the audit tree.

    ``integrated_hierarchy_edges.csv`` is a structural union of taxonomy
    merges and resolution-atlas relations. It is useful for auditing, but
    its edges must not define classifier labels or descendant masks.
    Taxonomy descendants come only from ``taxonomy_edges``. Resolution-atlas
    parents are handled separately by projecting each atlas child to the
    active training label(s) observed for that child.
    """

    edges = pd.read_csv(
        table_dir / "tables/taxonomy_edges.csv", dtype=str
    )[["parent", "child"]].drop_duplicates(["parent", "child"])
    children: dict[str, list[str]] = {}
    for row in edges.itertuples(index=False):
        children.setdefault(str(row.parent), []).append(str(row.child))
    terminal = cells[supervision_kind_key].astype(str).eq("leaf")
    leaves = sorted(
        cells.loc[terminal, supervision_node_key].dropna().astype(str).unique()
    )
    leaf_set = set(leaves)
    memo: dict[str, set[str]] = {}

    def descendants(node: str) -> set[str]:
        if node in memo:
            return memo[node]
        # A taxonomy-merge node is already in the classifier label universe;
        # stop there even if another audit graph contains descendants.
        if node in leaf_set:
            memo[node] = {node}
            return memo[node]

        answer = set()
        for child in children.get(node, []):
            answer.update(descendants(child))
        memo[node] = answer
        return answer

    atlas_parent_sets: dict[str, set[str]] = {}
    atlas_path = table_dir / "tables/resolution_atlas_edges.csv"
    training_label_key = (
        "assigned_taxonomy_node_id"
        if "assigned_taxonomy_node_id" in cells
        else supervision_node_key
    )
    if (
        use_resolution_atlas_parent_sets
        and atlas_path.exists()
        and "aligned_leaf_id" in cells
    ):
        atlas_edges = pd.read_csv(atlas_path, dtype=str)
        if {"parent", "child"}.issubset(atlas_edges.columns):
            aligned = cells["aligned_leaf_id"].astype(str)
            labels = cells[training_label_key]
            for row in atlas_edges[["parent", "child"]].itertuples(index=False):
                # RNA refinement can turn a former classifier leaf into an
                # internal taxonomy node.  Project every label observed for
                # the atlas child through the *current* taxonomy before
                # intersecting with the active classifier leaves.  A direct
                # ``& leaf_set`` silently discarded such internal labels and
                # allowed a single off-type terminal cell to become the
                # parent's only permitted class.
                child_labels: set[str] = set()
                observed = labels.loc[aligned.eq(str(row.child))].dropna()
                for value in observed.astype(str).unique():
                    if value in leaf_set:
                        child_labels.add(value)
                    else:
                        child_labels.update(descendants(value))
                atlas_parent_sets.setdefault(str(row.parent), set()).update(
                    child_labels
                )

    mask = np.ones((len(cells), len(leaves)), dtype=np.float32)
    supervision_kind = cells[supervision_kind_key].astype(str).to_numpy()
    for index, value in enumerate(cells[supervision_node_key]):
        # Cells explicitly marked unlabeled may still retain their original
        # node for protein-target translation/audit, but must not receive a
        # descendant-set constraint in SCANVI.  Keeping the all-ones row
        # makes the parent-set likelihood neutral for these observations.
        if supervision_kind[index] == "unlabeled":
            continue
        if pd.isna(value) or str(value) in {"", "nan", "None", "__ROOT__"}:
            continue
        node = str(value)
        allowed = (
            atlas_parent_sets.get(node, set())
            if node in atlas_parent_sets
            else descendants(node)
        )
        if not allowed:
            # If all active labels under an atlas parent were filtered, keep
            # the neutral mask and skip the parent-set loss rather than
            # inventing a classifier class.
            if node in atlas_parent_sets:
                continue
            raise ValueError(f"supervision node {value!r} has no training descendants")
        observed_allowed = [leaf in allowed for leaf in leaves]
        # A parent may contain only aligned leaves that have no direct cells
        # in the classifier label universe (because those cells were assigned
        # to a broader final TAXON node).  In that case there is no valid
        # descendant class to constrain against; keep the neutral all-ones
        # mask rather than creating an impossible all-zero likelihood.
        if any(observed_allowed):
            mask[index] = observed_allowed
    validate_taxonomy_masks(
        cells,
        leaves,
        mask,
        supervision_node_key,
        supervision_kind_key,
    )
    return leaves, mask


def validate_taxonomy_masks(
    cells: pd.DataFrame,
    leaves: list[str],
    mask: np.ndarray,
    supervision_node_key: str,
    supervision_kind_key: str,
) -> None:
    """Validate the training-semantic leaf and parent mask contract."""

    if len(leaves) != len(set(leaves)):
        raise ValueError("classifier leaf labels are duplicated")
    if mask.shape != (len(cells), len(leaves)):
        raise ValueError(
            "parent mask shape does not match cells and classifier leaves: "
            f"{mask.shape} != {(len(cells), len(leaves))}"
        )
    if np.any(mask.sum(axis=1) == 0):
        raise ValueError("at least one supervision row has an empty allowed set")

    leaf_to_index = {leaf: index for index, leaf in enumerate(leaves)}
    kinds = cells[supervision_kind_key].astype(str).to_numpy()
    nodes = cells[supervision_node_key].astype(str).to_numpy()
    for index, (kind, node) in enumerate(zip(kinds, nodes)):
        if kind != "leaf":
            continue
        if node not in leaf_to_index:
            raise ValueError(
                f"training leaf {node!r} is absent from classifier leaves"
            )
        expected = np.zeros(len(leaves), dtype=bool)
        expected[leaf_to_index[node]] = True
        if not np.array_equal(mask[index].astype(bool), expected):
            raise ValueError(
                f"training leaf {node!r} does not have an exact one-hot mask"
            )


def _apply_resolution_atlas_supervision(
    table_dir: Path,
    cells: pd.DataFrame,
    supervision_node_key: str,
    supervision_kind_key: str,
    *,
    enabled: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Promote atlas-parent cells without consulting the integrated tree.

    The integrated hierarchy is an audit-only union of several relation
    types. Training uses the final taxonomy label as its leaf label and uses
    only the explicit resolution-atlas parent IDs for parent supervision.
    """

    result = cells.copy()
    promoted = np.zeros(len(result), dtype=bool)
    if not enabled:
        result["resolution_atlas_parent_supervision"] = promoted
        return result, promoted

    atlas_path = table_dir / "tables/resolution_atlas_edges.csv"
    if not atlas_path.exists() or "atlas_node_id" not in result.columns:
        result["resolution_atlas_parent_supervision"] = promoted
        return result, promoted
    atlas_edges = pd.read_csv(atlas_path, dtype=str)
    if not {"parent", "child"}.issubset(atlas_edges.columns):
        result["resolution_atlas_parent_supervision"] = promoted
        return result, promoted
    parent_nodes = set(atlas_edges["parent"].dropna().astype(str))
    atlas_nodes = result["atlas_node_id"].astype(str)
    promoted_series = atlas_nodes.isin(parent_nodes)
    # A post-SCANVI RNA refinement is more specific than the older
    # resolution-atlas parent assignment.  Preserve cells that were
    # explicitly resolved from a refinement parent into a new terminal leaf;
    # otherwise the atlas promotion erases their labels before stage-two
    # training and leaves the new classifier head effectively unsupervised.
    if {
        "rna_assignment_level", "rna_refinement_parent_leaf",
        "assigned_taxonomy_node_id",
    }.issubset(result.columns):
        assigned = result["assigned_taxonomy_node_id"].astype(str)
        refinement_parent = result["rna_refinement_parent_leaf"].astype(str)
        refined_terminal = (
            result["rna_assignment_level"].astype(str).eq("terminal_leaf")
            & assigned.ne(refinement_parent)
        )
        promoted_series &= ~refined_terminal
    promoted = promoted_series.to_numpy()
    if promoted.any():
        result.loc[promoted, supervision_node_key] = atlas_nodes.loc[promoted]
        result.loc[promoted, supervision_kind_key] = "internal"
    result["resolution_atlas_parent_supervision"] = promoted
    return result, promoted


def _load_training_data(config: TrainingConfig):
    data = ad.read_h5ad(config.rna_h5ad)
    cells = pd.read_csv(
        config.taxonomy_dir / "tables/cell_taxonomy_assignments.csv",
        dtype={"cell_barcode": str},
    ).set_index("cell_barcode").reindex(data.obs_names.astype(str))
    if cells["batch"].isna().any():
        raise ValueError("taxonomy table does not cover all RNA cells")
    for column in (config.supervision_node_key, config.supervision_kind_key):
        if column not in cells:
            raise KeyError(f"taxonomy table lacks {column!r}")
    cells, atlas_parent_supervision = _apply_resolution_atlas_supervision(
        config.taxonomy_dir,
        cells,
        config.supervision_node_key,
        config.supervision_kind_key,
        enabled=config.use_resolution_atlas_parent_sets,
    )
    leaves, allowed_mask = taxonomy_masks(
        config.taxonomy_dir,
        cells,
        config.supervision_node_key,
        config.supervision_kind_key,
        use_resolution_atlas_parent_sets=config.use_resolution_atlas_parent_sets,
    )

    feature = data.var["feature_types"].astype(str).eq("Gene Expression").to_numpy()
    data = data[:, feature].copy()
    data.X = (
        data.X.tocsr().astype(np.float32)
        if sparse.issparse(data.X)
        else sparse.csr_matrix(data.X, dtype=np.float32)
    )
    data.layers["counts"] = data.X.copy()
    data.obs["batch"] = cells["batch"].astype(str).to_numpy()
    data.obs["known_supervision_node"] = cells[
        config.supervision_node_key
    ].astype(str).to_numpy()
    data.obs["resolution_atlas_parent_supervision"] = atlas_parent_supervision
    supervision_kind = cells[config.supervision_kind_key].astype(str)
    terminal = supervision_kind.eq("leaf").to_numpy()
    fully_unlabeled = supervision_kind.eq("unlabeled").to_numpy()
    assignment_level = np.select(
        [terminal, fully_unlabeled],
        ["terminal_leaf", "fully_unlabeled"],
        default="partial_parent",
    )
    data.obs["assignment_level"] = pd.Categorical(assignment_level)
    data.obsm[PARENT_SET_ACTIVE_KEY] = (
        (assignment_level == "partial_parent").astype(np.float32)[:, None]
    )
    observed = np.full(data.n_obs, UNKNOWN, dtype=object)
    observed[terminal] = cells.loc[
        terminal, config.supervision_node_key
    ].astype(str).to_numpy()
    data.obs["scanvi_label"] = pd.Categorical(observed)
    data.obsm[PARENT_MASK_KEY] = allowed_mask

    if not config.use_label_supervision:
        # Preserve SCANVI's non-empty label universe so the same architecture
        # remains constructible.  The CLI ablation sets all label-dependent
        # loss weights to zero; the protein target and RNA input are unchanged.
        terminal = np.zeros(data.n_obs, dtype=bool)
        assignment_level = np.full(
            data.n_obs, "fully_unlabeled", dtype=object
        )
        data.obs["assignment_level"] = pd.Categorical(assignment_level)
        data.obsm[PARENT_SET_ACTIVE_KEY] = np.zeros(
            (data.n_obs, 1), dtype=np.float32
        )
        allowed_mask = np.ones(
            (data.n_obs, len(leaves)), dtype=np.float32
        )
        data.obsm[PARENT_MASK_KEY] = allowed_mask

    protein_names: list[str] = []
    protein_values = np.zeros((data.n_obs, 0), dtype=np.float32)
    protein_mask = np.zeros((data.n_obs, 0), dtype=np.float32)
    if config.protein_target_h5ad is not None:
        protein = ad.read_h5ad(config.protein_target_h5ad)
        if not data.obs_names.astype(str).equals(protein.obs_names.astype(str)):
            raise ValueError("RNA and protein-target cell orders differ")
        if "normalized_mask" not in protein.layers:
            raise KeyError("protein target requires layers['normalized_mask']")
        protein_values = np.asarray(protein.X, dtype=np.float32)
        protein_mask = np.asarray(protein.layers["normalized_mask"], dtype=np.float32)
        protein_names = protein.var_names.astype(str).tolist()
        protein_loss_weight = np.ones_like(protein_values, dtype=np.float32)
        supervised = protein_mask.astype(bool)
        protein_variance = np.ones(len(protein_names), dtype=np.float32)
        if config.protein_variance_normalize:
            for protein_index in range(len(protein_names)):
                valid = supervised[:, protein_index]
                if valid.any():
                    protein_variance[protein_index] = np.var(
                        protein_values[valid, protein_index], dtype=np.float64
                    )
            protein_variance = np.maximum(
                protein_variance, float(config.protein_variance_epsilon)
            )
            protein_loss_weight *= (1.0 / protein_variance)[None, :]

        unknown_weight_names = sorted(
            set(config.protein_marker_weights).difference(protein_names)
        )
        if unknown_weight_names:
            raise ValueError(
                f"protein_marker_weights contains unknown markers: {unknown_weight_names}"
            )
        manual_marker_weight = np.asarray(
            [float(config.protein_marker_weights.get(name, 1.0)) for name in protein_names],
            dtype=np.float32,
        )
        if np.any(~np.isfinite(manual_marker_weight)) or np.any(manual_marker_weight <= 0):
            raise ValueError("all protein_marker_weights must be finite and positive")
        protein_loss_weight *= manual_marker_weight[None, :]

        cell_weight = np.ones(data.n_obs, dtype=np.float32)
        cell_weight_table = pd.DataFrame()
        if config.protein_celltype_sqrt_balance:
            supervised_cell = supervised.any(axis=1)
            supervision_label = cells[config.supervision_node_key].astype(str).to_numpy()
            counts = pd.Series(supervision_label[supervised_cell]).value_counts()
            raw_type_weight = 1.0 / np.sqrt(counts.astype(float))
            for label, weight in raw_type_weight.items():
                cell_weight[supervised_cell & (supervision_label == label)] = weight
            cell_weight[supervised_cell] /= cell_weight[supervised_cell].mean()
            cell_weight_table = pd.DataFrame({
                "cell_type": counts.index.astype(str),
                "n_supervised_cells": counts.to_numpy(dtype=int),
                "sqrt_balance_weight": [
                    float(cell_weight[
                        supervised_cell & (supervision_label == label)
                    ][0])
                    for label in counts.index
                ],
            })
        protein_loss_weight *= cell_weight[:, None]
        data.obsm[PROTEIN_TARGET_KEY] = protein_values
        data.obsm[PROTEIN_MASK_KEY] = protein_mask
        data.obsm[PROTEIN_LOSS_WEIGHT_KEY] = protein_loss_weight
        data.uns["protein_loss_variance"] = protein_variance
        data.uns["protein_loss_manual_marker_weight"] = manual_marker_weight
        data.uns["protein_loss_celltype_weights"] = cell_weight_table.to_dict("list")
    return (
        data, cells, leaves, allowed_mask, terminal, assignment_level,
        protein_names, protein_values, protein_mask,
    )


def _prediction_metrics(
    data: ad.AnnData,
    prediction: np.ndarray,
    terminal: np.ndarray,
    assignment_level: np.ndarray,
    compatible: np.ndarray,
    allowed_mass: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [
        ("ALL", np.ones(data.n_obs, dtype=bool)),
        ("terminal_leaf", terminal),
        ("partial_parent", assignment_level == "partial_parent"),
        ("fully_unlabeled", assignment_level == "fully_unlabeled"),
    ]
    truth_columns = [
        truth for truth in (
            "celltype_major", "celltype_minor", "cell_type", "subtype"
        ) if truth in data.obs
    ]
    for group, use in groups:
        for truth in truth_columns:
            nonempty = bool(use.any())
            rows.append({
                "group": group,
                "truth": truth,
                "n_cells": int(use.sum()),
                "n_predicted_leaves": int(pd.Series(prediction[use]).nunique()),
                "AMI": (
                    adjusted_mutual_info_score(data.obs.loc[use, truth], prediction[use])
                    if nonempty else np.nan
                ),
                "ARI": (
                    adjusted_rand_score(data.obs.loc[use, truth], prediction[use])
                    if nonempty else np.nan
                ),
                "compatible_fraction": float(compatible[use].mean()) if nonempty else np.nan,
                "mean_allowed_probability": (
                    float(allowed_mass[use].mean()) if nonempty else np.nan
                ),
                "mean_confidence": (
                    float(data.obs.loc[use, "assignment_confidence"].mean())
                    if nonempty else np.nan
                ),
                "mean_entropy": (
                    float(data.obs.loc[use, "assignment_entropy"].mean())
                    if nonempty else np.nan
                ),
            })
    return pd.DataFrame(rows)


def _classifier_comparison_metrics(
    data: ad.AnnData,
    classifiers: dict[str, tuple[np.ndarray, np.ndarray]],
    terminal: np.ndarray,
    assignment_level: np.ndarray,
    allowed_mask: np.ndarray,
    leaves: list[str],
) -> pd.DataFrame:
    """Compare official and auxiliary classifiers on the same cells."""

    rows: list[dict[str, object]] = []
    leaf_to_index = {leaf: index for index, leaf in enumerate(leaves)}
    truth_leaf = data.obs["known_supervision_node"].astype(str).to_numpy()
    groups = [
        ("ALL", np.ones(data.n_obs, dtype=bool)),
        ("terminal_leaf", terminal),
        ("partial_parent", assignment_level == "partial_parent"),
        ("fully_unlabeled", assignment_level == "fully_unlabeled"),
    ]
    truth_columns = [
        key for key in ("celltype_major", "celltype_minor", "cell_type")
        if key in data.obs
    ]
    for classifier, (prediction, probability) in classifiers.items():
        predicted_index = np.asarray([leaf_to_index[value] for value in prediction])
        compatible = allowed_mask[np.arange(data.n_obs), predicted_index].astype(bool)
        allowed_mass = (probability * allowed_mask).sum(axis=1)
        for group, use in groups:
            if not use.any():
                continue
            base = {
                "classifier": classifier,
                "group": group,
                "n_cells": int(use.sum()),
                "n_predicted_leaves": int(pd.Series(prediction[use]).nunique()),
                "leaf_accuracy": (
                    float(np.mean(prediction[use] == truth_leaf[use]))
                    if group == "terminal_leaf" else np.nan
                ),
                "leaf_macro_f1": (
                    float(f1_score(
                        truth_leaf[use], prediction[use], labels=leaves,
                        average="macro", zero_division=0,
                    )) if group == "terminal_leaf" else np.nan
                ),
                "leaf_balanced_accuracy": (
                    float(balanced_accuracy_score(truth_leaf[use], prediction[use]))
                    if group == "terminal_leaf" else np.nan
                ),
                "compatible_fraction": float(compatible[use].mean()),
                "mean_allowed_probability": float(allowed_mass[use].mean()),
                "mean_confidence": float(probability[use].max(axis=1).mean()),
            }
            if truth_columns:
                for truth in truth_columns:
                    base[f"AMI_{truth}"] = adjusted_mutual_info_score(
                        data.obs.loc[use, truth], prediction[use]
                    )
                    base[f"ARI_{truth}"] = adjusted_rand_score(
                        data.obs.loc[use, truth], prediction[use]
                    )
            rows.append(base)
    return pd.DataFrame(rows)


def _write_umap_reports(data: ad.AnnData, truth_columns: list[str], output_dir: Path):
    panels = [
        ("predicted_leaf", "Predicted terminal leaf"),
        *(
            [("rna_gene_predicted_leaf", "Direct RNA linear classifier")]
            if "rna_gene_predicted_leaf" in data.obs else []
        ),
        ("known_supervision_node", "Observed leaf/parent set"),
        *[(truth, truth) for truth in truth_columns],
        ("batch", "Batch"),
        ("assignment_level", "Supervision level"),
    ]
    n_columns = 3
    n_rows = int(np.ceil(len(panels) / n_columns))
    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(25, 7 * n_rows),
        constrained_layout=True, squeeze=False,
    )
    for axis, (column, title) in zip(axes.flat, panels):
        draw_categorical_umap(
            axis, data.obsm["X_umap_scanvi"], data.obs[column], title
        )
    for axis in axes.flat[len(panels):]:
        axis.set_visible(False)
    figure.suptitle(
        "Original SCANVI + descendant-set classifier mask (latent UMAP)", fontsize=18
    )
    figure.savefig(
        output_dir / "official_scanvi_parent_set_umap.png",
        dpi=230, bbox_inches="tight",
    )
    plt.close(figure)

    resolved_panels = [("final_leaf", "Final leaf (all cells)")]
    if "rna_gene_predicted_leaf" in data.obs:
        resolved_panels.append(
            ("rna_gene_predicted_leaf", "Direct RNA linear classifier")
        )
    resolved_panels.extend((truth, truth) for truth in truth_columns[:2])
    resolved_panels.append(("batch", "Batch"))
    n_rows = int(np.ceil(len(resolved_panels) / 2))
    figure, axes = plt.subplots(
        n_rows, 2, figsize=(17, 6 * n_rows),
        constrained_layout=True, squeeze=False,
    )
    for axis, (column, title) in zip(axes.flat, resolved_panels):
        draw_categorical_umap(
            axis, data.obsm["X_umap_scanvi"], data.obs[column], title
        )
    for axis in axes.flat[len(resolved_panels):]:
        axis.set_visible(False)
    figure.suptitle("Resolved SCANVI leaf assignments", fontsize=18)
    figure.savefig(
        output_dir / "final_leaf_umap.png", dpi=230, bbox_inches="tight"
    )
    plt.close(figure)


def train_parent_set_scanvi(config: TrainingConfig) -> dict[str, object]:
    """Train and persist the stable SCVI→SCANVI parent-set baseline."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    scvi.settings.seed = config.seed
    started = time.perf_counter()
    (
        data, _, leaves, allowed_mask, terminal, assignment_level,
        protein_names, protein_values, protein_mask,
    ) = _load_training_data(config)
    input_umap = (
        np.asarray(data.obsm["X_umap"], dtype=np.float32).copy()
        if "X_umap" in data.obsm else None
    )

    sc.pp.filter_genes(data, min_cells=config.gene_min_cells)
    if config.n_hvg >= data.n_vars:
        data.var["highly_variable"] = True
    elif config.hvg_selection_method == "batch_aware":
        sc.pp.highly_variable_genes(
            data, layer="counts", flavor="seurat_v3", n_top_genes=config.n_hvg,
            batch_key="batch", span=0.5, subset=False,
        )
    elif config.hvg_selection_method == "stable_per_batch":
        batch_values = data.obs["batch"].astype(str)
        batch_names = sorted(batch_values.unique())
        if len(batch_names) < 2:
            raise ValueError("stable_per_batch HVG selection requires >=2 batches")
        selected_by_batch = []
        ranks_by_batch = []
        for batch_name in batch_names:
            one_batch = data[batch_values.eq(batch_name).to_numpy()].copy()
            sc.pp.highly_variable_genes(
                one_batch, layer="counts", flavor="seurat_v3",
                n_top_genes=config.n_hvg, span=0.5, subset=False,
            )
            selected_by_batch.append(
                one_batch.var["highly_variable"].to_numpy(dtype=bool)
            )
            ranks_by_batch.append(
                one_batch.var["highly_variable_rank"].to_numpy(dtype=float)
            )
        selected_matrix = np.stack(selected_by_batch, axis=1)
        rank_matrix = np.stack(ranks_by_batch, axis=1)
        n_selected_batches = selected_matrix.sum(axis=1)
        selected_ranks = np.where(selected_matrix, rank_matrix, 0.0)
        has_candidate_rank = n_selected_batches > 0
        worst_selected_rank = np.full(data.n_vars, np.inf, dtype=float)
        mean_selected_rank = np.full(data.n_vars, np.inf, dtype=float)
        worst_selected_rank[has_candidate_rank] = np.where(
            selected_matrix[has_candidate_rank],
            rank_matrix[has_candidate_rank],
            -np.inf,
        ).max(axis=1)
        mean_selected_rank[has_candidate_rank] = (
            selected_ranks[has_candidate_rank].sum(axis=1)
            / n_selected_batches[has_candidate_rank]
        )
        order = np.lexsort((mean_selected_rank, worst_selected_rank, -n_selected_batches))
        chosen = order[: config.n_hvg]
        data.var["highly_variable"] = False
        data.var.iloc[chosen, data.var.columns.get_loc("highly_variable")] = True
        data.var["highly_variable_nbatches"] = n_selected_batches
        data.var["highly_variable_rank"] = np.nan
        data.var.iloc[
            chosen, data.var.columns.get_loc("highly_variable_rank")
        ] = np.arange(config.n_hvg, dtype=float)
        data.var["stable_hvg_worst_batch_rank"] = worst_selected_rank
        data.var["stable_hvg_mean_batch_rank"] = mean_selected_rank
    else:
        raise ValueError(
            "hvg_selection_method must be 'batch_aware' or "
            f"'stable_per_batch', got {config.hvg_selection_method!r}"
        )
    hvg = data.var["highly_variable"].to_numpy()
    pd.Series(data.var_names[hvg].astype(str), name="gene").to_csv(
        config.output_dir / "highly_variable_genes.csv", index=False
    )
    data = data[:, hvg].copy()
    gene_names = data.var_names.astype(str).tolist()
    gene_mean = gene_std = None
    if config.enable_rna_gene_classifier:
        gene_mean, gene_std = compute_log_normalized_gene_stats(
            data.layers["counts"]
        )
        pd.DataFrame({
            "gene": gene_names,
            "log_normalized_mean": gene_mean,
            "log_normalized_std": gene_std,
        }).to_csv(
            config.output_dir / "rna_gene_classifier_normalization.csv",
            index=False,
        )

    SCVI.setup_anndata(
        data, layer="counts", batch_key="batch", labels_key="scanvi_label"
    )
    manager = SCVI._get_most_recent_anndata_manager(data, required=True)
    custom_fields = [
        ObsmField(PARENT_MASK_KEY, PARENT_MASK_KEY),
        ObsmField(PARENT_SET_ACTIVE_KEY, PARENT_SET_ACTIVE_KEY),
    ]
    if protein_names:
        custom_fields.extend([
            ObsmField(PROTEIN_TARGET_KEY, PROTEIN_TARGET_KEY),
            ObsmField(PROTEIN_MASK_KEY, PROTEIN_MASK_KEY),
            ObsmField(PROTEIN_LOSS_WEIGHT_KEY, PROTEIN_LOSS_WEIGHT_KEY),
        ])
    manager.register_new_fields(custom_fields)
    vae = SCVI(data, n_latent=config.n_latent, gene_likelihood="nb")
    vae.train(
        max_epochs=config.scvi_epochs,
        early_stopping=config.early_stopping,
        accelerator=config.accelerator,
        devices=config.devices,
        batch_size=config.batch_size,
    )
    model = scvi.model.SCANVI.from_scvi_model(
        vae, unlabeled_category=UNKNOWN, labels_key="scanvi_label"
    )
    missing_fields = [
        field for field in custom_fields
        if field.registry_key not in model.adata_manager.data_registry
    ]
    if missing_fields:
        model.adata_manager.register_new_fields(missing_fields)
    if protein_names:
        if config.protein_independent_heads:
            model.module.protein_decoder = IndependentProteinDecoder(
                config.n_latent, config.protein_hidden_dim, len(protein_names)
            )
        else:
            model.module.protein_decoder = torch.nn.Sequential(
                torch.nn.Linear(config.n_latent, config.protein_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(config.protein_hidden_dim, len(protein_names)),
            )
    if config.rna_linear_classifier_ratio > 0:
        model.module.rna_linear_classifier = torch.nn.Linear(
            config.n_latent, len(leaves)
        )
    if config.enable_rna_gene_classifier:
        assert gene_mean is not None and gene_std is not None
        model.module.rna_gene_classifier = RNAGeneLinearClassifier(
            len(gene_names), len(leaves), gene_mean, gene_std
        )
    model._training_plan_cls = ParentSetTrainingPlan
    unknown_class_weights = sorted(
        set(config.classification_leaf_weights).difference(leaves)
    )
    if unknown_class_weights:
        raise ValueError(
            f"classification_leaf_weights contains unknown leaves: {unknown_class_weights}"
        )
    classification_leaf_weights = [
        float(config.classification_leaf_weights.get(leaf, 1.0)) for leaf in leaves
    ]
    if any((not np.isfinite(weight)) or weight < 1 for weight in classification_leaf_weights):
        raise ValueError("all classification_leaf_weights must be finite and >= 1")
    model.train(
        max_epochs=config.scanvi_epochs,
        early_stopping=config.early_stopping,
        accelerator=config.accelerator,
        devices=config.devices,
        batch_size=config.batch_size,
        plan_kwargs={
            "classification_ratio": config.classification_ratio,
            "classification_leaf_weights": classification_leaf_weights,
            "parent_set_ratio": config.parent_set_ratio,
            "parent_mask_key": PARENT_MASK_KEY,
            "parent_set_active_key": PARENT_SET_ACTIVE_KEY,
            "protein_ratio": config.protein_ratio,
            "protein_point_loss": config.protein_point_loss,
            "protein_point_loss_ratio": config.protein_point_loss_ratio,
            "protein_distribution_loss_ratio": (
                config.protein_distribution_loss_ratio
            ),
            "protein_target_key": PROTEIN_TARGET_KEY,
            "protein_mask_key": PROTEIN_MASK_KEY,
            "protein_loss_weight_key": PROTEIN_LOSS_WEIGHT_KEY,
            "rna_linear_classifier_ratio": config.rna_linear_classifier_ratio,
            "rna_gene_classifier_ratio": config.rna_gene_classifier_ratio,
            "rna_gene_classifier_l1_ratio": config.rna_gene_classifier_l1_ratio,
        },
    )

    latent = model.get_latent_representation().astype(np.float32)
    unconstrained_prediction = np.asarray(model.predict()).astype(str)
    soft = model.predict(soft=True)
    probability = soft.to_numpy() if isinstance(soft, pd.DataFrame) else np.asarray(soft)
    probability_names = soft.columns.astype(str).tolist() if isinstance(
        soft, pd.DataFrame
    ) else leaves
    if probability_names != leaves:
        reorder = [probability_names.index(leaf) for leaf in leaves]
        probability = probability[:, reorder]

    # Parent cells must resolve only within their effective descendant set.
    # Keep ordinary terminal-leaf predictions unconstrained so the official
    # SCANVI classifier remains the source of their errors/uncertainty.
    parent_active = assignment_level == "partial_parent"
    constrained_probability = probability.copy()
    constrained_probability[parent_active] = np.where(
        allowed_mask[parent_active].astype(bool),
        constrained_probability[parent_active],
        -np.inf,
    )
    constrained_index = np.argmax(constrained_probability, axis=1)
    prediction = unconstrained_prediction.copy()
    valid_parent_rows = parent_active & allowed_mask.any(axis=1)
    prediction[valid_parent_rows] = np.asarray(leaves, dtype=str)[
        constrained_index[valid_parent_rows]
    ]

    protein_prediction = np.zeros_like(protein_values)
    if protein_names:
        decoder = model.module.protein_decoder
        decoder.eval()
        decoder_device = next(decoder.parameters()).device
        with torch.no_grad():
            for start in range(0, len(latent), config.batch_size):
                stop = min(start + config.batch_size, len(latent))
                z = torch.from_numpy(latent[start:stop]).to(decoder_device)
                protein_prediction[start:stop] = decoder(z).cpu().numpy()
        data.obsm["protein_prediction"] = protein_prediction
        data.uns["protein_names"] = protein_names

    if hasattr(model.module, "rna_linear_classifier"):
        linear = model.module.rna_linear_classifier
        linear.eval(); device = next(linear.parameters()).device
        with torch.no_grad():
            logits = linear(torch.from_numpy(latent).to(device)).cpu().numpy()
        linear_probability = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        linear_index = np.argmax(linear_probability, axis=1)
        linear_prediction = np.asarray(leaves, dtype=str)[linear_index]
        linear_prediction[parent_active] = np.asarray(leaves, dtype=str)[
            np.argmax(np.where(allowed_mask[parent_active], linear_probability[parent_active], -np.inf), axis=1)
        ]
        data.obsm["rna_linear_leaf_probability"] = linear_probability.astype(np.float32)
        data.obs["rna_linear_predicted_leaf"] = pd.Categorical(linear_prediction, categories=leaves)
        data.uns["rna_linear_leaf_names"] = leaves

    gene_probability = None
    gene_prediction = None
    if hasattr(model.module, "rna_gene_classifier"):
        gene_classifier = model.module.rna_gene_classifier
        gene_probability = predict_gene_classifier(
            gene_classifier, data.layers["counts"], batch_size=config.batch_size
        )
        gene_prediction = np.asarray(leaves, dtype=str)[
            np.argmax(gene_probability, axis=1)
        ]
        gene_prediction[parent_active] = np.asarray(leaves, dtype=str)[
            np.argmax(
                np.where(
                    allowed_mask[parent_active],
                    gene_probability[parent_active],
                    -np.inf,
                ),
                axis=1,
            )
        ]
        data.obsm["rna_gene_leaf_probability"] = gene_probability
        data.obs["rna_gene_predicted_leaf"] = pd.Categorical(
            gene_prediction, categories=leaves
        )
        data.uns["rna_gene_leaf_names"] = leaves
        report_dir = config.output_dir / "rna_gene_classifier"
        write_gene_weight_reports(
            gene_classifier,
            gene_names,
            leaves,
            report_dir,
            top_n=config.rna_gene_classifier_top_n,
        )
        torch.save({
            "state_dict": gene_classifier.state_dict(),
            "gene_names": gene_names,
            "leaf_names": leaves,
            "normalization": "library_size_1e4_log1p_gene_standardized",
        }, report_dir / "rna_gene_linear_classifier.pt")

    data.obsm["X_scanvi"] = latent
    data.obsm["leaf_probability"] = probability.astype(np.float32)
    data.obsm["leaf_probability_parent_constrained"] = (
        np.where(np.isfinite(constrained_probability), constrained_probability, 0.0)
        .astype(np.float32)
    )
    data.obs["predicted_leaf"] = pd.Categorical(prediction, categories=leaves)
    data.obs["final_leaf"] = pd.Categorical(prediction, categories=leaves)
    data.obs["assignment_confidence"] = probability.max(axis=1).astype(np.float32)
    data.obs["assignment_entropy"] = (
        -(probability * np.log(probability + 1e-8)).sum(axis=1)
    ).astype(np.float32)
    data.uns["leaf_names"] = leaves
    data.uns["citepool_baseline_version"] = __version__
    if config.compute_umap:
        sc.pp.neighbors(data, use_rep="X_scanvi", n_neighbors=30, random_state=config.seed)
        sc.tl.umap(data, min_dist=0.25, random_state=config.seed)
        data.obsm["X_umap_scanvi"] = np.asarray(
            data.obsm["X_umap"], dtype=np.float32
        ).copy()
    if input_umap is not None:
        data.obsm["X_umap_input"] = input_umap

    leaf_to_index = {leaf: index for index, leaf in enumerate(leaves)}
    compatible = np.asarray([
        allowed_mask[index, leaf_to_index[label]] > 0
        for index, label in enumerate(prediction)
    ])
    allowed_mass = (probability * allowed_mask).sum(axis=1)
    metrics = _prediction_metrics(
        data, prediction, terminal, assignment_level, compatible, allowed_mass
    )
    metrics.to_csv(config.output_dir / "metrics.csv", index=False)
    classifier_outputs = {"official_scanvi": (prediction, probability)}
    if gene_probability is not None and gene_prediction is not None:
        classifier_outputs["direct_rna_linear"] = (
            gene_prediction, gene_probability
        )
    classifier_metrics = _classifier_comparison_metrics(
        data,
        classifier_outputs,
        terminal,
        assignment_level,
        allowed_mask,
        leaves,
    )
    classifier_metrics.to_csv(
        config.output_dir / "classifier_comparison_metrics.csv", index=False
    )
    assignment_columns: dict[str, object] = {
        "cell_barcode": data.obs_names.astype(str),
        "final_leaf": prediction,
        "confidence": data.obs["assignment_confidence"].to_numpy(),
        "input_supervision_node": data.obs[
            "known_supervision_node"
        ].astype(str).to_numpy(),
        "resolved_from_parent": assignment_level == "partial_parent",
    }
    if gene_prediction is not None and gene_probability is not None:
        assignment_columns["rna_gene_predicted_leaf"] = gene_prediction
        assignment_columns["rna_gene_confidence"] = gene_probability.max(axis=1)
    pd.DataFrame(assignment_columns).to_csv(
        config.output_dir / "final_leaf_assignments.csv", index=False
    )

    truth_columns = [
        truth for truth in (
            "celltype_major", "celltype_minor", "cell_type", "subtype"
        ) if truth in data.obs
    ]
    if config.compute_umap:
        _write_umap_reports(data, truth_columns, config.output_dir)

    if protein_names:
        variance = np.asarray(data.uns["protein_loss_variance"], dtype=float)
        manual_marker_weight = np.asarray(
            data.uns["protein_loss_manual_marker_weight"], dtype=float
        )
        pd.DataFrame({
            "protein": protein_names,
            "target_variance": variance,
            "inverse_variance_weight": (
                1.0 / variance
                if config.protein_variance_normalize
                else np.ones(len(protein_names))
            ),
            "manual_marker_weight": manual_marker_weight,
            "effective_marker_weight": manual_marker_weight * (
                1.0 / variance
                if config.protein_variance_normalize
                else np.ones(len(protein_names))
            ),
        }).to_csv(
            config.output_dir / "protein_loss_marker_weights.csv", index=False
        )
        celltype_weight_data = data.uns.get("protein_loss_celltype_weights", {})
        if celltype_weight_data:
            pd.DataFrame(celltype_weight_data).to_csv(
                config.output_dir / "protein_loss_celltype_weights.csv", index=False
            )
        torch.save(
            model.module.protein_decoder.state_dict(),
            config.output_dir / "protein_decoder.pt",
        )
        (config.output_dir / "protein_decoder_config.json").write_text(
            json.dumps({
                "architecture": (
                    "independent_heads"
                    if config.protein_independent_heads else "joint_head"
                ),
                "n_latent": config.n_latent,
                "hidden_dim": config.protein_hidden_dim,
                "n_proteins": len(protein_names),
                "protein_names": protein_names,
            }, indent=2) + "\n"
        )
        reconstructed = ad.AnnData(
            X=protein_prediction,
            obs=data.obs.copy(),
            var=pd.DataFrame(index=pd.Index(protein_names, name="protein")),
        )
        reconstructed.layers["target"] = protein_values
        reconstructed.layers["normalized_mask"] = protein_mask.astype(bool)
        reconstructed.uns["citepool_baseline_version"] = __version__
        reconstructed.write_h5ad(
            config.output_dir / "reconstructed_protein.h5ad", compression="gzip"
        )
        delattr(model.module, "protein_decoder")
    # Keep the official SCANVI checkpoint loadable without requiring the
    # optional sidecar class.  The interpretable classifier is saved above.
    if hasattr(model.module, "rna_gene_classifier"):
        delattr(model.module, "rna_gene_classifier")
    if hasattr(model.module, "rna_linear_classifier"):
        delattr(model.module, "rna_linear_classifier")
    model.save(config.output_dir / "scanvi_model", overwrite=True)
    data.write_h5ad(
        config.output_dir / "official_scanvi_parent_set.h5ad", compression="gzip"
    )
    summary = {
        "citepool_baseline_version": __version__,
        "scvi_tools_version": scvi.__version__,
        "modification": (
            "one descendant-set classifier likelihood for taxonomy and "
            "resolution-atlas internal-node cells"
        ),
        "resolution_atlas_parent_sets": bool(
            config.use_resolution_atlas_parent_sets
        ),
        "cluster_label_supervision_enabled": bool(config.use_label_supervision),
        "n_cells": int(data.n_obs),
        "n_batches": int(data.obs["batch"].nunique()),
        "n_terminal_leaves": len(leaves),
        "n_terminal_cells": int(terminal.sum()),
        "n_partial_parent_cells": int((assignment_level == "partial_parent").sum()),
        "n_resolution_atlas_parent_cells": int(
            data.obs["resolution_atlas_parent_supervision"].sum()
        ),
        "n_fully_unlabeled_cells": int((assignment_level == "fully_unlabeled").sum()),
        "classification_ratio": config.classification_ratio,
        "classification_leaf_weights": config.classification_leaf_weights,
        "rna_linear_classifier_enabled": bool(
            config.rna_linear_classifier_ratio > 0
        ),
        "rna_linear_classifier_ratio": config.rna_linear_classifier_ratio,
        "rna_gene_classifier_enabled": config.enable_rna_gene_classifier,
        "rna_gene_classifier_ratio": config.rna_gene_classifier_ratio,
        "rna_gene_classifier_l1_ratio": config.rna_gene_classifier_l1_ratio,
        "rna_gene_classifier_n_genes": (
            len(gene_names) if config.enable_rna_gene_classifier else 0
        ),
        "gene_min_cells": config.gene_min_cells,
        "hvg_selection_method": config.hvg_selection_method,
        "parent_set_ratio": config.parent_set_ratio,
        "protein_ratio": config.protein_ratio,
        "protein_point_loss": config.protein_point_loss,
        "protein_point_loss_ratio": config.protein_point_loss_ratio,
        "protein_distribution_loss_ratio": config.protein_distribution_loss_ratio,
        "protein_variance_normalize": config.protein_variance_normalize,
        "protein_celltype_sqrt_balance": config.protein_celltype_sqrt_balance,
        "protein_marker_weights": config.protein_marker_weights,
        "protein_independent_heads": config.protein_independent_heads,
        "protein_hidden_dim": config.protein_hidden_dim,
        "n_protein_markers": len(protein_names),
        "protein_decoder_uses_batch": False,
        "protein_masked_mae": float(
            np.abs(protein_prediction - protein_values)[
                protein_mask.astype(bool)
            ].mean()
        ) if protein_names else None,
        "runtime_minutes": (time.perf_counter() - started) / 60,
    }
    (config.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna-h5ad", type=Path, required=True)
    parser.add_argument("--taxonomy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-hvg", type=int, default=3000)
    parser.add_argument("--n-latent", type=int, default=32)
    parser.add_argument("--scvi-epochs", type=int, default=100)
    parser.add_argument("--scanvi-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--classification-ratio", type=float, default=50.0)
    parser.add_argument(
        "--rna-linear-classifier-ratio",
        type=float,
        default=0.0,
        help=(
            "Enable the legacy latent linear classifier with a positive loss "
            "weight. The baseline default 0 uses only the official SCANVI "
            "classifier to shape the embedding."
        ),
    )
    parser.add_argument("--rna-gene-classifier-ratio", type=float, default=50.0)
    parser.add_argument("--rna-gene-classifier-l1-ratio", type=float, default=0.001)
    parser.add_argument("--rna-gene-classifier-top-n", type=int, default=15)
    parser.add_argument(
        "--disable-rna-gene-classifier", action="store_true",
        help="Disable the interpretable direct RNA-to-leaf linear classifier.",
    )
    parser.add_argument("--parent-set-ratio", type=float, default=50.0)
    parser.add_argument("--protein-target-h5ad", type=Path, default=None)
    parser.add_argument("--protein-ratio", type=float, default=1.0)
    parser.add_argument("--protein-hidden-dim", type=int, default=256)
    parser.add_argument("--supervision-node-key", default="assigned_taxonomy_node_id")
    parser.add_argument("--supervision-kind-key", default="assigned_node_kind")
    parser.add_argument(
        "--disable-resolution-atlas-parent-sets",
        action="store_true",
        help="Do not use resolution_atlas coarse-to-fine edges as parent sets.",
    )
    parser.add_argument(
        "--disable-cluster-label-supervision",
        action="store_true",
        help=(
            "Ablation: train the embedding without CytoFuse cluster/atlas "
            "labels while retaining the same RNA and protein targets."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = train_parent_set_scanvi(TrainingConfig(
        rna_h5ad=args.rna_h5ad,
        taxonomy_dir=args.taxonomy_dir,
        output_dir=args.output_dir,
        protein_target_h5ad=args.protein_target_h5ad,
        n_hvg=args.n_hvg,
        n_latent=args.n_latent,
        scvi_epochs=args.scvi_epochs,
        scanvi_epochs=args.scanvi_epochs,
        batch_size=args.batch_size,
        classification_ratio=args.classification_ratio,
        rna_linear_classifier_ratio=args.rna_linear_classifier_ratio,
        rna_gene_classifier_ratio=args.rna_gene_classifier_ratio,
        rna_gene_classifier_l1_ratio=args.rna_gene_classifier_l1_ratio,
        rna_gene_classifier_top_n=args.rna_gene_classifier_top_n,
        enable_rna_gene_classifier=not args.disable_rna_gene_classifier,
        parent_set_ratio=args.parent_set_ratio,
        protein_ratio=args.protein_ratio,
        protein_hidden_dim=args.protein_hidden_dim,
        supervision_node_key=args.supervision_node_key,
        supervision_kind_key=args.supervision_kind_key,
        use_resolution_atlas_parent_sets=(
            not args.disable_resolution_atlas_parent_sets
        ),
        use_label_supervision=not args.disable_cluster_label_supervision,
        seed=args.seed,
    ))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
