"""Recursive node-local RNA-PCA refinement without pseudo-marker transfer.

The splitter first applies the legacy one-dimensional bimodality gate.  When
that gate is positive, it tests a node-local PC1/PC2 model in each sufficiently
large batch, chooses the component count by BIC, aligns component centres
across batches, and emits recurrent ``--``, ``-+``, ``+-`` and ``++`` states.
If the two-dimensional model is not reproducible, it falls back to a binary
GMM on one PC.  Unsupported batches stay at the parent and small isolated
fragments are exported as unlabeled cells for the later SCANVI stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import json
import pickle
from itertools import combinations, permutations
from pathlib import Path
import shutil
from typing import Dict, Optional

import anndata
import diptest
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture

from ..config import RNARefinementConfig


@dataclass
class RecursiveRNAPCANode:
    depth: int
    indices: Dict[str, list[str]]
    split_feature: str = "leaf"
    split_batches: list[str] = field(default_factory=list)
    unresolved_indices: Dict[str, list[str]] = field(default_factory=dict)
    unlabeled_indices: Dict[str, list[str]] = field(default_factory=dict)
    hvg_genes: list[str] = field(default_factory=list)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    aggregate_scores: dict[str, float] = field(default_factory=dict)
    split_mode: str = "leaf"
    split_features: list[str] = field(default_factory=list)
    multistate_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    children: Dict[str, "RecursiveRNAPCANode"] = field(default_factory=dict)
    left: Optional["RecursiveRNAPCANode"] = None
    right: Optional["RecursiveRNAPCANode"] = None

    @property
    def is_split(self) -> bool:
        return bool(self.children) or (
            self.left is not None and self.right is not None
        )

    def child_items(self):
        """Return labelled children while preserving binary-tree support."""
        if self.children:
            return list(self.children.items())
        result = []
        if self.left is not None:
            result.append(("-", self.left))
        if self.right is not None:
            result.append(("+", self.right))
        return result


def _bhattacharyya_separation(gmm: GaussianMixture) -> float:
    mu1, mu2 = gmm.means_[0], gmm.means_[1]
    sigma1, sigma2 = gmm.covariances_[0], gmm.covariances_[1]
    sigma = (sigma1 + sigma2) / 2
    d1 = distance.mahalanobis(mu1, mu2, np.linalg.inv(sigma)) ** 2 / 8
    d2 = (
        0.5 * np.log(np.linalg.det(sigma))
        - 0.25 * np.log(np.linalg.det(sigma1))
        - 0.25 * np.log(np.linalg.det(sigma2))
    )
    return float(1 - np.exp(-(d1 + d2)))


def _node_pca(
    rnadata: Dict[str, anndata.AnnData],
    n_hvg: int,
    n_pcs: int,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    pieces = []
    for batch, adata in rnadata.items():
        if adata.n_obs == 0:
            continue
        current = adata.copy()
        current.obs["_rna_split_batch"] = str(batch)
        pieces.append(current)
    if not pieces:
        return pd.DataFrame(), []
    combined = anndata.concat(pieces, join="inner", merge="same", index_unique=None)
    combined.obs["_rna_split_batch"] = pd.Categorical(combined.obs["_rna_split_batch"])
    sc.pp.highly_variable_genes(
        combined,
        n_top_genes=min(n_hvg, combined.n_vars),
        batch_key="_rna_split_batch",
    )
    hvg = combined.var_names[combined.var["highly_variable"]].astype(str).tolist()
    if len(hvg) < 2 or combined.n_obs < 3:
        return pd.DataFrame(), hvg
    pca_input = combined[:, hvg].copy()
    sc.pp.scale(pca_input)
    components = min(n_pcs, pca_input.n_obs - 1, pca_input.n_vars - 1)
    if components < 1:
        return pd.DataFrame(), hvg
    sc.tl.pca(pca_input, n_comps=components, random_state=seed)
    pca = pd.DataFrame(
        np.asarray(pca_input.obsm["X_pca"], dtype=np.float64),
        index=pca_input.obs_names.astype(str),
        columns=[f"CC_{index + 1}" for index in range(components)],
    )
    return pca, hvg


def _score_pcs(
    pca: pd.DataFrame,
    rnadata: Dict[str, anndata.AnnData],
    separation_cutoff: float,
    dip_cutoff: float,
    partition_cutoff: float,
    variance_cutoff: float,
    min_batch_cells: int,
    seed: int,
    bic_gain_per_cell: float = 0.1,
) -> pd.DataFrame:
    if not np.isfinite(bic_gain_per_cell) or bic_gain_per_cell < 0:
        raise ValueError("bic_gain_per_cell must be finite and nonnegative")
    rows = []
    for batch, adata in rnadata.items():
        cells = adata.obs_names.astype(str)
        if len(cells) < min_batch_cells:
            continue
        for pc in pca.columns:
            values = pca.loc[cells, [pc]].to_numpy(dtype=np.float64)
            dip = float(diptest.dipstat(values[:, 0]))
            gmm = GaussianMixture(n_components=2, random_state=seed).fit(values)
            single_gmm = GaussianMixture(n_components=1, random_state=seed).fit(values)
            bic_single = float(single_gmm.bic(values))
            bic_split = float(gmm.bic(values))
            bic_gain = (bic_single - bic_split) / len(cells)
            separation = _bhattacharyya_separation(gmm)
            weights = gmm.weights_
            partition = float(-(weights * np.log2(weights + 1e-12)).sum())
            prediction = gmm.predict(values)
            component_variances = [
                float(values[prediction == component, 0].var())
                if int((prediction == component).sum()) > 1
                else 0.0
                for component in range(2)
            ]
            min_variance = min(component_variances)
            fit = float(gmm.score(values))
            passed = bool(
                dip >= dip_cutoff
                and separation >= separation_cutoff
                and partition >= partition_cutoff
                and min_variance >= variance_cutoff
                and bic_gain >= bic_gain_per_cell
            )
            score = (
                fit * 0.2
                + separation
                + partition * 0.5
                + min_variance * 0.1
            )
            rows.append(
                {
                    "batch": str(batch),
                    "feature": pc,
                    "n_cells": int(len(cells)),
                    "dip": dip,
                    "separation": separation,
                    "partition": partition,
                    "min_variance": min_variance,
                    "gmm_fit": fit,
                    "bic_single": bic_single,
                    "bic_split": bic_split,
                    "bic_gain_per_cell": bic_gain,
                    "bic_gain_cutoff": float(bic_gain_per_cell),
                    "bic_passed": bool(bic_gain >= bic_gain_per_cell),
                    "score": score,
                    "passed": passed,
                    "mean_low": float(gmm.means_[:, 0].min()),
                    "mean_high": float(gmm.means_[:, 0].max()),
                }
            )
    return pd.DataFrame(rows)


def _valid_gmm(
    values: np.ndarray,
    n_components: int,
    seed: int,
    min_component_cells: int,
) -> tuple[GaussianMixture, np.ndarray, float] | None:
    """Fit a GMM only when every component has enough cells."""
    if values.shape[0] < n_components * min_component_cells:
        return None
    model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        reg_covar=1e-5,
        n_init=1,
        max_iter=200,
        random_state=seed,
    ).fit(values)
    prediction = model.predict(values)
    counts = np.bincount(prediction, minlength=n_components)
    if int(counts.min()) < min_component_cells:
        return None
    return model, prediction, float(model.bic(values))


def _multicomponent_quality(model: GaussianMixture) -> dict[str, float]:
    """Compute separation/partition/variance checks for a 2-D GMM."""
    separations = []
    for first, second in combinations(range(model.n_components), 2):
        mu1, mu2 = model.means_[first], model.means_[second]
        sigma1, sigma2 = model.covariances_[first], model.covariances_[second]
        sigma = (sigma1 + sigma2) / 2
        delta = mu1 - mu2
        d1 = float(delta @ np.linalg.pinv(sigma) @ delta) / 8
        sign, logdet = np.linalg.slogdet(sigma)
        sign1, logdet1 = np.linalg.slogdet(sigma1)
        sign2, logdet2 = np.linalg.slogdet(sigma2)
        if min(sign, sign1, sign2) <= 0:
            separation = 0.0
        else:
            d2 = 0.5 * logdet - 0.25 * logdet1 - 0.25 * logdet2
            separation = float(1 - np.exp(-(d1 + d2)))
        separations.append(separation)
    weights = model.weights_
    entropy = float(-(weights * np.log2(weights + 1e-12)).sum())
    normalized_partition = entropy / max(np.log2(model.n_components), 1.0)
    component_variances = [
        float(np.trace(covariance) / covariance.shape[0])
        for covariance in model.covariances_
    ]
    return {
        "min_pairwise_separation": float(min(separations)),
        "mean_pairwise_separation": float(np.mean(separations)),
        "partition": normalized_partition,
        "min_variance": float(min(component_variances)),
    }


def _canonical_quadrant_states(centers: np.ndarray) -> dict[int, str]:
    """Assign orientation-stable ``--``, ``-+``, ``+-``, ``++`` names."""
    centered = centers - centers.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    centered = centered / scale
    state_names = ["--", "-+", "+-", "++"]
    quadrants = np.asarray(
        [[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float
    )
    rows, columns = linear_sum_assignment(cdist(centered, quadrants))
    return {int(row): state_names[int(column)] for row, column in zip(rows, columns)}


def _match_component_centers(
    centers: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, float, float, tuple[int, ...]]:
    """Match a local component set to a reference, allowing label swaps."""
    cost = cdist(centers, reference)
    n_components = centers.shape[0]
    candidates = []
    for permutation in permutations(range(n_components)):
        values = np.asarray(
            [cost[permutation[index], index] for index in range(n_components)],
            dtype=float,
        )
        candidates.append((float(values.mean()), permutation))
    candidates.sort(key=lambda item: item[0])
    best_cost, best_permutation = candidates[0]
    second_cost = candidates[1][0] if len(candidates) > 1 else np.inf
    reordered = centers[np.asarray(best_permutation, dtype=int)]
    return (
        reordered,
        best_cost,
        float(second_cost - best_cost),
        tuple(best_permutation),
    )


def _map_components_to_reference(
    centers: np.ndarray,
    reference: np.ndarray,
) -> tuple[dict[int, int], float, float, np.ndarray]:
    """Map local components to a consensus set, allowing merges/missing states."""
    cost = cdist(centers, reference)
    n_local, n_reference = cost.shape
    if n_local == n_reference:
        _, _, margin, permutation = _match_component_centers(centers, reference)
        mapping = {
            int(local): int(ref)
            for ref, local in enumerate(permutation)
        }
        selected = cost[np.asarray(permutation, dtype=int), np.arange(n_reference)]
        margin_value = float(margin)
    elif n_local > n_reference:
        nearest = np.argmin(cost, axis=1)
        mapping = {int(local): int(ref) for local, ref in enumerate(nearest)}
        selected = cost[np.arange(n_local), nearest]
        ordered = np.sort(cost, axis=1)
        margin_value = (
            float(np.median(ordered[:, 1] - ordered[:, 0]))
            if n_reference > 1 else float("inf")
        )
    else:
        rows, columns = linear_sum_assignment(cost)
        mapping = {int(row): int(column) for row, column in zip(rows, columns)}
        selected = cost[rows, columns]
        local_best = np.min(cost, axis=1)
        margin_value = (
            float(np.median(np.sort(cost, axis=1)[:, 1] - local_best))
            if n_reference > 1 else float("inf")
        )
    mapped_centers = np.full_like(reference, np.nan, dtype=float)
    for ref in range(n_reference):
        local = [index for index, target in mapping.items() if target == ref]
        if local:
            mapped_centers[ref] = np.mean(centers[local], axis=0)
    return mapping, float(np.mean(selected)), margin_value, mapped_centers


def _find_2d_multistate_split(
    pca: pd.DataFrame,
    rnadata: Dict[str, anndata.AnnData],
    min_batch_cells: int,
    min_split_batches: int,
    min_child_cells: int,
    min_states: int,
    max_components: int,
    max_balanced_cells: int,
    posterior_cutoff: float,
    bic_gain_per_cell: float,
    separation_cutoff: float,
    partition_cutoff: float,
    variance_cutoff: float,
    max_center_distance: float,
    min_match_margin: float,
    seed: int,
) -> dict[str, object] | None:
    """Find recurrent 2-D states using batch-local BIC and center matching."""
    if pca.shape[1] < 2:
        return None
    pair = (pca.columns[0], pca.columns[1])
    feature_name = "+".join(pair)
    cache: dict[str, dict[str, object]] = {}
    batch_rows = []
    for batch_index, (batch, adata) in enumerate(rnadata.items()):
        cells = adata.obs_names.astype(str)
        if len(cells) < min_batch_cells:
            continue
        values = pca.loc[cells, list(pair)].to_numpy(dtype=np.float64)
        location = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-8] = 1.0
        values = (values - location) / scale
        min_component = max(10, int(np.ceil(0.02 * len(values))))
        fits = {}
        for k in range(1, min(max_components, 4) + 1):
            fit = _valid_gmm(values, k, seed + batch_index * 11 + k, min_component)
            if fit is not None:
                fits[k] = fit
        if 1 not in fits:
            continue
        best_k = min(fits, key=lambda k: fits[k][2])
        model, raw, bic = fits[best_k]
        gain = (fits[1][2] - bic) / len(values)
        quality = (
            _multicomponent_quality(model)
            if best_k > 1 else {
                "min_pairwise_separation": 0.0,
                "mean_pairwise_separation": 0.0,
                "partition": 0.0,
                "min_variance": 0.0,
            }
        )
        accepted = bool(
            best_k > 1
            and gain >= bic_gain_per_cell
            and quality["min_pairwise_separation"] >= separation_cutoff
            and quality["partition"] >= partition_cutoff
            and quality["min_variance"] >= variance_cutoff
        )
        score = gain + quality["min_pairwise_separation"] + 0.5 * quality["partition"] + 0.1 * quality["min_variance"]
        cache[str(batch)] = {
            "cells": cells, "values": values, "model": model, "raw": raw,
            "best_k": int(best_k), "accepted": accepted, "score": score,
        }
        batch_rows.append({
            "candidate_type": "2d_batch_bic", "feature": feature_name,
            "batch": str(batch), "n_cells": int(len(cells)),
            "n_components": int(best_k), "bic": float(bic),
            "bic_gain_per_cell": float(gain), "score": float(score),
            **quality, "states": "", "passed": accepted,
        })
    eligible = {batch: value for batch, value in cache.items() if value["accepted"]}
    if not eligible:
        return None
    counts = pd.Series([int(value["best_k"]) for value in eligible.values()], dtype=int).value_counts()
    max_count = int(counts.max())
    consensus_k = sorted(int(k) for k, value in counts.items() if value == max_count)[0]
    if consensus_k < min_states or max_count < min_split_batches:
        return None
    same_k = {batch: value for batch, value in eligible.items() if int(value["best_k"]) == consensus_k}
    if len(same_k) < min_split_batches:
        return None
    reference_batch = max(same_k, key=lambda batch: float(same_k[batch]["score"]))
    reference = np.asarray(same_k[reference_batch]["model"].means_, dtype=float)
    aligned = {}
    for _ in range(3):
        aligned = {}
        for batch, value in eligible.items():
            aligned[batch] = _map_components_to_reference(
                np.asarray(value["model"].means_, dtype=float), reference
            )
        reliable = [value[3] for value in aligned.values() if value[1] <= max_center_distance and value[2] >= min_match_margin]
        if len(reliable) < min_split_batches:
            break
        updated = np.nanmean(np.asarray(reliable, dtype=float), axis=0)
        reference = np.where(np.isfinite(updated), updated, reference)
    if not aligned:
        return None
    reliable_batches = {
        batch for batch, (_, cost, margin, _) in aligned.items()
        if cost <= max_center_distance and margin >= min_match_margin
    }
    if len(reliable_batches) < min_split_batches:
        return None
    state_names = _canonical_quadrant_states(reference)
    assignments: dict[str, tuple[pd.Index, np.ndarray]] = {}
    state_batch_support: dict[str, set[str]] = {state: set() for state in state_names.values()}
    state_cell_count: dict[str, int] = {state: 0 for state in state_names.values()}
    for batch in sorted(reliable_batches):
        value = eligible[batch]
        mapping = aligned[batch][0]
        raw = np.asarray(value["raw"])
        probability = value["model"].predict_proba(value["values"]).max(axis=1)
        labels = np.asarray([
            state_names[mapping[int(component)]] if int(component) in mapping else "?"
            for component in raw
        ], dtype=object)
        labels[probability < posterior_cutoff] = "?"
        cells = value["cells"]
        assignments[batch] = (cells, labels)
        for state in sorted(set(labels) - {"?"}):
            state_batch_support[state].add(batch)
            state_cell_count[state] += int(np.sum(labels == state))
    recurrent = sorted(
        state for state, batches in state_batch_support.items()
        if len(batches) >= min_split_batches and state_cell_count[state] >= min_child_cells
    )
    if len(recurrent) < min_states:
        return None
    consensus_row = pd.DataFrame([{
        "candidate_type": "2d_consensus", "feature": feature_name,
        "batch": "__CONSENSUS__", "n_cells": int(sum(len(value["cells"]) for value in eligible.values())),
        "n_components": int(consensus_k),
        "bic_gain_per_cell": float(np.mean([value["score"] for value in same_k.values()])),
        "n_eligible_batches": int(len(eligible)), "n_same_k_batches": int(len(same_k)),
        "n_aligned_batches": int(len(reliable_batches)),
        "n_merged_batches": int(sum(int(value["best_k"]) > consensus_k for value in eligible.values())),
        "n_lower_k_batches": int(sum(int(value["best_k"]) < consensus_k for value in eligible.values())),
        "median_match_distance": float(np.median([aligned[b][1] for b in reliable_batches])),
        "min_match_margin": float(min(aligned[b][2] for b in reliable_batches)),
        "states": ";".join(recurrent), "passed": True,
    }])
    return {
        "score": float(np.mean([cache[b]["score"] for b in reliable_batches])),
        "feature": feature_name, "features": list(pair), "states": recurrent,
        "assignments": assignments, "split_batches": sorted(reliable_batches),
        "diagnostics": pd.concat([consensus_row, pd.DataFrame(batch_rows)], ignore_index=True, sort=False),
    }


def resolve_min_split_batches(value: int | None, n_batches: int) -> int:
    """Resolve automatic support once using the full dataset batch count."""
    if value is None:
        return max(1, n_batches // 2)
    if value < 1:
        raise ValueError("min_split_batches must be at least 1")
    return value


def RecursiveRNAPCASplit(
    rnadata: Dict[str, anndata.AnnData],
    separation_cutoff: float = 0.5,
    n_hvg: int = 500,
    n_pcs: int = 5,
    min_node_cells: int = 50,
    min_batch_cells: int = 100,
    min_split_batches: int | None = None,
    min_child_cells: int = 50,
    small_fragment_max_cells: int = 50,
    small_fragment_separation: float = 0.8,
    dip_cutoff: float = 0.00495,
    partition_cutoff: float = 0.2,
    variance_cutoff: float = 0.3,
    enable_2d: bool = True,
    min_2d_states: int = 2,
    max_2d_components: int = 4,
    max_2d_balanced_cells: int = 1500,
    two_d_posterior_cutoff: float = 0.6,
    two_d_bic_gain_per_cell: float = 0.1,
    two_d_variance_cutoff: float = 0.01,
    two_d_max_center_distance: float = 4.0,
    two_d_min_match_margin: float = 0.0,
    max_depth: int = 8,
    seed: int = 2026,
    depth: int = 0,
    one_d_bic_gain_per_cell: float | None = None,
) -> RecursiveRNAPCANode:
    """Recursively split confident batches and retain the others at parents."""
    min_split_batches = resolve_min_split_batches(min_split_batches, len(rnadata))
    one_d_bic_cutoff = (
        two_d_bic_gain_per_cell
        if one_d_bic_gain_per_cell is None else one_d_bic_gain_per_cell
    )
    if not np.isfinite(one_d_bic_cutoff) or one_d_bic_cutoff < 0:
        raise ValueError("one_d_bic_gain_per_cell must be finite and nonnegative")
    clean = {str(key): value.copy() for key, value in rnadata.items() if value.n_obs > 0}
    indices = {
        batch: adata.obs_names.astype(str).tolist() for batch, adata in clean.items()
    }
    node = RecursiveRNAPCANode(depth=depth, indices=indices)
    total_cells = sum(len(cells) for cells in indices.values())
    if total_cells < min_node_cells or depth >= max_depth or not clean:
        return node

    pca, hvg = _node_pca(clean, n_hvg=n_hvg, n_pcs=n_pcs, seed=seed + depth)
    node.hvg_genes = hvg
    if pca.empty:
        return node
    diagnostics = _score_pcs(
        pca,
        clean,
        separation_cutoff=separation_cutoff,
        dip_cutoff=dip_cutoff,
        partition_cutoff=partition_cutoff,
        variance_cutoff=variance_cutoff,
        min_batch_cells=min_batch_cells,
        seed=seed + depth,
        bic_gain_per_cell=one_d_bic_cutoff,
    )
    node.diagnostics = diagnostics

    # Only attempt the two-dimensional state model after the same one-
    # dimensional bimodality gate used by the legacy splitter.  This prevents
    # BIC from manufacturing a 2-D split in an otherwise unimodal node.
    multistate = None
    one_d_trigger = bool(
        not diagnostics.empty and diagnostics["passed"].astype(bool).any()
    )
    if enable_2d and one_d_trigger:
        multistate = _find_2d_multistate_split(
            pca, clean,
            min_batch_cells=min_batch_cells,
            min_split_batches=min_split_batches,
            min_child_cells=min_child_cells,
            min_states=min_2d_states,
            max_components=max_2d_components,
            max_balanced_cells=max_2d_balanced_cells,
            posterior_cutoff=two_d_posterior_cutoff,
            bic_gain_per_cell=two_d_bic_gain_per_cell,
            separation_cutoff=separation_cutoff,
            partition_cutoff=partition_cutoff,
            variance_cutoff=two_d_variance_cutoff,
            max_center_distance=two_d_max_center_distance,
            min_match_margin=two_d_min_match_margin,
            seed=seed + depth,
        )
    if multistate is not None:
        node.multistate_diagnostics = multistate["diagnostics"]
        recurrent = list(multistate["states"])
        child_data: dict[str, Dict[str, anndata.AnnData]] = {state: {} for state in recurrent}
        unresolved_2d: Dict[str, list[str]] = {}
        for batch, adata in clean.items():
            assignment = multistate["assignments"].get(batch)
            if assignment is None:
                unresolved_2d[batch] = adata.obs_names.astype(str).tolist()
                continue
            cells, labels = assignment
            resolved = np.zeros(len(cells), dtype=bool)
            for state in recurrent:
                take = labels == state
                if bool(take.any()):
                    child_data[state][batch] = adata[cells[take]].copy()
                    resolved |= take
            if bool((~resolved).any()):
                unresolved_2d[batch] = cells[~resolved].astype(str).tolist()

        state_counts = {
            state: sum(part.n_obs for part in batches.values())
            for state, batches in child_data.items()
        }
        small_states = {
            state for state, count in state_counts.items()
            if small_fragment_max_cells > 0 and count < small_fragment_max_cells
        }
        kept_states = [state for state in recurrent if state not in small_states]
        if len(kept_states) >= min_2d_states:
            for state in small_states:
                for batch, part in child_data[state].items():
                    node.unlabeled_indices.setdefault(batch, []).extend(
                        part.obs_names.astype(str).tolist()
                    )
            node.split_mode = "2d_multistate"
            node.split_feature = str(multistate["feature"])
            node.split_features = list(multistate["features"])
            node.split_batches = sorted({
                batch for state in kept_states for batch in child_data[state]
            })
            node.unresolved_indices = unresolved_2d
            child_kwargs = dict(
                separation_cutoff=separation_cutoff, n_hvg=n_hvg, n_pcs=n_pcs,
                min_node_cells=min_node_cells, min_batch_cells=min_batch_cells,
                min_split_batches=min_split_batches, min_child_cells=min_child_cells,
                small_fragment_max_cells=small_fragment_max_cells,
                small_fragment_separation=small_fragment_separation,
                dip_cutoff=dip_cutoff, partition_cutoff=partition_cutoff,
                variance_cutoff=variance_cutoff, enable_2d=enable_2d,
                min_2d_states=min_2d_states, max_2d_components=max_2d_components,
                max_2d_balanced_cells=max_2d_balanced_cells,
                two_d_posterior_cutoff=two_d_posterior_cutoff,
                two_d_bic_gain_per_cell=two_d_bic_gain_per_cell,
                one_d_bic_gain_per_cell=one_d_bic_gain_per_cell,
                two_d_variance_cutoff=two_d_variance_cutoff,
                two_d_max_center_distance=two_d_max_center_distance,
                two_d_min_match_margin=two_d_min_match_margin,
                max_depth=max_depth, seed=seed, depth=depth + 1,
            )
            node.children = {
                state: RecursiveRNAPCASplit(child_data[state], **child_kwargs)
                for state in kept_states
            }
            return node

    if diagnostics.empty or not diagnostics["passed"].any():
        return node

    passed = diagnostics[diagnostics["passed"]]
    n_batches = max(len(clean), 1)
    aggregate = passed.groupby("feature")["score"].agg(["mean", "max", "count"])
    # A batch-specific bimodality must not create a global taxonomy branch.
    # Apply this when admitting the split, so rejected cells stay at the
    # current node rather than creating children that need post-hoc pruning.
    eligible = aggregate[aggregate["count"] >= min_split_batches].copy()
    if eligible.empty:
        return node
    aggregate["aggregate_score"] = (
        aggregate["mean"] + aggregate["max"] + aggregate["count"] / n_batches
    )
    node.aggregate_scores = aggregate["aggregate_score"].to_dict()
    best_feature = str(
        aggregate.loc[eligible.index, "aggregate_score"].idxmax()
    )
    split_batches = sorted(
        passed.loc[passed["feature"].eq(best_feature), "batch"].astype(str).unique()
    )

    left_data: Dict[str, anndata.AnnData] = {}
    right_data: Dict[str, anndata.AnnData] = {}
    unresolved: Dict[str, list[str]] = {}
    for batch, adata in clean.items():
        if batch not in split_batches:
            unresolved[batch] = adata.obs_names.astype(str).tolist()
            continue
        values = pca.loc[adata.obs_names.astype(str), [best_feature]].to_numpy()
        gmm = GaussianMixture(n_components=2, random_state=seed + depth).fit(values)
        raw = gmm.predict(values)
        low_component = int(np.argmin(gmm.means_[:, 0]))
        left_take = raw == low_component
        right_take = ~left_take
        if left_take.any():
            left_data[batch] = adata[left_take].copy()
        if right_take.any():
            right_data[batch] = adata[right_take].copy()

    left_count = sum(value.n_obs for value in left_data.values())
    right_count = sum(value.n_obs for value in right_data.values())
    if left_count < min_child_cells or right_count < min_child_cells:
        return node

    selected_rows = passed.loc[
        passed["feature"].eq(best_feature)
        & passed["batch"].astype(str).isin(split_batches)
    ]
    median_separation = float(selected_rows["separation"].median())
    small_left = bool(
        small_fragment_max_cells > 0 and left_count < small_fragment_max_cells
    )
    small_right = bool(
        small_fragment_max_cells > 0 and right_count < small_fragment_max_cells
    )
    if (small_left or small_right) and median_separation < small_fragment_separation:
        return node
    filter_left = small_left and median_separation >= small_fragment_separation
    filter_right = small_right and median_separation >= small_fragment_separation
    if filter_left or filter_right:
        filtered: Dict[str, list[str]] = {}
        for batch, part in (
            list(left_data.items()) if filter_left else []
        ) + (
            list(right_data.items()) if filter_right else []
        ):
            filtered.setdefault(batch, []).extend(part.obs_names.astype(str).tolist())
        filtered_set = {cell for values in filtered.values() for cell in values}
        retained = {}
        for batch, adata in clean.items():
            keep = ~adata.obs_names.astype(str).isin(filtered_set)
            if bool(keep.any()):
                retained[batch] = adata[keep].copy()
        if not retained:
            node.indices = {}
            node.unlabeled_indices = filtered
            return node
        replacement = RecursiveRNAPCASplit(
            retained,
            separation_cutoff=separation_cutoff,
            n_hvg=n_hvg,
            n_pcs=n_pcs,
            min_node_cells=min_node_cells,
            min_batch_cells=min_batch_cells,
            min_split_batches=min_split_batches,
            min_child_cells=min_child_cells,
            small_fragment_max_cells=small_fragment_max_cells,
            small_fragment_separation=small_fragment_separation,
            dip_cutoff=dip_cutoff,
            partition_cutoff=partition_cutoff,
            variance_cutoff=variance_cutoff,
            enable_2d=enable_2d,
            min_2d_states=min_2d_states,
            max_2d_components=max_2d_components,
            max_2d_balanced_cells=max_2d_balanced_cells,
            two_d_posterior_cutoff=two_d_posterior_cutoff,
            two_d_bic_gain_per_cell=two_d_bic_gain_per_cell,
            one_d_bic_gain_per_cell=one_d_bic_gain_per_cell,
            two_d_variance_cutoff=two_d_variance_cutoff,
            two_d_max_center_distance=two_d_max_center_distance,
            two_d_min_match_margin=two_d_min_match_margin,
            max_depth=max_depth,
            seed=seed,
            depth=depth,
        )
        for batch, cells in filtered.items():
            replacement.unlabeled_indices.setdefault(batch, []).extend(cells)
        return replacement

    node.split_feature = best_feature
    node.split_features = [best_feature]
    node.split_mode = "1d_binary"
    node.split_batches = split_batches
    node.unresolved_indices = unresolved
    child_kwargs = dict(
        separation_cutoff=separation_cutoff,
        n_hvg=n_hvg,
        n_pcs=n_pcs,
        min_node_cells=min_node_cells,
        min_batch_cells=min_batch_cells,
        min_split_batches=min_split_batches,
        min_child_cells=min_child_cells,
        small_fragment_max_cells=small_fragment_max_cells,
        small_fragment_separation=small_fragment_separation,
        dip_cutoff=dip_cutoff,
        partition_cutoff=partition_cutoff,
        variance_cutoff=variance_cutoff,
        enable_2d=enable_2d,
        min_2d_states=min_2d_states,
        max_2d_components=max_2d_components,
        max_2d_balanced_cells=max_2d_balanced_cells,
        two_d_posterior_cutoff=two_d_posterior_cutoff,
        two_d_bic_gain_per_cell=two_d_bic_gain_per_cell,
        one_d_bic_gain_per_cell=one_d_bic_gain_per_cell,
        two_d_variance_cutoff=two_d_variance_cutoff,
        two_d_max_center_distance=two_d_max_center_distance,
        two_d_min_match_margin=two_d_min_match_margin,
        max_depth=max_depth,
        seed=seed,
        depth=depth + 1,
    )
    node.left = RecursiveRNAPCASplit(left_data, **child_kwargs)
    node.right = RecursiveRNAPCASplit(right_data, **child_kwargs)
    return node


def _node_cells(node: RecursiveRNAPCANode) -> pd.Index:
    return pd.Index(
        [cell for values in node.indices.values() for cell in map(str, values)],
        dtype=str,
    )


def _export_tree(
    tree: RecursiveRNAPCANode,
    parent_leaf: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Export one partial-label tree without reading a ground-truth column."""

    edges: list[dict[str, object]] = []
    nodes: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    diagnostics: list[pd.DataFrame] = []

    def visit(node: RecursiveRNAPCANode, path: str, parent_id: str | None) -> None:
        split = node.is_split
        if not path:
            node_id = parent_leaf
        elif split:
            node_id = f"{parent_leaf}__RNA_NODE_{path}"
        else:
            node_id = f"{parent_leaf}__RNA_LEAF_{path}"
        cells = _node_cells(node)
        unresolved = pd.Index(
            [
                cell
                for values in node.unresolved_indices.values()
                for cell in map(str, values)
            ],
            dtype=str,
        )
        assigned_here = unresolved if split else cells
        unlabeled = pd.Index(
            [
                cell
                for values in node.unlabeled_indices.values()
                for cell in map(str, values)
            ],
            dtype=str,
        )
        nodes.append({
            "tree_node_id": node_id,
            "parent_id": parent_id or "",
            "binary_path": path or "ROOT",
            "depth_within_refinement": int(node.depth),
            "node_kind": "rna_internal" if split else "rna_leaf",
            "split_mode": getattr(node, "split_mode", "1d_binary" if split else "leaf"),
            "split_feature": node.split_feature if split else "",
            "split_features": "+".join(getattr(node, "split_features", [])),
            "split_batches": ";".join(node.split_batches),
            "n_cells": int(len(cells)),
            "n_assigned_here": int(len(assigned_here)),
            "n_filtered_unlabeled": int(len(unlabeled)),
        })
        if parent_id is not None:
            edges.append({"parent": parent_id, "child": node_id})
        for cell in assigned_here:
            assignments.append({
                "cell_barcode": str(cell),
                "rna_refinement_parent_leaf": parent_leaf,
                "assigned_taxonomy_node_id": node_id,
                "assigned_node_kind": "internal" if split else "leaf",
                "rna_assignment_level": (
                    "partial_parent" if split else "terminal_leaf"
                ),
            })
        for cell in unlabeled:
            assignments.append({
                "cell_barcode": str(cell),
                "rna_refinement_parent_leaf": parent_leaf,
                "assigned_taxonomy_node_id": pd.NA,
                "assigned_node_kind": "unlabeled",
                "rna_assignment_level": "fully_unlabeled",
            })
        node_diagnostics = []
        if not node.diagnostics.empty:
            node_diagnostics.append(node.diagnostics.copy())
        if not node.multistate_diagnostics.empty:
            node_diagnostics.append(node.multistate_diagnostics.copy())
        if node_diagnostics:
            current = pd.concat(node_diagnostics, ignore_index=True, sort=False)
            current.insert(0, "tree_node_id", node_id)
            current.insert(1, "binary_path", path or "ROOT")
            current["selected_feature"] = (
                current["feature"].eq(node.split_feature) & current["passed"]
            )
            diagnostics.append(current)
        if split:
            branch_codes = {"-": "0", "+": "1", "--": "0", "-+": "1", "+-": "2", "++": "3"}
            for label, child in node.child_items():
                code = branch_codes.get(str(label), str(label))
                visit(child, path + code, node_id)

    visit(tree, "", None)
    return (
        pd.DataFrame(edges, columns=["parent", "child"]),
        pd.DataFrame(nodes),
        pd.DataFrame(assignments),
        pd.concat(diagnostics, ignore_index=True)
        if diagnostics else pd.DataFrame(),
    )


def refine_taxonomy_with_rna(
    config: RNARefinementConfig,
) -> dict[str, object]:
    """Refine first-pass predicted leaves and emit a second-pass taxonomy.

    Only RNA values, batch identity, and first-pass predicted leaves are used.
    Batches that do not support a selected split remain assigned to the parent.
    """

    if config.output_dir.resolve() == config.taxonomy_dir.resolve():
        raise ValueError("RNA refinement output must differ from the source taxonomy")
    output_tables = config.output_dir / "tables"
    if output_tables.exists():
        shutil.rmtree(output_tables)
    output_tables.mkdir(parents=True, exist_ok=True)
    tree_root = config.output_dir / "rna_trees"
    if tree_root.exists():
        shutil.rmtree(tree_root)
    source_tables = config.taxonomy_dir / "tables"
    for source in source_tables.iterdir():
        if source.is_file():
            shutil.copy2(source, output_tables / source.name)

    rna = anndata.read_h5ad(config.rna_h5ad)
    first = anndata.read_h5ad(config.initial_model_h5ad)
    rna.obs_names = rna.obs_names.astype(str)
    first.obs_names = first.obs_names.astype(str)
    if not rna.obs_names.equals(first.obs_names):
        first = first[rna.obs_names].copy()
    if "predicted_leaf" not in first.obs:
        raise KeyError("first-pass model lacks obs['predicted_leaf']")
    if config.batch_key not in rna.obs:
        if config.batch_key not in first.obs:
            raise KeyError(
                f"RNA refinement requires batch column {config.batch_key!r}"
            )
        rna.obs[config.batch_key] = first.obs[config.batch_key].astype(str).to_numpy()
    # Strip every other annotation before splitting, making the no-known-label
    # contract structural rather than merely conventional.
    rna.obs = rna.obs[[config.batch_key]].copy()

    if "feature_types" in rna.var:
        gene = rna.var["feature_types"].astype(str).eq("Gene Expression").to_numpy()
        if gene.any():
            rna = rna[:, gene].copy()
    else:
        rna = rna.copy()
    if hasattr(rna.X, "tocsr"):
        rna.X = rna.X.tocsr().astype(np.float32)
    else:
        rna.X = np.asarray(rna.X, dtype=np.float32)
    totals = np.asarray(rna.X.sum(axis=1)).ravel()
    if np.allclose(totals, np.round(totals)):
        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)

    min_split_batches = resolve_min_split_batches(
        config.min_split_batches, rna.obs[config.batch_key].nunique()
    )
    parent = first.obs["predicted_leaf"].astype(str).to_numpy()
    available = sorted(pd.unique(parent))
    requested = (
        available
        if config.parent_leaves is None
        else [str(value) for value in config.parent_leaves]
    )
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"requested first-pass leaves are absent: {missing}")

    all_edges: list[pd.DataFrame] = []
    all_nodes: list[pd.DataFrame] = []
    all_assignments: list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []
    split_parent_count = 0
    for parent_leaf in requested:
        selected = np.flatnonzero(parent == parent_leaf)
        node_rna = rna[selected].copy()
        by_batch = {
            str(batch): node_rna[
                node_rna.obs[config.batch_key].astype(str).eq(str(batch)).to_numpy()
            ].copy()
            for batch in sorted(
                node_rna.obs[config.batch_key].astype(str).unique()
            )
        }
        tree = RecursiveRNAPCASplit(
            by_batch,
            separation_cutoff=config.separation_cutoff,
            n_hvg=config.n_hvg,
            n_pcs=config.n_pcs,
            min_node_cells=config.min_node_cells,
            min_batch_cells=config.min_batch_cells,
            min_split_batches=min_split_batches,
            min_child_cells=config.min_child_cells,
            small_fragment_max_cells=config.small_fragment_max_cells,
            small_fragment_separation=config.small_fragment_separation,
            dip_cutoff=config.dip_cutoff,
            partition_cutoff=config.partition_cutoff,
            variance_cutoff=config.variance_cutoff,
            enable_2d=config.enable_2d,
            min_2d_states=config.min_2d_states,
            max_2d_components=config.max_2d_components,
            max_2d_balanced_cells=config.max_2d_balanced_cells,
            two_d_posterior_cutoff=config.two_d_posterior_cutoff,
            two_d_bic_gain_per_cell=config.two_d_bic_gain_per_cell,
            one_d_bic_gain_per_cell=config.one_d_bic_gain_per_cell,
            two_d_variance_cutoff=config.two_d_variance_cutoff,
            two_d_max_center_distance=config.two_d_max_center_distance,
            two_d_min_match_margin=config.two_d_min_match_margin,
            max_depth=config.max_depth,
            seed=config.seed,
        )
        parent_dir = tree_root / parent_leaf
        parent_dir.mkdir(parents=True, exist_ok=True)
        with (parent_dir / "tree.pickle").open("wb") as handle:
            pickle.dump(tree, handle)
        edges, nodes, assignments, diagnostics = _export_tree(tree, parent_leaf)
        observed = assignments["cell_barcode"].astype(str)
        expected = set(node_rna.obs_names.astype(str))
        if observed.duplicated().any() or set(observed) != expected:
            raise ValueError(f"RNA refinement does not cover {parent_leaf} exactly once")
        edges.to_csv(parent_dir / "tree_edges.csv", index=False)
        nodes.to_csv(parent_dir / "tree_nodes.csv", index=False)
        assignments.to_csv(parent_dir / "cell_assignments.csv", index=False)
        diagnostics.to_csv(parent_dir / "pca_diagnostics.csv", index=False)
        if not edges.empty:
            split_parent_count += 1
            all_edges.append(edges)
        all_nodes.append(nodes)
        all_assignments.append(assignments)
        if not diagnostics.empty:
            all_diagnostics.append(diagnostics)

    # Leaves not requested for refinement still use the first-pass prediction.
    untouched = ~np.isin(parent, requested)
    if untouched.any():
        all_assignments.append(pd.DataFrame({
            "cell_barcode": rna.obs_names[untouched].astype(str),
            "rna_refinement_parent_leaf": parent[untouched],
            "assigned_taxonomy_node_id": parent[untouched],
            "assigned_node_kind": "leaf",
            "rna_assignment_level": "not_requested",
        }))

    refined = pd.concat(all_assignments, ignore_index=True).set_index("cell_barcode")
    refined = refined.reindex(rna.obs_names.astype(str))
    required = [
        "rna_refinement_parent_leaf", "assigned_node_kind",
        "rna_assignment_level",
    ]
    if refined.index.duplicated().any() or refined[required].isna().any(axis=None):
        raise ValueError("refined taxonomy must assign every cell exactly once")
    invalid_missing = (
        refined["assigned_taxonomy_node_id"].isna()
        & ~refined["assigned_node_kind"].astype(str).eq("unlabeled")
    )
    if invalid_missing.any():
        raise ValueError("only unlabeled cells may omit a taxonomy node")
    base_cells = pd.read_csv(
        source_tables / "cell_taxonomy_assignments.csv",
        dtype={"cell_barcode": str},
    ).set_index("cell_barcode").reindex(rna.obs_names.astype(str))
    if base_cells.isna().all(axis=1).any():
        raise ValueError("source taxonomy does not cover every RNA cell")
    original_edges = pd.read_csv(source_tables / "taxonomy_edges.csv", dtype=str)
    new_edges = (
        pd.concat(all_edges, ignore_index=True)
        if all_edges else pd.DataFrame(columns=["parent", "child"])
    )
    combined_edges = pd.concat([original_edges, new_edges], ignore_index=True)
    combined_edges.drop_duplicates().to_csv(
        output_tables / "taxonomy_edges.csv", index=False
    )
    refinement_nodes = pd.concat(all_nodes, ignore_index=True)
    refinement_nodes.to_csv(output_tables / "rna_refinement_nodes.csv", index=False)
    nodes_path = source_tables / "taxonomy_nodes.csv"
    if nodes_path.exists():
        original_nodes = pd.read_csv(nodes_path)
        split_parents = set(new_edges["parent"].astype(str))
        if "tree_node_id" in original_nodes and "node_kind" in original_nodes:
            original_nodes.loc[
                original_nodes["tree_node_id"].astype(str).isin(split_parents),
                "node_kind",
            ] = "internal"
        appended = refinement_nodes.loc[
            ~refinement_nodes["tree_node_id"].astype(str).isin(
                original_nodes["tree_node_id"].astype(str)
            )
        ]
        pd.concat([original_nodes, appended], ignore_index=True).to_csv(
            output_tables / "taxonomy_nodes.csv", index=False
        )
    if all_diagnostics:
        pd.concat(all_diagnostics, ignore_index=True).to_csv(
            output_tables / "rna_refinement_diagnostics.csv", index=False
        )
    new_edges.to_csv(output_tables / "rna_refinement_edges.csv", index=False)

    for column in refined.columns:
        base_cells[column] = refined[column].to_numpy()
    parent_of = combined_edges.drop_duplicates("child").set_index("child")[
        "parent"
    ].astype(str).to_dict()

    def taxonomy_path(node: object) -> str:
        path = [str(node)]
        seen = set(path)
        while path[-1] in parent_of:
            ancestor = parent_of[path[-1]]
            if ancestor in seen:
                raise ValueError("cycle detected in refined taxonomy")
            path.append(ancestor)
            seen.add(ancestor)
        return ";".join(reversed(path))

    base_cells["taxonomy_path"] = base_cells["assigned_taxonomy_node_id"].apply(
        lambda value: taxonomy_path(value) if pd.notna(value) else ""
    )
    base_cells["taxonomy_root"] = base_cells["taxonomy_path"].str.split(";").str[0]
    base_cells.index.name = "cell_barcode"
    base_cells.reset_index().to_csv(
        output_tables / "cell_taxonomy_assignments.csv", index=False
    )

    forest_path = source_tables / "taxonomy_forest.json"
    forest = json.loads(forest_path.read_text())
    children = {str(k): list(map(str, v)) for k, v in forest["children"].items()}
    for row in new_edges.itertuples(index=False):
        children.setdefault(str(row.parent), []).append(str(row.child))
    forest["children"] = {key: sorted(set(value)) for key, value in children.items()}
    (output_tables / "taxonomy_forest.json").write_text(
        json.dumps(forest, indent=2, ensure_ascii=False) + "\n"
    )
    summary = {
        "min_split_batches": min_split_batches,
        "min_split_batches_policy": "dataset_half_floor" if config.min_split_batches is None else "explicit",
        "known_labels_used": False,
        "grouping_source": "first_pass_model.obs[predicted_leaf]",
        "unresolved_batch_policy": "retain_at_parent",
        "n_requested_parent_leaves": len(requested),
        "n_split_parent_leaves": split_parent_count,
        "n_new_edges": int(len(new_edges)),
        "n_final_leaf_labels": int(
            base_cells["assigned_taxonomy_node_id"]
            [base_cells["assigned_node_kind"].astype(str).eq("leaf")].nunique()
        ),
        "n_partial_parent_cells": int(
            base_cells["assigned_node_kind"].astype(str).eq("internal").sum()
        ),
        "n_fully_unlabeled_cells": int(
            base_cells["assigned_node_kind"].astype(str).eq("unlabeled").sum()
        ),
        "output_taxonomy": str(config.output_dir),
    }
    (config.output_dir / "refinement_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna-h5ad", type=Path, required=True)
    parser.add_argument("--initial-model-h5ad", type=Path, required=True)
    parser.add_argument("--taxonomy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--parent-leaves", nargs="+")
    parser.add_argument("--separation-cutoff", type=float, default=0.5)
    parser.add_argument("--n-hvg", type=int, default=500)
    parser.add_argument("--n-pcs", type=int, default=5)
    parser.add_argument("--min-node-cells", type=int, default=50)
    parser.add_argument("--min-batch-cells", type=int, default=100)
    parser.add_argument("--min-split-batches", type=int, default=None, help="Default: max(1, total dataset batches // 2); an integer overrides automatic support.")
    parser.add_argument("--min-child-cells", type=int, default=50)
    parser.add_argument("--small-fragment-max-cells", type=int, default=50)
    parser.add_argument("--small-fragment-separation", type=float, default=0.8)
    parser.add_argument("--dip-cutoff", type=float, default=0.00495)
    parser.add_argument("--partition-cutoff", type=float, default=0.2)
    parser.add_argument("--variance-cutoff", type=float, default=0.3)
    parser.add_argument("--disable-2d", action="store_true")
    parser.add_argument("--min-2d-states", type=int, default=2)
    parser.add_argument("--max-2d-components", type=int, default=4)
    parser.add_argument("--max-2d-balanced-cells", type=int, default=1500)
    parser.add_argument("--two-d-posterior-cutoff", type=float, default=0.6)
    parser.add_argument("--two-d-bic-gain-per-cell", type=float, default=0.1)
    parser.add_argument(
        "--one-d-bic-gain-per-cell", type=float, default=None,
        help="1-D BIC gain per cell; defaults to the 2-D BIC threshold.",
    )
    parser.add_argument("--two-d-variance-cutoff", type=float, default=0.01)
    parser.add_argument("--two-d-max-center-distance", type=float, default=4.0)
    parser.add_argument("--two-d-min-match-margin", type=float, default=0.0)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = refine_taxonomy_with_rna(RNARefinementConfig(
        rna_h5ad=args.rna_h5ad,
        initial_model_h5ad=args.initial_model_h5ad,
        taxonomy_dir=args.taxonomy_dir,
        output_dir=args.output_dir,
        batch_key=args.batch_key,
        parent_leaves=tuple(args.parent_leaves) if args.parent_leaves else None,
        separation_cutoff=args.separation_cutoff,
        n_hvg=args.n_hvg,
        n_pcs=args.n_pcs,
        min_node_cells=args.min_node_cells,
        min_batch_cells=args.min_batch_cells,
        min_split_batches=args.min_split_batches,
        min_child_cells=args.min_child_cells,
        small_fragment_max_cells=args.small_fragment_max_cells,
        small_fragment_separation=args.small_fragment_separation,
        dip_cutoff=args.dip_cutoff,
        partition_cutoff=args.partition_cutoff,
        variance_cutoff=args.variance_cutoff,
        enable_2d=not args.disable_2d,
        min_2d_states=args.min_2d_states,
        max_2d_components=args.max_2d_components,
        max_2d_balanced_cells=args.max_2d_balanced_cells,
        two_d_posterior_cutoff=args.two_d_posterior_cutoff,
        two_d_bic_gain_per_cell=args.two_d_bic_gain_per_cell,
        one_d_bic_gain_per_cell=args.one_d_bic_gain_per_cell,
        two_d_variance_cutoff=args.two_d_variance_cutoff,
        two_d_max_center_distance=args.two_d_max_center_distance,
        two_d_min_match_margin=args.two_d_min_match_margin,
        max_depth=args.max_depth,
        seed=args.seed,
    ))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
