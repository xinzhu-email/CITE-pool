"""Ternary protein signatures, strict batch alignment, and taxonomy building."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


UNKNOWN = "?"
POSITIVE = "+"
NEGATIVE = "-"
VALID_STATES = {UNKNOWN, POSITIVE, NEGATIVE}
DEFAULT_CONFIDENCE = 1.0

@dataclass(frozen=True)
class StateThresholds:
    """Thresholds used to turn posterior evidence into ``+``, ``-``, or ``?``."""

    min_marker_weight: float = 0.25
    min_state_confidence: float = 0.60
    min_state_separation: float = 0.15
    min_cluster_fraction: float = 0.60
    min_marker_effect: float = 0.10
    min_cell_state_probability: float = 0.60
    max_markers_per_cluster: int | None = 40
    max_markers_per_state: int | None = 20


def _marker_weights(marker_params: pd.DataFrame) -> np.ndarray:
    if "configuration_weight" in marker_params:
        values = marker_params["configuration_weight"]
    elif "phenotype_weight" in marker_params:
        values = marker_params["phenotype_weight"]
    elif "marker_weight" in marker_params:
        values = marker_params["marker_weight"]
    else:
        values = pd.Series(0.0, index=marker_params.index)
    return np.nan_to_num(
        pd.to_numeric(values, errors="coerce").to_numpy(float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clip(0.0, 1.0)


def _wilson_lower_bound(
    fraction: np.ndarray,
    n: int,
    *,
    z: float = 1.96,
) -> np.ndarray:
    """Conservative binomial support that accounts for cluster size."""

    values = np.asarray(fraction, dtype=float).clip(0.0, 1.0)
    if n <= 0:
        return np.zeros_like(values)
    denominator = 1.0 + z * z / n
    center = values + z * z / (2.0 * n)
    radius = z * np.sqrt(
        values * (1.0 - values) / n + z * z / (4.0 * n * n)
    )
    return np.clip((center - radius) / denominator, 0.0, 1.0)


def build_cluster_marker_states(
    *,
    batch_id: str,
    cluster_labels: np.ndarray,
    marker_names: Iterable[str],
    low_prob: np.ndarray,
    high_prob: np.ndarray,
    middle_prob: np.ndarray | None = None,
    call_n_states: np.ndarray | None = None,
    marker_params: pd.DataFrame,
    thresholds: StateThresholds = StateThresholds(),
) -> pd.DataFrame:
    """Build a complete long-form cluster-marker ternary configuration.

    Every state is called against the marker model fitted inside this batch.
    A marker that is not reliably bimodal in this batch remains unknown and
    contributes neither agreement nor conflict during alignment.
    """

    labels = np.asarray(cluster_labels, dtype=str)
    markers = np.asarray(list(marker_names), dtype=str)
    low = np.asarray(low_prob, dtype=float)
    high = np.asarray(high_prob, dtype=float)
    if low.shape != high.shape or low.shape != (len(labels), len(markers)):
        raise ValueError("low/high posterior matrices must be cells x markers")
    middle = (
        np.zeros_like(low)
        if middle_prob is None
        else np.asarray(middle_prob, dtype=float)
    )
    if middle.shape != low.shape:
        raise ValueError("middle posterior matrix must be cells x markers")
    if len(marker_params) != len(markers):
        raise ValueError("marker_params must have exactly one row per marker")

    params = marker_params.reset_index(drop=True).copy()
    weights = _marker_weights(params)
    n_states = pd.to_numeric(
        params.get("n_states", pd.Series(2, index=params.index)),
        errors="coerce",
    ).fillna(1).to_numpy(int)
    if call_n_states is None:
        call_states = n_states.copy()
    else:
        call_states = np.asarray(call_n_states, dtype=int)
        if call_states.shape != (len(markers),):
            raise ValueError("call_n_states must have one value per marker")
    status = params.get(
        "status", pd.Series("unknown", index=params.index)
    ).fillna("unknown").astype(str).to_numpy()
    reliable_status = np.array(
        [
            (s in {"valid_bimodal", "weak_bimodal", "unknown"})
            or s.startswith("valid")
            for s in status
        ],
        dtype=bool,
    )

    cell_state = np.full((len(labels), len(markers)), -1, dtype=np.int8)
    cell_conf = np.zeros((len(labels), len(markers)), dtype=float)
    for j in range(len(markers)):
        if call_states[j] == 3:
            probabilities = np.column_stack(
                [low[:, j], middle[:, j], high[:, j]]
            )
            states = np.argmax(probabilities, axis=1)
        elif call_states[j] == 2:
            probabilities = np.column_stack([low[:, j], high[:, j]])
            states = np.where(np.argmax(probabilities, axis=1) == 1, 2, 0)
        else:
            probabilities = np.ones((len(labels), 1), dtype=float)
            states = np.full(len(labels), -1, dtype=np.int8)
        confidence = np.max(probabilities, axis=1)
        states[confidence < thresholds.min_cell_state_probability] = -1
        cell_state[:, j] = states
        cell_conf[:, j] = confidence
    records: list[dict] = []
    unique_clusters = sorted(
        np.unique(labels),
        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)),
    )
    for cluster in unique_clusters:
        inside = labels == str(cluster)
        outside = ~inside
        n_cells = int(inside.sum())
        mean_low = low[inside].mean(axis=0)
        mean_middle = middle[inside].mean(axis=0)
        mean_high = high[inside].mean(axis=0)
        mean_probabilities = np.column_stack(
            [mean_low, mean_middle, mean_high]
        )
        dominant_idx = np.argmax(mean_probabilities, axis=1)
        dominant_prob = np.max(mean_probabilities, axis=1)
        second_prob = np.partition(mean_probabilities, -2, axis=1)[:, -2]
        separation = dominant_prob - second_prob
        support_fraction = np.array(
            [
                np.mean(cell_state[inside, j] == dominant_idx[j])
                for j in range(len(markers))
            ],
            dtype=float,
        )
        support_lower = _wilson_lower_bound(support_fraction, n_cells)
        outside_low = low[outside].mean(axis=0) if outside.any() else np.zeros(len(markers))
        outside_middle = middle[outside].mean(axis=0) if outside.any() else np.zeros(len(markers))
        outside_high = high[outside].mean(axis=0) if outside.any() else np.zeros(len(markers))
        outside_probabilities = np.column_stack(
            [outside_low, outside_middle, outside_high]
        )
        inside_dom = mean_probabilities[np.arange(len(markers)), dominant_idx]
        outside_dom = outside_probabilities[np.arange(len(markers)), dominant_idx]
        effect = (inside_dom - outside_dom) * weights
        dominant_sum = mean_probabilities * n_cells
        dominant_sum = dominant_sum[np.arange(len(markers)), dominant_idx]
        dominance_a = dominant_sum + 1.0
        dominance_b = n_cells - dominant_sum + 1.0
        dominance_posterior = 1.0 - beta_dist.cdf(
            0.5, dominance_a, dominance_b
        )

        identity_callable = (
            np.isin(call_states, [2, 3])
            & reliable_status
            & (weights >= thresholds.min_marker_weight)
            & (dominant_prob >= thresholds.min_state_confidence)
            & (separation >= thresholds.min_state_separation)
            & (support_fraction >= thresholds.min_cluster_fraction)
            & (dominance_posterior >= 0.99)
            & ((call_states != 3) | np.isin(dominant_idx, [0, 2]))
        )
        evidence_score = (
            np.maximum(effect, 0.0)
            * dominant_prob
            * separation
            * support_fraction
        )
        call_confidence = weights * np.power(
            np.maximum(
                dominant_prob
                * separation
                * support_lower
                * dominance_posterior,
                0.0,
            ),
            0.25,
        )
        rankable = identity_callable & (effect >= thresholds.min_marker_effect)
        selected = np.flatnonzero(rankable)
        if thresholds.max_markers_per_state is not None:
            state_limited = []
            per_state = max(int(thresholds.max_markers_per_state), 0)
            for state_idx in (0, 2):
                candidates = selected[dominant_idx[selected] == state_idx]
                ranked = candidates[
                    np.argsort(-evidence_score[candidates], kind="stable")
                ]
                state_limited.extend(ranked[:per_state].tolist())
            selected = np.asarray(sorted(set(state_limited)), dtype=int)
        if thresholds.max_markers_per_cluster is not None:
            total = max(int(thresholds.max_markers_per_cluster), 0)
            selected = selected[
                np.argsort(-evidence_score[selected], kind="stable")
            ][:total]
        retained = np.zeros(len(markers), dtype=bool)
        retained[selected] = True
        calls = np.full(len(markers), UNKNOWN, dtype=object)
        calls[identity_callable & (dominant_idx == 2)] = POSITIVE
        calls[identity_callable & (dominant_idx == 0)] = NEGATIVE
        call_confidence[~identity_callable] = 0.0

        for j, marker in enumerate(markers):
            records.append(
                {
                    "batch": str(batch_id),
                    "cluster": str(cluster),
                    "local_node_id": f"{batch_id}|{cluster}",
                    "n_cells": n_cells,
                    "marker": str(marker),
                    "state": str(calls[j]),
                    "mean_low_prob": float(mean_low[j]),
                    "mean_middle_prob": float(mean_middle[j]),
                    "mean_high_prob": float(mean_high[j]),
                    "state_confidence": float(dominant_prob[j]),
                    "state_separation": float(separation[j]),
                    "cluster_state_fraction": float(support_fraction[j]),
                    "cluster_state_fraction_lower": float(support_lower[j]),
                    "marker_weight": float(weights[j]),
                    "marker_effect": float(effect[j]),
                    "dominance_posterior": float(dominance_posterior[j]),
                    "marker_n_states": int(call_states[j]),
                    "marker_status": str(status[j]),
                    "is_callable": bool(identity_callable[j]),
                    "is_retained": bool(retained[j]),
                    "evidence_score": float(evidence_score[j]),
                    "call_confidence": float(call_confidence[j]),
                }
            )
    return pd.DataFrame.from_records(records)


def signature_dicts(state_table: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Convert a long state table into node -> marker -> state mappings."""

    required = {"local_node_id", "marker", "state"}
    missing = required - set(state_table.columns)
    if missing:
        raise KeyError(f"state table missing columns: {sorted(missing)}")
    bad = set(state_table["state"].astype(str)) - VALID_STATES
    if bad:
        raise ValueError(f"invalid ternary states: {sorted(bad)}")
    return {
        str(node): dict(zip(sub["marker"].astype(str), sub["state"].astype(str)))
        for node, sub in state_table.groupby("local_node_id", sort=True)
    }


def confidence_dicts(
    state_table: pd.DataFrame,
    *,
    node_column: str = "local_node_id",
) -> dict[str, dict[str, float]]:
    """Return node-marker confidences, supporting legacy state-only tables."""

    if node_column not in state_table:
        raise KeyError(f"state table missing node column: {node_column}")
    confidence_column = next(
        (
            column
            for column in ("confidence", "call_confidence")
            if column in state_table
        ),
        None,
    )
    output: dict[str, dict[str, float]] = {}
    for node, sub in state_table.groupby(node_column, sort=True):
        states = sub["state"].astype(str).to_numpy()
        if confidence_column is None:
            values = np.where(states == UNKNOWN, 0.0, DEFAULT_CONFIDENCE)
        else:
            values = (
                pd.to_numeric(sub[confidence_column], errors="coerce")
                .fillna(0.0)
                .to_numpy(float)
                .clip(0.0, 1.0)
            )
            values[states == UNKNOWN] = 0.0
        output[str(node)] = dict(zip(sub["marker"].astype(str), values))
    return output


def consolidate_equivalent_local_clusters(
    state_table: pd.DataFrame,
    *,
    min_shared_observed: int = 3,
    min_reciprocal_coverage: float = 0.80,
    min_call_confidence: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse batch-local fragments with equivalent reliable configurations.

    A pair is compatible when it has no opposing ``+``/``-`` call and enough
    shared calls covering both configurations. Low-confidence calls are unknown
    under the same standard used by cross-batch alignment. Groups are complete-link, and
    a reliable coarse-to-fine fork remains unresolved.
    """

    if not 0.0 <= float(min_reciprocal_coverage) <= 1.0:
        raise ValueError("min_reciprocal_coverage must be between 0 and 1")
    if not 0.0 <= float(min_call_confidence) <= 1.0:
        raise ValueError("min_call_confidence must be between 0 and 1")
    state_table = state_table.copy()
    confidence_column = next(
        (column for column in ("call_confidence", "confidence")
         if column in state_table),
        None,
    )
    if confidence_column is not None:
        confidence = pd.to_numeric(
            state_table[confidence_column], errors="coerce"
        ).fillna(0.0)
        weak = confidence < float(min_call_confidence)
        state_table.loc[weak, "state"] = UNKNOWN
        state_table.loc[weak, confidence_column] = 0.0
    signatures = signature_dicts(state_table)
    confidences = confidence_dicts(state_table)
    meta = (
        state_table[["local_node_id", "batch", "cluster", "n_cells"]]
        .drop_duplicates("local_node_id")
        .set_index("local_node_id")
    )
    batch_by_node = meta["batch"].astype(str).to_dict()
    local_pair_evidence: dict[tuple[str, str], dict] = {}
    for _, batch_meta in meta.groupby("batch", sort=True):
        batch_nodes = sorted(batch_meta.index.astype(str))
        for index, left in enumerate(batch_nodes[:-1]):
            for right in batch_nodes[index + 1:]:
                local_pair_evidence[(left, right)] = _pair_evidence(
                    signatures[left],
                    signatures[right],
                    confidences[left],
                    confidences[right],
                )
    coarse_to_fine_pairs = _coarse_to_fine_pairs(
        local_pair_evidence,
        signatures,
        confidences,
        batch_by_node,
        min_shared_observed=min_shared_observed,
    )

    def compatible(left: str, right: str) -> bool:
        evidence = _pair_evidence(signatures[left], signatures[right])
        if evidence["n_conflicts"]:
            return False
        shared = int(evidence["n_shared_observed"])
        if shared < max(int(min_shared_observed), 1):
            return False
        if float(evidence["minimum_configuration_coverage"]) < float(
            min_reciprocal_coverage
        ):
            return False
        return tuple(sorted((left, right))) not in coarse_to_fine_pairs

    groups: list[list[str]] = []
    for batch, batch_meta in meta.groupby("batch", sort=True):
        del batch
        nodes = sorted(
            batch_meta.index.astype(str),
            key=lambda node: (-int(meta.loc[node, "n_cells"]), node),
        )
        batch_groups: list[list[str]] = []
        for node in nodes:
            candidates = [
                group for group in batch_groups
                if all(compatible(node, member) for member in group)
            ]
            if candidates:
                best = max(
                    candidates,
                    key=lambda group: (
                        min(
                            _pair_evidence(signatures[node], signatures[member])[
                                "minimum_configuration_coverage"
                            ]
                            for member in group
                        ),
                        min(
                            _pair_evidence(signatures[node], signatures[member])[
                                "n_shared_observed"
                            ]
                            for member in group
                        ),
                        sum(int(meta.loc[x, "n_cells"]) for x in group),
                    ),
                )
                best.append(node)
            else:
                batch_groups.append([node])
        groups.extend(batch_groups)

    mapping_rows: list[dict] = []
    collapsed_parts: list[pd.DataFrame] = []
    numeric_columns = state_table.select_dtypes(include=[np.number]).columns.tolist()
    for members in groups:
        representative = max(
            members,
            key=lambda node: (int(meta.loc[node, "n_cells"]), node),
        )
        representative_cluster = str(meta.loc[representative, "cluster"])
        total_cells = sum(int(meta.loc[node, "n_cells"]) for node in members)
        for node in members:
            mapping_rows.append(
                {
                    "local_node_id_original": node,
                    "local_node_id": representative,
                    "batch": str(meta.loc[node, "batch"]),
                    "cluster_original": str(meta.loc[node, "cluster"]),
                    "cluster": representative_cluster,
                    "n_cells_original": int(meta.loc[node, "n_cells"]),
                    "n_cells_consolidated": int(total_cells),
                    "was_consolidated": len(members) > 1,
                }
            )
        subset = state_table[state_table["local_node_id"].isin(members)].copy()
        for marker, marker_rows in subset.groupby("marker", sort=False):
            del marker
            row = marker_rows.loc[
                marker_rows["local_node_id"].eq(representative)
            ].iloc[0].copy()
            weights = marker_rows["n_cells"].to_numpy(float)
            observed = set(marker_rows["state"].astype(str)) - {UNKNOWN}
            row["state"] = next(iter(observed)) if len(observed) == 1 else UNKNOWN
            for column in numeric_columns:
                if column == "n_cells":
                    continue
                values = pd.to_numeric(marker_rows[column], errors="coerce").to_numpy(float)
                finite = np.isfinite(values)
                if finite.any():
                    row[column] = np.average(values[finite], weights=weights[finite])
            for column in marker_rows.select_dtypes(include=[bool]).columns:
                row[column] = bool(marker_rows[column].any())
            row["batch"] = str(meta.loc[representative, "batch"])
            row["cluster"] = representative_cluster
            row["local_node_id"] = representative
            row["n_cells"] = int(total_cells)
            if row["state"] == UNKNOWN:
                for column in ("call_confidence", "confidence"):
                    if column in row:
                        row[column] = 0.0
            collapsed_parts.append(row.to_frame().T)

    collapsed = pd.concat(collapsed_parts, ignore_index=True)
    for column in state_table.columns:
        try:
            collapsed[column] = collapsed[column].astype(state_table[column].dtype)
        except (TypeError, ValueError):
            pass
    return collapsed[state_table.columns], pd.DataFrame(mapping_rows)


def alignment_marker_reliability(
    state_table: pd.DataFrame,
    *,
    minimum: float = 0.05,
) -> pd.DataFrame:
    """Report local configuration availability without cross-batch rescaling.

    Each batch has its own Gaussian model, so cross-batch posterior magnitudes
    are not used to alter marker weights. Unavailable states remain unknown and
    are ignored pairwise.
    """

    del minimum
    rows = []
    for marker, sub in state_table.groupby("marker", sort=True):
        callable_mask = sub["state"].astype(str).isin({POSITIVE, NEGATIVE})
        total_batches = sub["batch"].astype(str).nunique()
        callable_batches = sub.loc[callable_mask, "batch"].astype(str).nunique()
        rows.append(
            {
                "marker": str(marker),
                "n_batches": int(total_batches),
                "n_callable_batches": int(callable_batches),
                "n_local_clusters": int(len(sub)),
                "n_callable_local_clusters": int(callable_mask.sum()),
                "callable_batch_fraction": float(
                    callable_batches / max(total_batches, 1)
                ),
                "batch_eta_squared": np.nan,
                "alignment_reliability": 1.0,
                "batch_dominated": False,
            }
        )
    return pd.DataFrame(rows)


def _state_specificity(
    signatures: dict[str, dict[str, str]],
    *,
    minimum: float = 0.10,
) -> dict[tuple[str, str], float]:
    """Compute a bounded, data-driven rarity weight for each marker state."""

    floor = float(np.clip(minimum, 0.0, 1.0))
    n_nodes = len(signatures)
    if n_nodes == 0:
        return {}
    markers = sorted(set().union(*(signature for signature in signatures.values())))
    denominator = np.log(n_nodes + 1.0)
    output: dict[tuple[str, str], float] = {}
    for marker in markers:
        for state in (POSITIVE, NEGATIVE):
            count = sum(
                signature.get(marker, UNKNOWN) == state
                for signature in signatures.values()
            )
            rarity = (
                np.log((n_nodes + 1.0) / (count + 1.0)) / denominator
                if denominator > 0
                else 0.0
            )
            output[(marker, state)] = floor + (1.0 - floor) * float(rarity)
    return output


def _aggregate_compatible_signatures(
    members: Iterable[str],
    signatures: dict[str, dict[str, str]],
) -> dict[str, str]:
    """Union compatible evidence; raise if any descendant states conflict."""

    members = list(members)
    markers = sorted(set().union(*(signatures[node] for node in members)))
    output: dict[str, str] = {}
    for marker in markers:
        observed = {
            signatures[node].get(marker, UNKNOWN)
            for node in members
        } - {UNKNOWN}
        if observed == {POSITIVE, NEGATIVE}:
            raise ValueError(f"incompatible members at marker {marker}")
        output[marker] = next(iter(observed)) if observed else UNKNOWN
    return output


def _aggregate_compatible_confidences(
    members: Iterable[str],
    signatures: dict[str, dict[str, str]],
    confidences: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Aggregate aligned evidence and penalize missing member support."""

    members = list(members)
    markers = sorted(set().union(*(signatures[node] for node in members)))
    output: dict[str, float] = {}
    for marker in markers:
        observed = [
            confidences[node].get(marker, 0.0)
            for node in members
            if signatures[node].get(marker, UNKNOWN) in {POSITIVE, NEGATIVE}
        ]
        coverage = len(observed) / len(members)
        output[marker] = (
            float(np.median(observed) * coverage) if observed else 0.0
        )
    return output


def _pair_evidence(
    left: dict[str, str],
    right: dict[str, str],
    left_confidence: dict[str, float] | None = None,
    right_confidence: dict[str, float] | None = None,
    specificity: dict[tuple[str, str], float] | None = None,
) -> dict:
    left_confidence = left_confidence or {}
    right_confidence = right_confidence or {}
    specificity = specificity or {}
    markers = sorted(set(left) | set(right))
    shared_pos = [
        marker for marker in markers
        if left.get(marker, UNKNOWN) == right.get(marker, UNKNOWN) == POSITIVE
    ]
    shared_neg = [
        marker for marker in markers
        if left.get(marker, UNKNOWN) == right.get(marker, UNKNOWN) == NEGATIVE
    ]
    conflicts = [
        marker for marker in markers
        if {left.get(marker, UNKNOWN), right.get(marker, UNKNOWN)}
        == {POSITIVE, NEGATIVE}
    ]
    shared_observed = shared_pos + shared_neg
    observed_left = sum(
        left.get(marker, UNKNOWN) in {POSITIVE, NEGATIVE}
        for marker in markers
    )
    observed_right = sum(
        right.get(marker, UNKNOWN) in {POSITIVE, NEGATIVE}
        for marker in markers
    )
    jointly_observed = len(shared_observed) + len(conflicts)

    def common_weight(marker: str, state: str) -> float:
        confidence = min(
            left_confidence.get(marker, DEFAULT_CONFIDENCE),
            right_confidence.get(marker, DEFAULT_CONFIDENCE),
        )
        return confidence * specificity.get((marker, state), DEFAULT_CONFIDENCE)

    weighted_pos = sum(common_weight(marker, POSITIVE) for marker in shared_pos)
    weighted_neg = sum(common_weight(marker, NEGATIVE) for marker in shared_neg)
    weighted_conflicts = 0.0
    for marker in conflicts:
        confidence = min(
            left_confidence.get(marker, DEFAULT_CONFIDENCE),
            right_confidence.get(marker, DEFAULT_CONFIDENCE),
        )
        state_weight = max(
            specificity.get((marker, POSITIVE), DEFAULT_CONFIDENCE),
            specificity.get((marker, NEGATIVE), DEFAULT_CONFIDENCE),
        )
        weighted_conflicts += confidence * state_weight
    return {
        "n_shared_observed": len(shared_observed),
        "n_observed_left": int(observed_left),
        "n_observed_right": int(observed_right),
        "n_jointly_observed": int(jointly_observed),
        "configuration_coverage_left": float(
            jointly_observed / max(observed_left, 1)
        ),
        "configuration_coverage_right": float(
            jointly_observed / max(observed_right, 1)
        ),
        "minimum_configuration_coverage": float(
            min(
                jointly_observed / max(observed_left, 1),
                jointly_observed / max(observed_right, 1),
            )
        ),
        "n_common_positive": len(shared_pos),
        "n_common_negative": len(shared_neg),
        "n_conflicts": len(conflicts),
        "weighted_shared_observed": weighted_pos + weighted_neg,
        "weighted_common_positive": weighted_pos,
        "weighted_common_negative": weighted_neg,
        "weighted_conflicts": weighted_conflicts,
        "shared_positive_markers": ";".join(shared_pos),
        "shared_negative_markers": ";".join(shared_neg),
        "conflict_markers": ";".join(conflicts),
    }


def _unique_best_partners(
    group_ids: list[str],
    candidate_scores: dict[tuple[str, str], tuple],
    *,
    absolute_tolerance: float = 0.05,
    relative_tolerance: float = 0.02,
) -> dict[str, str]:
    """Return only candidates whose score is uniquely best for each group."""

    def compare(left: tuple, right: tuple) -> int:
        for left_value, right_value in zip(left, right):
            tolerance = max(
                float(absolute_tolerance),
                float(relative_tolerance)
                * max(abs(float(left_value)), abs(float(right_value))),
            )
            difference = float(left_value) - float(right_value)
            if difference > tolerance:
                return 1
            if difference < -tolerance:
                return -1
        return 0

    output: dict[str, str] = {}
    for group in group_ids:
        candidates = []
        for pair, score in candidate_scores.items():
            if group not in pair:
                continue
            other = pair[1] if pair[0] == group else pair[0]
            candidates.append((other, score))
        if not candidates:
            continue
        best = [
            (other, score)
            for other, score in candidates
            if all(
                other == competitor
                or compare(score, competitor_score) > 0
                for competitor, competitor_score in candidates
            )
        ]
        if len(best) == 1:
            output[group] = best[0][0]
    return output


def _robust_group_profile(
    members: Iterable[str],
    signatures: dict[str, dict[str, str]],
    confidences: dict[str, dict[str, float]],
    batches: dict[str, str],
    *,
    consensus_fraction: float = 2.0 / 3.0,
) -> tuple[dict[str, str], dict[str, float]]:
    """Estimate a batch-robust prototype without propagating one-off states."""

    members = list(members)
    n_batches = len({batches[node] for node in members})
    required_batches = 1 if n_batches == 1 else 2
    markers = sorted(set().union(*(signatures[node] for node in members)))
    profile: dict[str, str] = {}
    profile_confidence: dict[str, float] = {}
    for marker in markers:
        votes = {POSITIVE: 0.0, NEGATIVE: 0.0}
        observed_batches: set[str] = set()
        state_confidences = {POSITIVE: [], NEGATIVE: []}
        for node in members:
            state = signatures[node].get(marker, UNKNOWN)
            if state not in {POSITIVE, NEGATIVE}:
                continue
            confidence = float(confidences[node].get(marker, 0.0))
            votes[state] += confidence
            state_confidences[state].append(confidence)
            observed_batches.add(batches[node])
        total = votes[POSITIVE] + votes[NEGATIVE]
        if len(observed_batches) < required_batches or total <= 0:
            profile[marker] = UNKNOWN
            profile_confidence[marker] = 0.0
            continue
        state = max((POSITIVE, NEGATIVE), key=lambda value: votes[value])
        agreement = votes[state] / total
        if agreement < consensus_fraction:
            profile[marker] = UNKNOWN
            profile_confidence[marker] = 0.0
            continue
        coverage = len(observed_batches) / max(n_batches, 1)
        profile[marker] = state
        profile_confidence[marker] = float(
            np.median(state_confidences[state])
            * agreement
            * (0.5 + 0.5 * coverage)
        )
    return profile, profile_confidence


def _alignment_candidate_pairs(
    nodes: list[str],
    signatures: dict[str, dict[str, str]],
    confidences: dict[str, dict[str, float]],
    specificity: dict[tuple[str, str], float],
    batches: dict[str, str],
    *,
    max_neighbors: int | None = None,
    chunk_size: int = 512,
) -> set[tuple[str, str]]:
    """Generate cross-batch candidates using chunked signed-state similarity."""

    if len(nodes) < 2:
        return set()
    markers = sorted(set().union(*(signatures[node] for node in nodes)))
    marker_index = {marker: index for index, marker in enumerate(markers)}
    matrix = np.zeros((len(nodes), len(markers)), dtype=np.float32)
    for row, node in enumerate(nodes):
        for marker, state in signatures[node].items():
            if state not in {POSITIVE, NEGATIVE}:
                continue
            confidence = max(float(confidences[node].get(marker, 0.0)), 0.0)
            state_weight = max(float(specificity.get((marker, state), 1.0)), 0.0)
            sign = 1.0 if state == POSITIVE else -1.0
            matrix[row, marker_index[marker]] = sign * np.sqrt(
                confidence * state_weight
            )
    norms = np.linalg.norm(matrix, axis=1)
    nonzero = norms > 0
    matrix[nonzero] /= norms[nonzero, None]
    n_batches = len(set(batches.values()))
    if max_neighbors is None:
        max_neighbors = max(32, min(128, 2 * n_batches))
    k = min(max(int(max_neighbors), 1), len(nodes) - 1)
    batch_array = np.asarray([batches[node] for node in nodes], dtype=object)
    indices_by_batch = {
        batch: np.flatnonzero(batch_array == batch)
        for batch in sorted(set(batch_array))
    }
    pairs: set[tuple[str, str]] = set()
    for start in range(0, len(nodes), max(int(chunk_size), 1)):
        stop = min(start + max(int(chunk_size), 1), len(nodes))
        similarity = matrix[start:stop] @ matrix.T
        similarity[batch_array[start:stop, None] == batch_array[None, :]] = -np.inf
        local_rows = np.arange(stop - start)
        similarity[local_rows, np.arange(start, stop)] = -np.inf
        if k < len(nodes):
            neighbor_idx = np.argpartition(similarity, -k, axis=1)[:, -k:]
        else:
            neighbor_idx = np.tile(np.arange(len(nodes)), (stop - start, 1))
        for local_row, candidates in enumerate(neighbor_idx):
            source = start + local_row
            for target in candidates:
                if not np.isfinite(similarity[local_row, target]):
                    continue
                left, right = sorted((nodes[source], nodes[int(target)]))
                pairs.add((left, right))
            # Global nearest neighbours can be dominated by abundant batches.
            # Keep two candidates per target batch so mosaic-specific markers
            # cannot hide a valid counterpart or a coarse-to-fine fork.
            for target_batch, target_indices in indices_by_batch.items():
                if target_batch == batch_array[source]:
                    continue
                target_scores = similarity[local_row, target_indices]
                n_keep = min(2, len(target_indices))
                if n_keep < len(target_indices):
                    selected = np.argpartition(target_scores, -n_keep)[-n_keep:]
                    targets = target_indices[selected]
                else:
                    targets = target_indices
                for target in targets:
                    if not np.isfinite(similarity[local_row, target]):
                        continue
                    left, right = sorted((nodes[source], nodes[int(target)]))
                    pairs.add((left, right))
    return pairs


def _coarse_to_fine_pairs(
    pair_evidence: dict[tuple[str, str], dict],
    signatures: dict[str, dict[str, str]],
    confidences: dict[str, dict[str, float]],
    batches: dict[str, str],
    *,
    min_shared_observed: int,
    min_parent_coverage: float = 0.70,
    min_child_confidence: float = 0.50,
) -> set[tuple[str, str]]:
    """Find compatible pairs made ambiguous by a resolving batch fork.

    If one node is compatible with two nodes from the same target batch and
    those target nodes explicitly conflict with one another, the source is a
    coarse parent. It must remain related to both children rather than being
    declared equivalent to whichever child happens to score first.
    """

    compatible_by_target: dict[tuple[str, str], list[str]] = {}
    minimum = max(int(min_shared_observed), 1)
    for (left, right), evidence in pair_evidence.items():
        if evidence["n_conflicts"] or evidence["n_shared_observed"] < minimum:
            continue
        shared = int(evidence["n_shared_observed"])
        if (
            shared / max(int(evidence["n_observed_left"]), 1)
            >= float(min_parent_coverage)
            and int(evidence["n_observed_right"]) > shared
        ):
            compatible_by_target.setdefault(
                (left, batches[right]), []
            ).append(right)
        if (
            shared / max(int(evidence["n_observed_right"]), 1)
            >= float(min_parent_coverage)
            and int(evidence["n_observed_left"]) > shared
        ):
            compatible_by_target.setdefault(
                (right, batches[left]), []
            ).append(left)

    ambiguous_directions: set[tuple[str, str]] = set()
    for key, candidates in compatible_by_target.items():
        candidates = sorted(set(candidates))
        if len(candidates) < 2:
            continue
        for index, left in enumerate(candidates[:-1]):
            parent = key[0]
            parent_signature = signatures[parent]
            resolving_fork = False
            for right in candidates[index + 1:]:
                sibling = _pair_evidence(signatures[left], signatures[right])
                conflict_markers = [
                    marker
                    for marker in sibling["conflict_markers"].split(";")
                    if marker
                ]
                if conflict_markers and all(
                    parent_signature.get(marker, UNKNOWN) == UNKNOWN
                    and min(
                        confidences[left].get(marker, 0.0),
                        confidences[right].get(marker, 0.0),
                    ) >= float(min_child_confidence)
                    for marker in conflict_markers
                ):
                    resolving_fork = True
                    break
            if resolving_fork:
                ambiguous_directions.add(key)
                break

    return {
        pair
        for pair in pair_evidence
        if (pair[0], batches[pair[1]]) in ambiguous_directions
        or (pair[1], batches[pair[0]]) in ambiguous_directions
    }


def _alignment_evidence(
    left: str,
    right: str,
    signatures: dict[str, dict[str, str]],
    confidences: dict[str, dict[str, float]],
    specificity: dict[tuple[str, str], float],
    *,
    use_weighted_scores: bool,
) -> dict:
    evidence = _pair_evidence(
        signatures[left],
        signatures[right],
        confidences[left] if use_weighted_scores else None,
        confidences[right] if use_weighted_scores else None,
        specificity if use_weighted_scores else None,
    )
    conflict_markers = [
        marker for marker in evidence["conflict_markers"].split(";") if marker
    ]
    strong_conflicts = sum(
        min(
            confidences[left].get(marker, DEFAULT_CONFIDENCE),
            confidences[right].get(marker, DEFAULT_CONFIDENCE),
        )
        >= 0.80
        for marker in conflict_markers
    )
    shared = (
        evidence["weighted_shared_observed"]
        if use_weighted_scores
        else float(evidence["n_shared_observed"])
    )
    conflicts = (
        evidence["weighted_conflicts"]
        if use_weighted_scores
        else float(evidence["n_conflicts"])
    )
    agreement = shared / max(shared + conflicts, 1e-12)
    evidence_strength = 1.0 - np.exp(-shared / 3.0)
    evidence.update(
        {
            "n_strong_conflicts": int(strong_conflicts),
            "agreement_fraction": float(agreement),
            "alignment_quality": float(agreement * evidence_strength),
        }
    )
    return evidence


def _batch_conflicts_are_tolerable(
    evidence: dict,
    *,
    max_batch_conflicts: int,
    allow_strong_conflicts: bool = False,
) -> bool:
    """Allow a small number of weak cross-batch state disagreements."""

    return (
        int(evidence["n_conflicts"]) <= max(int(max_batch_conflicts), 0)
        and (
            bool(allow_strong_conflicts)
            or int(evidence["n_strong_conflicts"]) == 0
        )
    )


def align_clusters(
    state_table: pd.DataFrame,
    *,
    min_shared_observed: int = 3,
    minimum_state_specificity: float = 0.10,
    score_absolute_tolerance: float = 0.05,
    score_relative_tolerance: float = 0.02,
    use_weighted_scores: bool = False,
    min_alignment_agreement: float = 0.80,
    min_configuration_coverage: float = 0.45,
    min_leaf_equivalence_coverage: float = 0.55,
    min_alignment_call_confidence: float = 0.50,
    min_prototype_batches: int = 2,
    max_candidate_neighbors: int | None = None,
    max_batch_conflicts: int = 2,
    allow_strong_conflicts: bool = False,
    small_component_max_cells: int = 100,
    small_component_target_ratio: float = 4.0,
    cross_batch_small_component_max_cells: int = 100,
    same_batch_small_component_target_ratio: float = 8.0,
    force_small_component_merge: bool = True,
    prototype_absorb_max_batches: int = 2,
    prototype_absorb_max_cells: int = 1000,
    prototype_absorb_min_agreement: float = 0.70,
    prototype_absorb_min_coverage: float = 0.35,
    collapse_oversplit_components: bool = True,
    oversplit_min_agreement: float = 0.92,
    oversplit_min_shared: int = 5,
    oversplit_min_coverage: float = 0.60,
) -> dict[str, pd.DataFrame]:
    """Align independently fitted configurations across batches.

    Unknown states encode unavailable resolving ability. They are never
    treated as matches, and asymmetric coarse/fine configurations remain
    unresolved unless enough of both configurations is jointly observable.
    Up to ``max_batch_conflicts`` weak disagreements may be absorbed when the
    nearest compatible nodes come from disjoint batches. Strong disagreements
    and same-batch collisions remain hard constraints.
    """

    for name, value in (
        ("min_configuration_coverage", min_configuration_coverage),
        ("min_leaf_equivalence_coverage", min_leaf_equivalence_coverage),
        ("min_alignment_call_confidence", min_alignment_call_confidence),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    equivalence_coverage = max(
        float(min_configuration_coverage),
        float(min_leaf_equivalence_coverage),
    )

    alignment_table = state_table.copy()
    confidence_column = next(
        (
            column
            for column in ("call_confidence", "confidence")
            if column in alignment_table
        ),
        None,
    )
    if confidence_column is None:
        downgraded_calls = alignment_table.iloc[0:0].copy()
        downgraded_calls["original_state"] = pd.Series(dtype=str)
        downgraded_calls["downgrade_reason"] = pd.Series(dtype=str)
    else:
        confidence = pd.to_numeric(
            alignment_table[confidence_column], errors="coerce"
        ).fillna(0.0)
        downgrade = (
            alignment_table["state"].astype(str).isin({POSITIVE, NEGATIVE})
            & (confidence < float(min_alignment_call_confidence))
        )
        downgraded_calls = alignment_table.loc[downgrade].copy()
        downgraded_calls["original_state"] = downgraded_calls["state"].astype(str)
        downgraded_calls["downgrade_reason"] = "low_alignment_call_confidence"
        alignment_table.loc[downgrade, "state"] = UNKNOWN
        alignment_table.loc[downgrade, confidence_column] = 0.0

    meta = (
        state_table[["local_node_id", "batch", "cluster", "n_cells"]]
        .drop_duplicates("local_node_id")
        .set_index("local_node_id")
    )
    signatures = signature_dicts(alignment_table)
    confidences = confidence_dicts(alignment_table)
    marker_reliability = alignment_marker_reliability(alignment_table)
    reliability_map = marker_reliability.set_index("marker")[
        "alignment_reliability"
    ].to_dict()
    for node in confidences:
        for marker in confidences[node]:
            confidences[node][marker] *= float(reliability_map.get(marker, 1.0))
    specificity = (
        _state_specificity(signatures, minimum=minimum_state_specificity)
        if use_weighted_scores
        else {}
    )
    nodes = sorted(signatures)
    batch_by_node = meta["batch"].astype(str).to_dict()
    candidate_pairs = _alignment_candidate_pairs(
        nodes,
        signatures,
        confidences,
        specificity,
        batch_by_node,
        max_neighbors=max_candidate_neighbors,
    )
    pair_evidence = {
        pair: _alignment_evidence(
            pair[0],
            pair[1],
            signatures,
            confidences,
            specificity,
            use_weighted_scores=use_weighted_scores,
        )
        for pair in sorted(candidate_pairs)
    }
    coarse_to_fine_pairs = _coarse_to_fine_pairs(
        pair_evidence,
        signatures,
        confidences,
        batch_by_node,
        min_shared_observed=min_shared_observed,
    )
    for pair, evidence in pair_evidence.items():
        evidence["coarse_to_fine_ambiguous"] = pair in coarse_to_fine_pairs
    eligible_pairs = {
        pair: evidence
        for pair, evidence in pair_evidence.items()
        if evidence["n_shared_observed"] >= max(int(min_shared_observed), 1)
        and _batch_conflicts_are_tolerable(
            evidence,
            max_batch_conflicts=max_batch_conflicts,
            allow_strong_conflicts=allow_strong_conflicts,
        )
        and evidence["agreement_fraction"] >= min_alignment_agreement
        and (
            evidence["n_conflicts"] == 0
            or evidence["minimum_configuration_coverage"] >= equivalence_coverage
        )
    }
    best_quality = {node: -np.inf for node in nodes}
    for (left, right), evidence in eligible_pairs.items():
        quality = evidence["alignment_quality"]
        best_quality[left] = max(best_quality[left], quality)
        best_quality[right] = max(best_quality[right], quality)

    parent = {node: node for node in nodes}
    members = {node: {node} for node in nodes}
    component_batches = {node: {batch_by_node[node]} for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> bool:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return False
        if not component_batches[left_root].isdisjoint(component_batches[right_root]):
            return False
        proposed = members[left_root] | members[right_root]
        component_conflicts = 0
        component_strong_conflicts = 0
        for marker in sorted(set().union(*(signatures[node] for node in proposed))):
            positive = [
                confidences[node].get(marker, 0.0)
                for node in proposed
                if signatures[node].get(marker, UNKNOWN) == POSITIVE
            ]
            negative = [
                confidences[node].get(marker, 0.0)
                for node in proposed
                if signatures[node].get(marker, UNKNOWN) == NEGATIVE
            ]
            if not positive or not negative:
                continue
            component_conflicts += 1
            strength = min(max(positive), max(negative))
            if strength >= 0.80:
                component_strong_conflicts += 1
        if component_conflicts > max(int(max_batch_conflicts), 0):
            return False
        if component_strong_conflicts and not allow_strong_conflicts:
            return False
        prototype_conflicts = 0
        for marker in sorted(
            set().union(*(signatures[node] for node in proposed))
        ):
            if reliability_map.get(marker, 1.0) < 0.25:
                continue
            votes = {POSITIVE: 0.0, NEGATIVE: 0.0}
            for node in proposed:
                state = signatures[node].get(marker, UNKNOWN)
                if state in votes:
                    votes[state] += confidences[node].get(marker, 0.0)
            total = votes[POSITIVE] + votes[NEGATIVE]
            if (
                min(votes.values()) >= 0.80
                and max(votes.values()) / total < 2.0 / 3.0
            ):
                prototype_conflicts += 1
        if prototype_conflicts and not allow_strong_conflicts:
            return False
        if prototype_conflicts > max(int(max_batch_conflicts), 0):
            return False
        if len(members[left_root]) < len(members[right_root]):
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        members[left_root].update(members.pop(right_root))
        component_batches[left_root].update(component_batches.pop(right_root))
        return True

    tolerance = max(float(score_absolute_tolerance), 0.0)
    audit_rows: list[dict] = []
    seed_pairs = []
    for pair, evidence in eligible_pairs.items():
        left, right = pair
        quality = evidence["alignment_quality"]
        near_best_left = quality >= best_quality[left] - tolerance
        near_best_right = quality >= best_quality[right] - tolerance
        if near_best_left and near_best_right:
            seed_pairs.append(pair)
    seed_pairs.sort(
        key=lambda pair: (
            pair_evidence[pair]["alignment_quality"],
            pair_evidence[pair]["n_shared_observed"],
            min(
                int(meta.loc[pair[0], "n_cells"]),
                int(meta.loc[pair[1], "n_cells"]),
            ),
            max(
                int(meta.loc[pair[0], "n_cells"]),
                int(meta.loc[pair[1], "n_cells"]),
            ),
        ),
        reverse=True,
    )

    selected_seed: set[tuple[str, str]] = set()
    for left, right in seed_pairs:
        evidence = pair_evidence[(left, right)]
        if (left, right) in coarse_to_fine_pairs:
            continue
        if union(left, right):
            selected_seed.add((left, right))

    for pair, evidence in pair_evidence.items():
        audit_rows.append(
            {
                "round": 1,
                "stage": "seed",
                "group_a": pair[0],
                "group_b": pair[1],
                "disjoint_batches": batch_by_node[pair[0]] != batch_by_node[pair[1]],
                "eligible": pair in eligible_pairs,
                "score": repr(
                    (
                        evidence["alignment_quality"],
                        evidence["weighted_shared_observed"],
                    )
                ),
                "selected_for_merge": pair in selected_seed,
                "supporting_batches": 0,
                **evidence,
            }
        )

    rescue_round = 1
    while True:
        rescue_round += 1
        roots = sorted({find(node) for node in nodes})
        groups = {root: tuple(sorted(members[root])) for root in roots}
        profiles = {}
        profile_confidences = {}
        for root, group_members in groups.items():
            profile, confidence = _robust_group_profile(
                group_members,
                signatures,
                confidences,
                batch_by_node,
            )
            profiles[root] = profile
            profile_confidences[root] = confidence

        component_candidates: set[tuple[str, str]] = set()
        for left, right in candidate_pairs:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                component_candidates.add(tuple(sorted((left_root, right_root))))
        rescue_evidence: dict[tuple[str, str], dict] = {}
        for left_root, right_root in sorted(component_candidates):
            if not component_batches[left_root].isdisjoint(
                component_batches[right_root]
            ):
                continue
            evidence = _pair_evidence(
                profiles[left_root],
                profiles[right_root],
                profile_confidences[left_root],
                profile_confidences[right_root],
                specificity,
            )
            shared = evidence["weighted_shared_observed"]
            conflicts = evidence["weighted_conflicts"]
            agreement = shared / max(shared + conflicts, 1e-12)
            strong_conflicts = sum(
                min(
                    profile_confidences[left_root].get(marker, 0.0),
                    profile_confidences[right_root].get(marker, 0.0),
                )
                >= 0.80
                for marker in evidence["conflict_markers"].split(";")
                if marker
            )
            supporting_left: set[str] = set()
            supporting_right: set[str] = set()
            refinement_ambiguous = False
            for left_node in groups[left_root]:
                for right_node in groups[right_root]:
                    pair = tuple(sorted((left_node, right_node)))
                    local = pair_evidence.get(pair)
                    if pair in coarse_to_fine_pairs:
                        refinement_ambiguous = True
                    if (
                        local is not None
                        and local["n_shared_observed"] >= max(int(min_shared_observed), 1)
                        and local["agreement_fraction"] >= min_alignment_agreement
                        and _batch_conflicts_are_tolerable(
                            local,
                            max_batch_conflicts=max_batch_conflicts,
                            allow_strong_conflicts=allow_strong_conflicts,
                        )
                        and local["minimum_configuration_coverage"]
                        >= min_configuration_coverage
                    ):
                        supporting_left.add(batch_by_node[left_node])
                        supporting_right.add(batch_by_node[right_node])
            required_left = min(
                max(int(min_prototype_batches), 1),
                len(component_batches[left_root]),
            )
            required_right = min(
                max(int(min_prototype_batches), 1),
                len(component_batches[right_root]),
            )
            eligible = (
                evidence["n_shared_observed"] >= max(int(min_shared_observed), 1)
                and agreement >= min_alignment_agreement
                and _batch_conflicts_are_tolerable(
                    {**evidence, "n_strong_conflicts": strong_conflicts},
                    # Group prototypes can accumulate disagreements through
                    # transitivity, so their rescue budget is deliberately
                    # tighter than the direct cross-batch matching budget.
                    max_batch_conflicts=min(max_batch_conflicts, 1),
                    allow_strong_conflicts=allow_strong_conflicts,
                )
                and evidence["minimum_configuration_coverage"]
                >= min_configuration_coverage
                and not refinement_ambiguous
                and len(supporting_left) >= required_left
                and len(supporting_right) >= required_right
            )
            quality = agreement * (
                1.0 - np.exp(-shared / 3.0)
            )
            rescue_evidence[(left_root, right_root)] = {
                **evidence,
                "n_strong_conflicts": int(strong_conflicts),
                "agreement_fraction": float(agreement),
                "alignment_quality": float(quality),
                "supporting_batches": len(supporting_left | supporting_right),
                "supporting_batches_a": len(supporting_left),
                "supporting_batches_b": len(supporting_right),
                "coarse_to_fine_ambiguous": bool(refinement_ambiguous),
                "eligible": bool(eligible),
            }

        best_for_root = {root: -np.inf for root in roots}
        for (left_root, right_root), evidence in rescue_evidence.items():
            if not evidence["eligible"]:
                continue
            quality = evidence["alignment_quality"]
            best_for_root[left_root] = max(best_for_root[left_root], quality)
            best_for_root[right_root] = max(best_for_root[right_root], quality)
        near_best_for_root = {root: [] for root in roots}
        for (left_root, right_root), evidence in rescue_evidence.items():
            if not evidence["eligible"]:
                continue
            quality = evidence["alignment_quality"]
            if quality >= best_for_root[left_root] - tolerance:
                near_best_for_root[left_root].append(right_root)
            if quality >= best_for_root[right_root] - tolerance:
                near_best_for_root[right_root].append(left_root)
        merge_candidates = []
        def no_same_batch_alternative(source: str, target: str) -> bool:
            target_batches = component_batches[target]
            return not any(
                alternative != target
                and not component_batches[alternative].isdisjoint(target_batches)
                for alternative in near_best_for_root[source]
            )

        for pair, evidence in rescue_evidence.items():
            if not evidence["eligible"]:
                continue
            left_root, right_root = pair
            quality = evidence["alignment_quality"]
            left_multi = len(component_batches[left_root]) > 1
            right_multi = len(component_batches[right_root]) > 1
            left_unique = near_best_for_root[left_root] == [right_root]
            right_unique = near_best_for_root[right_root] == [left_root]
            if (
                (left_multi and right_multi and left_unique and right_unique)
                or (
                    left_multi
                    and not right_multi
                    and right_unique
                    and no_same_batch_alternative(left_root, right_root)
                )
                or (
                    right_multi
                    and not left_multi
                    and left_unique
                    and no_same_batch_alternative(right_root, left_root)
                )
            ):
                merge_candidates.append(pair)
        merge_candidates.sort(
            key=lambda pair: (
                rescue_evidence[pair]["supporting_batches"],
                rescue_evidence[pair]["alignment_quality"],
            ),
            reverse=True,
        )
        selected_rescue: set[tuple[str, str]] = set()
        for left_root, right_root in merge_candidates:
            if union(left_root, right_root):
                selected_rescue.add((left_root, right_root))
        for pair, evidence in rescue_evidence.items():
            audit_rows.append(
                {
                    "round": rescue_round,
                    "stage": "prototype",
                    "group_a": pair[0],
                    "group_b": pair[1],
                    "disjoint_batches": True,
                    "score": repr(
                        (
                            evidence["alignment_quality"],
                            evidence["weighted_shared_observed"],
                        )
                    ),
                    "selected_for_merge": pair in selected_rescue,
                    **evidence,
                }
            )
        if not selected_rescue:
            break

    # Final aligned components with very few cells are often initialization
    # fragments. Absorb them only after stable prototypes have formed, so a
    # tiny weakly conflicting group cannot distort the main alignment pass.
    small_absorption_round = 0
    absorption_specificity = _state_specificity(
        signatures,
        minimum=minimum_state_specificity,
    )
    while max(int(small_component_max_cells), 0) > 0:
        roots = sorted({find(node) for node in nodes})
        component_cells = {
            root: sum(int(meta.loc[node, "n_cells"]) for node in members[root])
            for root in roots
        }
        small_roots = sorted(
            (
                root for root in roots
                if component_cells[root] <= int(small_component_max_cells)
            ),
            key=lambda root: (component_cells[root], root),
        )
        selected_absorption = None
        for small_root in small_roots:
            small_profile, small_confidence = _robust_group_profile(
                members[small_root], signatures, confidences, batch_by_node
            )
            candidates = []
            for target_root in roots:
                if target_root == small_root:
                    continue
                disjoint_batches = component_batches[target_root].isdisjoint(
                    component_batches[small_root]
                )
                if (
                    disjoint_batches
                    and component_cells[small_root]
                    > int(cross_batch_small_component_max_cells)
                ):
                    continue
                refinement_ambiguous = False
                for small_node in members[small_root]:
                    for target_node in members[target_root]:
                        pair = tuple(sorted((small_node, target_node)))
                        if pair in coarse_to_fine_pairs:
                            refinement_ambiguous = True
                            break
                    if refinement_ambiguous:
                        break
                if refinement_ambiguous:
                    continue
                target_ratio = float(small_component_target_ratio)
                if not disjoint_batches:
                    target_ratio = max(
                        target_ratio,
                        float(same_batch_small_component_target_ratio),
                    )
                if component_cells[target_root] < max(
                    2 * component_cells[small_root],
                    target_ratio * component_cells[small_root],
                ):
                    continue
                target_profile, target_confidence = _robust_group_profile(
                    members[target_root], signatures, confidences, batch_by_node
                )
                evidence = _pair_evidence(
                    small_profile,
                    target_profile,
                    small_confidence,
                    target_confidence,
                    absorption_specificity,
                )
                conflict_markers = [
                    marker for marker in evidence["conflict_markers"].split(";")
                    if marker
                ]
                lost_positive_markers = [
                    marker for marker in conflict_markers
                    if small_profile.get(marker, UNKNOWN) == POSITIVE
                    and target_profile.get(marker, UNKNOWN) == NEGATIVE
                ]
                positive_retention = evidence["n_common_positive"] / max(
                    evidence["n_common_positive"] + len(lost_positive_markers),
                    1,
                )
                strong_conflicts = sum(
                    min(
                        small_confidence.get(marker, 0.0),
                        target_confidence.get(marker, 0.0),
                    ) >= 0.80
                    for marker in conflict_markers
                )
                shared = float(evidence["n_shared_observed"])
                conflicts = float(evidence["n_conflicts"])
                agreement = shared / max(shared + conflicts, 1e-12)
                quality = agreement * (1.0 - np.exp(-shared / 3.0))
                required_shared = max(int(min_shared_observed), 1)
                required_agreement = float(min_alignment_agreement)
                required_coverage = float(min_configuration_coverage)
                if not disjoint_batches:
                    # Same-batch absorption is only for clear initialization
                    # fragments. It must be stricter than cross-batch rescue
                    # so rare biological states survive.
                    required_shared = max(required_shared, 5)
                    required_agreement = max(required_agreement, 0.90)
                    required_coverage = max(required_coverage, 0.60)
                eligible = (
                    evidence["n_shared_observed"] >= required_shared
                    and evidence["n_conflicts"] <= max(int(max_batch_conflicts), 0)
                    and (allow_strong_conflicts or strong_conflicts == 0)
                    and agreement >= required_agreement
                    and evidence["minimum_configuration_coverage"]
                    >= required_coverage
                    and evidence["n_common_positive"] >= 1
                    and positive_retention >= 0.60
                )
                candidates.append(
                    (
                        bool(eligible),
                        float(evidence["weighted_common_positive"]),
                        float(quality),
                        -float(evidence["weighted_conflicts"]),
                        float(evidence["minimum_configuration_coverage"]),
                        len(component_batches[target_root]),
                        component_cells[target_root],
                        target_root,
                        {
                            **evidence,
                            "n_strong_conflicts": int(strong_conflicts),
                            "agreement_fraction": float(agreement),
                            "alignment_quality": float(quality),
                            "small_positive_retention": float(positive_retention),
                            "lost_small_positive_markers": ";".join(
                                lost_positive_markers
                            ),
                        },
                    )
                )
            if not candidates:
                continue
            safe_candidates = [candidate for candidate in candidates if candidate[0]]
            if safe_candidates:
                selected = max(safe_candidates)
                merge_policy = "safe_weak_conflict"
            elif force_small_component_merge:
                fallback_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate[-1]["n_shared_observed"] >= 1
                    and candidate[-1]["n_conflicts"]
                    <= max(int(max_batch_conflicts), 0)
                    and (
                        allow_strong_conflicts
                        or candidate[-1]["n_strong_conflicts"] == 0
                    )
                ]
                if not fallback_candidates:
                    continue
                selected = max(fallback_candidates)
                merge_policy = "forced_nearest_large"
            else:
                continue
            eligible, _, _, _, _, _, _, target_root, evidence = selected
            disjoint_before_merge = component_batches[target_root].isdisjoint(
                component_batches[small_root]
            )
            parent[small_root] = target_root
            members[target_root].update(members.pop(small_root))
            component_batches[target_root].update(component_batches.pop(small_root))
            small_absorption_round += 1
            selected_absorption = (small_root, target_root)
            audit_rows.append(
                {
                    "round": rescue_round + small_absorption_round,
                    "stage": "small_component",
                    "group_a": small_root,
                    "group_b": target_root,
                    "disjoint_batches": bool(disjoint_before_merge),
                    "eligible": bool(eligible),
                    "score": repr(
                        (
                            evidence["alignment_quality"],
                            evidence["weighted_shared_observed"],
                        )
                    ),
                    "selected_for_merge": True,
                    "supporting_batches": len(component_batches[target_root]),
                    "small_component_cells": int(component_cells[small_root]),
                    "target_component_cells": int(component_cells[target_root]),
                    "small_component_merge_policy": merge_policy,
                    **evidence,
                }
            )
            break
        if selected_absorption is None:
            break

    # A second, broader pass absorbs batch-local fragments into stable
    # multi-batch prototypes. This is intentionally still configuration-based:
    # it tolerates weak batch-specific disagreements but never strong conflicts
    # or duplicated batches inside one aligned component.
    prototype_absorb_round = 0
    if (
        int(prototype_absorb_max_batches) > 0
        and int(prototype_absorb_max_cells) > 0
    ):
        while True:
            roots = sorted({find(node) for node in nodes})
            component_cells = {
                root: sum(int(meta.loc[node, "n_cells"]) for node in members[root])
                for root in roots
            }
            profiles = {}
            profile_confidences = {}
            for root in roots:
                profile, confidence = _robust_group_profile(
                    members[root], signatures, confidences, batch_by_node
                )
                profiles[root] = profile
                profile_confidences[root] = confidence
            candidates = []
            for source in roots:
                source_batches = component_batches[source]
                if (
                    len(source_batches) > int(prototype_absorb_max_batches)
                    or component_cells[source] > int(prototype_absorb_max_cells)
                ):
                    continue
                for target in roots:
                    if target == source:
                        continue
                    refinement_ambiguous = False
                    for source_node in members[source]:
                        for target_node in members[target]:
                            pair = tuple(sorted((source_node, target_node)))
                            if pair in coarse_to_fine_pairs:
                                refinement_ambiguous = True
                                break
                        if refinement_ambiguous:
                            break
                    if refinement_ambiguous:
                        continue
                    if not component_batches[target].isdisjoint(source_batches):
                        continue
                    if len(component_batches[target]) <= len(source_batches):
                        continue
                    if component_cells[target] < component_cells[source]:
                        continue
                    evidence = _pair_evidence(
                        profiles[source],
                        profiles[target],
                        profile_confidences[source],
                        profile_confidences[target],
                        absorption_specificity,
                    )
                    conflict_markers = [
                        marker for marker in evidence["conflict_markers"].split(";")
                        if marker
                    ]
                    strong_conflicts = sum(
                        min(
                            profile_confidences[source].get(marker, 0.0),
                            profile_confidences[target].get(marker, 0.0),
                        ) >= 0.80
                        for marker in conflict_markers
                    )
                    shared = float(evidence["n_shared_observed"])
                    conflicts = float(evidence["n_conflicts"])
                    agreement = shared / max(shared + conflicts, 1e-12)
                    quality = agreement * (1.0 - np.exp(-shared / 3.0))
                    eligible = (
                        evidence["n_shared_observed"] >= max(int(min_shared_observed), 1)
                        and evidence["n_conflicts"] <= max(int(max_batch_conflicts), 0)
                        and (allow_strong_conflicts or strong_conflicts == 0)
                        and agreement >= float(prototype_absorb_min_agreement)
                        and evidence["minimum_configuration_coverage"]
                        >= float(prototype_absorb_min_coverage)
                        and evidence["n_common_positive"] >= 1
                    )
                    if not eligible:
                        continue
                    candidates.append(
                        (
                            float(quality),
                            float(evidence["weighted_common_positive"]),
                            -float(evidence["weighted_conflicts"]),
                            len(component_batches[target]),
                            component_cells[target],
                            -component_cells[source],
                            source,
                            target,
                            {
                                **evidence,
                                "n_strong_conflicts": int(strong_conflicts),
                                "agreement_fraction": float(agreement),
                                "alignment_quality": float(quality),
                            },
                        )
                    )
            if not candidates:
                break
            _, _, _, _, _, _, source, target, evidence = max(candidates)
            disjoint_before_merge = component_batches[target].isdisjoint(
                component_batches[source]
            )
            parent[source] = target
            members[target].update(members.pop(source))
            component_batches[target].update(component_batches.pop(source))
            prototype_absorb_round += 1
            audit_rows.append(
                {
                    "round": rescue_round
                    + small_absorption_round
                    + prototype_absorb_round,
                    "stage": "prototype_absorption",
                    "group_a": source,
                    "group_b": target,
                    "disjoint_batches": bool(disjoint_before_merge),
                    "eligible": True,
                    "score": repr(
                        (
                            evidence["alignment_quality"],
                            evidence["weighted_shared_observed"],
                        )
                    ),
                    "selected_for_merge": True,
                    "supporting_batches": len(component_batches[target]),
                    "small_component_cells": int(component_cells[source]),
                    "target_component_cells": int(component_cells[target]),
                    "small_component_merge_policy": "prototype_absorption",
                    **evidence,
                }
            )

    oversplit_round = 0
    if collapse_oversplit_components:
        while True:
            roots = sorted({find(node) for node in nodes})
            component_cells = {
                root: sum(
                    int(meta.loc[node, "n_cells"])
                    for node in members[root]
                )
                for root in roots
            }
            profiles = {}
            profile_confidences = {}
            for root in roots:
                profile, confidence = _robust_group_profile(
                    members[root], signatures, confidences, batch_by_node
                )
                profiles[root] = profile
                profile_confidences[root] = confidence
            candidates = []
            for left_index, left in enumerate(roots[:-1]):
                for right in roots[left_index + 1:]:
                    # Equivalent aligned components may contain at most one
                    # node from each batch. Same-batch strict duplicates were
                    # already handled by local consolidation; any remaining
                    # granularity difference belongs in the taxonomy.
                    if not component_batches[left].isdisjoint(
                        component_batches[right]
                    ):
                        continue
                    refinement_ambiguous = False
                    for left_node in members[left]:
                        for right_node in members[right]:
                            pair = tuple(sorted((left_node, right_node)))
                            if pair in coarse_to_fine_pairs:
                                refinement_ambiguous = True
                                break
                        if refinement_ambiguous:
                            break
                    if refinement_ambiguous:
                        continue
                    evidence = _pair_evidence(
                        profiles[left],
                        profiles[right],
                        profile_confidences[left],
                        profile_confidences[right],
                        absorption_specificity,
                    )
                    if evidence["n_shared_observed"] < max(
                        int(oversplit_min_shared), 1
                    ):
                        continue
                    shared = float(evidence["n_shared_observed"])
                    conflicts = float(evidence["n_conflicts"])
                    agreement = shared / max(shared + conflicts, 1e-12)
                    strong_conflicts = sum(
                        min(
                            profile_confidences[left].get(marker, 0.0),
                            profile_confidences[right].get(marker, 0.0),
                        ) >= 0.80
                        for marker in evidence["conflict_markers"].split(";")
                        if marker
                    )
                    if conflicts > 0:
                        smaller = min(component_cells[left], component_cells[right])
                        larger = max(component_cells[left], component_cells[right])
                        shared_batches = not component_batches[left].isdisjoint(
                            component_batches[right]
                        )
                        weak_fragment = (
                            shared_batches
                            and smaller <= 500
                            and larger >= 8.0 * smaller
                            and conflicts <= 1
                            and strong_conflicts == 0
                            and agreement >= 0.90
                            and evidence["minimum_configuration_coverage"] >= 0.60
                        )
                        if not weak_fragment:
                            continue
                    else:
                        if agreement < float(oversplit_min_agreement):
                            continue
                        if (
                            evidence["minimum_configuration_coverage"]
                            < float(oversplit_min_coverage)
                        ):
                            continue
                    quality = agreement * (1.0 - np.exp(-shared / 3.0))
                    candidates.append(
                        (
                            float(quality),
                            float(evidence["weighted_common_positive"]),
                            min(
                                component_cells[left],
                                component_cells[right],
                            ),
                            left,
                            right,
                            {
                                **evidence,
                                "n_strong_conflicts": int(strong_conflicts),
                                "agreement_fraction": float(agreement),
                                "alignment_quality": float(quality),
                            },
                        )
                    )
            if not candidates:
                break
            _, _, _, left, right, evidence = max(candidates)
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                continue
            if len(members[left_root]) < len(members[right_root]):
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root
            members[left_root].update(members.pop(right_root))
            component_batches[left_root].update(component_batches.pop(right_root))
            oversplit_round += 1
            audit_rows.append(
                {
                    "round": rescue_round
                    + small_absorption_round
                    + prototype_absorb_round
                    + oversplit_round,
                    "stage": "oversplit_collapse",
                    "group_a": left_root,
                    "group_b": right_root,
                    "disjoint_batches": False,
                    "eligible": True,
                    "score": repr(
                        (
                            evidence["alignment_quality"],
                            evidence["weighted_shared_observed"],
                        )
                    ),
                    "selected_for_merge": True,
                    "supporting_batches": len(component_batches[left_root]),
                    **evidence,
                }
            )

    final_groups = {
        root: tuple(sorted(members[root]))
        for root in sorted({find(node) for node in nodes})
    }
    ordered_groups = sorted(
        final_groups.values(),
        key=lambda members: (-len(members), members),
    )
    membership_rows = []
    state_rows = []
    for index, members in enumerate(ordered_groups, start=1):
        aligned_id = f"ALIGNED_{index:04d}"
        aggregate, aggregate_confidence = _robust_group_profile(
            members,
            signatures,
            confidences,
            batch_by_node,
        )
        for node in members:
            row = meta.loc[node]
            membership_rows.append(
                {
                    "aligned_node_id": aligned_id,
                    "local_node_id": node,
                    "batch": str(row["batch"]),
                    "cluster": str(row["cluster"]),
                    "n_cells": int(row["n_cells"]),
                }
            )
        for marker, state in sorted(aggregate.items()):
            observed_members = sum(
                signatures[node].get(marker, UNKNOWN) in {POSITIVE, NEGATIVE}
                for node in members
            )
            state_rows.append(
                {
                    "aligned_node_id": aligned_id,
                    "marker": marker,
                    "state": state,
                    "confidence": aggregate_confidence[marker],
                    "observed_member_fraction": observed_members / len(members),
                    "state_specificity": (
                        specificity.get((marker, state), 0.0)
                        if state in {POSITIVE, NEGATIVE}
                        else 0.0
                    ),
                }
            )
    return {
        "membership": pd.DataFrame(membership_rows),
        "states": pd.DataFrame(state_rows),
        "pair_audit": pd.DataFrame(audit_rows),
        "marker_reliability": marker_reliability,
        "downgraded_calls": downgraded_calls,
    }


def build_resolution_aware_atlas(
    aligned_membership: pd.DataFrame,
    aligned_states: pd.DataFrame,
    *,
    local_state_table: pd.DataFrame | None = None,
    min_shared_observed: int = 3,
    min_parent_coverage: float = 0.95,
    min_compatible_coverage: float = 0.50,
    min_extra_child_states: int = 1,
    min_common_positive: int = 1,
    min_parent_state_homogeneity: float = 0.80,
    minimum_state_specificity: float = 0.10,
) -> dict[str, pd.DataFrame | dict]:
    """Infer configuration containment without forcing coarse nodes to leaves.

    Equivalent configurations have already been grouped by :func:`align_clusters`.
    Here, an aligned node can become an internal atlas node when its reliable
    states are a conflict-free subset of a more resolved node. Each child gets
    at most one immediate parent, chosen as the most specific compatible
    ancestor; all other compatible relations remain available in the audit.
    """

    for name, value in (
        ("min_parent_coverage", min_parent_coverage),
        ("min_compatible_coverage", min_compatible_coverage),
        ("min_parent_state_homogeneity", min_parent_state_homogeneity),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    signatures = {
        str(node): dict(zip(sub["marker"].astype(str), sub["state"].astype(str)))
        for node, sub in aligned_states.groupby("aligned_node_id", sort=True)
    }
    confidences = confidence_dicts(aligned_states, node_column="aligned_node_id")
    specificity = _state_specificity(
        signatures,
        minimum=minimum_state_specificity,
    )
    membership_meta = aligned_membership.groupby("aligned_node_id", sort=True).agg(
        n_local_clusters=("local_node_id", "nunique"),
        n_batches=("batch", "nunique"),
        n_cells=("n_cells", "sum"),
    )
    observed_count = {
        node: sum(state in {POSITIVE, NEGATIVE} for state in signature.values())
        for node, signature in signatures.items()
    }

    # A mixed local cluster may have a dominant +/- call while still carrying
    # substantial support for the opposite state. Such a marker remains in the
    # reported configuration, but cannot be used to force that cluster to a
    # resolved leaf.
    coarse_signatures = {node: dict(signature) for node, signature in signatures.items()}
    coarse_confidences = {node: dict(values) for node, values in confidences.items()}
    if (
        local_state_table is not None
        and "cluster_state_fraction" in local_state_table
    ):
        local_to_aligned = aligned_membership.set_index("local_node_id")[
            "aligned_node_id"
        ].astype(str)
        local = local_state_table.copy()
        local["aligned_node_id"] = local["local_node_id"].astype(str).map(
            local_to_aligned
        )
        local = local[local["aligned_node_id"].notna()].copy()
        for (node, marker), sub in local.groupby(
            ["aligned_node_id", "marker"],
            sort=False,
        ):
            values = pd.to_numeric(
                sub["cluster_state_fraction"], errors="coerce"
            ).to_numpy(float)
            weights = pd.to_numeric(sub["n_cells"], errors="coerce").fillna(1.0)
            finite = np.isfinite(values)
            homogeneity = (
                float(np.average(values[finite], weights=weights.to_numpy(float)[finite]))
                if finite.any()
                else 0.0
            )
            if homogeneity < float(min_parent_state_homogeneity):
                coarse_signatures[str(node)][str(marker)] = UNKNOWN
                coarse_confidences[str(node)][str(marker)] = 0.0

    coarse_observed_count = {
        node: sum(state in {POSITIVE, NEGATIVE} for state in signature.values())
        for node, signature in coarse_signatures.items()
    }
    batches_by_node = {
        str(node): set(sub["batch"].astype(str))
        for node, sub in aligned_membership.groupby("aligned_node_id", sort=True)
    }

    relation_rows: list[dict] = []
    relation_by_pair: dict[tuple[str, str], dict] = {}
    compatible_by_parent: dict[str, list[str]] = {node: [] for node in signatures}
    for parent, child in itertools.permutations(sorted(signatures), 2):
        evidence = _pair_evidence(
            coarse_signatures[parent],
            signatures[child],
            coarse_confidences[parent],
            confidences[child],
            specificity,
        )
        parent_coverage = (
            evidence["n_shared_observed"] / max(coarse_observed_count[parent], 1)
        )
        compatible_coverage = evidence["n_shared_observed"] / max(
            min(coarse_observed_count[parent], observed_count[child]),
            1,
        )
        extra_child_states = observed_count[child] - evidence["n_shared_observed"]
        compatible = (
            evidence["n_conflicts"] == 0
            and evidence["n_shared_observed"] >= max(int(min_shared_observed), 1)
            and evidence["n_common_positive"] >= max(int(min_common_positive), 0)
            and compatible_coverage >= float(min_compatible_coverage)
        )
        subset_eligible = (
            compatible
            and parent_coverage >= float(min_parent_coverage)
            and extra_child_states >= max(int(min_extra_child_states), 1)
            and coarse_observed_count[parent] < observed_count[child]
        )
        row = {
            "parent": parent,
            "child": child,
            "relation": "coarse_to_fine",
            "eligible": bool(subset_eligible),
            "selected": False,
            "relation_reason": "configuration_subset" if subset_eligible else "",
            "parent_observed_states": coarse_observed_count[parent],
            "child_observed_states": observed_count[child],
            "parent_coverage": float(parent_coverage),
            "compatible_coverage": float(compatible_coverage),
            "extra_child_states": int(extra_child_states),
            **evidence,
        }
        relation_rows.append(row)
        relation_by_pair[(parent, child)] = row
        if compatible:
            compatible_by_parent[parent].append(child)

    # Ambiguity witness: if one configuration is compatible with two nodes
    # from the same batch that conflict with each other, it cannot identify
    # which fine state is present. The broad node therefore remains internal.
    for parent, children in compatible_by_parent.items():
        for left, right in itertools.combinations(sorted(children), 2):
            if batches_by_node[left].isdisjoint(batches_by_node[right]):
                continue
            sibling_evidence = _pair_evidence(
                signatures[left],
                signatures[right],
                confidences[left],
                confidences[right],
                specificity,
            )
            if sibling_evidence["n_conflicts"] == 0:
                continue
            sibling_conflict_markers = [
                marker
                for marker in sibling_evidence["conflict_markers"].split(";")
                if marker
            ]
            unstable_witness = any(
                signatures[parent].get(marker, UNKNOWN) in {POSITIVE, NEGATIVE}
                and coarse_signatures[parent].get(marker, UNKNOWN) == UNKNOWN
                for marker in sibling_conflict_markers
            )
            if sibling_evidence["n_conflicts"] < 3 and not unstable_witness:
                continue
            for child in (left, right):
                row = relation_by_pair[(parent, child)]
                row["eligible"] = True
                row["relation_reason"] = "ambiguous_between_conflicting_siblings"
                row["sibling_witness"] = right if child == left else left
                row["n_sibling_conflicts"] = sibling_evidence["n_conflicts"]
                row["unstable_conflict_witness"] = bool(unstable_witness)

    eligible_by_child: dict[str, list[dict]] = {node: [] for node in signatures}
    eligible_children: dict[str, list[str]] = {node: [] for node in signatures}
    for row in relation_rows:
        if row["eligible"]:
            eligible_by_child[row["child"]].append(row)
            eligible_children[row["parent"]].append(row["child"])

    selected_edges: list[dict] = []
    for child, candidates in eligible_by_child.items():
        if not candidates:
            continue
        candidates.sort(
            key=lambda row: (
                row["parent_observed_states"],
                row["weighted_shared_observed"],
                int(membership_meta.loc[row["parent"], "n_batches"]),
                row["parent"],
            ),
            reverse=True,
        )
        selected = candidates[0]
        selected["selected"] = True
        selected_edges.append(
            {
                "parent": selected["parent"],
                "child": child,
                "relation": "coarse_to_fine",
            }
        )

    edges = pd.DataFrame(
        selected_edges,
        columns=["parent", "child", "relation"],
    )
    children_by_parent = {
        parent: sorted(sub["child"].astype(str))
        for parent, sub in edges.groupby("parent", sort=True)
    } if not edges.empty else {}
    parent_by_child = (
        edges.set_index("child")["parent"].astype(str).to_dict()
        if not edges.empty
        else {}
    )

    graph_children = {
        parent: sorted(set(children))
        for parent, children in eligible_children.items()
        if children
    }

    def descendant_leaves(node: str, trail: frozenset[str] = frozenset()) -> tuple[str, ...]:
        if node in trail:
            return ()
        children = graph_children.get(node, [])
        if not children:
            return (node,)
        leaves: list[str] = []
        for child in children:
            leaves.extend(descendant_leaves(child, trail | {node}))
        return tuple(sorted(set(leaves))) or (node,)

    node_rows = []
    for node in sorted(signatures):
        leaves = descendant_leaves(node)
        is_internal = node in graph_children
        reasons = {
            relation_by_pair[(node, child)]["relation_reason"]
            for child in graph_children.get(node, [])
        }
        resolution_reason = (
            "ambiguous_between_conflicting_siblings"
            if "ambiguous_between_conflicting_siblings" in reasons
            else "configuration_subset"
            if reasons
            else "maximally_resolved_configuration"
        )
        node_rows.append(
            {
                "atlas_node_id": node,
                "atlas_node_kind": "internal" if is_internal else "leaf",
                "is_root": node not in parent_by_child,
                "n_observed_states": observed_count[node],
                "n_positive_states": sum(
                    state == POSITIVE for state in signatures[node].values()
                ),
                "n_negative_states": sum(
                    state == NEGATIVE for state in signatures[node].values()
                ),
                "n_local_clusters": int(membership_meta.loc[node, "n_local_clusters"]),
                "n_batches": int(membership_meta.loc[node, "n_batches"]),
                "n_cells": int(membership_meta.loc[node, "n_cells"]),
                "candidate_leaf_ids": ";".join(leaves),
                "n_candidate_leaves": len(leaves),
                "resolution_reason": resolution_reason,
            }
        )
    nodes = pd.DataFrame(node_rows)
    node_kind = nodes.set_index("atlas_node_id")["atlas_node_kind"].to_dict()
    candidate_leaves = nodes.set_index("atlas_node_id")["candidate_leaf_ids"].to_dict()
    resolution_reason = nodes.set_index("atlas_node_id")["resolution_reason"].to_dict()
    assignments = aligned_membership.copy()
    assignments["atlas_node_id"] = assignments["aligned_node_id"].astype(str)
    assignments["atlas_resolution"] = assignments["atlas_node_id"].map(node_kind)
    assignments["candidate_leaf_ids"] = assignments["atlas_node_id"].map(
        candidate_leaves
    )
    assignments["assignment_reason"] = assignments["atlas_node_id"].map(
        resolution_reason
    )
    roots = sorted(set(signatures) - set(parent_by_child))
    forest = {
        "roots": roots,
        "children": children_by_parent,
    }
    return {
        "nodes": nodes,
        "edges": edges,
        "relations": pd.DataFrame(relation_rows),
        "assignments": assignments,
        "forest": forest,
    }


def _intersect_parent_state(
    left: dict[str, str],
    right: dict[str, str],
) -> dict[str, str]:
    markers = sorted(set(left) | set(right))
    return {
        marker: (
            left.get(marker, UNKNOWN)
            if left.get(marker, UNKNOWN) == right.get(marker, UNKNOWN)
            and left.get(marker, UNKNOWN) in {POSITIVE, NEGATIVE}
            else UNKNOWN
        )
        for marker in markers
    }


def build_taxonomy(
    aligned_membership: pd.DataFrame,
    aligned_states: pd.DataFrame,
    *,
    min_tree_shared_observed: int = 1,
    min_tree_weighted_shared: float = 0.0,
    max_tree_conflicts: int | None = None,
    minimum_state_specificity: float = 0.10,
    score_absolute_tolerance: float = 0.05,
    score_relative_tolerance: float = 0.02,
    use_weighted_scores: bool = True,
) -> dict[str, pd.DataFrame | dict]:
    """Build a bottom-up forest with lexicographic, ambiguity-safe merging."""

    leaf_signatures = {
        str(node): dict(zip(sub["marker"].astype(str), sub["state"].astype(str)))
        for node, sub in aligned_states.groupby("aligned_node_id", sort=True)
    }
    leaf_confidences = confidence_dicts(
        aligned_states,
        node_column="aligned_node_id",
    )
    specificity = (
        _state_specificity(leaf_signatures, minimum=minimum_state_specificity)
        if use_weighted_scores
        else {}
    )
    membership = {
        str(node): tuple(sorted(sub["local_node_id"].astype(str)))
        for node, sub in aligned_membership.groupby("aligned_node_id", sort=True)
    }
    active = sorted(leaf_signatures)
    signatures = dict(leaf_signatures)
    confidences = dict(leaf_confidences)
    descendants = {node: (node,) for node in active}
    node_kind = {node: "aligned_leaf" for node in active}
    edges: list[dict] = []
    merges: list[dict] = []
    pair_audit: list[dict] = []
    next_internal = 1
    round_id = 0

    while len(active) > 1:
        round_id += 1
        scores: dict[tuple[str, str], tuple] = {}
        evidence_by_pair: dict[tuple[str, str], dict] = {}
        for left, right in itertools.combinations(sorted(active), 2):
            evidence = _pair_evidence(
                signatures[left],
                signatures[right],
                confidences[left] if use_weighted_scores else None,
                confidences[right] if use_weighted_scores else None,
                specificity if use_weighted_scores else None,
            )
            evidence_by_pair[(left, right)] = evidence
            if evidence["n_shared_observed"] < max(
                int(min_tree_shared_observed),
                1,
            ):
                continue
            if (
                use_weighted_scores
                and evidence["weighted_shared_observed"]
                < max(float(min_tree_weighted_shared), 0.0)
            ):
                continue
            if (
                max_tree_conflicts is not None
                and evidence["n_conflicts"] > max(int(max_tree_conflicts), 0)
            ):
                continue
            score = (
                -evidence["weighted_conflicts"],
                evidence["weighted_common_positive"],
                evidence["weighted_common_negative"],
                evidence["minimum_configuration_coverage"],
            )
            scores[(left, right)] = score

        unique_best = _unique_best_partners(
            sorted(active),
            scores,
            absolute_tolerance=score_absolute_tolerance if use_weighted_scores else 0.0,
            relative_tolerance=score_relative_tolerance if use_weighted_scores else 0.0,
        )
        merge_pairs = sorted(
            {
                tuple(sorted((left, right)))
                for left, right in unique_best.items()
                if unique_best.get(right) == left
            }
        )
        for pair, evidence in evidence_by_pair.items():
            score = scores.get(pair)
            pair_audit.append(
                {
                    "round": round_id,
                    "node_a": pair[0],
                    "node_b": pair[1],
                    "score": repr(score) if score is not None else "",
                    "eligible": score is not None,
                    "is_unique_best_for_a": unique_best.get(pair[0]) == pair[1],
                    "is_unique_best_for_b": unique_best.get(pair[1]) == pair[0],
                    "selected_for_merge": pair in merge_pairs,
                    **evidence,
                }
            )
        if not merge_pairs:
            break

        merged_nodes = set(itertools.chain.from_iterable(merge_pairs))
        new_active = [node for node in active if node not in merged_nodes]
        for left, right in merge_pairs:
            parent = f"TAXON_{next_internal:04d}"
            next_internal += 1
            evidence = evidence_by_pair[(left, right)]
            signatures[parent] = _intersect_parent_state(
                signatures[left], signatures[right]
            )
            confidences[parent] = {
                marker: (
                    min(
                        confidences[left].get(marker, 0.0),
                        confidences[right].get(marker, 0.0),
                    )
                    if state in {POSITIVE, NEGATIVE}
                    else 0.0
                )
                for marker, state in signatures[parent].items()
            }
            descendants[parent] = tuple(
                sorted(descendants[left] + descendants[right])
            )
            node_kind[parent] = "internal"
            edges.extend(
                [
                    {"parent": parent, "child": left},
                    {"parent": parent, "child": right},
                ]
            )
            merges.append(
                {
                    "round": round_id,
                    "parent": parent,
                    "child_a": left,
                    "child_b": right,
                    "score": repr(
                        (
                            -evidence["weighted_conflicts"],
                            evidence["weighted_common_positive"],
                            evidence["weighted_common_negative"],
                            evidence["minimum_configuration_coverage"],
                        )
                    ),
                    **evidence,
                }
            )
            new_active.append(parent)
        active = sorted(new_active)

    roots = sorted(active)
    all_local_members = {
        node: tuple(
            sorted(
                itertools.chain.from_iterable(
                    membership[leaf] for leaf in descendants[node]
                )
            )
        )
        for node in signatures
    }
    node_rows = []
    state_rows = []
    for node in sorted(signatures):
        sig = signatures[node]
        node_rows.append(
            {
                "tree_node_id": node,
                "node_kind": node_kind[node],
                "is_root": node in roots,
                "n_aligned_leaves": len(descendants[node]),
                "n_local_clusters": len(all_local_members[node]),
                "n_positive_markers": sum(x == POSITIVE for x in sig.values()),
                "n_negative_markers": sum(x == NEGATIVE for x in sig.values()),
                "n_unknown_markers": sum(x == UNKNOWN for x in sig.values()),
                "mean_observed_confidence": float(
                    np.mean(
                        [
                            confidences[node].get(marker, 0.0)
                            for marker, state in sig.items()
                            if state in {POSITIVE, NEGATIVE}
                        ]
                    )
                )
                if any(state in {POSITIVE, NEGATIVE} for state in sig.values())
                else 0.0,
                "aligned_leaf_descendants": ";".join(descendants[node]),
                "local_cluster_descendants": ";".join(all_local_members[node]),
            }
        )
        for marker, state in sorted(sig.items()):
            state_rows.append(
                {
                    "tree_node_id": node,
                    "marker": marker,
                    "state": state,
                    "confidence": confidences[node].get(marker, 0.0),
                    "state_specificity": (
                        specificity.get((marker, state), 0.0)
                        if state in {POSITIVE, NEGATIVE}
                        else 0.0
                    ),
                }
            )

    forest = {
        "roots": roots,
        "children": {
            node: [
                edge["child"] for edge in edges if edge["parent"] == node
            ]
            for node in sorted(signatures)
            if any(edge["parent"] == node for edge in edges)
        },
    }
    return {
        "nodes": pd.DataFrame(node_rows),
        "edges": pd.DataFrame(edges, columns=["parent", "child"]),
        "merges": pd.DataFrame(merges) if merges else pd.DataFrame(columns=["parent", "child_a", "child_b", "round"]),
        "states": pd.DataFrame(state_rows),
        "pair_audit": pd.DataFrame(pair_audit),
        "forest": forest,
    }


def build_cell_tree_assignments(
    cell_clusters: pd.DataFrame,
    aligned_membership: pd.DataFrame,
    tree_edges: pd.DataFrame,
    atlas_assignments: pd.DataFrame | None = None,
    tree_merges: pd.DataFrame | None = None,
    consensus_up_level: int = 2,
    final_min_shared_observed: int = 3,
    final_min_common_positive: int = 1,
    final_max_conflicts: int = 1,
    final_max_conflict_fraction: float = 0.10,
    merge_batch_complementary_children: bool = True,
    batch_complement_max_conflicts: int = 3,
    batch_complement_max_conflict_fraction: float = 0.20,
    primary_readout: str = "taxonomy",
) -> pd.DataFrame:
    """Attach aligned leaves and root-to-leaf taxonomy paths to cells."""

    cells = cell_clusters.copy()
    cells["local_node_id"] = (
        cells["batch"].astype(str) + "|" + cells["protein_phenotype"].astype(str)
    )
    leaf_map = aligned_membership.set_index("local_node_id")[
        "aligned_node_id"
    ].astype(str)
    cells["aligned_leaf_id"] = cells["local_node_id"].map(leaf_map)
    cells["cytofuse_leaf_id"] = cells["aligned_leaf_id"]
    if atlas_assignments is not None:
        atlas = atlas_assignments.drop_duplicates("local_node_id").set_index(
            "local_node_id"
        )
        for column in (
            "atlas_node_id",
            "atlas_resolution",
            "candidate_leaf_ids",
            "assignment_reason",
        ):
            cells[column] = cells["local_node_id"].map(atlas[column])
    if tree_edges.empty:
        parent_of: dict[str, str] = {}
    else:
        required_columns = {"parent", "child"}
        missing = required_columns - set(tree_edges.columns)
        if missing:
            raise ValueError(f"tree_edges is missing columns: {sorted(missing)}")
        children = tree_edges["child"].astype(str)
        if children.duplicated().any():
            duplicated = sorted(set(children[children.duplicated()].tolist()))
            raise ValueError(
                "taxonomy must be a forest with at most one parent per child; "
                f"duplicated children: {duplicated}"
            )
        parent_of = tree_edges.set_index("child")["parent"].astype(str).to_dict()
        for start in set(parent_of) | set(parent_of.values()):
            seen: set[str] = set()
            node = str(start)
            while node in parent_of:
                if node in seen:
                    raise ValueError(f"cycle detected in taxonomy at {node}")
                seen.add(node)
                node = parent_of[node]

    def path_for(leaf: str) -> str:
        if pd.isna(leaf):
            return ""
        path = [str(leaf)]
        seen = set(path)
        while path[-1] in parent_of:
            parent = parent_of[path[-1]]
            if parent in seen:
                raise ValueError("cycle detected in taxonomy")
            path.append(parent)
            seen.add(parent)
        return ";".join(reversed(path))

    cells["taxonomy_path"] = cells["aligned_leaf_id"].map(path_for)
    cells["taxonomy_root"] = cells["taxonomy_path"].str.split(";").str[0]
    up_level = max(int(consensus_up_level), 0)

    def consensus_for(path: str) -> str:
        parts = [part for part in str(path).split(";") if part]
        if not parts:
            return ""
        return parts[max(0, len(parts) - 1 - up_level)]

    cells["cytofuse_consensus_id"] = cells["taxonomy_path"].map(consensus_for)
    final_by_leaf = {
        leaf: leaf for leaf in cells["aligned_leaf_id"].dropna().astype(str).unique()
    }
    final_reason_by_leaf = {
        leaf: "retained_aligned_leaf" for leaf in final_by_leaf
    }
    if "batch" in aligned_membership:
        leaf_batches = {
            str(node): set(group["batch"].dropna().astype(str))
            for node, group in aligned_membership.groupby("aligned_node_id")
        }
    else:
        leaf_batches = {
            str(node): {
                str(local_node).split("|", 1)[0]
                for local_node in group["local_node_id"].dropna().astype(str)
            }
            for node, group in aligned_membership.groupby("aligned_node_id")
        }
    node_batches = {node: set(batches) for node, batches in leaf_batches.items()}
    available = set(final_by_leaf)
    if tree_merges is not None and not tree_merges.empty:
        required_shared = max(int(final_min_shared_observed), 1)
        required_positive = max(int(final_min_common_positive), 0)
        allowed_conflicts = max(int(final_max_conflicts), 0)
        allowed_fraction = float(final_max_conflict_fraction)
        if not 0.0 <= allowed_fraction <= 1.0:
            raise ValueError("final_max_conflict_fraction must be between 0 and 1")
        complement_allowed_conflicts = max(
            int(batch_complement_max_conflicts), 0
        )
        complement_allowed_fraction = float(
            batch_complement_max_conflict_fraction
        )
        if not 0.0 <= complement_allowed_fraction <= 1.0:
            raise ValueError(
                "batch_complement_max_conflict_fraction must be between 0 and 1"
            )
        for row in tree_merges.sort_values(["round", "parent"]).itertuples(
            index=False
        ):
            left, right, parent = str(row.child_a), str(row.child_b), str(row.parent)
            shared = int(row.n_shared_observed)
            conflicts = int(row.n_conflicts)
            left_batches = node_batches.get(left, set())
            right_batches = node_batches.get(right, set())
            node_batches[parent] = left_batches | right_batches
            batch_complementary = bool(
                left_batches - right_batches
                and right_batches - left_batches
            )
            if left not in available or right not in available:
                continue
            regular_evidence = (
                shared >= required_shared
                and int(row.n_common_positive) >= required_positive
                and conflicts <= allowed_conflicts
                and conflicts / max(shared, 1) <= allowed_fraction
            )
            complementary_evidence = (
                bool(merge_batch_complementary_children)
                and batch_complementary
                and shared >= required_shared
                and int(row.n_common_positive) >= required_positive
                and conflicts <= complement_allowed_conflicts
                and conflicts / max(shared, 1) <= complement_allowed_fraction
            )
            # A single-batch child can be absorbed at its candidate tree
            # parent; downstream RNA refinement can recover finer classes.
            # Retain shared-marker and conflict checks, but do not require
            # positive overlap or disjoint batch support for this override.
            single_batch_evidence = (
                (len(left_batches) == 1 or len(right_batches) == 1)
                and shared >= required_shared
                and conflicts <= allowed_conflicts
                and conflicts / max(shared, 1) <= allowed_fraction
            )
            if not (regular_evidence or complementary_evidence or single_batch_evidence):
                continue
            merge_reason = (
                "regular_configuration_evidence" if regular_evidence
                else "batch_complementary_siblings" if complementary_evidence
                else "single_batch_child_absorption"
            )
            for leaf, node in list(final_by_leaf.items()):
                if node in {left, right}:
                    final_by_leaf[leaf] = parent
                    final_reason_by_leaf[leaf] = merge_reason
            available.remove(left)
            available.remove(right)
            available.add(parent)
    if primary_readout not in {"taxonomy", "atlas"}:
        raise ValueError("primary_readout must be 'taxonomy' or 'atlas'")
    cells["cytofuse_final_id"] = cells["aligned_leaf_id"].astype(str).map(
        final_by_leaf
    )
    # Keep the resolution-atlas readout alongside the final taxonomy cut.  The
    # atlas node is the unsupervised coarse readout; the taxonomy cut remains
    # available for audit and optional downstream refinement.
    if "atlas_node_id" in cells:
        cells["cytofuse_atlas_id"] = cells["atlas_node_id"].astype(str)
        cells["cytofuse_atlas_id"] = cells["cytofuse_atlas_id"].replace(
            {"nan": pd.NA, "None": pd.NA}
        )
    else:
        cells["cytofuse_atlas_id"] = pd.NA
    atlas_id = cells["cytofuse_atlas_id"]
    atlas_id = atlas_id.where(atlas_id.notna() & atlas_id.ne(""), pd.NA)
    cells["cytofuse_primary_id"] = (
        atlas_id.fillna(cells["cytofuse_final_id"])
        if primary_readout == "atlas"
        else cells["cytofuse_final_id"]
    )
    cells["cytofuse_primary_readout"] = primary_readout
    cells["cytofuse_final_merge_reason"] = (
        cells["aligned_leaf_id"].astype(str).map(final_reason_by_leaf)
    )
    cells["cytofuse_final_resolution"] = np.where(
        cells["cytofuse_final_id"].eq(cells["cytofuse_leaf_id"]),
        "fine_leaf",
        "broader_parent",
    )
    cells["cytofuse_consensus_up_level"] = up_level
    cells["cytofuse_final_rule"] = (
        f"global_evidence_cut:shared>={max(int(final_min_shared_observed), 1)};"
        f"common_positive>={max(int(final_min_common_positive), 0)};"
        f"conflicts<={max(int(final_max_conflicts), 0)};"
        f"conflict_fraction<={float(final_max_conflict_fraction):g};"
        "single_batch_child_override=True;"
        "batch_complement_override="
        f"{bool(merge_batch_complementary_children)};"
        f"batch_complement_conflicts<={max(int(batch_complement_max_conflicts), 0)};"
        "batch_complement_conflict_fraction<="
        f"{float(batch_complement_max_conflict_fraction):g}"
    )
    return cells


def build_hierarchy_relation_graph(
    taxonomy: dict,
    resolution_atlas: dict,
) -> dict[str, pd.DataFrame]:
    """Expose merge and coarse-to-fine evidence as one unsupervised graph.

    ``taxonomy`` is the conservative binary merge forest used for the main
    cell readout. ``resolution_atlas`` contains configuration-containment
    relations that should remain available for downstream RNA refinement.
    They are intentionally kept as different edge types: a coarse-to-fine
    relation must not silently become an equivalence merge.
    """

    edge_rows: list[dict[str, str]] = []
    for row in taxonomy.get("edges", pd.DataFrame()).itertuples(index=False):
        edge_rows.append(
            {
                "parent": str(row.parent),
                "child": str(row.child),
                "relation": "configuration_merge",
                "source": "taxonomy",
            }
        )
    for row in resolution_atlas.get("edges", pd.DataFrame()).itertuples(index=False):
        edge_rows.append(
            {
                "parent": str(row.parent),
                "child": str(row.child),
                "relation": "coarse_to_fine",
                "source": "resolution_atlas",
            }
        )
    edges = pd.DataFrame(
        edge_rows,
        columns=["parent", "child", "relation", "source"],
    ).drop_duplicates()

    node_rows: dict[str, dict[str, str]] = {}
    for row in taxonomy.get("nodes", pd.DataFrame()).itertuples(index=False):
        node = str(row.tree_node_id)
        node_rows[node] = {
            "node_id": node,
            "taxonomy_kind": str(row.node_kind),
            "atlas_kind": "",
        }
    for row in resolution_atlas.get("nodes", pd.DataFrame()).itertuples(index=False):
        node = str(row.atlas_node_id)
        existing = node_rows.setdefault(
            node,
            {"node_id": node, "taxonomy_kind": "", "atlas_kind": ""},
        )
        existing["atlas_kind"] = str(row.atlas_node_kind)
    nodes = pd.DataFrame(
        list(node_rows.values()),
        columns=["node_id", "taxonomy_kind", "atlas_kind"],
    ).sort_values("node_id")
    return {"nodes": nodes.reset_index(drop=True), "edges": edges.reset_index(drop=True)}
