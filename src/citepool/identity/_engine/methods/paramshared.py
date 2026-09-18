from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import igraph as ig
import leidenalg
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

try:
    import torch
except ImportError:  # Optional GPU backend.
    torch = None

EPS = 1e-12


def resolve_compute_device(device: str) -> str:
    value = str(device).lower()
    if value not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    cuda_available = bool(
        torch is not None and torch.cuda.is_available()
    )
    if value == "auto":
        return "cuda" if cuda_available else "cpu"
    if value == "cuda" and not cuda_available:
        raise RuntimeError("device='cuda' requested but PyTorch CUDA is unavailable")
    return value


def _leiden_target_labels(
    features,
    target_components,
    random_state=0,
    n_neighbors=30,
    max_resolution_steps=14,
):
    """Create a fixed-cardinality Leiden proposal for structural splitting."""

    features = np.asarray(features, dtype=np.float64)
    n_cells = features.shape[0]
    target = int(np.clip(target_components, 1, n_cells))
    if target == 1 or n_cells < 3:
        return np.zeros(n_cells, dtype=int), 0.0

    neighbors = min(max(int(n_neighbors), 5), n_cells - 1)
    model = NearestNeighbors(n_neighbors=neighbors + 1, metric="euclidean")
    model.fit(features)
    distances, indices = model.kneighbors(features)
    distances, indices = distances[:, 1:], indices[:, 1:]
    positive = distances[distances > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    edge_weights = {}
    for left in range(n_cells):
        for right, distance in zip(indices[left], distances[left]):
            edge = (left, int(right)) if left < right else (int(right), left)
            weight = max(
                float(np.exp(-0.5 * (distance / max(scale, EPS)) ** 2)),
                0.05,
            )
            edge_weights[edge] = max(edge_weights.get(edge, 0.0), weight)
    graph = ig.Graph(n=n_cells, edges=list(edge_weights), directed=False)
    graph.es["weight"] = list(edge_weights.values())

    evaluated = {}

    def evaluate(resolution):
        key = float(resolution)
        if key not in evaluated:
            partition = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=key,
                seed=int(random_state),
                n_iterations=-1,
            )
            labels = np.asarray(partition.membership, dtype=int)
            evaluated[key] = (int(labels.max() + 1), labels)
        return evaluated[key]

    resolution = 1.0
    count, _ = evaluate(resolution)
    low = high = resolution
    for _ in range(10):
        if count == target:
            break
        if count < target:
            low = resolution
            resolution *= 2.0
            high = resolution
        else:
            high = resolution
            resolution /= 2.0
            low = resolution
        count, _ = evaluate(resolution)
        if evaluate(low)[0] <= target <= evaluate(high)[0]:
            break
    for _ in range(max(int(max_resolution_steps), 1)):
        midpoint = float(np.sqrt(max(low, 1e-6) * max(high, 1e-6)))
        count, _ = evaluate(midpoint)
        if count < target:
            low = midpoint
        elif count > target:
            high = midpoint
        else:
            break
    bounded = [item for item in evaluated.items() if item[1][0] <= target]
    candidates = bounded if bounded else list(evaluated.items())
    best_resolution, (_, labels) = min(
        candidates,
        key=lambda item: (
            abs(item[1][0] - target),
            abs(np.log(max(item[0], 1e-6))),
        ),
    )
    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(int), float(best_resolution)


def leiden_initial_labels(
    features,
    resolution=1.0,
    random_state=0,
    n_neighbors=30,
):
    """Initialize phenotypes at a user-specified Leiden resolution."""

    features = np.asarray(features, dtype=np.float64)
    n_cells = features.shape[0]
    if n_cells < 3:
        return np.zeros(n_cells, dtype=int), float(resolution)
    resolution = float(resolution)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("initial_resolution must be finite and positive")

    neighbors = min(max(int(n_neighbors), 5), n_cells - 1)
    model = NearestNeighbors(n_neighbors=neighbors + 1, metric="euclidean")
    model.fit(features)
    distances, indices = model.kneighbors(features)
    distances, indices = distances[:, 1:], indices[:, 1:]
    positive = distances[distances > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    edge_weights = {}
    for left in range(n_cells):
        for right, distance in zip(indices[left], distances[left]):
            edge = (left, int(right)) if left < right else (int(right), left)
            weight = max(
                float(np.exp(-0.5 * (distance / max(scale, EPS)) ** 2)),
                0.05,
            )
            edge_weights[edge] = max(edge_weights.get(edge, 0.0), weight)
    graph = ig.Graph(n=n_cells, edges=list(edge_weights), directed=False)
    graph.es["weight"] = list(edge_weights.values())
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=int(random_state),
        n_iterations=-1,
    )
    _, labels = np.unique(np.asarray(partition.membership, dtype=int), return_inverse=True)
    return labels.astype(int), resolution


def robust_zscore_per_marker(X, eps=1e-6):
    X = np.asarray(X, dtype=np.float64)
    center = np.nanmedian(X, axis=0)
    scale = (np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)) / 1.349
    fallback = np.nanstd(X, axis=0)
    scale = np.where(scale > eps, scale, fallback)
    scale = np.where(scale > eps, scale, 1.0)
    return (X - center) / scale, center, scale


def add_gaussian_jitter(X, noise_sd=1e-3, random_state=0, mode="all"):
    X = np.asarray(X, dtype=np.float64)
    if noise_sd is None or noise_sd <= 0 or mode == "none":
        return X.copy()
    rng = np.random.default_rng(random_state)
    out = X.copy()
    if mode == "all":
        out += rng.normal(0, noise_sd, size=out.shape)
    elif mode == "zero":
        mask = np.abs(out) <= EPS
        out[mask] += rng.normal(0, noise_sd, size=mask.sum())
    else:
        raise ValueError("mode must be one of {'all', 'zero', 'none'}")
    return out


def log_gaussian_1d(X, mean, variance):
    variance = np.maximum(np.asarray(variance, dtype=float), EPS)
    return (
        -0.5 * np.log(2.0 * np.pi * variance)[None, :]
        - 0.5 * (np.asarray(X) - np.asarray(mean)[None, :]) ** 2 / variance[None, :]
    )


def _log_gaussian_vec(x, mu, variance):
    variance = np.maximum(np.asarray(variance, dtype=float), EPS)
    return -0.5 * np.log(2.0 * np.pi * variance) - 0.5 * (x[:, None] - mu[None, :]) ** 2 / variance[None, :]


def bhattacharyya_distance_1d(mu1, var1, mu2, var2):
    var1 = np.maximum(var1, EPS)
    var2 = np.maximum(var2, EPS)
    return (
        0.25 * np.log(0.25 * (var1 / var2 + var2 / var1 + 2.0))
        + 0.25 * (mu1 - mu2) ** 2 / (var1 + var2)
    )


@dataclass
class CandidateFit:
    n_states: int
    weights: np.ndarray
    mu: np.ndarray
    variance: np.ndarray
    log_likelihood: float
    bic: float
    posterior: np.ndarray
    stability: float
    valid: bool
    reason: str = "ok"


def _fixed_two_state_candidate(
    values,
    means,
    common_variance,
    component_weights,
    variance_floor,
    reason,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot initialize marker states from no finite values")
    means = np.asarray(means, dtype=float)
    common_variance = max(float(common_variance), float(variance_floor))
    variances = np.full(2, common_variance, dtype=float)
    if component_weights is None:
        weights = np.full(2, 0.5, dtype=float)
    else:
        weights = np.asarray(component_weights, dtype=float).reshape(-1)
        if weights.size != 2 or np.any(~np.isfinite(weights)) or np.any(weights < 0):
            weights = np.full(2, 0.5, dtype=float)
        else:
            total = float(weights.sum())
            weights = weights / total if total > EPS else np.full(2, 0.5)
    log_joint = _log_gaussian_vec(values, means, variances) + np.log(
        np.maximum(weights, EPS)
    )[None, :]
    log_normalizer = logsumexp(log_joint, axis=1)
    posterior = np.exp(log_joint - log_normalizer[:, None])
    log_likelihood = float(log_normalizer.sum())
    bic = float(-2.0 * log_likelihood + 5.0 * np.log(values.size))
    return CandidateFit(
        n_states=2,
        weights=weights,
        mu=means,
        variance=variances,
        log_likelihood=log_likelihood,
        bic=bic,
        posterior=posterior,
        stability=1.0,
        valid=means[1] > means[0],
        reason="ok" if means[1] > means[0] else reason,
    )


def minmax_quartile_two_state_initialization(
    x,
    *,
    component_weights=None,
    variance_floor=1e-3,
):
    """Initialize two Gaussian states at 1/4 and 3/4 of the data range."""

    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot initialize marker states from no finite values")
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = max(upper - lower, 0.0)
    return _fixed_two_state_candidate(
        values,
        [lower + 0.25 * span, lower + 0.75 * span],
        (span / 4.0) ** 2,
        component_weights,
        variance_floor,
        "constant_marker",
    )


def empirical_quartile_two_state_initialization(
    x,
    *,
    component_weights=None,
    variance_floor=1e-3,
):
    """Use empirical Q25/Q75 as centers with a shared robust scale."""

    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot initialize marker states from no finite values")
    lower, upper = np.quantile(values, [0.25, 0.75])
    span = max(float(upper - lower), 0.0)
    return _fixed_two_state_candidate(
        values,
        [lower, upper],
        (span / 2.0) ** 2,
        component_weights,
        variance_floor,
        "collapsed_quartiles",
    )


def order_trimmed_quartile_two_state_initialization(
    x,
    *,
    boundary_order=3,
    component_weights=None,
    variance_floor=1e-3,
):
    """Use the kth smallest/largest values as robust range boundaries."""

    values = np.asarray(x, dtype=float)
    values = np.sort(values[np.isfinite(values)])
    order = max(int(boundary_order), 1)
    if values.size < 2 * order:
        raise ValueError("too few finite values for requested boundary order")
    lower = float(values[order - 1])
    upper = float(values[-order])
    span = max(upper - lower, 0.0)
    return _fixed_two_state_candidate(
        values,
        [lower + 0.25 * span, lower + 0.75 * span],
        (span / 4.0) ** 2,
        component_weights,
        variance_floor,
        "collapsed_trimmed_range",
    )


@dataclass
class MarkerStateStats:
    resp_sum: list[np.ndarray]
    x_sum: list[np.ndarray]
    x2_sum: list[np.ndarray]


class BaseCluster:
    """Heterogeneous marker-wise Gaussian state model."""

    def __init__(
        self,
        n_states,
        mu,
        variance,
        component_weights,
        marker_status,
        marker_weights,
        configuration_weights,
        model_selection_metrics,
        marker_names=None,
        variance_floor=1e-3,
    ):
        self.n_states = np.asarray(n_states, dtype=int)
        self.n_markers = int(self.n_states.size)
        self.max_states = int(self.n_states.max()) if self.n_markers else 0
        self.state_offsets = np.zeros(self.n_markers + 1, dtype=int)
        self.state_offsets[1:] = np.cumsum(self.n_states)
        self.total_states = int(self.state_offsets[-1])
        self.mu = [np.asarray(v, dtype=float).copy() for v in mu]
        self.variance = [np.maximum(np.asarray(v, dtype=float), variance_floor) for v in variance]
        self.initial_mu = [v.copy() for v in self.mu]
        self.initial_variance = [v.copy() for v in self.variance]
        self.component_weights = [np.asarray(v, dtype=float).copy() for v in component_weights]
        self.initial_component_weights = [value.copy() for value in self.component_weights]
        self.marker_status = np.asarray(marker_status, dtype=object)
        # marker_weights are used only by batch-local high-dimensional EM.
        self.marker_weights = np.asarray(marker_weights, dtype=float)
        self.prior_marker_weights = self.marker_weights.copy()
        # configuration_weights decide whether a marker can produce +/- calls.
        self.configuration_weights = np.asarray(configuration_weights, dtype=float)
        self.model_selection_metrics = pd.DataFrame(model_selection_metrics)
        self.marker_names = (
            np.asarray(marker_names, dtype=str)
            if marker_names is not None
            else np.asarray([f"marker_{j}" for j in range(self.n_markers)])
        )
        self.variance_floor = float(variance_floor)
        self.validate_parameters()

    @classmethod
    def initialize_marker_models(cls, X, marker_names=None, random_state=0, **kwargs):
        selector = MarkerModelSelector(random_state=random_state, **kwargs)
        n_markers = X.shape[1]
        marker_names = (
            np.asarray(marker_names, dtype=str)
            if marker_names is not None
            else np.asarray([f"marker_{j}" for j in range(n_markers)])
        )
        n_states, mu, var, comp_w = [], [], [], []
        status, cluster_weights, configuration_weights, records = [], [], [], []
        for j in range(n_markers):
            selected, metrics = selector.initialize_candidate_states(
                np.asarray(X[:, j], dtype=float)
            )
            n_states.append(selected.n_states)
            mu.append(selected.mu)
            var.append(selected.variance)
            comp_w.append(selected.weights)
            status.append(metrics["status"])
            cluster_weights.append(metrics["cluster_weight"])
            configuration_weights.append(metrics["configuration_weight"])
            metrics["marker"] = marker_names[j]
            records.append(metrics)
        return cls(
            n_states=n_states,
            mu=mu,
            variance=var,
            component_weights=comp_w,
            marker_status=status,
            marker_weights=cluster_weights,
            configuration_weights=configuration_weights,
            model_selection_metrics=records,
            marker_names=marker_names,
            variance_floor=selector.variance_floor,
        )

    def compute_log_probability(self, X):
        X = np.asarray(X, dtype=float)
        return [_log_gaussian_vec(X[:, j], self.mu[j], self.variance[j]) for j in range(self.n_markers)]

    def compute_independent_state_posteriors(self, X):
        logs = self.compute_log_probability(X)
        out = []
        for j, lp in enumerate(logs):
            prior = np.maximum(self.component_weights[j], EPS)
            prior = prior / prior.sum()
            joint = lp + np.log(prior)[None, :]
            out.append(np.exp(joint - logsumexp(joint, axis=1, keepdims=True)))
        return out

    def update_parameters(self, X, stats: MarkerStateStats, shrinkage_strength=0.0):
        state_orders = []
        for j in range(self.n_markers):
            if self.marker_weights[j] <= 0 or np.sum(stats.resp_sum[j]) <= EPS:
                state_orders.append(np.arange(self.n_states[j], dtype=int))
                continue
            rsum = np.maximum(stats.resp_sum[j], EPS)
            empirical_mu = stats.x_sum[j] / rsum
            empirical_var = np.maximum(
                stats.x2_sum[j] / rsum - empirical_mu**2,
                self.variance_floor,
            )
            if shrinkage_strength and shrinkage_strength > 0:
                lam = float(shrinkage_strength)
                mu_new = (
                    rsum * empirical_mu + lam * self.initial_mu[j]
                ) / (rsum + lam)
                var_new = (
                    rsum * empirical_var + lam * self.initial_variance[j]
                ) / (rsum + lam)
            else:
                mu_new = empirical_mu
                var_new = empirical_var
            order = np.argsort(mu_new)
            state_orders.append(order)
            self.mu[j] = mu_new[order]
            self.variance[j] = np.maximum(var_new[order], self.variance_floor)
            weights = rsum[order] / np.maximum(rsum.sum(), EPS)
            self.component_weights[j] = np.maximum(weights, EPS) / np.maximum(weights.sum(), EPS)
        self.validate_parameters()
        return state_orders

    def update_configuration_reliability(
        self,
        min_bhattacharyya=1.0,
        min_component_weight=0.01,
    ):
        """Update strict configuration reliability without changing EM weights."""

        metrics = (
            self.model_selection_metrics.set_index("marker")
            if "marker" in self.model_selection_metrics
            else pd.DataFrame()
        )
        for j in range(self.n_markers):
            if self.n_states[j] < 2:
                self.configuration_weights[j] = 0.0
                continue
            mu = np.asarray(self.mu[j], dtype=float)
            var = np.asarray(self.variance[j], dtype=float)
            comp = np.asarray(self.component_weights[j], dtype=float)
            if len(mu) != 2 or np.min(comp) < min_component_weight:
                self.configuration_weights[j] = 0.0
                self.marker_status[j] = "rare_state"
                continue
            marker = self.marker_names[j]
            candidate_bd = float(
                bhattacharyya_distance_1d(mu[0], var[0], mu[1], var[1])
            )
            # Q25/Q75 candidates deliberately use a common robust variance, so
            # their distance is a construction property rather than evidence
            # that the observed marker distribution is bimodal.  Keep those
            # candidates for high-dimensional EM, but use the independently
            # fitted GMM distance for the final configuration reliability call.
            bd = (
                float(metrics.loc[marker].get("bhattacharyya_distance", 0.0))
                if marker in metrics.index
                else candidate_bd
            )
            bic_gain = (
                float(metrics.loc[marker].get("bic_gain_12", 0.0))
                if marker in metrics.index
                else 0.0
            )
            stability = (
                float(metrics.loc[marker].get("fit_stability", 1.0))
                if marker in metrics.index
                else 1.0
            )
            bic_score = max(bic_gain, 0.0) / (max(bic_gain, 0.0) + 50.0)
            bd_score = bd / (bd + max(float(min_bhattacharyya), EPS))
            balance = min(1.0, np.min(comp) / max(float(min_component_weight), EPS))
            reliable = bic_gain >= 10.0 and bd >= min_bhattacharyya
            if reliable:
                self.configuration_weights[j] = float(
                    np.clip(bic_score * bd_score * balance * stability, 0.0, 1.0)
                )
                self.marker_status[j] = "valid_bimodal"
            else:
                self.configuration_weights[j] = 0.0
                self.marker_status[j] = "weak_bimodal"
        self.model_selection_metrics["cluster_weight"] = self.marker_weights
        self.model_selection_metrics["bhattacharyya_distance_final"] = [
            (
                float(
                    bhattacharyya_distance_1d(
                        self.mu[j][0],
                        self.variance[j][0],
                        self.mu[j][1],
                        self.variance[j][1],
                    )
                )
                if self.n_states[j] == 2
                else np.nan
            )
            for j in range(self.n_markers)
        ]
        self.model_selection_metrics["configuration_bhattacharyya_distance"] = [
            (
                float(metrics.loc[self.marker_names[j]].get(
                    "bhattacharyya_distance", 0.0
                ))
                if self.marker_names[j] in metrics.index
                else np.nan
            )
            for j in range(self.n_markers)
        ]
        self.model_selection_metrics["configuration_weight"] = (
            self.configuration_weights
        )
        # Compatibility for downstream code written before the weight split.
        self.model_selection_metrics["phenotype_weight"] = (
            self.configuration_weights
        )
        if "status" in self.model_selection_metrics:
            self.model_selection_metrics["status"] = self.marker_status

    def validate_parameters(self):
        for j in range(self.n_markers):
            if len(self.mu[j]) != self.n_states[j]:
                raise ValueError(f"marker {j}: mu length does not match n_states.")
            if len(self.variance[j]) != self.n_states[j]:
                raise ValueError(f"marker {j}: variance length does not match n_states.")
            if np.any(np.diff(self.mu[j]) < 0):
                order = np.argsort(self.mu[j])
                self.mu[j] = self.mu[j][order]
                self.variance[j] = self.variance[j][order]
                self.component_weights[j] = self.component_weights[j][order]
            self.variance[j] = np.maximum(self.variance[j], self.variance_floor)

    def to_marker_params(self, center=None, scale=None, n_iter=0):
        center = np.zeros(self.n_markers) if center is None else np.asarray(center)
        scale = np.ones(self.n_markers) if scale is None else np.asarray(scale)
        rows = []
        metrics = self.model_selection_metrics.set_index("marker") if "marker" in self.model_selection_metrics else pd.DataFrame()
        for j, marker in enumerate(self.marker_names):
            row = metrics.loc[marker].to_dict() if marker in metrics.index else {}
            mu_norm = self.mu[j]
            var_norm = self.variance[j]
            weights = self.component_weights[j]
            row.update({
                "marker": marker,
                "n_states": int(self.n_states[j]),
                "status": str(self.marker_status[j]),
                "cluster_weight_prior": float(self.prior_marker_weights[j]),
                "cluster_weight": float(self.marker_weights[j]),
                "configuration_weight": float(self.configuration_weights[j]),
                "phenotype_weight": float(self.configuration_weights[j]),
                "marker_weight": float(self.configuration_weights[j]),
                "unimodal_state": "not_applicable",
                "component_weights": weights.tolist(),
                "component_means_norm": mu_norm.tolist(),
                "component_means_original": (mu_norm * scale[j] + center[j]).tolist(),
                "component_variances_norm": var_norm.tolist(),
                "component_variances_original": (
                    var_norm * scale[j] ** 2
                ).tolist(),
                "n_iter": int(n_iter),
            })
            if self.n_states[j] == 1:
                if mu_norm[0] >= 0.5:
                    unimodal_state = "high"
                elif mu_norm[0] <= -0.5:
                    unimodal_state = "low"
                else:
                    unimodal_state = "unassigned"
                row.update({
                    "unimodal_mean_norm": float(mu_norm[0]),
                    "unimodal_mean_original": float(mu_norm[0] * scale[j] + center[j]),
                    "unimodal_var": float(var_norm[0]),
                    "unimodal_state": unimodal_state,
                })
            else:
                row.update({
                    "low_mean_norm": float(mu_norm[0]),
                    "high_mean_norm": float(mu_norm[-1]),
                    "low_mean_original": float(mu_norm[0] * scale[j] + center[j]),
                    "high_mean_original": float(mu_norm[-1] * scale[j] + center[j]),
                    "low_var": float(var_norm[0]),
                    "high_var": float(var_norm[-1]),
                    "low_weight": float(weights[0]),
                    "high_weight": float(weights[-1]),
                })
            rows.append(row)
        return pd.DataFrame(rows)


class MarkerModelSelector:
    """Fit marginal GMM evidence and initialize recoverable marker states."""

    def __init__(
        self,
        random_state=0,
        variance_floor=1e-3,
        n_init=5,
        min_component_weight=0.02,
        min_component_weight_3=0.03,
        bic_gain_12=10.0,
        bic_gain_23=20.0,
        icl_gain_23=5.0,
        min_bhattacharyya=0.08,
        stability_tol=0.35,
        weak_cluster_weight_floor=0.05,
        component_prior_floor=0.01,
    ):
        self.random_state = int(random_state)
        self.variance_floor = float(variance_floor)
        self.n_init = int(n_init)
        self.min_component_weight = float(min_component_weight)
        self.min_component_weight_3 = float(min_component_weight_3)
        self.bic_gain_12 = float(bic_gain_12)
        self.bic_gain_23 = float(bic_gain_23)
        self.icl_gain_23 = float(icl_gain_23)
        self.min_bhattacharyya = float(min_bhattacharyya)
        self.stability_tol = float(stability_tol)
        self.weak_cluster_weight_floor = float(weak_cluster_weight_floor)
        self.component_prior_floor = float(component_prior_floor)
        if not 0.0 < self.component_prior_floor < 0.5:
            raise ValueError("component_prior_floor must be between 0 and 0.5")

    def fit_candidate_model(self, x, n_states):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        n = x.size
        if n < max(10, n_states * 5):
            mu = np.full(n_states, np.nanmean(x) if n else 0.0)
            var = np.full(n_states, np.nanvar(x) + self.variance_floor if n else 1.0)
            post = np.full((max(n, 1), n_states), 1.0 / n_states)
            return CandidateFit(n_states, np.ones(n_states) / n_states, mu, var, -np.inf, np.inf, post, 0.0, False, "too_few_cells")
        if n_states == 1:
            mu = np.array([np.mean(x)])
            var = np.array([max(np.var(x), self.variance_floor)])
            ll = float(_log_gaussian_vec(x, mu, var).sum())
            bic = -2.0 * ll + 2.0 * np.log(n)
            post = np.ones((n, 1))
            return CandidateFit(1, np.ones(1), mu, var, ll, bic, post, 1.0, True)

        quantiles = [0.25, 0.75] if n_states == 2 else [0.15, 0.50, 0.85]
        means_init = np.quantile(x, quantiles).reshape(-1, 1)
        best = None
        all_mu = []
        for seed in range(self.n_init):
            try:
                gm = GaussianMixture(
                    n_components=n_states,
                    covariance_type="diag",
                    reg_covar=self.variance_floor,
                    n_init=1,
                    max_iter=300,
                    random_state=self.random_state + seed,
                    means_init=means_init if seed == 0 else None,
                )
                gm.fit(x.reshape(-1, 1))
                mu = gm.means_.ravel()
                var = gm.covariances_.reshape(n_states, -1)[:, 0]
                weights = gm.weights_.ravel()
                order = np.argsort(mu)
                mu, var, weights = mu[order], np.maximum(var[order], self.variance_floor), weights[order]
                post = gm.predict_proba(x.reshape(-1, 1))[:, order]
                ll = float(gm.score(x.reshape(-1, 1)) * n)
                bic = float(gm.bic(x.reshape(-1, 1)))
                fit = CandidateFit(n_states, weights, mu, var, ll, bic, post, 1.0, True)
                all_mu.append(mu)
                if best is None or fit.bic < best.bic:
                    best = fit
            except Exception:
                continue
        if best is None:
            one = self.fit_candidate_model(x, 1)
            return CandidateFit(n_states, np.ones(n_states) / n_states, np.repeat(one.mu, n_states), np.repeat(one.variance, n_states), -np.inf, np.inf, np.ones((n, n_states)) / n_states, 0.0, False, "fit_failed")
        if len(all_mu) > 1:
            arr = np.vstack(all_mu)
            stability = 1.0 / (1.0 + float(np.mean(np.std(arr, axis=0))))
        else:
            stability = 1.0
        best.stability = stability
        collapse = np.any(best.variance <= self.variance_floor * 1.01)
        rare = np.min(best.weights) < (self.min_component_weight_3 if n_states == 3 else self.min_component_weight)
        best.valid = not collapse and not rare
        best.reason = "ok" if best.valid else ("variance_collapse" if collapse else "rare_state")
        return best

    def select_marker_state_count(self, x):
        """Return the legacy hard 1-vs-2 selection for diagnostics."""

        m1 = self.fit_candidate_model(x, 1)
        m2 = self.fit_candidate_model(x, 2)
        bhat2 = (
            float(bhattacharyya_distance_1d(m2.mu[0], m2.variance[0], m2.mu[1], m2.variance[1]))
            if len(m2.mu) == 2
            else 0.0
        )
        conf2 = float(np.mean(np.max(m2.posterior, axis=1))) if m2.posterior.size else 0.0
        bic_gain_12 = float(m1.bic - m2.bic)
        candidate_two_state = m2.valid and bic_gain_12 > 0.0
        reliable_two_state = (
            candidate_two_state
            and bic_gain_12 >= self.bic_gain_12
            and bhat2 >= self.min_bhattacharyya
        )
        if candidate_two_state:
            selected = m2
            status = "valid_bimodal" if reliable_two_state else "weak_bimodal"
        else:
            selected = m1
            status = "rare_state" if m2.reason == "rare_state" else "unimodal"
        if candidate_two_state:
            evidence = max(bic_gain_12, 0.0) / (max(bic_gain_12, 0.0) + 50.0)
            bd_score = bhat2 / (bhat2 + max(self.min_bhattacharyya, EPS))
            balance = min(
                1.0,
                np.min(m2.weights) / max(self.min_component_weight, EPS),
            )
            soft_weight = (
                evidence * bd_score * balance * conf2 * selected.stability
            )
            cluster_weight = max(self.weak_cluster_weight_floor, soft_weight)
            configuration_weight = soft_weight if reliable_two_state else 0.0
        else:
            cluster_weight = 0.0
            configuration_weight = 0.0
        metrics = {
            "n_states": int(selected.n_states),
            "status": status,
            "cluster_weight": float(np.clip(cluster_weight, 0.0, 1.0)),
            "configuration_weight": float(
                np.clip(configuration_weight, 0.0, 1.0)
            ),
            "phenotype_weight": float(
                np.clip(configuration_weight, 0.0, 1.0)
            ),
            "bic_1": float(m1.bic),
            "bic_2": float(m2.bic),
            "bic_gain_12": bic_gain_12,
            "bhattacharyya_distance": float(bhat2),
            "posterior_confidence": float(conf2),
            "fit_stability": float(selected.stability),
            "selected_component_weights": selected.weights.tolist(),
            "selected_means_norm": selected.mu.tolist(),
            "selected_variances_norm": selected.variance.tolist(),
        }
        return selected, metrics

    def initialize_candidate_states(self, x):
        """Initialize two recoverable states while retaining GMM evidence."""

        values = np.asarray(x, dtype=float)
        values = values[np.isfinite(values)]
        m1 = self.fit_candidate_model(values, 1)
        m2 = self.fit_candidate_model(values, 2)
        bhattacharyya = (
            float(
                bhattacharyya_distance_1d(
                    m2.mu[0], m2.variance[0], m2.mu[1], m2.variance[1]
                )
            )
            if len(m2.mu) == 2
            else 0.0
        )
        bic_gain = float(m1.bic - m2.bic)
        bic_probability = float(
            1.0 / (1.0 + np.exp(-np.clip(0.5 * bic_gain, -40.0, 40.0)))
        )
        nonoverlap = float(1.0 - np.exp(-max(bhattacharyya, 0.0)))
        gmm_confidence = float(
            np.clip(
                np.sqrt(bic_probability * nonoverlap) * m2.stability,
                0.0,
                1.0,
            )
        )
        component_weights = np.asarray(m2.weights, dtype=float)
        if component_weights.size != 2 or np.any(~np.isfinite(component_weights)):
            component_weights = np.full(2, 0.5)
        component_weights = np.maximum(
            component_weights,
            self.component_prior_floor,
        )
        component_weights /= component_weights.sum()
        initialized = empirical_quartile_two_state_initialization(
            values,
            component_weights=component_weights,
            variance_floor=self.variance_floor,
        )
        posterior_confidence = float(np.mean(np.max(m2.posterior, axis=1)))
        clustering_weight = float(
            self.weak_cluster_weight_floor
            + (1.0 - self.weak_cluster_weight_floor) * gmm_confidence
        )
        metrics = {
            "n_states": 2,
            "status": "candidate_two_state",
            "initialization": "empirical_q25_q75",
            "cluster_weight": clustering_weight,
            "configuration_weight": 0.0,
            "phenotype_weight": 0.0,
            "gmm_confidence": gmm_confidence,
            "gmm_bic_probability": bic_probability,
            "gmm_nonoverlap": nonoverlap,
            "bic_1": float(m1.bic),
            "bic_2": float(m2.bic),
            "bic_gain_12": bic_gain,
            "bhattacharyya_distance": bhattacharyya,
            "posterior_confidence": posterior_confidence,
            "fit_stability": float(m2.stability),
            "gmm_component_weights": m2.weights.tolist(),
            "gmm_component_means_norm": m2.mu.tolist(),
            "gmm_component_variances_norm": m2.variance.tolist(),
            "selected_component_weights": initialized.weights.tolist(),
            "selected_means_norm": initialized.mu.tolist(),
            "selected_variances_norm": initialized.variance.tolist(),
        }
        return initialized, metrics


class ParamSharedCluster:
    """Finite phenotype mixture over marker states."""

    def __init__(
        self,
        base_cluster: BaseCluster,
        initial_resolution=1.0,
        n_components=1,
        alpha_theta=0.2,
        alpha_pi=0.2,
        random_state=0,
        device="cpu",
    ):
        self.base_cluster = base_cluster
        # n_components is retained only for constructing low-level diagnostic
        # models; fit() always replaces it with Leiden resolution initialization.
        self.n_components = int(n_components)
        self.alpha_theta = float(alpha_theta)
        self.alpha_pi = float(alpha_pi)
        self.random_state = int(random_state)
        self.device = resolve_compute_device(device)
        self.pi = np.full(self.n_components, 1.0 / self.n_components)
        self.theta_flat = np.zeros((self.n_components, base_cluster.total_states), dtype=float)
        self.resp = None
        self.labels_ = None
        self.log_likelihood = -np.inf
        self.history = []
        self.initialization_method = "leiden"
        self.initialization_resolution = float(initial_resolution)
        self.max_structural_components = 1
        self._initialization_features = None
        self._initialization_marker_slices = {}
        self.theta_sharpening = 0.0
        self.last_split_diagnostic = {}
        self.last_prune_protected = []

    def _init_theta_from_labels(self, state_post, labels):
        K = self.n_components
        n = labels.size
        self.pi = np.maximum(np.bincount(labels, minlength=K).astype(float), EPS)
        self.pi /= self.pi.sum()
        for j, post in enumerate(state_post):
            start, end = self.base_cluster.state_offsets[j], self.base_cluster.state_offsets[j + 1]
            for k in range(K):
                idx = labels == k
                if idx.any():
                    theta = post[idx].sum(axis=0)
                else:
                    theta = post.sum(axis=0)
                theta = np.maximum(theta + self.alpha_theta, EPS)
                theta = theta / theta.sum()
                self.theta_flat[k, start:end] = theta

    def initialize(self, X):
        state_post = self.base_cluster.compute_independent_state_posteriors(X)
        active = self.base_cluster.marker_weights > 0
        features = []
        feature_offset = 0
        for j, post in enumerate(state_post):
            if active[j]:
                feature = post * np.sqrt(self.base_cluster.marker_weights[j])
                features.append(feature)
                self._initialization_marker_slices[j] = slice(
                    feature_offset, feature_offset + feature.shape[1]
                )
                feature_offset += feature.shape[1]
        if features:
            self._initialization_features = np.concatenate(features, axis=1).astype(
                np.float32, copy=False
            )
        if len(features) == 0:
            labels = np.zeros(X.shape[0], dtype=int)
            self.n_components = 1
            self.pi = np.ones(1)
            self.theta_flat = np.zeros((1, self.base_cluster.total_states), dtype=float)
        else:
            F = self._initialization_features
            labels, resolution = leiden_initial_labels(
                F,
                resolution=self.initialization_resolution,
                random_state=self.random_state,
            )
            self.initialization_resolution = resolution
            self.n_components = int(labels.max() + 1)
            self.pi = np.full(self.n_components, 1.0 / self.n_components)
            self.theta_flat = np.zeros(
                (self.n_components, self.base_cluster.total_states), dtype=float
            )
        self.max_structural_components = min(
            64,
            max(
                8,
                2 * self.n_components,
                int(round(np.sqrt(X.shape[0] / 20.0))),
            ),
        )
        self._init_theta_from_labels(state_post, labels)
        return self

    def e_step(self, X, return_stats=False, return_marker_stats=True):
        if self.device == "cuda":
            return self._e_step_torch(
                X,
                return_stats=return_stats,
                return_marker_stats=return_marker_stats,
            )
        logs = self.base_cluster.compute_log_probability(X)
        n = X.shape[0]
        K = self.n_components
        log_resp = np.log(np.maximum(self.pi, EPS))[None, :].repeat(n, axis=0)
        denom_by_marker = []
        for j, lp in enumerate(logs):
            w = self.base_cluster.marker_weights[j]
            if w <= 0:
                denom_by_marker.append(None)
                continue
            start, end = self.base_cluster.state_offsets[j], self.base_cluster.state_offsets[j + 1]
            theta = np.maximum(self.theta_flat[:, start:end], EPS)
            tmp = lp[:, None, :] + np.log(theta)[None, :, :]
            denom = logsumexp(tmp, axis=2)
            log_resp += w * denom
            denom_by_marker.append((tmp, denom))
        norm = logsumexp(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp - norm)
        ll = float(np.sum(norm))
        self.resp = resp
        self.log_likelihood = ll
        self.labels_ = np.argmax(resp, axis=1).astype(str)
        if not return_stats:
            return resp, ll, None
        marker_stats = None
        if return_marker_stats:
            resp_sum = [np.zeros(L, dtype=float) for L in self.base_cluster.n_states]
            x_sum = [np.zeros(L, dtype=float) for L in self.base_cluster.n_states]
            x2_sum = [np.zeros(L, dtype=float) for L in self.base_cluster.n_states]
        theta_counts = np.zeros_like(self.theta_flat)
        for j, item in enumerate(denom_by_marker):
            L = self.base_cluster.n_states[j]
            start, end = self.base_cluster.state_offsets[j], self.base_cluster.state_offsets[j + 1]
            if item is None:
                post = self.base_cluster.compute_independent_state_posteriors(X[:, [j]])[0] if False else None
                theta_counts[:, start:end] = 1.0 / L
                continue
            tmp, denom = item
            p_state_given_k = np.exp(tmp - denom[:, :, None])
            weighted = resp[:, :, None] * p_state_given_k
            counts_k = np.maximum(
                weighted.sum(axis=0) + self.alpha_theta - 1.0,
                EPS,
            )
            theta_counts[:, start:end] = counts_k
            if return_marker_stats:
                total_state = weighted.sum(axis=1)
                resp_sum[j] = total_state.sum(axis=0)
                x_sum[j] = (total_state * X[:, j:j + 1]).sum(axis=0)
                x2_sum[j] = (total_state * (X[:, j:j + 1] ** 2)).sum(axis=0)
        if return_marker_stats:
            marker_stats = MarkerStateStats(resp_sum, x_sum, x2_sum)
        return resp, ll, (marker_stats, theta_counts)

    def _e_step_torch(self, X, return_stats=False, return_marker_stats=True):
        """Run the phenotype E-step and sufficient statistics on CUDA."""

        X_array = np.asarray(X, dtype=np.float32)
        n = X_array.shape[0]
        K = self.n_components
        with torch.no_grad():
            x_tensor = torch.as_tensor(X_array, device="cuda")
            pi = torch.as_tensor(self.pi, dtype=torch.float32, device="cuda")
            theta_flat = torch.as_tensor(
                self.theta_flat, dtype=torch.float32, device="cuda"
            )
            log_resp = torch.log(torch.clamp(pi, min=EPS))[None, :].expand(n, K).clone()

            def marker_terms(j):
                mu = torch.as_tensor(
                    self.base_cluster.mu[j], dtype=torch.float32, device="cuda"
                )
                variance = torch.as_tensor(
                    self.base_cluster.variance[j], dtype=torch.float32, device="cuda"
                )
                values = x_tensor[:, j:j + 1]
                log_probability = -0.5 * (
                    np.log(2.0 * np.pi)
                    + torch.log(torch.clamp(variance, min=EPS))[None, :]
                    + (values - mu[None, :]) ** 2
                    / torch.clamp(variance, min=EPS)[None, :]
                )
                start, end = self.base_cluster.state_offsets[j:j + 2]
                theta = torch.clamp(theta_flat[:, start:end], min=EPS)
                temporary = log_probability[:, None, :] + torch.log(theta)[None, :, :]
                denominator = torch.logsumexp(temporary, dim=2)
                return temporary, denominator

            for j in range(self.base_cluster.n_markers):
                weight = float(self.base_cluster.marker_weights[j])
                if weight <= 0:
                    continue
                _, denominator = marker_terms(j)
                log_resp += weight * denominator
            normalization = torch.logsumexp(log_resp, dim=1, keepdim=True)
            resp_tensor = torch.exp(log_resp - normalization)
            resp = resp_tensor.cpu().numpy().astype(float, copy=False)
            ll = float(normalization.sum().item())

            self.resp = resp
            self.log_likelihood = ll
            self.labels_ = np.argmax(resp, axis=1).astype(str)
            if not return_stats:
                return resp, ll, None

            marker_stats = None
            if return_marker_stats:
                resp_sum = [np.zeros(L, dtype=float) for L in self.base_cluster.n_states]
                x_sum = [np.zeros(L, dtype=float) for L in self.base_cluster.n_states]
                x2_sum = [np.zeros(L, dtype=float) for L in self.base_cluster.n_states]
            theta_counts = np.zeros_like(self.theta_flat)
            for j in range(self.base_cluster.n_markers):
                length = self.base_cluster.n_states[j]
                start, end = self.base_cluster.state_offsets[j:j + 2]
                if self.base_cluster.marker_weights[j] <= 0:
                    theta_counts[:, start:end] = 1.0 / length
                    continue
                temporary, denominator = marker_terms(j)
                state_given_component = torch.exp(
                    temporary - denominator[:, :, None]
                )
                weighted = resp_tensor[:, :, None] * state_given_component
                counts = torch.clamp(
                    weighted.sum(dim=0) + self.alpha_theta - 1.0,
                    min=EPS,
                )
                theta_counts[:, start:end] = counts.cpu().numpy()
                if return_marker_stats:
                    total_state = weighted.sum(dim=1)
                    resp_sum[j] = total_state.sum(dim=0).cpu().numpy()
                    values = x_tensor[:, j:j + 1]
                    x_sum[j] = (total_state * values).sum(dim=0).cpu().numpy()
                    x2_sum[j] = (total_state * values.square()).sum(dim=0).cpu().numpy()
            if return_marker_stats:
                marker_stats = MarkerStateStats(resp_sum, x_sum, x2_sum)
            return resp, ll, (marker_stats, theta_counts)

    def m_step(self, X, stats_pack, shrinkage_strength=0.0, update_marker_parameters=False):
        resp = self.resp
        stats, theta_counts = stats_pack
        counts = resp.sum(axis=0)
        prior = self.alpha_pi / max(self.n_components, 1)
        self.pi = np.maximum(counts + prior - 1.0, EPS)
        self.pi /= self.pi.sum()
        for j in range(self.base_cluster.n_markers):
            start, end = self.base_cluster.state_offsets[j], self.base_cluster.state_offsets[j + 1]
            block = theta_counts[:, start:end]
            block = np.maximum(block, EPS)
            block = block / block.sum(axis=1, keepdims=True)
            exponent = 1.0 + self.theta_sharpening * self.base_cluster.marker_weights[j]
            if exponent > 1.0:
                block = np.power(block, exponent)
                block /= block.sum(axis=1, keepdims=True)
            self.theta_flat[:, start:end] = block
        if not update_marker_parameters:
            return
        state_orders = self.base_cluster.update_parameters(
            X,
            stats,
            shrinkage_strength=shrinkage_strength,
        )
        for j, order in enumerate(state_orders):
            if np.array_equal(order, np.arange(len(order))):
                continue
            start, end = self.base_cluster.state_offsets[j:j + 2]
            self.theta_flat[:, start:end] = self.theta_flat[:, start:end][:, order]

    def _keep_components(self, keep):
        keep = np.asarray(keep, dtype=bool)
        self.pi = self.pi[keep]
        self.theta_flat = self.theta_flat[keep]
        if self.resp is not None:
            self.resp = self.resp[:, keep]
            self.resp /= np.maximum(self.resp.sum(axis=1, keepdims=True), EPS)
        self.n_components = int(keep.sum())
        self.pi /= self.pi.sum()

    def prune(
        self,
        n_cells,
        min_cells=20,
        min_fraction=0.001,
        protected_components=None,
    ):
        if self.n_components <= 1:
            return []
        effective = self.resp.sum(axis=0)
        threshold = max(float(min_cells), float(min_fraction) * n_cells)
        keep = (effective >= threshold) & (self.pi >= min_fraction)
        if protected_components:
            keep[np.asarray(protected_components, dtype=int)] = True
        if not keep.any():
            keep[np.argmax(effective)] = True
        removed = np.flatnonzero(~keep).tolist()
        if removed:
            self._keep_components(keep)
        return removed

    def _js_distance(self, a, b):
        weights = self.base_cluster.marker_weights
        total = weights.sum()
        if total <= EPS:
            return np.inf
        distance = 0.0
        for j, weight in enumerate(weights):
            if weight <= 0:
                continue
            start, end = self.base_cluster.state_offsets[j:j + 2]
            p = np.maximum(a[start:end], EPS)
            q = np.maximum(b[start:end], EPS)
            p, q = p / p.sum(), q / q.sum()
            mean = 0.5 * (p + q)
            js = 0.5 * np.sum(p * np.log(p / mean))
            js += 0.5 * np.sum(q * np.log(q / mean))
            distance += weight * js
        return float(distance / total)

    def _posterior_overlap(self, left, right):
        if self.resp is None:
            return 0.0
        shared = np.minimum(self.resp[:, left], self.resp[:, right]).sum()
        smaller = min(
            self.resp[:, left].sum(),
            self.resp[:, right].sum(),
        )
        return float(shared / max(smaller, EPS))

    def _snapshot(self):
        return {
            "n_components": self.n_components,
            "pi": self.pi.copy(),
            "theta_flat": self.theta_flat.copy(),
            "resp": None if self.resp is None else self.resp.copy(),
            "labels": None if self.labels_ is None else self.labels_.copy(),
            "log_likelihood": float(self.log_likelihood),
            "mu": [value.copy() for value in self.base_cluster.mu],
            "variance": [value.copy() for value in self.base_cluster.variance],
            "component_weights": [
                value.copy() for value in self.base_cluster.component_weights
            ],
        }

    def _restore(self, snapshot):
        self.n_components = int(snapshot["n_components"])
        self.pi = snapshot["pi"]
        self.theta_flat = snapshot["theta_flat"]
        self.resp = snapshot["resp"]
        self.labels_ = snapshot["labels"]
        self.log_likelihood = float(snapshot["log_likelihood"])
        self.base_cluster.mu = snapshot["mu"]
        self.base_cluster.variance = snapshot["variance"]
        self.base_cluster.component_weights = snapshot["component_weights"]

    def _bic(self, n_cells):
        state_parameters = int(np.sum(self.base_cluster.n_states - 1))
        phenotype_parameters = (
            max(self.n_components - 1, 0)
            + self.n_components * state_parameters
        )
        marker_parameters = int(
            np.sum(3 * self.base_cluster.n_states - 1)
        )
        total = phenotype_parameters + marker_parameters
        return float(-2.0 * self.log_likelihood + total * np.log(max(n_cells, 2)))

    def _structural_score(self, n_cells):
        """BIC plus an entropy penalty for heterogeneous phenotype states."""

        score = self._bic(n_cells)
        if self.theta_sharpening <= 0:
            return score
        heterogeneity = 0.0
        total_marker_weight = 0.0
        for j, weight in enumerate(self.base_cluster.marker_weights):
            if weight <= 0 or self.base_cluster.n_states[j] < 2:
                continue
            start, end = self.base_cluster.state_offsets[j:j + 2]
            theta = np.maximum(self.theta_flat[:, start:end], EPS)
            theta /= theta.sum(axis=1, keepdims=True)
            entropy = -np.sum(theta * np.log(theta), axis=1)
            heterogeneity += weight * float(np.sum(self.pi * entropy))
            total_marker_weight += weight
        heterogeneity /= max(total_marker_weight, 1.0)
        return float(score + 2.0 * self.theta_sharpening * n_cells * heterogeneity)

    def _configuration_conflicts(
        self,
        left,
        right,
        min_dominance=0.80,
        min_bhattacharyya=1.00,
        min_marker_weight=0.10,
    ):
        """Count reliable opposite states learned by EM, before final calls exist."""

        conflicts = 0
        for j in range(self.base_cluster.n_markers):
            if (
                self.base_cluster.n_states[j] < 2
                or self.base_cluster.marker_weights[j] < min_marker_weight
            ):
                continue
            mu = self.base_cluster.mu[j]
            variance = self.base_cluster.variance[j]
            distance = bhattacharyya_distance_1d(
                mu[0], variance[0], mu[-1], variance[-1]
            )
            if distance < min_bhattacharyya:
                continue
            start, end = self.base_cluster.state_offsets[j:j + 2]
            left_theta = self.theta_flat[left, start:end]
            right_theta = self.theta_flat[right, start:end]
            if (
                left_theta.max() >= min_dominance
                and right_theta.max() >= min_dominance
                and int(np.argmax(left_theta)) != int(np.argmax(right_theta))
            ):
                conflicts += 1
        return conflicts

    def _merge_pair(self, left, right):
        mass = self.pi[left] + self.pi[right]
        self.theta_flat[left] = (
            self.pi[left] * self.theta_flat[left]
            + self.pi[right] * self.theta_flat[right]
        ) / max(mass, EPS)
        self.pi[left] = mass
        if self.resp is not None:
            self.resp[:, left] += self.resp[:, right]
        keep = np.ones(self.n_components, dtype=bool)
        keep[right] = False
        self._keep_components(keep)

    def structural_prune(self, X, min_cells, min_fraction):
        self.last_prune_protected = []
        snapshot = self._snapshot()
        hard_counts = np.bincount(
            np.argmax(self.resp, axis=1), minlength=self.n_components
        )
        if np.any(hard_counts == 0) and np.any(hard_counts > 0):
            removed = np.flatnonzero(hard_counts == 0).tolist()
            self._keep_components(hard_counts > 0)
            self.e_step(X, return_stats=False)
            return removed
        effective = self.resp.sum(axis=0)
        threshold = max(float(min_cells), float(min_fraction) * X.shape[0])
        large = np.flatnonzero(effective >= threshold)
        for component in np.flatnonzero(effective < threshold):
            candidates = [other for other in large if other != component]
            if not candidates:
                continue
            nearest = min(
                candidates,
                key=lambda other: self._js_distance(
                    self.theta_flat[component], self.theta_flat[other]
                ),
            )
            if self._configuration_conflicts(
                component,
                nearest,
                min_dominance=0.80,
                min_bhattacharyya=1.00,
                min_marker_weight=0.10,
            ) > 0:
                self.last_prune_protected.append(int(component))
        old_score = self._structural_score(X.shape[0])
        removed = self.prune(
            X.shape[0],
            min_cells=min_cells,
            min_fraction=min_fraction,
            protected_components=self.last_prune_protected,
        )
        if not removed:
            return []
        self.e_step(X, return_stats=False)
        if self._structural_score(X.shape[0]) <= old_score:
            return removed
        self._restore(snapshot)
        return []

    def structural_merge(
        self,
        X,
        threshold,
        redundant_threshold,
        min_overlap,
        max_size_ratio,
    ):
        accepted = []
        while self.n_components > 1:
            candidates = []
            for left in range(self.n_components - 1):
                for right in range(left + 1, self.n_components):
                    if self._configuration_conflicts(left, right) > 0:
                        continue
                    distance = self._js_distance(
                        self.theta_flat[left], self.theta_flat[right]
                    )
                    overlap = self._posterior_overlap(left, right)
                    size_ratio = min(self.pi[left], self.pi[right]) / max(
                        self.pi[left], self.pi[right], EPS
                    )
                    close = distance < threshold
                    redundant = (
                        redundant_threshold is not None
                        and distance < redundant_threshold
                        and overlap >= min_overlap
                        and size_ratio <= max_size_ratio
                    )
                    if close or redundant:
                        candidates.append(
                            (distance, -overlap, size_ratio, left, right)
                        )
            if not candidates:
                break
            accepted_this_round = False
            for distance, negative_overlap, size_ratio, left, right in sorted(candidates):
                snapshot = self._snapshot()
                old_score = self._structural_score(X.shape[0])
                self._merge_pair(left, right)
                self.e_step(X, return_stats=False)
                new_score = self._structural_score(X.shape[0])
                if new_score > old_score:
                    self._restore(snapshot)
                    continue
                accepted.append({
                    "left": left,
                    "right": right,
                    "distance": distance,
                    "overlap": -negative_overlap,
                    "size_ratio": size_ratio,
                    "criterion_gain": old_score - new_score,
                })
                accepted_this_round = True
                break
            if not accepted_this_round:
                break
        return accepted

    def structural_split(self, X, min_cells, refinement_steps=2):
        self.last_split_diagnostic = {}
        if (
            self._initialization_features is None
            or self.n_components >= self.max_structural_components
            or self.n_components < 1
        ):
            return []
        hard = np.argmax(self.resp, axis=1)
        weights = self.base_cluster.marker_weights
        scores = []
        for component in range(self.n_components):
            cells = np.flatnonzero(hard == component)
            if cells.size < 2 * min_cells:
                continue
            marker_entropies = []
            for j, weight in enumerate(weights):
                if weight <= 0 or self.base_cluster.n_states[j] < 2:
                    continue
                start, end = self.base_cluster.state_offsets[j:j + 2]
                theta = np.maximum(self.theta_flat[component, start:end], EPS)
                theta /= theta.sum()
                normalized_entropy = float(
                    -np.sum(theta * np.log(theta)) / np.log(len(theta))
                )
                marker_entropies.append(weight * normalized_entropy)
            entropy = max(marker_entropies, default=0.0)
            residual = max(
                (
                    float(
                        np.var(
                            self._initialization_features[
                                cells, marker_slice
                            ],
                            axis=0,
                        ).sum()
                    )
                    for marker_slice in self._initialization_marker_slices.values()
                ),
                default=0.0,
            )
            scores.append((max(entropy, 5.0 * residual), cells.size, component, cells))
        if not scores:
            self.last_split_diagnostic = {"reason": "no_eligible_component"}
            return []
        # Test only the strongest unresolved component. Searching weaker
        # candidates after rejection over-fragments noisy real batches.
        for entropy, _, component, cells in sorted(scores, reverse=True)[:1]:
            accepted = self._try_structural_split_candidate(
                X,
                component=component,
                cells=cells,
                entropy=entropy,
                min_cells=min_cells,
                refinement_steps=refinement_steps,
            )
            if accepted:
                return accepted
        return []

    def _try_structural_split_candidate(
        self,
        X,
        *,
        component,
        cells,
        entropy,
        min_cells,
        refinement_steps,
    ):
        if entropy < 0.075:
            self.last_split_diagnostic = {
                "reason": "low_residual_heterogeneity",
                "score": entropy,
            }
            return []
        split_labels, resolution = _leiden_target_labels(
            self._initialization_features[cells],
            target_components=2,
            random_state=self.random_state + len(self.history) + component,
            n_neighbors=min(20, max(5, cells.size // 20)),
            max_resolution_steps=8,
        )
        counts = np.bincount(split_labels)
        if len(counts) != 2 or counts.min() < min_cells:
            self.last_split_diagnostic = {
                "reason": "invalid_child_size",
                "score": entropy,
                "counts": counts.tolist(),
            }
            return []

        child_features = [
            self._initialization_features[cells[split_labels == group]]
            for group in (0, 1)
        ]
        child_means = [feature.mean(axis=0) for feature in child_features]
        between_variance = float(np.sum((child_means[0] - child_means[1]) ** 2))
        within_variance = float(
            np.mean([
                np.var(feature, axis=0).sum() for feature in child_features
            ])
        )
        variance_ratio = between_variance / max(within_variance, EPS)
        coherent_posterior_split = variance_ratio >= 1.5

        current_independent = self.base_cluster.compute_independent_state_posteriors(X)
        independent = []
        for j, current in enumerate(current_independent):
            marker_slice = self._initialization_marker_slices.get(j)
            if marker_slice is None:
                independent.append(current)
                continue
            initial = np.asarray(
                self._initialization_features[:, marker_slice], dtype=float
            )
            initial /= np.maximum(initial.sum(axis=1, keepdims=True), EPS)
            independent.append(initial)
        group_states = [
            np.vstack([
                posterior[cells[split_labels == group]].mean(axis=0)
                for group in (0, 1)
            ])
            for posterior in independent
        ]
        state_contrasts = [
            float(np.max(np.abs(group_state[0] - group_state[1])))
            for group_state in group_states
        ]
        max_state_contrast = max(state_contrasts, default=0.0)
        if max_state_contrast < 0.25:
            self.last_split_diagnostic = {
                "reason": "no_strong_marker_state_contrast",
                "score": entropy,
                "counts": counts.tolist(),
                "variance_ratio": variance_ratio,
                "max_state_contrast": max_state_contrast,
            }
            return []

        snapshot = self._snapshot()
        old_score = self._structural_score(X.shape[0])
        new_theta = self.theta_flat[component].copy()
        for group in (0, 1):
            selected = cells[split_labels == group]
            target = self.theta_flat[component] if group == 0 else new_theta
            for j, posterior in enumerate(independent):
                start, end = self.base_cluster.state_offsets[j:j + 2]
                block = np.maximum(
                    posterior[selected].sum(axis=0) + self.alpha_theta - 1.0,
                    EPS,
                )
                target[start:end] = block / block.sum()
        for j, group_state in enumerate(group_states):
            if state_contrasts[j] < 0.10:
                continue
            if coherent_posterior_split:
                sharpened = np.power(np.maximum(group_state, EPS), 4.0)
                sharpened /= sharpened.sum(axis=1, keepdims=True)
                start, end = self.base_cluster.state_offsets[j:j + 2]
                self.theta_flat[component, start:end] = sharpened[0]
                new_theta[start:end] = sharpened[1]
            # A newly exposed state can be inaccessible after the preceding EM
            # pass has collapsed its global Gaussian. Re-anchor only markers
            # that distinguish the proposed children; the transaction snapshot
            # restores all parameters when the split criterion rejects it.
            self.base_cluster.mu[j] = self.base_cluster.initial_mu[j].copy()
            self.base_cluster.variance[j] = (
                self.base_cluster.initial_variance[j].copy()
            )
            self.base_cluster.component_weights[j] = (
                self.base_cluster.initial_component_weights[j].copy()
            )
        old_mass = self.pi[component]
        fraction = counts / counts.sum()
        self.pi[component] = old_mass * fraction[0]
        self.pi = np.append(self.pi, old_mass * fraction[1])
        self.theta_flat = np.vstack([self.theta_flat, new_theta])
        self.n_components += 1
        for _ in range(max(int(refinement_steps), 1)):
            _, _, stats = self.e_step(X, return_stats=True, return_marker_stats=False)
            self.m_step(X, stats, update_marker_parameters=False)
        self.e_step(X, return_stats=False)
        new_score = self._structural_score(X.shape[0])
        if new_score >= old_score:
            self.last_split_diagnostic = {
                "reason": "criterion_rejected",
                "score": entropy,
                "old_criterion": old_score,
                "new_criterion": new_score,
                "counts": counts.tolist(),
                "variance_ratio": variance_ratio,
                "max_state_contrast": max_state_contrast,
            }
            self._restore(snapshot)
            return []
        self.last_split_diagnostic = {
            "reason": "accepted",
            "score": entropy,
            "old_criterion": old_score,
            "new_criterion": new_score,
            "counts": counts.tolist(),
            "variance_ratio": variance_ratio,
            "max_state_contrast": max_state_contrast,
        }
        return [{
            "component": component,
            "resolution": resolution,
            "entropy": entropy,
            "left_cells": int(counts[0]),
            "right_cells": int(counts[1]),
            "criterion_gain": old_score - new_score,
            "variance_ratio": variance_ratio,
            "max_state_contrast": max_state_contrast,
        }]

    def reweight_markers_by_phenotype_discrimination(self, floor=0.25):
        """Downweight markers whose state proportions do not vary by phenotype."""

        floor = float(np.clip(floor, 0.0, 1.0))
        prior = self.base_cluster.prior_marker_weights
        updated = np.zeros_like(prior)
        for j in range(self.base_cluster.n_markers):
            if prior[j] <= 0 or self.base_cluster.n_states[j] < 2:
                continue
            start, end = self.base_cluster.state_offsets[j:j + 2]
            theta = np.maximum(self.theta_flat[:, start:end], EPS)
            theta /= theta.sum(axis=1, keepdims=True)
            global_state = np.sum(self.pi[:, None] * theta, axis=0)
            global_state /= global_state.sum()
            mean = 0.5 * (theta + global_state[None, :])
            js = 0.5 * np.sum(
                theta * np.log(theta / mean), axis=1
            )
            js += 0.5 * np.sum(
                global_state[None, :] * np.log(global_state[None, :] / mean),
                axis=1,
            )
            discrimination = float(
                np.sum(self.pi * js) / max(np.log(2.0), EPS)
            )
            discrimination = float(np.clip(discrimination, 0.0, 1.0))
            updated[j] = prior[j] * (
                floor + (1.0 - floor) * np.sqrt(discrimination)
            )
        self.base_cluster.marker_weights = updated
        return updated

    def update_marker_weights_from_em(self, floor=1e-3):
        """Let multivariate phenotype structure rescue or suppress candidate states."""

        floor = float(np.clip(floor, 0.0, 1.0))
        updated = np.zeros(self.base_cluster.n_markers, dtype=float)
        for j in range(self.base_cluster.n_markers):
            if self.base_cluster.n_states[j] < 2:
                continue
            start, end = self.base_cluster.state_offsets[j:j + 2]
            theta = np.maximum(self.theta_flat[:, start:end], EPS)
            theta /= theta.sum(axis=1, keepdims=True)
            global_state = np.sum(self.pi[:, None] * theta, axis=0)
            global_state /= global_state.sum()
            mean = 0.5 * (theta + global_state[None, :])
            js = 0.5 * np.sum(theta * np.log(theta / mean), axis=1)
            js += 0.5 * np.sum(
                global_state[None, :] * np.log(global_state[None, :] / mean),
                axis=1,
            )
            discrimination = float(
                np.clip(np.sum(self.pi * js) / max(np.log(2.0), EPS), 0.0, 1.0)
            )
            mu = self.base_cluster.mu[j]
            variance = self.base_cluster.variance[j]
            nonoverlap = 1.0 - np.exp(
                -bhattacharyya_distance_1d(
                    mu[0], variance[0], mu[-1], variance[-1]
                )
            )
            component_balance = float(
                np.clip(2.0 * np.min(self.base_cluster.component_weights[j]), 0.0, 1.0)
            )
            prior = float(np.clip(self.base_cluster.prior_marker_weights[j], 0.0, 1.0))
            marginal_evidence = np.clip((prior - 0.05) / 0.95, 0.0, 1.0)
            state_evidence = marginal_evidence + (
                1.0 - marginal_evidence
            ) * nonoverlap * component_balance
            evidence = max(
                0.05 * marginal_evidence,
                np.sqrt(discrimination) * state_evidence,
            )
            updated[j] = floor + (1.0 - floor) * evidence
        self.base_cluster.marker_weights = updated
        return updated

    def merge(
        self,
        threshold=0.03,
        redundant_threshold=None,
        min_overlap=0.15,
        max_size_ratio=0.20,
    ):
        merged = []
        while self.n_components > 1:
            candidates = []
            for left in range(self.n_components - 1):
                for right in range(left + 1, self.n_components):
                    distance = self._js_distance(
                        self.theta_flat[left],
                        self.theta_flat[right],
                    )
                    overlap = self._posterior_overlap(left, right)
                    size_ratio = min(self.pi[left], self.pi[right]) / max(
                        self.pi[left],
                        self.pi[right],
                        EPS,
                    )
                    close = distance < threshold
                    redundant = (
                        redundant_threshold is not None
                        and distance < redundant_threshold
                        and overlap >= min_overlap
                        and size_ratio <= max_size_ratio
                    )
                    if close or redundant:
                        candidates.append(
                            (distance, -overlap, size_ratio, left, right)
                        )
            if not candidates:
                break
            distance, neg_overlap, size_ratio, left, right = min(candidates)
            mass = self.pi[left] + self.pi[right]
            self.theta_flat[left] = (
                self.pi[left] * self.theta_flat[left]
                + self.pi[right] * self.theta_flat[right]
            ) / mass
            self.pi[left] = mass
            if self.resp is not None:
                self.resp[:, left] += self.resp[:, right]
            keep = np.ones(self.n_components, dtype=bool)
            keep[right] = False
            self._keep_components(keep)
            merged.append({
                "left": left,
                "right": right,
                "distance": distance,
                "overlap": -neg_overlap,
                "size_ratio": size_ratio,
            })
        return merged

    def fit(
        self,
        X,
        max_iter=100,
        tol=1e-5,
        shrinkage_strength=0.0,
        warmup_iter=5,
        structural_interval=5,
        min_component_cells=10,
        min_component_fraction=0.0005,
        merge_threshold=0.02,
        redundant_merge_threshold=None,
        redundant_min_overlap=0.15,
        redundant_max_size_ratio=0.20,
        reweight_markers=False,
        marker_reweight_interval=2,
        min_bhattacharyya=0.08,
        min_marker_component_weight=0.01,
        min_marker_posterior_confidence=0.60,
        enable_structural_split=True,
        max_accepted_splits=3,
        adaptive_marker_reweight=False,
        adaptive_marker_weight_floor=1e-3,
        theta_sharpening=0.0,
        update_marker_parameters=True,
        initialize_model=True,
    ):
        self.theta_sharpening = max(float(theta_sharpening), 0.0)
        if initialize_model:
            self.initialize(X)
        elif self.resp is None or self.resp.shape[0] != X.shape[0]:
            raise ValueError("continuation EM requires an initialized model")
        initialized_k = self.n_components
        previous = -np.inf
        previous_labels = None
        stable_rounds = 0
        accepted_split_count = 0
        update_marker_parameters = bool(update_marker_parameters)
        for iteration in range(int(max_iter)):
            _, ll, stats_pack = self.e_step(
                X,
                return_stats=True,
                return_marker_stats=update_marker_parameters,
            )
            labels = np.argmax(self.resp, axis=1)
            assignment_change = (
                1.0
                if previous_labels is None
                else float(np.mean(labels != previous_labels))
            )
            theta_before = self.theta_flat.copy()
            mu_before = [value.copy() for value in self.base_cluster.mu]
            variance_before = [value.copy() for value in self.base_cluster.variance]
            marker_shrinkage = 0.0 if iteration < warmup_iter else shrinkage_strength
            self.m_step(
                X,
                stats_pack,
                shrinkage_strength=marker_shrinkage,
                update_marker_parameters=update_marker_parameters,
            )
            if (
                reweight_markers
                and iteration + 1 >= warmup_iter
                and (iteration + 1 - warmup_iter) % marker_reweight_interval == 0
            ):
                self.base_cluster.update_configuration_reliability(
                    min_bhattacharyya=min_bhattacharyya,
                    min_component_weight=min_marker_component_weight,
                )
            max_theta_change = float(np.max(np.abs(self.theta_flat - theta_before)))
            max_peak_mean_change = max(
                float(np.max(np.abs(after - before)))
                for after, before in zip(self.base_cluster.mu, mu_before)
            )
            max_peak_variance_change = max(
                float(
                    np.max(
                        np.abs(
                            np.log(np.maximum(after, EPS))
                            - np.log(np.maximum(before, EPS))
                        )
                    )
                )
                for after, before in zip(
                    self.base_cluster.variance, variance_before
                )
            )
            pruned, merged, split = [], [], []
            if (
                iteration + 1 >= warmup_iter
                and (iteration + 1 - warmup_iter) % structural_interval == 0
            ):
                if adaptive_marker_reweight:
                    self.update_marker_weights_from_em(
                        floor=adaptive_marker_weight_floor
                    )
                pruned = self.structural_prune(
                    X,
                    min_cells=min_component_cells,
                    min_fraction=min_component_fraction,
                )
                merged = self.structural_merge(
                    X,
                    threshold=merge_threshold,
                    redundant_threshold=redundant_merge_threshold,
                    min_overlap=redundant_min_overlap,
                    max_size_ratio=redundant_max_size_ratio,
                )
                if (
                    enable_structural_split
                    and not pruned
                    and not merged
                    and (
                        max_accepted_splits is None
                        or accepted_split_count < max_accepted_splits
                    )
                ):
                    split = self.structural_split(
                        X,
                        min_cells=max(int(min_component_cells), 20),
                    )
                    accepted_split_count += int(bool(split))
                if pruned or merged or split:
                    self.e_step(X, return_stats=False)
                    ll = self.log_likelihood
            rel = (
                np.inf
                if not np.isfinite(previous)
                else abs(ll - previous) / (abs(previous) + 1.0)
            )
            self.history.append({
                "iteration": iteration + 1,
                "initialized_k": initialized_k,
                "log_likelihood": ll,
                "relative_change": rel,
                "assignment_change": assignment_change,
                "max_theta_change": max_theta_change,
                "max_peak_mean_change": max_peak_mean_change,
                "max_peak_variance_change": max_peak_variance_change,
                "n_components": self.n_components,
                "pruned": pruned,
                "prune_protected": self.last_prune_protected.copy(),
                "merged": merged,
                "split": split,
            })
            stable = (
                iteration + 1 >= warmup_iter
                and rel < tol
                and assignment_change < 1e-3
                and max_theta_change < 1e-3
                and max_peak_mean_change < 1e-3
                and max_peak_variance_change < 1e-3
                and not pruned
                and not merged
                and not split
            )
            stable_rounds = stable_rounds + 1 if stable else 0
            if stable_rounds >= 2:
                break
            previous = ll
            previous_labels = labels
        self.e_step(X, return_stats=False)
        return self

    def state_posteriors(self, X):
        logs = self.base_cluster.compute_log_probability(X)
        if self.resp is None or self.resp.shape[0] != X.shape[0]:
            self.e_step(X, return_stats=False)
        out = []
        for j, lp in enumerate(logs):
            start, end = self.base_cluster.state_offsets[j], self.base_cluster.state_offsets[j + 1]
            theta = np.maximum(self.theta_flat[:, start:end], EPS)
            tmp = lp[:, None, :] + np.log(theta)[None, :, :]
            denom = logsumexp(tmp, axis=2)
            p_state_given_k = np.exp(tmp - denom[:, :, None])
            out.append(np.sum(self.resp[:, :, None] * p_state_given_k, axis=1))
        return out


PhenotypeMixtureModel = ParamSharedCluster


class Optimizer:
    """EM optimizer for the phenotype mixture model."""

    def __init__(self, max_iter=100, tol=1e-5, variance_floor=1e-3):
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.variance_floor = float(variance_floor)
        self.history = []

    def initialize(self, X, marker_names=None, initial_resolution=1.0, random_state=0, **kwargs):
        base = BaseCluster.initialize_marker_models(
            X,
            marker_names=marker_names,
            random_state=random_state,
            variance_floor=self.variance_floor,
            **kwargs,
        )
        return ParamSharedCluster(
            base,
            initial_resolution=initial_resolution,
            random_state=random_state,
        )

    def em(self, model, X, shrinkage_strength=0.0):
        model.fit(X, max_iter=self.max_iter, tol=self.tol, shrinkage_strength=shrinkage_strength)
        self.history = model.history
        return model

    def gradient_descent(self, parameters, objective: Callable, gradient: Callable, learning_rate=1e-2):
        parameters = np.asarray(parameters, dtype=float).copy()
        for _ in range(self.max_iter):
            step = learning_rate * np.asarray(gradient(parameters), dtype=float)
            parameters -= step
            self.history.append(float(objective(parameters)))
            if np.linalg.norm(step) < self.tol:
                break
        return parameters


def _build_extreme_state_call_posteriors(
    X,
    state_post,
    base,
    marker_names,
    *,
    random_state=0,
    min_component_weight=0.025,
    min_component_weight_3=0.04,
    bic_gain_23=20.0,
    min_bhattacharyya=0.25,
    trimodal_markers=("CD4",),
):
    """Build ternary-call posteriors without changing phenotype EM.

    The phenotype model historically uses two states per marker.  For marker
    calling, independently test a three-component marginal model.  If it is
    strongly supported, only the lowest and highest components are eligible
    for ``-`` and ``+``; the middle component is deliberately exposed as
    ``?``.  This is a generic safeguard for trimodal markers such as CD4.
    """

    X = np.asarray(X, dtype=float)
    n_cells, n_markers = X.shape
    low = np.zeros((n_cells, n_markers), dtype=np.float32)
    middle = np.zeros((n_cells, n_markers), dtype=np.float32)
    high = np.zeros((n_cells, n_markers), dtype=np.float32)
    call_n_states = np.ones(n_markers, dtype=np.int16)
    metrics = []
    selector = MarkerModelSelector(
        random_state=random_state + 7919,
        n_init=3,
        min_component_weight=min_component_weight,
        min_component_weight_3=min_component_weight_3,
        bic_gain_23=bic_gain_23,
        min_bhattacharyya=min_bhattacharyya,
    )
    # Trimodal calling is deliberately restricted to the biologically
    # validated CD4 marker by default.  Screening every marker is an explicit
    # opt-in (``ALL``); otherwise a marker still has to pass the statistical
    # BIC/separation/weight guard below before its middle state is exposed.
    marker_selection = {str(x) for x in trimodal_markers} if trimodal_markers is not None else {"CD4"}
    allowed_markers = None if "ALL" in marker_selection else marker_selection

    for j in range(n_markers):
        posterior = np.asarray(state_post[j], dtype=float)
        if posterior.ndim != 2 or posterior.shape[1] < 2:
            metrics.append({
                "active": False,
                "bic_gain_23": np.nan,
                "min_adjacent_bhattacharyya": np.nan,
            })
            continue

        low[:, j] = posterior[:, 0].astype(np.float32)
        high[:, j] = posterior[:, -1].astype(np.float32)
        if allowed_markers is not None and str(marker_names[j]) not in allowed_markers:
            call_n_states[j] = 2
            metrics.append({
                "active": False,
                "bic_gain_23": np.nan,
                "min_adjacent_bhattacharyya": np.nan,
                "reason": "not_in_trimodal_marker_list",
            })
            continue
        m2 = selector.fit_candidate_model(X[:, j], 2)
        m3 = selector.fit_candidate_model(X[:, j], 3)
        adjacent = [
            float(
                bhattacharyya_distance_1d(
                    m3.mu[k], m3.variance[k],
                    m3.mu[k + 1], m3.variance[k + 1],
                )
            )
            for k in range(2)
        ] if m3.valid else []
        min_adjacent = min(adjacent) if adjacent else 0.0
        gain_23 = float(m2.bic - m3.bic)
        active = bool(
            int(base.n_states[j]) == 2
            and float(base.configuration_weights[j]) > 0.0
            and m2.valid
            and m3.valid
            and gain_23 >= float(bic_gain_23)
            and min_adjacent >= float(min_bhattacharyya)
            and np.min(m3.weights) >= float(min_component_weight_3)
        )
        if active:
            low[:, j] = m3.posterior[:, 0].astype(np.float32)
            middle[:, j] = m3.posterior[:, 1].astype(np.float32)
            high[:, j] = m3.posterior[:, 2].astype(np.float32)
            call_n_states[j] = 3
        else:
            call_n_states[j] = 2
        metrics.append({
            "active": active,
            "bic_gain_23": gain_23,
            "min_adjacent_bhattacharyya": float(min_adjacent),
            "reason": "trimodal_guard_active" if active else "criteria_not_met",
            "component_weights": m3.weights.tolist(),
            "component_means": m3.mu.tolist(),
        })
    return low, middle, high, call_n_states, metrics


def fit_shared_marker_gmm(
    X,
    marker_names=None,
    random_state=0,
    max_iter=100,
    tol=1e-5,
    min_component_weight=0.02,
    min_component_weight_3=0.03,
    min_bic_gain=10.0,
    bic_gain_12=None,
    bic_gain_23=20.0,
    icl_gain_23=5.0,
    trimodal_min_bhattacharyya=0.25,
    min_bhattacharyya=None,
    min_separation_score=None,
    min_separation=1.0,
    min_mean_posterior_conf=0.70,
    jitter_noise_sd=1e-3,
    jitter_mode="all",
    jitter_for_posterior=False,
    init_q_low=None,
    init_q_high=None,
    shrinkage=None,
    outlier_weight=None,
    student_df=None,
    initial_resolution=1.0,
    alpha_theta=0.2,
    alpha_pi=0.2,
    warmup_iter=5,
    structural_interval=5,
    min_component_cells=10,
    min_component_fraction=0.0005,
    merge_threshold=0.02,
    redundant_merge_threshold=None,
    redundant_min_overlap=0.15,
    redundant_max_size_ratio=0.20,
    reweight_markers=False,
    marker_reweight_interval=2,
    min_marker_component_weight=0.01,
    min_marker_posterior_confidence=0.60,
    min_configuration_bhattacharyya=1.0,
    marker_weight_multipliers=None,
    structural_marker_weight_multipliers=None,
    discrimination_reweight=False,
    discrimination_floor=0.25,
    theta_sharpening=0.5,
    adaptive_marker_reweight=False,
    update_marker_parameters=True,
    enable_structural_split=False,
    em_alpha=None,
    stability_tol=0.35,
    device="cpu",
    trimodal_markers=("CD4",),
):
    """Fit heterogeneous marker states and a finite phenotype mixture."""
    del init_q_low, init_q_high, outlier_weight, student_df
    del icl_gain_23
    X = np.asarray(X, dtype=np.float64)
    n_cells, n_markers = X.shape
    initial_resolution = float(initial_resolution)
    if not np.isfinite(initial_resolution) or initial_resolution <= 0:
        raise ValueError("initial_resolution must be finite and positive")
    marker_names = (
        np.asarray(marker_names, dtype=str)
        if marker_names is not None
        else np.asarray([f"marker_{j}" for j in range(n_markers)])
    )
    if marker_names.size != n_markers:
        raise ValueError("marker_names must match the number of columns in X.")
    if bic_gain_12 is None:
        bic_gain_12 = min_bic_gain
    del min_separation, min_separation_score, min_mean_posterior_conf
    del min_marker_posterior_confidence
    if min_bhattacharyya is None:
        min_bhattacharyya = 0.08

    X_fit = add_gaussian_jitter(X, noise_sd=jitter_noise_sd, random_state=random_state, mode=jitter_mode)
    Z_fit, center, scale = robust_zscore_per_marker(X_fit)
    Z_post = Z_fit if jitter_for_posterior else (X - center) / scale

    selector_kwargs = dict(
        min_component_weight=min_component_weight,
        bic_gain_12=bic_gain_12,
        min_bhattacharyya=min_bhattacharyya,
        stability_tol=stability_tol,
    )
    base = BaseCluster.initialize_marker_models(
        Z_fit,
        marker_names=marker_names,
        random_state=random_state,
        **selector_kwargs,
    )
    if marker_weight_multipliers is not None:
        multipliers = np.asarray(marker_weight_multipliers, dtype=float)
        if multipliers.shape != (n_markers,):
            raise ValueError("marker_weight_multipliers must have one value per marker.")
        if np.any(~np.isfinite(multipliers)) or np.any(multipliers < 0):
            raise ValueError("marker_weight_multipliers must be finite and non-negative.")
        base.marker_weights *= multipliers
        base.prior_marker_weights *= multipliers
        base.configuration_weights *= multipliers
        base.model_selection_metrics["platform_weight_multiplier"] = multipliers
        base.model_selection_metrics["cluster_weight"] = base.marker_weights
        base.model_selection_metrics["configuration_weight"] = (
            base.configuration_weights
        )
        base.model_selection_metrics["phenotype_weight"] = (
            base.configuration_weights
        )
    if structural_marker_weight_multipliers is not None:
        multipliers = np.asarray(structural_marker_weight_multipliers, dtype=float)
        if multipliers.shape != (n_markers,):
            raise ValueError(
                "structural_marker_weight_multipliers must have one value per marker."
            )
        if np.any(~np.isfinite(multipliers)) or np.any(multipliers < 0):
            raise ValueError(
                "structural_marker_weight_multipliers must be finite and non-negative."
            )
        base.marker_weights *= multipliers
        base.prior_marker_weights *= multipliers
        base.model_selection_metrics["structural_weight_multiplier"] = multipliers
        base.model_selection_metrics["cluster_weight"] = base.marker_weights
    if shrinkage is None:
        shrinkage = 0.0
    if em_alpha is not None:
        alpha_theta = em_alpha
    model = ParamSharedCluster(
        base,
        initial_resolution=initial_resolution,
        alpha_theta=alpha_theta,
        alpha_pi=alpha_pi,
        random_state=random_state,
        device=device,
    )
    model.fit(
        Z_fit,
        max_iter=max_iter,
        tol=tol,
        shrinkage_strength=shrinkage,
        warmup_iter=warmup_iter,
        structural_interval=structural_interval,
        min_component_cells=min_component_cells,
        min_component_fraction=min_component_fraction,
        merge_threshold=merge_threshold,
        redundant_merge_threshold=redundant_merge_threshold,
        redundant_min_overlap=redundant_min_overlap,
        redundant_max_size_ratio=redundant_max_size_ratio,
        reweight_markers=reweight_markers,
        marker_reweight_interval=marker_reweight_interval,
        min_bhattacharyya=min_bhattacharyya,
        min_marker_component_weight=min_marker_component_weight,
        min_marker_posterior_confidence=0.0,
        adaptive_marker_reweight=adaptive_marker_reweight,
        theta_sharpening=theta_sharpening,
        update_marker_parameters=update_marker_parameters,
        enable_structural_split=enable_structural_split,
    )
    first_pass_history = list(model.history)
    if discrimination_reweight and np.any(base.marker_weights > 0):
        model.update_marker_weights_from_em(floor=discrimination_floor * 0.01)
        model.history = []
        model.fit(
            Z_fit,
            max_iter=max_iter,
            tol=tol,
            shrinkage_strength=shrinkage,
            warmup_iter=warmup_iter,
            structural_interval=structural_interval,
            min_component_cells=min_component_cells,
            min_component_fraction=min_component_fraction,
            merge_threshold=merge_threshold,
            redundant_merge_threshold=redundant_merge_threshold,
            redundant_min_overlap=redundant_min_overlap,
            redundant_max_size_ratio=redundant_max_size_ratio,
            reweight_markers=False,
            min_bhattacharyya=min_bhattacharyya,
            min_marker_component_weight=min_marker_component_weight,
            min_marker_posterior_confidence=0.0,
            adaptive_marker_reweight=False,
            theta_sharpening=theta_sharpening,
            update_marker_parameters=update_marker_parameters,
            enable_structural_split=enable_structural_split,
            initialize_model=False,
        )
    model.e_step(Z_post, return_stats=False)
    state_post = model.state_posteriors(Z_post)
    base.update_configuration_reliability(
        min_bhattacharyya=min_configuration_bhattacharyya,
        min_component_weight=min_marker_component_weight,
    )

    call_low, call_middle, call_high, call_n_states, trimodal_metrics = (
        _build_extreme_state_call_posteriors(
            Z_post,
            state_post,
            base,
            marker_names,
            random_state=random_state,
            min_component_weight=min_component_weight,
            min_component_weight_3=min_component_weight_3,
            bic_gain_23=bic_gain_23,
            min_bhattacharyya=trimodal_min_bhattacharyya,
            trimodal_markers=trimodal_markers,
        )
    )
    high_prob = np.zeros((n_cells, n_markers), dtype=np.float32)
    low_prob = np.zeros((n_cells, n_markers), dtype=np.float32)
    unimodal_prob = np.zeros((n_cells, n_markers), dtype=np.float32)
    empty_prob = np.zeros((n_cells, n_markers), dtype=np.float32)
    state_argmax = np.zeros((n_cells, n_markers), dtype=np.int16)
    for j, post in enumerate(state_post):
        L = base.n_states[j]
        state_argmax[:, j] = np.argmax(post, axis=1)
        if L == 1:
            unimodal_prob[:, j] = 1.0
        else:
            low_prob[:, j] = post[:, 0].astype(np.float32)
            high_prob[:, j] = post[:, -1].astype(np.float32)
    marker_params = base.to_marker_params(center=center, scale=scale, n_iter=len(model.history))
    marker_params["call_n_states"] = call_n_states
    marker_params["trimodal_bic_gain_23"] = [
        record["bic_gain_23"] for record in trimodal_metrics
    ]
    marker_params["trimodal_min_adjacent_bhattacharyya"] = [
        record["min_adjacent_bhattacharyya"] for record in trimodal_metrics
    ]
    marker_params["trimodal_guard_active"] = [
        record["active"] for record in trimodal_metrics
    ]
    marker_params["trimodal_guard_reason"] = [
        record.get("reason", "criteria_not_met") for record in trimodal_metrics
    ]

    return {
        "high_prob": high_prob,
        "low_prob": low_prob,
        # These are used only for the final ternary marker call.  The main
        # phenotype EM remains unchanged; a reliable third component makes
        # the middle state unknown instead of silently folding it into '+'.
        "call_low_prob": call_low,
        "call_middle_prob": call_middle,
        "call_high_prob": call_high,
        "call_n_states": call_n_states,
        "trimodal_metrics": trimodal_metrics,
        "unimodal_prob": unimodal_prob,
        "empty_prob": empty_prob,
        "state_argmax": state_argmax,
        "state_posteriors": state_post,
        "phenotype_posterior": model.resp.astype(np.float32),
        "posterior_likelihood": model.resp.astype(np.float32),
        "phenotype_assignment": np.asarray(model.labels_, dtype=str),
        "phenotype_confidence": model.resp.max(axis=1).astype(np.float32),
        "phenotype_margin": (
            np.sort(model.resp, axis=1)[:, -1]
            - (
                np.sort(model.resp, axis=1)[:, -2]
                if model.n_components > 1
                else 0.0
            )
        ).astype(np.float32),
        "possible_secondary_phenotype": (
            np.argsort(-model.resp, axis=1)[:, 1].astype(str)
            if model.n_components > 1
            else np.full(n_cells, "", dtype=str)
        ),
        "phenotype_counts": np.bincount(model.resp.argmax(axis=1), minlength=model.n_components),
        "phenotype_proportions": model.pi.copy(),
        "theta_flat": model.theta_flat,
        "pi": model.pi,
        "marker_state_offsets": base.state_offsets.copy(),
        "marker_n_states": base.n_states.copy(),
        "marker_weights": base.marker_weights.copy(),
        "configuration_weights": base.configuration_weights.copy(),
        "marker_status": base.marker_status.copy(),
        "model_selection_metrics": base.model_selection_metrics.copy(),
        "log_likelihood_history": model.history,
        "component_history": model.history,
        "first_pass_history": first_pass_history,
        "marker_params": marker_params,
        "marker_center": center,
        "marker_scale": scale,
        "initialization_method": model.initialization_method,
        "initialization_resolution": model.initialization_resolution,
        "compute_device": model.device,
        "update_marker_parameters": bool(update_marker_parameters),
        "initialized_components": (
            int(first_pass_history[0]["initialized_k"])
            if first_pass_history else int(model.n_components)
        ),
        "max_structural_components": model.max_structural_components,
        "model": model,
        "optimizer": model,
        "n_iter": len(model.history),
    }


fit_phenotypes = fit_shared_marker_gmm
