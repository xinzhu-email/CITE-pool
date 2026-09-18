"""Regression tests for adaptive structural phenotype EM."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from citepool_baseline.cytofuse.methods.paramshared import (
    BaseCluster,
    MarkerStateStats,
    ParamSharedCluster,
    fit_shared_marker_gmm,
    leiden_initial_labels,
)


class StructuralEMRegressionTests(unittest.TestCase):
    @staticmethod
    def _six_phenotype_data():
        rng = np.random.default_rng(9)
        counts = np.asarray([250, 210, 180, 150, 90, 20])
        configuration = np.asarray(
            [
                [0, 0, 0, 1, 1, 0],
                [1, 0, 0, 0, 1, 0],
                [0, 1, 0, 1, 0, 0],
                [1, 1, 0, 0, 0, 1],
                [0, 0, 1, 1, 0, 1],
                [1, 1, 1, 0, 1, 1],
            ]
        )
        truth = np.repeat(np.arange(len(counts)), counts)
        values = rng.normal(
            np.where(configuration[truth] > 0, 1.8, -1.8),
            0.38,
        )
        order = rng.permutation(len(truth))
        return values[order], truth[order]

    def test_resolution_controls_granularity_without_peak_drift(self):
        values, truth = self._six_phenotype_data()
        observed_components = []
        for initial_resolution in (0.1, 2.0):
            with self.subTest(initial_resolution=initial_resolution):
                result = fit_shared_marker_gmm(
                    values,
                    marker_names=[f"M{i}" for i in range(values.shape[1])],
                    initial_resolution=initial_resolution,
                    random_state=2,
                    max_iter=25,
                    jitter_noise_sd=0.0,
                    update_marker_parameters=False,
                    discrimination_reweight=True,
                )
                labels = np.asarray(result["phenotype_assignment"])
                observed_components.append(np.unique(labels).size)
                self.assertGreater(adjusted_rand_score(truth, labels), 0.70)
                rare = truth == 5
                rare_recall = max(
                    np.mean(labels[rare] == label) for label in np.unique(labels)
                )
                self.assertGreater(rare_recall, 0.70)
                model = result["model"]
                for mu, initial_mu in zip(
                    model.base_cluster.mu,
                    model.base_cluster.initial_mu,
                ):
                    np.testing.assert_allclose(mu, initial_mu)
                for variance, initial_variance in zip(
                    model.base_cluster.variance,
                    model.base_cluster.initial_variance,
                ):
                    np.testing.assert_allclose(variance, initial_variance)
        self.assertLessEqual(observed_components[0], observed_components[1])

    def test_leiden_resolution_directly_controls_initial_granularity(self):
        rng = np.random.default_rng(17)
        angle = np.linspace(0, 8 * np.pi, 1200)
        features = np.column_stack(
            [angle * np.cos(angle), angle * np.sin(angle)]
        ) + rng.normal(0, 0.35, (1200, 2))
        coarse, _ = leiden_initial_labels(features, resolution=0.1)
        fine, _ = leiden_initial_labels(features, resolution=2.0)
        self.assertLess(np.unique(coarse).size, np.unique(fine).size)

    def test_peak_reordering_also_reorders_configuration_states(self):
        base = BaseCluster(
            n_states=[2],
            mu=[[-1.0, 1.0]],
            variance=[[1.0, 1.0]],
            component_weights=[[0.5, 0.5]],
            marker_status=["candidate_two_state"],
            marker_weights=[1.0],
            configuration_weights=[0.0],
            model_selection_metrics=pd.DataFrame([{"marker": "M0"}]),
            marker_names=["M0"],
        )
        model = ParamSharedCluster(base, n_components=2, random_state=0)
        model.resp = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        )
        model.theta_flat = np.asarray([[0.9, 0.1], [0.2, 0.8]])
        stats = MarkerStateStats(
            resp_sum=[np.asarray([2.0, 2.0])],
            x_sum=[np.asarray([2.0, -2.0])],
            x2_sum=[np.asarray([4.0, 4.0])],
        )
        theta_counts = np.asarray([[9.0, 1.0], [2.0, 8.0]])

        model.m_step(
            np.asarray([[1.0], [1.0], [-1.0], [-1.0]]),
            (stats, theta_counts),
            update_marker_parameters=True,
        )

        np.testing.assert_allclose(base.mu[0], [-1.0, 1.0])
        np.testing.assert_allclose(model.theta_flat, [[0.1, 0.9], [0.8, 0.2]])

    def test_default_m_step_keeps_initialized_peak_parameters_fixed(self):
        base = BaseCluster(
            n_states=[2],
            mu=[[-1.0, 1.0]],
            variance=[[1.0, 1.0]],
            component_weights=[[0.5, 0.5]],
            marker_status=["candidate_two_state"],
            marker_weights=[1.0],
            configuration_weights=[0.0],
            model_selection_metrics=pd.DataFrame([{"marker": "M0"}]),
            marker_names=["M0"],
        )
        model = ParamSharedCluster(base, n_components=2, random_state=0)
        model.resp = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        )
        model.theta_flat = np.asarray([[0.9, 0.1], [0.2, 0.8]])
        theta_counts = np.asarray([[9.0, 1.0], [2.0, 8.0]])

        model.m_step(
            np.asarray([[1.0], [1.0], [-1.0], [-1.0]]),
            (None, theta_counts),
        )

        np.testing.assert_allclose(base.mu[0], [-1.0, 1.0])
        np.testing.assert_allclose(base.variance[0], [1.0, 1.0])
        np.testing.assert_allclose(model.theta_flat, [[0.9, 0.1], [0.2, 0.8]])


if __name__ == "__main__":
    unittest.main()
