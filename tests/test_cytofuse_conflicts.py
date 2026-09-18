"""Direct tests for rare phenotype pruning at the EM stage."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from citepool_baseline.cytofuse.methods.paramshared import (
    BaseCluster,
    ParamSharedCluster,
)
from citepool_baseline.cytofuse.tree import (
    build_cell_tree_assignments, build_taxonomy,
    consolidate_equivalent_local_clusters,
)


class LocalConsolidationTests(unittest.TestCase):
    def states(self, right_confidence):
        return pd.DataFrame([
            {"local_node_id": f"batch|{cluster}", "batch": "batch",
             "cluster": str(cluster), "n_cells": 1000 if cluster == 0 else 50,
             "marker": marker, "state": state,
             "call_confidence": confidence}
            for cluster in (0, 1)
            for marker, state, confidence in (
                ("M1", "+", 0.9), ("M2", "-", 0.9), ("M3", "+", 0.9),
                ("M4", "+" if cluster == 0 else "-", right_confidence),
            )
        ])

    def test_weak_calls_do_not_fragment_equivalent_clusters(self):
        _, mapping = consolidate_equivalent_local_clusters(self.states(0.4))
        self.assertEqual(mapping.local_node_id.nunique(), 1)

    def test_reliable_conflict_protects_rare_cluster(self):
        _, mapping = consolidate_equivalent_local_clusters(self.states(0.9))
        self.assertEqual(mapping.local_node_id.nunique(), 2)

    def test_insufficient_coverage_preserves_coarse_node(self):
        states = self.states(0.4)
        extra = states.iloc[[0]].copy()
        for marker in ("M5", "M6"):
            extra["marker"] = marker
            states = pd.concat([states, extra.copy()], ignore_index=True)
        _, mapping = consolidate_equivalent_local_clusters(states)
        self.assertEqual(mapping.local_node_id.nunique(), 2)


def make_model(mu, variance):
    base = BaseCluster(
        n_states=[2],
        mu=[mu],
        variance=[variance],
        component_weights=[[0.99, 0.01]],
        marker_status=["candidate_two_state"],
        marker_weights=[1.0],
        configuration_weights=[0.0],
        model_selection_metrics=pd.DataFrame([{"marker": "M1"}]),
        marker_names=["M1"],
    )
    model = ParamSharedCluster(base, n_components=2, random_state=0)
    model.pi = np.asarray([0.99, 0.01])
    model.theta_flat = np.asarray([[0.999, 0.001], [0.001, 0.999]])
    return model


class StrongConflictPruningTests(unittest.TestCase):
    def test_rare_strong_conflict_is_not_pruned(self):
        rng = np.random.default_rng(3)
        X = np.concatenate([
            rng.normal(-2.0, 0.2, 990), rng.normal(2.0, 0.2, 10)
        ])[:, None]
        model = make_model([-2.0, 2.0], [0.04, 0.04])
        model.e_step(X)
        removed = model.structural_prune(X, min_cells=40, min_fraction=0.0005)
        self.assertEqual(removed, [])
        self.assertEqual(model.last_prune_protected, [1])
        self.assertEqual(model.n_components, 2)

    def test_rare_weak_overlap_remains_prunable(self):
        rng = np.random.default_rng(4)
        X = np.concatenate([
            rng.normal(-0.1, 1.0, 990), rng.normal(0.1, 1.0, 10)
        ])[:, None]
        model = make_model([-0.1, 0.1], [1.0, 1.0])
        model.resp = np.zeros((1000, 2), dtype=float)
        model.resp[:990, 0] = 1.0
        model.resp[990:, 1] = 1.0
        model.labels_ = np.argmax(model.resp, axis=1).astype(str)
        model.log_likelihood = -2000.0
        removed = model.structural_prune(X, min_cells=40, min_fraction=0.0005)
        self.assertNotIn(1, model.last_prune_protected)
        self.assertEqual(model.n_components, 1)
        self.assertEqual(removed, [1])

    def test_taxonomy_prefers_conflict_free_partner(self):
        membership = pd.DataFrame({
            "aligned_node_id": ["NK_A", "NK_B", "CONFLICTING"],
            "local_node_id": ["b1|0", "b2|0", "b3|0"],
        })
        core = {
            "CD16": "+", "CD45RA": "+", "CD3": "-", "CD8": "-",
            "M1": "-", "M2": "-", "M3": "-", "M4": "-",
        }
        signatures = {
            "NK_A": core,
            "NK_B": {**core, "M5": "-", "M6": "-"},
            "CONFLICTING": {
                **core, "CD3": "+", "CD8": "+", "M5": "-", "M6": "-",
            },
        }
        states = pd.DataFrame([
            {"aligned_node_id": node, "marker": marker, "state": state,
             "call_confidence": 1.0, "marker_weight": 1.0}
            for node, signature in signatures.items()
            for marker, state in signature.items()
        ])
        result = build_taxonomy(
            membership,
            states,
            min_tree_shared_observed=1,
            use_weighted_scores=False,
        )
        first = result["merges"].sort_values("round").iloc[0]
        self.assertEqual({first["child_a"], first["child_b"]}, {"NK_A", "NK_B"})
        self.assertEqual(int(first["n_conflicts"]), 0)

    def test_marker_overlap_can_merge_without_a_lineage_prior(self):
        membership = pd.DataFrame({
            "aligned_node_id": ["A", "B"],
            "local_node_id": ["b1|0", "b2|0"],
        })
        signatures = {
            "A": {"M1": "+", "M2": "-", "M3": "+", "CD3": "+"},
            "B": {"M1": "+", "M2": "-", "M3": "+", "CD19": "+"},
        }
        states = pd.DataFrame([
            {"aligned_node_id": node, "marker": marker, "state": state,
             "call_confidence": 1.0, "marker_weight": 1.0}
            for node, signature in signatures.items()
            for marker, state in signature.items()
        ])
        result = build_taxonomy(
            membership, states, min_tree_shared_observed=3,
            use_weighted_scores=False,
        )
        self.assertEqual(len(result["merges"]), 1)

    def test_multibatch_final_cut_does_not_merge_on_negative_evidence_only(self):
        membership = pd.DataFrame({
            "aligned_node_id": ["A", "A", "B", "B", "C", "C"],
            "local_node_id": ["b1|0", "b2|0", "b1|1", "b2|1", "b1|2", "b2|2"],
            "batch": ["b1", "b2", "b1", "b2", "b1", "b2"],
        })
        states = pd.DataFrame([
            {"aligned_node_id": node, "marker": marker, "state": state,
             "call_confidence": 1.0, "marker_weight": 1.0}
            for node, signature in {
                "A": {"NK": "+", "M1": "-", "M2": "-", "M3": "-"},
                "B": {"NK": "+", "M1": "-", "M2": "-", "M3": "-"},
                "C": {"NK": "?", "M1": "-", "M2": "-", "M3": "-"},
            }.items()
            for marker, state in signature.items()
        ])
        taxonomy = build_taxonomy(
            membership, states, min_tree_shared_observed=1,
            use_weighted_scores=False,
        )
        cells = pd.DataFrame({
            "cell_barcode": ["a", "b", "c"],
            "batch": ["b1", "b1", "b1"],
            "protein_phenotype": ["0", "1", "2"],
        })
        result = build_cell_tree_assignments(
            cells, membership, taxonomy["edges"],
            tree_merges=taxonomy["merges"],
            final_min_shared_observed=1,
        )
        final = result.set_index("cell_barcode")["cytofuse_final_id"]
        self.assertEqual(final["a"], final["b"])
        self.assertNotEqual(final["a"], final["c"])

    def test_single_batch_absorption_keeps_marker_and_conflict_checks(self):
        for shared, conflicts, expected in [(6, 0, True), (6, 1, False), (2, 0, False)]:
            with self.subTest(shared=shared, conflicts=conflicts):
                membership = pd.DataFrame({
                    "aligned_node_id": ["A", "B", "B"],
                    "local_node_id": ["b1|0", "b1|1", "b2|1"],
                    "batch": ["b1", "b1", "b2"],
                })
                cells = pd.DataFrame({
                    "cell_barcode": ["a", "b", "c"],
                    "batch": ["b1", "b1", "b2"],
                    "protein_phenotype": ["0", "1", "1"],
                })
                edges = pd.DataFrame({"parent": ["P", "P"], "child": ["A", "B"]})
                merges = pd.DataFrame([dict(round=1, parent="P", child_a="A", child_b="B",
                    n_shared_observed=shared, n_common_positive=0, n_conflicts=conflicts)])
                result = build_cell_tree_assignments(cells, membership, edges, tree_merges=merges)
                labels = result.set_index("cell_barcode").cytofuse_final_id
                self.assertEqual(labels["a"] == labels["b"], expected)
                if expected:
                    self.assertEqual(set(labels), {"P"})
                    self.assertEqual(set(result.cytofuse_final_merge_reason), {"single_batch_child_absorption"})


if __name__ == "__main__":
    unittest.main()
