from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from citepool_baseline.training import (
    _apply_resolution_atlas_supervision,
    taxonomy_masks,
)


class TaxonomyMaskTests(unittest.TestCase):
    def test_atlas_child_internal_label_expands_to_active_descendants(self) -> None:
        with TemporaryDirectory() as temporary:
            table_dir = Path(temporary)
            (table_dir / "tables").mkdir()
            pd.DataFrame(
                {
                    "parent": ["P", "P"],
                    "child": ["L1", "L2"],
                }
            ).to_csv(table_dir / "tables/taxonomy_edges.csv", index=False)
            pd.DataFrame(
                {
                    "parent": ["ATLAS_PARENT"],
                    "child": ["ATLAS_CHILD"],
                }
            ).to_csv(table_dir / "tables/resolution_atlas_edges.csv", index=False)

            cells = pd.DataFrame(
                {
                    "supervision_node": [
                        "L1", "L2", "L3", "ATLAS_PARENT", "P", "L3", np.nan,
                    ],
                    "supervision_kind": [
                        "leaf", "leaf", "leaf", "internal", "internal", "leaf", "unlabeled",
                    ],
                    "assigned_taxonomy_node_id": [
                        "L1", "L2", "L3", "ATLAS_PARENT", "P", "L3", np.nan,
                    ],
                    "aligned_leaf_id": [
                        "OTHER", "OTHER", "OTHER", "ATLAS_PARENT",
                        "ATLAS_CHILD", "ATLAS_CHILD", "ATLAS_CHILD",
                    ],
                }
            )

            leaves, mask = taxonomy_masks(
                table_dir,
                cells,
                "supervision_node",
                "supervision_kind",
            )

            parent_mask = mask[3].astype(bool)
            self.assertEqual(
                {leaf for leaf, allowed in zip(leaves, parent_mask) if allowed},
                {"L1", "L2", "L3"},
            )

    def test_atlas_promotion_preserves_explicit_rna_terminal_leaf(self) -> None:
        with TemporaryDirectory() as temporary:
            table_dir = Path(temporary)
            (table_dir / "tables").mkdir()
            pd.DataFrame(
                {
                    "parent": ["ATLAS_PARENT"],
                    "child": ["ATLAS_CHILD"],
                }
            ).to_csv(table_dir / "tables/resolution_atlas_edges.csv", index=False)
            cells = pd.DataFrame(
                {
                    "assigned_taxonomy_node_id": ["P__RNA_LEAF_0", "P"],
                    "assigned_node_kind": ["leaf", "internal"],
                    "atlas_node_id": ["ATLAS_PARENT", "ATLAS_PARENT"],
                    "rna_assignment_level": ["terminal_leaf", "partial_parent"],
                    "rna_refinement_parent_leaf": ["P", "P"],
                }
            )

            result, promoted = _apply_resolution_atlas_supervision(
                table_dir,
                cells,
                "assigned_taxonomy_node_id",
                "assigned_node_kind",
                enabled=True,
            )

            self.assertEqual(promoted.tolist(), [False, True])
            self.assertEqual(
                result["assigned_taxonomy_node_id"].tolist(),
                ["P__RNA_LEAF_0", "ATLAS_PARENT"],
            )
            self.assertEqual(
                result["assigned_node_kind"].tolist(),
                ["leaf", "internal"],
            )


if __name__ == "__main__":
    unittest.main()
