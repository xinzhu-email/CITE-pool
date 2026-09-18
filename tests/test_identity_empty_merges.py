"""Valid disconnected protein identities must survive a no-merge hierarchy."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import pandas as pd
from citepool_baseline.identity._engine.integrated_tree import build_integrated_hierarchy


class EmptyMergeHierarchyTests(unittest.TestCase):
    def test_no_merge_headers_and_legacy_blank_file_keep_identities(self):
        for content in ['parent,child_a,child_b,round\n', '\n']:
            with self.subTest(content=content), TemporaryDirectory() as tmp:
                tables = Path(tmp) / 'tables'; tables.mkdir()
                pd.DataFrame({'aligned_node_id': ['ALIGNED_A', 'ALIGNED_B'],
                              'local_node_id': ['b1|0', 'b1|1']}).to_csv(tables / 'alignment_membership.csv', index=False)
                pd.DataFrame({'cell_barcode': ['a', 'b'],
                    'cytofuse_final_id': ['ALIGNED_A', 'ALIGNED_B'],
                    'aligned_leaf_id': ['ALIGNED_A', 'ALIGNED_B']}).to_csv(tables / 'cell_taxonomy_assignments.csv', index=False)
                (tables / 'taxonomy_merges.csv').write_text(content)
                summary = build_integrated_hierarchy(Path(tmp))
                self.assertEqual(summary['n_taxonomy_merges_input'], 0)
                self.assertEqual(summary['n_aligned_nodes'], 2)
                assignments = pd.read_csv(tables / 'cell_taxonomy_assignments.csv')
                self.assertEqual(assignments.cytofuse_final_id.tolist(), ['ALIGNED_A', 'ALIGNED_B'])
                self.assertEqual(assignments.integrated_supervision_node_id.tolist(), ['ALIGNED_A', 'ALIGNED_B'])


if __name__ == '__main__': unittest.main()
