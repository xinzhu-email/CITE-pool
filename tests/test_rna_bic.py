"""Regression tests for BIC gating before 1-D RNA branches are admitted."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import anndata as ad
from citepool_baseline import rna_refinement as rr

class OneDimensionalBICTests(unittest.TestCase):
    def data(self, values):
        ids=[f"c{i}" for i in range(len(values))]
        a=ad.AnnData(np.zeros((len(values),2)),obs=pd.DataFrame(index=ids))
        return pd.DataFrame({"CC_1":values},index=ids),{"batch":a}

    def score(self, values, cutoff):
        pca,data=self.data(values)
        return rr._score_pcs(pca,data,separation_cutoff=0,dip_cutoff=0,
            partition_cutoff=0,variance_cutoff=0,min_batch_cells=100,
            seed=2026,bic_gain_per_cell=cutoff).iloc[0]

    def test_unimodal_candidate_rejected_even_when_other_gates_pass(self):
        row=self.score(np.random.default_rng(10).normal(size=1000),0.05)
        self.assertLess(row.bic_gain_per_cell,0.05)
        self.assertFalse(row.bic_passed)
        self.assertFalse(row.passed)
        self.assertAlmostEqual(row.bic_gain_per_cell,(row.bic_single-row.bic_split)/1000)

    def test_bimodal_candidate_requires_configured_gain(self):
        rng=np.random.default_rng(10)
        values=np.r_[rng.normal(-4,.5,500),rng.normal(4,.5,500)]
        row=self.score(values,0.05)
        self.assertTrue(row.passed)
        rejected=self.score(values,row.bic_gain_per_cell+0.01)
        self.assertFalse(rejected.passed)

    def test_threshold_inheritance_and_override_reach_recursive_children(self):
        rng=np.random.default_rng(10)
        pca,data=self.data(np.r_[rng.normal(-4,.5,500),rng.normal(4,.5,500)])
        original=rr._score_pcs
        for override,expected in [(None,0.05),(0.07,0.07)]:
            seen=[]
            def score(*args,**kwargs):
                seen.append(kwargs["bic_gain_per_cell"])
                return original(*args,**kwargs)
            def node_pca(rnadata,**kwargs):
                ids=[c for a in rnadata.values() for c in a.obs_names]
                return pca.loc[ids],[]
            with patch.object(rr,"_node_pca",side_effect=node_pca),patch.object(rr,"_score_pcs",side_effect=score):
                tree=rr.RecursiveRNAPCASplit(data,enable_2d=False,dip_cutoff=0,
                    separation_cutoff=0,partition_cutoff=0,variance_cutoff=0,
                    small_fragment_max_cells=0,min_split_batches=1,
                    two_d_bic_gain_per_cell=0.05,one_d_bic_gain_per_cell=override,max_depth=2)
            self.assertTrue(tree.is_split)
            self.assertGreaterEqual(len(seen),3)
            self.assertTrue(all(x==expected for x in seen))

if __name__=="__main__": unittest.main()
