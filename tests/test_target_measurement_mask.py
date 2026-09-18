from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import anndata as ad
import numpy as np
import pandas as pd
from citepool_baseline.config import TargetPreparationConfig
from citepool_baseline.targets import prepare_translated_protein_targets


class TargetMeasurementMaskTests(unittest.TestCase):
    def prepare(self, root, masked, full_variable_mask=False):
        counts=np.array([[1,9,99,0],[1,9,99,0],[1,9,99,999],[1,9,99,999]],dtype=np.float32)
        a=ad.AnnData(counts,obs=pd.DataFrame({'batch':['a','a','b','b']},index=['c1','c2','c3','c4']),
                     var=pd.DataFrame({'feature_types':['Gene Expression']+['Antibody Capture']*3},index=['rna','A','B','C']))
        mask=np.array([[1,1,0],[1,1,0],[1,1,1],[1,1,1]],dtype=bool)
        if masked:a.obsm['protein_measurement_mask']=np.c_[np.ones(4,dtype=bool),mask] if full_variable_mask else mask
        inp=root/'input.h5ad';a.write_h5ad(inp)
        tables=root/'cytofuse/tables';tables.mkdir(parents=True)
        pd.DataFrame({'cell_barcode':a.obs_names,'batch':a.obs.batch,'assigned_taxonomy_node_id':['ALIGNED_1']*4}).to_csv(tables/'cell_taxonomy_assignments.csv',index=False)
        pd.DataFrame({'tree_node_id':['ALIGNED_1']*3,'marker':['A','B','C'],'state':['+']*3}).to_csv(tables/'taxonomy_states.csv',index=False)
        result=prepare_translated_protein_targets(TargetPreparationConfig(inp,tables.parent,None,root/'out',all_proteins=True))
        return ad.read_h5ad(result['target']),mask

    def test_missing_protein_does_not_affect_clr_translation_or_loss(self):
        for full in [False,True]:
            with TemporaryDirectory() as td:
                out,mask=self.prepare(Path(td),True,full)
                np.testing.assert_array_equal(out.layers['normalized_mask'],mask)
                unit=np.log(10)
                expected=np.array([[-.75,.25,0],[-.75,.25,0],[-.75,.25,1],[-.75,.25,1]])*unit
                np.testing.assert_allclose(out.X,expected,atol=1e-6)
                self.assertTrue(np.isfinite(out.X).all())

    def test_unmasked_targets_preserve_complete_panel_formula(self):
        with TemporaryDirectory() as td:
            out,_=self.prepare(Path(td),False)
            x=np.log1p(np.array([[9,99,0],[9,99,0],[9,99,999],[9,99,999]],dtype=np.float32))
            clr=x-x.mean(axis=1,keepdims=True)
            expected=np.repeat(np.median(np.stack([clr[0],clr[2]]),axis=0)[None,:],4,axis=0)
            np.testing.assert_array_equal(out.X,expected)
            self.assertTrue(out.layers['normalized_mask'].all())

if __name__=='__main__':unittest.main()
