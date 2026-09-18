"""Model lifecycle and scientific data-contract tests, without expensive fitting."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from citepool_baseline.model import CITEPool
from citepool_baseline.model._data import prepare_input
from citepool_baseline import api


def input_data():
    data = ad.AnnData(sparse.csr_matrix([[1, 2], [3, 4]]),
        obs=pd.DataFrame({'donor': ['b1', 'b2']}, index=['c1', 'c2']),
        var=pd.DataFrame(index=['g1', 'g2']))
    data.obsm['adt'] = pd.DataFrame([[3, 0], [0, 5]], index=data.obs_names,
                                  columns=['CD3', 'CD19'])
    data.obsm['measured'] = np.array([[True, False], [False, True]])
    return data


def register(data):
    CITEPool.setup_anndata(data, batch_key='donor', protein_expression_obsm_key='adt',
                          protein_measurement_mask_obsm_key='measured')


def write_run(root, **kwargs):
    root.mkdir(exist_ok=True)
    model = root / 'citepool_refined'
    model.mkdir()
    obs = pd.DataFrame({'predicted_leaf': ['L1', 'L2']}, index=['c1', 'c2'])
    data = ad.AnnData(np.ones((2, 1)), obs=obs)
    data.obsm['X_scanvi'] = np.array([[1, 2], [3, 4]])
    data.obsm['leaf_probability_parent_constrained'] = np.array([[.4, 0], [.1, .9]])
    data.uns['leaf_names'] = ['L1', 'L2']
    data.write_h5ad(model / 'official_scanvi_parent_set.h5ad')
    protein = ad.AnnData(np.array([[.5, .2], [.3, .9]]), obs=obs,
                        var=pd.DataFrame(index=['CD3', 'CD19']))
    protein.write_h5ad(model / 'reconstructed_protein.h5ad')
    pd.DataFrame({'cell': ['c1', 'c2'], 'label': ['L1', 'L2']}).to_csv(model / 'final_leaf_assignments.csv', index=False)
    pd.DataFrame({'metric': ['ARI'], 'value': [.8]}).to_csv(model / 'metrics.csv', index=False)
    report = model / 'rna_gene_classifier'; report.mkdir()
    pd.DataFrame({'gene': ['g1'], 'weight': [1.]}).to_csv(report / 'rna_gene_classifier_all_weights.csv', index=False)
    (root / 'workflow_summary.json').write_text(json.dumps({'final_model_dir': str(model)}))
    (root / 'workflow_config.json').write_text(json.dumps(kwargs))
    return api.load_run(root)


class ModelApiTests(unittest.TestCase):
    def test_obsm_conversion_preserves_counts_masks_and_user_input(self):
        data = input_data(); register(data)
        converted = prepare_input(data)
        self.assertEqual(converted.shape, (2, 4))
        np.testing.assert_array_equal(converted.X.toarray(), [[1, 2, 3, 0], [3, 4, 0, 5]])
        np.testing.assert_array_equal(converted.obsm['protein_measurement_mask'], data.obsm['measured'])
        self.assertEqual(converted.obs.batch.tolist(), ['b1', 'b2'])
        self.assertEqual(data.shape, (2, 2))
        self.assertNotIn('batch', data.obs)

    def test_setup_rejects_normalized_counts_and_empty_measurement_rows(self):
        data = input_data(); data.X = data.X.astype(float) / 2
        with self.assertRaisesRegex(ValueError, 'raw counts'): register(data)
        data = input_data(); data.obsm['measured'][0] = False
        with self.assertRaisesRegex(ValueError, 'at least one measured'): register(data)

    def test_duplicate_protein_names_and_bad_mask_values_rejected(self):
        data = input_data(); data.obsm['adt'].columns = ['CD3', 'CD3']
        with self.assertRaisesRegex(ValueError, 'unique'): register(data)
        data = input_data(); data.obsm['measured'] = np.array([[2, 0], [0, 1]])
        with self.assertRaisesRegex(ValueError, 'boolean'): register(data)

    def test_array_proteins_require_names(self):
        data = input_data(); data.obsm['adt'] = data.obsm['adt'].to_numpy()
        with self.assertRaisesRegex(ValueError, 'protein_names'): register(data)
        data.uns['markers'] = ['CD3', 'CD19']
        CITEPool.setup_anndata(data, batch_key='donor', protein_expression_obsm_key='adt', protein_names_uns_key='markers')
        self.assertEqual(prepare_input(data).var.feature_name.iloc[-2:].tolist(), ['CD3', 'CD19'])

    def test_training_dispatch_and_portable_model_lifecycle(self):
        data = input_data(); register(data)
        model = CITEPool(data, initial_resolution=1.5)
        with self.assertRaisesRegex(RuntimeError, 'Train'): model.predict()
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            seen = {}
            def fake_fit(path, output, **kwargs):
                seen.update(kwargs)
                self.assertEqual(ad.read_h5ad(path).shape, (2, 4))
                return write_run(output, input_h5ad=str(path))
            with patch('citepool_baseline.model._citepool.fit', side_effect=fake_fit):
                model.train(root, device='cpu', scvi_epochs=1, scanvi_epochs=1)
            self.assertTrue(seen['all_proteins']); self.assertTrue(seen['enable_rna_refinement'])
            self.assertEqual(seen['initial_resolution'], 1.5)
            self.assertEqual(seen['batch_key'], 'batch')
            self.assertTrue(model.is_trained)
            self.assertEqual(model.predict().tolist(), ['L1', 'L2'])
            np.testing.assert_allclose(model.predict(soft=True).sum(axis=1), 1)
            np.testing.assert_array_equal(model.get_latent_representation(), [[1, 2], [3, 4]])
            self.assertEqual(model.get_reconstructed_protein().columns.tolist(), ['CD3', 'CD19'])
            self.assertEqual(model.get_gene_weights().gene.tolist(), ['g1'])
            saved = model.save(Path(tmp) / 'saved')
            # The original still exists: load must resolve saved files locally.
            loaded = CITEPool.load(saved)
            self.assertEqual(loaded.run_.final_model_dir, saved / 'citepool_refined')
            self.assertEqual(loaded.predict().tolist(), ['L1', 'L2'])
            self.assertTrue((saved / 'input.h5ad').is_file())
            self.assertNotIn('citepool-input-', (saved / 'workflow_config.json').read_text())
            with self.assertRaises(FileExistsError): model.save(saved)

    def test_legacy_aliases_share_implementation(self):
        from citepool_baseline import training, rna_refinement, targets
        from citepool_baseline.representation import _training
        from citepool_baseline.refinement import _rna_refinement
        from citepool_baseline.identity import _targets
        from citepool_baseline.cytofuse import runner
        from citepool_baseline.identity._engine import runner as engine
        self.assertIs(training, _training); self.assertIs(rna_refinement, _rna_refinement)
        self.assertIs(targets, _targets); self.assertIs(runner, engine)


if __name__ == '__main__': unittest.main()
