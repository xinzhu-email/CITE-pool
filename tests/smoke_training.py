"""Small CPU integration run exercising real protein/RNA/refinement stages.

Use an explicit scratch output location; this is not a benchmark experiment.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from citepool.model import CITEPool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    rng = np.random.default_rng(2026)
    n, ng = 600, 80
    identity = np.tile(np.repeat([0, 1], n // 4), 2)
    batch = np.repeat(['b1', 'b2'], n // 2)
    rates = np.full((n, ng), 3.)
    rates[identity == 0, :20] = 15
    rates[identity == 1, 20:40] = 15
    rna = rng.poisson(rates).astype(np.float32)
    protein_rates = np.full((n, 6), 2.)
    protein_rates[identity == 0, :3] = 100
    protein_rates[identity == 1, 3:] = 100
    proteins = rng.poisson(protein_rates).astype(np.float32)
    data = ad.AnnData(sparse.csr_matrix(rna),
        obs=pd.DataFrame({'batch': batch}, index=[f'c{i}' for i in range(n)]),
        var=pd.DataFrame(index=[f'g{i}' for i in range(ng)]))
    data.obsm['protein_counts'] = pd.DataFrame(proteins, index=data.obs_names,
                                              columns=['CD3', 'CD4', 'CD27', 'CD19', 'CD20', 'CD38'])
    CITEPool.setup_anndata(data, protein_expression_obsm_key='protein_counts')
    model = CITEPool(data, initial_resolution=.5)
    model.train(args.output_dir, device='cpu', scvi_epochs=1, scanvi_epochs=1,
        training_options={'n_hvg': 60, 'n_latent': 8, 'batch_size': 128,
                          'compute_umap': False, 'early_stopping': False})
    assert model.is_trained
    assert model.get_latent_representation().shape == (n, 8)
    assert model.get_reconstructed_protein().shape == (n, 6)
    assert model.predict().index.equals(data.obs_names)
    np.testing.assert_allclose(model.predict(soft=True).sum(axis=1), 1, atol=1e-6)
    assert len(model.get_gene_weights()) > 0
    restored = CITEPool.load(args.output_dir)
    np.testing.assert_allclose(restored.get_latent_representation(), model.get_latent_representation())
    print('PASS: real three-stage training without ground-truth labels; outputs and reload verified')


if __name__ == '__main__': main()
