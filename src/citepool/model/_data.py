"""AnnData registration and conversion to the internal count-data contract."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

REGISTRY_KEY = '_citepool_setup'


def _validate_counts(matrix, name):
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f'{name} must contain finite, nonnegative raw counts')
    if not np.allclose(values, np.rint(values), atol=1e-5, rtol=0):
        raise ValueError(f'{name} must contain raw counts, not normalized expression')


def register_anndata(adata, *, batch_key='batch', layer=None,
                     protein_expression_obsm_key=None, protein_names_uns_key=None,
                     protein_measurement_mask_obsm_key=None,
                     feature_type_key='feature_types'):
    """Validate raw counts and store a serializable input registry in ``uns``."""
    if adata.isbacked:
        raise ValueError('Load AnnData into memory before setup_anndata')
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError('Cell and feature names must be unique')
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError('AnnData must contain cells and RNA features')
    if batch_key not in adata.obs or adata.obs[batch_key].isna().any():
        raise ValueError(f'obs[{batch_key!r}] must contain a batch for every cell')
    counts = adata.X if layer is None else adata.layers[layer]
    _validate_counts(counts, 'RNA/feature matrix')
    if protein_expression_obsm_key is None:
        if protein_names_uns_key is not None:
            raise ValueError('protein_names_uns_key requires a protein obsm matrix')
        if feature_type_key not in adata.var:
            raise ValueError('Provide a protein obsm matrix or feature-type annotations')
        types = adata.var[feature_type_key].astype(str)
        if not types.eq('Gene Expression').any() or not types.eq('Antibody Capture').any():
            raise ValueError('Combined input needs Gene Expression and Antibody Capture features')
        proteins = counts[:, types.eq('Antibody Capture').to_numpy()]
        names = adata.var.loc[types.eq('Antibody Capture'), 'feature_name'].astype(str).tolist() if 'feature_name' in adata.var else adata.var_names[types.eq('Antibody Capture')].tolist()
        allowed_mask_shapes = {proteins.shape, adata.shape}
    else:
        proteins = adata.obsm[protein_expression_obsm_key]
        if isinstance(proteins, pd.DataFrame):
            names = proteins.columns.astype(str).tolist()
        elif protein_names_uns_key is not None:
            names = list(map(str, adata.uns[protein_names_uns_key]))
        else:
            raise ValueError('Array protein obsm input requires protein_names_uns_key')
        allowed_mask_shapes = {proteins.shape}
    if len(proteins.shape) != 2 or proteins.shape[0] != adata.n_obs or proteins.shape[1] == 0:
        raise ValueError('Protein matrix must have shape cells × proteins')
    if len(names) != proteins.shape[1] or len(set(names)) != len(names):
        raise ValueError('Protein names must be unique and match protein columns')
    _validate_counts(proteins, 'Protein matrix')
    if protein_measurement_mask_obsm_key is None and 'protein_measurement_mask' in adata.obsm:
        protein_measurement_mask_obsm_key = 'protein_measurement_mask'
    if protein_measurement_mask_obsm_key is not None:
        mask = adata.obsm[protein_measurement_mask_obsm_key]
        if mask.shape not in allowed_mask_shapes:
            raise ValueError('Protein measurement mask shape must match the protein matrix')
        values = mask.data if sparse.issparse(mask) else np.asarray(mask)
        if not np.isin(values, [0, 1]).all():
            raise ValueError('Measurement mask must contain only boolean/0/1 values')
        if protein_expression_obsm_key is None and mask.shape == adata.shape:
            mask = np.asarray(mask)[:, types.eq('Antibody Capture').to_numpy()] if isinstance(mask, pd.DataFrame) else mask[:, types.eq('Antibody Capture').to_numpy()]
        if (np.asarray(mask.sum(axis=1)).ravel() == 0).any():
            raise ValueError('Every cell must have at least one measured protein')
    # Empty strings are H5AD-safe representations of optional keys.
    adata.uns[REGISTRY_KEY] = dict(batch_key=batch_key, layer=layer or '',
        protein_expression_obsm_key=protein_expression_obsm_key or '',
        protein_names_uns_key=protein_names_uns_key or '',
        protein_measurement_mask_obsm_key=protein_measurement_mask_obsm_key or '',
        feature_type_key=feature_type_key)


def prepare_input(adata):
    """Create a separate count AnnData; never rewrite the user's expression."""
    from anndata import AnnData
    registry = adata.uns.get(REGISTRY_KEY)
    if registry is None:
        raise ValueError('Call CITEPool.setup_anndata(adata) before constructing the model')
    # Revalidate: setup may have been followed by edits to expression or keys.
    register_anndata(adata, **{k: (v or None) if k != 'batch_key' and k != 'feature_type_key' else v for k, v in registry.items()})
    counts = adata.X if not registry['layer'] else adata.layers[registry['layer']]
    obs = adata.obs.copy()
    obs['batch'] = obs[registry['batch_key']].astype(str)
    key = registry['protein_expression_obsm_key']
    if not key:
        var = adata.var.copy()
        var['feature_types'] = var[registry['feature_type_key']].astype(str)
        result = AnnData(X=counts.copy(), obs=obs, var=var)
    else:
        proteins = adata.obsm[key]
        names = proteins.columns.astype(str).tolist() if isinstance(proteins, pd.DataFrame) else list(map(str, adata.uns[registry['protein_names_uns_key']]))
        protein_values = proteins.to_numpy() if isinstance(proteins, pd.DataFrame) else proteins
        var = adata.var.copy()
        var['feature_types'] = 'Gene Expression'
        var['feature_name'] = adata.var_names.astype(str)
        protein_var = pd.DataFrame({'feature_types': 'Antibody Capture',
                                   'feature_name': names}, index=names)
        combined_var = pd.concat([var, protein_var])
        matrix = sparse.hstack([sparse.csr_matrix(counts), sparse.csr_matrix(protein_values)], format='csr')
        result = AnnData(X=matrix, obs=obs, var=combined_var)
        if not result.var_names.is_unique:
            # Preserve feature_name; only internal matrix IDs gain a suffix.
            result.var_names_make_unique()
    mask_key = registry['protein_measurement_mask_obsm_key']
    # Preserve the existing internal measurement-mask convention too.
    if not mask_key and 'protein_measurement_mask' in adata.obsm:
        mask_key = 'protein_measurement_mask'
    if mask_key:
        mask = adata.obsm[mask_key]
        if not key and mask.shape == adata.shape:
            mask = np.asarray(mask)[:, result.var.feature_types.eq('Antibody Capture').to_numpy()] if isinstance(mask, pd.DataFrame) else mask[:, result.var.feature_types.eq('Antibody Capture').to_numpy()]
        result.obsm['protein_measurement_mask'] = mask.copy()
    result.uns[REGISTRY_KEY] = dict(registry)
    return result
