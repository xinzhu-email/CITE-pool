"""Shared preprocessing helpers used by the stable benchmark."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def dense(matrix):
    """Return a dense NumPy view/copy for dense or sparse input."""

    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def normalize_log1p(
    x: sparse.spmatrix, target_sum: float = 1e4
) -> sparse.csr_matrix:
    """Library-normalize a count matrix and apply log1p."""

    x = sparse.csr_matrix(x, dtype=np.float32).copy()
    totals = np.asarray(x.sum(axis=1)).ravel()
    x = sparse.diags((target_sum / np.maximum(totals, 1.0)).astype(np.float32)) @ x
    x.data = np.log1p(x.data)
    return x.tocsr().astype(np.float32)


def prepare_unintegrated_pca(
    rna_h5ad: Path, hvg_csv: Path, cell_names: pd.Index, seed: int = 2026
) -> np.ndarray:
    """Build the fixed pre-integration RNA PCA required by scIB."""

    rna = ad.read_h5ad(rna_h5ad)
    if not pd.Index(rna.obs_names.astype(str)).equals(cell_names):
        raise ValueError("RNA and embedding cell orders differ")
    feature_type = rna.var["feature_types"].astype(str).to_numpy()
    gene_columns = np.flatnonzero(feature_type == "Gene Expression")
    gene_names = rna.var_names.astype(str).to_numpy()[gene_columns]
    if pd.Index(gene_names).duplicated().any():
        raise ValueError("Gene Expression feature names are not unique")
    hvg_names = pd.read_csv(hvg_csv)["gene"].astype(str)
    selected = pd.Index(gene_names).get_indexer(hvg_names)
    if np.any(selected < 0):
        missing = hvg_names[selected < 0].tolist()[:10]
        raise KeyError(f"HVGs missing from RNA input: {missing}")
    normalized = normalize_log1p(rna.X[:, gene_columns[selected]])
    pre = ad.AnnData(X=normalized)
    sc.pp.pca(pre, n_comps=min(50, normalized.shape[1] - 1), random_state=seed)
    return np.asarray(pre.obsm["X_pca"], dtype=np.float32)
