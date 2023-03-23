import numpy as np
import pandas as pd
import scanpy as sc
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from pyDIMM import DirichletMultinomialMixture
from utils import Utils


class MetaDIMM(object):
    """
    _summary_
    """

    def __init__(
        self,
    ) -> None:
        sc.settings.verbosity = 0
        sc.settings.set_figure_params(dpi=80, facecolor='white')
        return None

    @Utils.timer
    def filter(
        self,
        adata
    ) -> sc.AnnData:
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True
        )
        adata = adata[adata.obs.n_genes_by_counts < 2500, :]
        adata = adata[adata.obs.pct_counts_mt < 10, :].copy()
        return adata

    @Utils.timer
    def preprocess(
        self,
        adata,
        normalize=False,
        log1p=False,
        round=False,
        hvg=False,
        scale=False
    ) -> sc.AnnData:
        if normalize:
            sc.pp.normalize_total(adata, target_sum=1e4)
        if log1p:
            sc.pp.log1p(adata)
        if round:
            adata.X = np.round(adata.X)
        if hvg:
            # sc.pp.highly_variable_genes(
            #     adata, min_mean=0.0125, max_mean=3, min_disp=0.5
            # )
            sc.pp.highly_variable_genes(adata)
        if scale:
            sc.pp.scale(adata, max_value=10)
        # if len(np.unique(adata.obs['cell_type'])) > 1:
        #     sc.tl.rank_genes_groups(adata, 'cell_type', method='wilcoxon')
        return adata

    @Utils.timer
    def pca(
        self,
        adata,
    ) -> sc.AnnData:
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        return adata
    
    @Utils.timer
    def select_n_meta_pcs(
        self,
        adata
    ) -> int:
        i=0
        flag = False
        n_pcs = adata.obsm['X_pca'].shape[1]
        while not flag:
            x = adata.obsm['X_pca'][:,i].reshape(-1,1)
            gmm1 = sklearn.mixture.GaussianMixture(n_components=1)
            gmm1.fit(x)
            gmm2 = sklearn.mixture.GaussianMixture(n_components=2)
            gmm2.fit(x)
            if ((gmm1.bic(x) < gmm2.bic(x)) or (i+1 >= n_pcs)):
                flag = True
            i = i+1
        n_meta_pcs = min(i+1, n_pcs)
        return n_meta_pcs
    
    @Utils.timer
    def get_metagenes(
        self,
        adata,
        n_meta_pcs = None,
        loading_threshold = 0.03
    ) -> list:
        if n_meta_pcs == None:
            n_meta_pcs = self.select_n_meta_pcs(adata)
        pcs = adata.varm['PCs'].T
        n_tail = (pcs < -loading_threshold).sum(axis=1) + 1
        n_head = (pcs > loading_threshold).sum(axis=1) + 1
        pc_rank = pcs.argsort(axis=1)
        metagenes = []
        for i in range(n_meta_pcs):
            metagenes.append(pc_rank[i][:n_tail[i]])
            metagenes.append(pc_rank[i][-n_head[i]:])
        return metagenes

    @Utils.timer
    def merge(
        self,
        adata_raw,
        metagenes
    ) -> sc.AnnData:
        X = np.zeros((adata_raw.shape[0], len(metagenes)))
        for i, metagene in enumerate(metagenes):
            X[:, i] = adata_raw.X[:,metagene].sum(axis=1)
        adata_meta = sc.AnnData(X, obs=adata_raw.obs)
        return adata_meta

    @Utils.timer
    def dimm_cluster(
        self,
        adata,
        n_components,
        max_iter = 300,
        tol=1e-3,
        verbose=False
    ) -> pd.Categorical:
        print('Fitting data shape:', adata.shape)
        X = adata.X
        dimm = DirichletMultinomialMixture(n_components=n_components, tol=tol, max_iter=max_iter, verbose=2*verbose).fit(X)
        cluster_label = pd.Categorical(dimm.predict(X))
        posterior_prob = np.exp(dimm._estimate_log_prob_resp(X)[1].max(axis=1))
        return cluster_label, posterior_prob
    
    def score(
        self,
        adata,
        label='cell_type'
    ) -> None:
        RI = sklearn.metrics.rand_score(
                adata.obs['cell_type'], adata.obs[label])
        ARI = sklearn.metrics.adjusted_rand_score(
                adata.obs['cell_type'], adata.obs[label])
        print('Rand Index =', RI)
        print('Adjusted Rand Index =', ARI)
        return RI, ARI
    
    def plot_pc(
        self,
        adata,
        x_pc=0,
        y_pc=1,
        label='cell_type'
    ) -> None:
        factors = pd.DataFrame(adata.obsm['X_pca'])
        factors[label] = pd.Categorical(adata.obs[label])
        plt.figure(figsize=(6,6), dpi=64)
        sns.scatterplot(data=factors, x=x_pc, y=y_pc, hue=label).set(
            title='PC_plot', xlabel='PC_'+str(x_pc), ylabel='PC_'+str(y_pc))
        plt.show()
        return None


if __name__ == "__main__":
    pass
