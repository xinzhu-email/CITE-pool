import numpy as np
import pandas as pd
import networkx as nx
import scanpy as sc
from celltree import CellTree


class SimulationCellTree(CellTree):

    def _sample_dirichlet_multinomial(
        self,
        alpha,
        size
    ) -> np.ndarray:
        p_vals = np.random.dirichlet(alpha=alpha, size=size)
        results = []
        for p in p_vals:
            umi = np.random.lognormal(np.log(2000), 0.1)
            results.append(np.random.multinomial(n=umi, pvals=p))
        return np.array(results)

    def _up_regulated_alpha(
        self,
        alpha
    ) -> np.ndarray:
        n_genes = len(alpha)
        # alpha_up = alpha * (1 + abs(np.random.normal(loc=0, scale=1, size=n_genes)))
        alpha_up = alpha + \
            np.exp(np.random.normal(loc=np.log(5), scale=1, size=n_genes))
        return alpha_up

    def _generate_gene_alpha(
        self,
        n_genes_per_metagene,
        n_housekeeping_genes
    ) -> pd.DataFrame:
        """
        Expand the metagenes to multiple genes, with metagene affiliation information stored.
        Generate the alpha in Dirichlet distribution for each gene, as well as the up regulated alpha.

        Args:
            n_genes_per_metagene (_type_): _description_
            n_housekeeping_genes (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """
        n_genes = self.n_metagenes * n_genes_per_metagene + n_housekeeping_genes
        alpha = np.exp(np.random.normal(loc=np.log(1), scale=1, size=n_genes))
        alpha_up = self._up_regulated_alpha(alpha)
        alpha_df = pd.DataFrame(np.array([alpha, alpha_up]).T, columns=[
                                'base_alpha', 'up_regulated_alpha'])
        metagene_info = []
        for i in range(self.n_metagenes):
            metagene_info += [i] * n_genes_per_metagene
        metagene_info += [-1] * n_housekeeping_genes
        alpha_df['metagene'] = pd.Categorical(metagene_info)
        alpha_df.index = [f"Gene_{i:d}" for i in range(len(alpha_df))]
        return alpha_df

    def generate_metagene_regulation(
        self,
    ) -> None:
        """
        Generate metagene regulation information for each node in the cell tree.
        Saved in 'metagene_reg' attribute in each node, where 1 means no regulation, 2 means up regulated.

        Returns:
            None
        """
        self.n_metagenes = self.tree.number_of_edges()
        nx.set_node_attributes(self.tree, {0: np.ones(
            self.n_metagenes)}, name='metagene_reg')
        bfs_edges = nx.bfs_edges(self.tree, 0)
        DE_metagene_dict = {}
        i_metagene = 0
        for edge in bfs_edges:
            DE_metagene_dict[edge] = i_metagene
            metagene_reg = self.tree.nodes[edge[0]]['metagene_reg'].copy()
            metagene_reg[i_metagene] += 1
            nx.set_node_attributes(
                self.tree, {edge[1]: metagene_reg}, name='metagene_reg')
            i_metagene += 1
        nx.set_edge_attributes(self.tree, DE_metagene_dict, 'metagene_reg')
        return None

    def simulate_gene_expr(
        self,
        base_alpha=None,
        n_genes_per_meta=100,
        n_housekeeping_genes=500
    ) -> sc.AnnData:
        if base_alpha == None:
            alpha_df = self._generate_gene_alpha(
                n_genes_per_meta, n_housekeeping_genes)
        else:
            pass
        leaves = [node for node in self.tree.nodes() if self.tree.in_degree(
            node) != 0 and self.tree.out_degree(node) == 0]
        X = []
        cell_types = []
        for node in leaves:
            n_cells = self.tree.nodes[node]['n_cells']
            metagene_reg = self.tree.nodes[node]['metagene_reg']
            gene_reg = alpha_df['metagene'].isin(
                np.nonzero(metagene_reg > 1)[0])
            alpha = np.where(
                gene_reg == True, alpha_df['base_alpha'], alpha_df['up_regulated_alpha'])
            X.append(self._sample_dirichlet_multinomial(
                alpha=alpha, size=n_cells))
            cell_types += [node] * n_cells
        X = np.concatenate(X, axis=0)
        adata = sc.AnnData(X, dtype="int64")
        adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
        adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]
        adata.obs['cell_type'] = pd.Categorical(cell_types)
        adata.var['base_alpha'] = alpha_df['base_alpha']
        adata.var['up_regulated_alpha'] = alpha_df['up_regulated_alpha']
        adata.var['metagene'] = alpha_df['metagene']
        return adata
