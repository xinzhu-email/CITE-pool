import platform
import scanpy as sc
import anndata as ad
import pandas as pd

def get_dataset_path():
    if platform.system().lower() == 'linux':
        dspath = '/mnt/c/Users/ziqir/Desktop/iCloudDrive/scData'
    elif platform.system().lower() == 'darwin':
        dspath = '/Users/ziqi/Library/Mobile Documents/com~apple~CloudDocs/scData'
    else:
        raise Exception('System unknown. Path not determined.')
    return dspath

def load_zheng_data(names):
    dspath = get_dataset_path()
    def fetch_data(name):
        adata = sc.read_10x_mtx(dspath+'/Zheng_filtered/'+name+'/hg19/')
        adata.obs['cell_type'] = name
        return adata
    adata = ad.concat([fetch_data(name) for name in names], merge='same')
    adata.X = adata.X.toarray()
    return adata

def load_kong_data(names):
    dspath = get_dataset_path()
    def fetch_pbmc_data(name):
        adata = sc.read_h5ad(dspath+'/pbmc_purified/'+name+'.h5ad')
        return adata
    adata = ad.concat([fetch_pbmc_data(name) for name in names], merge='same')
    gating_label = pd.read_csv(dspath+'/pbmc_purified/pbmc_RNA_purified_label.csv', index_col=0)
    adata.obs['cell_type'] = gating_label.loc[adata.obs.index]['gating']
    adata.X = adata.X.toarray()
    return adata

def load_simulation_data(name):
    dspath = get_dataset_path()
    adata = sc.read_h5ad(dspath+'/simdata/'+name+'/data.h5ad')
    return adata

def sample_data(adata, conditions):
    index = []
    for cell_type, size in conditions:
        samples = np.random.choice(
                    np.where(adata.obs['cell_type'] == cell_type)[0], 
                    size, 
                    replace=False
                )
        index.append(samples)
    index = np.concatenate(index)
    return adata[index, ]