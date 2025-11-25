import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from scmidas.config import load_config
from scmidas.model import MIDAS
from scmidas.utils import load_predicted
import lightning as L
import torch

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

sc.set_figure_params(figsize=(4, 4))

path = '../data/result section 2/midas/all123_hvg500/'

data_config = [
    {'rna': path+'rnaHV.csv', 'adt': path+'adtHV.csv'},
    {'rna': path+'rnaHD-4.csv', 'adt': path+'adtHD-4.csv'},
    {'rna': path+'rnaP8.csv', 'adt': path+'adtP8.csv'},
]

dims_x = {
    'rna':[500],
    'adt': [72]
}

mask_config = [
    {'rna':path+'rna_mask.csv','adt':path+'adt_mask.csv'},
    {'rna':path+'rna_mask.csv','adt':path+'adt_mask.csv'},
    {'rna':path+'rna_mask.csv','adt':path+'adt_mask.csv'},
    # {'rna':path+'rna_mask_HV.csv','adt':path+'adt_mask_HV.csv'},
    # {'rna':path+'rna_mask_HD-4.csv','adt':path+'adt_mask_HD-4.csv'},
    # {'rna':path+'rna_mask_P8.csv','adt':path+'adt_mask_P8.csv'},
]

tree_label = pd.read_csv(path+'tree_label.csv',index_col=0)
tree_col = torch.tensor(tree_label.columns.astype(int), dtype=torch.int32)
tree_label = torch.tensor(tree_label.values, dtype=torch.float32)

# Configure MIDAS with the data
configs = load_config()
datasets, dims_s, s_joint, combs = MIDAS.configure_data_from_csv(data_config,mask_config,tree_label=tree_label,tree_cols=tree_col)
model = MIDAS.configure_data(configs, datasets, dims_x, dims_s, s_joint, combs)

trainer = L.Trainer(max_epochs=500)
trainer.fit(model=model)

model.predict(path,
        joint_latent=True,
        mod_latent=False,
        impute=False,
        batch_correct=True,
        translate=False,
        input=False)

# label = []
# batch_id = []
# for i in ['HD-4','P8']:
#     label.append(pd.read_csv(path+'/label_%s.csv'%i, index_col=0,header=None).values.flatten())
#     batch_id.append([i] * len(label[-1]))
# labels = np.concatenate(label)
# batch_ids = np.concatenate(batch_id)


# joint_embeddings = load_predicted(path, model.combs, joint_latent=True)

# adata_bio = sc.AnnData(joint_embeddings['z']['joint'][:, :model.dim_c])
# adata_tech = sc.AnnData(joint_embeddings['z']['joint'][:, model.dim_c:])

# adata_bio.obs['batch'] = batch_ids
# adata_bio.obs['label'] = labels
# adata_tech.obs['batch'] = batch_ids
# adata_tech.obs['label'] = labels

# output_path = '../output/midas/result section 2/'
# name = ['bio','tech']
# for i in range(2):
#     adata = [adata_bio, adata_tech][i]
#     sc.pp.neighbors(adata)
#     sc.tl.umap(adata)
#     # shuffle
#     sc.pp.subsample(adata, fraction=1)
#     sc.pl.umap(adata, color=['batch', 'label'], ncols=2, save=output_path+name[i])

