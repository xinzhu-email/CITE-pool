#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 23:49:04 2019

@author: lianqiuyu
"""

import pandas as pd
from CITEsort_rna.Matryoshka import Matryoshka
from CITEsort_rna.Visualize import visualize_tree
from CITEsort_rna.BTreeTraversal import BTreeTraversal
from CITEsort_rna.ReSplit import ReSplit, Choose_leaf
import pickle
import argparse
import os
import time
import matplotlib.pyplot as plt
import warnings
import scanpy as sc
from metadimm import MetaDIMM
warnings.filterwarnings("ignore")

#from sys import argv

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help = "The input path of CLR normalized data in .csv files with row as sample, col as feature.")
    parser.add_argument('-c','--cutoff',type = float, default=0.1, help = "The cutoff for merging components (default 0.1). It shoube a value between 0 and 1. The bigger value leads to split more aggressively, and ends in a more complicated tree.")
    parser.add_argument('-adt_output',default=None, help="The path of ADT clustering result, if none, consider no groups")
    parser.add_argument('-o', '--output', type=str, default='./CITEsort_out',help='Path to save output files.')
    parser.add_argument('--compact', action='store_true', default=False, help='Output a compact tree.')
    args = parser.parse_args()
    
    data_path = args.data_path
    output_path = args.output
    merge_cutoff = args.cutoff
    compact_flag = args.compact
    adt_output = args.adt_output
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    starttime = time.time()
    print('read data and run CITE-sort.')

    # data = pd.read_csv(data_path,header = 0, index_col=0)
    try:
        adata = sc.read_h5ad(data_path)
    except:
        adata = pd.read_csv(data_path,sep=',',index_col=0).T
        adata = sc.Anndata(adata)
        print(adata.shape, adata.obs_names[0], adata.var_names[0])
    if adt_output != None:
        f = open(adt_output+'tree.pickle','rb')
        tree = pickle.load(f)
        f.close()
        adata_sub = adata[tree.indices,:]
        md = MetaDIMM()
        # adata_sub = md.filter(adata_sub)
        adata_sub = md.preprocess(adata_sub, normalize=True, log1p=True, hvg=False, scale=False)
        tree = dfs(tree, adata_sub, merge_cutoff)
    else:
        # ct_list = ['CD4 Naive','CD8 Naive','CD14 Mono']
        # adata = adata[adata.obs['label_l2'].isin(ct_list),:]
        # adata = adata[[i for i in adata.obs_names 
        # if adata.obs.loc[i,'label_l2'] in ct_list and adata.obs.loc[i,'donor']=='P2' and adata.obs.loc[i,'time']==0],:]
        md = MetaDIMM()
        # adata = md.filter(adata)
        adata = md.preprocess(adata, normalize=True, log1p=True, hvg=False, scale=False)
        # sc.pp.scale(adata, max_value=10)
        # sc.tl.pca(adata,n_comps=10)
        tree, bic_list, min_bic_node = Choose_leaf(data=adata,merge_cutoff=merge_cutoff,use_parent=True)
        # adata.write_h5ad(output_path+'/adata_pp.h5ad')        
        # tree = ReSplit(adata,data_raw=adata_raw)
        # tree = ReSplit(data,merge_cutoff)
        #tree = Matryoshka(data,merge_cutoff)
        # print('done.\nplot tree.')
    visualize_tree(tree,adata,output_path,'tree',compact=compact_flag)
    
    f = open(output_path+'/tree.pickle','wb')
    pickle.dump(tree,f)
    f.close()
    
    # print('generate labels.')
    
    traversal = BTreeTraversal(tree,save_min_BIC=False)
    leaves_labels = traversal.get_leaf_label()
    leaves_labels.to_csv(output_path + '/leaf_labels.csv')
    
    # leaves_labels = traversal.get_leaf_label(BIC_node=True)
    # leaves_labels.to_csv(output_path + '/BIC_stop_labels.csv',index=False)

    endtime = time.time()

    print('Time using: ', round(endtime-starttime, 3),'secs')
    # plt.plot(list(range(len(bic_list))), bic_list)
    # plt.savefig('BIC_as_split.png')

def dfs(node, adata, merge_cutoff):
    if node.key == ('leaf',):
        adata_sub = adata[list(set(node.indices)&set(adata.obs_names)),:]
        print(adata_sub.shape)
        # print(len(node.indices),adata_sub.X.shape)
        # sc.pp.scale(adata_sub, max_value=10)
        # sc.tl.pca(adata_sub, n_comps=10)
        # node.stop = None
        node, bic_list, min_bic_node = Choose_leaf(data=adata_sub,merge_cutoff=merge_cutoff,use_parent=True) 
        return node       
    else:
        node.left = dfs(node.left, adata, merge_cutoff)
        node.right = dfs(node.right, adata, merge_cutoff)
        return node

if __name__ == "__main__":
    main()
    
    