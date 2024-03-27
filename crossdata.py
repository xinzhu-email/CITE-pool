import pandas as pd
from Classifier.BTree import BTree
from Classifier.Visualize import visualize_tree, visualize_modeltree
from Classifier.BTreeTraversal import BTreeTraversal
from Classifier.adtReSplit import ReSplit, Choose_leaf, CrossSplit, CrossNode, retrain, gm_proba
import pickle
import argparse
import os
import time
import matplotlib.pyplot as plt
import warnings
import scanpy as sc
warnings.filterwarnings("ignore")
import numpy as np

def olddfs(node, adata, merge_cutoff, prior_gene):
    if node.key == ('leaf',):
        adata_sub = adata[list(set(node.indices)&set(adata.obs_names)),:]
        print(adata_sub.shape)
        # print(len(node.indices),adata_sub.X.shape)
        # sc.pp.scale(adata_sub, max_value=10)
        # sc.tl.pca(adata_sub, n_comps=10)
        # node.stop = None
        node, bic_list, min_bic_node = Choose_leaf(data=adata_sub,prior_gene=prior_gene,merge_cutoff=merge_cutoff,use_parent=True) 
        return node       
    else:
        node.left = dfs(node.left, adata, merge_cutoff, prior_gene)
        node.right = dfs(node.right, adata, merge_cutoff, prior_gene)
        return node

def dfs(crossnode, adtdata, rnadata, merge_cutoff, useADT=True, ifretrain=False, fitgaussian=False, key=None):
    # if crossnode.modelnode.key == ('CD45RO','CD45RA',):
    #     if len(crossnode.modelnode.indices) != 8:
    #         crossnode.modelnode.key = ('leaf',)
    # if crossnode.modelnode == None:
    #     return crossnode
    
        

    if crossnode.modelnode.key == ('leaf',):#11174 and  crossnode.modelnode.val_cnt == 13665 
        # print(crossnode.modelnode.val_cnt)
        if ifretrain or fitgaussian:
            return crossnode
        if useADT:
            crossnode = CrossSplit(adtdata.copy(),merge_cutoff, weight=np.ones(len(list(rnadata.keys()))),rnadata=rnadata.copy())
        else:
            crossnode = CrossSplit({},merge_cutoff, weight=np.ones(len(list(rnadata.keys()))),rnadata=rnadata.copy())
        # adata_sub = adata[list(set(node.indices)&set(adata.obs_names)),:]
        # print(adata_sub.shape)
        # # print(len(node.indices),adata_sub.X.shape)
        # # sc.pp.scale(adata_sub, max_value=10)
        # # sc.tl.pca(adata_sub, n_comps=10)
        # # node.stop = None
        # node, bic_list, min_bic_node = Choose_leaf(data=adata_sub,prior_gene=prior_gene,merge_cutoff=merge_cutoff,use_parent=True) 
        return crossnode       
    elif crossnode.modelnode.key == ('leaf',):
        return crossnode
    else:
        if ifretrain:
            # if crossnode.modelnode.val_cnt == 11174 :#crossnode.modelnode.key == ('CC_1',)
            #     print(crossnode.modelnode.key)
            crossnode.modelnode.artificial_w = retrain(
                    crossnode.nodelist, rnadata.copy(), genes=crossnode.modelnode.artificial_w.index)
        nodelist = crossnode.nodelist
        lnodelist, rnodelist, ladt, radt, lrna, rrna = [], [], {}, {}, {}, {}
        for i in range(len(nodelist)):
            node = nodelist[i]
            
            if node != None:
                
                if node.key != ('leaf',):
                    if fitgaussian :
                        node = gm_proba(node, adtdata[i])
                    if useADT:
                        if len(adtdata[i]) > 0:
                            ladt[i], radt[i] = adtdata[i].loc[node.left_indices,:], adtdata[i].loc[node.right_indices,:]
                        else:
                            ladt[i], radt[i] = [],[]
                    lrna[i], rrna[i] = rnadata[i][node.left_indices,:], rnadata[i][node.right_indices,:]
                else:
                    ladt[i], radt[i], lrna[i], rrna[i] = [],[],[],[]
                lnodelist.append(node.left)
                rnodelist.append(node.right)
        lcrossnode = CrossNode(lnodelist, modelnode=crossnode.modelnode.left)
        rcrossnode = CrossNode(rnodelist, modelnode=crossnode.modelnode.right)
        crossnode.left = dfs(lcrossnode, ladt, lrna, merge_cutoff, useADT, ifretrain, fitgaussian)
        crossnode.right = dfs(rcrossnode, radt, rrna, merge_cutoff, useADT, ifretrain, fitgaussian)
        return crossnode
         
# adt_path = ['../SeuratV4/subdata/4_41_ADT.csv',
#             '../SeuratV4/subdata/4_42_ADT.csv',
#             '../SeuratV4/subdata/4_43_ADT.csv','../SeuratV4/subdata/4_44_ADT.csv']
            
# rna_path = ['../SeuratV4/subdata/4_41_RNA.h5ad',
#             '../SeuratV4/subdata/4_42_RNA.h5ad',
#             '../SeuratV4/subdata/4_43_RNA.h5ad', '../SeuratV4/subdata/4_44_RNA.h5ad']

# adt_path = ['../data/4tumor/subdata/ADT_1.csv','../data/4tumor/subdata/ADT_2.csv',
#             '../data/4tumor/subdata/ADT_3.csv','../data/4tumor/subdata/ADT_3.csv',]
            
# rna_path = ['../data/4tumor/subdata/RNA_1.h5ad','../data/4tumor/subdata/RNA_2.h5ad',
#             '../data/4tumor/subdata/RNA_3.h5ad','../data/4tumor/subdata/RNA_3.h5ad',]

day,adt_path,rna_path,= [0,3,7],[],[],
for i in day:
    for j in range(8):
        adt_path.append(('../SeuratV4/subdata/4_5/t'+str(i)+'p'+str(j+1)+'_ADT.csv'))
        rna_path.append('../SeuratV4/subdata/4_5/t'+str(i)+'p'+str(j+1)+'_RNA.h5ad')

output_path = '../output/7_5/new'
merge_cutoff = 0.1
compact_flag = True
current_treepath = '../output/7_5/rna'#'../data/4tumor/output/ADT_'
current_tree = (np.arange(0,25)).astype(str) #['0','1','2', '3','4'] #[]#
useadt, userna =   True, False,
ifretrain, fitgaussian = False, True


starttime = time.time()
print('read data and run CITE-sort.')

adtdata = {}
rnadata = {}
for i in range(len(adt_path)):
    if adt_path[i] != None:
        adtdata[i] = pd.read_csv(adt_path[i], header = 0, index_col=0, sep=',')
    else:
        adtdata[i] = []
    rnadata[i] = sc.read_h5ad(rna_path[i])
    rnadata[i].var_names_make_unique()

# prior_gene = dict({'exhausted':['TOX','TOX2','LAG3','BTLA','CBLB','ITCH','NEDD4'],
# 'TCM/TEM':['CCR7','CD44','IL7R','IL15RA','MBD2'],'Th17':['KLRB1','IL23R','CCR6','IL1R1','STAT3']})
prior_gene = {}
if len(current_tree) != 0:
    nodelist = []
    for i in range(len(current_tree)):
        f = open(current_treepath+'_'+current_tree[i]+'/tree.pickle','rb')
        tree = pickle.load(f)
        f.close()
        nodelist.append(tree)
    modelnode = nodelist.pop(0)
    crossnode = CrossNode(nodelist, modelnode=modelnode)

    if useadt:      
        crossnode = dfs(crossnode, adtdata, rnadata, merge_cutoff, useADT=True, ifretrain=ifretrain, fitgaussian=fitgaussian)
    else:
        crossnode = dfs(crossnode, {}, rnadata, merge_cutoff, useADT=False, ifretrain=ifretrain, fitgaussian=fitgaussian)
    # data_sub = adtdata.loc[list(set(tree.indices)&set(adtdata.index)),:]
    # tree = dfs(tree, data_sub, merge_cutoff,rnadata)
else:
    # nodelist=[BTree(('leaf',)) for i in range(len(adt_path))]
    crossnode = CrossSplit(adtdata.copy(),merge_cutoff, weight=np.ones(len(adt_path)),rnadata=rnadata.copy())
        
    # tree, bic_list, min_bic_node = Choose_leaf(data=data,merge_cutoff=merge_cutoff,rawdata=data.copy(),datarna=data_rna)

def inner_dfs(node, crossnode, i):
    # print(node.key)
    if node != None and crossnode.left != None:
        # print(crossnode.left.nodelist,crossnode.right.nodelist, node.key)
        node.left = inner_dfs(crossnode.left.nodelist[i], crossnode.left, i) 
        node.right = inner_dfs(crossnode.right.nodelist[i], crossnode.right, i)
    return node

def modeltree_dfs(node, crossnode):
    if node != None and crossnode.left != None:
        node.left = modeltree_dfs(crossnode.left.modelnode,  crossnode.left)
        node.right = modeltree_dfs(crossnode.right.modelnode,  crossnode.right)
    return node

if useadt == False and userna:
    output_path = output_path + '_rna'

for i in range(len(adtdata)):
    if ifretrain:
        break
    tree = inner_dfs(crossnode.nodelist[i], crossnode, i)
    output = output_path+'_'+str(i+1) #+ '_'
    if not os.path.exists(output):
        os.mkdir(output)
    visualize_tree(tree, adtdata[i], output, 'tree', compact=compact_flag, rnadata=rnadata[i].copy())
    f = open(output+'/tree.pickle','wb')
    pickle.dump(tree,f)
    f.close()
    # print('generate labels.')
    traversal = BTreeTraversal(tree,save_min_BIC=False)
    leaves_labels = traversal.get_leaf_label()
    leaves_labels.to_csv(output + '/leaf_labels.csv')

output = output_path+'_0'
if not os.path.exists(output):
    os.mkdir(output)
tree = modeltree_dfs(crossnode.modelnode, crossnode)

if ifretrain:
    f = open(output+'/tree_retrain.pickle','wb')
else:
    visualize_modeltree(tree, output, 'tree')
    f = open(output+'/tree.pickle','wb')
pickle.dump(tree,f)
f.close()

endtime = time.time()

print('Time using: ', round(endtime-starttime, 3),'secs')
# plt.plot(list(range(len(bic_list))), bic_list)
# plt.savefig('BIC_as_split.png')