import pandas as pd
from Classifier.BTree import BTree
from Classifier.Visualize import visualize_tree, visualize_modeltree
from Classifier.BTreeTraversal import BTreeTraversal
from Classifier.adtReSplit import CrossSplit, CrossNode, retrain, gm_proba, ReClassify
import pickle
import argparse
import os
import time
import matplotlib.pyplot as plt
import warnings
import scanpy as sc
warnings.filterwarnings("ignore")
import numpy as np
import scipy.sparse


def dfs(crossnode, adtdata, rnadata, merge_cutoff, useADT=True, ifretrain=False, fitgaussian=False, probafilter=False):
    # if crossnode.modelnode.key == ('CD14',):
    #     for node in nodelist:
    #         crossnode.
    #     if len(crossnode.modelnode.indices) != 8:
    #         crossnode.modelnode.key = ('leaf',)
    # if crossnode.modelnode == None:
    #     return crossnode


    if crossnode.modelnode.ind in [81]  :  # crossnode.modelnode.key == ('leaf',) 
        # print(crossnode.modelnode.key)

        if ifretrain or fitgaussian or reclass:
            return crossnode
        if useADT:
            crossnode = CrossSplit(adtdata.copy(),merge_cutoff, weight=np.ones(len(list(rnadata.keys()))),rnadata=rnadata.copy(),crossnode=crossnode)
        else:
            crossnode = CrossSplit({},merge_cutoff, weight=np.ones(len(list(rnadata.keys()))),rnadata=rnadata.copy(),crossnode=crossnode)
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
        nodelist = crossnode.nodelist

        ### Expand traning data at certain node
        # if crossnode.modelnode.key == ('CD45RA','CD45RO',):
        # # if crossnode.modelnode.key[0][:2]!='CC' and  len(crossnode.modelnode.indices) == 1:
        #     print(crossnode.modelnode.key)
        #     crossnode = CrossSplit(adtdata.copy(),merge_cutoff, weight=np.ones(len(list(rnadata.keys()))), rnadata=rnadata.copy(),crossnode=crossnode,marker_set=crossnode.modelnode.key)
        #     nodelist = crossnode.nodelist
        if ifretrain:
            # if crossnode.modelnode.ind in [5,11,12]: #or crossnode.modelnode.loss.detach().numpy()>10 or len(crossnode.modelnode.key)==2
            # print(crossnode.modelnode.key)
            crossnode.modelnode.artificial_w, crossnode.modelnode.embedding, crossnode.modelnode.loss = retrain(
                        crossnode.nodelist, rnadata.copy(), genes=crossnode.modelnode.artificial_w.index)
        if reclass:
            if crossnode.modelnode.ind == 4:
                crossnode = ReClassify(merge_cutoff, rnadata.copy(), crossnode)

        lnodelist, rnodelist, ladt, radt, lrna, rrna = [], [], {}, {}, {}, {}
        for i in range(len(nodelist)):
            node = nodelist[i]

            if node != None:
                if node.left is not None and node.right is not None:
                    # if len(node.indices) !=  len(node.left.indices) + len(node.right.indices):
                    #     # print(len(node.indices),len(node.left.indices) + len(node.right.indices))
                    #     crossnode.nodelist[i].indices = list(node.left.indices) + list(node.right.indices)
                    #     # print(len(crossnode.nodelist[i].indices))
                    if fitgaussian and crossnode.modelnode.key == ('CD14','CLEC12A',):
                        # print(crossnode.modelnode.key)
                        node = gm_proba(node, adtdata[i], probafilter)
                    # if useADT:
                    #     if len(adtdata[i]) > 0:
                    #         ladt[i], radt[i] = adtdata[i].loc[node.left.indices,:], adtdata[i].loc[node.right.indices,:]
                    #     else:
                    #         ladt[i], radt[i] = [],[]
                    # print(i,crossnode.modelnode.indices,len(rnadata[i]))
                    # lrna[i], rrna[i] = rnadata[i][node.left.indices,:], rnadata[i][node.right.indices,:]


                if node.left is not None:
                    # if node.left.left is not None and node.left.right is not None:
                    #     node.left.indices = list(set(list(node.left.indices)+list(node.left.left.indices) + list(node.left.right.indices)) )
                    node.left.indices = node.left_indices
                    # t = set(node.left.indices)-set(node.left_indices)
                    # if len(t)!=0:
                    #     print(i,t)

                    # if useADT:
                    #     if len(adtdata[i]) > 0:
                    #         ladt[i] = adtdata[i].loc[list(set(node.left.indices)),:]
                    #     else:
                    #         ladt[i] = []
                    # # print(rnadata[i])
                    # lrna[i] = rnadata[i][list(set(node.left.indices)),:]
                    # if node.right is None:
                    #     rrna[i] = rnadata[i][False,:]
                    #     radt[i] = []


                # if node.right is not None:
                    # if node.right.left is not None and node.right.right is not None:
                    #     node.right.indices = list(set(list(node.right.indices)+list(node.right.left.indices) + list(node.right.right.indices)))
                    # node.right.indices = node.right_indices
                    # t = set(node.right.indices)-set(node.right_indices)
                    # if len(t)!=0:
                    #     print(i,t)

                    # if useADT:
                    #     if len(adtdata[i]) > 0:
                    #         radt[i] = adtdata[i].loc[list(set(node.right.indices)),:]
                    #     else:
                    #         radt[i] = []
                    # rrna[i] = rnadata[i][list(set(node.right.indices)),:]
                    # if node.left is None:
                    #     lrna[i] = rnadata[i][False,:]
                    #     ladt[i] = []


                if node.left is None and node.right is None:
                    ladt[i], radt[i], lrna[i], rrna[i] = [],[],[],[]
                lnodelist.append(node.left)
                rnodelist.append(node.right)
            else:
                lnodelist.append(None)
                rnodelist.append(None)

        crossnode.modelnode.left.ind = 2*crossnode.modelnode.ind + 1
        crossnode.modelnode.right.ind = 2*crossnode.modelnode.ind + 2
        lcrossnode = CrossNode(lnodelist, modelnode=crossnode.modelnode.left)
        rcrossnode = CrossNode(rnodelist, modelnode=crossnode.modelnode.right)
        # crossnode.left = dfs(lcrossnode, ladt, lrna, merge_cutoff, useADT, ifretrain, fitgaussian, probafilter)
        # crossnode.right = dfs(rcrossnode, radt, rrna, merge_cutoff, useADT, ifretrain, fitgaussian, probafilter)
        crossnode.left = dfs(lcrossnode, adtdata, rnadata, merge_cutoff, useADT, ifretrain, fitgaussian, probafilter)
        crossnode.right = dfs(rcrossnode, adtdata, rnadata, merge_cutoff, useADT, ifretrain, fitgaussian, probafilter)

        # if crossnode.modelnode.ind in [12]: 
        #     print(crossnode.modelnode.ind)
        #     for i in range(len(crossnode.right.nodelist)):

        #         if crossnode.left.nodelist[i] is not None and crossnode.right.left.nodelist[i] is not None:
        #             # print('before:',len(crossnode.right.left.left.nodelist[i].indices))
        #             # if crossnode.right.left is None:
        #             #     crossnode.right.left = crossnode.left

        #             crossnode.left.nodelist[i].indices = crossnode.left.nodelist[i].indices.append(
        #                 crossnode.right.right.nodelist[i].indices)
        #             # print('after:',len(crossnode.right.left.left.nodelist[i].indices))
        #             # crossnode.left.nodelist[i].indices = crossnode.nodelist[i].indices
        #         # elif crossnode.left.nodelist[i] is not None:
        #         #     crossnode.right.nodelist[i] = crossnode.left.left.nodelist[i]
        #     crossnode.right = crossnode.right.left


        return crossnode

# adt_path = ['../SeuratV4/subdata/4_31_ADT.csv',
#             '../SeuratV4/subdata/4_32_ADT.csv',
#             '../SeuratV4/subdata/4_33_ADT.csv','../SeuratV4/subdata/4_34_ADT.csv']

# rna_path = ['../SeuratV4/subdata/4_31_RNA.h5ad',
#             '../SeuratV4/subdata/4_32_RNA.h5ad',
#             '../SeuratV4/subdata/4_33_RNA.h5ad', '../SeuratV4/subdata/4_34_RNA.h5ad']

# adt_path = ['../data/4tumor/subdata/ADT_1.csv','../data/4tumor/subdata/ADT_2.csv',
#             '../data/4tumor/subdata/ADT_3.csv','../data/4tumor/subdata/ADT_3.csv',]

# rna_path = ['../data/4tumor/subdata/RNA_1.h5ad','../data/4tumor/subdata/RNA_2.h5ad',
#             '../data/4tumor/subdata/RNA_3.h5ad','../data/4tumor/subdata/RNA_3.h5ad',]

day,adt_path,rna_path,path= [0,3,7],[],[],[]
# for i in day:
#     for j in range(8):
#         adt_path.append(('../SeuratV4/subdata/4_5/t'+str(i)+'p'+str(j+1)+'_ADT.csv'))
#         rna_path.append('../SeuratV4/subdata/4_5/t'+str(i)+'p'+str(j+1)+'_RNA.h5ad')

# for i in range(4,7):
#     adt_path.append('../simulator_out/simulator_SET2/ADT_simulator_'+str(i)+'.csv')
#     rna_path.append('../simulator_out/simulator_SET2/RNA_simulator_'+str(i)+'.h5ad')


# for i in range(6):
#     path.append('../data/PBMC DATA/'+str(i+1)+'/data.h5ad')
path.append('../data/RealSimulation/set1/data.h5ad')


output_path = '../output/RealSimulation/set1_fine'
merge_cutoff = 0.5
compact_flag = True
current_treepath = '../output/RealSimulation/set1_fine'#'../data/4tumor/output/ADT_'
current_tree = [ '0','1','2', '3','4'] #  (np.arange(0,9)).astype(str)
useadt, userna =  False,  True,  
ifretrain, fitgaussian, probafilter, reclass =    True, False, False,  False,  #  


starttime = time.time()
print('read data and run CITE-pool.')

adtdata = {}
rnadata = {}
for i in range(len(adt_path)):
    if adt_path[i] != None:
        adtdata[i] = pd.read_csv(adt_path[i], header = 0, index_col=0, sep=',')
    else:
        adtdata[i] = []
    rnadata[i] = sc.read_h5ad(rna_path[i])
    rnadata[i].var_names_make_unique()
    inbatch = [len(adt_path)]
    sc.pp.normalize_total(rnadata[i], target_sum=1e4)
    sc.pp.log1p(rnadata[i])

if len(adt_path)==0:
    j,inbatch = 0,[]
    # print(path)
    for i in range(len(path)):
        jpre = j
        adata = sc.read_h5ad(path[i])
        batch = adata.obs['batch'].cat.categories
        for b in batch:
            data = adata[adata.obs['batch']==b]
            adtdata[j] = data[:,data.var['feature_types']=='Antibody Capture']

            rnadata[j] = data[:,data.var['feature_types']=='Gene Expression']

            x = rnadata[j][:31,:20].X.toarray().flatten()
            if len(list(set(x))) <= 3:
                x = rnadata[j][:37,:60].X.toarray().flatten()
                if len(list(set(x))) <= 3:
                    x = rnadata[j][:37,:600].X.toarray().flatten()
            x = list(set(x))
            s = np.sort(x)
            if s[-2] == int(s[-2]) and s[-3] == int(s[-3]):
                # print(s)
                sc.pp.normalize_total(rnadata[j], target_sum=1e4)
                sc.pp.log1p(rnadata[j])

            del(data)

            adtdata[j] = adtdata[j].to_df()
            # print(adtdata[i].shape,adtdata[i].iloc[:,0].shape)
            # if scipy.sparse.issparse(rnadata[i])== False:
            #     print(rnadata[j].X.shape)
            #     rnadata[j].X = scipy.sparse.csr_matrix(rnadata[j].X)
            j = j + 1
            # if j-jpre > 1:
            #     break
        inbatch.append(j-jpre)
        print('dataset',i+1,'batch num:',j-jpre,'ADT num:',adtdata[j-1].shape[1])




# print(len(rnadata[5]))
# prior_gene = dict({'exhausted':['TOX','TOX2','LAG3','BTLA','CBLB','ITCH','NEDD4'],
# 'TCM/TEM':['CCR7','CD44','IL7R','IL15RA','MBD2'],'Th17':['KLRB1','IL23R','CCR6','IL1R1','STAT3']})
prior_gene = {}
dataid = 0
if len(current_tree) != 0:
    nodelist = []
    for i in range(len(adtdata)+1):
        # if current_tree[i] =='0':
        #     f = open(current_treepath+'_'+current_tree[i]+'/tree_retrain.pickle','rb')
        #     # f = open('../output/9_2/_0/tree_retrain.pickle','rb')
        # else:
        #     f = open(current_treepath+'_'+current_tree[i]+'/tree.pickle','rb')

        if i ==0:
            f = open(current_treepath+'/0'+'/tree.pickle','rb')
            # print('using retrained tree')
            tree = pickle.load(f)
            f.close()
            nodelist.append(tree)
            continue
        if not os.path.exists(current_treepath+'/'+str(dataid)+'/'+str(i)):
            dataid = dataid + 1
        f = open(current_treepath+'/'+str(dataid)+'/'+str(i)+'/tree.pickle','rb')

        tree = pickle.load(f)
        f.close()
        nodelist.append(tree)
    print('existed tree num:',len(nodelist))
    modelnode = nodelist.pop(0)
    modelnode.ind = 0
    crossnode = CrossNode(nodelist, modelnode=modelnode)

    if useadt:
        crossnode = dfs(crossnode, adtdata, rnadata.copy(), merge_cutoff, useADT=True, ifretrain=ifretrain, fitgaussian=fitgaussian, probafilter=probafilter)
    else:
        crossnode = dfs(crossnode, {}, rnadata.copy(), merge_cutoff, useADT=False, ifretrain=ifretrain, fitgaussian=fitgaussian, probafilter=probafilter)
    # data_sub = adtdata.loc[list(set(tree.indices)&set(adtdata.index)),:]
    # tree = dfs(tree, data_sub, merge_cutoff,rnadata)
else:
    # nodelist=[BTree(('leaf',)) for i in range(len(adt_path))]

    crossnode = CrossSplit(adtdata.copy(),merge_cutoff, weight=np.ones(len(adtdata)),rnadata=rnadata.copy())

    # tree, bic_list, min_bic_node = Choose_leaf(data=data,merge_cutoff=merge_cutoff,rawdata=data.copy(),datarna=data_rna)

temp_indices = {}
def inner_dfs(node, crossnode, i):
    # print(node.key)
    if node != None and crossnode.left != None:
        # print(crossnode.left.nodelist,crossnode.right.nodelist, node.key)
        node.left = inner_dfs(crossnode.left.nodelist[i], crossnode.left, i)
        node.right = inner_dfs(crossnode.right.nodelist[i], crossnode.right, i)


        # if i in range(1,9):
        #     if crossnode.modelnode.key == ('CD127',):
        #         temp_indices[i] = node.right.indices
        #         node.right.indices = []
        #     if crossnode.modelnode.key in [('CD20','CD21',),('CD16','CD19',),('CD14','CLEC12A')] or crossnode.modelnode.val_cnt == 83193:
        #         print(i,len(temp_indices[i]))
        #         node.indices = node.indices + temp_indices[i]
        #         print(len(node.indices))

    return node

def modeltree_dfs(node, crossnode):
    if node != None and crossnode.left != None:
        node.left = modeltree_dfs(crossnode.left.modelnode,  crossnode.left)
        node.right = modeltree_dfs(crossnode.right.modelnode,  crossnode.right)
    return node

# if useadt == False and userna:
#     output_path = output_path + '_rna'

output = output_path+'/0'
if not os.path.exists(output):
    os.mkdir(output)
modeltree = modeltree_dfs(crossnode.modelnode, crossnode)

if ifretrain:
    f = open(output+'/tree_retrain.pickle','wb')
else:
    visualize_modeltree(modeltree, output, 'tree')
    f = open(output+'/tree.pickle','wb')
pickle.dump(modeltree,f)
f.close()

batch, dataid  = 1, 1
for i in range(len(adtdata)):
    if ifretrain:
        break
    tree = inner_dfs(crossnode.nodelist[i], crossnode, i)
    tree.indices = rnadata[i].obs_names


    if batch > inbatch[0]:
        batch = 1
        inbatch.pop(0)
        dataid += 1
    output = output_path+'/'+ str(dataid)
    if not os.path.exists(output):
        print(output)
        os.mkdir(output)
    output = output + '/' + str(i+1)

    # output = output_path+'_'+str(i+1) #+ '_'
    print(output)
    if not os.path.exists(output):
        os.mkdir(output)
    visualize_tree(tree, adtdata[i], output, 'tree', compact=compact_flag, rnadata=rnadata[i].copy(),modeltree=modeltree)
    f = open(output+'/tree.pickle','wb')
    pickle.dump(tree,f)
    f.close()
    # print('generate labels.')
    traversal = BTreeTraversal(tree,save_min_BIC=False)
    leaves_labels = traversal.get_leaf_label()
    leaves_labels.to_csv(output + '/leaf_labels.csv')

    batch += 1




endtime = time.time()

print('Time using: ', round(endtime-starttime, 3),'secs')
# plt.plot(list(range(len(bic_list))), bic_list)
# plt.savefig('BIC_as_split.png')