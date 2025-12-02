#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:44:58 2020

@author: lianqiuyu
"""



from sklearn.neighbors import KNeighborsClassifier as knc
from scalable_gcca import learn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.nn as nn
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_decomposition import CCA, PLSRegression, PLSCanonical
from sklearn.preprocessing import normalize
from scipy.sparse import isspmatrix_csr
import logging
from Classifier.gcca import GCCA
import scanpy as sc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import diptest
import random
import copy
from BTree import BTree
from scipy.spatial import distance
import operator
from scipy import stats, sparse
import itertools
from scipy.stats import multivariate_normal, norm
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import sys
sys.path.append("./CITEsort")

#from scipy.signal import upfirdn
#import pandas as pd


def all_BIC(leaf_dict, n_features):
    ll, n_features, n_sample = 0, 0, 0
    for key, node in leaf_dict.items():
        ll = ll + node.ll * node.weight
        # n_features = n_features + len(node.key)
        n_sample = n_sample + len(node.indices)
        # node_name.append(str(key))
    cov_params = len(leaf_dict) * n_features * (n_features + 1) / 2.0
    mean_params = n_features * len(leaf_dict)
    n_param = int(cov_params + mean_params + len(leaf_dict) - 1)
    bic_score = -2 * ll * n_sample + n_param * np.log(n_sample)

    return bic_score


def assign_GMM(sample, mean_list, cov_list, weight, if_log=False, marker_list=None, confidence_threshold=0, throw=True):
    """confidence_threshold is used to not assign dots with low confidence to each group:
        a big confidence_threshold represents a more strict standard for confidential dots"""
    # confidence_threshold = (1-confidence_threshold) / len(weight) * 2
    index = sample.index
    # sample = np.array(sample)
    weight = np.array(weight)
    if if_log:
        type_num = np.log(weight/sum(weight))
    else:
        type_num = weight/sum(weight)

    p_prior = np.zeros(shape=(len(sample), len(weight)))
    for i in range(len(weight)):
        if if_log:
            # print('sample_null',sample.loc[:,marker_list].isnull().any().any(), 'mean_null', mean_list[i][marker_list].isnull().any().any(), 'cov_null', cov_list[i].loc[marker_list,marker_list].isnull().any().any())
            # print('marker len:',marker_list[i].shape[0])
            p_prior[:, i] = multivariate_normal.logpdf(np.array(sample.loc[:, marker_list]), mean=np.array(
                mean_list[i][marker_list]), cov=np.array(cov_list[i].loc[marker_list, marker_list]), allow_singular=True)
            p_prior[:, i] = p_prior[:, i] + type_num[i]

        else:
            p_prior[:, i] = multivariate_normal.pdf(np.array(sample.loc[:, marker_list]), mean=np.array(
                mean_list[i][marker_list]), cov=np.array(cov_list[i].loc[marker_list, marker_list]), allow_singular=True)
            p_prior[:, i] = p_prior[:, i] * type_num[i]
    # p_prior = -p_prior

    p_post = p_prior / (p_prior.sum(axis=1)[:, np.newaxis])
    if if_log:
        pred_label = np.argmin(p_post, axis=1)
    else:
        pred_label = np.argmax(p_post, axis=1)
    if throw:
        if if_log:
            pred_label = [pred_label[i] if p_post[i, pred_label[i]] < (
                1-confidence_threshold)/len(weight)*2 else -1 for i in range(len(pred_label))]
        else:
            pred_label = [pred_label[i] if p_post[i, pred_label[i]] >
                          confidence_threshold/len(weight)*2 else -1 for i in range(len(pred_label))]
    # print(p_post[:10,:])
    # print(pred_label[:10])
    # print(p_post[:5,:],pred_label[:5])
    pred_label = pd.Series(data=pred_label, index=index)
    return pred_label




def smooth(x, item=0, num=6):
    if False:
    # if x.columns[0][:2]!='CC':
        if x.min().min() >= 0:
            # i = [i for i in item][0]
            # print(value[0],value[1])
            x = x.apply(np.exp)
            # print('<0')
            # for i in x.columns:
            #     x.loc[:,i] += np.random.normal(loc=0, scale=1,size=x.shape[0]) * 0.01
            # return x
        # else:
            rawdata = np.apply_along_axis(
                lambda x: np.log(x+1) - np.mean(np.log(x+1)), 0, x)
            rawdata = pd.DataFrame(rawdata, index=x.index, columns=x.columns)

    # print('before',(x.isnull()).any())
    s = x.min().min()
    if x.columns[0][:2]!='CC' and s >= 0:
        # print('unprocessed','x.min:',s)
        y = np.apply_along_axis(lambda x: np.log(x+1) - np.mean(np.log(x+1)),0,x)   
        x = pd.DataFrame(y, index=x.index, columns=x.columns)
    # print('processed','x.min:',x.min().min())
    x.mask(x.isnull(),0)

    for i in x.columns:
        value = np.unique(x.loc[:, i].values.tolist())
        num = min(len(value), num)
        # print(x.shape,x.loc[:,i].shape,i)
        x.loc[:, i] += np.random.normal(loc=0, scale=1, size=x.shape[0]) * 0.01

        # for k in range(num-1):
        #     # print(x.loc[x.loc[:,i]==value[k],i])
        #     x.loc[x.loc[:,i]==value[k],i] += np.random.normal(loc=0, scale=1, size=sum(x.loc[:,i]==value[k])) * (value[k+1]-value[k])*0.1
        # print(i,':',len(value))

    # print('after',(x.isnull()).any())
    # print(x.columns)
    return x


def value_count(data):
    val_cnt = pd.Series(index=data.columns)
    for col in data.columns:
        val, cnt = np.unique(data[col].values.tolist(), return_counts=True)
        val_cnt[col] = len(val)
    return val_cnt


def GenModelNode(crossnode, best_feature, artificial_w, score_dict, outputmean=[0, 0], loss=0):
    modelnode = BTree(best_feature)
    leftm, rightm, cellnum, trainnum, traindataset = 0, 0, 0, 0, []

    if best_feature == ('leaf',):
        for node in crossnode.nodelist:
            if node != None:  # and node.indices != None:
                cellnum += len(node.indices)
        modelnode.val_cnt = cellnum
        modelnode.score_dict = score_dict
        crossnode.modelnode = modelnode
        return crossnode

    for i in range(len(crossnode.nodelist)):
        node = crossnode.nodelist[i]
        if node is not None:
            cellnum += len(node.indices)
            # continue
            # print(best_feature,node.key)
            if node.key != ('leaf',) and node.key != ('artificial',):
                # print(node.mean_l)
                leftm += node.mean_l
                rightm += node.mean_r
                trainnum += 1
                traindataset.append(i+1)

    leftm = leftm / trainnum
    rightm = rightm / trainnum

    modelnode.indices = traindataset
    modelnode.mean_l, modelnode.mean_r = leftm, rightm
    modelnode.val_cnt = cellnum
    modelnode.artificial_w = artificial_w
    modelnode.loss = loss
    modelnode.embedding = outputmean

    crossnode.modelnode = modelnode
    return crossnode


def outlier_filter(data):

    iqr = data.quantile([0.25, 0.5, 0.75])
    iqr.loc['iqr', :] = (iqr.loc[0.75, :] - iqr.loc[0.25, :])*3
    iqr.loc['min', :] = iqr.loc[0.25, :]-iqr.loc['iqr', :]
    iqr.loc['max', :] = iqr.loc[0.75, :]+iqr.loc['iqr', :]
    index = set(data.index)
    flag = False
    for col in iqr.columns:
        xmax, xmin = max(data.loc[:, col]), min(data.loc[:, col])
        if xmax-xmin > 5*iqr.loc['iqr', col]:
            flag = True
        if xmax-xmin < 3*iqr.loc['iqr', col]:
            continue
        index = index & set(data[data.loc[:, col] < iqr.loc['max', col]].index) & set(
            data[data.loc[:, col] > iqr.loc['min', col]].index)
        if len(data)-len(index) > 30 or len(index) < 50:
            index = set(data.index) - set(data[col].iloc[[np.argmax(
                list(data.loc[:, col])), np.argmin(list(data.loc[:, col]))]].index)
            break
    return list(index), flag




def retrain(nodelist, rnadata, adtdata, feature, modelnode):  # gene?
    for i in range(len(nodelist)):
        if i in modelnode.indices:
            nodelist[i].key = modelnode.key
    genes = cca_gene_selection(adtdata, rnadata, nodelist, feature)
    
    traindata = {}
    for i in range(len(nodelist)):
        node = nodelist[i]
        # print(node,adtdata[i])
        node, probs = gm_proba(node, adtdata[i])
        # if i  in [1,0]:
        #     continue
        
        if  len(node.indices) > 30 and node is not None and node.left is not None and node.right is not None : #
            rnadata[i] = rnadata[i][list(node.left.indices)+list(node.right.indices), :]
            rnadata[i].obs['label'] = 0
            rnadata[i].obs['label'].loc[node.right.indices] = 1

            # if len(node.left.indices) < 30  or len(node.right.indices) < 30:
            #     continue
            if node.key == ('artificial',) :
                marker = pd.DataFrame(0,index=rnadata[i].obs_names, columns=['artificial'])
            else:
                if feature[0][:2] == 'CC':
                    marker = node.embedding.loc[:,feature]
                else:
                    marker = adtdata[i].loc[:,feature]
            marker = marker.loc[marker.index.intersection(rnadata[i].obs_names),:]

            traindata[i] = rnadata[i][marker.index,genes] 
            traindata[i].obsm['marker'] = marker.values
            probs2d = pd.DataFrame(0.01,index=traindata[i].obs_names, columns=['0','1'])
            
            probs2d.loc[node.left.indices,'0'] = probs[node.left.indices]
            probs2d.loc[node.right.indices,'1'] = probs[node.right.indices]
            # print(probs2d)
            # print(probs2d.index.intersection(traindata[i].obs_names))
            traindata[i].obsm['probs'] = probs2d

            # print(traindata[i].shape,len(genes))
            
    print('------start retrain------')
    
    w, m0, m1, loss, deltaW, probs = LearnPseudoMaker(traindata)
    # w, m0, m1, loss = train_classifier(traindata)
    del(traindata)
    w = pd.Series(w.detach().numpy().reshape(-1), index=genes)
    for i in range(len(nodelist)): 
        nodelist[i].artificial_w = w  + pd.Series(deltaW[i].detach().numpy().reshape(-1), index=genes)
        p = probs[i].detach()
        pred =  pd.DataFrame(np.argmax(p, axis=1), index=rnadata[i].obs_names)
        nodelist[i].left_indices = pred[pred==0].index
        nodelist[i].right_indices = pred[pred==1].index
        # print(nodelist[i].artificial_w)


    return w, [m0, m1], loss, nodelist

from sklearn.mixture import GaussianMixture as GMM


def gmm_class(node, data):
    gm =  GMM(n_components=2)
    k_bic, _ = BIC(data, 4, 'bic')
    print(k_bic)
    if k_bic == 2:
        pred = gm.fit_predict(data)
        pred = pd.Series(index=data.index, data = pred)
        left = pred==np.argmin(gm.means_)

        # print(gm.means_, left, len(pred[~left]))
        node.left_indices = pred[left].index
        node.right_indices = pred[~left].index
        return node, False
    else:
        return node, True



def gm_proba(node, data={}, filter=False):
    # print(node.key)
    node.left_indices = node.left_indices.intersection(data.index)
    node.right_indices = node.right_indices.intersection(data.index)
    if len(node.left_indices) < 2 or len(node.right_indices) < 2:
        print(node.key)
        return node, pd.Series(0.99, index=node.indices,name='new')
    
    if node.key == ('artificial',) or len(node.artificial_w) == len(data):
        data = node.artificial_w
    elif node.key[0][:2] == 'CC':
        data = node.embedding.loc[:, node.key]
    else:
        data = data.loc[:, node.key]
        data = smooth(data.copy())
    
    data = data.loc[node.left_indices.append(node.right_indices),:]
    lind, rind = pd.Index(node.left_indices).intersection(
        data.index), pd.Index(node.right_indices).intersection(data.index)
    meanl, meanr = data.loc[lind, :].mean(
        axis=0), data.loc[rind, :].mean(axis=0)
    covl, covr = data.loc[lind, :].cov(), data.loc[rind, :].cov()
    # print(len(lind),len(rind))
    p00 = multivariate_normal.pdf(data.loc[lind, :], meanl, covl)
    p11 = multivariate_normal.pdf(data.loc[rind, :], meanr, covr)
    p01 = multivariate_normal.pdf(data.loc[lind, :], meanr, covr)
    p10 = multivariate_normal.pdf(data.loc[rind, :], meanl, covl)
    w = len(lind)/(len(rind) + len(lind))

    proba0 = w*p00/(w*p00 + (1-w)*p01)
    proba1 = (1-w)*p11/(w*p10 + (1-w)*p11)
    proba = pd.DataFrame(index=lind.append(rind), columns=[
                         'parent', 'child', 'new'])
    try:
        proba['parent'] = node.proba
    except:
        proba['parent'] = np.ones(len(node.indices))
        node.proba = np.ones(len(node.indices))

    # proba['child'] = 2
    proba.loc[node.left_indices, 'child'] = proba0
    proba.loc[node.right_indices, 'child'] =  proba1
    proba['new'] = proba['child'].mul(proba['parent'])
    node.left.proba = proba.loc[node.left_indices, 'new']
    node.right.proba = proba.loc[node.right_indices, 'new']
    # print(node.left.indices )

    if filter:
        # print('filter')
        node.left_indices =  node.left.proba[node.left.proba > 0.99].index
        node.right_indices = node.right.proba[node.right.proba > 0.99].index

    # print(node, proba['new'])
    return node, proba['new']


def RNApca(rnadata=None):
    pcs = {}
    for i in rnadata.keys():

        adata = rnadata[i].copy()
        if len(adata) < 100:
            pcs[i] = None
            continue
        # sc.pp.normalize_total(adata, target_sum=1e4)
        # sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_disp=0.25)
        adata = adata[:, adata.var.highly_variable]
        sc.pp.scale(adata)
        sc.pp.pca(adata, n_comps=5)
        pcs[i] = pd.DataFrame(adata.obsm['X_pca'], index=adata.obs_names, columns=[
                              'PC'+str(k+1) for k in range(5)])
    return pcs


def finish(nodelist, rnadata, feature_scores, separable_features, scores_ll):
    for i in range(len(nodelist)):
        node = nodelist[i]
        if node != None:
            node.indices = rnadata[i].obs_names
            node.left,node.right = None, None
            if len(feature_scores) == 0:
                node.stop = 'separable feature:'+str(len(feature_scores))
            elif max(list(feature_scores.values())) <= -10:
                node.stop = 'best feature score:' + \
                    str(max(list(feature_scores.values())))
            else:
                node.stop = 'KNN classifier low confidence'

    crossnode = CrossNode(nodelist)
    score_dict = dict(zip(separable_features, scores_ll))
    crossnode = GenModelNode(crossnode, ('leaf',),
                             artificial_w=None, score_dict=score_dict)
    return crossnode


class CrossNode():
    def __init__(self, nodelist, left=None, right=None, modelnode=None):
        self.nodelist = nodelist
        self.left = left
        self.right = right
        self.modelnode = modelnode


def RNApp(rnadata=None, nodelist=None):
    # set(rnadata[list(rnadata.keys())[0]].var_names)
    ppdata, gene_list = {}, []
    ppdata = {}
    # cellnum = [data.shape[0] for data in list(rnadata.values())]
    nrepeat = 1
    ncomp = 5

    # usedata = [np.random.choice(len(rnadata.keys()),min(6,len(rnadata)), replace=False) for i in range(nrepeat)]
    # print(usedata)

    # adata = sc.read_h5ad('./SeuratV4/subdata/4_5/4_50_RNA.h5ad')

    for i in rnadata.keys():
        if nodelist is not None:
            if nodelist[i] is not None:
                nodelist[i].indices = nodelist[i].indices.intersection(rnadata[i].obs_names)
                rnadata[i] = rnadata[i][list(set(nodelist[i].indices)),:] 
            else:
                rnadata[i] = rnadata[i][False,:]
        if i == list(rnadata.keys())[0]:
            adata = rnadata[i]
            continue
        if len(rnadata[i]) == 0 :
            continue
        # print(len(rnadata[i]))
        # if i == 6:
            # print(nodelist[i].proba)
        # rnadata[i] = rnadata[i][nodelist[i].proba>0.99,:]
        if len(adata) == 0:
            adata = rnadata[i]
        else:
            # adata.concatenate(rnadata[i],batch_key='batch',join='outer')
            adata = sc.concat([adata, rnadata[i]],join='inner')

        # print(i)
        # rnadata[3][rnadata[3].obs_names,:]
    # print(adata.obs_names[:10])
    # print(len(adata.var_names))
    # print(np.isinf(adata.X.toarray()).sum())  
    # print("Max value:", adata.X.max())
    # print("Any negative values:", (adata.X.toarray() < 0).any())  
    # adata = adata[:,:10]
    if len(adata) < 50:
        return {}
    # adata.obs['batch'] = pd.Categorical(adata.obs['batch'])
    adata.obs['batch'] = adata.obs['batch'].cat.remove_unused_categories()
    sc.pp.highly_variable_genes(adata, n_top_genes=500, batch_key='batch')#
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=ncomp)
    pcs = pd.DataFrame(index=adata.obs_names, data=adata.obsm['X_pca'])
    # indices, flag = outlier_filter(pcs)
    # print(len(pcs),len(indices))
    # adata = adata[indices,:]
    # sc.pp.highly_variable_genes(adata,min_disp=0.25)
    # adata = adata[:,adata.var.highly_variable]
    # sc.pp.scale(adata)
    # sc.pp.pca(adata, n_comps=ncomp)

    vars = list()
    for i in rnadata.keys():
        if len(rnadata[i]) < 30:
            vars.append(0)
            continue
        # print(rnadata[i])
        rnadata[i] = rnadata[i][list(
            set(rnadata[i].obs_names.intersection(adata.obs_names))), :]
        variance = adata[rnadata[i].obs_names, :].obsm['X_pca'][:, 0].var()
        variance += adata[rnadata[i].obs_names, :].obsm['X_pca'][:, 1].var()
        vars.append(variance+len(rnadata)*len(rnadata[i])/len(adata))
    # print(min(vars))
    # print(np.argsort(vars)[-3:])

    pre = min(len(rnadata), 8)
    usedata = [np.argsort(vars)[-pre:]]
    # print(usedata, rnadata.keys())

    # bestv = usedata[0][0]
    # bestdata = adata[list(set(rnadata[bestv].obs_names)&set(adata.obs_names))]
    # sc.tl.pca(bestdata,n_comps=ncomp)
    # loading = bestdata.varm['PCs']

    # print(usedata)
    # usedata = [sorted(vars, key=vars.get, reverse=True)[:4]]

    # cellnumlist = [len(rnadata[i]) for i in rnadata.keys()]
    # largest  = np.argmax(cellnumlist)
    # adata = rnadata[largest].copy()

    # print(sum(adata.X.toarray()[:100,:100]))

    # genes = pd.read_csv('../SeuratV4/subdata/4_5/degenes.csv')
    # gene_list = list(genes.iloc[:200,1])
    # gene_list = gene_list + list(genes.iloc[-200:,1])

    data_cc = {}
    for i in rnadata.keys():
        if len(rnadata[i]) < 100:
            data_cc[i] = []
            continue
        rnadata[i] = rnadata[i][list(set(rnadata[i].obs_names)&set(adata.obs_names)),:]
        data_cc[i] = adata[rnadata[i].obs_names,:].obsm['X_pca']

        # if i != bestv:
        #     data_cc[i] = adata[rnadata[i].obs_names,:].X.dot(loading)
        # else:
        #     data_cc[i] = bestdata.obsm['X_pca']
        data_cc[i] = pd.DataFrame(data=data_cc[i][:,:ncomp],index=rnadata[i].obs_names,columns=['CC_'+str(k+1) for k in range(ncomp)])
        indices = data_cc[i].index
    return data_cc

    for k in range(nrepeat):
        sortid = np.argsort(np.array([len(rnadata[i])
                            for i in usedata[k]]))[::-1]
        largeid = usedata[k][sortid[:4]]
        for i in usedata[k]:
            if len(rnadata[i]) < 100 or i not in largeid:  # or i%8==3
                usedata[k] = np.delete(usedata[k], np.where(usedata[k] == i))
                continue
            rnadata[i] = rnadata[i][list(
                set(rnadata[i].obs_names) & set(adata.obs_names)), :]
            if i in ppdata.keys():
                continue
            # if rnadata[i].shape[0] > 1000:

            #     rand = np.random.choice(
            #         rnadata[i].shape[0]-1, 1000, replace=False)
            #     rnadata[i] = rnadata[i][rand, :]

            # adata = rnadata[i].copy()
            # print(sum(adata.X.toarray()[:100,:100]))
            # sc.pp.normalize_total(adata, target_sum=1e4)
            # sc.pp.log1p(adata)
            # sc.pp.highly_variable_genes(adata,min_disp=1.5)

            ppdata[i] = adata[rnadata[i].obs_names,:].obsm['X_pca'].T
            ppdata[i] = normalize(ppdata[i],copy=True)

            # ppdata[i] = adata[rnadata[i].obs_names, :].X.toarray().T

            # ppdata[i] = adata[:,gene_list].X.toarray().T
            # adata = adata[:,adata.var.highly_variable]
            # sc.pp.scale(adata)
            # sc.pp.pca(adata, n_comps=100)

            # ppdata[i] = adata.obsm['X_pca'].T

            # ppdata[i] = sparse.csr_matrix(adata.obsm['X_pca'].T)
            # ppdata[i] = adata[:,adata.var['highly_variable']]

            # genes = [adata.var_names[x] for t in range(3) for x in np.argsort(adata.varm['PCs'][:,t])[-10:][::-1]]
            # genes = genes + [adata.var_names[x] for t in range(3) for x in np.argsort(adata.varm['PCs'][:,t])[:10][::-1]]
            # gene_list = gene_list + genes
            # gene_list  = gene_list & set(list(adata.var_names))
            # gene_list  = gene_list + list(ppdata[i].var_names)# & set(list(ppdata[i].var_names))

    data_cc = {}
    if len(ppdata.keys()) < 2:
        for i in ppdata.keys():
    # if True:
    #     for i in rnadata.keys():
            if len(rnadata[i]) < 100:  # or i%8 == 3
                data_cc[i] = []
                continue
            ppdata[i] = adata[rnadata[i].obs_names, :].obsm['X_pca'][:,:5]
            # ppdata[i] = normalize(ppdata[i],copy=True)
            temp = normalize(ppdata[i], copy=True)
            # temp = ppdata[i]
            data_cc[i] = pd.DataFrame(data=temp, index=rnadata[i].obs_names, columns=[
                                      'CC_'+str(k+1) for k in range(temp.shape[1])])
        return data_cc

    # for i in ppdata.keys():
    #     # ppdata[i] = ppdata[i][:,list(gene_list)].X.T
    #     gene_list = set(gene_list)
    #     if isspmatrix_csr(ppdata[i].X):
    #         ppdata[i] = ppdata[i][:,list(gene_list)].X.toarray().T
    #     else:
    #         ppdata[i] = ppdata[i][:,list(gene_list)].X.T

    score_list, loadingdict, vectordict = [], {}, {}

    for k in range(nrepeat):
        if len(usedata[k]) < 2:
            score_list.append(-10000)
            continue
        logging.root.setLevel(level=logging.INFO)
        gcca = GCCA(n_components=5)
        # gcca.fit([ppdata[i] for i in usedata[k]])
        # cca_vector = gcca.transform([ppdata[i] for i in usedata[k]])
        # # print(len([ppdata[i] for i in usedata[k]]))
        # # print(dict(zip(usedata[k],[ppdata[i] for i in usedata[k]])))
        # # cca_vector = gcca.fit_transform(dict(zip(usedata[k],[ppdata[i] for i in usedata[k]])))
        # cca_loading = gcca.h_list
        # print(gcca.eigvals)

        cca_vector, cca_loading = [], []
        # print(np.mean(cca_vector,axis=0).shape)
        mean_vector = np.mean(cca_vector, axis=0)  # [:,0]
        # print(len(cca_vector),cca_vector[0].shape)
        # mean_vector = cca_vector[np.argmax([len(rnadata[i]) for i in usedata[k]])][:,0]
        # mean_vector = np.mean([cca_vector[i][:,0] for i in range(len(cca_vector))],axis=0)

        cca = CCA(n_components=1)
        mincor, meanscore, num = 1, [], 0
        for j in rnadata.keys():
            if j in usedata[k]:
                continue
            if len(rnadata[j]) < 10:
                continue
            if j not in list(ppdata.keys()):
                rnadata[j] = rnadata[j][list(
                    set(rnadata[j].obs_names) & set(adata.obs_names)), :]
                # adata = rnadata[j].copy()

                # sc.pp.normalize_total(adata, target_sum=1e4)
                # sc.pp.log1p(adata)
                # sc.pp.highly_variable_genes(adata,min_disp=0.5)
                # adata = adata[:,adata.var.highly_variable]

                # adata = adata[:,list(gene_list)]
                # sc.pp.scale(adata)
                # sc.pp.pca(adata, n_comps=100)

                # ppdata[j] = adata.obsm['X_pca'].T

                # ppdata[j] = adata[:,list(gene_list)].X.toarray().T

                ppdata[j] = adata[rnadata[j].obs_names,:].obsm['X_pca'].T
                ppdata[j] = normalize(ppdata[j],copy=True)

                # ppdata[j] = adata[rnadata[j].obs_names, :].X.toarray().T
            continue

            cca.fit(ppdata[j], mean_vector.reshape(-1, 1))
            cor = cca.score(ppdata[j], mean_vector.reshape(-1, 1))
            # mincor = min(cor,mincor)
            meanscore.append(cor)
            num = num + 1
            # cca_loading = cca.y_rotations_
        # print(meanscore)
        meanscore = [0]
        score_list.append(np.median(meanscore))
        loadingdict[k] = cca_loading
        vectordict[k] = cca_vector

    # bestk=0
    # mean_vector = mean_vector[:,0]
    bestk = np.argmax(score_list)
    # print(usedata[bestk])
    # cca = CCA(n_components=5,max_iter=1000)
    # cca = PLSRegression(n_components=5, max_iter=1000)
    cca = PLSCanonical(n_components=5, max_iter=1000)
    if max(score_list) != -10000:
        # mean_vector = vectordict[bestk][0]
        mean_vector = ppdata[largeid[0]]
        # mean_vector = np.mean(vectordict[bestk],axis=0)

    # print('largeid',largeid)
    for i in rnadata.keys():
        if len(rnadata[i]) < 100:  # or i%8 == 3
            data_cc[i] = []
            continue
        # print('run cca on data:',i)
        if i == largeid[0]:
            cca.fit(ppdata[largeid[1]], ppdata[i] )
            cca_loading = cca.y_weights_
        else:
            cca.fit(ppdata[i], mean_vector)
            cca_loading = cca.x_weights_
        # if i not in usedata[bestk]:# i not in ppdata.keys():
        #     cca.fit(ppdata[i], mean_vector)
        #     # cor = cca.score(ppdata[i], mean_vector)
        #     # if cor < -50:
        #     #     data_cc[i] = []
        #     #     continue
        #     # print(i+1,cor)
        #     cca_loading = cca.x_loadings_
        #     # print(len(cca_loading),len(rnadata[i]),ppdata[i].shape[1])
        # else:
        #     # for k in range(nrepeat):
        #     #     if i in usedata[k]:
        #     dataid = [j for j in range(len(usedata[bestk])) if usedata[bestk][j]==i][0]
        #     # dataid = [j for j in range(len(usedata[bestk])) if usedata[bestk][j]==i][0]
        #     cca_loading = loadingdict[bestk][dataid]

        #     # print(len(cca_loading),len(rnadata[i]))

        data_cc[i] = pd.DataFrame(data=cca_loading, index=rnadata[i].obs_names, columns=[
                                  'CC_'+str(k+1) for k in range(5)])
        indices = data_cc[i].index
        # indices, flag = outlier_filter(data_cc[i])
        # data_cc[i] = data_cc[i].loc[indices,:]

        # temp = pd.DataFrame(data=temp,index=data_cc[i].loc[indices,:].index,columns=['CC_'+str(k+1) for k in range(5)])

    # print(ppdata.values())
    # model = learn(list(ppdata.values()), k=5, epochs=10, verbose=False)
    # cca_loading = model.Qs
    # print(cca_loading)

    # j = 0
    # for i in rnadata.keys():
    #     if i in ppdata.keys():
    #     # if len(rnadata[i]) < 250:
    #     #     continue
    #         data_cc[i] = pd.DataFrame(data=cca_loading[j][:,:5],index=rnadata[i].obs_names,columns=['CC_'+str(k+1) for k in range(5)])
    #         indices = data_cc[i].index
    #         # indices, flag = outlier_filter(data_cc[i])
    #         # data_cc[i] = data_cc[i].loc[indices,:]

    #         temp = normalize(data_cc[i].loc[indices,:],copy=True)
    #         data_cc[i] = pd.DataFrame(data=temp,index=data_cc[i].loc[indices,:].index,columns=['CC_'+str(k+1) for k in range(5)])

    #         j += 1
    #     else:
    #         data_cc[i] = []

    return data_cc

def ReUseCCA(nodelist):
    data_cc = {}
    for i in range(len(nodelist)):
        node = nodelist[i]
        if node is not None and node.embedding is not None:
            if len(node.embedding) != 0 and node.embedding.columns[0] == 'CC_1':
                data_cc[i] = node.embedding
                # print(data_cc[i])
    if len(data_cc) == 0:
        return data_cc, True
    else:
        # print('data_cc',len(data_cc))
        return data_cc, False

def ReClassify(merge_cutoff=0.1, rnadata=None, crossnode=None, adtdata=None):
    train_id = crossnode.modelnode.indices
    # print(train_id)
    w = crossnode.modelnode.artificial_w
    m0 = crossnode.modelnode.embedding[0]
    m1 = crossnode.modelnode.embedding[1]
    nodeind = crossnode.modelnode.ind
    # print(m0,m1)
    crossnode.nodelist = Classify(merge_cutoff, rnadata, m0,m1,w,crossnode.nodelist,train_id,nodeind,adtdata)
    return crossnode

def Classify(merge_cutoff=0.1, rnadata=None, m0=0,m1=0,w=None,nodelist=None,train_id=None,nodeind=None,adtdata=None):
    datanum = len(nodelist)
    for i in range(datanum):
        root = nodelist[i]
        
        # print(i, len(root.indices))
        # print(root.partitions)
        if root == None:
            continue
        # root.indices = root.left_indices.append(root.right_indices) 
        # if i in train_id:
        #     # root.score_dict = feature_scores
        #     # print('1',root.partitions)
        #     root, continue_deep = root_param(root, adtdata[i], best_feature)
        #     root.indices = rnadata[i].obs_names
        #     # print('last data shape',i,rnadata[i].shape)
        #     root.stop = None
        #     root.artificial_w = w
        s = i+1
        if len(root.indices) > 30  and  s == 1 : #(27<s < 45 or ) and int(i%3) == 0  and  s not in train_id 
            # print(s)
            if len(train_id) > 0:
                root.indices = root.indices.intersection(rnadata[i].obs_names)
                rnadata[i] = rnadata[i][root.indices,:]
                adata = rnadata[i].copy()
                
                try:
                    feature = np.dot(adata[:,w.index].X.toarray(), w.values)
                except:
                    feature = np.dot(adata[:,w.index].X, w.values)
                # if s == 3:
                #     feature = smooth(adtdata[i])
                #     feature = feature.loc[root.indices,'CD25'].values
                    

                feature = pd.DataFrame(
                        feature, index=adata.obs_names, columns=['artificial'])

                root.artificial_w = feature
                # root = gm_proba(root, feature, True)
                root, hardclass = gmm_class(root, feature)

                # root = artificial_feature(feature, root, min(merge_cutoff*3,0.4))
                # if s == 6:
                hardclass = True
                print('node', nodeind,'batch', s, 'hard class', hardclass)
                if hardclass:
                    root = FakeFeatureSeparate(
                        feature, root, merge_cutoff, min(m0, m1), max(m0, m1))

                # print( len(root.indices),feature.shape)
                root.indices = rnadata[i].obs_names
                if root.left is not None:
                    root.left.indices = root.left_indices
                else:
                    root.left = BTree(key=('leaf',),indices=root.left_indices)
                if root.right is not None:
                    root.right.indices = root.right_indices
                else:
                    root.right = BTree(key=('leaf',),indices=root.right_indices)

                # print('last data shape',i,rnadata[i].shape)
                plt.hist(feature.loc[root.left_indices,:],bins=100)
                plt.hist(feature.loc[root.right_indices,:],bins=100)
                plt.savefig('../output/figures/'+str(nodeind)+'_'+str(s)+'.png')
                plt.close()
                del(adata)
            # root.stop = 'No common ADT marker'
            nodelist[i] = root
    return nodelist


def CrossSplit(adtdata=None, merge_cutoff=0.1, weight=1, max_k=10, max_ndim=2, bic='bic', 
                marker_set=[], parent_key=set(), rnadata=None, crossnode=None):
    feature_scores, feature_sepnum = {}, {}
    datanum = len(rnadata.keys())

    useADT = True
    if crossnode is not None:
        nodelist_ = crossnode.nodelist

    # print(len(adtdata),len(rnadata))
    if adtdata == {} and datanum > 0:
        runcca = False
        if crossnode is not None:
            adtdata, runcca = ReUseCCA(nodelist_)
        runcca = True
        if runcca :
            if crossnode is not None:
                adtdata = RNApp(rnadata.copy(),nodelist_)
            else:
                adtdata = RNApp(rnadata.copy())
        # print(adtdata[0].shape)
        pcs = [None for i in range(datanum)]
        useADT = False
        # print(adtdata)
    # else:
    #     pcs = RNApca(rnadata.copy())
    if adtdata == {} and crossnode is not None:
        adtdata = RNApp(rnadata.copy(),nodelist_)

    nodelist = []
    if datanum < 2:
        print('==datanum:', datanum)
    for i in range(datanum):
        if crossnode is not None:
            if nodelist_[i] is None:
                rnadata[i] = rnadata[i][False,:]
                adtdata[i] = pd.DataFrame([])
            else:
                nodelist_[i].indices = nodelist_[i].indices.intersection(rnadata[i].obs_names)
                rnadata[i] = rnadata[i][nodelist_[i].indices,:]
                if useADT:
                    adtdata[i] = adtdata[i].loc[nodelist_[i].indices,:]
        if len(marker_set) != 0:
            node = nodelist_[i]
        #### 'len(marker_set) != 0' is used to test the process of retraining when only on training datasets with seperable feature. ###
        if len(marker_set) != 0 and node is not None and node.key not in [('leaf',), ('artificial',)]:
            nodelist.append(node)
            if node.key in feature_scores.keys():
                feature_scores[node.key], feature_sepnum[node.key] = max(
                    node.score_dict[node.key]+0.1, feature_scores[node.key]), feature_sepnum[node.key]+1
            else:
                feature_scores[node.key], feature_sepnum[node.key] = node.score_dict[node.key], 1
            continue
        separable_features, scores_ll = [], []
        if len(rnadata[i]) == 0:
            print('rnadata=none', i)
            nodelist.append(None)
            continue
        root = BTree(('leaf',))
        root.weight = weight[i]
        if crossnode is not None and nodelist_[i] is not None:
            root.left, root.right, root.weight = nodelist_[
                i].left, nodelist_[i].right, weight[i]
            root.ind = crossnode.modelnode.ind
        # print(adtdata[i].columns[0][:2])
        if i in adtdata.keys() and len(adtdata[i]) > 0 and adtdata[i].columns[0][:2] == 'CC':
            root.embedding = adtdata[i]
            # root.indices = adtdata[i].index.values.tolist()
        else:
            root.embedding = []
            # print(adtdata[i].columns[0][:2]=='CC')
            # root.indices = rnadata[i].obs_names.values.tolist()

        root.stop = None
        root.partitions = {}
        # print(i, root.stop, root.partitions)
        root.marker = marker_set
        # print(adtdata.keys())
        if i not in adtdata.keys() or (i in adtdata.keys() and len(adtdata[i]) == 0):
            root.indices = rnadata[i].obs_names.values.tolist()
            nodelist.append(root)
            # if i in list(rnadata.keys()):
            #     root.indices = rnadata[i].obs_names.values.tolist()
            #     nodelist.append(root)
            # print(i+1,len(adtdata[i]))
            # else:
            #     print('=====',root.embedding)
            #     nodelist.append(None)
            continue
        root.indices = adtdata[i].index.values.tolist()
        if useADT:
            data = smooth(adtdata[i].copy())

        else:
            data = adtdata[i]
        val_cnt = value_count(adtdata[i])
        root.val_cnt = val_cnt

        if True:
            root.mean = data.mean()
            root.cov = data.cov()

        else:
            root.mean = data.loc[:, root.marker].mean()
            root.cov = data.loc[:, root.marker].cov()
        # if len(root.indices) < 500:
        #    print(root.indices)

        if data.shape[0] < 100:
            root.all_clustering_dic = _set_small_leaf(data)
            root.stop = 'small size'
            root.left, root.right = None, None
            print(i+1, root.stop)
            nodelist.append(root)
            continue

        unimodal = GaussianMixture(1, covariance_type='full').fit(data)
        root.ll = root.weight * unimodal.lower_bound_
        root.bic = unimodal.bic(data)

        separable_features, bipartitions, scores_ll, all_clustering_dic, rescan =  HiScanFeatures(data,
                         root, merge_cutoff, max_k, max_ndim, bic, parent_key, marker_set)

        # rescan_features=list(set(scanfeatures)-set(parent_key))
        if rescan:  # save separable features if have rescaned
            root.rescan = True
            # print('root.rescan:',root.rescan)
            # root.separable_features = separable_features
        classifier = True
        if len(scores_ll) == 0:
            root.stop = 'no separable features'
            print(i+1,root.stop)
            root.all_clustering_dic = all_clustering_dic
            root.left, root.right = None, None
            nodelist.append(root)
            continue
        idx_best = np.argmax(scores_ll)
        # print(i+1, idx_best)
        if scores_ll[idx_best] <= -90:
            if scores_ll[idx_best] <= -900:
                root.stop = 'small subpopulation'
            elif scores_ll[idx_best] <= -90:
                root.stop = 'overly discrete features'
            print(i+1, root.stop)
            root.left, root.right = None, None
            root.all_clustering_dic = all_clustering_dic
            root.score_dict = dict(zip(separable_features, scores_ll))
            # for i in range(len(separable_features)):
            #     features = separable_features[i]
            #     if len(separable_features[i]) > 1:
            #         # balance = np.exp(-abs(sum(root.partitions[features])-sum(~root.partitions[features]))/len(root.indices)) 
            #         scores_ll[i] = scores_ll[i] 
            nodelist.append(root)
            continue

        while(scores_ll[idx_best] > -datanum):
            # if len(rnadata[i]) < 1000 or len(separable_features) <= 1:
            #     classifier = True
            #     break
            # print(scores_ll)

            best_feature = separable_features[idx_best]
            best_partition = bipartitions[best_feature]
            if sum(best_partition) < 30 or sum(~best_partition) < 30:
                scores_ll[idx_best] = -200  # -200 means 'partition small size'
                idx_best = np.argmax(scores_ll)
                continue

            # if len(best_feature) > 1:
            #     break
            # print(datarna.obs_names[:10])
            # print(separable_features[idx_best])
            if best_feature == ('CC_1',):
                break
            if best_feature in feature_scores.keys():
                # If other datasets have used this feature with higher score, skip test
                ###### Special s1 #####
                if feature_scores[best_feature] > scores_ll[idx_best]+0.5:
                    break
                best_score = np.max(list(feature_scores.values()))
                # If current highest score higher than this feature score, skip test
                if best_score > scores_ll[idx_best]:
                    break

            # classifier, overlap = LDA_test(rnadata[i][root.indices,:].copy(), best_partition)
            classifier = True
            

            if classifier == False:
                scores_ll[idx_best] = -datanum-overlap
            else:
                best_feature = separable_features[idx_best]
                break

            idx_best = np.argmax(scores_ll)

        root.score_dict = dict(zip(separable_features, scores_ll))
        # scores_ll = list(np.array(scores_ll) + 5)

        for j in range(len(separable_features)):
            feature = separable_features[j]
            if scores_ll[j] > -10:
                if feature not in feature_scores.keys():
                    feature_scores[feature] = scores_ll[j] + 0.1
                    feature_sepnum[feature] = 1
                else:
                    feature_sepnum[feature] += 1
                    feature_scores[feature] = max( scores_ll[j]+0.1*feature_sepnum[feature],feature_scores[feature])
                    # if 'CC_2' in feature :
                    #     feature_scores[feature] -= 2

        # if 'CD4-1' in feature_scores.keys():
        #     print(feature_scores)
        root.all_clustering_dic = all_clustering_dic
        root.partitions = bipartitions
        nodelist.append(root)
        print('Dataset:',i+1, root.score_dict)###1111

    if (len(feature_scores) == 1 and len(list(feature_scores.keys())[0]) == 1):
        # if list(feature_sepnum.values())[0] <2:
        feature_scores[list(feature_scores.keys())[0]] = -10
    
    for feature in feature_scores.keys():
        if len(feature) == 1:
            feature_scores[feature] = feature_scores[feature] + feature_sepnum[feature]/10
        else:
            feature_scores[feature] = feature_scores[feature] + feature_sepnum[feature]/5

    # print(feature_scores)
    while len(feature_sepnum) > 0:
        idx_best = np.argmax(list(feature_scores.values()))
        # idx_best = np.argmax(list(feature_sepnum.values()))
        best_feature = list(feature_scores.keys())[idx_best]
        # if best_feature == ('CC_3',) or best_feature == ('CC_1',):
        #     feature_scores[best_feature] -= 20
        # if len(marker_set) == 2:
        #     if len(best_feature) != 2:
        #         feature_scores[best_feature] -= 10
        if feature_sepnum[best_feature] >= 1 and feature_scores[best_feature] >= -10: 
        # The best feature should be separable in at least two datasets.
            break
        else:
            feature_scores[best_feature] -= 10
        # if max(list(feature_sepnum.values())) > 1:
        #     # The best feature should be separable in at least two datasets.
        #     # or feature_scores[best_feature] > 0:
        #     if feature_sepnum[best_feature] > 2: 
        #         break
        #     else:
        #         feature_scores[best_feature] -= 10
        # else:
        #     feature_scores[best_feature] -= 10
        # if len(marker_set) == 2:
        #     if len(best_feature) != 2:
        #         feature_scores[best_feature] -= 10
        
        # if len(best_feature) < 2:
        #     feature_scores[best_feature] -= 20

        if max(list(feature_scores.values())) <= -10:
            break

    # print('datanum',datanum,feature_scores)
    # or (feature_sepnum[best_feature]<2):
    if len(feature_scores) == 0 or max(list(feature_scores.values())) <= -10:
        # print('LDA test not passed',feature_scores,max(list(feature_scores.values())),feature_sepnum[best_feature])
        if len(marker_set) != 0:
            return crossnode
        crossnode = finish(nodelist, rnadata, feature_scores,
                           separable_features, scores_ll)
        return crossnode

    idx_best = np.argmax(list(feature_scores.values()))
    best_feature = list(feature_scores.keys())[idx_best]
    print('\n','best:', best_feature, feature_sepnum[best_feature],'with datasets', 'in   ', feature_scores)

    # print(nodelist)
    genes = []
    train_id = []
    for i in range(datanum):
        root = nodelist[i]
        if root == None :
            continue
        # print(i,':',root.partitions.keys())
        if root.stop == None and best_feature in root.partitions.keys() and root.score_dict[best_feature] > -10:
            # print(i,'train')
            partition = root.partitions[best_feature]
            # print('rna gene',len(rnadata[i].var_names))
            deg = gene_selection(rnadata[i][root.indices, :].copy(), partition)
            # print('rna gene',len(rnadata[i].var_names))
            genes += deg
            train_id.append(i)

    for i in range(datanum):
        if len(rnadata[i]) > 0:
            genes = list(set(genes) & set(rnadata[i].var_names))
    genes = list(set(genes))
    # print('ngene',len(genes))
    traindata = {}
    # print(rnadata)
    for id in train_id:
        root = nodelist[id]
        rnadata[id].var_names_make_unique()
        # print('rna gene',len(rnadata[i].var_names))
        adata = rnadata[id][nodelist[id].indices, :]

        traindata[id] = adata[:,genes]
        traindata[id].obs['label'] = 0
        
        flag = True
        # nodelist[id].partitions[best_feature], flag = knn(
        #     adtdata[id].loc[:, best_feature], pcs[id], root.partitions[best_feature])

        if flag != True:
            nodelist[id].stop = 'KNN low confidence'
            train_id.remove(id)
            continue

        partition = nodelist[id].partitions[best_feature]
        p1_mean = adtdata[id].loc[partition, best_feature].mean().values
        p2_mean = adtdata[id].loc[~partition, best_feature].mean().values
        if len(best_feature) == 1:
            flag = p1_mean < p2_mean
        else:
            dist = (p1_mean-p2_mean)**2
            dim = 1  # np.argmax(dist)
            p1_cosine = p1_mean[dim]/np.sqrt(sum(p1_mean**2))
            p2_cosine = p2_mean[dim]/np.sqrt(sum(p2_mean**2))
            flag = p1_cosine < p2_cosine
        if flag:
            traindata[id].obs['label'].loc[~partition] = 1
            # root.left_indices = adtdata.loc[partition,:].index
            # root.right_indices = adtdata.loc[~partition,:].index
        else:
            traindata[id].obs['label'].loc[partition] = 1
        # print(len(partition),sum(partition))
    if len(train_id) > 0:
        # print(traindata)
        w, m0, m1, loss = train_classifier(traindata)

        del(traindata)
        w = pd.Series(w.detach().numpy().reshape(-1), index=genes)
    else:
        crossnode = finish(nodelist, rnadata, feature_scores,
                           separable_features, scores_ll)
        return crossnode

    continue_deep = False
    for i in range(datanum):
        root = nodelist[i]
        # print(i, len(root.indices))
        # print(root.partitions)
        if root == None:
            continue
        if i in train_id:
            # if i in adtdata.keys() and adtdata[i].columns[0][:2] == 'CC' and len(adtdata[i]) != len(rnadata[i]):
            #     knn(adtdata)
            # root.score_dict = feature_scores
            # print('1',root.partitions)

            root, continue_deep = root_param(root, adtdata[i], best_feature)
            root.indices = rnadata[i].obs_names
            # print('last data shape',i,rnadata[i].shape)
            root.stop = None
            root.artificial_w = w

        if i not in train_id or (
                i in adtdata.keys() and adtdata[i].columns[0][:2] == 'CC' and len(adtdata[i]) != len(rnadata[i])):
            if len(train_id) > 0:
                adata = rnadata[i].copy()
                # x = adata[:31,:20].X.toarray().flatten()
                # if len(list(set(x))) <= 2:
                #     x = adata[:37,:60].X.toarray().flatten()
                #     if len(list(set(x))) <= 2:
                #         x = adata[:37,:600].X.toarray().flatten()
                # x = list(set(x))
                # if np.sort(x)[-2] != int(np.sort(x)[-2]):
                #     sc.pp.normalize_total(adata, target_sum=1e4)
                #     sc.pp.log1p(adata)
                # sc.pp.scale(adata, max_value=10, zero_center=False)
                try:
                    feature = np.dot(adata[:,genes].X.toarray(), w.values)
                except:
                    feature = np.dot(adata[:,genes].X, w.values)
                if i in train_id:
                    if len(best_feature) > 1:
                        feature = pd.DataFrame(
                            feature, index=adata.obs_names, columns=[best_feature])

                    else:
                        feature = pd.DataFrame(
                            feature, index=adata.obs_names, columns=[best_feature[0]])
                else:
                    feature = pd.DataFrame(
                        feature, index=adata.obs_names, columns=['artificial'])

                # root = artificial_feature(feature, root, min(merge_cutoff*3,0.4))
                root = FakeFeatureSeparate(
                    feature, root, merge_cutoff, min(m0, m1), max(m0, m1))
                if i in train_id:
                    root.key = best_feature
                root.artificial_w = feature
                root.indices = rnadata[i].obs_names
                # print('last data shape',i,rnadata[i].shape)
                del(adata)
            # root.stop = 'No common ADT marker'
        nodelist[i] = root

    if len(marker_set) != 0:
        ml, mr = crossnode.modelnode.left, crossnode.modelnode.right
    crossnode = CrossNode(nodelist)
    score_dict = dict(zip(separable_features, scores_ll))
    crossnode = GenModelNode(crossnode, best_feature, w, score_dict=score_dict, outputmean=[
                             min(m0, m1), max(m0, m1)], loss=loss)
    if len(marker_set) != 0:
        print(feature_sepnum)
        crossnode.modelnode.left, crossnode.modelnode.right = ml, mr

        return crossnode

    if continue_deep:
        # print(best_feature,)
        leftadt, rightadt, leftrna, rightrna, lw, rw = {}, {}, {}, {}, {}, {}
        # for i in range(datanum):
        if len(rnadata.keys()) == 0:
            return crossnode
        # print(rnadata.keys())
        for i in list(rnadata.keys()):
            leftadt[i], rightadt[i], leftrna[i], rightrna[i], lw[i], rw[i] = [
                ], [], [], [], [], []
            if nodelist[i] is None or nodelist[i].key == ('leaf',):
                continue
            if useADT and len(adtdata[i]) != 0:
                leftadt[i] = adtdata[i].loc[nodelist[i].left_indices, :]
                rightadt[i] = adtdata[i].loc[nodelist[i].right_indices, :]
            else:
                # print(i,nodelist[i].key, nodelist[i].stop)
                leftadt[i] = []
                rightadt[i] = []
            adtdata[i] = []
            leftrna[i] = rnadata[i][nodelist[i].left_indices, :]
            rightrna[i] = rnadata[i][nodelist[i].right_indices, :]
            rnadata[i] = []
            lw[i], rw[i] = nodelist[i].w_l, nodelist[i].w_r
        if useADT == False:
            # parent_key = crossnode.modelnode.key
            merge_cutoff = 0
            crossnode.left = CrossSplit(
                {}, merge_cutoff, lw, max_k, max_ndim, bic, marker_set, parent_key, leftrna)
            crossnode.right = CrossSplit(
                {}, merge_cutoff, rw, max_k, max_ndim, bic, marker_set, parent_key, rightrna)
        else:
            merge_cutoff = 0
            print(set(list(best_feature)), list(parent_key))
            parent_key = set(list(best_feature) + list(parent_key))
            crossnode.left = CrossSplit(
                leftadt, merge_cutoff, lw, max_k, max_ndim, bic, marker_set, parent_key, leftrna)
            crossnode.right = CrossSplit(
                rightadt, merge_cutoff, rw, max_k, max_ndim, bic, marker_set, parent_key, rightrna)
    return crossnode


import torch.optim as optim
import torch.nn.functional as F

def train_classifier(rnadata):
    if len(rnadata[list(rnadata.keys())[0]]) > 10000:
        batch_num = 32
    elif len(rnadata[list(rnadata.keys())[0]]) > 6000:
        batch_num = 16
    else:
        batch_num = 4
    print('batch num:',batch_num)
    class scRNAdata(Dataset):
        def __init__(self, adata):
            # x = adata[:31,:20].X.toarray().flatten()
            # if len(list(set(x))) <= 2:
            #     x = adata[:37,:60].X.toarray().flatten()
            #     if len(list(set(x))) <= 2:
            #         x = adata[:37,:600].X.toarray().flatten()
            # x = list(set(x))
            # if np.sort(x)[-2] != int(np.sort(x)[-2]):
            #     sc.pp.normalize_total(adata, target_sum=1e4)
            #     sc.pp.log1p(adata)
            # sc.pp.scale(adata, max_value=10, zero_center=False)
            
            try:
                self.data = adata.X.toarray()
            except:
                self.data = adata.X
            self.label = adata.obs['label'].values
            # self.index = (str(id)+ np.array(adata.obs_names)).tolist()

        def __getitem__(self, index):
            data = torch.tensor(self.data[index], dtype=torch.float)
            labels = torch.tensor(self.label[index], dtype=torch.float)
            return data, labels

        def __len__(self):
            return len(self.data)

    dataload = []
    for i in rnadata.keys():
        data = scRNAdata(rnadata[i].copy())
        dataload.append(DataLoader(data, batch_size=int(
            rnadata[i].shape[0]/batch_num), shuffle=True))
    #     dataload = ConcatDataset([dataload, data])
    # dataload = DataLoader(dataload, batch_size=int(len(dataload)/batch_num), shuffle=True)

    # import torch.nn.functional as F
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device} device")

    class Net(nn.Module):
        def __init__(self, ngenes, W0):
            super(Net, self).__init__()
            self.fc = nn.Linear(ngenes, 1, bias=True)
            # print('self',self.fc.weight.dtype)
            self.fc.weight = nn.Parameter(W0)
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax()

        def forward(self, x):
            x = self.fc(x)
            # x1 = self.sigmoid(x)
            return x

    lda = LinearDiscriminantAnalysis()
    # i = np.argmin([len(d) for d in rnadata.values()])
    # lda.fit(rnadata[list(rnadata.keys())[i]].X.toarray(),rnadata[list(rnadata.keys())[i]].obs['label'].values)
    lda.fit(data.data, data.label)

    w0 = torch.tensor(lda.coef_, dtype=torch.float32)
    # c0 = w0.detach().numpy()
    # print(len(data.label))
    # print(sum(data.label),w0[0,0])

    model = Net(rnadata[list(rnadata.keys())[0]].shape[1], w0).to(device)
    # print(model)

    class CrossLDAloss(nn.Module):
        def __init__(self):
            super(CrossLDAloss, self).__init__()

        def submtx(self, output, label):
            mask = label == 0
            # print(mask)
            return output[mask], output[~mask]

        def mean(self, y0, y1):
            # print(y0, y1)
            miu0, miu1 = torch.mean(y0), torch.mean(y1)

            if len(y0) == 0:
                print('miu',miu1.item())
                return torch.tensor(0), miu1
            elif len(y1) == 0:
                print('miu',miu0.item())
                return miu0, torch.tensor(0)
            # print('miu',miu0,miu1)
            return miu0, miu1
 
        def WithinDist(self, y, miu):
            if len(y) <= 2:
                return 0.0001
            # distmtx = torch.dot(y-miu, y-miu)
            norm = (miu.item() ** 2) #* (len(y))
            # print('Sw',norm, len(y))
            # return distmtx / norm
            return y.var() / norm

        def BetweenDist(self, miu0, miu1):
            norm = (miu0.item()**2 + miu1.item()**2) / 2
            # norm = (miu0**2 + miu1**2) / 2
            # print('Sb',norm)
            return ((miu1 - miu0) ** 2) / norm  # + (miu1-miu0)*0.01

        def forward(self, output: torch.Tensor, label: torch.Tensor, W: torch.Tensor):
            y0, y1 = self.submtx(output, label)
            miu0, miu1 = self.mean(y0, y1)
            # print(self.WithinDist(y0, miu0) , self.WithinDist(y1, miu1))
            Sw = self.WithinDist(y0, miu0) + self.WithinDist(y1, miu1)
            Sb = self.BetweenDist(miu0, miu1)
            wL1 = abs(W).mean()
            wL2 = (W ** 2).mean()/2
            outnorm = torch.norm(output, p=2)*0.01
            # wL1 = torch.norm(W, p=1, dim=0)
            # wL2 = torch.norm(W, p=2, dim=0)
            # print(label)
            loss = torch.relu(10-Sb/Sw)*10 # + (wL2+wL1*10)*0.001  + torch.relu(3-Sb)*0.05  + torch.relu(Sw-2)*0.05
            loss = loss  + outnorm  + (miu0**2 + miu1**2)*0.05 + wL1*0.1# + torch.relu(-miu1)*0.01  + torch.relu(miu0)*0.01
            return loss, miu0, miu1 ,  [Sw.item(),Sb.item(),outnorm.item(),wL1.item(),miu0.item(),miu1.item()]

    loss_fn1 = CrossLDAloss()
    loss_fn2 = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    def train(dataloader, model, loss_fn1, loss_fn2, optimizer):

        model.train()

        current = 0
        # for batch, data in enumerate(dataloader):
        for batch in range(batch_num):
            loss, size, miu0, miu1 = 0, 0, torch.empty(
                len(dataloader)), torch.empty(len(dataloader))
            for i in range(len(dataloader)):
                (X, y) = next(iter(dataloader[i]))
                # (X,y) = dataloader[i]
                X, y = X.to(device), y.to(device)
                cont = False
                for cls in [0,1]:
                    mask = (y == cls)
                    if mask.sum() == 0:  
                        print('Num of class',cls,'is 0')
                        cont = True
                if cont:
                    continue
                feature = model(X).squeeze(-1)
                loss1, miu0[i], miu1[i], losslist = loss_fn1(feature, y, model.fc.weight)

                loss2 = loss_fn2(feature, y)
                # if i == 1:
                #     loss1 = 4*loss1
                loss += loss1 + loss2
           

            # (X, y) = data
            # # print(y)
            # feature = model(X).squeeze(-1)
            # loss = loss_fn1(feature, y, model.fc.weight)
            # loss += loss_fn2(feature, y)*20
            if len(dataloader) > 1:
                loss += (miu0.var() + miu1.var()) *20
                # print(loss.item(),miu0.var(),miu1.var())
            # if batch % 100 == 1 :
            #     print(batch,losslist,miu0.var().item())
            # size += len(dataloader[i].dataset)
            current += len(X)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # if batch % 9 == 0:
            #     loss = loss.item()
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]",i+1,'train data')

        return loss, miu0.mean().item(), miu1.mean().item()

    epochs = 800
    loss0 = 10000
    for t in range(epochs):
        # print(f"Epoch {t+1}\n-------------------------------")
        loss, mean0, mean1 = train(
            dataload, model, loss_fn1, loss_fn2, optimizer)
        if (t+1) % 20 == 0:
            print(f"Epoch {t+1} loss: {loss:>7f}")
        if abs(loss-loss0) < 0.1 and loss < 6:
            break
        loss0 = loss
        # scheduler.step()
    weight = model.fc.weight

    weight = F.normalize(weight)
    # weight = model.module.features[0].weight
    # print(weight.shape))
    # print(w0[0,0],weight[0,0])
    return weight, mean0, mean1, loss



def LearnPseudoMaker(rnadata):
    if len(rnadata[list(rnadata.keys())[0]]) > 10000:
        batch_num = 32
    elif len(rnadata[list(rnadata.keys())[0]]) > 6000:
        batch_num = 16
    else:
        batch_num = 4
    print('batch num:',batch_num)
    class scRNAdata(Dataset):
        def __init__(self, adata):            
            try:
                self.data = adata.X.toarray()
            except:
                self.data = adata.X
            self.marker = adata.obsm['marker']
            # self.label = adata.obs['label'].values
            self.probs = adata.obsm['probs'].values.astype(float)
            # print(self.probs)

            # self.index = (str(id)+ np.array(adata.obs_names)).tolist()

        def __getitem__(self, index):
            data = torch.tensor(self.data[index], dtype=torch.float)
            marker = torch.tensor(self.marker[index], dtype=torch.float)
            # label = torch.tensor(self.label[index], dtype=torch.float)
            probs = torch.tensor(self.probs[index], dtype=torch.float)

            return data, marker, probs

        def __len__(self):
            return len(self.data)

    dataload = []
    for i in rnadata.keys():
        data = scRNAdata(rnadata[i].copy())
        dataload.append(DataLoader(data, batch_size=int(
            rnadata[i].shape[0]/batch_num), shuffle=True))
    #     dataload = ConcatDataset([dataload, data])
    # dataload = DataLoader(dataload, batch_size=int(len(dataload)/batch_num), shuffle=True)

    # import torch.nn.functional as F
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device} device")

    class Net(nn.Module):
        def __init__(self, ngenes, W0, datanum):
            super(Net, self).__init__()
            self.fc = nn.Linear(ngenes, 3, bias=True)
            self.fc.weight = nn.Parameter(torch.cat([W0,torch.randn(ngenes,2).T],dim=0))
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax()
            self.deltaW = nn.Parameter(torch.zeros(datanum, ngenes, 1))
            # self.h_head = nn.Linear(2,1)
            # self.probs_head = nn.Linear(2,2)

        def forward(self, x, dataid):
            # print(x.shape)
            # if x.dim() == 1:
            #     x = x.unsqueeze(0)
            deltah = torch.matmul(x, self.deltaW[dataid].squeeze(-1).t()).squeeze(-1)
            output = self.fc(x)
            # print(output.shape)
            h = output[:,0] + deltah
            logits = output[:,1:]
            probs = self.softmax(logits)

            # x1 = self.sigmoid(x)
            return h.unsqueeze(-1), probs

    cca = PLSRegression(n_components=1)
    # print(data.data)
    cca.fit(data.data, data.marker)
    # print(cca.x_weights_.T.tolist())
    Winit = cca.x_weights_.T.tolist()

    # lda = LinearDiscriminantAnalysis()
    # i = np.argmin([len(d) for d in rnadata.values()])
    # lda.fit(rnadata[list(rnadata.keys())[i]].X.toarray(),rnadata[list(rnadata.keys())[i]].obs['label'].values)
    # lda.fit(data.data, data.label)

    Winit = torch.tensor(Winit, dtype=torch.float32)
    # print(Winit.shape)
    # c0 = w0.detach().numpy()
    # print(len(data.label))
    # print(sum(data.label),w0[0,0])

    model = Net(rnadata[list(rnadata.keys())[0]].shape[1], Winit, len(rnadata)).to(device)
    # print(model)

    class LowDimClustering(nn.Module):
        def __init__(self):
            super(LowDimClustering, self).__init__()
        
        def mean(self, h, probs):
            center = torch.sum(probs * h, dim=0) / (torch.sum(probs, dim=0) + 1e-8)
            return center

        def distance(self, h, h_central):
            d = torch.sum((h-h_central)**2)
            # print('distance:',d)
            return d #/ ((h_central**2).detach()+1e-8)

        def correlation(self, h, m):
            h_2d = h.unsqueeze(1) if h.dim() == 1 else h  # [N, 1]
            m_2d = m.unsqueeze(1) if m.dim() == 1 else m  # [N, 1] 或 [N, C]

            # 中心化
            h_centered = h_2d - h_2d.mean(dim=0, keepdim=True)
            m_centered = m_2d - m_2d.mean(dim=0, keepdim=True)

            # 计算相关系数
            covariance = (h_centered * m_centered).mean(dim=0)  # [1] 或 [C]
            h_std = h_centered.std(dim=0)  # [1] 或 [C]
            m_std = m_centered.std(dim=0)  # [1] 或 [C]

            correlation = covariance / (h_std * m_std + 1e-8)
            # print('corr:',correlation)
            return (correlation).mean()

        def probability(self, h, center):
            temperature = 0.1
            center = center.unsqueeze(0)
            distance = torch.cdist(h, center)
            logits = -distance/temperature
            probs = torch.softmax(logits, dim=1)
            return probs

        def gmm(self, h, h_center):
            var = torch.var(h, dim=0, unbiased=True)
            mahalanobis = torch.sum((h-h_center)**2 / (var + 1e-8))
            log_det = torch.sum(torch.log(var+1e-8))

            log_likelihood = -0.5 * (
                    torch.log(torch.tensor(2 * torch.pi)) + 
                    log_det + mahalanobis
                )
            # component_log_likelyhood = (
            #     gaussian_log_likelihood.sum() + 
            #     h.shape[0]*torch.log
            # )
            return log_likelihood.sum()


        def forward(self, h: torch.Tensor, probs: torch.Tensor, m: torch.Tensor, W: torch.Tensor):
            
            center = (probs*h).sum(0) / (probs.sum(0)+1e-8)
            distm = probs * ((h-center.unsqueeze(0))**2)
            variance = torch.var(distm, dim=0).sum() / (torch.var(h).detach()+1e-8)
            distance = torch.sum(distm, dim=1)
            l_classify = torch.mean(distance) / (torch.var(h).detach()+1e-8)
            if m.sum() == 0:
                l_correlation = 0
            else:
                l_correlation = torch.relu(0.81-self.correlation(h, m[:,0])**2)
            prob_entropy = -torch.sum(probs*torch.log(probs+1e-8),dim=1).mean()
            type_entropy = -torch.mean(probs*torch.log(probs+1e-8),dim=0).mean()
            entropy = prob_entropy - type_entropy*0.5

            # print(entropy,probs.shape)

            wL1 = abs(W).mean()
            h_norm = torch.norm(h, p=2)
            # print(l_classify.shape, l_correlation.shape, center.sum())
            # l_correlation = 0
            l = l_classify + l_correlation*20 + torch.relu(1-center.var())*10 + variance*0.01 + entropy + \
                 center.sum()*0.001 + wL1*0.01 + h_norm*0.001

            return l, center, [probs.mean(0).detach(), l_correlation.detach(), center.detach(),   l_classify.item(), entropy.item(), center.var().item()]


    loss_fn1 = LowDimClustering()
    loss_fn2 = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    def train(dataloader, model, loss_fn1, loss_fn2, optimizer):

        model.train()

        current = 0
        # for batch, data in enumerate(dataloader):
        for batch in range(batch_num):
            loss, size, miu0, miu1 = 0, 0, torch.empty(
                len(dataloader)), torch.empty(len(dataloader))
            for i in range(len(dataloader)):
                (X, y, p) = next(iter(dataloader[i]))
                # (X,y) = dataloader[i]
                X, y, p = X.to(device), y.to(device), p.to(device)
                # cont = False
                # for cls in [0,1]:
                #     mask = (y == cls)
                #     if mask.sum() == 0:  
                #         print('Num of class',cls,'is 0')
                #         cont = True
                # if cont:
                #     continue
                h, probs = model(X,dataid=i)

                loss1, center, losslist = loss_fn1(h, probs, y, model.fc.weight[:,0])
                miu0[i], miu1[i] = center[0], center[1]
                loss1 = loss1 + torch.norm(model.deltaW[i], p=2)*10
                # loss2 = loss_fn2(feature, y)
                # if i == 1:
                #     loss1 = 4*loss1
                loss += loss1 
           

            # (X, y) = data
            # # print(y)
            # feature = model(X).squeeze(-1)
            # loss = loss_fn1(feature, y, model.fc.weight)
            # loss += loss_fn2(feature, y)*20
            if len(dataloader) > 1:
                loss += (miu0.var() + miu1.var()) * 10
                # print(loss.item(),miu0.var(),miu1.var())
            # if batch % 100 == 1 :
            #     print(loss.item(),losslist,miu0.var().item())
            # size += len(dataloader[i].dataset)
            current += len(X)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # if batch % 9 == 0:
            #     loss = loss.item()
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]",i+1,'train data')

        return loss, miu0.mean().item(), miu1.mean().item()

    epochs = 600
    loss0 = 10000
    for t in range(epochs):
        # print(f"Epoch {t+1}\n-------------------------------")
        loss, mean0, mean1 = train(
            dataload, model, loss_fn1, loss_fn2, optimizer)
        if (t+1) % 20 == 0:
            print(f"Epoch {t+1} loss: {loss:>7f}")
        if abs(loss-loss0) < 0.05 and loss < 0.01:
            break
        loss0 = loss
        # scheduler.step()
    weight = model.fc.weight[0,:]
    deltaW = model.deltaW
    probs = {}
    for i in range(len(rnadata)):
        probs[i] =  scRNAdata(rnadata[i].copy()).data.dot(model.fc.weight[1:,:].detach().numpy().T) 

    # print(weight.shape)
    # weight = F.normalize(weight)
    # weight = model.module.features[0].weight
    # print(weight.shape))
    # print(w0[0,0],weight[0,0])
    return weight, mean0, mean1, loss, deltaW, probs


def cca_gene_selection(adtdata, rnadata, nodelist, feature):
    allgenes,degenes = [],[]
    for i in rnadata.keys():
        node = nodelist[i]
        if node is None or node.key == ('artificial',) :
            # print(node, node.key)
            continue
        if feature[0][:2] == 'CC':
            # print(node.embedding.columns)
            y = node.embedding.loc[:,feature]
        else:
            y = adtdata[i].loc[:,feature]
        indices = node.indices.intersection(y.index)
        y = y.loc[indices,:]
        rna = rnadata[i][indices,:]
        sc.pp.highly_variable_genes(rna,n_top_genes=500)
        rna = rna[:,rna.var.highly_variable]
        x = rna.X.toarray()

        cca = PLSRegression(n_components=1)
        # print(x.shape, y.shape)
        cca.fit(x, y)
        # print(cca.x_weights_.T.tolist())
        loading = pd.Series(cca.x_weights_.T.tolist()[0], index=rna.var_names)
        genes = list(loading.nlargest(50).index )+list(loading.nsmallest(50).index)
        allgenes.extend(genes)

        if len(node.left_indices)<2 or len(node.right_indices)<2:
            continue
        rna.obs['split'] = str(0)
        rna.obs['split'].loc[node.right_indices] = 1
        rna.obs['split'] = pd.Categorical(rna.obs['split'])
        sc.tl.rank_genes_groups(rna, groupby='split',
                            method='t-test', n_genes=200)
        DE_genes = pd.DataFrame(
            rna.uns['rank_genes_groups']['names'][:50])
        degenes.extend(list(set(list(DE_genes.loc[:, '0'])+list(DE_genes.loc[:, '1']))))
    
    allgenes = list(set(allgenes+degenes))
    # allgenes = list(set(degenes))
    print(len(allgenes))
    for i in rnadata.keys():
        allgenes = rnadata[i].var_names.intersection(allgenes)

    
    # print(degenes)
    return allgenes
        


        


def gene_selection(adata_sub, bestpartition):
    # adata_sub = adata[node.indices,:].copy()
    sc.pp.filter_genes(adata_sub, min_cells=3)
    # sc.pp.normalize_total(adata_sub, target_sum=1e4)
    # sc.pp.log1p(adata_sub)
    adata_sub.obs['node_split'] = pd.Series(dtype='object') # pd.Categorical()
    adata_sub.obs['node_split'].loc[bestpartition] = str(0)
    adata_sub.obs['node_split'].loc[~bestpartition] = str(1)
    adata_sub.obs['node_split'] = pd.Categorical( adata_sub.obs['node_split'])
    sc.tl.rank_genes_groups(adata_sub, groupby='node_split',
                            method='t-test', n_genes=4000)
    if len(adata_sub) > 500 and len(adata_sub) < 5000:
        ngenes = 200
    else:
        ngenes = 1000
    DE_genes = pd.DataFrame(
        adata_sub.uns['rank_genes_groups']['names'][:ngenes])
    genes = list(set(list(DE_genes.loc[:, '0'])+list(DE_genes.loc[:, '1'])))
    return genes




def root_param(root, data, best_feature):
    best_partition = root.partitions[best_feature]
    if sum(best_partition) < 30 or sum(~best_partition) < 30:
        root.stop = 'partition small size'
        root.key = ('leaf',)
        # root = notSeparabelContinue(root, key=('partition small size',)+best_feature)
        return root, False
    root.key = best_feature

    # Calculate mean, std and weight for all dimension

    p1_mean = data.loc[best_partition, best_feature].mean()
    p2_mean = data.loc[~best_partition, best_feature].mean()
    p1_cov = data.loc[best_partition, best_feature].cov()
    p2_cov = data.loc[~best_partition, best_feature].cov()
    # if best_feature == ('artificial',):
    #     print(p1_mean, p2_mean)



    flag = True
    if len(p1_mean) == 1:
        flag = p1_mean.values > p2_mean.values
    else:
        dist = (p1_mean-p2_mean)**2
        dim = 0 #np.argmax(dist.values)
        p1_cosine = p1_mean.iloc[dim]/np.sqrt(sum(p1_mean**2))
        p2_cosine = p2_mean.iloc[dim]/np.sqrt(sum(p2_mean**2))
        flag = p1_cosine > p2_cosine
        # flag = p1_mean.iloc[dim] > p2_mean.iloc[dim]

    # if data.index[0][-1] == '3':
    #     if flag:
    #         flag = False
    #     else:
    #         flag = True
    if flag:
        # print(best_partition)
        root.right_indices = data.loc[best_partition, :].index
        root.w_r = sum(best_partition)/len(best_partition)
        root.mean_r = p1_mean
        root.cov_r = p1_cov
        root.left_indices = data.loc[~best_partition, :].index
        root.w_l = sum(~best_partition)/len(best_partition)
        root.mean_l = p2_mean
        root.cov_l = p2_cov
        root.where_dominant = 'right'
    else:
        root.right_indices = data.loc[~best_partition, :].index
        root.w_r = sum(~best_partition)/len(best_partition)
        root.mean_r = p2_mean
        root.cov_r = p2_cov
        root.left_indices = data.loc[best_partition, :].index
        root.w_l = sum(best_partition)/len(best_partition)
        root.mean_l = p1_mean
        root.cov_l = p1_cov
        root.where_dominant = 'left'
    
    refine_edge = False
    if refine_edge:

        left = data.loc[data[best_feature[0]] < root.mean_l.item()].index
        right = data.loc[data[best_feature[0]] > root.mean_r.item()].index

        root.left_indices = list(set(root.left_indices)-set(right)|set(left))
        root.right_indices = list(set(root.right_indices)-set(left)|set(right))

        root.mean_l = data.loc[root.left_indices, best_feature].mean()
        root.mean_r = data.loc[root.right_indices, best_feature].mean()
    
    
    return root, True


def notSeparabelContinue(root, key):
    root.key = key
    root.left_indices, root.right_indices = root.indices, root.indices
    root.mean_l, root.mean_r, root.w_l, root.w_r = 0, 0, 1, 1
    return root
import matplotlib.pyplot as plt

def FakeFeatureSeparate(feature, root, merge_cutoff, m0, m1):
    root.indices = feature.index.values.tolist()
    mean = (m0+m1)/2
    cut0, cut1 = mean - (mean-m0)*(1-merge_cutoff) * \
        0.01, mean + (m1-mean)*(1-merge_cutoff)*0.01
    cut0, cut1 = -12, -12
    # if feature.index[0][-1] == '3':
    #     cut0, cut1 = 2,2
    # if feature.index[0][-1] == '3':
    #     cut0, cut1 = 0.1, 0.1
    print('mean',m0,m1,'cut:',cut0,cut1)

    root.left_indices = feature[feature.iloc[:, 0] < cut0].index
    root.right_indices = feature[feature.iloc[:, 0] > cut1].index
    # if len(root.left_indices) < 30 or len(root.right_indices)<30:
    #     root.stop = 'partition small size'
    #     root.key = ('leaf',)
    #     return root
    root.key = tuple([feature.columns[0]])
    # if root.key == 'artificial':
    #     root.key = ('artificial',)
    root.mean_l = feature.loc[root.left_indices, :].mean()
    root.mean_r = feature.loc[root.right_indices, :].mean()
    # if feature.index[0][-1] == '3':
    #     print('new mean',root.mean_l.values,root.mean_r.values)
    #     print(feature.loc[root.left_indices, :].max().values,feature.loc[root.right_indices, :].min().values)
    #     plt.hist(feature.loc[root.left_indices, :],bins=60)
    #     plt.hist(feature.loc[root.right_indices, :],bins=60)
    #     # plt.show()
    #     # plt.savefig('fig')
    root.w_l, root.w_r = len(
        root.left_indices)/len(root.indices), len(root.right_indices)/len(root.indices),
    return root


def artificial_feature(feature, root, merge_cutoff):
    all_clustering_dic = root.all_clustering_dic
    val = len(np.unique(feature, return_counts=False))
    # print('0',feature.shape)
    val_cnt = pd.DataFrame(data=[val, ], columns=['artificial'])
    dip = diptest.dipstat(np.array(feature.iloc[:, 0]))
    all_clutering = Clustering(
        feature, merge_cutoff, max_k=5, bic='bic', val_cnt=val_cnt, soft=True, dip=dip)
    # print(all_clutering)
    all_clustering_dic[1][('artificial',)] = all_clutering
    if all_clustering_dic[1][('artificial',)]['mp_ncluster'] == 1:
        root.stop = 'Artificial feature not separable'
        # root = notSeparabelContinue(root, key=('not sepaeable', 'aritficial'))
        # root.key = ('artificial',)
        root.left_indices, root.right_indices = root.indices, root.indices
        root.mean_l, root.mean_r, root.w_l, root.w_r = 0, 0, 1, 1
        return root
    else:
        merged_label = all_clustering_dic[1][('artificial',)]['mp_clustering']
        # print('stop',root.stop)
        # print('2',root.partitions)
        root.partitions = {}
        root.partitions[('artificial',)] = np.array(merged_label) == 1
        # print(root.partitions[('artificial',)])
        root.key = ('artificial',)
        root, continue_deep = root_param(
            root, pd.DataFrame(feature), ('artificial',))
        return root


def HiScanFeatures(data, root, merge_cutoff, max_k, max_ndim, bic, parent_key={}, marker_set=None):

    # Try to separate on one dimension

    # key_marker = ['TIGIT','PD-1','CD25']
    # key_marker = ['CD16','CD4-2', 'CD3-2','CD3-1', 'CD19', 'CD4-1', 'CD8', 'CD27', 'CD14', 'CLEC12A', 'CD16', 'CD45RA', 'CD127', 'CD45RO','CD25',  'CD56-1']

    # if 'CD4-2' in data.columns or 'CD4-1' in data.columns:
    #     # key_marker = ['CD3-2', 'CD19','CD14','CD4-2','CD8','CD56-2']# , 'CD45RA', 'CD127'
    #     # 'CD43','CD161','TCR-V-7.2','CD4-2','CD8','CD3-2','CD56-1','CD4-1','CD3-1','CD27', 'CLEC12A', 'CD16','CD19','CD8','CD4-1''CD4-1','CD4-2','CD56-2', 'CD25',
    #     key_marker = ['CD3-1', 'CD3-2', 'CD19','CD14', #'CD4-1','CD4-2','CD8','CD45RA','CD45RO',
    #                    'CLEC12A', 'CD56-1', 'CD56-2']
    # elif 'humanCD44' in data.columns:
    #     key_marker = 'CD16, CD14, CD123, CD1c, CD33, CD235ab, CD34, CD2'
    #     key_marker = key_marker.split(', ')
    #     # key_marker = data.columns
    #     # print(key_marker)
    #     # CD20, CD21, CD73, IgD, IgM, Iglightchainl, IdA, IgG, IgD, IgM, Iglightchainl
    #     # CD62L, CD25, CD127_IL_7Ra, CD20, CD38, CD33, CD22, CD28, CD73, CD57Recombinant, CD95_Fas, CD45RA, CD45RO, CD197_CCR7, KLRG1_MAFA
    #     # 'CD3, CD4, CD8, CD56_NCAM, CD19, CD20, CD27, CD38, CD14, CD123, CD1c, CD33, CD235ab, CD34, TCR_Vd2, TCR_Va7_2, CD161'
    # elif 'CD27' in data.columns:
    #     # 'CD16', 'CD45RA', 'CD127-IL7Ra', 'CD45RO', 'CD69','CD25'
    #     key_marker = ['CD4', 'CD3', 'CD19', 'CD8a', 'CD27', 'CD14', ]

    # else:
    #     key_marker = data.columns
    #     # key_marker = ['CD4', 'CD3', 'CD19', 'CD8a', 'CD14',]
    # if marker_set is not None:
    #     key_marker = marker_set
    key_marker =['CD27','CD28',] #['CLEC12A','CD57','CD61','CD20','CD21']'CD45RA','CD45RO','CD25','CD197' 'CD4','CD8' 'CD16'
    # key_marker = ['CD27','CD185','Galectin-9','Integrin-7'] # ,'TCR-V-9','CD158b','CD158e1','CD196','CD195','CD27','CD28','CD25','CD16','CD69','CCR7','CD28','CD107a','CD195','CX3CR1','CD62L','CD161','CD194', 'CCR10', 'CD185'
    # key_marker = data.columns
    # if root.ind == 0:
    #     key_marker = ['CD3']
    # elif root.ind in range(1,15):
    #     key_marker = ['CD4','CD8','CD14','CD19','CD45RA','CD45RO']
    # else:
    #     key_marker = data.columns
    # key_marker = data.columns.intersection(key_marker)
    # if data.columns[0] != 'CC_1' and len(key_marker) == 0 or data.index[0][-2:]=='s2':# or (data.index[0][-2:]=='s1' and int(data.index[0][-4])>3):
    #     key_marker = data.columns[:2]
    #     if marker_set is not None:
    #         merge_cutoff = 0
    if data.columns[0] == 'CC_1':
        key_marker = data.columns
    # print(key_marker,merge_cutoff,root.indices)
    separable_features, bipartitions, scores, bic_list, all_clustering_dic, rescan = Scan(data,
                          root, merge_cutoff, max_k, max_ndim, bic, parent_key, key_marker, marker_set)
    # print(all_clustering_dic[1][('CD19',)]['filter']) #
    # if len(separable_features)==0:
    #     print('no key markers separable')
    # separable_features, bipartitions, scores, bic_list, all_clustering_dic, rescan = Scan(data,root,merge_cutoff,max_k,max_ndim,bic,parent_key,data.columns.values.tolist())

    # print('1',scores)
    return separable_features, bipartitions, scores, all_clustering_dic, rescan


def Scan(data, root, merge_cutoff, max_k, max_ndim, bic, parent_key={}, scanfeatures=[], marker_set=[]):
    ndim = 1
    all_clustering_dic = {}

    parent_key = {'CD103','CD123','CD2','CD98','CD49d','CD21','IgD'}
    separable_features, bipartitions, scores, bic_list, all_clustering_dic[ndim] = ScoreFeatures(data,
         root, merge_cutoff, max_k, ndim, bic, rescan_features=list(set(scanfeatures)-set(parent_key)))
    # print(all_clustering_dic[1][('CD3',)]['filter'])
    rescan = False
    # if True:
    if len(separable_features) < 1 or  max(scores) <= -90 or len(marker_set) == 2 : # 

        rescan_features = []
        for item in all_clustering_dic[ndim]:
            val = all_clustering_dic[ndim][item]['similarity_stopped']
            # 考虑阈值是否应该随着用户指定调整
            if val < min(merge_cutoff*2, 0.9):  # val > merge_cutoff and
                rescan_features.append(item[0])

        for ndim in range(2, max_ndim+1):
            # Num of feature not enough for hight dimension clustering
            # Add all features <0.5 to assign features(save mean of sebarable features)
            if len(rescan_features) < ndim:
                # Threshold is set to a softer one, when partition of its parents is not well, may cause wrong fragment.
                # Add all features <0.5 to assign features(save mean of sebarable features)
                # separable_features, bipartitions, scores, bic_list, all_clustering_dic[ndim] = ScoreFeatures(data,root,0.5,max_k,len(rescan_features),bic)
                break

            # threshold=0.5 or merge_cutoff?
            rescan_features = list(set(rescan_features))
            separable_features_, bipartitions_, scores_, bic_list_, all_clustering_dic[ndim] = ScoreFeatures(
                data, root, min(merge_cutoff, 0.8), max_k, ndim, bic, rescan_features)
            separable_features.extend(separable_features_)
            bipartitions.update(bipartitions_)
            scores.extend(scores_)
            bic_list.extend(bic_list_)
            if len(separable_features) >= 1:
                break
    # print('2',scores)
    return separable_features, bipartitions, scores, bic_list, all_clustering_dic, rescan


def ScoreFeatures(data, root, merge_cutoff, max_k, ndim, bic, rescan_features=None, separable_features=None, bipartitions=None, scores=None, bic_list=None, all_clustering=None, soft=False):

    F_set = rescan_features
    # if ndim == 2:
    #     print('two dimensional rescan features',len(rescan_features))
    if len(rescan_features) >= data.shape[1]-3:
        random.shuffle(F_set)
    # print('F_set',F_set[:5])
    # if soft:
    #     print('Rescan---F_set:',len(F_set),'clustering:',len(all_clustering.keys()))
    if soft == False:
        separable_features = []
        bipartitions = {}
        scores = []
        bic_list = []
        all_clustering = {}
        count = 0
    else:
        count = data.shape[1] - len(separable_features)

    # for item in itertools.combinations(F_set, ndim):
    #     if len(np.unique(data.loc[:,item].values.tolist()))>300:
    #         continueness = True
    #         break
    stop_flash = False
    # print(F_set)
    # When ndim=1, search one feature each time. When ndim=2, search two features each time.
    for item in itertools.combinations(F_set, ndim):
        item = tuple(sorted(item))
        # print(item)
        x = data.loc[:, item]
        # val = len(np.unique(x.values.tolist()))
        # print(item)
        # x = smooth(x,item, max(val,6))
        # if continueness==False and ndim==1 and val<300 and val>1 and len(x)>10*val:
        #     x = smooth(x,item, min(val,6))
        # data.loc[:,item] = x
        if soft:
            if stop_flash == False:
                all_clustering_temp = Clustering(x, merge_cutoff, max_k, bic, root.val_cnt[list(
                    item)], soft, dip=all_clustering[item]['dip'])
                if all_clustering_temp != None:
                    all_clustering[item] = all_clustering_temp
            else:
                continue
        else:
            if stop_flash == False:
                all_clustering[item] = Clustering(
                    x, merge_cutoff, max_k, bic, root.val_cnt[list(item)], soft)
            else:
                all_clustering[item] = _set_one_component(x)
                all_clustering[item]['filter'] = 'skip'
                continue

    # for item in all_clustering:
        if all_clustering[item]['mp_ncluster'] > 1:
            count = count + 1

            merged_label = all_clustering[item]['mp_clustering']
            labels, counts = np.unique(merged_label, return_counts=True)
            if len(counts) == 1 or np.min(counts) < 10:
                continue

            ll_gain = []  # np.zeros(len(labels))
            bic_mlabels = []
            # Choose the cluster with max ll gain in this subspace
            for mlabel in labels:
                # 可优化，merge后只有两类时只用算一次，因为算两次都是一样的
                assignment = merged_label == mlabel

                # marker_list = data.columns
                # Choose only marker to calculate loglikelyhood
                marker_list = list(item)
                # print('marker_list:',marker_list)

                # print('sum(assignment):',sum(assignment),'len(assignment)',len(assignment))
                if sum(assignment) < 20 or sum(~assignment) < 20:
                    ll_gain.append(-1000)
                    print(sum(assignment),sum(~assignment))
                    continue

                ### Avoid overly discrete features
                if data.columns[0][:2] != 'CC':
                    dist0 = min(data.loc[~assignment, marker_list].max(
                        axis=0) - data.loc[~assignment, marker_list].min(axis=0))
                    dist1 = min(data.loc[assignment, marker_list].max(
                        axis=0) - data.loc[assignment, marker_list].min(axis=0))
                    # print(sum(assignment),dist0, dist1)
                    if (min(dist0, dist1) < np.log1p(4) and len(marker_list) == 1) or min(dist0, dist1) < np.log1p(2):
                        ll_gain.append(-100 + min(dist0, dist1))
                        continue

                gmm1 = GaussianMixture(1, covariance_type='full')
                ll1 = gmm1.fit(data.loc[assignment, marker_list]
                               ).lower_bound_ * sum(assignment)/len(assignment)
                # print(np.array(data.loc[~assignment,marker_list]).shape)
                # proba0 = pd.Series(gmm1.predict_proba(np.array(data.loc[assignment,marker_list])),index=data.loc[assignment,:].index)
                # if proba0[proba0<0.99] < 30:
                #     ll_gain.append(-1000)
                #     continue
                # bic1 = gmm1.bic(data.loc[assignment,marker_list])

                gmm0 = GaussianMixture(1, covariance_type='full')
                ll0 = gmm0.fit(data.loc[~assignment, marker_list]
                               ).lower_bound_ * sum(~assignment)/len(assignment)
                # proba1 = pd.Series(gmm1.predict_proba(np.array(data.loc[~assignment,marker_list])),index=data.loc[~assignment,:].index)
                # if proba1[proba1<0.99] < 30:
                #     ll_gain.append(-1000)
                #     continue
                # bic0 = gmm0.bic(data.loc[~assignment,marker_list])

                gmm_ = GaussianMixture(1, covariance_type='full').fit(
                    data.loc[:, marker_list])
                ll_ = gmm_.lower_bound_

                ll_gain.append(((ll1 + ll0) - ll_)  ) # - abs(sum(assignment)-sum(~assignment))*0.0001 

                # bic_mlabels.append( bic1 + bic0 )            
            best_mlabel_idx = np.argmax(ll_gain)
            # if marker_list == ['CC_1','CC_4']:
            #     print(labels,ll_gain)
            #     best_mlabel_idx = np.argmin(all_clustering[item]['bp_pro'])
            best_mlabel = labels[best_mlabel_idx]

            bipartitions[item] = merged_label == best_mlabel
            # if data.columns[0][:2]!='CC':
            #     bipartitions[item] = knn(data.loc[:,item], pcs, bipartitions[item])
            if True:
            # if soft == False and all_clustering[item]['similarity_stopped'] >= 0.01:
                scores.append(
                    ll_gain[best_mlabel_idx] + 2*(merge_cutoff - all_clustering[item]['similarity_stopped']))
            else:
                if soft == False:
                    count = count - 1
                scores.append(ll_gain[best_mlabel_idx]-10)
            separable_features.append(item)
            # bic_list.append( bic_mlabels[best_mlabel_idx] )


            if count == 20 and soft == False:
                stop_flash = True
            if count == 10:
                stop_flash = True

    rescan_features = [item[0] for item in all_clustering.keys(
    ) if all_clustering[item]['filter'] == 'filted']

    # print('filted',len(rescan_features),'separable',len(scores))

    # if data.columns[0][:2]!='CC' and count < min(20,len(F_set)*0.6) and soft==False and (scores==[] or max(scores)<0) and ndim==1 and (len(F_set)-len(rescan_features))<len(F_set)/3:

    #     separable_features, bipartitions, scores, bic_list, all_clustering = ScoreFeatures(data,root,merge_cutoff,max_k,ndim,bic,
    #                     rescan_features, separable_features, bipartitions, scores, bic_list, all_clustering, soft=True)
    # print('3',scores)
    return separable_features, bipartitions, scores, bic_list, all_clustering

# def KnnAllNode(crossnode):
#     for i in 

def knn(data, pcs, assignment):
    #### Select model cells for knn classifier ####

    lind, rind = data[~assignment].index, data[assignment].index
    postp0, postp1 = postproba(data, lind, rind)
    postp0, postp1 = pd.Series(index=lind, data=postp0), pd.Series(
        index=rind, data=postp1)
    index0, index1 = postp0[postp0 > 0.9].index, postp1[postp1 > 0.9].index
    # Filtering cells with high posterior probability in surface marker space
    print('origin:', sum(~assignment), sum(assignment),
          'filterd:', len(index0), len(index1))
    if len(index0) < 20 or len(index1) < 20:
        return assignment, False

    pcs = pd.concat([pcs, data], axis=1)
    # print(pcs.shape)
    postp0, postp1 = postproba(pcs, index0, index1)
    postp0, postp1 = pd.Series(index=index0, data=postp0), pd.Series(
        index=index1, data=postp1)
    index0, index1 = postp0[postp0 > 0.9].index, postp1[postp1 > 0.9].index
    # Filtering cells with high posterior probability in RNA pca and marker space
    print('filterd:', len(index0), len(index1))
    if len(index0) < 10 or len(index1) < 10 or len(index0)+len(index1) == len(assignment):
        return assignment, False

    # KNN classifier
    label = pd.Series(index=list(index0)+list(index1))
    label.loc[index0], label.loc[index1] = 0, 1
    classifier = knc(n_neighbors=10, weights='distance')
    classifier.fit(pcs.loc[label.index, :], label.values)
    lowindex = list(set(data.index)-set(label.index))
    lowconf = classifier.predict(pcs.loc[lowindex])
    lowconf = pd.Series(index=lowindex, data=lowconf)
    newlabel = pd.concat([lowconf, label], axis=0)
    assignment = newlabel == 1
    print('after knn:', sum(~assignment), sum(assignment))
    if sum(~assignment) < 30 or sum(assignment) < 30:
        return assignment, False
    return assignment, True

def UnassignedKnn(node, modelnode, adtdata, rnadata):
    rnadata = rnadata[node.indices,:]
    feature = pd.DataFrame(index=node.indices,data=rnadata[node.indices,modelnode.artificial_w.index].X.dot(modelnode.artificial_w))
    sc.pp.scale(rnadata)
    sc.pp.pca(rnadata, n_comps=10)
    feature = pd.concat([feature, pd.DataFrame(rnadata.obsm['X_pca'], index=node.indices) ], axis=1) 
    if len(adtdata)!=0 and modelnode.key in adtdata.columns:
        adt = np.apply_along_axis(lambda x: np.log(x+1) - np.mean(np.log(x+1)),0,adtdata.loc[node.indices,modelnode.key])   
        feature = pd.concat([feature, pd.Series(adt,index=node.indices,name=modelnode.key)], axis=1)
    if modelnode.key[0][:2] == 'CC':
        if len(node.embedding.index.intersection(node.indices)) == len(node.indices):
            cca = node.embedding.loc[node.indices, modelnode.key]
            feature = pd.concat([feature, cca], axis=1)

    # KNN classifier
    label = pd.Series(index=list(node.left_indices)+list(node.right_indices))
    label.loc[node.left_indices], label.loc[node.right_indices] = 0, 1
    classifier = knc(n_neighbors=10, weights='distance')
    classifier.fit(feature.loc[label.index, :], label.values)
    lowindex = list(set(feature.index)-set(label.index))
    lowconf = classifier.predict(feature.loc[lowindex])
    lowconf = pd.Series(index=lowindex, data=lowconf)
    node.left_indices = node.left_indices.append(lowconf[lowconf==0].index)
    node.right_indices = node.right_indices.append(lowconf[lowconf==1].index)
    return node




def postproba(data, lind, rind):
    meanl, meanr = data.loc[lind, :].mean(
        axis=0), data.loc[rind, :].mean(axis=0)
    covl, covr = data.loc[lind, :].cov(), data.loc[rind, :].cov()
    p00 = multivariate_normal.pdf(data.loc[lind, :], meanl, covl)
    p11 = multivariate_normal.pdf(data.loc[rind, :], meanr, covr)
    p01 = multivariate_normal.pdf(data.loc[lind, :], meanr, covr)
    p10 = multivariate_normal.pdf(data.loc[rind, :], meanl, covl)
    w = len(lind)/(len(rind) + len(lind))

    proba0 = w*p00/(w*p00 + (1-w)*p01)
    proba1 = (1-w)*p11/(w*p10 + (1-w)*p11)
    return proba0, proba1


def Clustering(x, merge_cutoff, max_k, bic, val_cnt, soft=False, dip=None):

    # if x.columns[0] == 'CD45RA':
    #     print(x.columns[0],val_cnt.values)
    # print(val_cnt, len(val))
    # print(np.array(x))
    if soft == False and x.columns[0][:2] != 'CC':
        if x.shape[1] == 1:
            if val_cnt.values <= min(min(x.shape[0]/30, 50), x.shape[0]):
                clustering = _set_one_component(x)
                clustering['filter'] = 'filted: val_cnt' + \
                    str(val_cnt.values) + '<=' + \
                    str(min(min(x.shape[0]/20, 100), x.shape[0]))
                clustering['dip'] = 0
                return clustering
            dip = diptest.dipstat(np.array(x.iloc[:, 0]))
            # if dip < max((1-merge_cutoff)*0.008, 0.005):
            #     clustering = _set_one_component(x)
            #     clustering['filter'] = 'filted: dip<=' + str(max((1-merge_cutoff)*0.008, 0.005))
            #     clustering['dip'] = dip
            #     return clustering

    if soft == True:
        # if clustering['dip'] == None:
        #     dip = diptest.dipstat(np.array(x.iloc[:,0]))
        #     clustering = _set_one_component(x)
        if x.shape[1] == 1 and (val_cnt.values <= min(min(x.shape[0]/30, 50), x.shape[0]) or dip < max((1-merge_cutoff)*0.002, 0.001)):
            # print(x.columns[0],'second filted')
            if x.columns[0] == 'artificial':
                clustering = _set_one_component(x)
                clustering['filter'] = 'filted'
                return clustering
            else:
                # print(x.columns[0])
                return

    k_bic, _ = BIC(x, max_k, bic)
    # if soft==False:
    # print(x.columns.values,x.shape[0]/30,val_cnt.values,dip)

    if k_bic == 1:
        # if only one component, set values
        if soft and x.columns[0] != 'artificial':
            # print(x.columns[0])
            return
        clustering = _set_one_component(x)
        clustering['filter'] = 'too many components:'+str(k_bic)
    # elif x.shape[1] == 1 and max(val_cnt.values) < min(max(50, len(x)/10), len(x)/2):
    #     bp_gmm = GaussianMixture(k_bic).fit(x)
    #     clustering = merge_bhat(x, bp_gmm, 0)
    #     clustering['filter'] = 'weak variant value: ' + str(val_cnt)
    # elif x.columns[0][:2]!='CC' and ((k_bic>4 and min(val_cnt.values)<200) ) :
    #     if soft and x.columns[0] != 'artificial':
    #         return
    #     bp_gmm = GaussianMixture(int(k_bic/2)).fit(x)
    #     clustering = merge_bhat(x,bp_gmm,merge_cutoff)
    #     clustering['filter'] = 'too many components:'+str(k_bic)
    else:
        # print(val_cnt.index, val_cnt.values)
        bp_gmm = GaussianMixture(k_bic).fit(x)
        clustering = merge_bhat(x, bp_gmm, merge_cutoff)
        clustering['filter'] = 'variant value:' + str(val_cnt)
        # print(merge_cutoff,clustering['mp_ncluster'])
    clustering['dip'] = dip

    return clustering


def LDA_test(adata_sub, bestpartition):
    # adata_sub = adata[node.indices,:].copy()
    # sc.pp.filter_genes(adata_sub, min_cells=3)
    # sc.pp.normalize_total(adata_sub, target_sum=1e4)
    # sc.pp.log1p(adata_sub)
    adata_sub.obs['node_split'] = pd.Categorical(categories=['0','1'])  #pd.Series(dtype='object')
    adata_sub.obs['node_split'].loc[bestpartition] = str(0)
    adata_sub.obs['node_split'].loc[~bestpartition] = str(1)
    sc.tl.rank_genes_groups(adata_sub, groupby='node_split',
                            method='t-test', n_genes=4000)
    if len(adata_sub) > 1000 and len(adata_sub) < 5000:
        ngenes = 200
    else:
        ngenes = 1000
    DE_genes = pd.DataFrame(
        adata_sub.uns['rank_genes_groups']['names'][:ngenes])
    genes = list(set(list(DE_genes.loc[:, '0'])+list(DE_genes.loc[:, '1'])))
    clf = LinearDiscriminantAnalysis()
    rna_new = clf.fit_transform(adata_sub[adata_sub.obs['node_split'].isin(['0', '1']), genes].X.toarray(
    ), adata_sub[adata_sub.obs['node_split'].isin(['0', '1']), :].obs['node_split'])
    rna_new = pd.DataFrame(rna_new, index=adata_sub.obs_names)

    overlap = bhattacharyya_dist(rna_new.loc[bestpartition, :].mean(), rna_new.loc[~bestpartition, :].mean(),
                                 rna_new.loc[bestpartition, :].cov(), rna_new.loc[~bestpartition, :].cov())
    # print(np.exp(-overlap))
    if len(adata_sub) > 2000:
        cutoff = 0.02
    else:
        cutoff = 0.05 * (2-len(adata_sub)/10000)
    overlap = np.exp(-overlap)
    if overlap < cutoff:
        return True, overlap
    else:
        return False, overlap


def bhattacharyya_dist(mu1, mu2, Sigma1, Sigma2):
    Sig = (Sigma1+Sigma2)/2
    ldet_s = np.linalg.det(Sig)
    ldet_s1 = np.linalg.det(Sigma1)
    ldet_s2 = np.linalg.det(Sigma2)
    d1 = distance.mahalanobis(mu1, mu2, np.linalg.inv(Sig))**2/8
    d2 = 0.5*np.log(ldet_s) - 0.25*np.log(ldet_s1) - 0.25*np.log(ldet_s2)
    return d1+d2


def merge_bhat(x, bp_gmm, cutoff):

    clustering = {}
    clustering['bp_ncluster'] = bp_gmm.n_components
    clustering['bp_clustering'] = bp_gmm.predict(x)
    clustering['bp_pro'] = bp_gmm.weights_
    clustering['bp_mean'] = bp_gmm.means_
    clustering['bp_Sigma'] = bp_gmm.covariances_

    #clustering['last_pair_similarity'] = _get_last_pair_similarity_2D(x,bp_gmm)
    gmm = copy.deepcopy(bp_gmm)

    mu = gmm.means_
    Sigma = gmm.covariances_
    weights = list(gmm.weights_)
    posterior = gmm.predict_proba(x)

    current_ncluster = len(mu)
    mergedtonumbers = [int(item) for item in range(current_ncluster)]

    merge_flag = True
    clustering['bhat_dic_track'] = {}
    merge_time = 0

    while current_ncluster > 1 and merge_flag:

        bhat_dic = {}

        for c_pair in itertools.combinations(range(current_ncluster), 2):
            m1 = mu[c_pair[0], :]
            m2 = mu[c_pair[1], :]
            Sigma1 = Sigma[c_pair[0], :, :]
            Sigma2 = Sigma[c_pair[1], :, :]
            bhat_dic[c_pair] = np.exp(-bhattacharyya_dist(m1,
                                      m2, Sigma1, Sigma2))

        clustering['bhat_dic_track'][merge_time] = bhat_dic
        merge_time = merge_time + 1

        max_pair = max(bhat_dic.items(), key=operator.itemgetter(1))[0]
        max_val = bhat_dic[max_pair]

        if max_val > cutoff: # or (max_val<0.05 and len(x)<=1300):
            merged_i, merged_j = max_pair
            # update mergedtonumbers
            for idx, val in enumerate(mergedtonumbers):
                if val == merged_j:
                    mergedtonumbers[idx] = merged_i
                if val > merged_j:
                    mergedtonumbers[idx] = val - 1

            # update parameters
            weights[merged_i] = weights[merged_i] + weights[merged_j]

            posterior[:, merged_i] = posterior[:,
                                               merged_i] + posterior[:, merged_j]

            w = posterior[:, merged_i]/np.sum(posterior[:, merged_i])
            mu[merged_i, :] = np.dot(w, x)  # update

            x_centered = x.apply(lambda xx: xx-mu[merged_i, :], 1)
            Sigma[merged_i, :, :] = np.cov(x_centered.T, aweights=w, bias=1)

            del weights[merged_j]
            #weights = np.delete(weights,merged_j,0)
            mu = np.delete(mu, merged_j, 0)
            Sigma = np.delete(Sigma, merged_j, 0)
            posterior = np.delete(posterior, merged_j, 1)
            current_ncluster = current_ncluster - 1

        else:
            # print(x.columns,max_val)
            merge_flag = False

    clustering['similarity_stopped'] = np.min(list(bhat_dic.values()))
    # if x.columns[0] == 'CD4-2' or  clustering['similarity_stopped']==0:
    #     print('0 overlap',x.columns[0],mu.shape[0],mergedtonumbers)
    clustering['mp_ncluster'] = mu.shape[0]
    clustering['mergedtonumbers'] = mergedtonumbers
    clustering['mp_clustering'] = list(
        np.apply_along_axis(np.argmax, 1, posterior))

    return clustering


def _set_small_leaf(data):
    all_clustering_dic = {}
    all_clustering_dic[1] = {}

    F_set = data.columns.values.tolist()
    all_clustering = {}

    for item in itertools.combinations(F_set, 1):
        x = data.loc[:, item]
        all_clustering[item] = _set_one_component(x)

    all_clustering_dic[1] = all_clustering

    return all_clustering_dic


def _set_one_component(x):

    clustering = {}
    clustering['bp_ncluster'] = 1
    clustering['bp_clustering'] = [0]*len(x)
    clustering['bp_pro'] = [1]
    clustering['bp_mean'] = np.mean(x)
    clustering['bp_Sigma'] = np.var(x)
    clustering['bhat_dic_track'] = {}
    clustering['similarity_stopped'] = 1
    clustering['mp_ncluster'] = 1
    clustering['mp_clustering'] = [0]*len(x)
    clustering['mergedtonumbers'] = [0]

    return clustering


def BIC(X, max_k=6, bic='bic'):
    """return best k chosen with BIC method"""

    bic_list = _get_BIC_k(X, min(max_k, len(np.unique(X))))

    if bic == 'bic':
        return min(np.argmin(bic_list)+1, _FindElbow(bic_list)), bic_list
    elif bic == 'bic_min':
        return np.argmin(bic_list)+1, bic_list
    elif bic == 'bic_elbow':
        return _FindElbow(bic_list), bic_list


def _get_BIC_k(X, max_k):
    """compute BIC scores with k belongs to [1,max_k]"""
    bic_list = []
    for i in range(1, max_k+1):
        gmm_i = GaussianMixture(i).fit(X)
        bic_list.append(gmm_i.bic(X))
    return bic_list


def _FindElbow(bic_list):
    """return elbow point, defined as the farthest point from the line through the first and last points"""
    if len(bic_list) == 1:
        return 1
    else:
        a = bic_list[0] - bic_list[-1]
        b = len(bic_list) - 1
        c = bic_list[-1]*1 - bic_list[0]*len(bic_list)
        dis = np.abs(a*range(1, len(bic_list)+1) + b *
                     np.array(bic_list) + c)/np.sqrt(a**2+b**2)
        return np.argmax(dis)+1
