#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:44:58 2020

@author: lianqiuyu
"""

import sys
sys.path.append("./CITEsort_rna")

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal, norm
import itertools
from scipy import stats
import operator
from scipy.spatial import distance
from BTree import BTree
import copy
#from scipy.signal import upfirdn
#import pandas as pd
import random
import diptest
from metadimm import MetaDIMM
from pyDIMM import DirichletMultinomialMixture
import scanpy as sc
from functools import reduce

from sklearn import preprocessing
from sklearn.cross_decomposition import CCA
from scipy.sparse import isspmatrix_csr



def all_BIC(leaf_dict, n_features):
        ll, n_features, n_sample = 0, 0, 0
        for key,node in leaf_dict.items():
            ll = ll + node.ll * node.weight 
            # n_features = n_features + len(node.key)
            n_sample = n_sample + len(node.indices)
            # node_name.append(str(key))
        cov_params = len(leaf_dict) * n_features * (n_features + 1) / 2.0
        mean_params = n_features * len(leaf_dict)
        n_param = int(cov_params + mean_params + len(leaf_dict) - 1)
        bic_score = -2 * ll * n_sample + n_param * np.log(n_sample)

        return bic_score

def assign_GMM(sample, mean_list, cov_list, weight, if_log=False, confidence_threshold=0, throw=True):
    """confidence_threshold is used to not assign dots with low confidence to each group:
        a big confidence_threshold represents a more strict standard for confidential dots"""
    confidence_threshold = (1-confidence_threshold) / len(weight) * 2
    index = sample.index
    weight = np.array(weight)
    if if_log:
        type_num = np.log(weight/sum(weight))
    else:
        type_num = weight/sum(weight)
    
    p_prior = np.zeros(shape=(len(sample),len(weight)))
    for i in range(len(weight)):
        if if_log:
            print('sample:',sample.shape,'cov:',cov_list[i].shape)
            p_prior[:,i] = multivariate_normal.logpdf(np.array(sample), mean=np.array(mean_list[i]), cov=np.array(cov_list[i]),allow_singular=True)
            p_prior[:,i] = p_prior[:,i] + type_num[i]
            
        else:
            p_prior[:,i] = multivariate_normal.pdf(np.array(sample), mean=np.array(mean_list[i]), cov=np.array(cov_list[i]))   
            p_prior[:,i] = p_prior[:,i] * type_num[i]
    # p_prior = -p_prior 
    
    p_post = p_prior / (p_prior.sum(axis=1)[:,np.newaxis] )
    pred_label = np.argmin(p_post,axis=1)
    if throw:
        pred_label = [pred_label[i] if p_post[i,pred_label[i]]<confidence_threshold else -1 for i in range(len(pred_label)) ]
    # print(p_post[:10,:])
    # print(pred_label[:10])
    pred_label = pd.Series(data=pred_label,index=index)    
    return pred_label

def update_param(node, alldata, indices, merge_cutoff=0.1,max_k=10,max_ndim=2,bic='bic'):
    """Update: node: data, indices, (param not changed)
               child: data, mean, cov, weight"""
    if node.stop != None:
        return node
    weight = len(indices)/len(alldata)
    # print('len(indices)',len(indices))
    if len(indices) < len(list(node.key)) + 1 or len(indices) < 10:
        node.weight = weight 
        node.stop = 'small size'                        
        return node
    data = alldata[alldata.index.isin(indices)]

    if abs(len(indices)-len(node.indices)) < 0.1*len(node.indices):
        return node
    
    # node.data = data
    node.indices = indices #data.index.values.tolist()
    indices_l = list(set(indices)&set(node.left_indices))
    indices_r = list(set(indices)&set(node.right_indices))
    # data_l = data[data.index.isin(node.left_indices)]
    # data_r = data[data.index.isin(node.right_indices)]

    if len(indices_l) < 10 or len(indices_r) < 10:
        print('#### ReSplit ####')
        print(len(node.left_indices), ':', len(indices_l), len(node.right_indices),':', len(indices_r))
        node = ReSplit_pca(data, merge_cutoff, weight, max_k, max_ndim, bic,root=node)
        return node
    
    if abs(len(indices_l)-len(node.left_indices)) > 0.1*len(node.left_indices):
        node.left_indices = indices_l
        node.w_l = len(node.left_indices)/len(indices)
        data_l = data[indices_l,:]
        data_pc = pd.DataFrame(data=np.dot(data_l.X, node.pc_loading),index=indices_l)
        node.mean_l = data_pc.mean()
        node.cov_l = data_pc.cov()
    if abs(len(indices_r)-len(node.right_indices)) > 0.1*len(node.right_indices):
        node.right_indices = indices_r
        node.w_r = len(node.right_indices)/len(indices)
        data_r = data[indices_r,:]
        data_pc = pd.DataFrame(data=np.dot(data_r.X, node.pc_loading),index=indices_r)
        node.mean_r = data_pc.mean()
        node.cov_r = data_pc.cov()  
   
    return node


def smooth(x,item=0,num=6):
    # i = [i for i in item][0]
    # print(i,value[:5])
    # print(value[0],value[1])
    x = x.apply(np.expm1)
    print('before',(x.isnull()).any())
    for i in x.columns:
        value = np.unique(x.loc[:,i].values.tolist())
        num = min(len(value),num)
        x.loc[:,i] += np.random.normal(loc=0, scale=1,size=x.shape[0]) * 0.01
        # for k in range(num-1):
        #     # print(x.loc[x.loc[:,i]==value[k],i])
        #     x.loc[x.loc[:,i]==value[k],i] += np.random.normal(loc=0, scale=1, size=sum(x.loc[:,i]==value[k])) * (value[k+1]-value[k])*0.1
        # print(i,':',len(value))
    x = x.apply(np.log1p)
    x.mask(x.isnull(),0)
    print('after',(x.isnull()).any())
    return x

def value_count(data):
    val_cnt = pd.Series(index=data.columns)
    for col in data.columns:
        val,cnt = np.unique(data[col].values.tolist(),return_counts=True)        
        val_cnt[col] = len(val)
    return val_cnt

def prototype(leaf_list, data_pcs):
    mean_list, cov_list, w_list = [], [], []
    for node in leaf_list:
        data_pc = data_pcs.loc[node.indices,:]
        mean_list.append(data_pc.mean())
        cov_list.append(data_pc.cov())
        w_list.append(node.weight)
    return mean_list, cov_list, w_list

def outlier_filter(data):

    iqr = data.quantile([0.25,0.5,0.75])
    iqr.loc['iqr',:] = (iqr.loc[0.75,:] - iqr.loc[0.25,:])*4
    iqr.loc['min',:] = iqr.loc[0.25,:]-iqr.loc['iqr',:]
    iqr.loc['max',:] = iqr.loc[0.75,:]+iqr.loc['iqr',:]
    index = set(data.index)
    flag = False
    for col in iqr.columns:
        xmax, xmin = max(data.loc[:,col]), min(data.loc[:,col])
        if xmax-xmin > 5*iqr.loc['iqr',col]:
            flag = True
        if xmax-xmin < 3*iqr.loc['iqr',col]:
            continue
        index = index & set(data[data.loc[:,col]<iqr.loc['max',col]].index) & set(data[data.loc[:,col]>iqr.loc['min',col]].index)
        if len(data)-len(index) > 30 or len(index)<50:
            index = set(data.index) - set(data[col].iloc[[np.argmax(list(data.loc[:,col])),np.argmin(list(data.loc[:,col]))]].index)
            break  
    return list(index), flag



def pp_cca(prior_gene,data=None,merge_cutoff=0.1,weight=1,max_k=5,max_ndim=2,bic='bic',root=None, val_cnt=None, mean=[], cov=None, use_parent=False):
    if len(data) < 50:
        root = BTree(('leaf',))
        root.indices = data.obs_names.values.tolist()
        root.stop = 'small size and no pca'
        return root, []

    if use_parent == True:
        data_cc = pd.DataFrame(data=data.obsm['X_cca'],index=data.obs_names,columns=['CC'+str(int(i/2))+'_'+str(i%2) for i in range(data.obsm['X_cca'].shape[1])])
        # data_cc = pd.DataFrame(data=data.obsm['X_cca'],index=data.obs_names,columns=['CC'+str(i)+'_'+str(i) for i in range(data.obsm['X_cca'].shape[1])])
        data_cc = data_cc.drop(data_cc.columns[data_cc.std() == 0], axis=1)
        if data_cc.shape[1] > 0:
            root = ReSplit_pca(data_cc, merge_cutoff, weight, max_k, max_ndim, bic, mean=mean, cov=cov)
        else: 
            use_parent = False

    if use_parent==False or root.stop != None:
        data_cc = np.array([[]]).reshape((data.shape[0],0))
        for key in prior_gene.keys():
            gene_list = set(list(prior_gene[key])) & set(list(data.var_names))
            # print(prior_gene[key],data.var_names)
            # print(list(gene_list),[i for i in range(len(data.obs_names)) if data.obs_names[i] in list(gene_list)])
            if isspmatrix_csr(data.X):
                y = data[:,list(gene_list)].X.toarray()
                x = data[:,list(set(data.var_names)-set(gene_list))].X.toarray()
                # y = data[:,[i for i in range(len(data.var_names)) if data.var_names[i] in list(gene_list)]].X.toarray()
                # x = data[:,[i for i in range(len(data.var_names)) if data.var_names[i] in list(set(data.var_names)-set(gene_list))]].X.toarray()
                # print(x.shape,y.shape)
            else:
                y = data[:,list(gene_list)].X
                x = data[:,list(set(data.var_names)-set(gene_list))].X

            cca = CCA(n_components=1)
            cca.fit(x, y)
            xc, yc = cca.transform(x, y)
            
            mtx = np.concatenate((xc,yc), axis=1)
            # print(xc.shape,yc.shape,mtx.shape,data_cc.shape)
            data_cc = np.concatenate((data_cc,mtx), axis=1)
        data.obsm['X_cca'] = data_cc
        data_cc = pd.DataFrame(data=data.obsm['X_cca'],index=data.obs_names,columns=['CC'+str(int(i/2))+'_'+str(i%2) for i in range(data.obsm['X_cca'].shape[1])])
        # data_cc = pd.DataFrame(data=data.obsm['X_cca'],index=data.obs_names,columns=['CC'+str(i)+'_'+str(i) for i in range(data.obsm['X_cca'].shape[1])])

        data_cc = data_cc.drop(data_cc.columns[data_cc.std() == 0], axis=1)
        if data_cc.shape[1] == 0:
            pp_pca(data, merge_cutoff, weight, max_k, max_ndim, bic, mean, cov, use_parent=use_parent)
            root = BTree(('leaf',))
            root.indices = data.obs_names.values.tolist()
            root.stop = 'prior gene value are all same in cells'
            return root, []

        if len(mean)==0:
            mean = data_cc.mean()
            cov = data_cc.cov() 
        root = ReSplit_pca(data_cc, merge_cutoff, weight, max_k, max_ndim, bic, mean=mean, cov=cov)
        root.all_clustering_dic[0] = None
        return root, data.obsm['X_cca']
    else:
        return root, []



def pp_pca(data=None,merge_cutoff=0.1,weight=1,max_k=5,max_ndim=2,bic='bic',root=None, val_cnt=None, mean=[], cov=None, use_parent=False):
    ncomponents = 5
    if len(data) < 100:
        root = BTree(('leaf',))
        root.indices = data.obs_names.values.tolist()
        root.stop = 'small size and no pca'
        return root
    if use_parent == True:
        sc.tl.pca(data,n_comps=ncomponents)
        data_pc = pd.DataFrame(data=data.obsm['X_pca'],index=data.obs_names,columns=['PC'+str(i) for i in range(data.obsm['X_pca'].shape[1])])
        root = ReSplit_pca(data_pc, merge_cutoff, weight, max_k, max_ndim, bic, mean=mean, cov=cov)
    
    if use_parent==False or root.stop != None:
        sc.pp.filter_genes(data, min_cells=3)
        sc.pp.scale(data, max_value=10,  )
        sc.tl.pca(data, n_comps=ncomponents)
        # data = data
        data_pc = pd.DataFrame(data=data.obsm['X_pca'],index=data.obs_names,columns=['PC'+str(i) for i in range(data.obsm['X_pca'].shape[1])])
        indices, flag = outlier_filter(data_pc.iloc[:,:5])
        data_pc = data_pc.loc[indices,:]
        if len(data_pc) < 0.99 * len(data) or flag:
            if len(data_pc) < 50:
                root = BTree(('leaf',))
                root.indices = data_pc.index.values.tolist()
                root.all_clustering_dic = _set_small_leaf(data_pc)
                root.stop = 'small size'
                return root
            
            data = data[data_pc.index,:]
            sc.pp.filter_genes(data, min_cells=10)
            if data.shape[1] > 100:
                sc.pp.scale(data,  )
                sc.tl.pca(data,n_comps=ncomponents)
                data_pc = pd.DataFrame(data=data.obsm['X_pca'],index=data.obs_names,columns=['PC'+str(i) for i in range(data.obsm['X_pca'].shape[1])])    
                    
            else:
                if use_parent==False:
                    root = BTree(('leaf',))
                    root.indices = data_pc.index.values.tolist()
                    root.stop = 'few features'
                    return root
                return root
        if len(mean)==0:
            mean = data_pc.mean()
            cov = data_pc.cov()
        

        root = ReSplit_pca(data_pc, merge_cutoff, weight, max_k, max_ndim, bic, mean=mean, cov=cov)
        root.all_clustering_dic[0] = None
        return root
    else:
        return root
        # root.outliers = (flag,set(data.obs_names)-set(root.indices))

def Choose_leaf(leaf_dict=None,data=None,prior_gene={},bic_list=[],leaf_list=None,n_features=0,merge_cutoff=0.1,max_k=10,max_ndim=2,bic='bic',bic_stop=False,rawdata=None,use_parent=False):
    # leaf_dict only save index of current leaves, leaf_list save the sort of surrent leaves
    max_ll, max_root, separable = 0, None, False
    rawdata = data.copy()
    if leaf_dict == None:
        # print(prior_gene)
        if len(prior_gene.keys()) == 0:
            root = pp_pca(data=data,merge_cutoff=merge_cutoff,use_parent=False)
        else:
            root, newcca = pp_cca(prior_gene=prior_gene,data=data,merge_cutoff=merge_cutoff,use_parent=False)
            if len(newcca) != 0:
                data.obsm['X_cca'] = newcca
        
        
        if root.key != ('leaf',):
            # root.mean = data_pc.loc[:,root.key].mean()
            # root.cov = data_pc.loc[:,root.key].cov()
            root.ind = 0
            # root.pc_loading = pd.DataFrame(data.varm['PCs'][:,[int(key[-1]) for key in root.key]],index=data.var_names,columns=[str(root.ind)+key for key in root.key])
            # root.parent_pc = root.pc_loading
            
            # print(root.pc_loading.shape)
        leaf_dict = {0: root}
        max_root = root
        leaf_list = [root]
        
        
    ### _____Choose maxmum loglikely hood gain as new root_____
    
    # print(leaf_dict.items())
    for key in list(leaf_dict.keys()):
        node = leaf_dict[key]
        # print('stop:',node.stop)
        if node.stop != None:
            # leaf_dict.pop(node.ind)
            continue
        separable = True
        if node.score_ll >= max_ll:
            max_ll = node.score_ll
            max_root = node
        # else:
        #     leaf_dict.pop(node.ind)
        #     leaf_dict[node.ind] = node
    bic_min_node = leaf_list  
    if separable:
        print('node.key:',list(max_root.key))
        

        data_l = data[max_root.left_indices,:].copy()
        if len(prior_gene.keys()) == 0:
            max_root.left = pp_pca(data_l, merge_cutoff, max_root.weight * max_root.w_l, max_k, max_ndim, bic, mean=max_root.mean_l, cov=max_root.cov_l, use_parent=use_parent)
        else:
            max_root.left, newcca = pp_cca(prior_gene, data_l, merge_cutoff, max_root.weight * max_root.w_l, max_k, max_ndim, bic, mean=max_root.mean_l, cov=max_root.cov_l, use_parent=use_parent)
            if len(newcca) != 0:
                data.obsm['X_cca'] = np.zeros(shape=(data.shape[0],newcca.shape[1]))
                data[data_l.obs_names,:].obsm['X_cca'] = newcca
        max_root.left.ind = max(leaf_dict.keys()) + 1  
        leaf_dict[max_root.left.ind] = max_root.left
        del data_l


        data_r = data[max_root.right_indices,:].copy()
        if len(prior_gene.keys()) == 0:
            max_root.right = pp_pca(data_r, merge_cutoff, max_root.weight * max_root.w_r, max_k, max_ndim, bic, mean=max_root.mean_r, cov=max_root.cov_r,  use_parent=use_parent)
        else:
            max_root.right, newcca = pp_cca(prior_gene, data_r, merge_cutoff, max_root.weight * max_root.w_r, max_k, max_ndim, bic, mean=max_root.mean_r, cov=max_root.cov_r,  use_parent=use_parent)
            if len(newcca) != 0:
                data.obsm['X_cca'] = np.zeros(shape=(data.shape[0],newcca.shape[1]))
                data[data_r.obs_names,:].obsm['X_cca'] = newcca
        max_root.right.ind = max(leaf_dict.keys()) + 1
        del data_r    
        # del data
        leaf_dict[max_root.right.ind] = max_root.right
        # print('ind',max_root.ind)
        leaf_dict.pop(max_root.ind)
        leaf_list = [x for x in leaf_list if x!=max_root]
        leaf_list.append(max_root.left)
        leaf_list.append(max_root.right)

        
        if bic_stop == False:
            del data
            _, bic_list, bic_min_node = Choose_leaf(leaf_dict=leaf_dict, data=rawdata, prior_gene=prior_gene, bic_list=bic_list, leaf_list=leaf_list, n_features=n_features, rawdata=None, merge_cutoff=merge_cutoff, use_parent=use_parent)
        # if bic_score <= min(bic_list):
        #     bic_min_node = leaf_list
    # else:
    #     dfs = [node.parent_pc.T for node in leaf_list]
    #     df = reduce(lambda x, y: pd.concat([x,y], axis=0, join='inner'), dfs).T
    #     allPC_loading = df.loc[:,~df.columns.duplicated()]
    #     # print('dfs',dfs[0].columns[:2],dfs[1].columns[:2],'df',df.shape,'allPC_loading',allPC_loading.shape)
    #     data_PC = pd.DataFrame(data=np.dot(data.X, allPC_loading),index=data.obs_names,columns=allPC_loading.columns)
    #     mean_list, cov_list, w_list = prototype(leaf_list, data_PC)
    #     new_label = assign_GMM(data_PC, mean_list, cov_list, w_list, if_log=True, confidence_threshold=0.5, throw=False)
    #     for i in range(len(leaf_list)):
    #         node = leaf_list[i]
    #         sub_data = data[new_label==i]
    #         node.indices = sub_data.obs_names
        # Final assignment
    return max_root, bic_list, bic_min_node

def ReSplit(data=None,merge_cutoff=0.1,weight=1,max_k=10,max_ndim=2,bic='bic',marker_set=None, root=None, val_cnt=None, data_raw=None,ind=0):
    if root == None:
        root = BTree(('leaf',))
    root.indices = data.obs_names
    print(len(root.indices))
    root.ind = ind
    if data.shape[0] < 200:
        root.stop = 'small size'
        return root

    md = MetaDIMM()
    data = md.pca(data)
    metagenes = md.get_metagenes(data, n_meta_pcs=None, loading_threshold=0.1)
    if len(metagenes) == 0:
        root.stop = 'Not separable'
        return root
    root.key = str(metagenes[0][0])
    print(root.key)
    adata_meta = md.merge(data_raw, metagenes)
    cluster_label, posterior_prob = md.dimm_cluster(adata_meta, n_components=2, verbose=True)
    data.obs['MetaDIMM_prediction'] = cluster_label
    # sc.pl.umap(data, color='MetaDIMM_prediction')
    index_l = [i for i in range(adata_meta.shape[0]) if cluster_label[i]==0]
    index_r = [i for i in range(adata_meta.shape[0]) if cluster_label[i]==1]
    if len(index_l)<10 or len(index_r)<10:
        root.stop = 'Not separable'
        return root
    root.left = ReSplit(adata_meta[index_l,:],data_raw=data_raw[index_l,:],ind=0)
    root.right = ReSplit(adata_meta[index_r,:],data_raw=data_raw[index_r,:],ind=1)
    return root


def ReSplit_pca(data_pc=None,merge_cutoff=0.1,weight=1,max_k=5,max_ndim=2,bic='bic',root=None, val_cnt=None, mean=None, cov=None):

    if root == None:
        root = BTree(('leaf',))
        root.val_cnt = val_cnt
    root.indices = data_pc.index.values.tolist()

    # root.indices = data.obs_names
    print('cell nums:',len(root.indices))
    root.weight = weight
    root.stop = None

    if data_pc.columns[0][0] == 'C':
        root.embeddding = 'CCA'
    else:
        root.embeddding = 'PCA'

    if data_pc.shape[0] < 200:        
        root.all_clustering_dic = _set_small_leaf(data_pc)
        root.stop = 'small size'
        return root


    # if root.marker == None:
    if True:
        root.mean = mean
        root.cov = cov
    else:
        root.mean = data.loc[:,root.marker].mean()
        root.cov = data.loc[:,root.marker].cov()

    unimodal = GaussianMixture(1,covariance_type='full').fit(data_pc)
    root.ll = root.weight * unimodal.lower_bound_
    root.bic = unimodal.bic(data_pc)
    
    separable_features, bipartitions, scores_ll, bic_list, all_clustering_dic, rescan = HiScanFeatures(data_pc,root,merge_cutoff,max_k,max_ndim,bic)

    if len(separable_features) == 0:
        root.all_clustering_dic = all_clustering_dic
        root.stop = 'no separable features'
        # root.key = ('leaf',)
        return root

    idx_best = np.argmax(scores_ll)
    root.score_dict = dict(zip(separable_features, scores_ll))
    if np.max(scores_ll) <= -1000:
    #if root.bic < bic_list[idx_best]:
        root.all_clustering_dic = all_clustering_dic
        root.stop = 'spliting increases ll score'
        return root

    #idx_best = np.argmax(scores_ent)
    best_feature = separable_features[idx_best]
    best_partition = bipartitions[best_feature]
    best_score = scores_ll[idx_best]
    #best_weights = all_clustering_dic[len(best_feature)][best_feature]['weight']

    

    ## construct current node  
    if rescan: ## save separable features if have rescaned
        root.rescan = True
        print('root.rescan:',root.rescan)
        root.separable_features = separable_features

    root.key = best_feature
    root.all_clustering_dic = all_clustering_dic
    root.score_ll = best_score

    
    ### Calculate mean, std and weight for all dimension
    p1_mean = data_pc.loc[best_partition, root.key].mean()
    p2_mean = data_pc.loc[~best_partition, root.key].mean()
    p1_cov = data_pc.loc[best_partition, root.key].cov()
    p2_cov = data_pc.loc[~best_partition, root.key].cov()

    # print(data_pc.std(),p1_cov)
    prob1 = multivariate_normal.pdf(np.array(data_pc.loc[:,root.key]), mean=np.array(p1_mean), cov=np.array(p1_cov), allow_singular=True)
    prob2 = multivariate_normal.pdf(np.array(data_pc.loc[:,root.key]), mean=np.array(p2_mean), cov=np.array(p2_cov), allow_singular=True)
    prob1 = prob1/(prob1+prob2)
    prob2 = prob2/(prob1+prob2)

    partition1 = [best_partition[i] if prob1[i]>0.5 else False for i in range(len(prob1))]
    partition2 = [~best_partition[i] if prob2[i]>0.5 else False for i in range(len(prob2))]
    # partition1 = pd.DataFrame(data=[best_partition.iloc[i] if prob1[i]>0.8 else False for i in range(len(prob1))], index=best_partition.index )
    # partition2 = pd.DataFrame(data=[~best_partition.iloc[i] if prob2[i]>0.8 else False for i in range(len(prob2))], index=best_partition.index )
    print(sum(partition1),sum(partition2))
    if sum(partition1)<20 or sum(partition2)<20:
        root.stop = 'low confidence split'
        root.key = ('leaf',)
        return root

    flag = True
    if len(p1_mean) == 1:
        flag = p1_mean.values > p2_mean.values
    else:
        p1_cosine = sum(p1_mean)/np.sqrt(sum(p1_mean**2))
        p2_cosine = sum(p2_mean)/np.sqrt(sum(p2_mean**2))
        flag = p1_cosine > p2_cosine

    if flag:
        root.right_indices = data_pc.iloc[partition1, :].index
        root.w_r = sum(best_partition)/len(best_partition)
        root.mean_r = p1_mean
        # root.cov_r = p1_cov
        root.left_indices = data_pc.iloc[partition2, :].index 
        root.w_l = sum(best_partition)/len(best_partition)
        root.mean_l = p2_mean
        # root.cov_l = p2_cov
        root.where_dominant = 'right'
    else:
        root.right_indices = data_pc.iloc[partition2, :].index
        root.w_r = sum(best_partition)/len(best_partition)
        root.mean_r = p2_mean
        # root.cov_r = p2_cov
        root.left_indices = data_pc.iloc[partition1, :].index
        root.w_l = sum(best_partition)/len(best_partition)
        root.mean_l = p1_mean
        # root.cov_l = p1_cov
        root.where_dominant = 'left'

    
    ## recursion
    # root.left = ReSplit(child_left,merge_cutoff,weight * w_l,max_k,max_ndim,bic)
    # root.right = ReSplit(child_right,merge_cutoff,weight * w_r,max_k,max_ndim,bic)

    return root


def HiScanFeatures(data,root,merge_cutoff,max_k,max_ndim,bic):
    
    # Try to separate on one dimension
    ndim = 1
    all_clustering_dic = {}
    separable_features, bipartitions, scores, bic_list, all_clustering_dic[ndim] = ScoreFeatures(data,root,merge_cutoff,max_k,ndim,bic)
    
    rescan = False
    if len(separable_features) == 0:

        rescan_features = []
        for item in all_clustering_dic[ndim]:
            val = all_clustering_dic[ndim][item]['similarity_stopped']
            ### 考虑阈值是否应该随着用户指定调整
            if val < min(4*merge_cutoff,0.8):
                rescan_features.append(item[0])
        
        for ndim in range(2,max_ndim+1):
            # Num of feature not enough for hight dimension clustering                                                                                       
            #### Add all features <0.5 to assign features(save mean of sebarable features)
            if len(rescan_features) < ndim:
                # Threshold is set to a softer one, when partition of its parents is not well, may cause wrong fragment. 
                #### Add all features <0.5 to assign features(save mean of sebarable features)
                # separable_features, bipartitions, scores, bic_list, all_clustering_dic[ndim] = ScoreFeatures(data,root,0.5,max_k,len(rescan_features),bic)
                break
            
            ### threshold=0.5 or merge_cutoff?
            separable_features, bipartitions, scores,bic_list, all_clustering_dic[ndim] = ScoreFeatures(data,root,min(merge_cutoff*2,0.6),max_k,ndim,bic,rescan_features)
            if len(separable_features) >= 1:
                break
    return separable_features, bipartitions, scores, bic_list, all_clustering_dic, rescan
    


def ScoreFeatures(data,root,merge_cutoff,max_k,ndim,bic,rescan_features=None,):
 
    if rescan_features != None:
        F_set = rescan_features
    else:
        F_set = data.columns.values.tolist() # Feature list
    
    all_clustering = {}
    separable_features = []
    bipartitions = {}
    scores = []
    bic_list = []
    continueness = False

    # for item in itertools.combinations(F_set, ndim): 
    #     if len(np.unique(data.loc[:,item].values.tolist()))>300:
    #         continueness = True
    #         break

    for item in itertools.combinations(F_set, ndim): # When ndim=1, search one feature each time. When ndim=2, search two features each time.
        x = data.loc[:,item]
        # val = len(np.unique(x.values.tolist()))
        # print(item)
        # x = smooth(x,item, max(val,6))
        # if continueness==False and ndim==1 and val<300 and val>1 and len(x)>10*val:
        #     x = smooth(x,item, min(val,6))
        # data.loc[:,item] = x
        all_clustering[item] = Clustering(x,merge_cutoff,max_k,bic)
    
    for item in all_clustering:
        if all_clustering[item]['mp_ncluster'] > 1:
            
            merged_label = all_clustering[item]['mp_clustering']
            labels, counts = np.unique(merged_label, return_counts=True)
            if len(counts) == 1 or np.min(counts) < 5:
                continue
            
            ll_gain = []#np.zeros(len(labels))
            bic_mlabels = []
            # Choose the cluster with max ll gain in this subspace
            for mlabel in labels:
                ### 可优化，merge后只有两类时只用算一次，因为算两次都是一样的
                assignment = merged_label == mlabel

                # marker_list = data.columns
                marker_list = list(item) # Choose only marker to calculate loglikelyhood
                # print('marker_list:',marker_list)
                if sum(assignment) < 100 or sum(~assignment) < 100:
                    ll_gain.append(-1000)
                    continue

                # print('sum(assignment):',sum(assignment),'len(assignment)',len(assignment))
                gmm1 = GaussianMixture(1,covariance_type='full').fit(data.loc[assignment,marker_list]) 
                ll1 = gmm1.lower_bound_ * sum(assignment)/len(assignment)
                # bic1 = gmm1.bic(data.loc[assignment,marker_list]) 
                
                gmm0 = GaussianMixture(1,covariance_type='full').fit(data.loc[~assignment,marker_list])
                ll0 = gmm0.lower_bound_ * sum(~assignment)/len(assignment)
                # bic0 = gmm0.bic(data.loc[~assignment,marker_list]) 
                
                gmm_ = GaussianMixture(1,covariance_type='full').fit(data.loc[:,marker_list])
                ll_ = gmm_.lower_bound_

                ll_gain.append(  (ll1 + ll0) - ll_  )
                # bic_mlabels.append( bic1 + bic0 )
            best_mlabel_idx = np.argmax(ll_gain)
            best_mlabel = labels[best_mlabel_idx]
            
            bipartitions[item] = merged_label == best_mlabel
            scores.append( ll_gain[best_mlabel_idx] )
            separable_features.append(item)
            # bic_list.append( bic_mlabels[best_mlabel_idx] )
            
            # bipartitions[item] = all_clustering[item]['max_ent_p']
            # scores.append(all_clustering[item]['max_ent'])
    if len(scores)!=0 and max(scores) == -1000:
        separable_features = []
    return separable_features, bipartitions, scores, bic_list, all_clustering



def Clustering(x,merge_cutoff,max_k,bic,val_cnt=None):
    
    
    # print(val_cnt, len(val))
    # print(np.array(x))
    val,cnt = np.unique(x.values.tolist(),return_counts=True)  
    if x.shape[1]==1 and (len(val)<50 or diptest.dipstat(np.array(x.iloc[:,0]))<(1-merge_cutoff)*0.002):
            # val,cnt = np.unique(x.values.tolist(),return_counts=True)
            # print(val_cnt.index, val_cnt.values, len(val))
            clustering = _set_one_component(x) 
        # for i in range(x.shape[1]):
        #     x.iloc[:,i] += np.random.normal(loc=0, scale=1, size=len(x)) * 0.1
        # print(x.columns, len(val))
 
    else:
        # if x.shape[1]==1:
        #     dip = diptest.dipstat(np.array(x))
        #     if dip < 0.02:
        #         clustering = _set_one_component(x) 
        #         print(x.name)
        # print(x.columns, len(val))
        k_bic,_ = BIC(x,max_k,bic)
        # print(x.columns,k_bic)
    
        if k_bic == 1:    
            # print('k_bic=1')
            # if only one component, set values
            clustering = _set_one_component(x)      
        else:
            # print(val_cnt.index, val_cnt.values)
            bp_gmm = GaussianMixture(k_bic).fit(x)
            clustering = merge_bhat(x,bp_gmm,merge_cutoff)
            # import matplotlib.pyplot as plt
            # plt.hist(x,bins=1000)
            # plt.show()
            # print(clustering['mp_ncluster'])
            '''
            if clustering['mp_ncluster'] > 1:
    
                merged_label = clustering['mp_clustering']
                labels, counts = np.unique(merged_label, return_counts=True)
                
                per = counts/np.sum(counts)                 
                ents = [stats.entropy([per_i, 1-per_i],base=2) for per_i in per]
                clustering['max_ent'] = np.max(ents)
                best_cc_idx = np.argmax(ents)
                best_cc_label = labels[best_cc_idx]
                clustering['max_ent_p'] = merged_label == best_cc_label
            '''
    return clustering



def bhattacharyya_dist(mu1, mu2, Sigma1, Sigma2):
    Sig = (Sigma1+Sigma2)/2
    ldet_s = np.linalg.det(Sig)
    ldet_s1 = np.linalg.det(Sigma1)
    ldet_s2 = np.linalg.det(Sigma2)
    d1 = distance.mahalanobis(mu1,mu2,np.linalg.inv(Sig))**2/8
    d2 = 0.5*np.log(ldet_s) - 0.25*np.log(ldet_s1) - 0.25*np.log(ldet_s2)
    return d1+d2



def merge_bhat(x,bp_gmm,cutoff):

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
            m1 = mu[c_pair[0],:]
            m2 = mu[c_pair[1],:]
            Sigma1 = Sigma[c_pair[0],:,:]
            Sigma2 = Sigma[c_pair[1],:,:]
            bhat_dic[c_pair] = np.exp(-bhattacharyya_dist(m1, m2, Sigma1, Sigma2))

        clustering['bhat_dic_track'][merge_time] = bhat_dic
        merge_time = merge_time + 1
        
        max_pair = max(bhat_dic.items(), key=operator.itemgetter(1))[0]
        max_val = bhat_dic[max_pair]

        if max_val > cutoff or max_val<0.05:
            merged_i,merged_j = max_pair
            # update mergedtonumbers
            for idx,val in enumerate(mergedtonumbers):
                if val == merged_j:
                    mergedtonumbers[idx] = merged_i
                if val > merged_j:
                    mergedtonumbers[idx] = val - 1
                    
            # update parameters
            weights[merged_i] = weights[merged_i] + weights[merged_j]
            
            posterior[:,merged_i] = posterior[:,merged_i] + posterior[:,merged_j]
            
            w = posterior[:,merged_i]/np.sum(posterior[:,merged_i])
            mu[merged_i,:] = np.dot(w,x)# update                                 
            
            x_centered = x.apply(lambda xx: xx-mu[merged_i,:],1)
            Sigma[merged_i,:,:] = np.cov(x_centered.T,aweights=w,bias=1)

            del weights[merged_j]
            #weights = np.delete(weights,merged_j,0)
            mu = np.delete(mu,merged_j,0)
            Sigma = np.delete(Sigma,merged_j,0)
            posterior = np.delete(posterior,merged_j,1)
            current_ncluster = current_ncluster - 1

        else:
            merge_flag = False
    
    
    clustering['similarity_stopped'] = np.min(list(bhat_dic.values()))
    clustering['mp_ncluster'] = mu.shape[0]
    clustering['mergedtonumbers'] = mergedtonumbers
    clustering['mp_clustering'] = list(np.apply_along_axis(np.argmax,1,posterior))
    clustering['max_val'] = cutoff
    
    return clustering



def _set_small_leaf(data):
    all_clustering_dic = {}
    all_clustering_dic[1] = {}
    
    F_set = data.columns.values.tolist()
    all_clustering = {}

    for item in itertools.combinations(F_set, 1):
        x = data.loc[:,item]
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
    clustering['max_val'] = 'max_val'

    return clustering



def BIC(X, max_k = 10,bic = 'bic'):
    """return best k chosen with BIC method"""
    
    bic_list = _get_BIC_k(X, min(max_k,len(np.unique(X))))
    
    if bic == 'bic':   
        return min(np.argmin(bic_list)+1,_FindElbow(bic_list)),bic_list
    elif bic == 'bic_min':   
        return np.argmin(bic_list)+1,bic_list
    elif bic == 'bic_elbow':
        return _FindElbow(bic_list),bic_list

    
    
def _get_BIC_k(X, max_k):
    """compute BIC scores with k belongs to [1,max_k]"""
    bic_list = []
    for i in range(1,max_k+1):
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
        dis = np.abs(a*range(1,len(bic_list)+1) + b*np.array(bic_list) + c)/np.sqrt(a**2+b**2)
        return np.argmax(dis)+1


