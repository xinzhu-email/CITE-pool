"""
Created on Jan 15 23:44:58 2025

@author: xinzhujiang
"""

import sys
sys.path.append("./CITEpool")
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.nn as nn
import torch
import logging
from sklearn.cross_decomposition import CCA, PLSRegression, PLSCanonical
from sklearn.preprocessing import normalize
import scanpy as sc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import diptest
import random
import copy
from BTree import BTree
from scipy.spatial import distance
import operator
import itertools
from scipy.stats import multivariate_normal, norm
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import anndata



def RNApp(rnadata=None, nodelist=None):
    # set(rnadata[list(rnadata.keys())[0]].var_names)
    ppdata, gene_list = {}, []
    ppdata = {}
    # cellnum = [data.shape[0] for data in list(rnadata.values())]
    nrepeat = 1
    ncomp = 5
    # adata = rnadata[0]
    # adata = sc.concat(rnadata.values(),axis=0)
    for i in list(rnadata.keys()):

        if len(nodelist) != 0:
            if nodelist[i] is not None:
                nodelist[i].indices = nodelist[i].indices.intersection(rnadata[i].obs_names)
                rnadata[i] = rnadata[i][list(set(nodelist[i].indices)),:] 
            else:
                rnadata[i] = rnadata[i][False,:]

        if len(rnadata[i]) == 0 :
            continue

        if i == 0:
            adata = rnadata[i]
        else:
            adata = sc.concat([adata, rnadata[i]],axis=0)

    if len(adata) < 50:
        return {}
    # adata.obs['batch'] = pd.Categorical(adata.obs['batch'])
    adata.obs['batch'] = adata.obs['batch'].cat.remove_unused_categories()
    sc.pp.highly_variable_genes(adata, n_top_genes=500, batch_key='batch')#
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=ncomp)
    pcs = pd.DataFrame(index=adata.obs_names, data=adata.obsm['X_pca'])


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


def smooth(x, item=0, num=6):
    
    s = x.min().min()
    if s >= 0:
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

    return x

def value_count(data):
    val_cnt = pd.Series(index=data.columns)
    for col in data.columns:
        val, cnt = np.unique(data[col].values.tolist(), return_counts=True)
        val_cnt[col] = len(val)
    # print(val_cnt)
    return val_cnt

def bhattacharyya_dist(mu1, mu2, Sigma1, Sigma2):
    Sig = (Sigma1+Sigma2)/2
    ldet_s = np.linalg.det(Sig)
    ldet_s1 = np.linalg.det(Sigma1)
    ldet_s2 = np.linalg.det(Sigma2)
    d1 = distance.mahalanobis(mu1, mu2, np.linalg.inv(Sig))**2/8
    d2 = 0.5*np.log(ldet_s) - 0.25*np.log(ldet_s1) - 0.25*np.log(ldet_s2)
    return d1+d2

def gene_selection(rna):
    sc.pp.highly_variable_genes(rna,n_top_genes=500)
    rna = rna[:,rna.var.highly_variable]
    x = rna.X.toarray()
    y = rna.obsm['marker']
    pls = PLSRegression(n_components=1)
    pls.fit(x,y)
    loading = pd.Series(pls.x_weights_.T.tolist()[0], index=rna.var_names)
    genes = list(loading.nlargest(100).index )+list(loading.nsmallest(100).index)
    # print(genes)
    return genes


def CrossSplit(adtdata=None, rnadata=None, merge_cutoff=0.1, crossnode=None, runADT=True):
    
    cross_score = {}
    if crossnode is not None: # Tree finetune
        nodelist_ = crossnode.nodelist
    else: 
        nodelist_ = []

    # Use ADT or RNA as guide data
    if runADT:
        guidedata = adtdata
    else:
        guidedata = RNApp(rnadata.copy(), nodelist_)
    
    nodelist = []
    for i in range(len(guidedata)):

        if crossnode is not None:
            if nodelist_[i] is None:
                rnadata[i] = rnadata[i][False, :]
                guidedata[i] = pd.DataFrame([])
            else:
                indices = nodelist_[i].indices.intersection(rnadata[i].obs_names)
                nodelist_[i].indices = indices
                rnadata[i] = rnadata[i][indices,:]
                guidedata[i] = guidedata[i].loc[indices,:]

        
        if len(rnadata[i]) == 0 or len(guidedata[i]) == 0:
            nodelist.append(None)
            
            continue
        
        node = BTree(('leaf',))

        if i in guidedata.keys() and len(guidedata[i]) > 0 and guidedata[i].columns[0][:2] == 'CC':
            node.embedding = guidedata[i]
        else:
            node.embedding = []
        
        node.stop = None
        node.score_dict = {}
        node.indices = rnadata[i].obs_names.values.tolist()

        if len(guidedata[i]) == 0:
            nodelist.append(node)
            continue

        node.indices = guidedata[i].index.values.tolist()
        
        node.val_cnt = value_count(guidedata[i])

        if runADT:
            data = smooth(guidedata[i].copy())
        else:
            data = guidedata[i]
        
        if data.shape[0] < 100:
            nodelist.append(node)
            continue
        
        score = GmmFit(data, merge_cutoff, node.val_cnt)
        
        for feature in score.keys():
            if feature in cross_score.keys():
                cross_score[feature].append(score[feature])
            else:
                cross_score[feature] = [score[feature]]
        
        node.score_dict = score
        nodelist.append(node)
    
    w, loss = pd.Series(0), 10000
    if len(cross_score) > 0:

        feature_score = pd.Series(index=cross_score.keys())
        for feature in cross_score.keys():
            feature_score[feature] = np.mean(cross_score[feature]) + \
                 np.max(cross_score[feature]) + 2*len(cross_score[feature])/len(guidedata)

        # print(feature_score)
        best_feature = feature_score.index[feature_score.argmax()] 
        print('=== Best Feature:', best_feature,':',feature_score[best_feature],'in',len(cross_score[best_feature]), 'datasets ===')
        
        if feature_score[best_feature] > 1:
            

            traindata, genes = TrainData(rnadata.copy(), guidedata, best_feature, nodelist)
            w, m0, m1, loss, deltaW, probs = LearnPseudoMaker(traindata.copy())

            if loss < 10:
                nodelist, converge = assign(w, deltaW, probs, nodelist, traindata.copy(), genes, best_feature)
                if converge:
                    crossnode = CrossNode(nodelist)
                    crossnode = GenModelNode(crossnode, best_feature, w, cross_score, m0,m1, loss=loss)

                    leftadt, rightadt, leftrna, rightrna= {}, {}, {}, {}
                    for i in range(len(guidedata)):
                        leftrna[i] = rnadata[i][nodelist[i].left_indices, :].copy()
                        rightrna[i] = rnadata[i][nodelist[i].right_indices, :].copy()
                        leftadt[i] = adtdata[i].loc[nodelist[i].left_indices, :]
                        rightadt[i] = adtdata[i].loc[nodelist[i].right_indices, :]
                    
                    crossnode.left = CrossSplit(leftadt, leftrna, merge_cutoff, runADT=True)
                    crossnode.right = CrossSplit(rightadt, rightrna, merge_cutoff, runADT=True)
                    return crossnode
        
    if runADT:
        crossnode = CrossSplit(adtdata, rnadata, merge_cutoff, runADT=False)
        return crossnode
    else:
        crossnode = CrossNode(nodelist)
        crossnode = GenModelNode(crossnode, ('leaf',), w, score_dict=cross_score, loss=loss)
        return crossnode




class CrossNode():
    def __init__(self, nodelist, left=None, right=None, modelnode=None):
        self.nodelist = nodelist
        self.left = left
        self.right = right
        self.modelnode = modelnode

def GenModelNode(crossnode, best_feature, artificial_w, score_dict, m0=0, m1=0, loss=0):
    modelnode = BTree(best_feature)
    cellnum, trainnum, traindataset = 0, 0, []

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
                trainnum += 1
                traindataset.append(i+1)

    modelnode.indices = traindataset
    modelnode.val_cnt = cellnum
    modelnode.artificial_w = artificial_w
    modelnode.loss = loss
    modelnode.mean_l, modelnode.mean_r = m0, m1
    modelnode.score_dict = score_dict
    crossnode.modelnode = modelnode
    
    return crossnode



def TrainData(rnadata, guidedata, best_feature, nodelist):
    crossgenes = []
    traindata = {}
    for i in range(len(guidedata)):
        traindata[i] = rnadata[i].copy()
        if best_feature not in nodelist[i].score_dict.keys():
            traindata[i].obsm['marker'] = np.zeros(traindata[i].shape[0]).T
            continue
        traindata[i].obsm['marker'] = np.array(guidedata[i].loc[traindata[i].obs_names,best_feature].values)
        genes = gene_selection(traindata[i].copy())
        crossgenes.extend(genes)

    crossgenes = list(set(crossgenes))
    for i in range(len(guidedata)):
        crossgenes = traindata[i].var_names.intersection(crossgenes)

    for i in range(len(guidedata)):
        traindata[i] = traindata[i][:, crossgenes]

    return traindata.copy(), crossgenes




def assign(w, deltaW, probs, nodelist, traindata, genes, best_feature):
    w = pd.Series(w.detach().numpy().reshape(-1), index=genes)
    pmean = []
    for i in range(len(nodelist)):
        nodelist[i].key = best_feature 
        nodelist[i].artificial_w = w  + pd.Series(deltaW[i].detach().numpy().reshape(-1), index=genes)
        p = probs[i]
        pmean.append(p.mean(0))
        print(p.mean(0))
        pred = pd.Series(np.argmax(p, axis=1), index=traindata[i].obs_names)
        # print(pred[pred==0])
        nodelist[i].left_indices = pred[pred==0].index
        nodelist[i].right_indices = pred[pred==1].index
        nodelist[i].probs = p
        # print('left/right = ',len(nodelist[i].left_indices),'/', len(nodelist[i].right_indices))
    print(np.mean(pmean,axis=0))
    # if np.mean(pmean,axis=0).min() < 0.1:
    #     return nodelist, False
    return nodelist, True


def LearnPseudoMaker(rnadata):
    if len(rnadata[list(rnadata.keys())[0]]) > 10000:
        batch_num = 32
    elif len(rnadata[list(rnadata.keys())[0]]) > 6000:
        batch_num = 16
    else:
        batch_num = 4
    # print('batch num:',batch_num)
    class scRNAdata(Dataset):
        def __init__(self, adata):            
            try:
                self.data = adata.X.toarray()
            except:
                self.data = adata.X
            self.marker = adata.obsm['marker']
            # self.label = adata.obs['label'].values
            # self.index = (str(id)+ np.array(adata.obs_names)).tolist()

        def __getitem__(self, index):
            data = torch.tensor(self.data[index], dtype=torch.float)
            marker = torch.tensor(self.marker[index], dtype=torch.float)
            # label = torch.tensor(self.label[index], dtype=torch.float)
        
            return data, marker

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
    cca.fit(data.data, data.marker)
    Winit = cca.x_weights_.T.tolist()
    Winit = torch.tensor(Winit, dtype=torch.float32)


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
  
            h_centered = h_2d - h_2d.mean(dim=0, keepdim=True)
            m_centered = m_2d - m_2d.mean(dim=0, keepdim=True)

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
            
            prob_entropy = -torch.sum(probs*torch.log2(probs+1e-8),dim=1).mean()
            probsct = torch.mean(probs,dim=0)
            type_entropy = -(probsct*torch.log2(probsct)).mean()
            entropy = prob_entropy + (1-type_entropy) *1

            if m.sum() == 0:
                l_correlation = torch.tensor(0)
                # entropy = entropy * 4
                l_classify = l_classify*2
            else:
                l_correlation = torch.relu(0.81-self.correlation(h, m)**2)

            # print(entropy,probs.shape)

            wL1 = abs(W).mean()
            h_norm = torch.norm(h, p=2)
            # print(l_classify.shape, l_correlation.shape, center.sum())
            # l_correlation = 0
            l = l_classify + l_correlation*2 + torch.relu(1-center.var())*10 + variance*0.01 + entropy + \
                 center.sum()*0.001 + wL1*0.01 + h_norm*0.001

            return l, center, [probsct.detach(), l_correlation.detach(), center.detach(),   l_classify.item(), entropy.item()]


    loss_fn1 = LowDimClustering()
    loss_fn2 = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    def train(dataloader, model, loss_fn1, loss_fn2, optimizer):

        model.train()

        # for batch, data in enumerate(dataloader):
        for batch in range(batch_num):
            loss, size, miu0, miu1 = 0, 0, torch.empty(
                len(dataloader)), torch.empty(len(dataloader))
            for i in range(len(dataloader)):
                (X, y) = next(iter(dataloader[i]))
                if X.shape[0] < 2:
                    continue
                # (X,y) = dataloader[i]
                X, y = X.to(device), y.to(device)

                h, probs = model(X,dataid=i)

                loss1, center, losslist = loss_fn1(h, probs, y, model.fc.weight[:,0])
                miu0[i], miu1[i] = center[0], center[1]
                loss1 = loss1 + torch.norm(model.deltaW[i], p=2)*10
                loss += loss1 
                # if batch % 100 == 0 :
                #     print(loss.item(),losslist,miu0.var().item())

            # feature = model(X).squeeze(-1)
            # loss = loss_fn1(feature, y, model.fc.weight)
            # loss += loss_fn2(feature, y)*20
            if len(dataloader) > 1:
                loss += (miu0.var() + miu1.var()) * 10
                # print(loss.item(),miu0.var(),miu1.var())
            
            # size += len(dataloader[i].dataset)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # if batch % 9 == 0:
            #     loss = loss.item()
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]",i+1,'train data')

        return loss, miu0.mean().item(), miu1.mean().item()

    epochs = 101
    loss0 = 10000
    for t in range(epochs):
        # print(f"Epoch {t+1}\n-------------------------------")
        loss, mean0, mean1 = train(
            dataload, model, loss_fn1, loss_fn2, optimizer)
        if (t+1) % 100 == 0:
            print(f"Epoch {t+1} loss: {loss:>7f}")
        if abs(loss-loss0) < 0.05 and loss < 0.01:
            break
        loss0 = loss
        # scheduler.step()
    weight = model.fc.weight[0,:]
    deltaW = model.deltaW
    probs = {}
    for i in range(len(rnadata)):
        probs[i] = scRNAdata(rnadata[i].copy()).data.dot(model.fc.weight[1:,:].detach().numpy().T) 
        probs[i] = torch.softmax(torch.tensor(probs[i]),dim=1).detach()
        
    # print(weight.shape)
    # weight = F.normalize(weight)
    # weight = model.module.features[0].weight
    # print(weight.shape))
    # print(w0[0,0],weight[0,0])
    return weight, mean0, mean1, loss.item(), deltaW, probs

    

def GmmFit(data, cutoff, val_cnt):
    score = {}
    for i in range(data.shape[1]):
        col = data.columns[i]
        x = data.iloc[:,[i]]

        if col in ['CD158','CD158e1','CD146','CD62L','CD2']:
            continue
        # if col in ['CD28']:
        #     print('val_cnt: ',val_cnt[col],min(min(x.shape[0]/20, 70), x.shape[0]),diptest.dipstat(np.array(x.iloc[:, 0])))

        if val_cnt[col] < min(min(x.shape[0]/20, 60), x.shape[0]):
            # print('val_cnt: ',col, '=', val_cnt[col])
            continue
        dip = diptest.dipstat(np.array(x.iloc[:, 0]))
        if dip < max((1-cutoff)*0.008, 0.005):
            # print('dip: ',col, '=', dip)
            continue
        gmm = GaussianMixture(n_components=2)
        gmm.fit(x)
        
        miu1, miu2 = gmm.means_[0,:], gmm.means_[1,:]
        sigma1, sigma2 = gmm.covariances_[0,:,:], gmm.covariances_[1,:,:]
        pai1, pai2 = gmm.weights_[0],  gmm.weights_[1]  

        partition = -pai1*np.log2(pai1)-pai2*np.log2(pai2)
        if partition < 0.2:
            continue
        
        sep = 1 - np.exp(-bhattacharyya_dist(miu1, miu2, sigma1, sigma2))

        if sep < 1-cutoff:
            # print('sep: ',col, '=', sep)
            continue
        
        y = gmm.predict(x)
        var = [x[y==1].var().values, x[y==1].var().values]

        if np.min(var) < 0.3:
            # print('variation: ',col, '=', var)
            continue
        
        fit = gmm.score(x)

        # print(col, fit, sep, partition, var)
        score[col] = fit*0.2 + sep + partition*0.9 + np.min(var)*0.1

    return score



def dfs(crossnode, adtdata, rnadata, merge_cutoff, nodeid, ifretrain):

    if len(nodeid) == 0 and crossnode.modelnode.key == ('leaf',) or crossnode.modelnode.ind in nodeid:
        crossnode = CrossSplit(adtdata.copy(), rnadata.copy(), merge_cutoff, crossnode=crossnode)
        return crossnode

    elif crossnode.modelnode.key == ('leaf',):
        return crossnode
    else:
        nodelist = crossnode.nodelist

        # if ifretrain:
        #     if crossnode.modelnode.ind in nodeid: # 
        #         print(crossnode.modelnode.key)
        #         crossnode.modelnode.artificial_w, crossnode.modelnode.embedding, crossnode.modelnode.loss, crossnode.nodelist = retrain(
        #                         crossnode.nodelist, rnadata.copy(), adtdata.copy(), crossnode.modelnode.key, crossnode.modelnode) # 

        lnodelist, rnodelist, ladt, radt, lrna, rrna = [], [], {}, {}, {}, {} 
        for i in range(len(nodelist)):
            node = nodelist[i]
            node.indices = node.indices.intersection(rnadata[i].obs_names)
            
            if node is not None:
                if node.left is not None:                    
                    node.left.indices = node.left_indices

                if node.right is not None:
                    node.right.indices = node.right_indices
               
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
        crossnode.left = dfs(lcrossnode, adtdata, rnadata, merge_cutoff, nodeid, ifretrain)
        crossnode.right = dfs(rcrossnode, adtdata, rnadata, merge_cutoff, nodeid,  ifretrain)
           
        return crossnode


def CellEmbedding(tree, rna):
    tree.ind = 0
    queue, idxlist = [tree], [0]
    embedding, treelabel = pd.DataFrame(index=rna.obs_names), pd.DataFrame(index=rna.obs_names)
    

    while len(queue)>0:
        node = queue.pop(0)
        ind = node.ind

        if node.key != ('leaf',):
            
            if ind not in embedding.columns:
                embedding[ind] = 0

            vars = rna.var_names.intersection(node.artificial_w.index)
            embedding.loc[:, ind] = rna[:,vars].X.dot(node.artificial_w[vars].values)
            treelabel[ind] = 0
            treelabel.loc[node.left_indices, node.ind] = -1
            treelabel.loc[node.right_indices, node.ind] = 1

            node.left.ind, node.right.ind = 2*node.ind+1, 2*node.ind+2 
            queue.append(node.left)
            queue.append(node.right)
            idxlist.append(2*ind+1)
            idxlist.append(2*ind+2)
    
    return embedding, treelabel
