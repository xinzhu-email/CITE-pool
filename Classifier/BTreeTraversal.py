#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 20:26:57 2019

@author: lqyair
"""
import sys
sys.path.append("./CITEsort_rna")

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from Visualize import visualize_node, visualize_pair, visualize_umap, visualize_2dim, visualize_tree

   
'''
from matplotlib import pyplot as plt
from matplotlib import cm,colors
from mpl_toolkits.axes_grid1 import axes_grid
#import seaborn as sns
import pdb
'''

class BTreeTraversal:
    
    def __init__(self,tree,method='bfs',min_BIC_node=None,nodename=None,tree_summary=None,leaf_summary=None,ll=None,n_components=None,save_min_BIC=False):
        
        #print('initializing...')
        
        self.tree = tree
        self.method = method
        self.save_min_BIC = save_min_BIC
        self.min_BIC_node = min_BIC_node
        if self.method == 'bfs':
            self.nodelist = self.levelOrderTraversal()
        if self.method == 'dfs':
            self.nodelist = self.preorderTraversal()

        # templist = list(range(len(self.nodelist)))
        # for node in self.nodelist:
        #     templist[node.ind] = node
        # self.nodelist = templist
        # if min_BIC_node!=None:
        #     self.bicminname = [str(x.ind)+'_'+'_'.join(x.key) for x in self.min_BIC_node]
        # self.nodename = [str(x.ind)+'_'+'_'.join(x.key) for x in self.nodelist]

        # print(self.bicminname)
        # print(self.nodename)
        # self.nodename = [str(i)+'_'+nodename_temp[i] for i in range(len(nodename_temp))]
        nodename_temp = ['_'.join(x.key) for x in self.nodelist]
        self.nodename = [str(i)+'_'+nodename_temp[i] for i in range(len(nodename_temp))]
        self.tree_summary, self.leaf_summary = self.summarize()
        if 'll' in self.tree.__dir__():
            self.ll = self.leaf_summary['ll'].sum()
            self.n_components = self.leaf_summary.shape[0]
        
      
    def summarize(self):
        if 'll' in self.tree.__dir__():
            tree_summary = pd.DataFrame({'Count':[len(x.indices) for x in self.nodelist],
                                         'Weight':[x.weight for x in self.nodelist],
                                         'Stop':[x.stop for x in self.nodelist],
                                         'll':[x.ll for x in self.nodelist]
                                         },index=self.nodename)
        else:
            tree_summary = pd.DataFrame({'Count':[len(x.indices) for x in self.nodelist] },index=self.nodename)  

        leaf_summary = tree_summary.loc[[x for x in self.nodename if x.split('_')[1]=='leaf'],:]
        # leaf_summary = tree_summary.loc[[i for i in range(len(self.nodename)) if self.nodename[i].split('_')[1]=='leaf'],:]
        leaf_summary = leaf_summary.sort_values(by='Count',ascending=False)
        
        return tree_summary,leaf_summary
    
    
    
    def get_ll(self):
        
        self.ll_tot = sum([x.ll for idx,x in enumerate(self.nodelist) if self.nodename[idx].split('_')[1]=='leaf'])
    
    
    def get_node(self,nodeID):
        return self.nodelist[nodeID]
 
    
    
    def generateLabel(self):
        """generate label file (binary matrix: Num.cell x Num.node, X_ij = 1 means cell i is attached to node j.)"""
        
        label = pd.DataFrame(np.zeros([len(self.tree.indices),len(self.nodename)]),index=self.tree.indices,columns=self.nodename)

        for i in range(len(self.nodename)):
            label.loc[self.nodelist[i].indices,self.nodename[i]] = 1

        return label
    
    
    def finetune_leaf(self, method, nodeID, label):
        for i in range(len(self.nodelist)):
            node = self.nodelist[i]
            node.ind = i
        if method == 'bfs':
            self.nodelist = self.levelOrderTraversal(finetune=True, nodeID=nodeID)
        else:
            self.nodelist = self.preorderTraversal()
        for i in range(len(self.nodelist)):
            node = self.nodelist[i]
            if self.nodelist[i].key == ('leaf',):
                label.loc[self.nodelist[i].indices,'Label'] = str(i)+'_leaf'
        return label

    
    def get_leaf_label(self, BIC_node=False, midnodelist=None, midnames=None, finetune=False, method='bfs',nodeID=None):
        """generate label (one column, indicating which leaf cells are assigned.)"""
        label = pd.DataFrame({'GEM':self.tree.indices,'Label':[None]*len(self.tree.indices)},index=self.tree.indices)
        
        if midnodelist != None:
            for i in range(len(midnodelist)):
                label.loc[midnodelist[i].indices,'Label'] = midnames[i]
            return label['Label']
        elif BIC_node:
            for i in range(len(self.bicminname)):
                label.loc[self.min_BIC_node[i].indices,'Label'] = self.bicminname[i]
        elif finetune:
            label = self.finetune_leaf(method,nodeID,label)
            return self.nodelist, label
        else:
            for i in range(len(self.nodename)):
                if self.nodename[i].split('_')[1] == 'leaf':
                    label.loc[self.nodelist[i].indices,'Label'] = self.nodename[i]

        return label

    def get_ll(self, nodeID):
        node = self.nodelist[nodeID]
        return node.score_dict
    
    def plot_node(self,data, nodeID, viz_dim = 1, **plot_para):
        """plot the specified node (default: savefig=False,savepath='.')"""
        if viz_dim == 1:
            visualize_node(data, node = self.nodelist[nodeID], nodename = self.nodename[nodeID], **plot_para)
        if viz_dim == 2:
            visualize_pair(data, node = self.nodelist[nodeID], nodename = self.nodename[nodeID],  **plot_para)
    

    def plot_node_umap(self, data, nodeID):
        node = self.nodelist[nodeID]
        nodelist = [x for x in self.nodelist if x.ind<=node.ind]
        nodenames = [str(x.ind)+'_'+'_'.join(x.key) for x in nodelist]
        label = self.get_leaf_label(midnodelist=nodelist, midnames=nodenames)
        visualize_umap(data, label=label)
    

    def plot_node_2dim_old(self, data,  nodeID, marker1, marker2):
        node = self.nodelist[nodeID]
        subdata1, subdata2 = data.loc[node.indices, marker1], data.loc[node.indices, marker2]
        label = pd.DataFrame(index=subdata1.index, columns=['label'])
        if node.key == 'leaf':
            label.loc[node.indices, 'label'] = 0

        else:
            
            label.loc[node.left.indices, 'label'] = 1
            label.loc[node.right.indices, 'label'] = 2
            # print(list(label['label']))
        visualize_2dim(subdata1, subdata2, label['label'])


    def plot_node_2dim_new(self, data, nodeID, marker1, marker2, hist=False, savefig=False):
        node = self.nodelist[nodeID]
        # print(node.key)
        if node.key[0] == 'leaf':
            subdata1, subdata2 = data.loc[node.indices, marker1], data.loc[node.indices, marker2]
            label = pd.DataFrame(index=subdata1.index, columns=['label'])
            label.loc[node.indices, 'label'] = 0
        else:
            subdata1, subdata2 = data.loc[node.left.indices+node.right.indices, marker1], data.loc[node.left.indices+node.right.indices, marker2]
            label = pd.DataFrame(index=subdata1.index, columns=['label'])
            label.loc[node.left.indices, 'label'] = 1
            label.loc[node.right.indices, 'label'] = 2
        visualize_2dim(subdata1, subdata2, label['label'],title='node_'+str(node.ind)+' cell types',hist=hist,savefig=savefig)
    
    def plot_leaf_size(self):
        
        leaf_size = self.leaf_summary['Count']
        leaf_prop = self.leaf_summary['Proportion']
        
        fig, ax1 = plt.subplots(1,2,figsize=(12,4))
        
        # plot number/proportion of cells in each leaf
        color = 'tab:red'
        ax1[0].set_xlabel('leaf',fontsize=20)
        ax1[0].set_ylabel('Proportion', color=color,fontsize=20)
        ax1[0].plot(range(len(leaf_prop)),leaf_prop, color=color,marker='o')
        if len(leaf_prop) >= 5:
            plt.xticks(np.arange(0, len(leaf_prop), len(leaf_prop)//5))
        else:
            plt.xticks(np.arange(0, len(leaf_prop), 1))
        ax1[0].tick_params(axis='y', labelcolor=color,labelsize=15)
        ax1[0].tick_params(axis='x', labelsize=15)
        ax1[0].set_title('Num. of cells in leaf',fontsize=20,pad=20)

        ax2 = ax1[0].twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Count', color=color,fontsize=20)  # we already handled the x-label with ax1
        ax2.plot(range(len(leaf_size)),leaf_size, color=color,marker='o')
        ax2.tick_params(axis='y', labelcolor=color,labelsize=15)

        
        # plot cumulative number/proportion of cells in each leaf
        color = 'tab:red'
        ax1[1].set_xlabel('leaf',fontsize=20)
        ax1[1].set_ylabel('Proportion', color=color,fontsize=20)
        ax1[1].plot(range(len(leaf_prop)),leaf_prop.cumsum(), color=color,marker='o')
        if len(leaf_prop) >= 5:
            plt.xticks(np.arange(0, len(leaf_prop), len(leaf_prop)//5))
        else:
            plt.xticks(np.arange(0, len(leaf_prop), 1))
        ax1[1].tick_params(axis='y', labelcolor=color,labelsize=15)
        ax1[1].tick_params(axis='x', labelsize=15)
        ax1[1].set_title('Cumulative num. of cell in leaf',fontsize=20,pad=20)

        ax2 = ax1[1].twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Count', color=color,fontsize=20)  # we already handled the x-label with ax1
        ax2.plot(range(len(leaf_size)),leaf_size.cumsum(), color=color,marker='o')
        ax2.tick_params(axis='y', labelcolor=color,labelsize=15)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
        
    
    
    '''
    def track_marker(self,data,n_big_leaf,**plot_para):
        """track marker distributions in big leafs. (at most 12 leafs. default para: savefig=False,outpath='.')"""
        n_big_leaf = min(n_big_leaf,12) 
        
        savefig = plot_para.get('savefig',False)
        outpath = plot_para.get('outpath','.')
             
        big_leaf = self.leaf_summary.index.values.tolist()[0:n_big_leaf]
        markers = data.columns.values.tolist()
        node_plot = [self.nodename[0]] + big_leaf
        
        cmap = cm.get_cmap('Set3')
        col_dic = dict(zip(big_leaf,[colors.to_hex(cmap(i)) for i in range(len(big_leaf))]))
        col_dic[node_plot[0]] = '#999999' # col for all cells
        
        nrows = np.ceil(len(markers)/5)
        ncols = 5
        naxes = len(node_plot)
        f = plt.figure(figsize=(10, naxes))
        for i, m in enumerate(markers):
            ag = axes_grid.Grid(f, (nrows, ncols, i+1), (naxes, 1), axes_pad=0)
            for j in range(naxes):
                leaf_idx = int(node_plot[j].split('_')[0])
                ag[j].hist(data.loc[self.nodelist[leaf_idx].indices,m], 
                           color = col_dic[node_plot[j]], density = True, bins = 'auto')
                ag[j].axvline(0, linestyle='dashed', linewidth=2)
                ag[j].yaxis.set_ticks([])
                ag[j].xaxis.set_ticks([])
                if j%naxes == 0:
                    ag[j].set_title(markers[i],fontsize=15)
                if i%ncols == 0:
                    ag[j].set_ylabel(str(leaf_idx),fontsize=12)

        plt.subplots_adjust(wspace=0.1, hspace=0.3)
        if savefig == True:
            plt.savefig(outpath+'/track_marker_in_big_leafs.png')
        plt.show()
    '''
    
    
    # dfs
    def preorderTraversal(self, cutID):

        node = self.tree
        if node is None:
            return

        nodelist = []
        myStack = []

        while node or myStack:
            while node:
                nodelist.append(node)
                myStack.append(node)
                node = node.left
            node = myStack.pop()
            node = node.right   

        return nodelist


    def all_BIC(self, queue):
        ll, n_features = 0, 0
        n_sample = len(self.tree.indices)
        node_name = []
        for key,node in queue.items():
            ll = ll + node.ll * node.weight 
            n_features = n_features + len(node.all_clustering_dic)
            node_name.append(str(key))
            if len(node.all_clustering_dic) >2:
                print(node.all_clustering_dic)
        cov_params = len(queue) * n_features * (n_features + 1) / 2.0
        mean_params = n_features * len(queue)
        n_param = int(cov_params + mean_params + len(queue) - 1)
        bic = -2 * ll * n_sample + n_param * np.log(n_sample)
        print(node_name) 
        return bic, node_name

    # bfs
    def _levelOrderTraversal(self, finetune=False): 
        #print('bfs...')
        node = self.tree
        if node is None: 
            return

        queue = [] 
        nodelist = []
        BIC_list, n_cluster_list = [], []
        bic_queue = {}
        leaf_ind = 0

        queue.append(node) 
        nodelist.append(node)
        bic_queue[leaf_ind] = node
        node_dict = {}
        
        while(len(queue) > 0): 
            node = queue.pop(0)  
            name = str(node)
            if self.save_min_BIC and node.left is not None and node.right is not None:
                bic_queue = {key:val for key, val in bic_queue.items() if str(val) != name}

            if node.left is not None:                
                nodelist.append(node.left)
                queue.append(node.left)
                # leaf_ind = leaf_ind + 1
                # bic_queue[leaf_ind] = node.left

            if node.right is not None: 
                nodelist.append(node.right)
                queue.append(node.right)
                # leaf_ind = leaf_ind + 1
                # bic_queue[leaf_ind] = node.right
                if self.save_min_BIC:
                    bic, node_name = self.all_BIC(bic_queue)
                    BIC_list.append(bic)
                    n_cluster_list.append(len(node_name))
                    node_dict[bic] = node_name

        if self.save_min_BIC:
            min_node = min(list(node_dict.keys()))
            min_list = node_dict[min_node]
            print(n_cluster_list, BIC_list)
            plt.plot(n_cluster_list, BIC_list)
            plt.savefig('BIC_as_split.png')
            # print('nodelist0',nodelist)
            return nodelist, min_list
        
        else:
            return nodelist


    def levelOrderTraversal(self, root=None, finetune=False, nodeID=None): 
        #print('bfs...')
        if root != None:
            node = root
        else:
            node = self.tree
        if node is None: 
            return

        queue = [] 
        nodelist = []

        queue.append(node) 
        nodelist.append(node)
        cutID = []

        while(len(queue) > 0): 
            node = queue.pop(0)   
            if finetune:
                node, cutID = bfs_finetune(node, nodeID, cutID)
                nodelist[node.ind] = node

            if node.left is not None: 
                nodelist.append(node.left)
                queue.append(node.left)

            if node.right is not None: 
                nodelist.append(node.right)
                queue.append(node.right) 
            
            

        return nodelist
 
def bfs_finetune(node, nodeID, cutID):
    if node.ind in nodeID:
        node.key = ('leaf',)
        cutID.append(node.left.ind)
        cutID.append(node.right.ind)
        return node, cutID
            
    if node.ind in cutID:
        node.key = ('cutleaf',)
        if node.left is not None:
            cutID.append(node.left.ind)
            cutID.append(node.right.ind)
    return node, cutID


