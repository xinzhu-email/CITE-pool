#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:51:16 2019

@author: lqyair
"""

#import pandas as pd
import numpy as np
#from BTreeTraversal import BTreeTraversal
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import matplotlib
import scanpy as sc

#node = traversal.get_node(0)
#nodename = traversal.nodename[0]

def visualize_node(data,node,nodename,**plot_para):
    
    #matplotlib.rcParams['figure.dpi'] = 200

    # plot_para: savefig, outpath, 
    savefig = plot_para.get('savefig',False)
    savepath = plot_para.get('savepath','.')
    savename = plot_para.get('savename','.')
    
    current_indices = list(set(node.indices)&set(data.index)) 
    # current_indices =list(set(current_indices)&set(node.embedding.index))
    # # print(current_indices)
    # node_data = node.embedding.loc[current_indices,:]
    if len(node.embedding) != 0:
        current_indices = list(set(current_indices)&set(node.embedding.index))
        # print(current_indices)
        node_data = node.embedding.loc[current_indices,:]
    else:
        node_data = data.loc[current_indices,:]

    node_data['artificial'] = 0
    if node.key == ('artificial',):
        print('artificial')
        node_data['artificial'] = node.artificial_w
        node.all_clustering_dic = {1:node.artificial_w}
    
    
    # plt.figure(figsize=(12,((len(node.all_clustering_dic[1])-1)//5+1)*2), dpi=70)
    plt.style.use('seaborn-white')
    #ax.tick_params(axis='both', which='major', labelsize=10)

    
    if node.key == ('leaf',) and node_data.shape[0] <= 20 :
        markers = node_data.columns.values.tolist()
        for i in range(len(markers)):
            X = node_data.loc[:,markers[i]].values.reshape(-1, 1)
            plt.figure(figsize=(12,((data.shape[1]-1)//5+1)*2+0.5), dpi=70)
            plt.subplot( (len(markers)-1)//5+1,5,i+1 )
            plt.hist(X,bins=30, density = True, color = "lightblue")
            plt.ylabel('density',fontsize=10)
            plt.title( markers[i],fontsize=12)

    else:
        all_clustering = node.all_clustering_dic[1]
        markers = list(all_clustering.keys())
        plt.figure(figsize=(12,((len(node.all_clustering_dic[1])-1)//5+1)*2+0.5), dpi=70)
        for i in range(len(markers)):
            
            X = node_data.loc[:,markers[i]].values.reshape(-1, 1)
            
            
            plt.subplot( (len(markers)-1)//5+1,5,i+1 )

            bins = np.linspace(min(X),max(X),500)
            cols = ['r','g','b','c','m','y','darkorange','lightgreen','lightpink','darkgray']
    
            bp_ncluster = int(all_clustering[markers[i]]['bp_ncluster'])
            mp_ncluster = 1 # default
            weights = all_clustering[markers[i]]['bp_pro']
            means = all_clustering[markers[i]]['bp_mean']
            sigmas = np.sqrt(all_clustering[markers[i]]['bp_Sigma'])
            
            y = np.zeros((len(bins),bp_ncluster))
            
            for k in range(bp_ncluster):
                y[:,k] = (weights[k] * stats.norm.pdf(bins, means[k], sigmas[k]))[:,0]
                plt.plot(bins,y[:,k],linewidth=0.6,color='black')

            if bp_ncluster > 1:
                mp_ncluster = all_clustering[markers[i]]['mp_ncluster']
                mergedtonumbers = all_clustering[markers[i]]['mergedtonumbers']
                
                for k in range(mp_ncluster):
            
                    merged_idx = [idx for idx,val in enumerate(mergedtonumbers) if val == k]
                    y_merged = np.apply_along_axis(sum,1,y[:,merged_idx])
        
                    plt.plot(bins,y_merged,cols[k],linewidth=2,linestyle='-.')
                    
            subfig_title = '_'.join(markers[i])+' ('+str(mp_ncluster)+'|'+str(bp_ncluster)+') ' + str(round(all_clustering[markers[i]]['similarity_stopped'],2))
            
            if markers[i] == node.key:
                plt.title( subfig_title,fontsize=12,color='red')
            else: 
                plt.title( subfig_title,fontsize=12,color='darkgrey' if mp_ncluster <= 1 else 'black')
                
            plt.hist(X,bins=30, density = True, color = "lightblue")
            plt.ylabel('density',fontsize=10)
    
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.4,wspace=0.45)
    plt.suptitle(nodename+' | '+str(len(current_indices))+' cells',fontsize=15,color="darkblue")
    plt.subplots_adjust(top=0.85)
    #plt.savefig(savepath+'/visualize_node.png')
    if savefig == True:
        plt.savefig(savepath+'/'+savename+'_'+nodename+'.png') 
    plt.show()   
    
    




#import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def visualize_pair(data,node,nodename,**plot_para):
    
    # plot_para: savefig, outpath, 
    savefig = plot_para.get('savefig',False)
    savepath = plot_para.get('savepath','.')
    savename = plot_para.get('savename','.')
    
    all_clustering = node.all_clustering_dic[2]
    marker_pairs = list(all_clustering.keys())
    current_indices = node.indices 
    

    if len(node.embedding) != 0:
        current_indices =node.embedding.index
        # print(current_indices)
        data = node.embedding.loc[current_indices,:]
    else:
        data = data.loc[current_indices,:]
    data['artificial'] = 0
    # print(data)
    if node.key == ('artificial',):
        print('artificial')
        data['artificial'] = node.artificial_w
        node.all_clustering_dic = {1:node.artificial_w}

    plt.figure(figsize=(12,((len(marker_pairs)-1)//5+1)*2.5), dpi=96)
    sns.set_style("white")
    
    for i in range(len(marker_pairs)):
    
        marker1,marker2 = marker_pairs[i]
        X1 = data.loc[current_indices, marker1]
        X2 = data.loc[current_indices, marker2]
        
        bp_clustering = all_clustering[marker_pairs[i]]['bp_clustering']
        mp_clustering = all_clustering[marker_pairs[i]]['mp_clustering']
        
        mp_ncluster = all_clustering[marker_pairs[i]]['mp_ncluster']
        bp_ncluster = all_clustering[marker_pairs[i]]['bp_ncluster']

        # print(node.embedding.index.value_counts())
        data_pair = pd.DataFrame({marker1:X1,marker2:X2,
                              'bp':bp_clustering,
                              'mp':mp_clustering},index=current_indices)

        plt.subplot( (len(marker_pairs)-1)//5+1,5,i+1 )
        
        #shapes = ['s','X','+']
        #markers = dict(zip(np.unique(mp_clustering),[shapes[idx] for idx in range(mp_ncluster)]))
        sns.scatterplot(x=marker1, y=marker2,hue="bp",style="mp",
                        data=data_pair,s=10,legend=False)

        marker_pair_joint = marker_pairs[i][0]+'_'+marker_pairs[i][1]
        subfig_title = marker_pair_joint+' ('+str(mp_ncluster)+'|'+str(bp_ncluster)+') ' + str(round(all_clustering[marker_pairs[i]]['similarity_stopped'],2))
        
        if marker_pairs[i] == node.key:
            plt.title( subfig_title,fontsize=12,color='red')
        else: 
            plt.title( subfig_title,fontsize=12,color='darkgrey' if mp_ncluster <= 1 else 'black')
            
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.6,wspace=0.45)
    plt.suptitle(nodename+' | '+str(len(current_indices))+' cells',fontsize=15,color="darkblue")
    plt.subplots_adjust(top=0.85)
    #plt.savefig(savepath+'/visualize_node.png')
    if savefig == True:
        plt.savefig(savepath+'/'+savename+'_'+nodename+'.png') 
    plt.show() 
    
  
def visualize_umap(data, label):
    import umap
    import umap.plot
    mapper = umap.UMAP().fit(data)
    umap.plot.points(mapper, labels=label)


def visualize_2dim(dim1, dim2, label, title='Split Cells',hist=False, savefig=False):
    # print(list(label))
    # fig, ax = plt.subplots()
    # n, bins, patches = plt.hist(dim1, bins=100, density=1, alpha=0.5, orientation='vertical')
    # n, bins, patches = plt.hist(dim2, bins=100, density=1, alpha=0.5, orientation='horizontal') 
    if hist:  
        sns.distplot(dim1,kde=True,color='green',vertical=False)
        sns.distplot(dim2,kde=True,color='green',vertical=True)
    plt.scatter(dim1, dim2, c=list(label), cmap='tab20b', s=2)
    plt.xlabel(dim1.name)
    plt.ylabel(dim2.name)
    plt.title(title)
    # legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    # a,b=scatter.legend_elements()
    # b=['$\\mathdefault {left}$',
    #     '$\\mathdefault{right}$']
    # legend1 = ax.legend(a,b, title="Classes")

    # plt.title()
    if savefig:
        plt.savefig('.png') 
    plt.show()


def plot_keymarker(data,traversal,node_ID,dpi=5,savepath=None):

    node = traversal.get_node(node_ID)
    
    current_indices = node.indices
    node_data = data.loc[current_indices,:]
        
    marker_dkey = node.key
    
    if len(marker_dkey) == 1:
        marker = marker_dkey[0]
    
        clustering = node.all_clustering_dic[1][marker_dkey]
        
        X = node_data.loc[:,marker_dkey].values.reshape(-1, 1)
        
        bins = np.linspace(min(X),max(X),500)
        cols = ['firebrick','navy','lightgreen','darkorange']
        
        bp_ncluster = int(clustering['bp_ncluster'])
        mp_ncluster = 1 # default
        weights = clustering['bp_pro']
        means = clustering['bp_mean']
        sigmas = np.sqrt(clustering['bp_Sigma'])
        
        y = np.zeros((len(bins),bp_ncluster))
        
        #plt.figure(figsize=(4,3), dpi=24)
        plt.style.use('seaborn-white')
        matplotlib.rcParams['axes.linewidth'] = 0.1
        fig, ax = plt.subplots(figsize=(4,3), dpi=dpi)
        
        for k in range(bp_ncluster):
            y[:,k] = (weights[k] * stats.norm.pdf(bins, means[k], sigmas[k]))[:,0]
            plt.plot(bins,y[:,k],linewidth=0.05,color='black')
        
        mp_ncluster = clustering['mp_ncluster']
        
        # red -- component with bigger mean
        mp_means = []
        for i in range(mp_ncluster):
            mp_means.append(np.mean(X[np.array(clustering['mp_clustering'])==i,0]))
        
        idx = list(np.argsort(mp_means))
        idx.reverse()
        
        mergedtonumbers = clustering['mergedtonumbers']
        
        for k in range(mp_ncluster):
        
            merged_idx = [ii for ii,val in enumerate(mergedtonumbers) if val == k]
            y_merged = np.apply_along_axis(sum,1,y[:,merged_idx])
        
            plt.plot(bins,y_merged,cols[idx.index(k)],linewidth=0.8,linestyle='--')
        
        #subfig_title = str(node_ID) + '_'+ marker# +' ('+str(mp_ncluster)+'|'+str(bp_ncluster)+') ' + str(round(clustering['similarity_stopped'],2))
        
        plt.hist(X,bins=30, density = True, color = "lightblue",linewidth=0)
        
        #plt.title( subfig_title,fontsize=16)
        plt.ylabel('density',fontsize=18)
        plt.xlabel(marker,fontsize=18)
        plt.subplots_adjust(top=0.8, bottom=0.2, left=0.15, right=0.9, hspace=0.2,wspace=0.8)
        ax.tick_params(axis='both', which='major', labelsize=10)
        if savepath is not None:
            plt.savefig(savepath+'/'+str(node_ID)+'_'+marker+'.pdf') 
        plt.show()
    
    if len(marker_dkey) == 2:
        
        marker1,marker2 = marker_dkey
        
        subdata = node_data.loc[:,marker_dkey]
        clustering = node.all_clustering_dic[2][marker_dkey]
        cols = ['firebrick','navy','lightgreen','darkorange']
    
        mp_ncluster = clustering['mp_ncluster']
        #mp_clustering = clustering['mp_clustering']
        componentidx = np.array(clustering['mp_clustering'])==1
        p1_mean = node_data.loc[componentidx,marker_dkey].mean()
        p2_mean = node_data.loc[~componentidx,marker_dkey].mean()
        
        p1_cosine = sum(p1_mean)/np.sqrt(sum(p1_mean**2))
        p2_cosine = sum(p2_mean)/np.sqrt(sum(p2_mean**2))
        
        plt.style.use('seaborn-white')
        matplotlib.rcParams['axes.linewidth'] = 0.1
        fig, ax = plt.subplots(figsize=(4,3), dpi=dpi)
        
        if p1_cosine > p2_cosine:
            plt.scatter(subdata.loc[componentidx,marker1],subdata.loc[componentidx,marker2],c='firebrick',s=1)
            plt.scatter(subdata.loc[~componentidx,marker1],subdata.loc[~componentidx,marker2],c='navy',s=1)
        else:
            plt.scatter(subdata.loc[componentidx,marker1],subdata.loc[componentidx,marker2],c='navy',s=1)
            plt.scatter(subdata.loc[~componentidx,marker1],subdata.loc[~componentidx,marker2],c='firebrick',s=1)
        
        sns.kdeplot(subdata[marker1], subdata[marker2], ax=ax, n_levels = 5, cmap = 'Wistia')

        plt.xlabel(marker1,fontsize=18)
        plt.ylabel(marker2,fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.subplots_adjust(top=0.8, bottom=0.2, left=0.15, right=0.9, hspace=0.2,wspace=0.8)
        if savepath is not None:
            plt.savefig(savepath+'/'+str(node_ID)+'_'+marker1+'_'+marker2+'.pdf') 

        plt.show()
        


from subprocess import call
def visualize_tree(root,data,outpath,filename,compact=False,rnadata=None):
    """write tree structure into .dot and .png files."""
    
    # open a file, and design general format
    tree_dot = open(outpath+'/'+filename+'.dot','w') 
    tree_dot.writelines('digraph Tree {')
    tree_dot.writelines('node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;')
    tree_dot.writelines('edge [fontname=helvetica] ;')


    #tree_dot = _write_tree_bfs(root,tree_dot)
        # Base Case 
    if root is None: 
        return
    
    
    # Create an empty queue for level order traversal 
    queue = [] 
    nodelist = []
    idxStack = []
    indStack = []
    
    tot_cells = len(root.indices)
    #means_in_root = root.marker_summary['mean']
    #stds_in_root = root.marker_summary['std']
    # rawdata = data
    # if len(data) == 0:
    #     data = root.embedding

    # data['artificial'] = 0
    # print(data['artificial'])
    means_in_root = data.mean(axis = 0) 
    means_in_root['artificial'] = 0
    means_in_root = pd.concat([means_in_root,pd.Series(data=np.zeros(10),index=['CC_'+str(i+1) for i in range(10)])],axis=0)
    stds_in_root = data.std(axis = 0)
    stds_in_root['artificial'] = 1
    stds_in_root = pd.concat([stds_in_root,pd.Series(data=np.ones(10),index=['CC_'+str(i+1) for i in range(10)])],axis=0)
    markers = means_in_root.index.values.tolist()
    
    # auxiliary parameters for color display
    branch_col = pd.Series({1:'#ffccccff',2:'#ffff99ff',3:'#CC99CC',4:'#99CCFF'})   
    leaf_col = matplotlib.colors.Normalize(vmin=0, vmax=np.log(tot_cells))
    
    node = root
    
    # Enqueue Root and initialize height 
    queue.append(node) 
    
    i = 0
    #print(str(node.ind)+'_'+root.key)
    if node.key == ('artificial',):
        tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.key)+ \
                        '\\nNum: '+str(len(node.indices))+ \
                        '",fillcolor="#ff9966ff",fontsize=25];')  
    else:
        all_clustering = node.all_clustering_dic[len(node.key)]
        bp_ncluster = all_clustering[node.key]['bp_ncluster']
        mp_ncluster = all_clustering[node.key]['mp_ncluster']
        tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.key)+ \
                            '\\nNum: '+str(len(node.indices))+ \
                            '\\n('+str(mp_ncluster)+'|'+str(bp_ncluster)+')",fillcolor="#ff9966ff",fontsize=25];')  
    nodelist.append(node.key)
    idxStack.append(i)
    # indStack.append(node.ind)
    
    while(len(queue) > 0): 
        # Print front of queue and remove it from queue 
        node = queue.pop(0) 
        # ind = indStack.pop(0)
        idx = idxStack.pop(0)
        ######
        # if node.key[0] == 'not separable' or node.key[0] == 'partition small size':
        ######
        # if node.key[0][:2] == 'CC':
        #     data = root.embedding
        # else:
        #     data = rawdata

        if node.key == ('artificial',):
            markers = [('artificial',)]
            # means_in_root['artificial'] = 0
            # adata = rnadata[node.indices,node.artificial_w.index].copy()
            # sc.pp.normalize_total(adata, target_sum=1e4)
            # sc.pp.log1p(adata)
            # data.loc[node.indices,'artificial'] = np.dot(adata.X.toarray(),node.artificial_w)
            # data['artificial'] = 0
            # data.loc[list(node.indices), 'artificial'] = node.artificial_w
            stds_in_root['artificial'] = node.artificial_w.std()
            # print(data['artificial'])

                
        # left child 
        if node.left is not None and node.left.key != ('cutleaf',): 
            nodelist.append(node.left.key)
            queue.append(node.left)
            i = i + 1
            idxStack.append(i)
            # indStack.append(node.left.ind)
            #print(str(i)+'_'+node.left.key)
            
            percent = str(round(len(node.left.indices)/tot_cells*100,2))+'%'
            mean_temp = node.mean_l
            
            # mean_temp['artificial'] = data['artificial'].mean()
            
            if node.left.key == ('leaf',):
                # left leaf node     
                # print(node.left.ind,node.left.key)  
                if compact:
                    offset_in_leaf = ''
                else:
                    temp = (mean_temp - means_in_root)/stds_in_root
                    offset_in_leaf = '\n' + markers[0]+': '+str(round(temp[markers[0]],2))
                    for k in range(1,len(markers)):
                        offset_in_leaf = offset_in_leaf + '\n' + markers[k]+': '+ str(round(temp[markers[k]],2))
                    
                col =  matplotlib.colors.to_hex(matplotlib.cm.Greens(leaf_col(np.log(len(node.left.indices)))))
                tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.left.key)+'\\n'+ \
                                    str(len(node.left.indices))+ ' ('+percent+')\\n'+ \
                                    offset_in_leaf+'",fillcolor="'+col+'",fontsize=20];')
            elif node.left.key != ('cutleaf',):
                # left branch node
                # print(node.left.key)
                if node.left.key == ('artificial',):
                    tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.left.key)+'\\n'+ \
                                    str(len(node.left.indices))+' ('+percent+')\\n'+ \
                                    '",fillcolor="'+branch_col[len(node.left.key)]+'",fontsize=25];')
                else:
                    all_clustering = node.left.all_clustering_dic[len(node.left.key)]
                    bp_ncluster = all_clustering[node.left.key]['bp_ncluster']
                    mp_ncluster = all_clustering[node.left.key]['mp_ncluster']
                    
                    tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.left.key)+'\\n'+ \
                                        str(len(node.left.indices))+' ('+percent+')\\n'+ \
                                        '('+str(mp_ncluster)+'|'+str(bp_ncluster)+')",fillcolor="'+branch_col[len(node.left.key)]+'",fontsize=25];')

            # edge from parent to left node
            offset = ''
            # print(mean_temp, means_in_root, stds_in_root)
            # if node.left.key in data.columns:
            #     mean_temp = data.loc[node.left.indices,:].mean() 
            # elif node.key in node.left.mean.index:
            #     mean_temp = node.left.mean
            # else:
            #     mean_temp = pd.concat([data.loc[node.left.indices,:].mean(), node.left.mean],axis=0)
            # if len(root.left.embedding) != 0:
            #     mean_temp = pd.concat([data.loc[node.left.indices,:].mean(), node.left.mean],axis=0)
            # print(mean_temp, means_in_root, stds_in_root)
            if nodelist[idx][0][:2]== 'CC' and len(nodelist[idx])>1:
                val = (mean_temp.values - means_in_root['artificial'])/stds_in_root['artificial']
                offset = offset + str(round(val[0],2))
            else:
                for m in nodelist[idx]:
                    # print(mean_temp, means_in_root, stds_in_root)
                    val = (mean_temp[m] - means_in_root[m])/stds_in_root[m]
                    offset = offset + str(round(val,2))+'\n'
                
            #print(str(idx)+'->'+str(i))
            tree_dot.writelines(str(idx)+' -> '+str(i)+ ' [labeldistance=3, label = "'+offset+'",fontsize=25, color='+['black','red'][node.where_dominant=='left']+\
                                ', style='+['solid','bold'][node.where_dominant=='left']+'];')

        # right child 
        if node.right is not None and node.right.key != ('cutleaf',): 
            nodelist.append(node.right.key)
            queue.append(node.right) 
            i = i + 1
            idxStack.append(i)
            # indStack.append(node.right.ind)
            #print(str(i)+'_'+node.right.key)
            
            percent = str(round(len(node.right.indices)/tot_cells*100,2))+'%'
            mean_temp = None
            mean_temp = node.mean_r
            # if node.right.key[0] in data.columns:
            #     mean_temp = data.loc[node.right.indices,:].mean() 
            # elif node.key[0] in node.right.mean.index:
            #     mean_temp = node.right.mean
            # else:
            #     mean_temp = pd.concat([data.loc[node.right.indices,:].mean(), node.right.mean],axis=0)
             
            # mean_temp['artificial'] = data['artificial'].mean()

            if node.right.key == ('leaf',):
                # print(node.right.ind,node.right.key)
                # right leaf node
                if compact:
                    offset_in_leaf = ''
                else:
                    temp = (mean_temp - means_in_root)/stds_in_root
                    offset_in_leaf = '\n' + markers[0]+': '+str(round(temp[markers[0]],2))
                    for k in range(1,len(markers)):
                        offset_in_leaf = offset_in_leaf + '\n' + markers[k]+': '+ str(round(temp[markers[k]],2))

                col =  matplotlib.colors.to_hex(matplotlib.cm.Greens(leaf_col(np.log(len(node.right.indices)))))
                tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.right.key)+'\\n'+ \
                                    str(len(node.right.indices))+ ' ('+percent+')'+'\\n'+ \
                                    offset_in_leaf+'",fillcolor="'+col+'",fontsize=20];')

            elif node.right.key != ('cutleaf',):
                if node.right.key == ('artificial',):
                    tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.right.key)+'\\n'+ \
                                    str(len(node.right.indices))+' ('+percent+')\\n'+ \
                                    '",fillcolor="'+branch_col[len(node.right.key)]+'",fontsize=25];')
                # right branch node
                # print(node.right.key)
                else:
                    # print(node.right.key)
                    all_clustering = node.right.all_clustering_dic[len(node.right.key)]
                    bp_ncluster = all_clustering[node.right.key]['bp_ncluster']
                    mp_ncluster = all_clustering[node.right.key]['mp_ncluster']
                    
                    tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.right.key)+'\\n'+ \
                                        str(len(node.right.indices))+' ('+percent+')\\n'+ \
                                        '('+str(mp_ncluster)+'|'+str(bp_ncluster)+')",fillcolor="'+branch_col[len(node.right.key)]+'",fontsize=25];')

            # edge from parent to right node
            offset = ''
            # # val = (mean_temp - means_in_root[m])/stds_in_root[m]
            # offset = str(round(mean_temp,2))
            # if len(root.right.embedding) != 0:
            #     mean_temp = pd.concat([data.loc[node.right.indices,:].mean(), node.right.mean],axis=0)
            if nodelist[idx][0][:2]== 'CC' and len(nodelist[idx])>1:
                val = (mean_temp.values - means_in_root['artificial'])/stds_in_root['artificial']
                offset = offset + str(round(val[0],2))
            else:
                for m in nodelist[idx]:
                    val = (mean_temp[m] - means_in_root[m])/stds_in_root[m]
                    offset = offset + str(round(val,2))+'\n'
                # print(m,str(round(val,2)))
            # print(str(idx)+'->'+str(node.ind))
            tree_dot.writelines(str(idx)+' -> '+str(i)+' [labeldistance=3, label = "'+offset+'",fontsize=25, color='+['black','red'][node.where_dominant=='right']+ \
                                ', style='+['solid','bold'][node.where_dominant=='right']+'];')
    
    # main body is completed
  
    tree_dot.writelines('}')
    tree_dot.close()

    # Convert to png using system command (requires Graphviz)
    import os
    print(os.getcwd())
    call(['dot', '-Tpdf', outpath+'/'+filename+'.dot', '-o', outpath+'/'+filename+'.pdf', '-Gdpi=100'])
    
    
    # Display in jupyter notebook
    #Image(filename = outpath+'/GatingTree.png')

def visualize_modeltree(root,outpath,filename):
    """write tree structure into .dot and .png files."""
    
    # open a file, and design general format
    tree_dot = open(outpath+'/'+filename+'.dot','w') 
    tree_dot.writelines('digraph Tree {')
    tree_dot.writelines('node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;')
    tree_dot.writelines('edge [fontname=helvetica] ;')


    #tree_dot = _write_tree_bfs(root,tree_dot)
        # Base Case 
    if root is None: 
        return
    
    
    # Create an empty queue for level order traversal 
    queue = [] 
    nodelist = []
    idxStack = []
    indStack = []
    

    tot_cells = [root.val_cnt]
    #means_in_root = root.marker_summary['mean']
    #stds_in_root = root.marker_summary['std']
    # data['artificial'] = 0
    # print(data['artificial'])
    # means_in_root = data.mean(axis = 0) 
    # stds_in_root = data.std(axis = 0)
    # stds_in_root['artificial'] = 1
    # markers = means_in_root.index.values.tolist()
    
    # auxiliary parameters for color display
    branch_col = pd.Series({1:'#ffccccff',2:'#ffff99ff',3:'#CC99CC',4:'#99CCFF'})   
    print(tot_cells)
    leaf_col = matplotlib.colors.Normalize(vmin=0, vmax=np.log(tot_cells))
    
    node = root
    
    # Enqueue Root and initialize height 
    queue.append(node) 
    
    i = 0
    #print(str(node.ind)+'_'+root.key)
    # all_clustering = node.all_clustering_dic[len(node.key)]
    # bp_ncluster = all_clustering[node.key]['bp_ncluster']
    # mp_ncluster = all_clustering[node.key]['mp_ncluster']
    ndata = str(node.val_cnt)
    if len(node.indices)<=4:
        training_datasets = str(node.indices)
    else:
        training_datasets = str(len(node.indices))
    tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.key)+ \
                        '\\n'+ ndata +'\\n'+training_datasets+'",fillcolor="#ff9966ff",fontsize=25];')  
    nodelist.append(node.key)
    idxStack.append(i)
    # indStack.append(node.ind)
    
    while(len(queue) > 0): 
        # Print front of queue and remove it from queue 
        node = queue.pop(0) 
        # ind = indStack.pop(0)
        idx = idxStack.pop(0)
        # if node.key == ('artificial',):
        #     markers = [('artificial',)]
        #     # means_in_root['artificial'] = 0
        #     adata = rnadata[node.indices,node.artificial_w.index].copy()
        #     sc.pp.normalize_total(adata, target_sum=1e4)
        #     sc.pp.log1p(adata)
        #     data.loc[node.indices,'artificial'] = np.dot(adata.X.toarray(),node.artificial_w)
        #     stds_in_root['artificial'] = data.loc[node.indices,'artificial'].std()
            # print(data['artificial'])

                
        # left child 
        if node.left is not None and node.left.key != ('cutleaf',): 
            nodelist.append(node.left.key)
            queue.append(node.left)
            i = i + 1
            idxStack.append(i)
            # indStack.append(node.left.ind)
            #print(str(i)+'_'+node.left.key)
            
            # percent = str(round(len(node.left.indices)/tot_cells*100,2))+'%'
            # mean_temp = data.loc[node.left.indices,:].mean() 
            
            if node.left.key == ('leaf',):                   
                col =  matplotlib.colors.to_hex(matplotlib.cm.Greens(leaf_col(np.log(node.left.val_cnt))))
                tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.left.key)+'\\n'+ \
                                    str(node.left.val_cnt)+'",fillcolor="'+col+'",fontsize=20];')
            elif node.left.key != ('cutleaf',):
                if len(node.left.indices)<=4:
                    ndata = str(node.left.indices)
                else:
                    ndata = str(len(node.left.indices))
                tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.left.key)+'\\n'+ \
                                    str(node.left.val_cnt)+'\\n'+ ndata +'",fillcolor="'+branch_col[len(node.left.key)]+'",fontsize=25];')

            # edge from parent to left node
            offset = ''
            if nodelist[idx][0][:2]== 'CC' and len(nodelist[idx])>1:
                val = node.mean_l.values
                offset = offset + str(round(val[0],2))
            else:
                for m in nodelist[idx]:
                    # val = (mean_temp[m] - means_in_root[m])/stds_in_root[m]
                    # print(node.mean_l, m)
                    val = node.mean_l[m]#/(node.mean_r[m]-node.mean_l[m])
                    offset = offset + str(round(val,2))+'\n'
            # offset = str(round(node.mean_l/(node.mean_r-node.mean_l),2))
            #print(str(idx)+'->'+str(i))
            tree_dot.writelines(str(idx)+' -> '+str(i)+ ' [labeldistance=3, label = "'+offset+'",fontsize=25, color='+['black','red'][node.where_dominant=='left']+\
                                ', style='+['solid','bold'][node.where_dominant=='left']+'];')

        # right child 
        if node.right is not None and node.right.key != ('cutleaf',): 
            nodelist.append(node.right.key)
            queue.append(node.right) 
            i = i + 1
            idxStack.append(i)
            # indStack.append(node.right.ind)
            #print(str(i)+'_'+node.right.key)
            
            # percent = str(round(len(node.right.indices)/tot_cells*100,2))+'%'
            # mean_temp = data.loc[node.right.indices,:].mean() 

            if node.right.key == ('leaf',):  
                col =  matplotlib.colors.to_hex(matplotlib.cm.Greens(leaf_col(np.log(node.right.val_cnt))))
                tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.right.key)+'\\n'+ \
                                    str(node.right.val_cnt)+ '",fillcolor="'+col+'",fontsize=20];')

            elif node.right.key != ('cutleaf',):
                if len(node.right.indices)<=4:
                    ndata = str(node.right.indices)
                else:
                    ndata = str(len(node.right.indices))
                tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.right.key)+'\\n'+ \
                                    str(node.right.val_cnt)+'\\n'+ ndata +'",fillcolor="'+branch_col[len(node.right.key)]+'",fontsize=25];')

            # edge from parent to right node
            offset = ''
            if nodelist[idx][0][:2]== 'CC' and len(nodelist[idx])>1:
                val = node.mean_r.values
                offset = offset + str(round(val[0],2))
            else:
                for m in nodelist[idx]:
                    # print(node.mean_r, m)
                    # val = (mean_temp[m] - means_in_root[m])/stds_in_root[m]
                    val = node.mean_r[m]#/(node.mean_r[m]-node.mean_l[m])
                    offset = offset + str(round(val,2))+'\n'
            # offset = str(round(node.mean_r/(node.mean_r-node.mean_l),2))
            #print(str(idx)+'->'+str(node.ind))
            tree_dot.writelines(str(idx)+' -> '+str(i)+' [labeldistance=3, label = "'+offset+'",fontsize=25, color='+['black','red'][node.where_dominant=='right']+ \
                                ', style='+['solid','bold'][node.where_dominant=='right']+'];')
    
    # main body is completed
  
    tree_dot.writelines('}')
    tree_dot.close()

    # Convert to png using system command (requires Graphviz)
    import os
    print(os.getcwd())
    call(['dot', '-Tpdf', outpath+'/'+filename+'.dot', '-o', outpath+'/'+filename+'.pdf', '-Gdpi=100'])


def visualize_classifytree(root,outpath,filename):
    """write tree structure into .dot and .png files."""
    
    # open a file, and design general format
    tree_dot = open(outpath+'/'+filename+'.dot','w') 
    tree_dot.writelines('digraph Tree {')
    tree_dot.writelines('node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;')
    tree_dot.writelines('edge [fontname=helvetica] ;')

    #tree_dot = _write_tree_bfs(root,tree_dot)
        # Base Case 
    if root is None: 
        return
    # Create an empty queue for level order traversal 
    queue = [] 
    nodelist = []
    idxStack = []
    indStack = []
    

    tot_cells = len(root.indices)
    


    branch_col = pd.Series({1:'#ffccccff',2:'#ffff99ff',3:'#CC99CC',4:'#99CCFF'})   

    leaf_col = matplotlib.colors.Normalize(vmin=0, vmax=np.log(tot_cells))

    # print(tot_cells, leaf_col)

    node = root
    
    # Enqueue Root and initialize height 
    queue.append(node) 
    
    i = 0

    ndata = str(node.val_cnt)
    tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.key)+ \
                        '\\nNum: '+str(len(node.indices))+'\\n'+ ndata +' datasets'+'",fillcolor="#ff9966ff",fontsize=25];')  
    nodelist.append(node.key)
    idxStack.append(i)
    # indStack.append(node.ind)
    
    while(len(queue) > 0): 
        # Print front of queue and remove it from queue 
        node = queue.pop(0) 
        # ind = indStack.pop(0)
        idx = idxStack.pop(0)                
        # left child 
        if node.left is not None and node.left.key != ('cutleaf',): 
            nodelist.append(node.left.key)
            queue.append(node.left)
            i = i + 1
            idxStack.append(i)
            # indStack.append(node.left.ind)
            #print(str(i)+'_'+node.left.key)
            
            # percent = str(round(len(node.left.indices)/tot_cells*100,2))+'%'
            # mean_temp = data.loc[node.left.indices,:].mean() 
            
            if node.left.key == ('leaf',):                   
                col =  matplotlib.colors.to_hex(matplotlib.cm.Greens(leaf_col(np.log(len(node.left.indices)))))
                # print(i,len(node.left.indices),col)
                tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.left.key)+'\\n'+ \
                                    str(len(node.left.indices))+'",fillcolor="'+col+'",fontsize=20];')

            elif node.left.key != ('cutleaf',):
                ndata = str(node.left.val_cnt)
                tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.left.key)+'\\n'+ \
                                    str(len(node.left.indices))+'\\n'+ ndata +' datasets'+'",fillcolor="'+branch_col[len(node.left.key)]+'",fontsize=25];')
            # edge from parent to left node
            offset = ''
            if nodelist[idx][0][:2]== 'CC' and len(nodelist[idx])>1:
                val = node.mean_l.values
                offset = offset + str(round(val[0],2))
            else:
                for m in nodelist[idx]:
                    # val = (mean_temp[m] - means_in_root[m])/stds_in_root[m]
                    # print(node.mean_l, m)
                    val = node.mean_l[m]#/(node.mean_r[m]-node.mean_l[m])
                    offset = offset + str(round(val,2))+'\n'

            tree_dot.writelines(str(idx)+' -> '+str(i)+ ' [labeldistance=3, label = "'+offset+'",fontsize=25, color='+['black','red'][node.where_dominant=='left']+\
                                ', style='+['solid','bold'][node.where_dominant=='left']+'];')

        # right child 
        if node.right is not None and node.right.key != ('cutleaf',): 
            nodelist.append(node.right.key)
            queue.append(node.right) 
            i = i + 1
            idxStack.append(i)
            
            if node.right.key == ('leaf',):  
                col =  matplotlib.colors.to_hex(matplotlib.cm.Greens(leaf_col(np.log(len(node.right.indices)))))
                # print(i,len(node.right.indices),col)
                tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.right.key)+'\\n'+ \
                                    str(len(node.right.indices))+ '",fillcolor="'+col+'",fontsize=20];')

            elif node.right.key != ('cutleaf',):
                ndata = str(node.right.val_cnt)
                tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.right.key)+'\\n'+ \
                                    str(len(node.right.indices))+'\\n'+ ndata +' datasets'+'",fillcolor="'+branch_col[len(node.right.key)]+'",fontsize=25];')
            # edge from parent to right node
            offset = ''
            if nodelist[idx][0][:2]== 'CC' and len(nodelist[idx])>1:
                val = node.mean_r.values
                offset = offset + str(round(val[0],2))
            else:
                for m in nodelist[idx]:
                    # print(node.mean_r, m)
                    # val = (mean_temp[m] - means_in_root[m])/stds_in_root[m]
                    val = node.mean_r[m]#/(node.mean_r[m]-node.mean_l[m])
                    offset = offset + str(round(val,2))+'\n'
            # offset = str(round(node.mean_r/(node.mean_r-node.mean_l),2))
            #print(str(idx)+'->'+str(node.ind))
            tree_dot.writelines(str(idx)+' -> '+str(i)+' [labeldistance=3, label = "'+offset+'",fontsize=25, color='+['black','red'][node.where_dominant=='right']+ \
                                ', style='+['solid','bold'][node.where_dominant=='right']+'];')
    
    # main body is completed
  
    tree_dot.writelines('}')
    tree_dot.close()

    # Convert to png using system command (requires Graphviz)
    import os
    print(os.getcwd())
    call(['dot', '-Tpdf', outpath+'/'+filename+'.dot', '-o', outpath+'/'+filename+'.pdf', '-Gdpi=100'])