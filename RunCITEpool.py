import pandas as pd
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

from CITEpool.BTree import BTree
from CITEpool.NodeSplit import CrossSplit, dfs, CrossNode, CellEmbedding
from CITEpool.Visualize import visualize_tree, visualize_modeltree
from CITEpool.BTreeTraversal import BTreeTraversal


def main(data_path, output_path, 
        cutoff=0.1, current_treepath=None, FinetuneNode=[], ifretrain=False):

    # path = ['../data/mosaic celltype/data3.h5ad']
    # output_path = '../output/mosaic celltype/3tem_test++'
    # merge_cutoff = 0.45
    # current_treepath = '../output/mosaic celltype/3tem_test++'
    # current_tree = ['0','1','2','3']
    # ifretrain = False
    # FinetuneNonde = [5]

    starttime = time.time()
    print('Read data and run CITE-pool.')

    adtdata, rnadata = {}, {}

    ## Load Files
    j, inbatch = 0, []
    for i in range(len(data_path)):
        jpre = j
        adata = sc.read_h5ad(data_path[i])
        batch = adata.obs['batch'].cat.categories
        for b in batch:
            data = adata[adata.obs['batch']==b]
            adtdata[j] = data[:,data.var['feature_types']=='Antibody Capture']
            
            rnadata[j] = data[:,data.var['feature_types']=='Gene Expression']
            
            x = rnadata[j][:31,:20].X.sum()
            if x == int(x):
                # print(s)
                sc.pp.normalize_total(rnadata[j], target_sum=1e4)
                sc.pp.log1p(rnadata[j])
                
            del(data)
            adtdata[j] = adtdata[j].to_df()
            j = j + 1

        inbatch.append(j-jpre)
        print('dataset',i+1,'batch num:',j-jpre,'ADT num:',adtdata[j-1].shape[1])


    ## Constrcut tree for integration and clustering
    dataid = 0
    if current_treepath is None:
        crossnode = CrossSplit(adtdata.copy(),rnadata.copy(),cutoff)
    else:
        nodelist = []
        for i in range(len(adtdata)+1):

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
    
        crossnode = dfs(crossnode, adtdata, rnadata.copy(), cutoff, nodeid=FinetuneNode, ifretrain=ifretrain)
    
    ## Tree dfs of each dataset
    def inner_dfs(node, crossnode, i, ind):
        if node is not None:
            node.ind = ind
            if crossnode.left is not None:
                node.left = inner_dfs(crossnode.left.nodelist[i], crossnode.left, i, 2*ind+1) 
                node.right = inner_dfs(crossnode.right.nodelist[i], crossnode.right, i, 2*ind+2)
        return node

    ## Tree dfs for model tree
    def modeltree_dfs(node, crossnode):
        if node is not None and crossnode.left is not None:
            node.left = modeltree_dfs(crossnode.left.modelnode,  crossnode.left)
            node.right = modeltree_dfs(crossnode.right.modelnode,  crossnode.right)
        return node

    ## Save model tree
    output = output_path+'/0'
    if not os.path.exists(output):
        os.mkdir(output)
    modeltree = modeltree_dfs(crossnode.modelnode, crossnode)

    visualize_modeltree(modeltree, output, 'tree')
    f = open(output+'/tree.pickle','wb')
    pickle.dump(modeltree,f)
    f.close()

    ## Save dataset tree, embedding, leaflabel and treelabel
    batch, dataid  = 1, 1
    for i in range(len(adtdata)):
        # if ifretrain and reclass == False:
        #     break
        tree = inner_dfs(crossnode.nodelist[i], crossnode, i, 0)
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

        embedding, treelabel = CellEmbedding(tree, rnadata[i].copy())
        embedding.to_csv(output+ '/embedding.csv')
        treelabel.to_csv(output+ '/treelabel.csv')

        visualize_tree(tree, adtdata[i], output, 'tree', rnadata=rnadata[i].copy(),modeltree=modeltree)
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



def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run CITE-pool analysis')
    
    parser.add_argument('--path', nargs='+', required=True,
                       help='List of input data file paths')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output directory path')
    parser.add_argument('--cutoff', type=float, default=0.1,
                       help='Bimodal overlap threshold, default 0.1')
    parser.add_argument('--current_treepath', type=str, default=None,
                       help='Path to existing tree for continued training')
    parser.add_argument('--FinetuneNonde', nargs='+', type=int, default=[],
                       help='List of nodes to fine-tune')
    parser.add_argument('--ifretrain', action='store_true',
                       help='Whether to retrain')
    
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    main(
        data_path=args.data_path,
        output_path=args.output_path,
        cutoff=args.cutoff,
        current_treepath=args.current_treepath,
        FinetuneNonde=args.FinetuneNonde,
        ifretrain=args.ifretrain
    )