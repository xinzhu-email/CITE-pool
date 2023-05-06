#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 23:49:04 2019

@author: lianqiuyu
"""

import pandas as pd
from CITEsort.Matryoshka import Matryoshka
from CITEsort.Visualize import visualize_tree
from CITEsort.BTreeTraversal import BTreeTraversal
from CITEsort.ReSplit import ReSplit, Choose_leaf
import pickle
import argparse
import os
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#from sys import argv

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help = "The input path of CLR normalized data in .csv files with row as sample, col as feature.")
    parser.add_argument('-c','--cutoff',type = float, default=0.1, help = "The cutoff for merging components (default 0.1). It shoube a value between 0 and 1. The bigger value leads to split more aggressively, and ends in a more complicated tree.")
    parser.add_argument('-o', '--output', type=str, default='./CITEsort_out',help='Path to save output files.')
    parser.add_argument('--compact', action='store_true', default=False, help='Output a compact tree.')
    args = parser.parse_args()
    
    data_path = args.data_path
    output_path = args.output
    merge_cutoff = args.cutoff
    compact_flag = args.compact
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    starttime = time.time()
    print('read data and run CITE-sort.')

    data = pd.read_csv(data_path,header = 0, index_col=0)
    tree, bic_list, min_bic_node = Choose_leaf(data=data,merge_cutoff=merge_cutoff,rawdata=data.copy())
    # tree = ReSplit(data,merge_cutoff)
    #tree = Matryoshka(data,merge_cutoff)
    print('done.\nplot tree.')
    visualize_tree(tree,data,output_path,'tree',compact=compact_flag)
    
    f = open(output_path+'/tree.pickle','wb')
    pickle.dump(tree,f)
    f.close()
    
    print('generate labels.')
    
    traversal = BTreeTraversal(tree,save_min_BIC=False, min_BIC_node=min_bic_node)
    leaves_labels = traversal.get_leaf_label()
    leaves_labels.to_csv(output_path + '/leaf_labels.csv')
    # leaves_labels = traversal.get_leaf_label(BIC_node=True)
    # leaves_labels.to_csv(output_path + '/BIC_stop_labels.csv',index=False)

    endtime = time.time()

    print('Time using: ', round(endtime-starttime, 3),'secs')
    plt.plot(list(range(len(bic_list))), bic_list)
    # plt.savefig('BIC_as_split.png')

if __name__ == "__main__":
    main()
    
    