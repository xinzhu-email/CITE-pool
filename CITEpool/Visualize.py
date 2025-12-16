import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import matplotlib
import scanpy as sc
from subprocess import call


def visualize_modeltree(root,outpath,filename):
    """write tree structure into .dot and .png files."""
    
    # open a file, and design general format
    tree_dot = open(outpath+'/'+filename+'.dot','w') 
    tree_dot.writelines('digraph Tree {')
    tree_dot.writelines('node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;')
    tree_dot.writelines('edge [fontname=helvetica] ;')

    queue = [] 
    nodelist = {}
    idxStack = []

    tot_cells = [root.val_cnt]

    branch_col = pd.Series({1:'#ffccccff',2:'#ffff99ff',3:'#CC99CC',4:'#99CCFF'})   
    print(tot_cells)
    leaf_col = matplotlib.colors.Normalize(vmin=0, vmax=np.log(tot_cells))
    
    node = root

    queue.append(node) 
    
    i = 0

    ndata = str(node.val_cnt)
    if len(node.indices) <= 4: 
        training_datasets = str(node.indices)
    else:
        training_datasets = str(len(node.indices))
    tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+node.key+ \
                        '\\n'+ ndata +'\\n'+training_datasets+'",fillcolor="#ff9966ff",fontsize=25];')  
    nodelist[i] = node.key
    idxStack.append(i)

    while(len(queue) > 0): 
        # Print front of queue and remove it from queue 
        node = queue.pop(0) 
        # ind = indStack.pop(0)
        idx = idxStack.pop(0)
        
        # left child 
        if node.left is not None: 
            queue.append(node.left)
            i = 2*idx + 1
            nodelist[i] = node.left.key
            idxStack.append(i)
            
            if node.left.key == ('leaf',):                   
                col =  matplotlib.colors.to_hex(matplotlib.cm.Greens(leaf_col(np.log(node.left.val_cnt))))
                tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.left.key)+'\\n'+ \
                                    str(node.left.val_cnt)+'",fillcolor="'+col+'",fontsize=20];')
            else:
                if len(node.left.indices)<=4:
                    ndata = str(node.left.indices)
                else:
                    ndata = str(len(node.left.indices))
                tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+node.left.key+'\\n'+ \
                                    str(node.left.val_cnt)+'\\n'+ ndata +'",fillcolor="'+branch_col[1]+'",fontsize=25];')

            # edge from parent to left node
            offset = ''
            if nodelist[idx][0][:2]== 'CC' and len(nodelist[idx])>1:
                val = node.mean_l
                offset = offset + str(round(val,2))
            else:
                val = node.mean_l #/(node.mean_r[m]-node.mean_l[m])
                offset = offset + str(round(val,2))+'\n'

            tree_dot.writelines(str(idx)+' -> '+str(i)+ ' [labeldistance=3, label = "'+offset+'",fontsize=25, color='+['black','red'][node.where_dominant=='left']+\
                                ', style='+['solid','bold'][node.where_dominant=='left']+'];')

        # right child 
        if node.right is not None: 
            queue.append(node.right) 
            i = 2*idx + 2
            nodelist[i] = node.right.key
            idxStack.append(i)

            if node.right.key == ('leaf',):  
                col =  matplotlib.colors.to_hex(matplotlib.cm.Greens(leaf_col(np.log(node.right.val_cnt))))
                tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.right.key)+'\\n'+ \
                                    str(node.right.val_cnt)+ '",fillcolor="'+col+'",fontsize=20];')

            else:
                if len(node.right.indices)<=4:
                    ndata = str(node.right.indices)
                else:
                    ndata = str(len(node.right.indices))
                tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+node.right.key+'\\n'+ \
                                    str(node.right.val_cnt)+'\\n'+ ndata +'",fillcolor="'+branch_col[1]+'",fontsize=25];')

            # edge from parent to right node
            offset = ''
            if nodelist[idx][0][:2]== 'CC' and len(nodelist[idx])>1:
                val = node.mean_r
                offset = offset + str(round(val,2))
            else:
                val = node.mean_r#/(node.mean_r[m]-node.mean_l[m])
                offset = offset + str(round(val,2))+'\n'

            tree_dot.writelines(str(idx)+' -> '+str(i)+' [labeldistance=3, label = "'+offset+'",fontsize=25, color='+['black','red'][node.where_dominant=='right']+ \
                                ', style='+['solid','bold'][node.where_dominant=='right']+'];')

    tree_dot.writelines('}')
    tree_dot.close()

    # Convert to png using system command (requires Graphviz)
    import os
    print(os.getcwd())
    call(['dot', '-Tpdf', outpath+'/'+filename+'.dot', '-o', outpath+'/'+filename+'.pdf', '-Gdpi=100'])



def visualize_tree(root,data,outpath,filename,compact=True,rnadata=None,modeltree=None):
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
    nodelist = {}
    idxStack = []

    modelqueue = []
    
    tot_cells = len(root.indices)
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
    modelqueue.append(modeltree)
    
    i = 0
    #print(str(node.ind)+'_'+root.key)

    tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+node.key+ \
                    '\\nNum: '+str(len(node.indices))+ \
                    '",fillcolor="#ff9966ff",fontsize=25];')  
    
    nodelist[i] = node.key
    idxStack.append(i)
    # indStack.append(node.ind)
    
    while(len(queue) > 0): 
        # Print front of queue and remove it from queue 
        node = queue.pop(0) 
        modeltree = modelqueue.pop(0)
        # ind = indStack.pop(0)
        idx = idxStack.pop(0)

        if node is None:
            if modeltree.key != ('leaf',):
                i = 2*idx + 1
                idxStack.append(i)
                col =  matplotlib.colors.to_hex(matplotlib.cm.Greens(leaf_col(np.log(0))))
                tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(('leaf',))+'\\n'+ \
                                    str(0)+ ' (0%)\\n'+ \
                                    ''+'",fillcolor="'+col+'",fontsize=20];')
                tree_dot.writelines(str(idx)+' -> '+str(i)+ ' [labeldistance=3, label = "'+'",fontsize=25, color='+['black','red']['left'=='left']+\
                                ', style='+['solid','bold']['left'=='left']+'];')
                nodelist[i] = ('leaf',)
                i = 2*idx + 2
                idxStack.append(i)
                tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(('leaf',))+'\\n'+ \
                                    str(0)+ ' (0%)\\n'+ \
                                    ''+'",fillcolor="'+col+'",fontsize=20];')
                tree_dot.writelines(str(idx)+' -> '+str(i)+ ' [labeldistance=3, label = "'+'",fontsize=25, color='+['black','red']['left'=='right']+\
                                ', style='+['solid','bold']['left'=='right']+'];')
                nodelist[i] = ('leaf',)
                queue.append(None)
                queue.append(None)
                modelqueue.append(modeltree.left)
                modelqueue.append(modeltree.right)
        
        elif node is not None:

            if node.key == ('artificial',):
                markers = ('artificial',)
                stds_in_root['artificial'] = node.artificial_w.std()
                # print(data['artificial'])
   
            # left child 
            if node.left is not None: 
                queue.append(node.left)
                modelqueue.append(modeltree.left)
                i = 2*idx + 1
                idxStack.append(i)
                nodelist[i] = node.left.key
                
                percent = str(round(len(node.left.indices)/tot_cells*100,2))+'%'
                 
                
                if node.left.key == ('leaf',):     
                    col =  matplotlib.colors.to_hex(matplotlib.cm.Greens(leaf_col(np.log(len(node.left.indices)))))
                    tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.left.key)+'\\n'+ \
                                        str(len(node.left.indices))+ ' ('+percent+')\\n'+ \
                                        '",fillcolor="'+col+'",fontsize=20];')
                else:
                    # left branch node

                    tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+node.left.key+'\\n'+ \
                                    str(len(node.left.indices))+' ('+percent+')\\n'+ \
                                    '",fillcolor="'+branch_col[1]+'",fontsize=25];')

                #print(str(idx)+'->'+str(i))
                tree_dot.writelines(str(idx)+' -> '+str(i)+ ' [labeldistance=3, label = "'+'",fontsize=25, color='+['black','red'][node.where_dominant=='left']+\
                                    ', style='+['solid','bold'][node.where_dominant=='left']+'];')
                
                if node.right is None:
                    i = 2*idx + 2
                    idxStack.append(i)
                    col =  matplotlib.colors.to_hex(matplotlib.cm.Greens(leaf_col(0)))
                    tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(('leaf',))+'\\n'+ \
                                        str(0)+ ' (0%)\\n'+ \
                                        ''+'",fillcolor="'+col+'",fontsize=20];')
                    tree_dot.writelines(str(idx)+' -> '+str(i)+ ' [labeldistance=3, label = "'+'",fontsize=25, color='+['black','red'][node.where_dominant=='left']+\
                                    ', style='+['solid','bold'][node.where_dominant=='right']+'];')
                    nodelist[i] = ('leaf',)                    
                    queue.append(None)
                    modelqueue.append(modeltree.right)


            # right child 
            if node.right is not None : 
                if node.left is None:
                    i = 2*idx + 1
                    idxStack.append(i)
                    col =  matplotlib.colors.to_hex(matplotlib.cm.Greens(leaf_col(np.log(0))))
                    tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(('leaf',))+'\\n'+ \
                                        str(0)+ ' (0%)\\n'+ \
                                        ''+'",fillcolor="'+col+'",fontsize=20];')
                    tree_dot.writelines(str(idx)+' -> '+str(i)+ ' [labeldistance=3, label = "'+'",fontsize=25, color='+['black','red'][node.where_dominant=='left']+\
                                    ', style='+['solid','bold'][node.where_dominant=='left']+'];')
                    nodelist[i] = ('leaf',)     
                    queue.append(None)
                    modelqueue.append(modeltree.left)

                queue.append(node.right) 
                modelqueue.append(modeltree.right)
                i = 2*idx + 2
                nodelist[i] = node.right.key
                idxStack.append(i)
                # indStack.append(node.right.ind)
                #print(str(i)+'_'+node.right.key)
                
                percent = str(round(len(node.right.indices)/tot_cells*100,2))+'%'
                
                if node.right.key == ('leaf',):
                    # print(node.right.ind,node.right.key)
                    # right leaf node

                    col =  matplotlib.colors.to_hex(matplotlib.cm.Greens(leaf_col(np.log(len(node.right.indices)))))
                    tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+'_'.join(node.right.key)+'\\n'+ \
                                        str(len(node.right.indices))+ ' ('+percent+')'+'\\n'+ \
                                        '",fillcolor="'+col+'",fontsize=20];')

                else:

                    tree_dot.writelines(str(i)+' [label="'+str(i)+'_'+node.right.key+'\\n'+ \
                                        str(len(node.right.indices))+' ('+percent+')\\n'+ \
                                        '",fillcolor="'+branch_col[1]+'",fontsize=25];')
                    
                
                tree_dot.writelines(str(idx)+' -> '+str(i)+' [labeldistance=3, label = "'+'",fontsize=25, color='+['black','red'][node.where_dominant=='right']+ \
                                    ', style='+['solid','bold'][node.where_dominant=='right']+'];')

    # main body is completed
  
    tree_dot.writelines('}')
    tree_dot.close()

    # Convert to png using system command (requires Graphviz)
    import os
    print(os.getcwd()+outpath)
    call(['dot', '-Tpdf', outpath+'/'+filename+'.dot', '-o', outpath+'/'+filename+'.pdf', '-Gdpi=100'])
    
    
    # Display in jupyter notebook
    #Image(filename = outpath+'/GatingTree.png')
    return idxStack
