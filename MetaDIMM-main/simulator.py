import time
import os
from simulationcelltree import SimulationCellTree


def main():
    """
    Generate a simulation cell tree.
    """
    tree = SimulationCellTree(8000)
    tree.differentiate(0)
    tree.differentiate(1)
    tree.differentiate(2)
    # tree.differentiate(3)
    # tree.differentiate(4)
    # tree.differentiate(5)
    # tree.differentiate(6)
    tree.generate_metagene_regulation()
    adata = tree.simulate_gene_expr()
    timestamp=time.strftime('%Y%m%d%H%M%S')
    os.mkdir('./simdata/'+timestamp)
    tree.draw(save_path='./simdata/'+timestamp)
    adata.write(filename='./simdata/'+timestamp+'/data.h5ad')


if __name__ == "__main__":
    main()
