import networkx as nx
import matplotlib.pyplot as plt


class CellTree(object):

    def __init__(
        self, n_cells,
    ) -> None:
        self.tree = nx.DiGraph()
        self.tree.add_node(0)
        nx.set_node_attributes(self.tree, {0: n_cells}, name='n_cells')
        return None

    def differentiate(
        self,
        node,
        n_successors=2,
        n_successors_cells=None
    ) -> None:
        assert self.tree.has_node(node), 'Node %s does not exist.' % node
        assert self.tree.out_degree(
            node) == 0, 'Node %s has beed differentiate.' % node
        if n_successors_cells == None:
            n_successors_cells = [self.tree.nodes[node]
                                  ['n_cells'] // n_successors] * n_successors
            n_successors_cells[0] += self.tree.nodes[node]['n_cells'] - \
                sum(n_successors_cells)
        assert len(
            n_successors_cells) == n_successors, 'Length of n_successors_cells not equal to n_successors.'
        assert sum(
            n_successors_cells) == self.tree.nodes[node]['n_cells'], 'Number of cells in successors not equal to number of cells in node to be differentiated.'
        n = self.tree.number_of_nodes()
        for new_node in range(n, n+n_successors):
            self.tree.add_edge(node, new_node)
            nx.set_node_attributes(
                self.tree, {new_node: n_successors_cells[new_node-n]}, 'n_cells')
        return None

    def draw(
        self,
        save_path=None
    ) -> None:
        plt.figure(dpi=256)
        pos = nx.nx_agraph.graphviz_layout(self.tree, prog='dot')
        nx.draw(
            self.tree,
            labels=nx.get_node_attributes(self.tree, 'n_cells'),
            pos=pos,
            node_size=1500,
            node_color='pink'
        )
        nx.draw_networkx_edge_labels(
            self.tree, pos=pos, edge_labels=nx.get_edge_attributes(self.tree, 'metagene_reg'))
        for node in pos:
            pos[node] = list(pos[node])
            pos[node][1] += 10
        nx.draw_networkx_labels(self.tree, pos=pos)
        if save_path != None:
            plt.savefig(save_path + '/cell_tree.png')
        return None
