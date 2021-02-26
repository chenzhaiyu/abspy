import networkx as nx
from pathlib import Path


class AdjacencyGraph:
    def __init__(self, graph=None):
        self.graph = graph
        self.uid = list(graph.nodes) if graph else None  # passed graph.nodes are sorted

    def load_graph(self, filepath):
        filepath = Path(filepath)
        if filepath.suffix == '.adjlist':
            self.graph = nx.read_adjlist(filepath)
            self.uid = self._sort_uid()  # loaded graph.nodes are unordered string
        else:
            raise NotImplementedError('File format not supported: {}'.format(filepath.suffix))

    def assign_weights(self, weights):
        """
        Assign weights to every nodes. weights is a dict in respect to each node.
        """
        assert isinstance(weights, dict)
        nx.set_node_attributes(self.graph, weights, name='probability')

    def cut(self):
        pass

    def draw(self):
        import matplotlib.pyplot as plt
        plt.subplot(121)
        nx.draw(self.graph, with_labels=True, font_weight='bold')
        plt.show()

    def _uid_to_index(self, query):
        """
        Convert index in the node list to that in the cell list.
        The rationale behind is #nodes == #cells (when a primitive is settled down).
        :param query: query index in the node list.
        """
        return self.uid.index(query)

    def _sort_uid(self):
        return sorted([int(i) for i in self.graph.nodes])
