from pathlib import Path
import networkx as nx
from sage.all import RR
from scipy.spatial import ConvexHull


from absp import attach_to_log

logger = attach_to_log()


class AdjacencyGraph:
    def __init__(self, graph=None):
        self.graph = graph
        self.uid = list(graph.nodes) if graph else None  # passed graph.nodes are sorted

    def load_graph(self, filepath):
        """
        Load graph from an external file.
        """
        filepath = Path(filepath)
        if filepath.suffix == '.adjlist':
            logger.info('loading graph from {}'.format(filepath))
            self.graph = nx.read_adjlist(filepath)
            self.uid = self._sort_uid()  # loaded graph.nodes are unordered string
        else:
            raise NotImplementedError('file format not supported: {}'.format(filepath.suffix))

    def assign_weights_to_n_links(self, cells, mode='radius', normalise=True, factor=1.0, backend='Qhull'):
        """
        Assign weights to edges between every cell. weights is a dict with respect to each pair of nodes.
        """

        radius = [None] * len(self.graph.edges)
        area = [None] * len(self.graph.edges)

        if mode == 'radius_overlap':
            for i, (m, n) in enumerate(self.graph.edges):
                # compute interface
                interface = cells[self._uid_to_index(m)].intersection(cells[self._uid_to_index(n)])
                # the maximal distance from the center to a vertex -> inverted: penalising small radius
                radius[i] = RR(interface.radius())

            for i, (m, n) in enumerate(self.graph.edges):
                max_radius = max(radius)
                self.graph[m][n].update({'capacity': ((max_radius - radius[
                    i]) / max_radius if normalise else max_radius - radius[i]) * factor})

        elif mode == 'area_overlap':
            for i, (m, n) in enumerate(self.graph.edges):
                # compute interface
                interface = cells[self._uid_to_index(m)].intersection(cells[self._uid_to_index(n)])
                # area of the overlap -> inverted: penalising small area
                if backend == 'Qhull':
                    # 'volume' is the area of the convex hull when input points are 2-dimensional
                    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
                    area[i] = ConvexHull(interface.affine_hull_projection().vertices_list()).volume
                else:
                    # slower computation
                    area[i] = RR(interface.affine_hull_projection().volume())

            for i, (m, n) in enumerate(self.graph.edges):
                max_area = max(area)
                self.graph[m][n].update(
                    {'capacity': ((max_area - area[i]) / max_area if normalise else max_area - area[i]) * factor})

        elif mode == 'area_misalign':
            # area of the mis-aligned regions from both cells
            raise NotImplementedError

    def assign_weights_to_st_links(self, weights):
        """
        Assign weights to edges between each cell and the s-t nodes. weights is a dict in respect to each node.
        the weights can be the occupancy probability or the signed distance of the cells
        """
        for i in self.uid:
            self.graph.add_edge(i, 's', capacity=weights[i])
            self.graph.add_edge(i, 't', capacity=1 - weights[i])  # make sure
    
    def cut(self):
        """
        Perform cutting operation.
        """
        cut_value, partition = nx.algorithms.flow.minimum_cut(self.graph, 's', 't')
        reachable, non_reachable = partition
        reachable.remove('s')
        logger.info('cut_value: {}'.format(cut_value))
        logger.info('number of extracted cells: {}'.format(len(reachable)))
        return cut_value, reachable

    def draw(self):
        """
        Naively draw the graph with nodes represented by their UID.
        """
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

    def _index_to_uid(self, query):
        """
        Convert index to node UID.
        """
        return self.uid[query]

    def _sort_uid(self):
        """
        Sort UID for graph structure loaded from an external file.
        """
        return sorted([int(i) for i in self.graph.nodes])

    def to_indices(self, uids):
        """
        Convert UIDs to indices.
        """
        return [self._uid_to_index(i) for i in uids]

    def to_dict(self, weights_list):
        """
        Convert a weight list to weight dict keyed by self.uid.
        """
        return {self.uid[i]: weight for i, weight in enumerate(weights_list)}
