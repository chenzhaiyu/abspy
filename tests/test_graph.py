import unittest
from unittest.mock import MagicMock, patch
import networkx as nx
from abspy.graph import AdjacencyGraph

class TestAdjacencyGraph(unittest.TestCase):
    def setUp(self):
        # Create a simple graph with a few nodes and edges
        self.graph = nx.Graph()
        self.graph.add_nodes_from([0, 1, 2])
        self.graph.add_edges_from([(0, 1), (1, 2)])
        self.adjacency_graph = AdjacencyGraph(graph=self.graph, quiet=True)
        self.adjacency_graph.uid = [0, 1, 2]  # Explicitly set the uid list
        
        # Mock cells
        self.cells = [MagicMock() for _ in range(3)]
        
        # Set up the cells to have mock intersection method
        for i, cell in enumerate(self.cells):
            # Create a mock interface
            interface = MagicMock()
            interface.radius.return_value = 1.0
            interface.affine_hull_projection.return_value.vertices_list.return_value = [(0, 0), (1, 1), (0, 1)]
            interface.affine_hull_projection.return_value.volume.return_value = 1.0
            interface.n_vertices.return_value = 3
            
            # Set up cell methods
            cell.intersection.return_value = interface
            cell.vertices_list.return_value = [(0, 0), (1, 0), (1, 1), (0, 1)]
            cell.volume.return_value = 1.0
            cell.facets.return_value = []
        
        # Mock RR and ConvexHull
        self.patcher1 = patch('abspy.graph.RR', side_effect=lambda x: x)
        self.patcher2 = patch('scipy.spatial.ConvexHull')
        self.mock_RR = self.patcher1.start()
        self.mock_ConvexHull = self.patcher2.start()
        self.mock_ConvexHull.return_value.volume = 1.0
    
    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()
    
    def test_assign_weights_radius_overlap(self):
        self.adjacency_graph.assign_weights_to_n_links(self.cells, attribute='radius_overlap')
        for u, v in self.graph.edges():
            self.assertIn('capacity', self.graph[u][v])
    
    def test_assign_weights_area_overlap(self):
        self.adjacency_graph.assign_weights_to_n_links(self.cells, attribute='area_overlap')
        for u, v in self.graph.edges():
            self.assertIn('capacity', self.graph[u][v])
    
    def test_cache_interfaces(self):
        self.adjacency_graph.assign_weights_to_n_links(self.cells, attribute='radius_overlap', cache_interfaces=True)
        self.assertTrue(len(self.adjacency_graph._cached_interfaces) > 0)
        
    def test_no_cache_interfaces(self):
        self.adjacency_graph.assign_weights_to_n_links(self.cells, attribute='radius_overlap', cache_interfaces=False)
        self.assertEqual(len(self.adjacency_graph._cached_interfaces), 0)
    
    def test_normalise_true(self):
        self.adjacency_graph.assign_weights_to_n_links(self.cells, attribute='radius_overlap', normalise=True)
        for u, v in self.graph.edges():
            self.assertIn('capacity', self.graph[u][v])
            self.assertLessEqual(self.graph[u][v]['capacity'], 1.0)
    
    def test_normalise_false(self):
        self.adjacency_graph.assign_weights_to_n_links(self.cells, attribute='radius_overlap', normalise=False)
        for u, v in self.graph.edges():
            self.assertIn('capacity', self.graph[u][v])
    
    def test_factor(self):
        factor = 2.0
        self.adjacency_graph.assign_weights_to_n_links(self.cells, attribute='radius_overlap', factor=factor)
        for u, v in self.graph.edges():
            self.assertIn('capacity', self.graph[u][v])
    
    def test_engine_qhull(self):
        self.adjacency_graph.assign_weights_to_n_links(self.cells, attribute='area_overlap', engine='Qhull')
        for u, v in self.graph.edges():
            self.assertIn('capacity', self.graph[u][v])
    
    def test_engine_non_qhull(self):
        self.adjacency_graph.assign_weights_to_n_links(self.cells, attribute='area_overlap', engine='non_Qhull')
        for u, v in self.graph.edges():
            self.assertIn('capacity', self.graph[u][v])

if __name__ == '__main__':
    unittest.main()