from pathlib import Path

from abspy import AdjacencyGraph
import networkx as nx

dir_tests= Path(__file__).parent

def example_graph():
    adjacency_graph = AdjacencyGraph()
    adjacency_graph.load_graph(dir_tests / 'test_data' / 'test_graph.adjlist')
    adjacency_graph.draw()


if __name__ == '__main__':
    example_graph()
