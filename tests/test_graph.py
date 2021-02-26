from absp import AdjacencyGraph
import networkx as nx


def example_graph():
    adjacency_graph = AdjacencyGraph()
    adjacency_graph.load_graph('./test_data/test_graph.adjlist')
    adjacency_graph.draw()


if __name__ == '__main__':
    example_graph()
