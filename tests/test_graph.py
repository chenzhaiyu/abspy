from pathlib import Path

from abspy import AdjacencyGraph


dir_tests = Path(__file__).parent


def example_graph():
    """
    Simple adjacency graph from external (.adjlist) file.
    """
    # initialise adjacency graph
    adjacency_graph = AdjacencyGraph()

    # load graph from file
    adjacency_graph.load_graph(dir_tests / 'test_data' / 'test_graph.adjlist')

    # simply draw the graph in 2D
    adjacency_graph.draw()


if __name__ == '__main__':
    example_graph()
