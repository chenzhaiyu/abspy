from absp import VertexGroup, AdjacencyGraph, CellComplex
import numpy as np
import random


def example_combined():
    vertex_group = VertexGroup(filepath='./test_data/test_points.vg')
    planes, bounds = np.array(vertex_group.planes), np.array(vertex_group.bounds)

    cell_complex = CellComplex(planes, bounds, build_graph=True)
    cell_complex.prioritise_planes()
    cell_complex.construct()
    cell_complex.print_info()

    graph = AdjacencyGraph(cell_complex.graph)
    weights_list = [random.random() for i in range(cell_complex.num_cells)]
    weights_dict = graph.to_dict(weights_list)

    graph.assign_weights_to_n_links(cell_complex.cells, attribute='area_overlap', factor=0.1)  # provided by the cell complex
    graph.assign_weights_to_st_links(weights_dict)  # provided by the neural network prediction
    _, _ = graph.cut()


if __name__ == '__main__':
    example_combined()
