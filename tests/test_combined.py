from absp import VertexGroup, AdjacencyGraph, CellComplex
import numpy as np
import random
random.seed(100)


def sigmoid(x):
    # can safely ignore RuntimeWarning: overflow encountered in exp
    return 1 / (1 + np.exp(-x))


def example_combined():
    vertex_group = VertexGroup(filepath='./test_data/test_points.vg')
    vertex_group.normalise_to_centroid_and_scale()
    planes, bounds, points = np.array(vertex_group.planes), np.array(vertex_group.bounds), np.array(
        vertex_group.points_grouped, dtype=object)

    cell_complex = CellComplex(planes, bounds, points, build_graph=True)
    cell_complex.refine_planes()
    cell_complex.prioritise_planes()
    cell_complex.construct()
    cell_complex.print_info()

    graph = AdjacencyGraph(cell_complex.graph)

    # provided by the neural network prediction
    weights_list = np.array([random.random() for _ in range(cell_complex.num_cells)])
    weights_list *= cell_complex.volumes(multiplier=10e3)
    # weights_list = sigmoid(weights_list)

    weights_dict = graph.to_dict(weights_list)

    graph.assign_weights_to_n_links(cell_complex.cells, attribute='area_overlap',
                                    factor=0.1, cache_interfaces=True)  # provided by the cell complex
    graph.assign_weights_to_st_links(weights_dict)
    _, _ = graph.cut()
    graph.save_surface_obj(filepath='../output/surface.obj', engine='rendering')


if __name__ == '__main__':
    example_combined()
