import numpy as np
from absp import CellComplex


def example_cell_complex():
    planes = np.array([[0, 1, 0, -50], [0, 0, 1, -50]])
    bounds = np.array([[[0, 50, 0], [100, 50, 100]], [[0, 0, 50], [100, 50, 50]]])
    initial_bound = [[0, 0, 0], [100, 100, 100]]

    cell_complex = CellComplex(planes, bounds, initial_bound=initial_bound, build_graph=True)
    # refine_planes() requires point coordinates
    # cell_complex.refine_planes(normalise_normal=True)
    cell_complex.prioritise_planes()
    cell_complex.construct()
    cell_complex.print_info()
    cell_complex.save_obj('../output/cells.obj')
    cell_complex.save_plm('../output/cells.plm')


if __name__ == '__main__':
    example_cell_complex()
