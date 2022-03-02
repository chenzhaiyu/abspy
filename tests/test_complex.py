from pathlib import Path

import numpy as np
from abspy import CellComplex

dir_tests = Path(__file__).parent


def example_cell_complex():
    """
    Simple CellComplex construction from specified planes and bounds.
    """
    # start from two planes
    planes = np.array([[0, 1, 0, -50], [0, 0, 1, -50]])

    # specify the bounds
    bounds = np.array([[[0, 50, 0], [100, 50, 100]], [[0, 0, 50], [100, 50, 50]]])

    # specify the initial bound
    initial_bound = [[0, 0, 0], [100, 100, 100]]

    # initialise the cell complex
    cell_complex = CellComplex(planes, bounds, initial_bound=initial_bound, build_graph=True)

    # prioritise certain (vertical by default) planes
    cell_complex.prioritise_planes()

    # construct the complex
    cell_complex.construct()

    # print out info
    cell_complex.print_info()

    # visualise the cell complex (only if trimesh installation is found)
    cell_complex.visualise()

    # save cell complex to OBJ and PLM files
    cell_complex.save_obj(dir_tests / 'test_output' / 'cells.obj', use_mtl=True)
    cell_complex.save_plm(dir_tests / 'test_output' / 'cells.plm')


if __name__ == '__main__':
    example_cell_complex()
