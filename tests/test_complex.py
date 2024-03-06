from pathlib import Path

import numpy as np
import trimesh

from abspy import CellComplex

dir_tests = Path(__file__).parent


def example_cell_complex_from_planes():
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

    # boundary cells
    cells_boundary = cell_complex.cells_boundary()
    print(f'boundary cells: {cells_boundary}')

    # cell representatives
    num_representatives = 1000
    representatives = cell_complex.cell_representatives(location='skeleton', num=num_representatives)
    representatives = np.concatenate(representatives, axis=0)
    colours = np.zeros([num_representatives * cell_complex.num_cells, 4])
    for i in range(cell_complex.num_cells):
        colours[num_representatives * i: num_representatives * (i + 1)] = trimesh.visual.color.random_color()

    # visualise cell representatives
    representatives_vis = trimesh.PointCloud(representatives, colours)
    representatives_vis.export(dir_tests / 'test_output' / 'test_vis.ply', file_type='ply')
    try:
        representatives_vis.show()
    except (Exception, ImportError) as e:
        print(f'visualization skipped: {e}')

    # save cell complex CC file
    cell_complex.save(dir_tests / 'test_output' / 'complex.cc')

    # save cells to OBJ and PLM files
    cell_complex.save_obj(dir_tests / 'test_output' / 'cells.obj', use_mtl=True)
    cell_complex.save_plm(dir_tests / 'test_output' / 'cells.plm')


def example_cell_complex_from_mesh():
    """
    CellComplex construction from reference mesh.
    """
    # vertex group reference from mesh
    from abspy import VertexGroupReference
    vertex_group_reference = VertexGroupReference(filepath=dir_tests / 'test_data' / 'test_church.obj', num_samples=10000)

    # perturb plane normal vectors
    vertex_group_reference.perturb(sigma=0.001)

    # construct cell complex
    cell_complex = CellComplex(vertex_group_reference.planes, vertex_group_reference.aabbs,
                               vertex_group_reference.obbs, build_graph=True, quiet=False)
    cell_complex.construct()
    cell_complex.print_info()

    # cells inside reference mesh
    cells_in_mesh = cell_complex.cells_in_mesh(dir_tests / 'test_data' / 'test_church.obj', engine='distance')

    # save cell complex CC file
    cell_complex.save(dir_tests / 'test_output' / 'test_complex.cc')

    #  Save cells to OBJ and PLM files
    cell_complex.save_obj(dir_tests / 'test_output' / 'test_cells.obj', use_mtl=True, indices_cells=cells_in_mesh)
    cell_complex.save_plm(dir_tests / 'test_output' / 'test_cells.plm')

    # visualise the inside cells (only if pyglet installation is found and valid indices are provided)
    if len(cells_in_mesh):
        try:
            cell_complex.visualise(indices_cells=cells_in_mesh, temp_dir='./test_output/')
        except (Exception, ImportError) as e:
            print(f'visualization skipped: {e}')


if __name__ == '__main__':
    example_cell_complex_from_planes()
    example_cell_complex_from_mesh()

