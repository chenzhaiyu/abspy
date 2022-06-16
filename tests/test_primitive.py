from pathlib import Path

from abspy import VertexGroup, VertexGroupReference

dir_tests = Path(__file__).parent


def example_extract_primitives_vg():
    """
    Extract primitives from VertexGroup (.vg) file.
    """
    vertex_group = VertexGroup(filepath=dir_tests / 'test_data' / 'test_points.vg')

    # save planes and bounds as numpy (.npy) files
    vertex_group.save_planes_npy(dir_tests / 'test_output' / 'primitives_planes_vg.npy')
    vertex_group.save_bounds_npy(dir_tests / 'test_output' / 'primitives_bounds_vg.npy')


def example_extract_primitives_bvg():
    """
    Extract primitives from VertexGroup (.vg) file.
    """
    vertex_group = VertexGroup(filepath=dir_tests / 'test_data' / 'test_points.bvg')

    # save planes and bounds as numpy (.npy) files
    vertex_group.save_planes_npy(dir_tests / 'test_output' / 'primitives_planes_bvg.npy')
    vertex_group.save_bounds_npy(dir_tests / 'test_output' / 'primitives_bounds_bvg.npy')


def example_extract_reference_primitives():
    """
    Extract primitives from VertexGroupReference (.ply) file.
    """
    vertex_group_reference = VertexGroupReference(filepath=dir_tests / 'test_data' / 'test_mesh.ply')

    # save extracted primitives as a VertexGroup (.vg) file
    vertex_group_reference.save_primitives_vg(dir_tests / 'test_output' / 'reference.vg')


if __name__ == '__main__':
    example_extract_primitives_vg()
    example_extract_primitives_bvg()
    example_extract_reference_primitives()
