import numpy as np
from pathlib import Path

from abspy import VertexGroup, VertexGroupReference

dir_tests = Path(__file__).parent


def example_extract_primitives_vg():
    """
    Extract primitives from VertexGroup (.vg) file.
    """
    vertex_group = VertexGroup(filepath=dir_tests / 'test_data' / 'test_points.vg')

    # append additional planes
    additional_planes = [[1, 2, 3, 4], [5, 6, 7, 8]]
    additional_points = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                         [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]
    vertex_group.append_planes(additional_planes=additional_planes, additional_points=additional_points)

    # save planes and bounds as numpy (.npy) files
    vertex_group.save_planes_npy(dir_tests / 'test_output' / 'primitives_planes_vg.npy')
    vertex_group.save_aabbs_npy(dir_tests / 'test_output' / 'primitives_bounds_vg.npy')

    # save extracted primitives to a Vertex group (.vg) file
    vertex_group.save_vg(dir_tests / 'test_output' / 'primitives.vg')


def example_extract_primitives_bvg():
    """
    Extract primitives from VertexGroup (.vg) file.
    """
    vertex_group = VertexGroup(filepath=dir_tests / 'test_data' / 'test_points.bvg')

    # append additional planes
    additional_planes = [[1, 2, 3, 4], [5, 6, 7, 8]]
    additional_points = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                         [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]
    vertex_group.append_planes(additional_planes=additional_planes, additional_points=additional_points)

    # save planes and bounds as numpy (.npy) files
    vertex_group.save_planes_npy(dir_tests / 'test_output' / 'primitives_planes_bvg.npy')
    vertex_group.save_aabbs_npy(dir_tests / 'test_output' / 'primitives_bounds_bvg.npy')

    # save extracted primitives to a binary Vertex group (.bvg) file
    vertex_group.save_bvg(dir_tests / 'test_output' / 'primitives.bvg')


def example_extract_reference_primitives():
    """
    Extract primitives from VertexGroupReference (.ply) file.
    """
    vertex_group_reference = VertexGroupReference(filepath=dir_tests / 'test_data' / 'test_mesh.ply', num_samples=10000)

    # inject points
    vertex_group_reference.inject_points(np.random.rand(100, 3) - 0.5, threshold=0.05, overwrite=True,
                                         keep_bottom=True, keep_wall=True, compute_normal=True,
                                         pseudo_normal=False, pseudo_size=0)

    # save point cloud
    vertex_group_reference.save_cloud(dir_tests / 'test_output' / 'reference.ply')

    # save extracted primitives to both a Vertex Group (.vg) file and a binary Vertex group (.bvg) file
    vertex_group_reference.save_vg(dir_tests / 'test_output' / 'reference.vg')
    vertex_group_reference.save_bvg(dir_tests / 'test_output' / 'reference.bvg')


if __name__ == '__main__':
    example_extract_primitives_vg()
    example_extract_primitives_bvg()
    example_extract_reference_primitives()
