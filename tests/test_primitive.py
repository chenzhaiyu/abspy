from pathlib import Path

from pyabsp import VertexGroup, VertexGroupReference

dir_tests= Path(__file__).parent

def example_extract_primitives():
    vertex_group = VertexGroup(filepath=dir_tests / 'test_data' / 'test_points.vg')
    vertex_group.save_planes_npy(dir_tests / 'test_output' / 'primitives_planes.npy')
    vertex_group.save_bounds_npy(dir_tests / 'test_output' / 'primitives_bounds.npy')


# def example_extract_reference_primitives():
#     vertex_group_reference = VertexGroupReference(filepath='../data/reference.ply')
#     vertex_group_reference.save_vg(filepath='../output/reference.vg')


if __name__ == '__main__':
    example_extract_primitives()
    # example_extract_reference_primitives()
