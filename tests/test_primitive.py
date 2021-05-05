from absp import VertexGroup, VertexGroupReference


def example_extract_primitives():

    vertex_group = VertexGroup(filepath='../data/Fig1.vg')
    vertex_group.save_planes_npy('../output/primitives_planes.npy')
    vertex_group.save_bounds_npy('../output/primitives_bounds.npy')


def example_extract_reference_primitives():
    vertex_group_reference = VertexGroupReference(filepath='../data/reference.ply')
    vertex_group_reference.save_vg(filepath='../output/reference.vg')


if __name__ == '__main__':
    example_extract_primitives()
    example_extract_reference_primitives()
