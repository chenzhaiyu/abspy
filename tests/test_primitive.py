from absp import VertexGroup


def example_extract_primitives():

    vertex_group = VertexGroup(filepath='../data/Fig1.vg')
    vertex_group.save_planes_npy('../output/primitives_planes.npy')
    vertex_group.save_bounds_npy('../output/primitives_bounds.npy')


if __name__ == '__main__':
    example_extract_primitives()

