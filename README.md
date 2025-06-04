<img src="https://raw.githubusercontent.com/chenzhaiyu/abspy/main/docs/source/_static/images/logo.png" width="480"/>

-----------
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://img.shields.io/pypi/v/abspy)](https://pypi.python.org/pypi/abspy/) [![PyPI downloads](https://img.shields.io/pypi/dm/abspy?color=blue)](https://pypi.python.org/pypi/abspy/) [![Build status](https://readthedocs.org/projects/abspy/badge/)](https://abspy.readthedocs.io/en/latest/)

## Introduction

***abspy*** is a Python tool for 3D *adaptive binary space partitioning* and beyond. It adaptively partitions ambient 3D space into a linear cell complex using planar primitives, dynamically generating an adjacency graph in the process. Designed primarily for compact surface reconstruction, *abspy* also supports a range of other applications.

<div align="center" width="480">
  <img src="https://raw.githubusercontent.com/chenzhaiyu/abspy/main/docs/source/_static/images/animation.gif"><br>
</div>

## Key features

* Manipulation of planar primitives from point cloud or reference mesh
* Linear cell complex creation with adaptive binary space partitioning (a-BSP)
* Dynamic BSP-tree ([NetworkX](https://networkx.org/) graph) updated locally upon primitive insertion
* Support of polygonal surface reconstruction with graph cut
* Compatible data structure with [Easy3D](https://github.com/LiangliangNan/Easy3D) on point cloud, primitive, mesh and cell complex
* Robust spatial operations underpinned by the rational ring from [SageMath](https://www.sagemath.org/)'s exact kernel

## Installation

### All-in-one installation

Create a conda environment with the latest *abspy* release and all its dependencies installed:

```bash
git clone https://github.com/chenzhaiyu/abspy && cd abspy
conda env create -f environment.yml && conda activate abspy
```

### Manual installation

Still easy! Create a conda environment and enter it: 

```bash
conda create --name abspy python=3.11 && conda activate abspy
```

Install the dependencies:

```bash
# You may replace `conda` with `mamba` for faster package parsing
# conda install mamba -c conda-forge
conda install -c conda-forge networkx numpy tqdm scikit-learn matplotlib colorlog scipy trimesh rtree pyglet sage=10.2 
```

Preferably, the latest *abspy* release can be found and installed via [PyPI](https://pypi.org/project/abspy/):

```bash
pip install abspy
```

Otherwise, you can install the latest version locally:

```bash
git clone https://github.com/chenzhaiyu/abspy && cd abspy
pip install .
```

## Quick start

### Example 1 - Reconstruction from point cloud

The example loads a point cloud to `VertexGroup` (`.vg`), partitions ambient space into a cell complex, creates the adjacency graph, and extracts the object's outer surface.

```python
from abspy import VertexGroup, AdjacencyGraph, CellComplex

# load a point cloud in VertexGroup
vertex_group = VertexGroup(filepath='tutorials/data/test_points.vg')

# normalise point cloud
vertex_group.normalise_to_centroid_and_scale()

# initialise cell complex
cell_complex = CellComplex(vertex_group.planes, vertex_group.aabbs, vertex_group.obbs, vertex_group.points_grouped, build_graph=True, additional_planes=None)

# refine planar primitives
cell_complex.refine_planes()

# prioritise certain planes (e.g., vertical ones)
cell_complex.prioritise_planes(prioritise_verticals=False)

# construct cell complex
cell_complex.construct()

# print info about cell complex
cell_complex.print_info()

# cells inside reference mesh
cells_in_mesh = cell_complex.cells_in_mesh('tutorials/data/test_mesh.ply')

# build adjacency graph from cell complex
adjacency_graph = AdjacencyGraph(cell_complex.graph)

# calculate volumes
volumes = cell_complex.volumes(multiplier=10e5)
volumes = [0.1 if i in cells_in_mesh else vol for i, vol in enumerate(volumes)]

# assign graph weights
adjacency_graph.assign_weights_to_n_links(cell_complex.cells, attribute='area_overlap', factor=0.000, cache_interfaces=True)
adjacency_graph.assign_weights_to_st_links(adjacency_graph.to_dict(volumes))

# perform graph cut to extract surface
_, _ = adjacency_graph.cut()

# save surface model to an OBJ file
adjacency_graph.save_surface_obj('tutorials/output/surface.obj', engine='mesh')
```

### Example 2 - Convex decomposition from mesh

The example loads a mesh to `VertexGroupReference`, partitions ambient space into a cell complex, identifies cells inside reference mesh, and visualizes the cells.

```python
from abspy import VertexGroupReference
vertex_group_reference = VertexGroupReference(filepath='tutorials/data/test_mesh.ply')

# initialise cell complex
cell_complex = CellComplex(vertex_group_reference.planes, vertex_group_reference.aabbs, vertex_group_reference.obbs, build_graph=True)

# construct cell complex 
cell_complex.construct()

# cells inside reference mesh
cells_in_mesh = cell_complex.cells_in_mesh('tutorials/data/test_mesh.ply', engine='distance')

# save cell complex file
cell_complex.save('tutorials/output/complex.cc')

# visualise the inside cells
if len(cells_in_mesh):
    cell_complex.visualise(indices_cells=cells_in_mesh)
```

Please find the usage of *abspy* at [API reference](https://abspy.readthedocs.io/en/latest/api.html), with self-explanatory examples in [`./tutorials`](https://github.com/chenzhaiyu/abspy/blob/main/tutorials). 
For the data structure of a `.vg`/`.bvg` file, please refer to [VertexGroup](https://abspy.readthedocs.io/en/latest/vertexgroup.html).

## Testing *abspy*

To run the test suite, first install [`pytest`](https://docs.pytest.org/en/stable/), and execute all tests:

```bash
pytest
```

## Contributing to *abspy*

Please see the [Contribution Guide](https://github.com/chenzhaiyu/abspy/blob/main/CONTRIBUTING.md) for more information. 

## FAQ

* **How can I install *abspy* on Windows**

For Windows users, you may need to build [SageMath from source](https://doc.sagemath.org/html/en/installation/source.html) or install all other dependencies into a [pre-built SageMath environment](https://doc.sagemath.org/html/en/installation/binary.html). Otherwise, virtualization with [docker](https://www.docker.com/) may come to the rescue.

* **How can I use *abspy* for surface reconstruction?**

As shown in [`Example 1`](https://github.com/chenzhaiyu/abspy/tree/main#example-1---reconstruction-from-point-cloud), the surface is defined between adjacent cells where one is *inside* and the other *outside*. For more information, please refer to ***[Points2Poly](https://github.com/chenzhaiyu/points2poly)*** which integrates ***abspy*** with deep implicit fields, and ***[PolyGNN](https://github.com/chenzhaiyu/polygnn)*** which learns a piecewise planar occupancy function supported by ***abspy***, for 3D building reconstruction.

![adaptive](https://raw.githubusercontent.com/chenzhaiyu/abspy/main/docs/source/_static/images/surface.png)


## License

See the [MIT](https://raw.githubusercontent.com/chenzhaiyu/abspy/main/LICENSE) license for details.

## Citation

If you use *abspy* in a scientific work, please consider citing the paper:

```bibtex
@article{chen2022points2poly,
  title = {Reconstructing compact building models from point clouds using deep implicit fields},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume = {194},
  pages = {58-73},
  year = {2022},
  issn = {0924-2716},
  doi = {https://doi.org/10.1016/j.isprsjprs.2022.09.017},
  url = {https://www.sciencedirect.com/science/article/pii/S0924271622002611},
  author = {Zhaiyu Chen and Hugo Ledoux and Seyran Khademi and Liangliang Nan}
}
```
