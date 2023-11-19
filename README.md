<img src="https://raw.githubusercontent.com/chenzhaiyu/abspy/main/docs/source/_static/images/logo.png" width="480"/>

-----------
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://img.shields.io/pypi/v/abspy)](https://pypi.python.org/pypi/abspy/) [![Build status](https://readthedocs.org/projects/abspy/badge/)](https://abspy.readthedocs.io/en/latest/)

## Introduction

***abspy*** is a Python tool for 3D *adaptive binary space partitioning* and beyond: an ambient 3D space is adaptively partitioned to form a linear cell complex with pre-detected planar primitives in a point cloud, where an adjacency graph is dynamically obtained. Though the tool is designed primarily to support compact surface reconstruction, it can be extrapolated to other applications as well.

<div align="center" width="480">
  <img src="https://raw.githubusercontent.com/chenzhaiyu/abspy/main/docs/source/_static/images/animation.gif"><br>
</div>

## Key features

* Manipulation of planar primitives detected from point clouds
* Linear cell complex creation with adaptive binary space partitioning (a-BSP)
* Dynamic BSP-tree ([NetworkX](https://networkx.org/) graph) updated locally upon insertion of primitives
* Support of polygonal surface reconstruction from graph cuts
* Compatible data structure with [Easy3D](https://github.com/LiangliangNan/Easy3D) on point clouds, primitives, cell complexes and surfaces
* Robust Boolean spatial operations underpinned by the rational ring from [SageMath](https://www.sagemath.org/)'s exact kernel

## Installation

### All-in-one installation

Create a conda environment with the latest *abspy* release and all its dependencies installed:

```bash
git clone https://github.com/chenzhaiyu/abspy && cd abspy
conda env create -f environment.yml && conda activate abspy
```

### Manual installation

Create a conda environment and enter it: 

```bash
conda create --name abspy python=3.10 && conda activate abspy
```

Install the dependencies:

```bash
conda install -c conda-forge networkx numpy tqdm scikit-learn matplotlib colorlog scipy trimesh rtree pyglet sage=10.0 
```

Alternatively, you can use [mamba](https://github.com/mamba-org/mamba) for faster package parsing installation:

```bash
conda install mamba -c conda-forge
mamba install -c conda-forge networkx numpy tqdm scikit-learn matplotlib colorlog scipy trimesh rtree pyglet sage=10.0 
```

You might want to install [pyembree](https://github.com/trimesh/embreex) for better ray-tracing:

```bash
pip install embreex
```

Preferably, *abspy* can be found and easily installed via [PyPI](https://pypi.org/project/abspy/):

```bash
pip install abspy
```

Otherwise, you can install the latest version locally:

```
pip install .
```

## Quick start

Here is an example of loading a point cloud in `VertexGroup` (`.vg`), partitioning the ambient space into candidate convexes, creating the adjacency graph, and extracting the object's outer surface.

```python
import numpy as np
from abspy import VertexGroup, AdjacencyGraph, CellComplex

# load a point cloud in VertexGroup 
vertex_group = VertexGroup(filepath='points.vg')

# normalise the point cloud
vertex_group.normalise_to_centroid_and_scale()

# additional planes to append (e.g., the bounding planes)
additional_planes = [[0, 0, 1, -1], [1, 2, 3, -4]]

# initialise CellComplex from planar primitives
cell_complex = CellComplex(vertex_group.planes, vertex_group.bounds, vertex_group.points_grouped, build_graph=True, additional_planes=additional_planes)

# refine planar primitives
cell_complex.refine_planes()

# prioritise certain planes (e.g., vertical ones)
cell_complex.prioritise_planes(prioritise_verticals=True)

# construct CellComplex 
cell_complex.construct()

# print info on the cell complex
cell_complex.print_info()

# build adjacency graph of the cell complex
adjacency_graph = AdjacencyGraph(cell_complex.graph)

# assign weights (e.g., SDF values provided by neural network prediction) to graph 
weights_dict = adjacency_graph.to_dict(...)
adjacency_graph.assign_weights_to_n_links(cell_complex.cells, attribute='area_overlap', factor=0.001, cache_interfaces=True)
adjacency_graph.assign_weights_to_st_links(weights_dict)

# perform graph-cut to extract surface
_, _ = adjacency_graph.cut()

# save surface model to an OBJ file
adjacency_graph.save_surface_obj('surface.obj', engine='rendering')
```

Usage of *abspy* can be found at [API reference](https://abspy.readthedocs.io/en/latest/api.html). For the data structure of a `.vg`/`.bvg` file, please refer to [VertexGroup](https://abspy.readthedocs.io/en/latest/vertexgroup.html).


## FAQ

* **How can I install *abspy* on Windows?**

For Windows users, you may need to build [SageMath from source](https://doc.sagemath.org/html/en/installation/source.html) or install all other dependencies into a [pre-built SageMath environment](https://doc.sagemath.org/html/en/installation/binary.html). Otherwise, virtualization with [docker](https://www.docker.com/) may come to the rescue.

* **How can I use *abspy* for surface reconstruction?**

With a constructed cell complex, the surface can be addressed by graph cut &mdash; in between adjacent cells where one being *inside* and the other being *outside* &mdash; exactly where the cut is performed. For more information, please refer to ***[Points2Poly](https://github.com/chenzhaiyu/points2poly)*** which wraps ***abspy*** for building surface reconstruction.

![adaptive](https://raw.githubusercontent.com/chenzhaiyu/abspy/main/docs/source/_static/images/surface.png)


## License

[MIT](https://raw.githubusercontent.com/chenzhaiyu/abspy/main/LICENSE)

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
