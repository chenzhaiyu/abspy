<img src="https://raw.githubusercontent.com/chenzhaiyu/abspy/docs/docs/source/_static/images/logo.png" width="480"/>

-----------
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://badge.fury.io/py/abspy.svg)](https://pypi.python.org/pypi/abspy/) [![Build status](https://readthedocs.org/projects/abspy/badge/)](https://abspy.readthedocs.io/en/latest/)

## Introduction

**abspy** is a Python tool for 3D adaptive binary space partitioning and beyond: an ambient 3D space is recessively partitioned into non-overlapping convexes with pre-detected planar primitives, where the adjacency graph is dynamically obtained. It is implemented initially for surface reconstruction, but can be extrapolated to other applications nevertheless.

![partition](https://raw.githubusercontent.com/chenzhaiyu/abspy/main/docs/source/_static/images/partition.png)

An exact kernel of [SageMath](https://www.sagemath.org/) is used for robust Boolean spatial operations. This rational-based representation help avoid degenerate cases that may otherwise result in inconsistencies in the geometry.

## Installation

### Install requirements

All dependencies except for [SageMath](https://www.sagemath.org/) can be easily installed with [PyPI](https://pypi.org/):

```bash
pip install -r requirements.txt
```

Optionally, install [trimesh](https://github.com/mikedh/trimesh) and [pyglet](https://github.com/pyglet/pyglet) for benchmarking and visualisation, respectively:

```bash
pip install trimesh pyglet
```

### Install SageMath

For Linux and macOS users, the easist is to install from [conda-forge](https://conda-forge.org/):

```bash
conda config --add channels conda-forge
conda install sage
```

Alternatively, you can use [mamba](https://github.com/mamba-org/mamba) for faster parsing and package installation:

```bash
conda config --add channels conda-forge
conda install mamba
mamba install sage
```

For Windows users, you may have to build SageMath from source or install all other dependencies into a [pre-built SageMath environment](https://doc.sagemath.org/html/en/installation/binary.html).

### Install abspy

**abspy** can be found and installed via [PyPI](https://pypi.org/project/abspy/):

```bash
pip install abspy
```

## Quick start

Here is an example of loading a point cloud in `VertexGroup` (`.vg`), partitioning the ambient space into candidate convexes, creating the adjacency graph and extracting the outer surface of the object. For the data structure of a `.vg` file, please refer to [VertexGroup](https://raw.githubusercontent.com/chenzhaiyu/abspy/main/docs/source/vertexgroup.md).

```python
import numpy as np
from abspy import VertexGroup, AdjacencyGraph, CellComplex

# load a point cloud in VertexGroup 
vertex_group = VertexGroup(filepath='points.vg')

# normalise the point cloud
vertex_group.normalise_to_centroid_and_scale()

# retrieve planes, bounds and points from VertexGroup
planes, bounds, points = np.array(vertex_group.planes), np.array(vertex_group.bounds), np.array(vertex_group.points_grouped, dtype=object)

# additional planes to append (e.g., the bounding planes)
additional_planes = [[0, 0, 1, -bounds[:, 0, 2].min()]]

# initialise CellComplex from planar prititives
cell_complex = CellComplex(planes, bounds, points, build_graph=True, additional_planes=additional_planes)

# refine planar primitives
cell_complex.refine_planes()

# prioritise certain planes
cell_complex.prioritise_planes()

# construct CellComplex 
cell_complex.construct()

# print info on the cell complex
cell_complex.print_info()

# visualise the cell complex (only if trimesh installation is found)
cell_complex.visualise()

# build adjacency graph of the cell complex
graph = AdjacencyGraph(cell_complex.graph)

# apply random weights (could instead be the predicted probability
# for each convex being selected as composing the object in practice)
weights_list = np.array([random.random() for _ in range(cell_complex.num_cells)])
weights_list *= cell_complex.volumes(multiplier=10e5)
weights_dict = graph.to_dict(weights_list)

# assign weights to n-links and st-links to the graph
graph.assign_weights_to_n_links(cell_complex.cells, attribute='area_overlap', factor=0.1, cache_interfaces=True)
graph.assign_weights_to_st_links(weights_dict)

# perform graph-cut
_, _ = graph.cut()

# save surface model to an obj file
graph.save_surface_obj('surface.obj', engine='rendering')
```

## Misc

* **Why adaptive?**

Adaptive space partitioning can significantly reduce computations for cell complex creation, compared to an exhaustive partitioning strategy. The excessive number of cells not only hinders computation but also inclines to defective surfaces on subtle structures where inaccurate labels are more likely to be assigned.

![adaptive](https://raw.githubusercontent.com/chenzhaiyu/abspy/main/docs/source/_static/images/adaptive.png)

* **What weights to assign to the adjacency graph?**

There are two kinds of weights to assign to an S-T graph: either to *n-links* or to *st-links*. For surface reconstruction using [graph-cut](https://en.wikipedia.org/wiki/Cut_(graph_theory)), assign the predicted probability of occupancy for each cell to the *st-links*, while assign overlap area to the *n-links*. Read this [paper](https://arxiv.org/pdf/2112.13142.pdf) for more information on this Markov random field formulation.

## License

[MIT](https://raw.githubusercontent.com/chenzhaiyu/abspy/main/LICENSE)

## Citation

If you use abspy in a scientific work, please cite:

```latex
@article{chen2021reconstructing,
  title={Reconstructing Compact Building Models from Point Clouds Using Deep Implicit Fields},
  author={Chen, Zhaiyu and Khademi, Seyran and Ledoux, Hugo and Nan, Liangliang},
  journal={arXiv preprint arXiv:2112.13142},
  year={2021}
}
```
