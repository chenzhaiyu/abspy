# pyabsp: a Python tool for 3D adaptive binary space partitioning

## Introduction

This repository implements adaptive binary space partitioning: an ambient 3D space is recessively partitioned into non-overlapping convexes with pre-detected planar primitives.

![doc/partition.png](./doc/partition.png)

An exact kernal of [SageMath](https://www.sagemath.org/) is used for robust Boolean spatial operations. The rational-based exact representation avoids degenerate cases that may otherwise cause inconsistencies in the geometry.

## Installation

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

For Windows users, consider building SageMath from source or install all other dependencies into a [pre-built SageMath environment](https://doc.sagemath.org/html/en/installation/binary.html).

### Install other requirements

```bash
pip install -r requirements.txt
```

Optionally, install [trimesh](https://github.com/mikedh/trimesh) and [pyglet](https://github.com/pyglet/pyglet) for benchmarking and visualisation, respectively:

```bash
pip install trimesh pyglet
```

### Install pyabsp

```
pip install pyabsp
```

## Usage

### Primitive detection

Primitives are defined in `VertexGroup` (a group of vertices for each primitive). To create such data from a point cloud, you can use `Mapple` in [Easy3D](https://github.com/LiangliangNan/Easy3D).

### Classes and methods

See `./tests/test_{primitive, complex, graph}.py` for how to use classes `VertexGroup`, `CellComplex`, `AdjacencyGraph`, respectively, or for a combined workflow with `./tests/test_combined.py`.

