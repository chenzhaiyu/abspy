---
title: 'abspy: A Python package for 3D adaptive binary space partitioning and modeling'
tags:
  - Python
  - 3D modeling
  - point cloud
  - binary space partitioning
  - surface reconstruction
authors:
  - name: Zhaiyu Chen
    orcid: 0000-0001-7084-0994
    affiliation: "1"
affiliations:
 - name: Technical University of Munich, Germany
   index: 1
date: 20 February 2025
bibliography: paper.bib
---

# Summary

Efficient and robust space partitioning of 3D space underpins many applications in computer graphics. *abspy* is a Python package for adaptive binary space partitioning and 3D modeling. At its core, *abspy* constructs a plane arrangement by recursively subdividing 3D space with planar primitives to form a linear cell complex that reflects the underlying geometric structure. This adaptive scheme iteratively refines the spatial decomposition, producing a compact representation that supports efficient 3D modeling and analysis. Built on robust arithmetic kernels and interoperable data structures, *abspy* exposes a high‑level, Python‑native API that lets researchers prototype advanced 3D pipelines such as surface reconstruction, volumetric analysis, and feature extraction for machine learning.

# Statement of need

In many scientific and engineering workflows, raw 3D data must be transformed into compact volumetric representations for visualization, simulation, and analysis of complex 3D environments. Sensor-acquired point clouds and photogrammetric meshes often contain hundreds of millions of irregular, noisy samples that are expensive to store, transmit, and query, and that provide little inherent structure. Converting such data into a well-defined spatial decomposition, in which each cell represents a coherent geometric region, reduces memory usage, spares downstream algorithms from ad-hoc filtering and enables exact spatial predicates such as intersection, containment, and adjacency. The central challenge is to obtain this decomposition without inheriting the aliasing artefacts of coarse voxel grids or the combinatorial explosion produced by exhaustive plane arrangements [@edelsbrunner1986constructing].

*How can 3D space be partitioned into a minimal yet meaningful set of units that faithfully capture geometric detail?* Adaptive binary space partitioning (BSP) [@murali1997consistent] addresses this challenge by dynamically subdividing 3D space according to the data's intrinsic features, constructing a plane arrangement that conforms to geometric boundaries. Despite its theoretical advantages, a high-level implementation suitable for modern research use cases has been missing. *abspy* fills this gap with a modern, Python-native API for adaptive partitioning that delivers reliable and efficient 3D modeling. Researchers can thus tackle large-scale and complex reconstruction challenges with great reliability and precision. Unlike general-purpose libraries such as CGAL [@fabri2009cgal] and Open3D [@zhou2018open3d], which mainly supply low-level geometric utilities, *abspy* uniquely integrates adaptive BSP and exact arithmetic within a single, Python-native framework, facilitating direct integration into research workflows tailored for compact, dynamic 3D modeling tasks and beyond.

\autoref{fig:2d} illustrates how *abspy* uses user-supplied or extracted planar primitives to recursively subdivide space into a convex cell complex. Beginning with user-supplied planes, we insert each plane only into the BSP-tree leaves whose bounding boxes it intersects. This local, on-demand splitting keeps every cell convex, updates adjacency on the fly, and, by skipping unrelated plane pairs, avoids the prohibitive cost of exhaustive arrangements. The result is a compact cell complex. Initially developed for research purposes, *abspy* has since evolved to support diverse 3D applications, particularly for compact surface reconstructions. It has been employed in several research projects and featured in scientific publications [@chen2022points2poly; @chen2024polygnn; @sulzer2024concise] as well as graduate students' projects.

![A 2D illustration for adaptive binary space partitioning. The ambient space is recursively partitioned into a cell complex with the insertion of planar primitives.\label{fig:2d}](assets/2d.png){ width=95% }


# Overview of features


The core features of *abspy* include:

- **Planar primitive manipulation:** *abspy* extracts planar primitives from point clouds or meshes, and refines or perturbs the primitives to control noise, ensuring a robust representation of the input geometric structures.
- **Linear cell complex construction:** Using adaptive binary space partitioning, *abspy* recursively subdivides 3D space into a cell complex that reflects the data’s spatial layout. This structured decomposition facilitates efficient spatial queries and further processing.
- **Dynamic graph generation:** The package iteratively constructs and updates a BSP-tree as a dynamic graph, supporting operations like connectivity analysis and neighborhood searches.
- **Robust spatial operations:** By leveraging the rational ring from SageMath [@sage2024sage], *abspy* performs exact geometric computations that avoid inaccuracies associated with floating-point arithmetic, leading to reliable intersection tests and robust boundary determinations.
- **Surface reconstruction:** *abspy* identifies boundaries between interior and exterior cells to reconstruct polygonal surfaces using graph cuts, enabling the generation of surface models suitable for visualization and analysis.
- **Ease of integration:** With a Pythonic interface, comprehensive documentation, and practical examples, *abspy* integrates smoothly into existing research workflows. Its data structures are designed for interoperability with tools such as NetworkX [@hagberg2008exploring] and Easy3D [@nan2021easy3d], facilitating further development.

\autoref{fig:3d} showcases a few geometric representations that *abspy* can provide. From an input point cloud, for example, the library can manipulate points grouped by their supporting planar primitives, build a linear cell complex with its adaptive partitioning routine, extract a watertight polygonal surface from that complex through its graph-cut API, and sample representative points from the cells with the built-in samplers. These representations allow users to choose whichever level of abstraction best fits their application or to combine several of them as needed. The [documentation of *abspy*](https://abspy.readthedocs.io/) contains tutorials for the package, its API reference, file format specifications, etc.

![An overview of representations supported by *abspy*, shown left to right: points grouped and colored by planar primitives; cell complex of colored polyhedral volumes generated by adaptive space partitioning; reconstructed surface mesh; and sampled representative points from each cell volume, colored to match their cells.\label{fig:3d}](assets/3d.png){ width=95% }

# Acknowledgements

I acknowledge the feedback provided by [Liangliang Nan](https://github.com/LiangliangNan), [Hugo Ledoux](https://github.com/hugoledoux), Yuqing Wang, and Qian Bai, which greatly contributed to the development of this package.

# References
