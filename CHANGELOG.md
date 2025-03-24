# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Unit tests integrated with GitHub Actions workflow.

### Changed
- Rename previous `tests` to `tutorials`.
- `CellComplex.prioritise_planes` defaults `prioritise_verticals=False`.

## [0.2.6] - 2025-02-03
### Added
- `engine='mesh'` for `AdjacencyGraph.save_surface_obj` for mesh output (https://github.com/chenzhaiyu/abspy/issues/18).
- `.github/workflows/tests.yml`
- `CONTRIBUTING.md`

### Removed
- `engine='projection'` and `engine='sorted'` for `AdjacencyGraph.save_surface_obj`.

### Changed
- Rename `engine='rendering'` &rarr; `engine='soup'` for `AdjacencyGraph.save_surface_obj`.
- Reduce `environment.yml` to core dependencies.
- Replace SDF input with fixed values in `tests/test_combined.py`.

## [0.2.5] - 2024-12-01
### Added
- `VertexGroupReference.bottom_indices`.
- `VertexGroupReference.wall_indices`.
- `VertexGroupReference.perturb` to perturb plane normals.
- `VertexGroupReference.inject_points` to inject and overwrite points.
- `VertexGroup.save_cloud` and `VertexGroupReference.save_cloud` to save point cloud and optionally normals.
- Copyright notice.

### Changed
- `VertexGroupReference.bounds` &rarr; `VertexGroupReference.aabbs`.
- `VertexGroup.save_bounds_npy` &rarr; `VertexGroup.save_aabbs_npy`.
- Correct SDF file `test_sdf.npy` for improved complex with OBB.
- Update Sphinx requirements.
- Fix writing (`.vg`/`.bvg`) by removing a place (previously KSR would crash).
- Fix reading multi-line points from VG.

## [0.2.4] - 2023-12-02
### Added
- Options to refit and skip global vertex group.
- OBB pre-intersection test to reduce runtime and maintain compactness.
- An example for convex decomposition from mesh in README.
- RTD configuration `.readthedocs.yaml`.
- A complex church test mesh.

### Removed
- Obsolete PyPi `requirements.txt`.
- The `Why adaptive` section in README.

### Changed
- A few trivial optimizations.
- Rename test meshes.
- Upgrade `sage=10.0`.

## [0.2.3] - 2023-05-03
### Added
- Issue templates.
- GitHub workflow to publish package.
- Animation GIF in README.
- Download stats badge in README.
- License attribute in `setup.cfg`.
- `AdjacencyGraph.to_uids()`.
- option `location=random_t` and `location=boundary` in `CellComplex.cell_representatives`.
- Conda configuration file `environment.yml`.
- `CellComplex.cells_boundary()`.

### Removed
- Download stats in README.

### Changed
- Fix docs autofunction names `__init__`.
- Fix planar segment extraction from `VertexGroupReference`.
- Clean Misc section in README.
- Rename `location=random` to `location=random_r` (rejection-based) in `CellComplex.cell_representatives()`.

## [0.2.2] - 2022-11-07
### Added
- `version.py`

### Changed
- Sphinx finds local abspy package.
- Fix docs autofunction name `CellComplex.cells_in_mesh`.
- Clarify variable in README example.

## [0.2.1] - 2022-11-02
### Added
- Changelog.
- `bumpver` configuration in `pyproject.toml`.
- `CellComplex.save()` method to dump complex object.
- Option to enable multi-processing neighbour intersections (though much slower for now).
- Option to disable logging.

### Changed
- Introduce tolerance to `_bbox_intersect()` to deal with degenerate cases.
- Add `pathlib.Path` to supported filepath types in docstring.
- Add function entries to API.
- Change MTL filename to the same as corresponding OBJ.
- Change bibtex entry for citation.
