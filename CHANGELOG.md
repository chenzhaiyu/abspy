# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `bumpver` configuration in `pyproject.toml`.

## [0.2.0] - 2022-11-02
### Added
- Changelog.
- `CellComplex.save()` method to dump complex object.
- Option to enable multi-processing neighbour intersections (though much slower for now).
- Option to disable logging.

### Changed
- Introduce tolerance to `_bbox_intersect()` to deal with degenerate cases.
- Add `pathlib.Path` to supported filepath types in docstring.
- Add function entries to API.
- Change MTL filename to the same as corresponding OBJ.
- Change bibtex entry for citation.