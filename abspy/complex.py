# Copyright (c) 2022 Zhaiyu Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
complex.py
----------

Cell complex from planar primitive arrangement.

A linear cell complex is constructed from planar primitives
with adaptive binary space partitioning: upon insertion of a primitive
only the local cells that are intersecting it will be updated,
so will be the corresponding adjacency graph of the complex.
"""

import os
import string
from pathlib import Path
import itertools
import heapq
from copy import copy
from random import random, choices, uniform, randint
import pickle
import time
import multiprocessing

import numpy as np
from tqdm import trange
import networkx as nx
import trimesh
from sage.all import polytopes, QQ, RR, Polyhedron
from sage.geometry.triangulation.point_configuration import PointConfiguration
from sklearn.decomposition import PCA

from .logger import attach_to_log
from .primitive import VertexGroup

logger = attach_to_log()


class CellComplex:
    """
    Class of cell complex from planar primitive arrangement.
    """

    def __init__(self, planes, aabbs, obbs=None, points=None, initial_bound=None, initial_padding=0.1, additional_planes=None,
                 build_graph=False, quiet=False):
        """
        Init CellComplex.
        Class of cell complex from planar primitive arrangement.

        Parameters
        ----------
        planes: (n, 4) float
            Plana parameters
        aabbs: (n, 2, 3) float
            Corresponding bounding box bounds of the planar primitives
        obbs: (n, 4, 3) float
            Corresponding oriented bounds of the planar primitives
        points: (n, ) object of float
            Points grouped into primitives, points[any]: (m, 3)
        initial_bound: None or (2, 3) float
            Initial bound to partition
        build_graph: bool
            Build the cell adjacency graph if set True.
        additional_planes: None or (n, 4) float
            Additional planes to append to the complex,
            can be missing planes due to occlusion or incapacity of RANSAC
        quiet: bool
            Disable logging and progress bar if set True
        """
        self.quiet = quiet
        if self.quiet:
            logger.disabled = True

        self.aabbs = aabbs  # numpy.array over RDF
        self.obbs = obbs      # numpy.array over RDF
        self.planes = planes  # numpy.array over RDF
        self.points = points
        # missing planes due to occlusion or incapacity of RANSAC
        self.additional_planes = additional_planes

        self.initial_bound = initial_bound if initial_bound else self._pad_bound(
            [np.amin(aabbs[:, 0, :], axis=0), np.amax(aabbs[:, 1, :], axis=0)],
            padding=initial_padding)
        self.cells = [self._construct_initial_cell()]  # list of QQ
        self.cells_bounds = [self.cells[0].bounding_box()]  # list of QQ

        if build_graph:
            self.graph = nx.Graph()
            self.graph.add_node(0)  # the initial cell
            self.index_node = 0  # unique for every cell ever generated
        else:
            self.graph = None

        self.constructed = False

    def _construct_initial_cell(self):
        """
        Construct initial bounding cell.

        Return
        ----------
        as_object: Polyhedron object
            Polyhedron object of the initial cell,
            a cuboid with 12 triangular facets.
        """
        return polytopes.cube(
            intervals=[[QQ(self.initial_bound[0][i]), QQ(self.initial_bound[1][i])] for i in range(3)])

    def refine_planes(self, theta=10 * 3.1416 / 180, epsilon=0.005, normalise_normal=False):
        """
        Refine planar primitives.

        First, compute the angle of the supporting planes for each pair of planar primitives.
        Then, starting from the pair with the smallest angle, test if the following two conditions are met:
        (a) the angle between is lower than a threshold. (b) more than a specified number of points lie on
        both primitives. Merge the two primitives and fit a new plane if the conditions are satisfied.

        Parameters
        ----------
        theta: float
            Angle tolerance, primitive pair has to be less than this tolerance to be refined
        epsilon: float
            Distance tolerance, primitive pair has to be less than this tolerance to be refined
        normalise_normal: bool
            Normalise normal if set True
        """
        if self.points is None:
            raise ValueError('point coordinates are needed for plane refinement')
        logger.info('refining planar primitives')
        if self.additional_planes:
            logger.warning('additional planes are not refined')

        # shallow copy of the primitives
        planes = list(copy(self.planes))
        aabbs = list(copy(self.aabbs))
        obbs = list(copy(self.obbs))
        points = list(copy(self.points))

        # pre-compute cosine of theta
        theta_cos = np.cos(theta)

        # priority queue storing pairwise planes and their angles
        priority_queue = []

        # compute angles and save them to the priority queue
        for i, j in itertools.combinations(range(len(planes)), 2):
            # no need to normalise as PCA in primitive.py already does it
            angle_cos = np.abs(np.dot(planes[i][:3], planes[j][:3]))
            if normalise_normal:
                angle_cos /= (np.linalg.norm(planes[i][:3]) * np.linalg.norm(planes[j][:3]))
            heapq.heappush(priority_queue, [-angle_cos, i, j])  # negate to use max-heap

        # indices of planes to be merged
        to_merge = set()

        while priority_queue:
            # pair with the smallest angle
            pair = heapq.heappop(priority_queue)

            if -pair[0] > theta_cos:  # negate back to use max-heap
                # distance from the center of a primitive to the supporting plane of the other
                distance = np.abs(
                    np.dot((np.array(points[pair[1]]).mean(axis=0) - np.array(points[pair[2]]).mean(axis=0)),
                           planes[pair[1]][:3]))

                # assume normalised data so that epsilon can be an arbitrary number in (0, 1]
                if distance < epsilon and pair[1] not in to_merge and pair[2] not in to_merge:
                    # merge the two planes
                    points_merged = np.concatenate([points[pair[1]], points[pair[2]]])
                    planes_merged, obbs_merged = VertexGroup.fit_plane(points_merged)
                    aabbs_merged = [np.min([aabbs[pair[1]][0], aabbs[pair[2]][0]], axis=0).tolist(),
                                    np.max([aabbs[pair[1]][1], aabbs[pair[2]][1]], axis=0).tolist()]

                    # update to_merge
                    to_merge.update({pair[1]})
                    to_merge.update({pair[2]})

                    # push the merged plane into the heap
                    for i, p in enumerate(planes):
                        if i not in to_merge:
                            angle_cos = np.abs(np.dot(planes_merged[:3], p[:3]))
                            heapq.heappush(priority_queue, [-angle_cos, i, len(planes)])  # placeholder

                    # update the actual data with merged ones
                    points.append(points_merged)
                    planes.append(planes_merged)
                    aabbs.append(aabbs_merged)
                    obbs.append(obbs_merged)

            else:
                # no more possible coplanar pairs can exist in this priority queue
                break

        # delete the merged pairs
        for i in sorted(to_merge, reverse=True):
            del planes[i]
            del aabbs[i]
            del obbs[i]
            del points[i]

        logger.info('{} pairs of planes merged'.format(len(to_merge)))

        self.planes = np.array(planes)
        self.aabbs = np.array(aabbs)
        self.obbs = np.array(obbs)
        self.points = np.array(points, dtype=object)

    def prioritise_planes(self, prioritise_verticals=True):
        """
        Prioritise certain planes to favour building reconstruction.

        First, vertical planar primitives are accorded higher priority than horizontal or oblique ones
        to avoid incomplete partitioning due to missing data about building facades.
        Second, in the same priority class, planar primitives with larger areas are assigned higher priority
        than smaller ones, to make the final cell complex as compact as possible.
        Note that this priority setting is designed exclusively for building models.

        Parameters
        ----------
        prioritise_verticals: bool
            Prioritise vertical planes if set True
        """
        logger.info('prioritising planar primitives')
        # compute the priority
        indices_sorted_planes = self._sort_planes()

        if prioritise_verticals:
            indices_vertical_planes = self._vertical_planes(slope_threshold=0.9)
            bool_vertical_planes = np.in1d(indices_sorted_planes, indices_vertical_planes)
            indices_priority = np.append(indices_sorted_planes[bool_vertical_planes],
                                         indices_sorted_planes[np.invert(bool_vertical_planes)])
        else:
            indices_priority = indices_sorted_planes

        # reorder both the planes and their bounds
        self.planes = self.planes[indices_priority]
        self.aabbs = self.aabbs[indices_priority]

        # reorder obbs and points if not None
        if self.obbs is not None:
            self.obbs = self.obbs[indices_priority]
        if self.points is not None:
            self.points = self.points[indices_priority]

        # append additional planes with the highest priority
        if self.additional_planes:
            self.planes = np.concatenate([self.additional_planes, self.planes], axis=0)
            additional_bounds = [[[-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]]] * len(self.additional_planes)
            self.aabbs = np.concatenate([additional_bounds, self.aabbs], axis=0)  # never miss an intersection
            if self.obbs is not None:
                additional_obbs = [[[-np.inf, -np.inf, -np.inf], [-np.inf, np.inf, -np.inf], [np.inf, np.inf, -np.inf],
                                    [np.inf, -np.inf, -np.inf]]] * len(self.additional_planes)
                self.obbs = np.concatenate([additional_obbs, self.obbs], axis=0)
            if self.points is not None:
                additional_points = np.empty((1)) * len(self.additional_planes)
                self.points = np.concatenate([additional_points, self.points], axis=0)

        logger.debug('ordered planes: {}'.format(self.planes))
        logger.debug('ordered aabbs: {}'.format(self.aabbs))
        logger.debug('ordered obbs: {}'.format(self.obbs))

    def _vertical_planes(self, slope_threshold=0.9, epsilon=10e-5):
        """
        Return vertical planes.

        Parameters
        ----------
        slope_threshold: float
            Slope threshold, above which the planes are considered vertical
        epsilon: float
            Trivial term to avoid NaN

        Returns
        -------
        as_int: (n,) int
            Indices of the vertical planar primitives
        """
        slope_squared = (self.planes[:, 0] ** 2 + self.planes[:, 1] ** 2) / (self.planes[:, 2] ** 2 + epsilon)
        return np.where(slope_squared > slope_threshold ** 2)[0]

    def _sort_planes(self, mode='norm'):
        """
        Sort planes.

        Parameters
        ----------
        mode: str
            Mode for sorting, can be 'volume' or 'norm'

        Returns
        -------
        as_int: (n,) int
            Indices by which the planar primitives are sorted based on their bounding box volume
        """
        if mode == 'volume':
            volume = np.prod(self.aabbs[:, 1, :] - self.aabbs[:, 0, :], axis=1)
            return np.argsort(volume)[::-1]
        elif mode == 'norm':
            sizes = np.linalg.norm(self.aabbs[:, 1, :] - self.aabbs[:, 0, :], ord=2, axis=1)
            return np.argsort(sizes)[::-1]
        elif mode == 'area':
            # project the points supporting each plane onto the plane
            # https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
            raise NotImplementedError
        else:
            raise ValueError('mode has to be "volume" or "norm"')

    @staticmethod
    def _pad_bound(bound, padding=0.00):
        """
        Pad bound.

        Parameters
        ----------
        bound: (2, 3) float
            Bound of the query planar primitive
        padding: float
            Padding factor, defaults to 0.05.

        Returns
        -------
        as_float: (2, 3) float
            Padded bound
        """
        extent = bound[1] - bound[0]
        return [bound[0] - extent * padding, bound[1] + extent * padding]

    def _intersect_bound_plane(self, bound, plane, exhaustive=False, epsilon=10e-8):
        """
        Pre-intersection test between query primitive and existing cells, based on AABBs and plane parameters.

        Parameters
        ----------
        bound: (2, 3) float
            Bound of the query planar primitive
        plane: (4,) float
            Plane parameters
        exhaustive: bool
            Exhaustive partitioning, only for benchmarking
        epsilon: float
            Distance tolerance

        Returns
        -------
        as_int: (n,) int
            Indices of existing cells whose bounds intersect with bounds of the query primitive
            and intersect with the supporting plane of the primitive
        """
        if exhaustive:
            return np.arange(len(self.cells_bounds))

        # each planar primitive partitions only the 3D cells that intersect with it
        cells_bounds = np.array(self.cells_bounds)  # easier array manipulation
        center_targets = np.mean(cells_bounds, axis=1)  # N * 3
        extent_targets = cells_bounds[:, 1, :] - cells_bounds[:, 0, :]  # N * 3

        if bound[0][0] == -np.inf:
            intersection_bound = np.arange(len(self.cells_bounds))

        else:
            # intersection with existing cells' AABB
            center_query = np.mean(bound, axis=0)  # 3,
            center_distance = np.abs(center_query - center_targets)  # N * 3
            extent_query = bound[1] - bound[0]  # 3,

            # abs(center_distance) * 2 < (query extent + target extent) for every dimension -> intersection
            intersection_bound = \
                np.where(np.all(center_distance * 2 < extent_query + extent_targets + epsilon, axis=1))[0]

        # plane-AABB intersection test from extracted intersection_bound only
        # https://gdbooks.gitbooks.io/3dcollisions/content/Chapter2/static_aabb_plane.html
        # compute the projection interval radius of AABB onto L(t) = center + t * normal
        radius = np.dot(extent_targets[intersection_bound] / 2, np.abs(plane[:3]))
        # compute distance of box center from plane
        distance = np.dot(center_targets[intersection_bound], plane[:3]) + plane[3]
        # intersection between plane and AABB occurs when distance falls within [-radius, +radius] interval
        intersection_plane = np.where(np.abs(distance) <= radius + epsilon)[0]

        return intersection_bound[intersection_plane]

    def _intersect_obb(self, plane, obb, indices_cells, exhaustive=False, epsilon=10e-8):
        """
        Further-intersection test between query primitive and existing cells, based on their OBBs.

        Parameters
        ----------
        plane: (4,) float
            Plane parameters
        obb: (4, 3) float
            Oriented bounds of the planar primitive
        indices_cells: (n,) int
            Indices of existing cells whose bounds intersect with bounds of the query primitive
            and intersect with the supporting plane of the primitive in the first intersection test
        exhaustive: bool
            Exhaustive partitioning, only for benchmarking
        epsilon: float
            Distance tolerance

        Returns
        -------
        as_int: (n,) int
            Indices of existing cells whose bounds intersect with bounds of the query primitive
            and intersect with the supporting plane of the primitive
        """
        if exhaustive:
            return indices_cells

        vertices_query = obb  # 4 * 3
        norm_query = self._normalize(plane[:3])  # 3,
        edges_query = np.array([obb[0, :] - obb[1, :], obb[1, :] - obb[2, :]])  # 2 * 3
        edges_query = self._normalize(edges_query)  # 2 * 3

        vertices_targets = [cell.vertices_list() for i, cell in enumerate(self.cells) if i in indices_cells]
        norms_targets = [self._cell_norm(vertices) for vertices in vertices_targets]
        edges_targets = np.array(norms_targets)  # N * 3 * 3

        # all vertex lists have the same length for parallelization
        max_vertices = max([len(vertices) for vertices in vertices_targets])
        vertices_targets = [vertices + [vertices[-1]] * (max_vertices - len(vertices)) for vertices in vertices_targets]
        vertices_targets = np.array(vertices_targets)  # N * M * 3

        # 4 possible separating axis
        # 1: plane normal
        axis = norm_query  # 3,
        projection_query = np.einsum('ni,i->n', vertices_query, axis)  # 4,
        min_projection_query = np.min(projection_query, axis=0)  # 1,
        max_projection_query = np.max(projection_query, axis=0)  # 1,
        projection_targets = np.einsum('nmi,i->nm', vertices_targets, axis)
        min_projection_targets = np.min(projection_targets, axis=1)  # m,
        max_projection_targets = np.max(projection_targets, axis=1)  # m,
        condition1 = (min_projection_query > max_projection_targets + epsilon) | (
                min_projection_targets > max_projection_query + epsilon)

        # 2: cell_obb normals
        axis = norms_targets  # N * 3 * 3
        projection_query = np.einsum('nmi,bi->nmb', axis, vertices_query)  # N * 3 * 4
        min_projection_query = np.min(projection_query, axis=2)  # N * 3
        max_projection_query = np.max(projection_query, axis=2)  # N * 3
        projection_targets = np.einsum('nmi,nbi->nmb', axis, vertices_targets)  # N * 3 * M
        min_projection_targets = np.min(projection_targets, axis=2)  # N * 3 
        max_projection_targets = np.max(projection_targets, axis=2)  # N * 3
        condition2 = (min_projection_query > max_projection_targets + epsilon) | (
                min_projection_targets > max_projection_query + epsilon)
        condition2 = np.any(condition2, axis=1)

        # 3: plane normal cross plane_obb edges
        axis = np.cross(norm_query, edges_query)  # 2 * 3
        axis = self._normalize(axis)  # 2 * 3
        projection_query = np.einsum('ni,mi->nm', axis, vertices_query)  # 2 * 4
        min_projection_query = np.min(projection_query, axis=1)  # 2 * 3
        max_projection_query = np.max(projection_query, axis=1)  # 2 * 3
        projection_targets = np.einsum('ni,mbi->mnb', axis, vertices_targets)
        min_projection_targets = np.min(projection_targets, axis=2)
        max_projection_targets = np.max(projection_targets, axis=2)  # m * 2 * 3
        condition3 = (min_projection_query > max_projection_targets + epsilon) | (
                min_projection_targets > max_projection_query + epsilon)
        condition3 = np.any(condition3, axis=1)

        # 4: cell_obb edges cross plane_obb edges
        axis = np.cross(edges_query[np.newaxis, np.newaxis, :, :], edges_targets[:, :, np.newaxis, :])  # N * 3 * 2 * 3
        axis = axis.reshape(-1, 6, 3)  # N * 6 * 3
        axis = self._normalize(axis)  # N * 6 * 3
        projection_query = np.einsum('nmi,bi->nmb', axis, vertices_query)  # N * 6 * 4
        min_projection_query = np.min(projection_query, axis=2)  # N * 6
        max_projection_query = np.max(projection_query, axis=2)  # N * 6
        projection_targets = np.einsum('mni,mbi->mnb', axis, vertices_targets)  # N * 6 * M
        min_projection_targets = np.min(projection_targets, axis=2)  # N * 6
        max_projection_targets = np.max(projection_targets, axis=2)  # N * 6
        condition4 = (min_projection_query > max_projection_targets + epsilon) | (
                min_projection_targets > max_projection_query + epsilon)
        condition4 = np.any(condition4, axis=1)

        # combine all conditions
        condition = condition1 | condition2 | condition3 | condition4
        condition = np.array([not item for item in condition])
        intersection_plane = np.where(condition)[0]
        intersection_plane = indices_cells[intersection_plane]

        # if the OBB is unbounded, the intersection is incomplete
        if obb[0][0] == -np.inf and np.any(indices_cells == 0) and not np.any(intersection_plane == 0):
            intersection_plane = np.append(intersection_plane, 0)

        return intersection_plane

    @staticmethod
    def _normalize(axis):
        """
        Normalize axis.

        Parameters
        ----------
        axis: (n_0, n_1, ..., n_k, 3) float
        
        Returns
        -------
        as_float: (n_0, n_1, ..., n_k, 3) float
            Normalized axis
        """
        eps = 1e-12
        enom = np.linalg.norm(axis, axis=-1, keepdims=True)
        enom_clipped = np.clip(enom, eps, None)
        return axis / enom_clipped

    @staticmethod
    def _cell_norm(vertices):
        """
        Compute normals of a cell's oriented bounding box.

        Parameters
        ----------
        vertices: (n, 3) float
            cell's vertices list

        Returns
        -------        
        normals: (3,3) float
            Normals of the cell's oriented bounding box
        """
        pca = PCA(n_components=3)
        pca.fit(vertices)
        normals = pca.components_
        return normals

    @staticmethod
    def _inequalities(plane):
        """
        Inequalities from plane parameters.

        Parameters
        ----------
        plane: (4,) float
            Plane parameters

        Returns
        -------
        positive: (4,) float
            Inequality of the positive half-plane
        negative: (4,) float
            Inequality of the negative half-plane
        """
        positive = [QQ(plane[-1]), QQ(plane[0]), QQ(plane[1]), QQ(plane[2])]
        negative = [QQ(-element) for element in positive]
        return positive, negative

    def _index_node_to_cell(self, query):
        """
        Convert index in the node list to that in the cell list.
        The rationale behind is #nodes == #cells (when a primitive is settled down).

        Parameters
        ----------
        query: int
            Query index in the node list

        Returns
        -------
        as_int: int
            Query index in the cell list
        """
        return list(self.graph.nodes).index(query)

    def _intersect_neighbour(self, kwargs):
        """
        Intersection test between partitioned cells and neighbouring cell.
        Implemented for multi-processing across all neighbours.

        Parameters
        ----------
        kwargs: (int, Polyhedron object, Polyhedron object, Polyhedron object)
            (neighbour index, positive cell, negative cell, neighbouring cell)
        """
        n, cell_positive, cell_negative, cell_neighbour = kwargs['n'], kwargs['positive'], kwargs['negative'], kwargs[
            'neighbour']

        interface_positive = cell_positive.intersection(cell_neighbour)

        if interface_positive.dim() == 2:
            # this neighbour can connect with either or both children
            self.graph.add_edge(self.index_node + 1, n)
            interface_negative = cell_negative.intersection(cell_neighbour)
            if interface_negative.dim() == 2:
                self.graph.add_edge(self.index_node + 2, n)
        else:
            # this neighbour must otherwise connect with the other child
            self.graph.add_edge(self.index_node + 2, n)

    def construct(self, exhaustive=False, num_workers=0):
        """
        Construct cell complex.

        Two-stage primitive-in-cell predicate.
        (1) bounding boxes of primitive and existing cells are evaluated
        for possible intersection. (2), a strict intersection test is performed.

        Generated cells are stored in self.cells.
        * query the bounding box intersection.
        * optional: intersection test for polygon and edge in each potential cell.
        * partition the potential cell into two. rewind if partition fails.

        Parameters
        ----------
        exhaustive: bool
            Do exhaustive partitioning if set True
        
        num_workers: int
            Number of workers for multi-processing, disabled if set 0
        """
        logger.info('constructing cell complex')
        tik = time.time()

        pool = None
        if num_workers > 0:
            pool = multiprocessing.Pool(processes=num_workers)

        pbar = range(len(self.aabbs)) if self.quiet else trange(len(self.aabbs))
        for i in pbar:  # kinetic for each primitive
            # bounding box intersection test
            # indices of existing cells with potential intersections
            indices_cells = self._intersect_bound_plane(self.aabbs[i], self.planes[i], exhaustive)

            if self.obbs is not None:
                indices_cells = self._intersect_obb(self.planes[i], self.obbs[i], indices_cells, exhaustive)

            assert len(indices_cells), 'intersection failed! check the initial bound'

            # half-spaces defined by inequalities
            # no change_ring() here (instead, QQ() in _inequalities) speeds up 10x
            # init before the loop could possibly speed up a bit
            hspace_positive, hspace_negative = [Polyhedron(ieqs=[inequality]) for inequality in
                                                self._inequalities(self.planes[i])]

            # partition the intersected cells and their bounds while doing mesh slice plane
            indices_parents = []

            for index_cell in indices_cells:
                cell_positive = hspace_positive.intersection(self.cells[index_cell])
                cell_negative = hspace_negative.intersection(self.cells[index_cell])

                if cell_positive.dim() != 3 or cell_negative.dim() != 3:
                    # if cell_positive.is_empty() or cell_negative.is_empty():
                    """
                    cannot use is_empty() predicate for degenerate cases:
                        sage: Polyhedron(vertices=[[0, 1, 2]])
                        A 0-dimensional polyhedron in ZZ^3 defined as the convex hull of 1 vertex
                        sage: Polyhedron(vertices=[[0, 1, 2]]).is_empty()
                        False
                    """
                    continue

                # incrementally build the adjacency graph
                if self.graph is not None:
                    # append the two nodes (UID) being partitioned
                    self.graph.add_node(self.index_node + 1)
                    self.graph.add_node(self.index_node + 2)

                    # append the edge in between
                    self.graph.add_edge(self.index_node + 1, self.index_node + 2)

                    # get neighbours of the current cell from the graph
                    neighbours = self.graph[list(self.graph.nodes)[index_cell]]  # index in the node list

                    if neighbours:
                        # get the neighbouring cells to the parent
                        cells_neighbours = [self.cells[self._index_node_to_cell(n)] for n in neighbours]

                        # adjacency test between both created cells and their neighbours
                        # todo:
                        #   Avoid 3d-3d intersection if possible. Unsliced neighbours connect with only one child;
                        #   sliced neighbors connect with both children.

                        kwargs = []
                        for n, cell in zip(neighbours, cells_neighbours):
                            kwargs.append(
                                {'n': n, 'positive': cell_positive, 'negative': cell_negative, 'neighbour': cell})

                        if pool is None:
                            for k in kwargs:
                                self._intersect_neighbour(k)
                        else:
                            pool.map(self._intersect_neighbour, kwargs)

                    # update cell id
                    self.index_node += 2

                self.cells.append(cell_positive)
                self.cells.append(cell_negative)

                # incrementally cache the bounds for created cells
                self.cells_bounds.append(cell_positive.bounding_box())
                self.cells_bounds.append(cell_negative.bounding_box())

                indices_parents.append(index_cell)

            # delete the parent cells and their bounds. this does not affect the appended ones
            for index_parent in sorted(indices_parents, reverse=True):
                del self.cells[index_parent]
                del self.cells_bounds[index_parent]

                # remove the parent node (and subsequently its incident edges) in the graph
                if self.graph is not None:
                    self.graph.remove_node(list(self.graph.nodes)[index_parent])

        self.constructed = True
        logger.info('cell complex constructed: {:.2f} s'.format(time.time() - tik))

    def visualise(self, indices_cells=None, temp_dir='./'):
        """
        Visualise the cells using trimesh.

        pyglet installation is needed for the visualisation.

        Parameters
        ----------
        indices_cells: None or (n,) int
            Indices of cells to be visualised
        temp_dir: str or Path
            Temp dir to save intermediate visualisation
        """
        if self.constructed:
            try:
                import pyglet
            except ImportError:
                logger.warning('pyglet installation not found; skip visualisation')
                return
            if indices_cells is not None and len(indices_cells) == 0:
                raise ValueError('no indices provided')

            filename_stem = ''.join(choices(string.ascii_uppercase + string.digits, k=5))
            filename_mesh = filename_stem + '.obj'
            filename_mtl = filename_stem + '.mtl'

            self.save_obj(filepath=temp_dir + filename_mesh, indices_cells=indices_cells, use_mtl=True)
            scene = trimesh.load_mesh(temp_dir + filename_mesh)
            scene.show()
            os.remove(temp_dir + filename_mesh)
            os.remove(temp_dir + filename_mtl)
        else:
            raise RuntimeError('cell complex has not been constructed')

    @property
    def num_cells(self):
        """
        Number of cells in the complex.
        """
        return len(self.cells)

    @property
    def num_planes(self):
        """
        Number of planes in the complex, excluding the initial bounding box.
        """
        return len(self.planes)

    def volumes(self, multiplier=1.0, engine='Qhull'):
        """
        list of cell volumes.

        Parameters
        ----------
        multiplier: float
            Multiplier to the volume
        engine: str
            Engine to compute volumes, can be 'Qhull' or 'Sage' with native SageMath

        Returns
        -------
        as_float: list of float
            Volumes of cells
        """
        if engine == 'Qhull':
            from scipy.spatial import ConvexHull
            volumes = [None for _ in range(len(self.cells))]
            for i, cell in enumerate(self.cells):
                try:
                    volumes[i] = ConvexHull(cell.vertices_list()).volume * multiplier
                except:
                    # degenerate floating-point
                    volumes[i] = RR(cell.volume()) * multiplier
            return volumes

        elif engine == 'Sage':
            return [RR(cell.volume()) * multiplier for cell in self.cells]

        else:
            raise ValueError('engine must be either "Qhull" or "Sage"')

    def cell_representatives(self, location='center', num=1):
        """
        Return representatives of cells in the complex.

        Parameters
        ----------
        location: str
            'center' represents the average of the vertices of the polyhedron,
            'centroid' represents the center of mass/volume,
            'random_r' represents random point(s) by rejection,
            'random_t' represents random point(s) by tetrahedralization,
            'skeleton' represents skeleton point(s),
            'boundary' represents boundary point(s) by triangulation
        num: int
            number of samples per cell, only applies to 'random' and 'star'

        Returns
        -------
        as_float: (n, 3) float for 'center' and 'centroid', or (m, n, 3) for 'random' and 'skeleton' and 'boundary'
            Representatives of cells in the complex.
        """
        points = []

        if location == 'center':
            return [cell.center() for cell in self.cells]

        elif location == 'centroid':
            return [cell.centroid() for cell in self.cells]

        elif location == 'random_r':
            # strict random by sampling and rejection
            for cell in self.cells:
                bbox = cell.bounding_box()
                points_cell = []
                while len(points_cell) < num:
                    sample = (uniform(bbox[0][0], bbox[1][0]), uniform(bbox[0][1], bbox[1][1]),
                              uniform(bbox[0][2], bbox[1][2]))
                    if cell.contains(sample):
                        points_cell.append(sample)
                points.append(points_cell)
            return points

        elif location == 'random_t':
            # strict random by triangulation and sampling
            def tetrahedron_volume(a, b, c, d, epsilon=10e-6):
                ab = b - a
                ac = c - a
                ad = d - a
                v = abs(np.dot(ab, np.cross(ac, ad))) / 6.0 + epsilon  # epsilon here to prevent empty volume
                return v

            for cell in self.cells:
                points_cell = []

                # triangulate cell
                vertices = cell.vertices()
                point_config = PointConfiguration(vertices)
                triangulation = point_config.triangulate()
                vertices = np.array([[vertices[v] for v in t] for t in triangulation])
                volumes = [tetrahedron_volume(*tetra) for tetra in vertices]
                options = list(range(len(volumes)))

                for _ in range(num):
                    # select one tetrahedron with volume-based probability
                    choice = choices(options, volumes)[0]

                    # compute vertex-based probability
                    u, v, w = sorted([random() for _ in range(3)])
                    u, v, w = u, v - u, w - v

                    # randomly sample one point within the tetrahedron
                    v1, v2, v3, v4 = vertices[choice]
                    point = u * v1 + v * v2 + w * v3 + (1 - u - v - w) * v4
                    points_cell.append(point)
                points.append(points_cell)
            return points

        elif location == 'skeleton':
            # star-shaped sampling
            for cell in self.cells:
                vertices = cell.vertices_list()
                if num <= len(vertices):
                    # vertices given high priority
                    points.append(choices(vertices, k=num))
                else:
                    num_per_vertex = num // len(vertices)
                    num_remainder = num % len(vertices)
                    centroid = cell.centroid()
                    points_cell = []
                    for vertex in vertices[:-1]:
                        points_cell.extend([vertex + (centroid - np.array(vertex)) / num_per_vertex * i
                                            for i in range(num_per_vertex)])
                    # last vertex consumes remainder points
                    points_cell.extend(
                        [vertices[-1] + (centroid - np.array(vertices[-1])) / (num_remainder + num_per_vertex)
                         * i for i in range(num_remainder + num_per_vertex)])
                    points.append(points_cell)
            return points

        elif location == 'boundary':
            # boundary sampling
            def triangle_area(a, b, c, epsilon=10e-6):
                ab = b - a
                ac = c - a
                cross_prod = np.cross(ab, ac)
                area = 0.5 * np.linalg.norm(cross_prod) + epsilon
                return area

            for cell in self.cells:
                points_cell = []

                # get the list of facets
                facets = cell.facets()

                # aggregate all triangles
                triangles = []
                for f in facets:
                    vertices = f.vertices()
                    point_config = PointConfiguration(vertices)
                    triangulation = point_config.triangulate()  # degrade to 2D triangulation
                    vertices = np.array([[vertices[v] for v in t] for t in triangulation])
                    triangles.extend(vertices)

                # assign triangle-wise probabilities proportional to their areas
                areas = [triangle_area(*triangle) for triangle in triangles]
                options = list(range(len(areas)))

                for _ in range(num):
                    # select one triangle with area-based probability
                    choice = choices(options, areas)[0]

                    # compute vertex-based probability
                    u, v = sorted([random() for _ in range(2)])
                    u, v = u, v - u

                    # randomly sample one point within the triangle
                    v1, v2, v3 = triangles[choice]
                    point = u * v1 + v * v2 + (1 - u - v) * v3
                    points_cell.append(point)
                points.append(points_cell)
            return points

        else:
            raise ValueError(
                "expected 'center', 'centroid', 'random_r', 'random_t', 'skeleton' or 'boundary' as mode, got {}".format(
                    location))

    def cells_boundary(self, epsilon=1e-5):
        """
        Return indices of boundary cells (touching the bounds).
        These cells could be excluded from a valid reconstruction, only when a significant padding exists.

        Parameters
        ----------
        epsilon: float
            Distance tolerance

        Returns
        -------
        as_int: (n, ) int
            Indices of boundary cells
        """
        initial_bound = np.array(self.initial_bound)
        indices = []
        for index, cell_bound in enumerate(self.cells_bounds):
            if (cell_bound[0] < np.array(initial_bound[0]) + epsilon).any() \
                    or (cell_bound[1] > initial_bound[1] - epsilon).any():
                indices.append(index)
        return indices

    def cells_in_mesh(self, filepath_mesh, engine='distance'):
        """
        Return indices of cells that are inside a reference mesh.

        Parameters
        ----------
        filepath_mesh: str or Path
            Filepath to reference mesh
        engine: str
            Engine to compute predicate, can be 'embree' for pyembree (up to x50 faster but not stable),
            'ray_triangle' for a slower ray tracer, or 'distance' for signed distance

        Returns
        -------
        as_int: (n, ) int
            Indices of cells being inside the reference mesh
        """
        mesh = trimesh.load_mesh(filepath_mesh)
        centers = self.cell_representatives(location='center')

        if engine == 'embree':
            # not stable, and no exact returns, use with caution
            # https://trimsh.org/trimesh.ray.ray_pyembree.html
            # https://github.com/mikedh/trimesh/issues/242
            # https://github.com/mikedh/trimesh/issues/306
            logger.warning('ray tracing unstable; use with caution')
            contains = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh).contains_points(centers)
            return contains.nonzero()[0]
        
        elif engine == 'ray_triangle':
            # not stable, use with caution
            # https://trimsh.org/trimesh.ray.ray_triangle.html
            # https://github.com/mikedh/trimesh/issues/242
            logger.warning('ray tracing unstable; use with caution')
            contains = mesh.contains(centers)
            return contains.nonzero()[0]

        elif engine == 'distance':
            # https://trimsh.org/trimesh.proximity.html
            distances = trimesh.proximity.signed_distance(mesh, centers)
            return (distances >= 0).nonzero()[0]
        else:
            raise ValueError("expected 'embree', 'ray_triangle' or 'distance' as engine, got {}".format(engine))

    def print_info(self):
        """
        Print info to console.
        """
        logger.info('number of planes: {}'.format(self.num_planes))
        logger.info('number of cells: {}'.format(self.num_cells))

    def save(self, filepath):
        """
        Save the cell complex to a CC file.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save CC file, '.cc' suffix recommended
        """
        if self.constructed:
            # create the dir if not exists
            with open(filepath, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        else:
            raise RuntimeError('cell complex has not been constructed')

    def save_npy(self, filepath):
        """
        Save the cells to an npy file (deprecated).

        Parameters
        ----------
        filepath: str or Path
            Filepath to save npy file
        """
        if self.constructed:
            # create the dir if not exists
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            np.save(filepath, self.cells, allow_pickle=True)
        else:
            raise RuntimeError('cell complex has not been constructed')

    @staticmethod
    def _obj_str(cells, use_mtl=False, filename_mtl='colours.mtl'):
        """
        Convert a list of cells into a string of obj format.

        Parameters
        ----------
        cells: list of Polyhedra objects
            Polyhedra cells
        use_mtl: bool
            Use mtl attribute in obj if set True
        filename_mtl: None or str
            Material filename

        Returns
        -------
        scene_str: str
            String representation of the object
        material_str: str
            String representation of the material
        """
        scene = None
        for cell in cells:
            scene += cell.render_solid()

        # directly save the obj string from scene.obj() will bring the inverted facets
        scene_obj = scene.obj_repr(scene.default_render_params())
        if len(cells) == 1:
            scene_obj = [scene_obj]
        scene_str = ''
        material_str = ''

        if use_mtl:
            scene_str += f'mtllib {filename_mtl}\n'

        for o in range(len(cells)):
            scene_str += scene_obj[o][0] + '\n'

            if use_mtl:
                scene_str += scene_obj[o][1] + '\n'
                material_str += 'newmtl ' + scene_obj[o][1].split()[1] + '\n'
                material_str += 'Kd {:.3f} {:.3f} {:.3f}\n'.format(random(), random(), random())  # diffuse colour

            scene_str += '\n'.join(scene_obj[o][2]) + '\n'
            scene_str += '\n'.join(scene_obj[o][3]) + '\n'  # contents[o][4] are the interior facets
        return scene_str, material_str

    def save_obj(self, filepath, indices_cells=None, use_mtl=False):
        """
        Save polygon soup of indexed convexes to an obj file.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save obj file
        indices_cells: (n,) int
            Indices of cells to save to file
        use_mtl: bool
            Use mtl attribute in obj if set True
        """
        if self.constructed:
            # create the dir if not exists
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            cells = [self.cells[i] for i in indices_cells] if indices_cells is not None else self.cells
            scene_str, material_str = self._obj_str(cells, use_mtl=use_mtl, filename_mtl=f'{filepath.stem}.mtl')

            with open(filepath, 'w') as f:
                f.writelines(scene_str)
            if use_mtl:
                with open(filepath.with_name(f'{filepath.stem}.mtl'), 'w') as f:
                    f.writelines(material_str)
        else:
            raise RuntimeError('cell complex has not been constructed')

    def save_plm(self, filepath, indices_cells=None):
        """
        Save polygon soup of indexed convexes to a plm file (polyhedron mesh in Mapple).

        Parameters
        ----------
        filepath: str or Path
            Filepath to save plm file
        indices_cells: (n,) int
            Indices of cells to save to file
        """
        if self.constructed:
            # create the dir if not exists
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            num_vertices = 0
            info_vertices = ''
            info_facets = ''
            info_header = ''

            cells = [self.cells[i] for i in indices_cells] if indices_cells is not None else self.cells

            scene = None
            for cell in cells:
                scene += cell.render_solid()
                num_vertices += cell.n_vertices()

            info_header += '#vertices {}\n'.format(num_vertices)
            info_header += '#cells {}\n'.format(len(cells))

            with open(filepath, 'w') as f:
                contents = scene.obj_repr(scene.default_render_params())
                for o in range(len(cells)):
                    info_vertices += '\n'.join([st[2:] for st in contents[o][2]]) + '\n'
                    info_facets += str(len(contents[o][3])) + '\n'
                    for st in contents[o][3]:
                        info_facets += str(len(st[2:].split())) + ' '  # number of vertices on this facet
                        info_facets += ' '.join([str(int(n) - 1) for n in st[2:].split()]) + '\n'
                f.writelines(info_header + info_vertices + info_facets)

        else:
            raise RuntimeError('cell complex has not been constructed')
