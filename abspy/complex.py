"""
complex.py
----------

Cell complex from planar primitive arrangement.

A linear cell complex is constructed from planar primitives
with adaptive binary space partitioning: upon insertion of a primitive
only the local cells that are intersecting it will be updated,
so will be the corresponding adjacency graph of the complex.
"""

from pathlib import Path
import itertools
import heapq
from copy import copy
from random import random, choices
import time

import numpy as np
from tqdm import trange
import networkx as nx
from sage.all import polytopes, QQ, RR, Polyhedron

from .logger import attach_to_log
from .primitive import VertexGroup

logger = attach_to_log()


class CellComplex:
    """
    Class of cell complex from planar primitive arrangement.
    """
    def __init__(self, planes, bounds, points=None, initial_bound=None, initial_padding=0.1, additional_planes=None,
                 build_graph=False):
        """
        Init CellComplex.
        Class of cell complex from planar primitive arrangement.

        Parameters
        ----------
        planes: (n, 4) float
            Plana parameters
        bounds: (n, 2, 3) float
            Corresponding bounding box bounds of the planar primitives
        initial_bound: None or (2, 3) float
            Initial bound to partition
        build_graph: bool
            Build the cell adjacency graph if set True.
        additional_planes: None or (n, 4) float
            Additional planes to append to the complex,
            can be missing planes due to occlusion or incapacity of RANSAC
        """
        self.bounds = bounds  # numpy.array over RDF
        self.planes = planes  # numpy.array over RDF
        self.points = points

        # missing planes due to occlusion or incapacity of RANSAC
        self.additional_planes = additional_planes

        self.initial_bound = initial_bound if initial_bound else self._pad_bound(
            [np.amin(bounds[:, 0, :], axis=0), np.amax(bounds[:, 1, :], axis=0)],
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
        bounds = list(copy(self.bounds))
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
            # the pair with smallest angle
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
                    planes_merged = VertexGroup.fit_plane(points_merged)
                    bounds_merged = [np.min([bounds[pair[1]][0], bounds[pair[2]][0]], axis=0).tolist(),
                                     np.max([bounds[pair[1]][1], bounds[pair[2]][1]], axis=0).tolist()]

                    # update to_merge
                    to_merge.update({pair[1]})
                    to_merge.update({pair[2]})

                    # push the merged plane into the heap
                    for i, p in enumerate(planes):
                        if i not in to_merge:
                            angle_cos = np.abs(np.dot(planes_merged[:3], p[:3]))
                            heapq.heappush(priority_queue, [-angle_cos, i, len(planes)])  # placeholder

                    # update the actual data with the merged ones
                    points.append(points_merged)
                    bounds.append(bounds_merged)
                    planes.append(planes_merged)

            else:
                # no more possible coplanar pairs can exist in this priority queue
                break

        # delete the merged pairs
        for i in sorted(to_merge, reverse=True):
            del points[i]
            del bounds[i]
            del planes[i]

        logger.info('{} pairs of planes merged'.format(len(to_merge)))

        self.planes = np.array(planes)
        self.bounds = np.array(bounds)
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
        self.bounds = self.bounds[indices_priority]

        # append additional planes with highest priority
        if self.additional_planes:
            self.planes = np.concatenate([self.additional_planes, self.planes], axis=0)
            additional_bounds = [[[-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]]] * len(self.additional_planes)
            self.bounds = np.concatenate([additional_bounds, self.bounds], axis=0)  # never miss an intersection

        logger.debug('ordered planes: {}'.format(self.planes))
        logger.debug('ordered bounds: {}'.format(self.bounds))

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
            volume = np.prod(self.bounds[:, 1, :] - self.bounds[:, 0, :], axis=1)
            return np.argsort(volume)[::-1]
        elif mode == 'norm':
            sizes = np.linalg.norm(self.bounds[:, 1, :] - self.bounds[:, 0, :], ord=2, axis=1)
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

    def _bbox_intersect(self, bound, plane, exhaustive=False, padding=None):
        """
        Bounding box intersection test.

        Parameters
        ----------
        bound: (2, 3) float
            Bound of the query planar primitive
        plane: (4,) float
            Plane parameters
        padding: None or float
            Padding for existing cells
        exhaustive: bool
            Exhaustive partitioning, only for benchmarking.

        Returns
        -------
        as_int: (n,) int
            Indices of existing cells whose bounds intersect with bounds of the query primitive
            and intersect with the supporting plane of the primitive
        """

        # todo: alpha-shape/convex hull to reduce unnecessary partitioning?
        # each planar primitive partitions only the 3D cells that intersect with it
        cells_bounds = np.array(self.cells_bounds)  # easier array manipulation
        if padding:
            bound = self._pad_bound(bound, padding=padding)
        center_targets = np.mean(cells_bounds, axis=1)  # N * 3
        extent_targets = cells_bounds[:, 1, :] - cells_bounds[:, 0, :]  # N * 3

        if bound[0][0] == -np.inf:
            intersection_bound = np.arange(len(self.cells_bounds))

        else:
            # intersection with existing cell AABB
            center_query = np.mean(bound, axis=0)  # 3,
            center_distance = np.abs(center_query - center_targets)  # N * 3
            extent_query = bound[1] - bound[0]  # 3,

            # abs(center_distance) * 2 < (query extent + target extent) for every dimension -> intersection
            intersection_bound = np.where(np.all(center_distance * 2 < extent_query + extent_targets, axis=1))[0]

        # plane-AABB intersection test from extracted intersection_bound only
        # https://gdbooks.gitbooks.io/3dcollisions/content/Chapter2/static_aabb_plane.html
        # compute the projection interval radius of AABB onto L(t) = center + t * normal
        radius = np.dot(extent_targets[intersection_bound] / 2, np.abs(plane[:3]))
        # compute distance of box center from plane
        distance = np.dot(center_targets[intersection_bound], plane[:3]) + plane[3]
        # intersection between plane and AABB occurs when distance falls within [-radius, +radius] interval
        intersection_plane = np.where(np.abs(distance) <= radius)[0]

        if exhaustive:
            return np.arange(len(self.cells_bounds))

        return intersection_bound[intersection_plane]

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

    def construct(self, exhaustive=False):
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
        """
        logger.info('constructing cell complex')
        tik = time.time()

        for i in trange(len(self.bounds)):  # kinetic for each primitive
            # bounding box intersection test
            # indices of existing cells with potential intersections
            indices_cells = self._bbox_intersect(self.bounds[i], self.planes[i], exhaustive)
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
                        #   avoid 3d-3d intersection if possible. those unsliced neighbours connect with only one child
                        #   - reduce computation by half - can be further reduced using vertices/faces instead of
                        #   polyhedron intersection. those sliced neighbors connect with both children

                        for n, cell in enumerate(cells_neighbours):

                            interface_positive = cell_positive.intersection(cell)
                            interface_negative = cell_negative.intersection(cell)

                            if interface_positive.dim() == 2:  # strictly a face
                                self.graph.add_edge(self.index_node + 1, list(neighbours)[n])
                            if interface_negative.dim() == 2:
                                self.graph.add_edge(self.index_node + 2, list(neighbours)[n])

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

        Trimesh/pyglet installation are needed for the visualisation.

        Parameters
        ----------
        indices_cells: None or (n,) int
            Indices of cells to be visualised
        temp_dir: str
            Temp dir to save intermediate visualisation
        """
        if self.constructed:
            import os
            import string
            try:
                import trimesh
                import pyglet
            except ImportError:
                logger.warning('trimesh/pyglet installation not found. skip visualisation')
                return
            temp_filename = ''.join(choices(string.ascii_uppercase + string.digits, k=5)) + '.obj'
            self.save_obj(filepath=temp_dir + temp_filename, indices_cells=indices_cells, use_mtl=True)
            scene = trimesh.load_mesh(temp_dir + temp_filename)
            scene.show()
            os.remove(temp_dir + temp_filename)
            os.remove(temp_dir + 'colours.mtl')
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
            Engine to compute volumes, can be 'Qhull' or 'native' with native SageMath

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

        elif engine == 'native':
            return [RR(cell.volume()) * multiplier for cell in self.cells]

        else:
            raise ValueError('engine must be either "Qhull" or "native"')

    def cell_representatives(self, location='center'):
        """
        Return representatives of cells in the complex.

        Parameters
        ----------
        location: str
            'center' represents the average of the vertices of the polyhedron,
            'centroid' represents the center of mass/volume.

        Returns
        -------
        as_float: (n, 3) float
            Representatives of cells in the complex.
        """
        if location == 'center':
            return [cell.center() for cell in self.cells]
        elif location == 'centroid':
            return [cell.centroid() for cell in self.cells]
        else:
            raise ValueError("expected 'mass' or 'centroid' as mode, got {}".format(location))

    def print_info(self):
        """
        Print info to console.
        """
        logger.info('number of planes: {}'.format(self.num_planes))
        logger.info('number of cells: {}'.format(self.num_cells))

    def save_npy(self, filepath):
        """
        Save the cells to an npy file (deprecated).

        Parameters
        ----------
        filepath: str
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
    def _obj_str(cells, use_mtl=False):
        """
        Convert a list of cells into a string of obj format.

        Parameters
        ----------
        cells: list of Polyhedra objects
            Polyhedra cells
        use_mtl: bool
            Use mtl attribute in obj if set True

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
        scene_str = ''
        material_str = ''

        if use_mtl:
            scene_str += 'mtllib colours.mtl\n'

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
        filepath: str
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
            scene_str, material_str = self._obj_str(cells, use_mtl=use_mtl)

            with open(filepath, 'w') as f:
                f.writelines(scene_str)
            if use_mtl:
                with open(filepath.with_name('colours.mtl'), 'w') as f:
                    f.writelines(material_str)
        else:
            raise RuntimeError('cell complex has not been constructed')

    def save_plm(self, filepath, indices_cells=None):
        """
        Save polygon soup of indexed convexes to a plm file (polyhedron mesh in Mapple).

        Parameters
        ----------
        filepath: str
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
