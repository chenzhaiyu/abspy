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
primitive.py
----------

Process detected planar primitives.

Primitives are supported in vertex group format (.vg, .bvg).
Mapple as in [Easy3D](https://github.com/LiangliangNan/Easy3D)
can be used to generate such primitives from point clouds.
Otherwise, one can refer to the vertex group file format specification
attached to the README document.
"""

from random import random
from pathlib import Path
from functools import reduce
import struct

import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA

from .logger import attach_to_log

logger = attach_to_log()

try:
    import open3d as o3d
except ModuleNotFoundError:
    logger.warning('Open3D import failed')
    o3d = None


class VertexGroup:
    """
    Class for manipulating planar primitives.
    """

    def __init__(self, filepath, process=True, quiet=False, refit=True, global_group=False):
        """
        Init VertexGroup.
        Class for manipulating planar primitives.

        Parameters
        ----------
        filepath: str or Path
            Filepath to vertex group file (.vg) or binary vertex group file (.bvg)
        process: bool
            Immediate processing if set True
        quiet: bool
            Disable logging if set True
        refit: bool
            Refit plane parameters if set True
        global_group: bool
            Remove the first group as an unnecessary global one containing all subgroups if set True
        """
        if quiet:
            logger.disabled = True

        if isinstance(filepath, str):
            self.filepath = Path(filepath)
        else:
            self.filepath = filepath
        self.processed = False
        self.points = None
        self.planes = None
        self.aabbs = None
        self.obbs = None
        self.points_grouped = None
        self.points_ungrouped = None

        self.vgroup_ascii = self.load_file()
        self.vgroup_binary = None

        self.refit = refit
        self.global_group = global_group
        self.normals = None

        if process:
            self.process()

    def load_file(self):
        """
        Load (ascii / binary) vertex group file.
        """
        if self.filepath.suffix == '.vg':
            with open(self.filepath, 'r') as fin:
                return fin.readlines()

        elif self.filepath.suffix == '.bvg':
            # define size constants
            _SIZE_OF_INT = 4
            _SIZE_OF_FLOAT = 4
            _SIZE_OF_PARAM = 4
            _SIZE_OF_COLOR = 3

            vgroup_ascii = ''
            with open(self.filepath, 'rb') as fin:
                # points
                num_points = struct.unpack('i', fin.read(_SIZE_OF_INT))[0]
                points = struct.unpack('f' * num_points * 3, fin.read(_SIZE_OF_FLOAT * num_points * 3))
                vgroup_ascii += f'num_points: {num_points}\n'
                vgroup_ascii += ' '.join(map(str, points)) + '\n'

                # colors
                num_colors = struct.unpack("i", fin.read(_SIZE_OF_INT))[0]
                vgroup_ascii += f'num_colors: {num_colors}\n'

                # normals
                num_normals = struct.unpack("i", fin.read(_SIZE_OF_INT))[0]
                normals = struct.unpack('f' * num_normals * 3, fin.read(_SIZE_OF_FLOAT * num_normals * 3))
                vgroup_ascii += f'num_normals: {num_normals}\n'
                vgroup_ascii += ' '.join(map(str, normals)) + '\n'

                # groups
                num_groups = struct.unpack("i", fin.read(_SIZE_OF_INT))[0]
                vgroup_ascii += f'num_groups: {num_groups}\n'

                group_counter = 0
                while group_counter < num_groups:
                    group_type = struct.unpack("i", fin.read(_SIZE_OF_INT))[0]
                    num_group_parameters = struct.unpack("i", fin.read(_SIZE_OF_INT))[0]
                    group_parameters = struct.unpack("f" * _SIZE_OF_PARAM, fin.read(_SIZE_OF_INT * _SIZE_OF_PARAM))
                    group_label_size = struct.unpack("i", fin.read(_SIZE_OF_INT))[0]
                    # be reminded that vg <-> bvg in Mapple does not maintain group order
                    group_label = struct.unpack("c" * group_label_size, fin.read(group_label_size))
                    group_color = struct.unpack("f" * _SIZE_OF_COLOR, fin.read(_SIZE_OF_FLOAT * _SIZE_OF_COLOR))
                    group_num_point = struct.unpack("i", fin.read(_SIZE_OF_INT))[0]
                    group_points = struct.unpack("i" * group_num_point, fin.read(_SIZE_OF_INT * group_num_point))
                    num_children = struct.unpack("i", fin.read(_SIZE_OF_INT))[0]

                    vgroup_ascii += f'group_type: {group_type}\n'
                    vgroup_ascii += f'num_group_parameters: {num_group_parameters}\n'
                    vgroup_ascii += 'group_parameters: ' + ' '.join(map(str, group_parameters)) + '\n'
                    vgroup_ascii += 'group_label: ' + ''.join(map(str, group_label)) + '\n'
                    vgroup_ascii += 'group_color: ' + ' '.join(map(str, group_color)) + '\n'
                    vgroup_ascii += f'group_num_point: {group_num_point}\n'
                    vgroup_ascii += ' '.join(map(str, group_points)) + '\n'
                    vgroup_ascii += f'num_children: {num_children}\n'

                    group_counter += 1

                # convert vgroup_ascii to list
                return vgroup_ascii.split('\n')

        else:
            raise ValueError(f'unable to load {self.filepath}, expected *.vg or .bvg.')

    def process(self):
        """
        Start processing vertex group.
        """
        logger.info('processing {}'.format(self.filepath))
        self.points = self.get_points()
        self.planes, self.aabbs, self.obbs, self.points_grouped, self.points_ungrouped = self.get_primitives()
        self.processed = True

    def get_points(self, row=1):
        """
        Get points from vertex group.

        Parameters
        ----------
        row: int
            Row number where points are specified, defaults to 1 for filename.vg

        Returns
        ----------
        as_float: (n, 3) float
            Point cloud
        """
        pc_lines = []

        for line in self.vgroup_ascii[row:]:
            # stop reading when the 'num_colors' keyword is found
            if 'num_colors' in line:
                break
            pc_lines.append(line)

        pc = np.fromstring(' '.join(pc_lines), sep=' ')
        return np.reshape(pc, (-1, 3))

    def get_primitives(self):
        """
        Get primitives from vertex group.

        Returns
        ----------
        params: (n, 4) float
            Plane parameters
        aabbs: (n, 2, 3) float
            Axis-aligned bounding boxes of the primitives
        obbs: (n, 4, 3) float
            Oriented bounding boxes of the primitives
        groups: (n, m, 3) float
            Groups of points
        ungrouped_points: (u, 3) float
            Points that belong to no group
        """
        is_primitive = [line.startswith('group_num_point') for line in self.vgroup_ascii]
        is_parameter = [line.startswith('group_parameters') for line in self.vgroup_ascii]

        primitives = [self.vgroup_ascii[line] for line in np.where(is_primitive)[0] + 1]  # lines of groups in the file
        parameters = [self.vgroup_ascii[line] for line in np.where(is_parameter)[0]]

        # remove global group if there is one
        if self.global_group:
            primitives = primitives[1:]
            parameters = parameters[1:]

        if len(primitives) != len(parameters):
            raise ValueError('group attributes mismatch')

        params = []
        aabbs = []
        obbs = []
        groups = []
        grouped_indices = set()  # indices of points being grouped
        for i, p in enumerate(primitives):
            point_indices = np.fromstring(p, sep=' ', dtype=np.int64)
            grouped_indices.update(point_indices)
            points = self.points[point_indices]

            if len(point_indices) == 0:
                # empty group -> global bounds and no refit
                if self.refit:
                    logger.warning('refit skipped for empty group')
                param = np.array([float(j) for j in parameters[i][18:-1].split()])
                aabb = self._points_bound(self.points)
                obb = aabb

            else:
                # valid group -> local bounds and optional refit
                if self.refit:
                    param, obb = self.fit_plane(points, mode='PCA')
                else:
                    param = np.array([float(j) for j in parameters[i][18:-1].split()])
                    _, obb = self.fit_plane(points, mode='PCA')
                aabb = self._points_bound(points)

            if param is None or len(param) != 4:
                logger.warning(f'bad parameter skipped: {param}')
                continue

            params.append(param)
            aabbs.append(aabb)
            obbs.append(obb)
            groups.append(points)

        ungrouped_indices = set(range(len(self.points))).difference(grouped_indices)
        ungrouped_points = self.points[list(ungrouped_indices)]  # points that belong to no groups
        return (np.array(params), np.array(aabbs), np.array(obbs), np.array(groups, dtype=object),
                np.array(ungrouped_points))

    @staticmethod
    def _points_bound(points):
        """
        Get bounds (AABB) of the points.

        Parameters
        ----------
        points: (n, 3) float
            Points
        Returns
        ----------
        as_float: (2, 3) float
            Bounds (AABB) of the points
        """
        return np.array([np.amin(points, axis=0), np.amax(points, axis=0)])

    def normalise_from_centroid_and_scale(self, centroid, scale, num=None):
        """
        Normalising points.

        Centroid and scale are provided to be mitigated, which are identical with the return of
        scale_and_offset() such that the normalised points align with the corresponding mesh.
        Notice the difference with normalise_points_to_centroid_and_scale().

        Parameters
        ----------
        centroid: (3,) float
            Centroid of the points to be mitigated
        scale: float
            Scale of the points to be mitigated
        num: None or int
            If specified, random sampling is performed to ensure the identical number of points

        Returns
        ----------
        None: NoneType
            Normalised (and possibly sampled) self.points
        """
        # mesh_to_sdf.utils.scale_to_unit_sphere()
        self.points = (self.points - centroid) / scale

        # update planes and bounds as point coordinates has changed
        self.planes, self.aabbs, self.obbs, self.points_grouped, self.points_ungrouped = self.get_primitives()

        # safely sample points after planes are extracted
        if num:
            choice = np.random.choice(self.points.shape[0], num, replace=True)
            self.points = self.points[choice, :]

    def normalise_to_centroid_and_scale(self, centroid=(0, 0, 0), scale=1.0, num=None):
        """
        Normalising points to the provided centroid and scale. Notice
        the difference with normalise_points_from_centroid_and_scale().

        Parameters
        ----------
        centroid: (3,) float
            Desired centroid of the points
        scale: float
            Desired scale of the points
        num: None or int
            If specified, random sampling is performed to ensure the identical number of points

        Returns
        ----------
        None: NoneType
            Normalised (and possibly sampled) self.points
        """
        ######################################################
        # this does not lock the scale
        # offset = np.mean(points, axis=0)
        # denominator = np.max(np.ptp(points, axis=0)) / scale
        ######################################################
        bounds = np.ptp(self.points, axis=0)
        center = np.min(self.points, axis=0) + bounds / 2
        offset = center
        self.points = (self.points - offset) / (bounds.max() * scale) + centroid

        # update planes and bounds as point coordinates has changed
        self.planes, self.aabbs, self.obbs, self.points_grouped, self.points_ungrouped = self.get_primitives()

        # safely sample points after planes are extracted
        if num:
            choice = np.random.choice(self.points.shape[0], num, replace=True)
            self.points = self.points[choice, :]

    @staticmethod
    def fit_plane(points, mode='PCA'):
        """
        Fit plane parameters for a point set.

        Parameters
        ----------
        points: (n, 3) float
            Points to be fit
        mode: str
            Mode of plane fitting,
            'PCA' (recommended) or 'LSA' (may introduce distortions)

        Returns
        ----------
        param: (4,) float
            Plane parameters, (a, b, c, d) as in a * x + b * y + c * z = -d
        obb: (4,3) float
            Oriented bounding box of the plane
        """
        assert mode == 'PCA' or mode == 'LSA'

        if len(points) < 3:
            logger.warning('plane fitting skipped given #points={}'.format(len(points)))
            return None

        if mode == 'LSA':
            # AX = B
            logger.warning('LSA introduces distortions when the plane crosses the origin')
            param = np.linalg.lstsq(points, np.expand_dims(np.ones(len(points)), 1), rcond=None)
            param = np.append(param[0], -1)
            obb = None

        else:
            # PCA followed by shift
            pca = PCA(n_components=3)
            pca.fit(points)
            eig_vec = pca.components_
            points_trans = pca.transform(points)
            point_min = np.amin(points_trans, axis=0)
            point_max = np.amax(points_trans, axis=0)
            obb = np.array([[point_min[0], point_min[1], 0], [point_min[0], point_max[1], 0],
                            [point_max[0], point_max[1], 0], [point_max[0], point_min[1], 0]])
            obb = pca.inverse_transform(obb)

            logger.debug('explained_variance_ratio: {}'.format(pca.explained_variance_ratio_))

            # normal vector of minimum variance
            normal = eig_vec[2, :]  # (a, b, c) normalized
            centroid = np.mean(points, axis=0)

            # every point (x, y, z) on the plane satisfies a * x + b * y + c * z = -d
            # taking centroid as a point on the plane
            d = -centroid.dot(normal)
            param = np.append(normal, d)

        return param, obb

    def append_planes(self, additional_planes, additional_points=None):
        """
        Append planes to vertex group. The provided planes can be accompanied by optional supporting points.
        Notice these additional planes differ from `additional_planes` in `complex.py`: the former apply to
        the VertexGroup data structure thus is generic to applications, while the latter apply to only the
        CellComplex data structure.

        Parameters
        ----------
        additional_planes: (m, 4) float
            Plane parameters
        additional_points: None or (m, n, 3) float
            Points that support planes
        """
        if additional_points is None:
            # this may still find use cases where plane parameters are provided as-is
            logger.warning('no supporting points provided. only appending plane parameters')
        else:
            assert len(additional_planes) == len(additional_points)
            # direct appending would not work
            combined = np.zeros(len(self.points_grouped) + len(additional_points), dtype=object)
            combined[:len(self.points_grouped)] = self.points_grouped
            combined[len(self.points_grouped):] = [np.array(g) for g in additional_points]
            self.points_grouped = combined
        self.planes = np.append(self.planes, additional_planes, axis=0)

    def save_vg(self, filepath):
        """
        Save vertex group into a vg file.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save vg file
        """
        logger.info('writing vertex group into {}'.format(filepath))

        if isinstance(filepath, str):
            assert filepath.endswith('.vg')
        elif isinstance(filepath, Path):
            assert filepath.suffix == '.vg'
        assert self.planes is not None and self.points_grouped is not None

        points_grouped = np.concatenate(self.points_grouped)
        points_ungrouped = self.points_ungrouped

        # points
        out = ''
        out += 'num_points: {}\n'.format(len(points_grouped) + len(points_ungrouped))
        for i in points_grouped.flatten():
            # https://stackoverflow.com/questions/54367816/numpy-np-fromstring-not-working-as-hoped
            out += '{} '.format(i)
        for i in points_ungrouped.flatten():
            out += '{} '.format(i)

        # colors (no color needed)
        out += '\nnum_colors: {}'.format(0)

        # normals
        out += '\nnum_normals: {}\n'.format(len(points_grouped) + len(points_ungrouped))
        for i, group in enumerate(self.points_grouped):
            for _ in group:
                out += '{} {} {} '.format(*self.planes[i][:3])
        for i in range(len(points_ungrouped)):
            out += '{} {} {} '.format(0, 0, 0)

        # groups
        num_groups = len(self.points_grouped)
        out += '\nnum_groups: {}\n'.format(num_groups)
        j_base = 0
        for i in range(num_groups):
            out += 'group_type: {}\n'.format(0)
            out += 'num_group_parameters: {}\n'.format(4)
            out += 'group_parameters: {} {} {} {}\n'.format(*self.planes[i])
            out += 'group_label: group_{}\n'.format(i)
            out += 'group_color: {} {} {}\n'.format(random(), random(), random())
            out += 'group_num_point: {}\n'.format(len(self.points_grouped[i]))
            for j in range(j_base, j_base + len(self.points_grouped[i])):
                out += '{} '.format(j)
            j_base += len(self.points_grouped[i])
            out += '\nnum_children: {}\n'.format(0)

        # additional groups without supporting points
        num_planes = len(self.planes)  # num_planes >= num_groups
        for i in range(num_groups, num_planes):
            out += 'group_type: {}\n'.format(0)
            out += 'num_group_parameters: {}\n'.format(4)
            out += 'group_parameters: {} {} {} {}\n'.format(*self.planes[i])
            out += 'group_label: group_{}\n'.format(i)
            out += 'group_color: {} {} {}\n'.format(random(), random(), random())
            out += 'group_num_point: {}\n'.format(0)
            out += 'num_children: {}\n'.format(0)

        with open(filepath, 'w') as fout:
            fout.writelines(out)

    def save_bvg(self, filepath):
        """
        Save vertex group into a bvg file.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save vg file
        """
        logger.info('writing vertex group into {}'.format(filepath))

        if isinstance(filepath, str):
            assert filepath.endswith('.bvg')
        elif isinstance(filepath, Path):
            assert filepath.suffix == '.bvg'
        assert self.planes is not None and self.points_grouped is not None

        points_grouped = np.concatenate(self.points_grouped)
        points_ungrouped = self.points_ungrouped

        # points
        out = [struct.pack('i', len(points_grouped) + len(points_ungrouped))]
        for i in points_grouped.flatten():
            # https://stackoverflow.com/questions/54367816/numpy-np-fromstring-not-working-as-hoped
            out.append(struct.pack('f', i))
        for i in points_ungrouped.flatten():
            out.append(struct.pack('f', i))

        # colors (no color needed)
        out.append(struct.pack('i', 0))

        # normals
        out.append(struct.pack('i', len(points_grouped) + len(points_ungrouped)))
        for i, group in enumerate(self.points_grouped):
            for _ in group:
                out.append(struct.pack('fff', *self.planes[i][:3]))
        for i in range(len(points_ungrouped)):
            out.append(struct.pack('fff', 0, 0, 0))

        # groups
        num_groups = len(self.points_grouped)
        out.append(struct.pack('i', num_groups))
        j_base = 0
        for i in range(num_groups):
            out.append(struct.pack('i', 0))
            out.append(struct.pack('i', 4))
            out.append(struct.pack('ffff', *self.planes[i]))
            out.append(struct.pack('i', 6 + len(str(i))))
            out.append(struct.pack(f'{(6 + len(str(i)))}s', bytes('group_{}'.format(i), encoding='ascii')))
            out.append(struct.pack('fff', random(), random(), random()))
            out.append(struct.pack('i', len(self.points_grouped[i])))

            for j in range(j_base, j_base + len(self.points_grouped[i])):
                out.append(struct.pack('i', j))

            j_base += len(self.points_grouped[i])
            out.append(struct.pack('i', 0))

        # additional groups without supporting points
        num_planes = len(self.planes)  # num_planes >= num_groups
        for i in range(num_groups, num_planes):
            out.append(struct.pack('i', 0))
            out.append(struct.pack('i', 4))
            out.append(struct.pack('ffff', *self.planes[i]))
            out.append(struct.pack('i', 6 + len(str(i))))
            out.append(struct.pack(f'{(6 + len(str(i)))}s', bytes('group_{}'.format(i), encoding='ascii')))
            out.append(struct.pack('fff', random(), random(), random()))
            out.append(struct.pack('i', 0))
            out.append(struct.pack('i', 0))

        with open(filepath, 'wb') as fout:
            fout.writelines(out)

    def save_planes_txt(self, filepath):
        """
        Save plane parameters into a txt file.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save txt file
        """
        with open(filepath, 'w') as fout:
            logger.info('writing plane parameters into {}'.format(filepath))
            out = [''.join(str(n) + ' ' for n in line.tolist()) + '\n' for line in self.planes]
            fout.writelines(out)

    def save_planes_npy(self, filepath):
        """
        Save plane params into an npy file.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save npy file
        """
        logger.info('writing plane parameters into {}'.format(filepath))
        np.save(filepath, self.planes)

    def save_aabbs_npy(self, filepath):
        """
        Save plane AABBs into an npy file.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save npy file
        """
        logger.info('writing plane bounds into {}'.format(filepath))
        np.save(filepath, self.aabbs)

    def save_cloud(self, filepath):
        """
        Save point cloud into a common 3D format. Support formats: xyzn, xyzrgb, pts, ply, pcd.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save point cloud file
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        if self.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(self.normals)
        o3d.io.write_point_cloud(str(filepath), pcd)


class VertexGroupReference:
    """
    Class of reference vertex group sampled from meshes.
    """

    def __init__(self, filepath, num_samples=10000, process=True, quiet=False):
        """
        Init VertexGroupReference.
        Class of reference vertex group sampled from meshes.

        Parameters
        ----------
        filepath: str or Path
            Filepath to a mesh
        num_samples: int
            Number of sampled points
        process: bool
            Immediate processing if set True
        quiet: bool
            Disable logging if set True
        """
        if quiet:
            logger.disabled = True
        import trimesh

        self.filepath = filepath
        self.num_samples = num_samples
        self.processed = False
        self.points = None
        self.planes = None
        self.aabbs = None
        self.obbs = None
        self.points_grouped = None

        self.mesh = trimesh.load_mesh(self.filepath)
        self.normals = None

        if process:
            self.process()

    @staticmethod
    def _points_bound(points):
        """
        Get bounds (AABB) of the points.

        Parameters
        ----------
        points: (n, 3) float
            Points

        Returns
        ----------
        as_float: (2, 3) float
            Bounds (AABB) of the points
        """
        return np.array([np.amin(points, axis=0), np.amax(points, axis=0)])

    def process(self):
        """
        Start processing mesh data.
        """
        logger.info('processing {}'.format(self.filepath))

        # sample on all faces
        samples, face_indices = self.mesh.sample(count=self.num_samples, return_index=True)  # face_indices match facets

        planes = []
        aabbs = []
        obbs = []
        groups = []

        for facet in self.mesh.facets:  # a list of face indices for coplanar adjacent faces
            # group corresponding samples by facet
            points = []
            for face_index in facet:
                sample_indices = np.where(face_indices == face_index)[0]
                if len(sample_indices) > 0:
                    points.append(samples[sample_indices])

            # vertices
            vertices = reduce(np.union1d, self.mesh.faces[facet])  # indices of vertices
            vertices = self.mesh.vertices[vertices]  # coordinates of vertices

            # append vertices in case there is no sampled points in this group
            points.append(vertices)
            points = np.concatenate(points)

            # calculate parameters
            plane, obb = VertexGroup.fit_plane(vertices)

            planes.append(plane)
            aabbs.append(self._points_bound(vertices))
            obbs.append(obb)
            groups.append(points)

        # self.mesh.facets do not cover all faces
        faces_extracted = np.concatenate(self.mesh.facets)
        faces_remainder = np.setdiff1d(np.arange(len(self.mesh.faces)), faces_extracted)

        for face_index in faces_remainder:
            # group corresponding samples by faces
            points = []
            sample_indices = np.where(face_indices == face_index)[0]
            if len(sample_indices) > 0:
                points.append(samples[sample_indices])

            # vertices
            vertices = self.mesh.faces[face_index]
            vertices = self.mesh.vertices[vertices]

            # append vertices in case there is no sampled points in this group
            points.append(vertices)
            points = np.concatenate(points)

            # calculate parameters
            plane, obb = VertexGroup.fit_plane(vertices)
            planes.append(plane)
            aabbs.append(self._points_bound(vertices))
            obbs.append(obb)
            groups.append(points)

        self.points = np.concatenate(groups)
        self.planes = np.array(planes)
        self.aabbs = np.array(aabbs)
        self.obbs = np.array(obbs)
        self.points_grouped = np.array(groups, dtype=object)
        self.processed = True

    @property
    def bottom_indices(self, epsilon=0.01):
        """
        Group indices of bottom facets.

        Parameters
        ----------
        epsilon: (1,) float
            Tolerance for horizontality and minimum Z predicates

        Returns
        ----------
        as_int: (n,) int
            Indices of bottom groups
        """
        is_horizontal = np.logical_and(np.abs(self.planes[:, 0]) < epsilon, np.abs(self.planes[:, 1]) < epsilon)
        horizontal_indices = np.where(is_horizontal)[0]

        z = -self.planes[is_horizontal][:, 2] * self.planes[is_horizontal][:, 3]
        is_bottom = z < (np.min(z) + epsilon)

        return horizontal_indices[is_bottom]

    @property
    def wall_indices(self, epsilon=0.01):
        """
        Group indices of wall facets.

        Parameters
        ----------
        epsilon: (1,) float
            Tolerance for verticality predicate

        Returns
        ----------
        as_int: (n,) int
            Indices of wall groups
        """
        wall_indices = []
        is_vertical = np.abs(self.planes[:, 2]) < epsilon
        vertical_indices = np.where(is_vertical)[0]

        num_facets = len(self.mesh.facets)
        faces_remainder = np.setdiff1d(np.arange(len(self.mesh.faces)), np.concatenate(self.mesh.facets))

        # extract bottom vertices
        vertices_bottom = set()  # indices of bottom vertices
        for b in self.bottom_indices:
            if b < num_facets:
                faces = self.mesh.faces[self.mesh.facets[b]]
                vertices_bottom.update(reduce(np.union1d, faces))
            else:
                vertices_bottom.update(self.mesh.faces[faces_remainder[b - num_facets]])

        # extract vertical indices
        for v in vertical_indices:
            if v < num_facets:
                faces = self.mesh.faces[self.mesh.facets[v]]
                vertices_vertical = reduce(np.union1d, faces)  # indices of vertices
            else:
                vertices_vertical = self.mesh.faces[faces_remainder[v - num_facets]]  # indices of vertices

            if set(vertices_vertical).intersection(vertices_bottom):
                wall_indices.append(v)
        return np.array(wall_indices)

    def perturb(self, sigma):
        """
        Perturb plane normals with Gaussian noise.

        Parameters
        ----------
        sigma: (1,) float
            Gaussian noise standard deviation
        """
        # length of noise vector
        l = np.abs(np.random.normal(loc=0, scale=sigma, size=self.planes.shape[0]))

        # normal noise
        na = np.random.normal(loc=0, scale=1, size=self.planes.shape[0])
        nb = np.random.normal(loc=0, scale=1, size=self.planes.shape[0])
        nc = np.random.normal(loc=0, scale=1, size=self.planes.shape[0])

        # normalized noise vector
        n = np.array([na, nb, nc]).T / np.sqrt(na ** 2 + nb ** 2 + nc ** 2)[:, np.newaxis] * l[:, np.newaxis]

        # apply noise
        self.planes[:, :3] += n

        # re-normalize normal
        norms = np.linalg.norm(self.planes[:, :3], axis=1)
        self.planes[:, :3] /= norms[:, np.newaxis]

    def inject_points(self, points, threshold=0.05, overwrite=True, keep_bottom=False, keep_wall=False,
                      compute_normal=False, pseudo_normal=False, pseudo_size=1):
        """
        Inject unordered points to vertex groups.

        Parameters
        ----------
        points: (n, 3) float
            Point cloud
        threshold: (1,) float
            Distance threshold
        overwrite: bool
            Overwrite sampled points with input points if set True, otherwise append
        keep_bottom: bool
            Keep sampled points on bottom if set True, effective only when overwrite is True
        keep_wall: bool
            Keep sampled points on walls if set True, effective only when overwrite is True
        compute_normal: bool
            Compute normal if set True, otherwise inherit normal from plane parameters
        pseudo_normal: bool
            Take pseudo points into account for normal estimation if set True, otherwise random normal for them,
            not implemented
        pseudo_size: (1,) int
            Minimal group size for pseudo group without injected points,
            only apply to dangling groups without injected points, effective only when overwrite is True
        """
        assert self.points is not None

        # build KD-tree
        kdtree = KDTree(self.points)

        # query the KD-tree to find nearest neighbors and minimal distances
        min_distances, min_reference_indices = kdtree.query(points, k=1)
        min_distances = min_distances.flatten()
        min_reference_indices = min_reference_indices.flatten()

        # determine the group for each minimal reference point index
        group_boundaries = np.cumsum([0] + [chunk.shape[0] for chunk in self.points_grouped])
        min_reference_groups = np.digitize(min_reference_indices, group_boundaries) - 1
        min_reference_groups[min_distances > threshold] = -1

        # append points to groups (append, overwrite, keep all, or keep minimal)
        pseudo_groups = []
        for i in range(len(self.points_grouped)):
            group_mask = min_reference_groups == i
            if not overwrite:
                self.points_grouped[i] = np.concatenate([self.points_grouped[i], points[group_mask]], axis=0)
            elif (keep_bottom and i in self.bottom_indices) or (keep_wall and i in self.wall_indices):
                pass
            elif any(group_mask):
                self.points_grouped[i] = points[group_mask]
            else:
                self.points_grouped[i] = self.points_grouped[i][:pseudo_size]
                pseudo_groups.append(i)

        # update points and optionally their normals
        self.points = np.concatenate(self.points_grouped)
        self.points = np.concatenate([self.points, points[min_reference_groups == -1]])
        if compute_normal:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=4, max_nn=300))
            pcd.orient_normals_consistent_tangent_plane(k=15)
            self.normals = np.asarray(pcd.normals)

    def save_vg(self, filepath):
        """
        Save primitives into a vg file.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save vg file
        """
        if isinstance(filepath, str):
            assert filepath.endswith('.vg')
        elif isinstance(filepath, Path):
            assert filepath.suffix == '.vg'
        assert self.planes is not None and self.points_grouped is not None

        # points
        out = ''
        out += 'num_points: {}\n'.format(len(self.points))
        for i in self.points.flatten():
            # https://stackoverflow.com/questions/54367816/numpy-np-fromstring-not-working-as-hoped
            out += '{} '.format(i)

        # colors (no color needed)
        out += '\nnum_colors: {}'.format(0)

        # normals
        out += '\nnum_normals: {}\n'.format(len(self.points))
        if self.normals is None:
            for i, group in enumerate(self.points_grouped):
                for _ in group:
                    out += '{} {} {} '.format(*self.planes[i][:3])
            num_remainder_points = len(self.points) - sum(len(g) for g in self.points_grouped)
            for _ in range(num_remainder_points):
                out += '{} {} {} '.format(random(), random(), random())
        else:
            for n in self.normals:
                out += '{} {} {} '.format(*n)

        # groups
        out += '\nnum_groups: {}\n'.format(len(self.points_grouped))
        j_base = 0
        for i, group in enumerate(self.points_grouped):
            out += 'group_type: {}\n'.format(0)
            out += 'num_group_parameters: {}\n'.format(4)
            out += 'group_parameters: {} {} {} {}\n'.format(*self.planes[i])
            out += 'group_label: group_{}\n'.format(i)
            out += 'group_color: {} {} {}\n'.format(random(), random(), random())
            out += 'group_num_point: {}\n'.format(len(self.points_grouped[i]))
            for j in range(j_base, j_base + len(self.points_grouped[i])):
                out += '{} '.format(j)
            j_base += len(self.points_grouped[i])
            out += '\nnum_children: {}\n'.format(0)

        with open(filepath, 'w') as fout:
            fout.writelines(out)

    def save_bvg(self, filepath):
        """
        Save primitives into a bvg file.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save bvg file
        """

        if isinstance(filepath, str):
            assert filepath.endswith('.bvg')
        elif isinstance(filepath, Path):
            assert filepath.suffix == '.bvg'
        assert self.planes is not None and self.points_grouped is not None

        # points
        out = [struct.pack('i', len(self.points))]
        for i in self.points.flatten():
            # https://stackoverflow.com/questions/54367816/numpy-np-fromstring-not-working-as-hoped
            out.append(struct.pack('f', i))

        # colors (no color needed)
        out.append(struct.pack('i', 0))

        # normals
        out.append(struct.pack('i', len(self.points)))
        if self.normals is None:
            for i, group in enumerate(self.points_grouped):
                for _ in group:
                    out.append(struct.pack('fff', *self.planes[i][:3]))
            num_remainder_points = len(self.points) - sum(len(g) for g in self.points_grouped)
            for _ in range(num_remainder_points):
                out.append(struct.pack('fff', random(), random(), random()))
        else:
            for n in self.normals:
                out.append(struct.pack('fff', *n))

        # groups
        out.append(struct.pack('i', len(self.points_grouped)))
        j_base = 0
        for i, group in enumerate(self.points_grouped):
            out.append(struct.pack('i', 0))
            out.append(struct.pack('i', 4))
            out.append(struct.pack('ffff', *self.planes[i]))
            out.append(struct.pack('i', 6 + len(str(i))))
            out.append(struct.pack(f'{(6 + len(str(i)))}s', bytes('group_{}'.format(i), encoding='ascii')))
            out.append(struct.pack('fff', random(), random(), random()))
            out.append(struct.pack('i', len(self.points_grouped[i])))

            for j in range(j_base, j_base + len(self.points_grouped[i])):
                out.append(struct.pack('i', j))

            j_base += len(self.points_grouped[i])
            out.append(struct.pack('i', 0))

        with open(filepath, 'wb') as fout:
            fout.writelines(out)

    def save_cloud(self, filepath):
        """
        Save point cloud into a common 3D format. Support formats: xyzn, xyzrgb, pts, ply, pcd.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save point cloud file
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        if self.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(self.normals)
        o3d.io.write_point_cloud(str(filepath), pcd)
