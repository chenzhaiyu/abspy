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
import struct

import numpy as np
from sklearn.decomposition import PCA

from .logger import attach_to_log

logger = attach_to_log()


class VertexGroup:
    """
    Class for manipulating planar primitives.
    """

    def __init__(self, filepath, process=True):
        """
        Init VertexGroup.
        Class for manipulating planar primitives.

        Parameters
        ----------
        filepath: str or Path
            Filepath to vertex group file (.vg) or binary vertex group file (.bvg)
        process: bool 
            Immediate processing if set True
        """
        if isinstance(filepath, str):
            self.filepath = Path(filepath)
        else:
            self.filepath = filepath
        self.processed = False
        self.points = None
        self.planes = None
        self.bounds = None
        self.points_grouped = None
        self.points_ungrouped = None

        self.vgroup_ascii = self.load_file()
        self.vgroup_binary = None

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
        self.planes, self.bounds, self.points_grouped, self.points_ungrouped = self.get_primitives()
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
        pc = np.fromstring(self.vgroup_ascii[row], sep=' ')
        return np.reshape(pc, (-1, 3))

    def get_primitives(self):
        """
        Get primitives from vertex group.

        Returns
        ----------
        params: (n, 4) float
            Plane parameters
        bounds: (n, 2, 3) float
            Bounding box of the primitives
        groups: (n, m, 3) float
            Groups of points
        ungrouped_points: (u, 3) float
            Points that belong to no group
        """
        is_primitive = [line.startswith('group_num_point') for line in self.vgroup_ascii]
        primitives = [self.vgroup_ascii[line] for line in np.where(is_primitive)[0] + 1]  # lines of groups in the file
        params = []
        bounds = []
        groups = []
        grouped_indices = set()  # indices of points being grouped
        for i, p in enumerate(primitives):
            point_indices = np.fromstring(p, sep=' ').astype(np.int64)
            grouped_indices.update(point_indices)
            points = self.points[point_indices]
            param = self.fit_plane(points, mode='PCA')
            if param is None:
                continue
            params.append(param)
            bounds.append(self._points_bound(points))
            groups.append(points)
        ungrouped_indices = set(range(len(self.points))).difference(grouped_indices)
        ungrouped_points = self.points[list(ungrouped_indices)]  # points that belong to no groups
        return np.array(params), np.array(bounds), np.array(groups, dtype=object), np.array(ungrouped_points)

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
        self.planes, self.bounds, self.points_grouped, _ = self.get_primitives()

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
        self.planes, self.bounds, self.points_grouped, _ = self.get_primitives()

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
        """
        assert mode == 'PCA' or mode == 'LSA'
        if len(points) < 3:
            logger.warning('plane fitting skipped given #points={}'.format(len(points)))
            return None
        if mode == 'LSA':
            # AX = B
            logger.warning('LSA introduces distortions when the plane crosses the origin')
            param = np.linalg.lstsq(points, np.expand_dims(np.ones(len(points)), 1))
            param = np.append(param[0], -1)
        else:
            # PCA followed by shift
            pca = PCA(n_components=3)
            pca.fit(points)
            eig_vec = pca.components_
            logger.debug('explained_variance_ratio: {}'.format(pca.explained_variance_ratio_))

            # normal vector of minimum variance
            normal = eig_vec[2, :]  # (a, b, c)
            centroid = np.mean(points, axis=0)

            # every point (x, y, z) on the plane satisfies a * x + b * y + c * z = -d
            # taking centroid as a point on the plane
            d = -centroid.dot(normal)
            param = np.append(normal, d)

        return param

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
            out += ' ' + str(i)
        for i in points_ungrouped.flatten():
            out += ' ' + str(i)

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
        filepath: str
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
        filepath: str
            Filepath to save npy file
        """
        logger.info('writing plane parameters into {}'.format(filepath))
        np.save(filepath, self.planes)

    def save_bounds_npy(self, filepath):
        """
        Save plane bounds into an npy file.

        Parameters
        ----------
        filepath: str
            Filepath to save npy file
        """
        logger.info('writing plane bounds into {}'.format(filepath))
        np.save(filepath, self.bounds)


class VertexGroupReference:
    """
    Class of reference vertex group sampled from meshes.
    """

    def __init__(self, filepath, num_samples=10000, process=True):
        """
        Init VertexGroupReference.
        Class of reference vertex group sampled from meshes.

        Parameters
        ----------
        filepath: str
            Filepath to a mesh
        num_samples: int
            Number of sampled points
        process: bool
            Immediate processing if set True
        """
        import trimesh

        self.filepath = filepath
        self.num_samples = num_samples
        self.processed = False
        self.points = None
        self.planes = []
        self.bounds = []
        self.points_grouped = []

        self.mesh = trimesh.load_mesh(self.filepath)

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
        from functools import reduce

        # sample on all faces
        samples, face_indices = self.mesh.sample(count=self.num_samples, return_index=True)  # face_indices match facets

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
            plane = VertexGroup.fit_plane(vertices)
            self.planes.append(plane)
            self.bounds.append(self._points_bound(vertices))
            self.points_grouped.append(points)
        self.points = np.concatenate(self.points_grouped)

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
            out += ' ' + str(i)

        # colors (no color needed)
        out += '\nnum_colors: {}'.format(0)

        # normals
        out += '\nnum_normals: {}\n'.format(len(self.points))
        for i, group in enumerate(self.points_grouped):
            for _ in group:
                out += '{} {} {} '.format(*self.planes[i][:3])

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
        for i, group in enumerate(self.points_grouped):
            for _ in group:
                out.append(struct.pack('fff', *self.planes[i][:3]))

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
