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

import numpy as np

from sklearn.decomposition import PCA
from tqdm import tqdm

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
        filepath: str
            Filepath to vertex group file (.vg)
        process: bool 
            Immediate processing if set True
        """
        self.filepath = filepath
        self.processed = False
        self.points = None
        self.planes = None
        self.bounds = None
        self.points_grouped = None

        with open(filepath, 'r') as fin:
            self.vgroup = fin.readlines()

        if process:
            self.process()

    def process(self):
        """
        Start processing vertex group.
        """
        logger.info('processing {}'.format(self.filepath))
        self.points = self.get_points()
        self.planes, self.bounds, self.points_grouped = self.get_primitives()
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
        pc = np.fromstring(self.vgroup[row], sep=' ')
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
        """
        is_primitive = [line.startswith('group_num_point') for line in self.vgroup]
        primitives = [self.vgroup[line] for line in np.where(is_primitive)[0] + 1]  # lines of groups in the file
        params = []
        bounds = []
        groups = []
        for i, p in enumerate(primitives):
            points = self.points[np.fromstring(p, sep=' ').astype(np.int64)]
            param = self.fit_plane(points, mode='PCA')
            if param is None:
                continue
            params.append(param)
            bounds.append(self._points_bound(points))
            groups.append(points)
        return params, bounds, groups

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
        self.planes, self.bounds, self.points_grouped = self.get_primitives()

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
        self.planes, self.bounds, self.points_grouped = self.get_primitives()

        # safely sample points after planes are extracted
        if num:
            choice = np.random.choice(self.points.shape[0], num, replace=True)
            self.points = self.points[choice, :]

    @staticmethod
    def fit_plane(points, mode='PCA'):
        """
        Fitting plane parameters for a point set.

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

    def save_planes_vg(self, filepath, row=1):
        """
        Save plane parameters into a vg file.

        Parameters
        ----------
        filepath: str
            Filepath to save vg file
        row: int
            Row number where points are specified, defaults to 1 for filename.vg
        """
        # https://github.com/numpy/numpy/issues/17704
        # self.vgroup[row] = np.array2string(self.points.flatten(), threshold=100000000, separator=' ')
        out = ''
        logger.info('writing vertex group into {}'.format(filepath))
        for i in tqdm(self.points.flatten(), desc='writing vg'):
            # https://stackoverflow.com/questions/54367816/numpy-np-fromstring-not-working-as-hoped
            out += ' ' + str(i)
        self.vgroup[row] = out
        with open(filepath, 'w') as fout:
            fout.writelines(self.vgroup)

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

    def __init__(self, filepath, process=True):
        """
        Init VertexGroupReference.
        Class of reference vertex group sampled from meshes.

        Parameters
        ----------
        filepath: str
            Filepath to a mesh
        process: bool
            Immediate processing if set True
        """
        import trimesh

        self.filepath = filepath
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

    def process(self, num=10000):
        """
        Start processing mesh data.

        Parameters
        ----------
        num: int
            Number of points to sample from mesh
        """
        from functools import reduce

        # sample on all faces
        samples, face_indices = self.mesh.sample(count=num, return_index=True)  # face_indices match facets

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

    def save_primitives_vg(self, filepath):
        """
        Save primitives into a vg file.

        Parameters
        ----------
        filepath: str
            Filepath to save vg file
        """
        from random import random
        assert self.planes and self.points_grouped

        # points
        out = ''
        out += 'num_points: {}\n'.format(len(self.points))
        for i in self.points.flatten():
            # https://stackoverflow.com/questions/54367816/numpy-np-fromstring-not-working-as-hoped
            out += ' ' + str(i)

        # normals
        out += '\nnum_normals: {}\n'.format(len(self.points))
        for i, group in enumerate(self.points_grouped):
            # https://stackoverflow.com/questions/54367816/numpy-np-fromstring-not-working-as-hoped
            for _ in group:
                out += '{} {} {} '.format(*self.planes[i][:3])

        # colors (no color needed)
        out += '\nnum_colors: {}\n'.format(0)

        # primitives
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
