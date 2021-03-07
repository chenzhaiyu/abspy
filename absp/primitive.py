"""
Management of planar primitive data from Mapple.
Mapple can be downloaded from https://3d.bk.tudelft.nl/liangliang/software/Mapple.zip
or built from https://github.com/LiangliangNan/Easy3D/tree/master/applications/Mapple.

Prerequisites:
* Vertex group file (.vg/.bvg) is generated by Mapple and saved.

"""

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from .logger import attach_to_log

logger = attach_to_log()


class VertexGroup:

    def __init__(self, filepath, process=True):
        """
        :param filepath: filepath to .vg file. str.
        :param process: optional. Immediate processing if set True.
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
        Start processing the data.
        """
        logger.info('processing {}'.format(self.filepath))
        self.points = self.get_points()
        self.planes, self.bounds, self.points_grouped = self.get_primitives()
        self.processed = True

    def get_points(self, row=1):
        """
        :param row: optional. row number where points are specified. defaults to 1 for .vg.
        :return: points data. N * 3 array.
        """
        pc = np.fromstring(self.vgroup[row], sep=' ')
        return np.reshape(pc, (-1, 3))

    def get_primitives(self):
        """
        :return: plane parameters as N * 4 array and their bounds as N * 2 * 3 array.
        """
        is_primitive = [line.startswith('group_num_point') for line in self.vgroup]
        primitives = [self.vgroup[line] for line in np.where(is_primitive)[0] + 1]  # lines of groups in the file
        params = []
        bounds = []
        groups = []
        for i, p in enumerate(primitives):
            points = self.points[np.fromstring(p, sep=' ').astype(np.int64)]
            param = self.fit_plane(points, mode='PCA')
            params.append(param)
            bounds.append(self._points_bound(points))
            groups.append(points)
        return params, bounds, groups

    @staticmethod
    def _points_bound(points):
        """
        :return: bounds (AABB) of the points. 2 * 3 array.
        """
        return np.array([np.amin(points, axis=0), np.amax(points, axis=0)])

    def normalise_from_centroid_and_scale(self, centroid, scale, num=None):
        """
        Normalising points. Centroid and scale are provided to be mitigated, which are identical with the return of
        scale_and_offset() such that the normalised points align with the corresponding mesh. Notice the difference
        with normalise_points_to_centroid_and_scale().
        :param centroid: centroid of the points to be mitigated
        :param scale: scale of the points to be mitigated
        :param num: if specified, random sampling is performed to ensure the identical number of points
        :return: None. normalised (and possibly sampled) self.points whose centroid should locate at (0, 0, 0) and scale be as provided
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
        Normalising points to the provided centroid and scale. Notice the difference
        with normalise_points_from_centroid_and_scale().
        :param centroid: desired centroid of the points
        :param scale: desired scale of the points
        :param num: if specified, random sampling is performed to ensure the identical number of points
        :return: None. normalised self.points
        """
        ######################################################
        # this does not lock the scale
        # offset = np.mean(points, axis=0)
        # denominator = np.max(np.ptp(points, axis=0)) / scale
        ######################################################
        center = np.min(self.points, axis=0) + np.ptp(self.points, axis=0) / 2
        offset = center
        denominator = 1 / scale
        self.points = (self.points - offset) / denominator + centroid

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
        :param points: ndarray, n * 3 points
        :param mode: 'PCA' (recommended) or 'LSA' (may introduce distortions)
        :return: fitted parameters
        """
        assert mode == 'PCA' or mode == 'LSA'
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

    def save_planes_vg(self, filepath, row_pc=1):
        """
        Save (processed) plane params into a vg file.
        """
        # https://github.com/numpy/numpy/issues/17704
        # self.vgroup[row_pc] = np.array2string(self.points.flatten(), threshold=100000000, separator=' ')
        out = ''
        logger.info('writing vertex group into {}'.format(filepath))
        for i in tqdm(self.points.flatten(), desc='writing vg'):
            # https://stackoverflow.com/questions/54367816/numpy-np-fromstring-not-working-as-hoped
            out += ' ' + str(i)
        self.vgroup[row_pc] = out
        with open(filepath, 'w') as fout:
            fout.writelines(self.vgroup)

    def save_planes_txt(self, filepath):
        """
        Save plane params into a txt file.
        """
        with open(filepath, 'w') as fout:
            logger.info('writing plane parameters into {}'.format(filepath))
            outs = [''.join(str(n) + ' ' for n in line.tolist()) + '\n' for line in self.planes]
            fout.writelines(outs)

    def save_planes_npy(self, filepath):
        """
        Save plane params into an npy file.
        """
        logger.info('writing plane parameters into {}'.format(filepath))
        np.save(filepath, self.planes)

    def save_bounds_npy(self, filepath):
        """
        Save plane bounds into an npy file.
        """
        logger.info('writing plane bounds into {}'.format(filepath))
        np.save(filepath, self.bounds)
