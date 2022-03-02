"""
Benchmarking adaptive binary space partitioning with exhaustive hyperplane arrangement.

"""

import numpy as np
import glob
import time
from pathlib import Path

from abspy import attach_to_log
from abspy import VertexGroup
from abspy import CellComplex

logger = attach_to_log(filepath='benchmark.log')


try:
    from sage.all import *
    logger.info('SageMath installation found')
    sage_installed = True
except ModuleNotFoundError:
    logger.warning('SageMath is not installed; hyperplane arrangements benchmark will not run')
    sage_installed = False


def sage_hyperplane_arrangements(planes):
    """
    Hyperplane arrangement with SageMath.
    The SageMath binaries can be downloaded from https://www.sagemath.org/download.html.
    The installation is documented at https://doc.sagemath.org/html/en/installation/.
    """
    # hyperplane arrangements and bounded region extraction
    logger.info('starting hyperplane arrangements')
    hyperplane_arrangement = HyperplaneArrangements(QQ, ('x', 'y', 'z'))
    arrangements = hyperplane_arrangement([[tuple(plane[:3]), plane[-1]] for plane in planes])
    convexes = arrangements.bounded_regions()
    logger.info('number of cells: {}'.format(len(convexes)))


def pipeline_adaptive_partition(planes, bounds, save_file, filename=None):
    """
    Adaptive binary partition as implemented.
    """
    logger.info('starting adaptive partitioning')
    tik = time.time()
    cell_complex = CellComplex(planes, bounds, build_graph=True)
    cell_complex.prioritise_planes()
    cell_complex.construct()
    cell_complex.print_info()
    logger.info('runtime pipeline_adaptive_partition: {:.2f} s\n'.format(time.time() - tik))

    if save_file and filename and filename.suffix == '.obj':
        cell_complex.save_obj(filepath=Path(filename).with_suffix('.obj'))
    if save_file and filename and filename.suffix == '.plm':
        cell_complex.save_plm(filepath=Path(filename).with_suffix('.plm'))


def pipeline_exhaustive_partition(planes, bounds, save_file, filename=None):
    """
    Exhaustive binary partition as implemented.
    """
    logger.info('starting exhaustive partitioning')
    tik = time.time()
    cell_complex = CellComplex(planes, bounds, build_graph=True)
    cell_complex.prioritise_planes()
    cell_complex.construct(exhaustive=True)
    cell_complex.print_info()
    logger.info('runtime pipeline_exhaustive_partition: {:.2f} s\n'.format(time.time() - tik))

    if save_file and filename and filename.suffix == '.obj':
        cell_complex.save_obj(filepath=Path(filename).with_suffix('.obj'))
    if save_file and filename and filename.suffix == '.plm':
        cell_complex.save_plm(filepath=Path(filename).with_suffix('.plm'))


def run_benchmark(data_dir='./data/*.vg', save_file=False):
    """
    Run benchmark among pipeline_adaptive_partition, pipeline_exhaustive_partition, and sage_hyperplane_arrangement.
    """

    logger.info('---------- start benchmarking ----------')

    for filename in glob.glob(data_dir)[:]:

        # # Fig4f and Fig4i are defected: having vertex groups of 2 points. failing at PCA calculation
        # if 'Fig4f' in filename or 'Fig4i' in filename or not filename.endswith('.vg'):
        #     continue

        vertex_group = VertexGroup(filepath=filename)
        planes, bounds = np.array(vertex_group.planes), np.array(vertex_group.bounds)

        pipeline_adaptive_partition(planes, bounds, save_file, filename=Path(filename).with_suffix('.plm'))
        pipeline_exhaustive_partition(planes, bounds, save_file, filename=Path(filename).with_suffix('.plm'))

        if sage_installed:
            tok = time.time()
            sage_hyperplane_arrangements(planes)
            logger.info('runtime sage_hyperplane_arrangements: {:.2f} s\n'.format(time.time() - tok))


if __name__ == '__main__':
    run_benchmark(data_dir='tests/test_data/*.vg', save_file=False)
