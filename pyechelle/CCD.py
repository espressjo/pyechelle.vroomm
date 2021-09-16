import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import autologging
import h5py
import numpy as np
from numba import int32, float64, int64, njit

logger = logging.getLogger('CCD')


@njit(
    int32[:, :](float64[:], float64[:], int64, int64, int64, int64),
    parallel=True,
    nogil=True,
    cache=True,
)
def bin_2d(x, y, xmin=0, xmax=4096, ymin=0, ymax=4096):
    """  Bin XY position into 2d grid and throw away data outside the limits.

    Args:
        x (np.ndarray): X positions
        y (np.ndarray): Y positions
        xmin (int): minimum X value of the grid
        xmax (int): maximum X value of the grid
        ymin (int): minimum Y value of the grid
        ymax (int): maximum Y value of the grid

    Returns:
        np.ndarray: binned XY positions
    """
    grid = np.zeros((ymax - ymin, xmax - xmin), dtype=np.int32)

    valid_idx = np.logical_and(x >= xmin, y >= ymin)
    valid_idx = np.logical_and(valid_idx, y < ymax)
    valid_idx = np.logical_and(valid_idx, x < xmax)
    x = x[valid_idx]
    y = y[valid_idx]
    for xx, yy in zip(x, y):
        grid[int(yy), int(xx)] += 1
    return grid


def _bin_2d(x, y, xmin=0, xmax=4096, ymin=0, ymax=4096):
    splits = 24
    with ThreadPoolExecutor(max_workers=splits) as pool:
        chunk = x.shape[0] // splits
        chunksx = [x[i * chunk:(i + 1) * chunk] for i in range(splits)]
        chunksy = [y[i * chunk:(i + 1) * chunk] for i in range(splits)]
        f = partial(_bin_2d, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        results = pool.map(f, chunksx, chunksy)
        results = np.sum(results)
    return results


def read_ccd_from_hdf(path):
    with h5py.File(path, "r") as h5f:
        # read in CCD information
        Nx = h5f[f"CCD"].attrs['Nx']
        Ny = h5f[f"CCD"].attrs['Ny']
        ps = h5f[f"CCD"].attrs['pixelsize']
        return CCD(xmin=0, xmax=Nx, ymax=Ny, pixelsize=ps)


@autologging.traced(logger)
@autologging.logged(logger)
class CCD:
    def __init__(self, name='detector', xmin=0, xmax=4096, ymin=0, ymax=4096, maxval=65536, pixelsize=9):
        self.data = np.zeros(((ymax - ymin), (xmax - xmin)), dtype=np.int32)
        self.name = name
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.maxval = maxval
        self.pixelsize = pixelsize

    def add_photons(self, x_positions, y_positions):
        self.data += bin_2d(x_positions, y_positions, self.xmin, self.xmax, self.ymin, self.ymax)

    def add_readnoise(self, std=3.):
        self.data += np.asarray(np.random.normal(0., std, self.data.shape).round(0), dtype=np.int32)

    def add_bias(self, value: int = 1000):
        """Adds a bias value to the detector counts

        Args:
            value: bias value to be added. If float will get rounded to next integer

        Returns:
            None
        """
        self.data += value

    def clip(self):
        if np.any(self.data < 0):
            logger.warning('There is data <0 which will be clipped. Make sure you e.g. apply the bias before the '
                           'readnoise.')
            self.data[self.data < 0] = 0
        if np.any(self.data > self.maxval):
            self.data[self.data > self.maxval] = self.maxval
