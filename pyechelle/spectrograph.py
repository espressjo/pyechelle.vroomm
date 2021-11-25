import logging
from random import randrange, random

import h5py
import numpy as np
from numba import njit, int32, float64
from numba.types import UniTuple

from pyechelle.CCD import CCD
from pyechelle.randomgen import AliasSample, sample_alias_2d
from pyechelle.transformation import AffineTransformation

par = True
nogil = True


@njit(
    UniTuple(float64[:], 2)(
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
    ),
    parallel=par,
    nogil=nogil,
    cache=True,
)
def trace_native(x_vec, y_vec, sx, sy, rot, shear, tx, ty):
    """ Performs 'raytracing' for a given wavelength vector and XY input vectors

    Args:
        x_vec (np.ndarray): random X positions within the slit
        y_vec (np.ndarray): random Y positions within the slit
        sx (float): desired scaling in X direction
        sy (float):  desired scalinig in Y direction
        rot (float): desired slit rotation [rad]
        shear (float): desired slit shear
        tx (float): tx of affine matrix
        ty (float): ty of affine matrix

    Returns:
        np.ndarray: transformed XY positions for given input
    """
    m0 = sx * np.cos(rot)
    m1 = -sy * np.sin(rot + shear)
    m2 = tx
    m3 = sx * np.sin(rot)
    m4 = sy * np.cos(rot + shear)
    m5 = ty
    # do transformation
    xpos = m0 * x_vec + m1 * y_vec + m2
    ypos = m3 * x_vec + m4 * y_vec + m5
    return xpos, ypos


@njit(
    UniTuple(float64[:], 2)(
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
    ),
    parallel=par,
    nogil=nogil,
    cache=True,
)
def trace(x_vec, y_vec, m0, m1, m2, m3, m4, m5):
    """ Performs 'raytracing' for a given wavelength vector and XY input vectors

    Args:
        x_vec (np.ndarray): random X positions within the slit
        y_vec (np.ndarray): random Y positions within the slit
        m0 (float): matrix element 00
        m1 (float):  matrix element 10
        m2 (float): matrix element 20
        m3 (float): matrix element 01
        m4 (float): matrix element 11
        m5 (float): matrix element 21

    Returns:
        np.ndarray: transformed XY positions for given input
    """
    # do transformation
    return m0 * x_vec + m1 * y_vec + m2, m3 * x_vec + m4 * y_vec + m5


@njit(UniTuple(float64[:], 2)(float64[:, :], int32), parallel=True, nogil=True, cache=True)
def draw_from_2darray(data, N):
    x_cords = np.zeros(N, dtype=float64)
    y_cords = np.zeros(N, dtype=float64)
    for i in range(N):
        x = randrange(0, data.shape[1])
        y = randrange(0, data.shape[0])
        z = random()
        while z > data[y, x]:
            x = randrange(0, data.shape[1])
            y = randrange(0, data.shape[0])
            z = random()
        x_cords[i] = x
        y_cords[i] = y
    return x_cords, y_cords


def draw_from_2darray_alias(sampler, N):
    return sampler.sample(N)


class Spectrograph:
    pass


class PSF:
    def __init__(self, wl, data):
        self.wl = wl
        self.data = data / np.sum(data)
        self.sampler = AliasSample(np.ravel(data) / data.sum())

        # #
        # plt.figure()
        # plt.imshow(self.data,origin='lower')
        # # plt.show()
        # x = []
        # y = []
        # for i in range(1000):
        #     xx , yy = self.draw_xy_alias(1)
        #     x.append(xx)
        #     y.append(yy)
        # # plt.figure()
        # plt.scatter(x,y,s=1,alpha=0.5)
        # plt.show()

    def draw_xy_alias(self, N):
        return sample_alias_2d(self.data, N)

    def draw_xy(self, N):
        return draw_from_2darray(self.data, N)


class PSFs:
    def __init__(self):
        self.wl = []
        self.psfs = []
        self.idx = None
        self.sampling = []

    def add_psf(self, wl, data, sampling):
        self.wl.append(wl)
        self.psfs.append(PSF(wl, data))
        self.sampling.append(sampling)
        self.idx = None

    def prepare_lookup(self):
        self.idx = np.argsort(self.wl)
        self.wl = np.array(self.wl)[self.idx]
        self.psfs = np.array(self.psfs)[self.idx]
        self.sampling = np.array(self.sampling)[self.idx]

    def draw_xy(self, wl):
        Xlist = np.empty(len(wl))
        Ylist = np.empty(len(wl))
        bins = np.hstack((self.wl - np.mean(np.ediff1d(self.wl) / 2.), self.wl[-1] + np.mean(np.ediff1d(self.wl)) / 2.))
        idx = np.digitize(wl, bins) - 1
        for i in np.unique(idx):
            x, y = sample_alias_2d(self.psfs[i].data, np.count_nonzero(idx == i))
            Xlist[idx == i] = (x - self.psfs[i].data.shape[1] / 2.) * self.sampling[i]
            Ylist[idx == i] = (y - self.psfs[i].data.shape[0] / 2.) * self.sampling[i]
        return Xlist, Ylist


class ZEMAX(Spectrograph):
    def __init__(self, path, fiber: int = 1, n_lookup_table: int = 10000):
        """
        Load spectrograph model from ZEMAX based .hdf model.

        Args:
            path: file path
            fiber: which fiber
            n_lookup_table: number of entries in lookup
        """
        super().__init__()
        self.transformations = {}
        self.order_keys = {}
        self.psfs = {}
        self.field_shape = "round"
        self.fibers = lambda: self.order_keys
        self.CCD = None
        self.efficiency = None

        self.blaze = None
        self.gpmm = None
        self.name = None
        self.modelpath = path

        with h5py.File(path, "r") as h5f:
            # read in grating information
            self.name = h5f[f"Spectrograph"].attrs['name']
            self.blaze = h5f[f"Spectrograph"].attrs['blaze']
            self.gpmm = h5f[f"Spectrograph"].attrs['gpmm']

            Nx = h5f[f"CCD"].attrs['Nx']
            Ny = h5f[f"CCD"].attrs['Ny']
            ps = h5f[f"CCD"].attrs['pixelsize']
            self.CCD = CCD(xmin=0, xmax=Nx, ymax=Ny, pixelsize=ps)
            self.field_shape = h5f[f"fiber_{fiber}"].attrs["field_shape"]
            try:
                self.efficiency = h5f[f"fiber_{fiber}"].attrs["efficiency"]
            except KeyError:
                logging.warning(f'No spectrograph efficiency data found for fiber {fiber}.')
                self.efficiency = None
            for g in h5f[f"fiber_{fiber}"]:
                if not "psf" in g:
                    data = h5f[f"fiber_{fiber}/{g}"][()]
                    data = np.sort(data, order='wavelength')
                    self.transformations[g] = AffineTransformation(*data.view((data.dtype[0], len(data.dtype.names))).T)
                    self.transformations[g].make_lookup_table(n_lookup_table)
                if "psf" in g:
                    self.psfs[g] = PSFs()
                    for wl in h5f[f"fiber_{fiber}/{g}"]:
                        self.psfs[g].add_psf(h5f[f"fiber_{fiber}/{g}/{wl}"].attrs['wavelength'],
                                             h5f[f"fiber_{fiber}/{g}/{wl}"][()],
                                             h5f[f"fiber_{fiber}/{g}/{wl}"].attrs['dataSpacing'])
                    self.psfs[g].prepare_lookup()
        self.order_keys = list(self.transformations.keys())
        self.orders = [int(o[5:]) for o in self.order_keys]
        print(f"Available orders: {self.orders}")

    def get_wavelength_range(self, order):
        return self.transformations[f"order{order}"].min_wavelength(), self.transformations[
            f"order{order}"].max_wavelength()

