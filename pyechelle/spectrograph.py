import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, int32, float64
from numba.types import UniTuple

# import pandas as pd
from parmap import parmap

from pyechelle.CCD import bin_2d
from pyechelle.sources import Phoenix, Etalon
from pyechelle.transformation import AffineTransformation
from pyechelle.randomgen import AliasSample, samplealias2d
from random import randrange, random

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
def trace(x_vec, y_vec, sx, sy, rot, shear, tx, ty):
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


@njit(float64[:, :](int32), parallel=par, nogil=nogil, cache=True)
def generate_slit_xy(N):
    """  Generate uniform distributed XY position within a unit box

    Args:
        N (int):  number of random numbers

    Returns:
        np.ndarray: random XY position
    """
    x = np.random.random(N)
    y = np.random.random(N)
    return np.vstack((x, y))

@njit(UniTuple(float64[:], 2)(float64[:,:], int32), parallel=True, nogil=True, cache=True)
def draw_from_2darray(data, N):
    X = np.zeros(N, dtype=float64)
    Y = np.zeros(N, dtype=float64)
    for i in range(N):
        x = randrange(0, data.shape[1])
        y = randrange(0, data.shape[0])
        z = random()
        while z > data[y, x]:
            x = randrange(0, data.shape[1])
            y = randrange(0, data.shape[0])
            z = random()
        X[i] = x
        Y[i] = y
    return X, Y

def draw_from_2darray_alias(sampler, N):
    return sampler.sample(N)

class Spectrograph:
    pass

class PSF():
    def __init__(self, wl, data):
        self.wl = wl
        self.data = data / np.max(data)
        self.sampler = AliasSample(np.ravel(data)/data.sum())

        #
        # plt.figure()
        # plt.imshow(self.data,origin='lower')
        # # plt.show()
        # x = []
        # y = []
        # for i in range(1000):
        #     xx , yy = self.draw_xy()
        #     x.append(xx)
        #     y.append(yy)
        # # plt.figure()
        # plt.scatter(x,y,s=1,alpha=0.5)
        # plt.show()
    def draw_xy_alias(self, N):
        return samplealias2d(self.data, N)

    def draw_xy(self, N):
        return draw_from_2darray(self.data, N)

class PSFs:
    def __init__(self):
        self.wl = []
        self.psfs = []
        self.idx = None
        self.sampling = []

    def add_psf(self, wl, data,sampling):
        self.wl.append(wl)
        self.psfs.append(PSF(wl,data))
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
        bins = np.hstack((self.wl - np.mean(np.ediff1d(self.wl)/2.), self.wl[-1] + np.mean(np.ediff1d(self.wl))/2.))
        idx = np.digitize(wl, bins)-1
        for i in np.unique(idx):
            x, y = samplealias2d(self.psfs[i].data, np.count_nonzero(idx == i))
            Xlist[idx == i] = x
            Ylist[idx == i] = y
        # for w in wl:
        # # np.digitize(wl, np.hstack((self.wl - np.mean(np.ediff1d(self.wl)), ))
        #     idx = np.searchsorted(self.wl, w)
        #     idx = max(0, idx)
        #     idx = min(len(self.idx)-1, idx)
        #     x, y = self.psfs[idx].draw_xy(1)
        #     Xlist.append(x[0])
        #     Ylist.append(y[0])
        return Xlist, Ylist


def trace_par(o,t,psfs,n):
    xy = generate_slit_xy(n)
    # xy = np.zeros((2,n), dtype=np.float64)
    # spec = Etalon(min_wl=t.min_wavelength(), max_wl=t.max_wavelength())
    spec = Phoenix(min_wl=t.min_wavelength(), max_wl=t.max_wavelength())
    wltest = spec.draw_wavelength(n)
    # wltest = np.random.uniform(np.min(t.wl), np.max(t.wl), n)
    sx, sy, rot, shear, tx, ty = t.get_transformations_lookup(wltest)
    transformed = trace(xy[0], xy[1], sx, sy, rot, shear, tx, ty)
    #     # transformed = np.array(transformed)[:, :2].T
    X, Y = psfs[f"psf_{o[:5]}" + "_" + f"{o[5:]}"].draw_xy(wltest)
    # transformed[0] = transformed[0] + X
    # transformed[1] = transformed[1] + Y
    xx = transformed[0] + np.array(X) * psfs[f"psf_{o[:5]}" + "_" + f"{o[5:]}"].sampling[0] / 9.5
    yy = transformed[1] + np.array(Y) * psfs[f"psf_{o[:5]}" + "_" + f"{o[5:]}"].sampling[0] / 9.5
    #     # transformed += self.generate_psf_distortion(N)
    return img
    # img += bin_2d(xx, yy, ymin=0, ymax=4096, xmax=4096, xmin=0)


class ZEMAX(Spectrograph):
    def __init__(self, path, fiber=3, n_lookup_table=10000):
        """
        Load spectrograph model from ZEMAX based .hdf model.

        Args:
            path: file path
            fiber: which fiber
            n_lookup_table: number of entries in lookup
        """
        super().__init__()
        self.transformations = {}
        self.orders = {}
        self.psfs = {}
        self.fibers = lambda: self.orders
        self.CCD = None

        with h5py.File(path, "r") as h5f:
            for g in h5f[f"fiber_{fiber}"]:
                print(g)
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
        self.orders = list(self.transformations.keys())

    def plot_transformations(self, order=None):
        plt.figure()
        for t, k in self.transformations.items():
            plt.plot(k.lookup_table_sx)
        plt.show()

    def generate_slit(self, N):
        return generate_slit_xy(N)

    def generate_2d_spectrum(self, wl_vector):
        n = len(wl_vector)

        img = np.zeros((2048, 2048))
        t1 = time.time()
        # imgs = parmap.map(trace_par, self.transformations.items(), (self.psfs, n))
        for o, t in self.transformations.items():
            print(o)

            xy = self.generate_slit(n)
            # xy = np.zeros((2,n), dtype=np.float64)
            # spec = Etalon(min_wl=t.min_wavelength(), max_wl=t.max_wavelength())
            # spec = Phoenix(min_wl=t.min_wavelength(), max_wl=t.max_wavelength())
            spec = Etalon(d=3, min_wl=t.min_wavelength(), max_wl=t.max_wavelength())
            wltest = spec.draw_wavelength(n)
            # wltest = np.random.uniform(np.min(t.wl), np.max(t.wl), n)
            sx, sy, rot, shear, tx, ty = t.get_transformations_lookup(wltest)
            transformed = trace(xy[0], xy[1], sx, sy, rot, shear, tx, ty)
            xx, yy = transformed
            #     # transformed = np.array(transformed)[:, :2].T
            # X, Y = self.psfs[f"psf_{o[:5]}"+"_"+f"{o[5:]}"].draw_xy(wltest)
            # transformed[0] = transformed[0] + X
            # transformed[1] = transformed[1] + Y
            # xx = transformed[0] + np.array(X) * self.psfs[f"psf_{o[:5]}"+"_"+f"{o[5:]}"].sampling[0] / 6.5
            # yy = transformed[1] + np.array(Y) * self.psfs[f"psf_{o[:5]}"+"_"+f"{o[5:]}"].sampling[0] / 6.5
            #     # transformed += self.generate_psf_distortion(N)
            img += bin_2d(xx, yy, ymin=0, ymax=2048, xmax=2048, xmin=0)
        # img = np.array(imgs).sum(axis=0)
        # img = bin_2d(**imgs, ymin=0, ymax=4096, xmax=4096, xmin=0)
        t2 = time.time()
        print(t2 - t1)
        return img


if __name__ == "__main__":
    from pathlib import Path

    dir_path = Path(__file__).resolve().parent.parent

    spec = ZEMAX(dir_path.joinpath("/home/stuermer/Nextcloud/work/tmp/cubespec.hdf"), 1)

    img = spec.generate_2d_spectrum(np.empty((int(5e5),)))
    plt.figure()
    plt.imshow(img)
    plt.show()
    # print(np.max(img), np.min(img))
    # spec.plot_transformations()
