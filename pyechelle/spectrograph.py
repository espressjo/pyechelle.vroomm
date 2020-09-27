import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, int32, float64
from numba.types import UniTuple

# import pandas as pd
from pyechelle.sources import Phoenix
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


class Spectrograph:
    pass


class ZEMAX(Spectrograph):
    def __init__(self, path, fiber=3):
        super().__init__()
        self.transformations = {}
        self.orders = {}
        self.psfs = {}
        self.fibers = lambda: self.orders

        with h5py.File(path, "r") as h5f:
            for g in h5f[f"fiber_{fiber}"]:
                if not "psf" in g:
                    # data = pd.DataFrame(h5f[f"fiber_{fiber}/{g}"][()])
                    data = h5f[f"fiber_{fiber}/{g}"][()]
                    data = np.sort(data, order='wavelength')

                    # data = data.sort_values("wavelength")
                    self.transformations[g] = AffineTransformation(*data.view((data.dtype[0], len(data.dtype.names))).T)
                    self.transformations[g].make_lookup_table(10000)
                if "psf" in g:
                    for wl in h5f[f"fiber_{fiber}/{g}"]:
                        pass

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

        img = np.zeros((4096, 4096))
        t1 = time.time()
        # imgs = parmap.map(trace2d, self.transformations.items(), n)
        for o, t in self.transformations.items():
            print(o)

            xy = self.generate_slit(n)
            # xy = np.zeros((2,n), dtype=np.float64)
            # spec = Etalon(min_wl=t.min_wavelength(), max_wl=t.max_wavelength())
            spec = Phoenix(min_wl=t.min_wavelength(), max_wl=t.max_wavelength())
            wltest = spec.draw_wavelength(n)
            # wltest = np.random.uniform(np.min(t.wl), np.max(t.wl), n)
            sx, sy, rot, shear, tx, ty = t.get_transformations_lookup(wltest)
            transformed = trace(xy[0], xy[1], sx, sy, rot, shear, tx, ty)
            #     # transformed = np.array(transformed)[:, :2].T
            #
            #     # transformed += self.generate_psf_distortion(N)
            img += bin_2d(*transformed, ymin=0, ymax=4096, xmax=4096, xmin=0)
        # img = np.array(imgs).sum(axis=0)
        t2 = time.time()
        print(t2 - t1)
        return img


if __name__ == "__main__":
    spec = ZEMAX(
        "/home/stuermer/Repos/cpp/EchelleSimulator/data/spectrographs/MaroonX.hdf"
    )

    img = spec.generate_2d_spectrum(np.empty((int(5e6),)))
    plt.figure()
    plt.imshow(img)
    plt.show()
    # print(np.max(img), np.min(img))
    # spec.plot_transformations()
