from __future__ import annotations

import math
from dataclasses import dataclass, field

import h5py
import numpy as np
import scipy.interpolate

from efficiency import SystemEfficiency
from pyechelle.CCD import CCD


@dataclass
class AffineTransformation:
    sx: float
    sy: float
    rot: float
    shear: float
    tx: float
    ty: float
    wavelength: float

    m0: float = field(init=False)
    m1: float = field(init=False)
    m2: float = field(init=False)
    m3: float = field(init=False)
    m4: float = field(init=False)
    m5: float = field(init=False)

    def __post_init__(self):
        self.m0 = self.sx * math.cos(self.rot)
        self.m1 = -self.sy * math.sin(self.rot + self.shear)
        self.m2 = self.tx
        self.m3 = self.sx * math.sin(self.rot)
        self.m4 = self.sy * math.cos(self.rot + self.shear)
        self.m5 = self.ty

    def __le__(self, other):
        return self.wavelength <= other.wavelength

    def __lt__(self, other):
        return self.wavelength < other.wavelength

    def asarray(self):
        return self.m0, self.m1, self.m2, self.m3, self.m4, self.m5


@dataclass
class PSF:
    wavelength: float
    data: np.ndarray
    sampling: float

    def __le__(self, other):
        return self.wavelength <= other.wavelength

    def __lt__(self, other):
        return self.wavelength < other.wavelength

    def __str__(self):
        res = f'PSF@\n{self.wavelength:.4f}micron\n'
        letters = ['.', ':', 'o', 'x', '#', '@']
        norm = np.max(self.data)
        for d in self.data:
            for dd in d:
                i = int(math.floor(dd / norm / 0.2))
                res += letters[i]
            res += '\n'
        return res


class Spectrograph:
    """ Entire Spectrograph

    Describes an entire spectrograph.
    """

    def get_fibers(self, ccd_index: int = 1) -> list[int]:
        raise NotImplementedError

    def get_orders(self, fiber: int = 1, ccd_index: int = 1) -> list[int]:
        raise NotImplementedError

    def get_transformation(self, wavelength: float, order: int, fiber: int = 1,
                           ccd_index: int = 1) -> AffineTransformation:
        """ Transformation matrix

        Args:
            wavelength: wavelength [micron]
            order: diffraction order
            fiber: fiber index
            ccd_index: CCD index

        Returns:
            transformation matrix
        """
        raise NotImplementedError

    def get_psf(self, wavelength: float, order: int, fiber: int = 1, ccd_index: int = 1) -> PSF:
        """ PSF

        Args:
            wavelength: wavelength [micron]
            order: diffraction order
            fiber: fiber index
            ccd_index: ccd index

        Returns:
            PSF
        """
        raise NotImplementedError

    def get_wavelength_range(self, order: int | None = None, fiber: int | None = None, ccd_index: int | None = None) -> \
            tuple[
                float, float]:
        """ Wavelength range

        Returns minimum and maximum wavelength of the entire spectrograph unit or an individual order if specified.
        Args:
            ccd_index: CCD index
            fiber: fiber index
            order: diffraction order

        Returns:
            minimum and maximum wavelength [microns]
        """
        raise NotImplementedError

    def get_ccd(self, ccd_index: int | None = None) -> CCD | dict[int, CCD]:
        raise NotImplementedError

    def get_field_shape(self, fiber: int, ccd_index: int) -> str:
        pass

    def get_efficiency(self, ccd_index: int) -> SystemEfficiency:
        pass


class SimpleSpectrograph(Spectrograph):
    def __init__(self):
        self._ccd = {1: CCD()}
        self._fibers = {}
        self._orders = {}
        self._transformations = {}
        for c in self._ccd.keys():
            self._fibers[c] = [1]
            self._orders[c] = {}
            for f in self._fibers.keys():
                self._orders[c][f] = [1]

    def get_fibers(self, ccd_index: int = 1) -> list[int]:
        return self._fibers[ccd_index]

    def get_orders(self, fiber: int = 1, ccd_index: int = 1) -> list[int]:
        return self._orders[ccd_index][fiber]

    def get_transformation(self, wavelength: float, order: int, fiber: int = 1,
                           ccd_index: int = 1) -> AffineTransformation:

        return AffineTransformation(1.0, 1.0, 0., 0., (wavelength - 0.5) * 1000, fiber * 10., wavelength)

    def gauss_map(self, size_x, size_y=None, sigma_x=5., sigma_y=None):
        if size_y is None:
            size_y = size_x
        if sigma_y is None:
            sigma_y = sigma_x

        assert isinstance(size_x, int)
        assert isinstance(size_y, int)

        x0 = size_x // 2
        y0 = size_y // 2

        x = np.arange(0, size_x, dtype=float)
        y = np.arange(0, size_y, dtype=float)[:, np.newaxis]

        x -= x0
        y -= y0

        exp_part = x ** 2 / (2 * sigma_x ** 2) + y ** 2 / (2 * sigma_y ** 2)
        return 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-exp_part)

    def get_psf(self, wavelength: float, order: int, fiber: int = 1, ccd_index: int = 1) -> PSF:
        return PSF(wavelength, self.gauss_map(11, sigma_x=1.5, sigma_y=2.), 1.5)

    def get_wavelength_range(self, order: int | None = None, fiber: int | None = None, ccd_index: int | None = None) -> \
            tuple[
                float, float]:
        pass

    def get_ccd(self, ccd_index: int | None = None) -> CCD | dict[int, CCD]:
        if ccd_index is None:
            return self._ccd
        else:
            return self._ccd[ccd_index]


class ZEMAX(Spectrograph):
    def __init__(self, path):
        self._CCDs = {}
        self._ccd_keys = []
        self.path = path
        self._h5f = None

        self._transformations = {}
        self._spline_transformations = {}
        self._psfs = {}
        self._efficiency = {}

        # try:
        #     self.efficiency = self.h5f[f"fiber_{fiber}"].attrs["efficiency"]
        # except KeyError:
        #     logging.warning(f'No spectrograph efficiency data found for fiber {fiber}.')
        #     self.efficiency = None

        # self.blaze = self.h5f[f"Spectrograph"].attrs['blaze']
        # self.gpmm = self.h5f[f"Spectrograph"].attrs['gpmm']

        # self.name = self.h5f[f"Spectrograph"].attrs['name']
        self._field_shape = {}

        self._orders = {}

        self.CCD = [self._read_ccd_from_hdf]

    @property
    def h5f(self):
        if self._h5f is None:
            self._h5f = h5py.File(self.path, "r")
        return self._h5f

    def _read_ccd_from_hdf(self, k) -> CCD:
        # read in CCD information
        nx = self.h5f[f"CCD_{k}"].attrs['Nx']
        ny = self.h5f[f"CCD_{k}"].attrs['Ny']
        ps = self.h5f[f"CCD_{k}"].attrs['pixelsize']
        return CCD(xmax=nx, ymax=ny, pixelsize=ps)

    def get_fibers(self, ccd_index: int = 1) -> list[int]:
        return [int(k[6:]) for k in self.h5f[f"CCD_{ccd_index}"].keys() if "fiber" in k]

    def get_field_shape(self, fiber: int, ccd_index: int) -> str:
        if ccd_index not in self._field_shape.keys():
            self._field_shape[ccd_index] = {}
        if fiber not in self._field_shape[ccd_index].keys():
            self._field_shape[ccd_index][fiber] = self.h5f[f"CCD_{ccd_index}/fiber_{fiber}"].attrs["field_shape"]
        return self._field_shape[ccd_index][fiber]

    def get_orders(self, fiber: int = 1, ccd_index: int = 1) -> list[int]:
        if ccd_index not in self._orders.keys():
            self._orders[ccd_index] = {}
        if fiber not in self._orders[ccd_index].keys():
            self._orders[ccd_index][fiber] = [int(k[5:]) for k
                                              in self.h5f[f"CCD_{ccd_index}/fiber_{fiber}/"].keys() if "psf" not in k]
            self._orders[ccd_index][fiber].sort()
        return self._orders[ccd_index][fiber]

    def transformations(self, order: int, fiber: int = 1, ccd_index: int = 1) -> list[AffineTransformation]:
        if ccd_index not in self._transformations.keys():
            self._transformations[ccd_index] = {}
        if order not in self._transformations[ccd_index].keys():
            try:
                self._transformations[ccd_index][order] = [AffineTransformation(*af)
                                                           for af in
                                                           self.h5f[f"CCD_{ccd_index}/fiber_{fiber}/order{order}"][()]]
                self._transformations[ccd_index][order].sort()

            except KeyError:
                raise KeyError(
                    f"You asked for the affine transformation metrices in diffraction order {order}. "
                    f"But this data is not available")

        return self._transformations[ccd_index][order]

    def spline_transformations(self, order: int, fiber: int = 1, ccd_index: int = 1) -> callable(float):
        if ccd_index not in self._spline_transformations.keys():
            self._spline_transformations[ccd_index] = {}
        if order not in self._spline_transformations[ccd_index].keys():
            tfs = self.transformations(order, fiber, ccd_index)
            sx = [t.sx for t in tfs]
            sy = [t.sy for t in tfs]
            rot = [t.rot for t in tfs]
            shear = [t.shear for t in tfs]
            tx = [t.tx for t in tfs]
            ty = [t.ty for t in tfs]
            wl = [t.wavelength for t in tfs]

            self._spline_transformations[ccd_index][order] = (scipy.interpolate.UnivariateSpline(wl, sx),
                                                              scipy.interpolate.UnivariateSpline(wl, sy),
                                                              scipy.interpolate.UnivariateSpline(wl, rot),
                                                              scipy.interpolate.UnivariateSpline(wl, shear),
                                                              scipy.interpolate.UnivariateSpline(wl, tx),
                                                              scipy.interpolate.UnivariateSpline(wl, ty))
        return self._spline_transformations[ccd_index][order]

    def get_transformation(self, wavelength: float, order: int, fiber: int = 1,
                           ccd_index: int = 1) -> AffineTransformation:

        return AffineTransformation(
            *tuple([float(ev(wavelength)) for ev in self.spline_transformations(order, fiber, ccd_index)]), wavelength)
        #
        # idx = min(range(len(self.transformations(order, fiber, ccd_index))),
        #           key=lambda i: abs(self.transformations(order, fiber, ccd_index)[i].wavelength - wavelength))
        # return self.transformations(order, fiber, ccd_index)[idx]

    def psfs(self, order: int, fiber: int = 1, ccd_index: int = 1) -> list[PSF]:
        if ccd_index not in self._psfs.keys():
            self._psfs[ccd_index] = {}
        if order not in self._psfs[ccd_index].keys():
            try:
                self._psfs[ccd_index][order] = \
                    [PSF(self.h5f[f"CCD_{ccd_index}/fiber_{fiber}/psf_order_{order}/{wl}"].attrs['wavelength'],
                         self.h5f[f"CCD_{ccd_index}/fiber_{fiber}/psf_order_{order}/{wl}"][()],
                         self.h5f[f"CCD_{ccd_index}/fiber_{fiber}/psf_order_{order}/{wl}"].attrs[
                             'dataSpacing'])
                     for wl in self.h5f[f"CCD_{ccd_index}/fiber_{fiber}/psf_order_{order}"]]

            except KeyError:
                raise KeyError(f"You asked for the PSFs in diffraction order {order}. But this data is not available")
            self._psfs[ccd_index][order].sort()
        return self._psfs[ccd_index][order]

    def get_psf(self, wavelength: float, order: int, fiber: int = 1, ccd_index: int = 1) -> PSF:
        # find the nearest PSF:
        idx = min(range(len(self.psfs(order, fiber, ccd_index))),
                  key=lambda i: abs(self.psfs(order, fiber, ccd_index)[i].wavelength - wavelength))
        return self.psfs(order, fiber, ccd_index)[idx]

    def get_wavelength_range(self, order: int | None = None, fiber: int | None = None, ccd_index: int | None = None) -> \
            tuple[
                float, float]:
        min_w = []
        max_w = []

        if ccd_index is None:
            new_ccd_index = self.available_ccd_keys()
        else:
            new_ccd_index = [ccd_index]

        for ci in new_ccd_index:
            if fiber is None:
                new_fiber = self.get_fibers(ci)
            else:
                new_fiber = [fiber]

            for f in new_fiber:
                if order is None:
                    new_order = self.get_orders(f, ci)
                else:
                    new_order = [order]
                for o in new_order:
                    min_w.append(self.transformations(o, f, ci)[0].wavelength)
                    max_w.append(self.transformations(o, f, ci)[-1].wavelength)
        return min(min_w), max(max_w)

    def available_ccd_keys(self) -> list[int]:
        if not self._ccd_keys:
            self._ccd_keys = [int(k[4:]) for k in self.h5f[f"/"].keys() if "CCD" in k]
        return self._ccd_keys

    def get_ccd(self, ccd_index: int | None = None) -> CCD | dict[int, CCD]:
        if ccd_index is None:
            return dict(zip(self.available_ccd_keys(), [self._read_ccd_from_hdf(k) for k in self.available_ccd_keys()]))

        if ccd_index not in self._CCDs:
            self._CCDs[ccd_index] = self._read_ccd_from_hdf(ccd_index)
        return self._CCDs[ccd_index]

    def __exit__(self):
        if self._h5f:
            self._h5f.close()


if __name__ == "__main__":
    simple = SimpleSpectrograph()
    print(simple.get_transformation(0.503, 1))
    print(simple.get_psf(0.503, 1))
    # plt.figure()
    # plt.imshow(simple.get_psf(0.503, 1).data)
    # plt.show()
    #

    zm = ZEMAX("/home/stuermer/PycharmProjects/new_Models/models/MaroonX.hdf")
    # print(zm.get_CCD())
    # print(zm.get_fibers())
    # print(zm.get_orders(1, 1))
    # zm.get_psf(int(10, 10, 1, 1)

    print(zm.get_psf(0.6088, 100, 1, 1))

    print(zm.get_transformation(0.610, 100))
    print(zm.get_transformation(0.6101, 100))
    print(zm.get_wavelength_range(100))
    # print(zm.get_wavelength_range(100, 1))
    # print(zm.get_wavelength_range(fiber=2))
    # print(zm.get_wavelength_range(100, 3))
    # # print(zm.get_field_shape(3))
    # zm = ZEMAXUnit("/home/stuermer/PycharmProjects/pyechelle/pyechelle/models/MaroonX.hdf", 1)
    # psf = zm.get_psf(0.608, 100)
    # for o in zm.orders:
    #     min_w, max_w = zm.get_wavelength_range(o)
    #     print(zm.get_psf((min_w + max_w) / 2., o))
    #     print(zm.get_transformation((min_w + max_w) / 2., o))
    #
    # print(zm.orders)
    # print(zm.get_wavelength_range())
    # print(zm.get_wavelength_range(100))
    #
    # tf = zm.get_transformation(0.608, 100)
    # print(tf)
