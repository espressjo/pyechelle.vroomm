from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import h5py
import numpy as np

import pyechelle.CCD
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


@dataclass
class PSF:
    wavelength: float
    data: np.ndarray
    sampling: float

    def __le__(self, other):
        return self.wavelength <= other.wavelength

    def __lt__(self, other):
        return self.wavelength < other.wavelength


@dataclass
class Field:
    """ Input Field

        Describes the input field of a spectrograph i.e. a slit or a fiber
        Attributes:
            name (str): name of the field
            shape (str): shape string of the field (e.g. circular, octagonal, rectangular)
        """
    name: str
    shape: str


class Spectrograph:
    """ Entire Spectrograph

    Describes an entire spectrograph, i.e. a collection of SpectrographUnits
    """

    def get_spectrograph_unit(self, field: int = 1, detector: int = 1) -> SpectrographUnit:
        raise NotImplementedError


class SpectrographUnit:
    """ SpectrographUnit

    Defines a spectrograph unit i.e. everything that is needed to describe a single input field and a single detector.
    Attributes:
        field_shape (str): an input field shape (circular, octagonal etc.)
        transformations (dict[int, dict[float, AffineTransformation]]): affine transformation matrices describing the optical behaviour
        orders (list[int]): list of diffraction orders
        CCD (CCD): detector object describing the CCD.
    """

    def __init__(self):
        self.field_shape = None
        self.orders = None
        self.CCD = None

    def get_transformation(self, wavelength: float | Iterable[float], order: int):
        """ Transformation matrix

        Args:
            wavelength: wavelength [micron]
            order: diffraction order

        Returns:
            transformation matrix
        """
        raise NotImplementedError

    def get_psf(self, wavelength: float | Iterable[float], order: int) -> PSF | Iterable[PSF]:
        """ PSF

        Args:
            wavelength: wavelength [micron]
            order: diffraction order

        Returns:
            PSF
        """
        raise NotImplementedError

    def get_wavelength_range(self, order: int | None = None) -> tuple[float, float]:
        """ Wavelength range

        Returns minimum and maximum wavelength of the entire spectrograph unit or an individual order if specified.
        Args:
            order: diffraction order

        Returns:
            minimum and maximum wavelength [microns]
        """
        raise NotImplementedError


class PSFs:
    def __init__(self):
        self.wl = []
        self.psfs = []
        self.idx = None
        self.sampling = []

    def add_psf(self, wl, data, sampling):
        self.wl.append(wl)
        self.psfs.append(data)
        self.sampling.append(sampling)
        self.idx = None

    def prepare_lookup(self):
        self.idx = np.argsort(self.wl)
        self.wl = np.array(self.wl)[self.idx]
        self.psfs = np.array(self.psfs)[self.idx]
        self.sampling = np.array(self.sampling)[self.idx]


class ZEMAX(Spectrograph):
    def __init__(self, path):
        self.CCD = pyechelle.CCD.read_ccd_from_hdf(path)
        self.path = path

    def get_spectrograph_unit(self, field: int = 1, detector: int = 1) -> SpectrographUnit:
        return ZEMAXUnit(self.path, field)


class ZEMAXUnit(SpectrographUnit):
    """ ZEMAXUnit
    Attributes:
        psfs (dict[int, list[PSF]]): tabulated PSFs
        transformations (dict[int, list[AffineTransformation]): tabulated affine transformation matrices
    """

    def __init__(self, path, fiber: int = 1):
        """
        Load spectrograph model from ZEMAX based .hdf model.

        Args:
            path: file path
            fiber: which fiber
        """
        super().__init__()
        self.transformations = {}
        self.order_keys = {}
        self.psfs = {}
        self.fiber = fiber
        self.field_shape = "round"
        self.CCD = None
        self.efficiency = None

        self.blaze = None
        self.gpmm = None
        self.name = None
        self.modelpath = path
        self.h5f = h5py.File(path, "r")

        # self.CCD = pyechelle.CCD.read_ccd_from_hdf(path)
        #
        # with h5py.File(path, "r") as h5f:
        #     # read in grating information
        #     self.name = h5f[f"Spectrograph"].attrs['name']
        #     self.blaze = h5f[f"Spectrograph"].attrs['blaze']
        #     self.gpmm = h5f[f"Spectrograph"].attrs['gpmm']
        #
        #     self.field_shape = h5f[f"fiber_{fiber}"].attrs["field_shape"]
        #     try:
        #         self.efficiency = h5f[f"fiber_{fiber}"].attrs["efficiency"]
        #     except KeyError:
        #         logging.warning(f'No spectrograph efficiency data found for fiber {fiber}.')
        #         self.efficiency = None
        #     for g in h5f[f"fiber_{fiber}"]:
        #         if not "psf" in g:
        #
        #             data = h5f[f"fiber_{fiber}/{g}"][()]
        #             data = np.sort(data, order='wavelength')
        #             self.transformations[g] = AffineTransformation(*data.view((data.dtype[0], len(data.dtype.names))).T)
        #         if "psf" in g:
        #             self.psfs[g] = PSFs()
        #             for wl in h5f[f"fiber_{fiber}/{g}"]:
        #                 self.psfs[g].add_psf(h5f[f"fiber_{fiber}/{g}/{wl}"].attrs['wavelength'],
        #                                      h5f[f"fiber_{fiber}/{g}/{wl}"][()],
        #                                      h5f[f"fiber_{fiber}/{g}/{wl}"].attrs['dataSpacing'])
        #             self.psfs[g].prepare_lookup()
        # self.order_keys = list(self.transformations.keys())
        # self.orders = [int(o[5:]) for o in self.order_keys]
        # print(f"Available orders: {self.orders}")

    def get_wavelength_range(self, order: int | None = None) -> tuple[float, float]:
        if order is not None:
            return self.transformations[f"order{order}"].min_wavelength(), self.transformations[
                f"order{order}"].max_wavelength()
        else:
            minw = min([self.transformations[f"order{order}"].min_wavelength() for order in self.orders])
            maxw = max([self.transformations[f"order{order}"].max_wavelength() for order in self.orders])
            return minw, maxw

    def get_transformation(self, wavelength: float | Iterable[float], order: int):
        if order not in self.transformations.keys():
            # read in transformations for this order
            try:
                self.transformations[order] = [AffineTransformation(*af)
                                               for af in self.h5f[f"fiber_{self.fiber}/order{order}"][()]]
                self.transformations[order].sort()
                
            except KeyError:
                raise KeyError(
                    f"You asked for the affine transformation metrices in diffraction order {order}. "
                    f"But this data is not available")
        idx = min(range(len(self.transformations[order])),
                  key=lambda i: abs(self.transformations[order][i].wavelength - wavelength))
        return self.transformations[order][idx]

    def get_psf(self, wavelength: float, order: int) -> PSF:
        if order not in self.psfs.keys():
            # read in PSFs for this order
            try:
                self.psfs[order] = [PSF(self.h5f[f"fiber_{self.fiber}/psf_order_{order}/{wl}"].attrs['wavelength'],
                                        self.h5f[f"fiber_{self.fiber}/psf_order_{order}/{wl}"][()],
                                        self.h5f[f"fiber_{self.fiber}/psf_order_{order}/{wl}"].attrs['dataSpacing'])
                                    for wl in self.h5f[f"fiber_{self.fiber}/psf_order_{order}"]]
            except KeyError:
                raise KeyError(
                    f"You asked for the PSFs in diffraction order {order}. But this data is not available")
            self.psfs[order].sort()

        # find the nearest PSF:
        idx = min(range(len(self.psfs[order])), key=lambda i: abs(self.psfs[order][i].wavelength - wavelength))
        return self.psfs[order][idx]

    def __exit__(self):
        self.h5f.close()


if __name__ == "__main__":
    zm = ZEMAXUnit("/home/stuermer/Repos/python/pyechelle/pyechelle/models/MaroonX.hdf", 1)
    # psf = zm.get_psf(0.608, 100)
    tf = zm.get_transformation(0.608, 100)
    print(tf)
