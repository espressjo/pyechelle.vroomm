from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import h5py
import numpy as np

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


class Spectrograph:
    """ Entire Spectrograph

    Describes an entire spectrograph, i.e. a collection of SpectrographUnits
    """

    def get_spectrograph_unit(self, fiber: int = 1, detector: int = 1) -> SpectrographUnit:
        raise NotImplementedError


class SpectrographUnit:
    """ SpectrographUnit

    Defines a spectrograph unit i.e. everything that is needed to describe a single input field and a single detector.
    Attributes:
        field_shape (str): an input field shape (circular, octagonal etc.)
        transformations (dict[int, dict[float, AffineTransformation]]): affine transformation matrices describing the optical behaviour
        # orders (list[int]): list of diffraction orders
        CCD (CCD): detector object describing the CCD.
    """

    def __init__(self):
        self.field_shape = None
        self.orders = []
        self.CCD = None

    def get_transformation(self, wavelength: float, order: int):
        """ Transformation matrix

        Args:
            wavelength: wavelength [micron]
            order: diffraction order

        Returns:
            transformation matrix
        """
        raise NotImplementedError

    def get_psf(self, wavelength: float, order: int) -> PSF:
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


class ZEMAX(Spectrograph):
    def __init__(self, path):
        self.path = path

    def get_spectrograph_unit(self, field: int = 1, detector: int = 1) -> SpectrographUnit:
        return ZEMAXUnit(self.path, field)


class ZEMAXUnit(SpectrographUnit):
    """ ZEMAXUnit
    Attributes:
        _psfs (dict[int, list[PSF]]): tabulated PSFs
        _transformations (dict[int, list[AffineTransformation]): tabulated affine transformation matrices
        fiber (int): Fiber/Field number as specified in .hdf File
        h5f (h5py.File): .hdf file handle
    """

    def __init__(self, path, fiber: int = 1):
        """
        Load spectrograph model from ZEMAX based .hdf model.

        Args:
            path: file path
            fiber: which fiber
        """
        super().__init__()
        self.h5f = h5py.File(path, "r")
        self.fiber = fiber
        self._transformations = {}
        self._psfs = {}

        try:
            self.efficiency = self.h5f[f"fiber_{fiber}"].attrs["efficiency"]
        except KeyError:
            logging.warning(f'No spectrograph efficiency data found for fiber {fiber}.')
            self.efficiency = None

        self.blaze = self.h5f[f"Spectrograph"].attrs['blaze']
        self.gpmm = self.h5f[f"Spectrograph"].attrs['gpmm']

        self.name = self.h5f[f"Spectrograph"].attrs['name']
        self.field_shape = self.h5f[f"fiber_{self.fiber}"].attrs["field_shape"]

        self._orders = []

        self.CCD = self.read_ccd_from_hdf

    def read_ccd_from_hdf(self) -> CCD:
        # read in CCD information
        Nx = self.h5f[f"CCD"].attrs['Nx']
        Ny = self.h5f[f"CCD"].attrs['Ny']
        ps = self.h5f[f"CCD"].attrs['pixelsize']
        return CCD(xmax=Nx, ymax=Ny, pixelsize=ps)

    @property
    def orders(self) -> list[int]:
        if not self._orders:
            self._orders = [int(k[5:]) for k in self.h5f[f"fiber_{self.fiber}/"].keys() if "psf" not in k]
            self._orders.sort()
        return self._orders

    @orders.setter
    def orders(self, ol: list):
        self._orders = ol

    def transformations(self, order) -> list[AffineTransformation]:
        if order not in self._transformations.keys():
            try:
                self._transformations[order] = [AffineTransformation(*af)
                                                for af in self.h5f[f"fiber_{self.fiber}/order{order}"][()]]
                self._transformations[order].sort()

            except KeyError:
                raise KeyError(
                    f"You asked for the affine transformation metrices in diffraction order {order}. "
                    f"But this data is not available")

        return self._transformations[order]

    def get_wavelength_range(self, order: int | None = None) -> tuple[float, float]:
        if order is not None:
            # here we use that transformations are sorted by wavelength
            return self.transformations(order)[0].wavelength, self.transformations(order)[-1].wavelength
        else:
            min_w = min([self.transformations(o)[0].wavelength for o in self.orders])
            max_w = max([self.transformations(o)[-1].wavelength for o in self.orders])
            return min_w, max_w

    def get_transformation(self, wavelength: float, order: int):
        idx = min(range(len(self.transformations(order))),
                  key=lambda i: abs(self.transformations(order)[i].wavelength - wavelength))
        return self.transformations(order)[idx]

    def get_psf(self, wavelength: float, order: int) -> PSF:
        if order not in self._psfs.keys():
            # read in PSFs for this order
            try:
                self._psfs[order] = [PSF(self.h5f[f"fiber_{self.fiber}/psf_order_{order}/{wl}"].attrs['wavelength'],
                                         self.h5f[f"fiber_{self.fiber}/psf_order_{order}/{wl}"][()],
                                         self.h5f[f"fiber_{self.fiber}/psf_order_{order}/{wl}"].attrs['dataSpacing'])
                                     for wl in self.h5f[f"fiber_{self.fiber}/psf_order_{order}"]]
            except KeyError:
                raise KeyError(
                    f"You asked for the PSFs in diffraction order {order}. But this data is not available")
            self._psfs[order].sort()

        # find the nearest PSF:
        idx = min(range(len(self._psfs[order])), key=lambda i: abs(self._psfs[order][i].wavelength - wavelength))
        return self._psfs[order][idx]

    def __exit__(self):
        self.h5f.close()


if __name__ == "__main__":
    zm = ZEMAXUnit("/home/stuermer/PycharmProjects/pyechelle/pyechelle/models/MaroonX.hdf", 1)
    # psf = zm.get_psf(0.608, 100)
    for o in zm.orders:
        min_w, max_w = zm.get_wavelength_range(o)
        print(zm.get_psf((min_w + max_w) / 2., o))
        print(zm.get_transformation((min_w + max_w) / 2., o))
    #
    # print(zm.orders)
    # print(zm.get_wavelength_range())
    # print(zm.get_wavelength_range(100))
    #
    # tf = zm.get_transformation(0.608, 100)
    # print(tf)
