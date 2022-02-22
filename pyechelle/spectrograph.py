from __future__ import annotations

import logging
from dataclasses import dataclass

import h5py
import numpy as np
import scipy.interpolate

from pyechelle.CCD import CCD
from pyechelle.efficiency import SystemEfficiency, GratingEfficiency, TabulatedEfficiency, ConstantEfficiency
from pyechelle.optics import AffineTransformation, PSF


@dataclass
class Spectrograph:
    """ Abstract spectrograph model

    Describes all methods that a spectrograph model must have to be used in a simulation. \n
    When subclassing, all methods need to be implemented in the subclass.

    A spectrograph model as at least one CCD (with CCD_index 1), at least one field/fiber (with fiber index 1),
    and at least one diffraction order.
    """
    name: str = 'Spectrograph'

    def get_fibers(self, ccd_index: int = 1) -> list[int]:
        """ Fields/fiber indices

        Args:
            ccd_index: CCD index

        Returns:
            available fields/fiber indices
        """
        raise NotImplementedError

    def get_orders(self, fiber: int = 1, ccd_index: int = 1) -> list[int]:
        """ Diffraction orders

        Args:
            fiber: fiber/field index
            ccd_index: CCD index

        Returns:
            available diffraction order(s) for given indices
        """
        raise NotImplementedError

    def get_transformation(self, wavelength: float | np.ndarray, order: int, fiber: int = 1,
                           ccd_index: int = 1) -> AffineTransformation | np.ndarray:
        """ Transformation matrix/matrices

        Args:
            wavelength: wavelength(s) [micron]
            order: diffraction order
            fiber: fiber index
            ccd_index: CCD index

        Returns:
            transformation matrix/matrices
        """
        raise NotImplementedError

    def get_psf(self, wavelength: float | None, order: int, fiber: int = 1, ccd_index: int = 1) -> PSF | list[PSF]:
        """ PSF

        PSFs are tabulated. When wavelength is provided, the closest available PSF of the model is returned.

        When wavelength is None, all PSFs for that particular order (and fiber and CCD index) are returned.

        Args:
            wavelength: wavelength [micron] or None
            order: diffraction order
            fiber: fiber index
            ccd_index: ccd index

        Returns:
            PSF(s)
        """
        raise NotImplementedError

    def get_wavelength_range(self, order: int | None = None, fiber: int | None = None, ccd_index: int | None = None) \
            -> tuple[float, float]:
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
        """ Get CCD object(s)

        When index is provided the corresponding CCD object is returned.\n
        If no index is provided, all available CCDs are return as a dict with the index as key.

        Args:
            ccd_index: CCD index

        Returns:
            CCD object(s)
        """
        raise NotImplementedError

    def get_field_shape(self, fiber: int, ccd_index: int) -> str:
        """ Shape of field/fiber

        Returning the field/fiber shape for the given indices as a string.
        See slit.py for currently implemented shapes.

        Args:
            fiber: fiber index
            ccd_index: ccd index

        Returns:
            field/fiber shape as string (e.g. rectangular, octagonal)
        """
        raise NotImplementedError

    def get_efficiency(self, fiber: int, ccd_index: int) -> SystemEfficiency:
        """ Spectrograph efficiency

        Args:
            fiber: fiber/field index
            ccd_index: CCD index

        Returns:
            System efficiency for given indices
        """
        raise NotImplementedError


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

    def get_transformation(self, wavelength: float | np.ndarray, order: int, fiber: int = 1,
                           ccd_index: int = 1) -> AffineTransformation | list[AffineTransformation]:
        if isinstance(wavelength, float):
            return AffineTransformation(0.0, 1.0, 10., 0., (wavelength - 0.5) * wavelength * 100000. + 2000.,
                                        fiber * 10. + 2000., wavelength)
        else:
            return np.vstack([AffineTransformation(0.0, 1.0, 10., 0., (w - 0.5) * w * 100000. + 2000.,
                                                   fiber * 10. + 2000., w).as_matrix() for w in wavelength]).T

    @staticmethod
    def gauss_map(size_x, size_y=None, sigma_x=5., sigma_y=None):
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

    def get_psf(self, wavelength: float | None, order: int, fiber: int = 1, ccd_index: int = 1) -> PSF | list[PSF]:
        if wavelength is None:
            wl = np.linspace(*self.get_wavelength_range(order, fiber, ccd_index), 20)
            return [PSF(w, self.gauss_map(11, sigma_x=3., sigma_y=10.), 1.5) for w in wl]
        else:
            return PSF(wavelength, self.gauss_map(11, sigma_x=3., sigma_y=10.), 1.5)

    def get_wavelength_range(self, order: int | None = None, fiber: int | None = None, ccd_index: int | None = None) \
            -> tuple[float, float]:
        return 0.4, 0.6

    def get_ccd(self, ccd_index: int | None = None) -> CCD | dict[int, CCD]:
        if ccd_index is None:
            return self._ccd
        else:
            return self._ccd[ccd_index]

    def get_field_shape(self, fiber: int, ccd_index: int) -> str:
        return 'rectangular'

    def get_efficiency(self, fiber: int, ccd_index: int) -> SystemEfficiency:
        return SystemEfficiency([ConstantEfficiency(1.0)], 'System')


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
        return CCD(n_pix_x=nx, n_pix_y=ny, pixelsize=ps)

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
        if fiber not in self._transformations[ccd_index].keys():
            self._transformations[ccd_index][fiber] = {}
        if order not in self._transformations[ccd_index][fiber].keys():
            try:
                self._transformations[ccd_index][fiber][order] = [AffineTransformation(*af)
                                                                  for af in
                                                                  self.h5f[
                                                                      f"CCD_{ccd_index}/fiber_{fiber}/order{order}"][
                                                                      ()]]
                self._transformations[ccd_index][fiber][order].sort()

            except KeyError:
                raise KeyError(
                    f"You asked for the affine transformation matrices in diffraction order {order}. "
                    f"But this data is not available")

        return self._transformations[ccd_index][fiber][order]

    def spline_transformations(self, order: int, fiber: int = 1, ccd_index: int = 1) -> callable(float):
        if ccd_index not in self._spline_transformations.keys():
            self._spline_transformations[ccd_index] = {}
        if fiber not in self._spline_transformations[ccd_index].keys():
            self._spline_transformations[ccd_index][fiber] = {}
        if order not in self._spline_transformations[ccd_index][fiber].keys():
            tfs = self.transformations(order, fiber, ccd_index)
            m0 = [t.m0 for t in tfs]
            m1 = [t.m1 for t in tfs]
            m2 = [t.m2 for t in tfs]
            m3 = [t.m3 for t in tfs]
            m4 = [t.m4 for t in tfs]
            m5 = [t.m5 for t in tfs]
            wl = [t.wavelength for t in tfs]

            self._spline_transformations[ccd_index][fiber][order] = (scipy.interpolate.UnivariateSpline(wl, m0),
                                                                     scipy.interpolate.UnivariateSpline(wl, m1),
                                                                     scipy.interpolate.UnivariateSpline(wl, m2),
                                                                     scipy.interpolate.UnivariateSpline(wl, m3),
                                                                     scipy.interpolate.UnivariateSpline(wl, m4),
                                                                     scipy.interpolate.UnivariateSpline(wl, m5))
        return self._spline_transformations[ccd_index][fiber][order]

    def get_transformation(self, wavelength: float | np.ndarray, order: int, fiber: int = 1,
                           ccd_index: int = 1) -> AffineTransformation | list[AffineTransformation]:
        if isinstance(wavelength, float):
            return AffineTransformation(
                *tuple([float(ev(wavelength)) for ev in self.spline_transformations(order, fiber, ccd_index)]),
                wavelength)
        else:
            return [ev(wavelength) for ev in self.spline_transformations(order, fiber, ccd_index)]

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

    def get_psf(self, wavelength: float | None, order: int, fiber: int = 1, ccd_index: int = 1) -> PSF | list[PSF]:
        if wavelength is None:
            return self.psfs(order, fiber, ccd_index)
        else:
            # find the nearest PSF:
            idx = min(range(len(self.psfs(order, fiber, ccd_index))),
                      key=lambda i: abs(self.psfs(order, fiber, ccd_index)[i].wavelength - wavelength))
            return self.psfs(order, fiber, ccd_index)[idx]

    def get_wavelength_range(self, order: int | None = None, fiber: int | None = None, ccd_index: int | None = None) \
            -> tuple[float, float]:
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

    def get_efficiency(self, fiber: int, ccd_index: int) -> SystemEfficiency:
        ge = GratingEfficiency(self.h5f[f"CCD_{ccd_index}/Spectrograph"].attrs['blaze'],
                               self.h5f[f"CCD_{ccd_index}/Spectrograph"].attrs['blaze'],
                               self.h5f[f"CCD_{ccd_index}/Spectrograph"].attrs['gpmm'])

        if ccd_index not in self._efficiency.keys():
            self._efficiency[ccd_index] = {}
        if fiber not in self._efficiency[ccd_index].keys():
            try:
                self._efficiency[ccd_index][fiber] = \
                    SystemEfficiency([ge,
                                      TabulatedEfficiency('System', *self.h5f[f"CCD_{ccd_index}/fiber_{fiber}"].attrs[
                                          "efficiency"])], 'System')

            except KeyError:
                logging.warning(f'No spectrograph efficiency data found for fiber {fiber}.')
                self._efficiency[ccd_index][fiber] = SystemEfficiency([ge], 'System')
        return self._efficiency[ccd_index][fiber]

    def __exit__(self):
        if self._h5f:
            self._h5f.close()


class InteractiveZEMAX(Spectrograph):

    def get_fibers(self, ccd_index: int = 1) -> list[int]:
        pass

    def get_orders(self, fiber: int = 1, ccd_index: int = 1) -> list[int]:
        pass

    def get_transformation(self, wavelength: float | np.ndarray, order: int, fiber: int = 1,
                           ccd_index: int = 1) -> AffineTransformation | list[AffineTransformation]:
        pass

    def get_psf(self, wavelength: float | None, order: int, fiber: int = 1, ccd_index: int = 1) -> PSF | list[PSF]:
        pass

    def get_wavelength_range(self, order: int | None = None, fiber: int | None = None, ccd_index: int | None = None) -> \
            tuple[float, float]:
        pass

    def get_ccd(self, ccd_index: int | None = None) -> CCD | dict[int, CCD]:
        pass

    def get_field_shape(self, fiber: int, ccd_index: int) -> str:
        pass

    def get_efficiency(self, fiber: int, ccd_index: int) -> SystemEfficiency:
        pass


class Disturber(Spectrograph):

    def __init__(self, spec: Spectrograph, d_tx=0., d_ty=0., d_rot=0., d_shear=0., d_sx=0., d_sy=0.):
        self.spec = spec
        for method in dir(Spectrograph):
            if method.startswith('get_') and method != 'get_transformation':
                setattr(self, method, getattr(self.spec, method))
        self.disturber_matrix = AffineTransformation(d_rot, d_sx, d_sy, d_shear, d_tx, d_ty, None)

    def get_transformation(self, wavelength: float | np.ndarray, order: int, fiber: int = 1,
                           ccd_index: int = 1) -> AffineTransformation | np.ndarray:
        if isinstance(wavelength, float):
            return self.spec.get_transformation(wavelength, order, fiber, ccd_index) + self.disturber_matrix
        else:
            return self.spec.get_transformation(wavelength, order, fiber, ccd_index) + \
                   np.expand_dims(self.disturber_matrix.as_matrix(), axis=-1)


if __name__ == "__main__":
    simple = SimpleSpectrograph()
    # print(simple.get_transformation(0.503, 1))
    wl = np.linspace(*simple.get_wavelength_range(), 100)

    print(simple.get_transformation(wl, 1))

    dis = Disturber(simple, d_tx=1.)
    print(dis.get_transformation(wl, 1))

    print(simple.get_transformation(wl, 1) - dis.get_transformation(wl, 1))
    # print(simple.get_psf(0.503, 1))
    # plt.figure()
    # plt.imshow(simple.get_psf(0.503, 1).data)
    # plt.show()
    #

    # zm = ZEMAX("/home/stuermer/PycharmProjects/new_Models/models/MaroonX.hdf")
    # print(zm.get_CCD())
    # print(zm.get_fibers())
    # print(zm.get_orders(1, 1))
    # zm.get_psf(int(10, 10, 1, 1)

    # print(zm.get_psf(0.6088, 100, 1, 1))
    #
    # print(zm.get_transformation(0.610, 100, 2, 1, ))
    # print(zm.get_transformation(0.610, 100, 3, 1, ))
    # print(zm.get_transformation(0.610, 100, 4, 1, ))

    # wl = np.linspace(*zm.get_wavelength_range(100,1,1), 1000)
    # zm.get_transformation(wl)

    # print(zm.get_transformation(0.6101, 100))
    # print(zm.get_wavelength_range(100))
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
