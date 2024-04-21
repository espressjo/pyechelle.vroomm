from __future__ import annotations

import logging
import urllib.request
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError

import h5py
import numpy as np
from matplotlib import pyplot as plt

try:
    import zospy
    from zospy.functions.lde import find_surface_by_comment
    from zospy.zpcore import OpticStudioSystem
except ImportError:
    pass
except RuntimeError:
    pass
except:
    pass

from pyechelle.CCD import CCD
from pyechelle.efficiency import (
    SystemEfficiency,
    GratingEfficiency,
    TabulatedEfficiency,
    ConstantEfficiency,
)
from pyechelle.optics import (
    AffineTransformation,
    PSF,
    TransformationSet,
    convert_matrix,
    apply_matrix,
    find_affine,
    decompose_affine_matrix,
)

from ftplib import FTP


def check_FTP_url_exists(url):
    assert url.lower().startswith("ftp://")
    url = url[6:]
    ftp_host = url.split("/")[0]
    file_path = "/".join(url.split("/")[1:])

    try:
        ftp = FTP(ftp_host)
        ftp.login()
        file_list = ftp.nlst(file_path)
        ftp.quit()
        return file_path in file_list
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False


def check_url_exists(url: str) -> bool:
    """
    Check if URL exists.
    Args:
        url: url to be tested

    Returns:
        if URL exists
    """
    if url.lower().startswith("ftp:"):
        return check_FTP_url_exists(url)
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return float(response.headers["Content-length"]) > 0
    except URLError:
        return False


def check_for_spectrograph_model(model_name: str | Path, download=True) -> Path:
    """
    Check if spectrograph model exists locally. Otherwise: Download if download is true (default) or check if URL to
    spectrograph model is valid (this is mainly for testing purpose).

    Args:
        model_name: name of spectrograph model or path to .hdf file. See models/available_models.txt for valid names
        download: download flag

    Returns:
        path to spectrograph model
    """
    if isinstance(model_name, Path):
        return model_name
    else:
        if model_name.endswith(".hdf"):
            model_name = model_name.rstrip(".hdf")

        # look in models folder
        file_path = (
            Path(__file__)
            .resolve()
            .parent.joinpath("models")
            .joinpath(f"{model_name}.hdf")
        )
        # if it doesn't exist look in current working directory
        if not file_path.is_file():
            file_path = Path.cwd().resolve().joinpath(f"{model_name}.hdf")
        # if it doesn't exist try to download
        if not file_path.is_file():
            url = f"https://cloud.stuermer.science/s/d28p3aTrDqJPSC4/download?path=/&files={model_name}.hdf"
            if download:
                print(
                    f"Spectrograph model {model_name} not found locally. Trying to download from {url}..."
                )
                Path(Path(__file__).resolve().parent.joinpath("models")).mkdir(
                    parents=False, exist_ok=True
                )
                file_path = (
                    Path(__file__)
                    .resolve()
                    .parent.joinpath("models")
                    .joinpath(f"{model_name}.hdf")
                )
                with urllib.request.urlopen(url) as response, open(
                    file_path, "wb"
                ) as out_file:
                    data = response.read()
                    out_file.write(data)
            else:
                check_url_exists(url)
        logging.info(f"Using the model file {file_path.absolute()}")
        return file_path


@dataclass
class Spectrograph:
    """Abstract spectrograph model

    Describes all methods that a spectrograph model must have to be used in a simulation. \n
    When subclassing, all methods need to be implemented in the subclass.

    A spectrograph model as at least one CCD (with CCD_index 1), at least one field/fiber (with fiber index 1),
    and at least one diffraction order.
    """

    name: str = "Spectrograph"

    def get_fibers(self, ccd_index: int = 1) -> list[int]:
        """Fields/fiber indices

        Args:
            ccd_index: CCD index

        Returns:
            available fields/fiber indices
        """
        raise NotImplementedError

    def get_orders(self, fiber: int = 1, ccd_index: int = 1) -> list[int]:
        """Diffraction orders

        Args:
            fiber: fiber/field index
            ccd_index: CCD index

        Returns:
            available diffraction order(s) for given indices
        """
        raise NotImplementedError

    def get_transformation(
        self,
        wavelength: float | np.ndarray,
        order: int,
        fiber: int = 1,
        ccd_index: int = 1,
    ) -> AffineTransformation | np.ndarray:
        """Transformation matrix/matrices

        Args:
            wavelength: wavelength(s) [micron]
            order: diffraction order
            fiber: fiber index
            ccd_index: CCD index

        Returns:
            transformation matrix/matrices
        """
        raise NotImplementedError

    def get_psf(
        self, wavelength: float | None, order: int, fiber: int = 1, ccd_index: int = 1
    ) -> PSF | list[PSF]:
        """PSF

        PSFs are tabulated. When wavelength is provided, the closest available PSF of the model is returned.

        When wavelength is None, all PSFs for that particular order (and fiber and CCD index) are returned.

        Args:
            wavelength: wavelength [micron] or None
            order: diffraction order
            fiber: fiber index
            ccd_index: CCD index

        Returns:
            PSF(s)
        """
        raise NotImplementedError

    def get_wavelength_range(
        self,
        order: int | None = None,
        fiber: int | None = None,
        ccd_index: int | None = None,
    ) -> tuple[float, float]:
        """Wavelength range

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
        """Get CCD object(s)

        When index is provided the corresponding CCD object is returned.\n
        If no index is provided, all available CCDs are return as a dict with the index as key.

        Args:
            ccd_index: CCD index

        Returns:
            CCD object(s)
        """
        raise NotImplementedError

    def get_field_shape(self, fiber: int, ccd_index: int) -> str:
        """Shape of field/fiber

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
        """Spectrograph efficiency

        Args:
            fiber: fiber/field index
            ccd_index: CCD index

        Returns:
            System efficiency for given indices
        """
        raise NotImplementedError

    def __eq__(self, other: Spectrograph):
        equal = True
        equal_ccd = self.get_ccd() == other.get_ccd()
        if not equal_ccd:
            print(f"CCDs are equal: {equal_ccd}")
            equal = False
        for c in self.get_ccd().keys():
            equal_fields = self.get_fibers(c) == other.get_fibers(c)
            if not equal_fields:
                print(f"Fields for CCD {c} are equal: {equal_fields}")
                equal = False
            for f in self.get_fibers(c):
                # test field shapes
                equal_field_shapes = self.get_field_shape(
                    f, c
                ) == other.get_field_shape(f, c)
                if not equal_field_shapes:
                    print(
                        f"Field shapes for CCD {c} and fiber {f} are equal: {equal_field_shapes}"
                    )
                    equal = False
                # test orders
                equal_orders = self.get_orders(f, c) == other.get_orders(f, c)
                if not equal_orders:
                    print(f"Orders for CCD {c} and fiber {f} are equal: {equal_orders}")
                    equal = False
                for o in self.get_orders(f, c):
                    # test wavelength range
                    equal_wavelength_range = self.get_wavelength_range(
                        o, f, c
                    ) == other.get_wavelength_range(o, f, c)
                    if not equal_wavelength_range:
                        print(
                            f"Wavelength range for CCD {c} and fiber {f} and order {o} are equal: {equal_wavelength_range}"
                        )
                        equal = False
                    # test transformations
                    test_wl = np.linspace(*self.get_wavelength_range(o, f, c), 100)

                    s_transf = self.get_transformation(test_wl, o, f, c)
                    o_transf = other.get_transformation(test_wl, o, f, c)
                    equal_transf = (s_transf == o_transf).all()
                    if not equal_transf:
                        print(
                            f"Transformations for CCD {c} and fiber {f} and order {o} are equal: {equal_transf}"
                        )
                        equal = False
                    # test PSFs
                    equal_psf = True
                    for wl in test_wl:
                        s_psfs = self.get_psf(wl, o, f, c)
                        o_psfs = other.get_psf(wl, o, f, c)
                        equal_psf = equal_psf and (s_psfs == o_psfs)
                    if not equal_psf:
                        print(
                            f"PSFs for CCD {c} and fiber {f} and order {o} are equal: {equal_psf}"
                        )
                        equal = False
        return equal


class SimpleSpectrograph(Spectrograph):
    """Simple spectrograph model

    This is a simple spectrograph model that can be used for testing purposes and to demonstrate how a Spectrograph
    model should be implemented.
    It has one CCD, one fiber, one order, and a simple affine transformation.
    The PSF is a Gaussian with a sigma of 3 pixels in X and 10 pixels in Y.

    """

    def __init__(self, name: str = "SimpleSpectrograph"):
        self.name = name
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

    def get_transformation(
        self,
        wavelength: float | np.ndarray,
        order: int,
        fiber: int = 1,
        ccd_index: int = 1,
    ) -> AffineTransformation | list[AffineTransformation]:
        if isinstance(wavelength, float):
            return AffineTransformation(
                0.0,
                1.0,
                10.0,
                0.0,
                (wavelength - 0.5) * wavelength * 100000.0 + 2000.0,
                fiber * 10.0 + 2000.0,
                wavelength,
            )
        else:
            ts = TransformationSet(
                [
                    AffineTransformation(
                        0.0,
                        1.0,
                        10.0,
                        0.0,
                        (w - 0.5) * w * 100000.0 + 2000.0,
                        fiber * 10.0 + 2000.0,
                        w,
                    )
                    for w in wavelength
                ]
            )
            return ts.get_affine_transformations(wavelength)

    @staticmethod
    def gauss_map(size_x, size_y=None, sigma_x=5.0, sigma_y=None):
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

        exp_part = x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)
        return 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-exp_part)

    def get_psf(
        self, wavelength: float | None, order: int, fiber: int = 1, ccd_index: int = 1
    ) -> PSF | list[PSF]:
        if wavelength is None:
            wl = np.linspace(*self.get_wavelength_range(order, fiber, ccd_index), 20)
            return [
                PSF(w, self.gauss_map(11, sigma_x=3.0, sigma_y=10.0), 1.5) for w in wl
            ]
        else:
            return PSF(wavelength, self.gauss_map(11, sigma_x=3.0, sigma_y=10.0), 1.5)

    def get_wavelength_range(
        self,
        order: int | None = None,
        fiber: int | None = None,
        ccd_index: int | None = None,
    ) -> tuple[float, float]:
        return 0.4, 0.6

    def get_ccd(self, ccd_index: int | None = None) -> CCD | dict[int, CCD]:
        if ccd_index is None:
            return self._ccd
        else:
            return self._ccd[ccd_index]

    def get_field_shape(self, fiber: int, ccd_index: int) -> str:
        return "rectangular"

    def get_efficiency(self, fiber: int, ccd_index: int) -> SystemEfficiency:
        return SystemEfficiency([ConstantEfficiency(1.0)], "System")


class AtmosphericDispersion(Spectrograph):
    """
    Atmospheric Dispersion

    Simulates atmospheric dispersion from 300nm to 2microns.
    The model does not include atmospheric transmission as an efficiency model.
    In case you want to use this, please explicitly add the atmospheric model via the simulator.

    Attributes:
        zd: zenith distance [deg]
        pressure: air pressure [Pa]
        temperature: air temperature [K]
        r_wl: undisturbed wavelength [micron]
        pixel_scale: arcsec per pixel [arcsec]
        seeing: seeing at reference_wavelength [arcsec]
        ccd_size: number of pixels in X and Y direction
    """

    def __init__(
        self,
        zd: float,
        pressure=775e2,
        temperature=283.15,
        reference_wavelength: float = 0.35,
        pixel_scale: float = 0.1,
        seeing: float = 1.0,
        ccd_size: int = 80,
        name: str = "AtmosphericDispersion",
    ):
        """Constructor

        Args:
            zd: zenith distance [deg]
            pressure: air pressue [Pa]
            temperature: air temperature [K]
            reference_wavelength: undisturbed wavelength [micron]
            pixel_scale: arcsec per pixel [arcsec]
            seeing: seeing at reference_wavelength [arcsec]
            ccd_size: number of pixels in X and Y direction for simulated output
            name: name of the spectrograph model
        """
        self.name = name
        self.ccd_size = ccd_size
        self._ccd = {1: CCD(self.ccd_size, self.ccd_size, pixelsize=1)}
        self._fibers = {}
        self._orders = {}
        self._transformations = {}
        self.pressure = pressure
        self.temperature = temperature
        self.r_wl = reference_wavelength
        self.pixel_scale = pixel_scale
        self.zd = zd
        self.seeing = seeing
        for c in self._ccd.keys():
            self._fibers[c] = [1]
            self._orders[c] = {}
            for f in self._fibers.keys():
                self._orders[c][f] = [1]

    def refractive_index(self, wl: float | np.ndarray) -> float | np.ndarray:
        """Calculates the refractive index depending on atmospheric condition and ZD

        reference:
        https://www.ls.eso.org/sci/facilities/lasilla/instruments/feros/Projects/ADC/references/refraction/index.html

        Args:
            wl: wavelength(s) [micron]

        Returns:
            refractive index at given wavelength for given conditions
        """
        p0 = 1013.25e2
        t0 = 288.15
        return (
            (
                64.328
                + 29498.1e-6 / (146e-6 - (1 / (wl * 1000.0)) ** 2)
                + 255.4e-6 / (41e-4 - (1 / (wl * 1000.0)) ** 2)
            )
            * self.pressure
            / self.temperature
            * t0
            / p0
        )

    def delta_R(self, wl: np.ndarray | float) -> float | np.ndarray:
        """Differential refraction

        Returns difference in refraction for given wavelength(s)
        Args:
            wl: wavelength(s) [micron]

        Returns:
            refraction [arcsec]
        """
        return (
            206264.80625
            / 1e6
            * ((self.refractive_index(wl) - 1) - (self.refractive_index(self.r_wl) - 1))
            * np.tan(np.deg2rad(self.zd))
        )

    def get_fibers(self, ccd_index: int = 1) -> list[int]:
        return self._fibers[ccd_index]

    def get_orders(self, fiber: int = 1, ccd_index: int = 1) -> list[int]:
        return self._orders[ccd_index][fiber]

    def get_transformation(
        self,
        wavelength: float | np.ndarray,
        order: int,
        fiber: int = 1,
        ccd_index: int = 1,
    ) -> AffineTransformation | list[AffineTransformation]:
        if isinstance(wavelength, float):
            return AffineTransformation(
                0.0,
                1.0,
                1.0,
                0.0,
                self._ccd[1].n_pix_x / 2.0,
                self._ccd[1].n_pix_y / 2.0
                - self.delta_R(wavelength) / self.pixel_scale,
                wavelength,
            )
        else:
            ts = TransformationSet(
                [
                    AffineTransformation(
                        0.0,
                        1.0,
                        1.0,
                        0.0,
                        self._ccd[1].n_pix_x / 2.0,
                        self._ccd[1].n_pix_y / 2.0 - self.delta_R(w) / self.pixel_scale,
                        w,
                    )
                    for w in wavelength
                ]
            )
            return ts.get_affine_transformations(wavelength)

    def seeing_disc_diameter(self, wl):
        """Returns seeing disc diameter in pixel for given pixel scale and seeing"""
        seeing_wl = self.seeing * (wl / self.r_wl) ** (-1.0 / 5.0)
        return (seeing_wl / self.pixel_scale) / 2.0

    def get_psf(
        self, wavelength: float | None, order: int, fiber: int = 1, ccd_index: int = 1
    ) -> PSF | list[PSF]:
        if wavelength is None:
            wl = np.linspace(*self.get_wavelength_range(order, fiber, ccd_index), 20)
            return [
                PSF(
                    w,
                    SimpleSpectrograph.gauss_map(
                        50,
                        sigma_x=self.seeing_disc_diameter(w),
                        sigma_y=self.seeing_disc_diameter(w),
                    ),
                    1.0,
                )
                for w in wl
            ]
        else:
            return PSF(
                wavelength,
                SimpleSpectrograph.gauss_map(
                    50,
                    sigma_x=self.seeing_disc_diameter(wavelength),
                    sigma_y=self.seeing_disc_diameter(wavelength),
                ),
                1.0,
            )

    def get_wavelength_range(
        self,
        order: int | None = None,
        fiber: int | None = None,
        ccd_index: int | None = None,
    ) -> tuple[float, float]:
        return 0.3, 2.0

    def get_ccd(self, ccd_index: int | None = None) -> CCD | dict[int, CCD]:
        if ccd_index is None:
            return self._ccd
        else:
            return self._ccd[ccd_index]

    def get_field_shape(self, fiber: int, ccd_index: int) -> str:
        return "singlemode"

    def get_efficiency(self, fiber: int, ccd_index: int) -> SystemEfficiency:
        return SystemEfficiency([ConstantEfficiency(1.0)], "System")


class ZEMAX(Spectrograph):
    """ZEMAX spectrograph model

    This is a spectrograph model that can be used to read in an HDF model and use it for simulations.
    """

    def __init__(self, path: str | Path, name: str = "ZEMAX Model"):
        self.name = name
        self.path = check_for_spectrograph_model(path)
        self._CCDs = {}
        self._ccd_keys = []
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
        nx = self.h5f[f"CCD_{k}"].attrs["Nx"]
        ny = self.h5f[f"CCD_{k}"].attrs["Ny"]
        ps = self.h5f[f"CCD_{k}"].attrs["pixelsize"]
        return CCD(n_pix_x=nx, n_pix_y=ny, pixelsize=ps)

    def get_fibers(self, ccd_index: int = 1) -> list[int]:
        return [int(k[6:]) for k in self.h5f[f"CCD_{ccd_index}"].keys() if "fiber" in k]

    def get_field_shape(self, fiber: int, ccd_index: int) -> str:
        if ccd_index not in self._field_shape.keys():
            self._field_shape[ccd_index] = {}
        if fiber not in self._field_shape[ccd_index].keys():
            fs = self.h5f[f"CCD_{ccd_index}/fiber_{fiber}"].attrs["field_shape"]
            self._field_shape[ccd_index][fiber] = (
                fs.decode("utf-8") if isinstance(fs, bytes) else fs
            )
        return self._field_shape[ccd_index][fiber]

    def get_orders(self, fiber: int = 1, ccd_index: int = 1) -> list[int]:
        if ccd_index not in self._orders.keys():
            self._orders[ccd_index] = {}
        if fiber not in self._orders[ccd_index].keys():
            self._orders[ccd_index][fiber] = [
                int(k[5:])
                for k in self.h5f[f"CCD_{ccd_index}/fiber_{fiber}/"].keys()
                if "psf" not in k
            ]
            self._orders[ccd_index][fiber].sort()
        return self._orders[ccd_index][fiber]

    def transformations(
        self, order: int, fiber: int = 1, ccd_index: int = 1
    ) -> list[AffineTransformation]:
        if ccd_index not in self._transformations.keys():
            self._transformations[ccd_index] = {}
        if fiber not in self._transformations[ccd_index].keys():
            self._transformations[ccd_index][fiber] = {}
        if order not in self._transformations[ccd_index][fiber].keys():
            try:
                self._transformations[ccd_index][fiber][order] = [
                    AffineTransformation(*af)
                    for af in self.h5f[f"CCD_{ccd_index}/fiber_{fiber}/order{order}"][
                        ()
                    ]
                ]
                self._transformations[ccd_index][fiber][order].sort()

            except KeyError:
                raise KeyError(
                    f"You asked for the affine transformation matrices in diffraction order {order}. "
                    f"But this data is not available"
                )

        return self._transformations[ccd_index][fiber][order]

    def spline_transformations(
        self, order: int, fiber: int = 1, ccd_index: int = 1
    ) -> TransformationSet:
        if ccd_index not in self._spline_transformations.keys():
            self._spline_transformations[ccd_index] = {}
        if fiber not in self._spline_transformations[ccd_index].keys():
            self._spline_transformations[ccd_index][fiber] = {}
        if order not in self._spline_transformations[ccd_index][fiber].keys():
            tfs = self.transformations(order, fiber, ccd_index)
            self._spline_transformations[ccd_index][fiber][order] = TransformationSet(
                tfs
            )
        return self._spline_transformations[ccd_index][fiber][order]

    def get_transformation(
        self,
        wavelength: float | np.ndarray,
        order: int,
        fiber: int = 1,
        ccd_index: int = 1,
    ) -> AffineTransformation | list[AffineTransformation] | np.ndarray:
        return self.spline_transformations(
            order, fiber, ccd_index
        ).get_affine_transformations(wavelength)

    def psfs(self, order: int, fiber: int = 1, ccd_index: int = 1) -> list[PSF]:
        if ccd_index not in self._psfs.keys():
            self._psfs[ccd_index] = {}
        if order not in self._psfs[ccd_index].keys():
            try:
                self._psfs[ccd_index][order] = [
                    PSF(
                        self.h5f[
                            f"CCD_{ccd_index}/fiber_{fiber}/psf_order_{order}/{wl}"
                        ].attrs["wavelength"],
                        self.h5f[
                            f"CCD_{ccd_index}/fiber_{fiber}/psf_order_{order}/{wl}"
                        ][()],
                        self.h5f[
                            f"CCD_{ccd_index}/fiber_{fiber}/psf_order_{order}/{wl}"
                        ].attrs["dataSpacing"],
                    )
                    for wl in self.h5f[
                        f"CCD_{ccd_index}/fiber_{fiber}/psf_order_{order}"
                    ]
                ]

            except KeyError:
                raise KeyError(
                    f"You asked for the PSFs in diffraction order {order}. But this data is not available"
                )
            self._psfs[ccd_index][order].sort()
        return self._psfs[ccd_index][order]

    def get_psf(
        self, wavelength: float | None, order: int, fiber: int = 1, ccd_index: int = 1
    ) -> PSF | list[PSF]:
        if wavelength is None:
            return self.psfs(order, fiber, ccd_index)
        else:
            # find the nearest PSF:
            idx = min(
                range(len(self.psfs(order, fiber, ccd_index))),
                key=lambda i: abs(
                    self.psfs(order, fiber, ccd_index)[i].wavelength - wavelength
                ),
            )
            return self.psfs(order, fiber, ccd_index)[idx]

    def get_wavelength_range(
        self,
        order: int | None = None,
        fiber: int | None = None,
        ccd_index: int | None = None,
    ) -> tuple[float, float]:
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
            self._ccd_keys = [int(k[4:]) for k in self.h5f["/"].keys() if "CCD" in k]
        return self._ccd_keys

    def get_ccd(self, ccd_index: int | None = None) -> CCD | dict[int, CCD]:
        if ccd_index is None:
            return dict(
                zip(
                    self.available_ccd_keys(),
                    [self._read_ccd_from_hdf(k) for k in self.available_ccd_keys()],
                )
            )

        if ccd_index not in self._CCDs:
            self._CCDs[ccd_index] = self._read_ccd_from_hdf(ccd_index)
        return self._CCDs[ccd_index]

    def get_efficiency(self, fiber: int, ccd_index: int) -> SystemEfficiency:
        try:
            ge = GratingEfficiency(
                self.h5f[f"CCD_{ccd_index}/Spectrograph"].attrs["blaze"],
                self.h5f[f"CCD_{ccd_index}/Spectrograph"].attrs["blaze"],
                self.h5f[f"CCD_{ccd_index}/Spectrograph"].attrs["gpmm"],
            )
        except KeyError:
            logging.warning(
                "No information about the blaze and other grating parameters are found in the .hdf file."
            )
            ge = ConstantEfficiency("Spectrograph", eff=1.0)

        if ccd_index not in self._efficiency.keys():
            self._efficiency[ccd_index] = {}
        if fiber not in self._efficiency[ccd_index].keys():
            try:
                self._efficiency[ccd_index][fiber] = SystemEfficiency(
                    [
                        ge,
                        TabulatedEfficiency(
                            "System",
                            *self.h5f[f"CCD_{ccd_index}/fiber_{fiber}"].attrs[
                                "efficiency"
                            ],
                        ),
                    ],
                    "System",
                )

            except KeyError:
                logging.warning(
                    f"No spectrograph efficiency data found for fiber {fiber}."
                )
                self._efficiency[ccd_index][fiber] = SystemEfficiency([ge], "System")
        return self._efficiency[ccd_index][fiber]

    def __exit__(self):
        if self._h5f:
            self._h5f.close()

    def __str__(self):
        return self.name + f": {self.path.name}"


FieldPoint = namedtuple("FieldPoint", ["x", "y"])


@dataclass
class Field:
    """
    Spectrograph field/fiber

    Used to simplify field specifications in ZEMAX. Handles the conversion from 'normal' coordinates to 'normalized'
    coordinates as required by ZEMAX.
    """

    center: FieldPoint
    field_size: tuple[float, float]
    name: str
    shape: str = "circle"

    def __post_init__(self):
        self.points = [
            self.center,
            FieldPoint(
                self.center.x + self.field_size[0] / 2.0,
                self.center.y + self.field_size[1] / 2.0,
            ),
            FieldPoint(
                self.center.x - self.field_size[0] / 2.0,
                self.center.y + self.field_size[1] / 2.0,
            ),
            FieldPoint(
                self.center.x - self.field_size[0] / 2.0,
                self.center.y - self.field_size[1] / 2.0,
            ),
            FieldPoint(
                self.center.x + self.field_size[0] / 2.0,
                self.center.y - self.field_size[1] / 2.0,
            ),
        ]

        self.normalized_points = [
            FieldPoint(p.x / self.maxX, p.y / self.maxY) for p in self.points
        ]

    @property
    def maxX(self):
        return np.max(np.abs(self.points)[:, 0])

    @property
    def maxY(self):
        return np.max(np.abs(self.points)[:, 1])

    def push_to_zos(self, oss: OpticStudioSystem):
        """
        Deletes all current fields in the ZEMAX system and replaces them with the fives
        points defining the current field.
        Args:
            oss: OpticStudioSystem design

        Returns:

        """
        oss.SystemData.Fields.DeleteAllFields()
        oss.SystemData.Fields.Normalization = (
            zospy.constants.SystemData.FieldNormalizationType.Rectangular
        )
        for p in self.points[
            1:
        ]:  # skip the [0., 0.] field, because it is not deleted by DeleteAllFields() !
            oss.SystemData.Fields.AddField(p.x, p.y, 1.0)


class InteractiveZEMAX(Spectrograph):
    """
    Interactive ZEMAX

    Class that interacts with ZEMAX and pulls the required information from the ZEMAX model.
    Can be used in conjunction with HDFBuilder to generate an .hdf spectrograph model file.

    You can connect to ZEMAX either via an interactive extension to a running instance
    (make sure you've activated to accept connections from extensions in OpticStudio) or
    via a standalone process. For performance reasons, the latter is recommended.

    """

    def __init__(self, name: str, zemax_filepath: str | Path | None = None):
        """
        Constructor

        Args:
            name: name of the spectrograph
            zemax_filepath: path to .zmx / .zos spectrograph file. If None, a connection to a running OpticStudio
             session via an extension is attempted (make sure to accept external connection in OpticStudio).

        """
        self.name = name
        self._fields: list[Field] = []
        self._ccd = {}
        self._orders = {}
        self.logger = logging.getLogger("InteractiveZEMAX")
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        self._zos = zospy.ZOS()
        self._zos.wakeup()
        if zemax_filepath is None:
            self._zos.connect_as_extension()
            self._oss = self._zos.get_primary_system()
        else:
            if isinstance(zemax_filepath, str):
                zemax_filepath = Path(zemax_filepath)
                if not zemax_filepath.is_file():
                    raise FileNotFoundError("ZEMAX file is not found.")
            self._zos.create_new_application()
            self._oss = self._zos.get_primary_system()
            self._oss.load(str(zemax_filepath))

        self._lde = self._oss.LDE

        self._zmx_grating = None
        self._blaze = None  # blaze angle in degrees
        self._theta = None  # theta angles in degrees
        self._gamma = None  # gamma angles in degrees
        self._groves_per_micron = None  # groves per micron as in ZEMAX
        self._current_order = None  # current diffraction order
        self._current_field = None

        self._psf_setting_sampling_image = "64x64"
        self._psf_setting_sampling_pupil = "64x64"
        self._psf_setting_image_delta = 0.25

    def psf_settings(
        self,
        image_sampling: str = "64x64",
        pupil_sampling="64x64",
        image_delta: float = 0.25,
    ):
        """Globally sets PSF settings in ZEMAX

        Sets image/pupil sampling and data spacing / image delta of PSF in ZEMAX.
        Note: the PSF image has then a total size of image_sampling * image_delta (in microns).
        Make sure the PSF image_sampling is 'large enough' to not artificially cut the PSF wings.

        Args:
            image_sampling (str): image sampling of PSF. Must be "32x32", "64x64", "128x128", "256x256", "512x512",
                                  "1024x1024" or "2048x2048"
            pupil_sampling (str): pupil sampling of PSF. Must be "32x32", "64x64", "128x128", "256x256", "512x512",
                                  "1024x1024" or "2048x2048"
            image_delta (float): image delta (step size) in microns

        Returns:

        """
        available_modes = [
            "32x32",
            "64x64",
            "128x128",
            "256x256",
            "512x512",
            "1024x1024",
            "2048x2048",
        ]
        assert (
            image_sampling in available_modes
        ), f"image_sampling must be in {available_modes}"
        assert (
            pupil_sampling in available_modes
        ), f"pupil_sampling must be in {available_modes}"

        self._psf_setting_sampling_image = image_sampling
        self._psf_setting_sampling_pupil = pupil_sampling
        self._psf_setting_image_delta = image_delta

    def _zos_set_grating(self, surface: int | str = "Echelle"):
        if isinstance(surface, str):
            surface_name = surface
            surface = find_surface_by_comment(self._lde, surface)[0].SurfaceNumber
            self.logger.info(
                f'Found Echelle grating, named "{surface_name}" at surface {surface}'
            )

        self._zmx_grating = self._lde.GetSurfaceAt(surface)
        self._groves_per_micron = self._zmx_grating.GetCellAt(
            12
        ).DoubleValue  # groves per micron as in ZEMAX
        self._current_order = self._zmx_grating.GetCellAt(
            13
        )  # current diffraction order
        self._order_sign = -1 if self._current_order.DoubleValue < 0 else 1

    def _zos_set_current_order(self, order: int):
        """Set current diffraction order

        Args:
            order (int): Set current diffraction order. Signs are ignored.

        Returns:

        """
        self._current_order.DoubleValue = self._order_sign * abs(order)

    def _blaze_wl(self, order: int = None) -> float:
        """Blaze wavelength [nm]

        Args:
            order (int): diffraction order, if None, self._current_order will be used

        Returns:
            blaze wavelength [nm]
        """
        if order is None:
            order = self._current_order.DoubleValue
        alpha_rad = np.deg2rad(self._blaze) + np.deg2rad(self._theta)
        beta_rad = np.deg2rad(self._blaze) - np.deg2rad(self._theta)
        grp = 1.0 / self._groves_per_micron
        c0 = grp * np.cos(np.deg2rad(self._gamma))
        c1 = c0 * (np.sin(alpha_rad) + np.sin(beta_rad))
        return abs(c1 / order)

    def _FSR(self, order: int = None) -> tuple[float, float]:
        """Free spectral range

        Args:
            order (int): diffraction order, if None, self._current_order is used

        Returns:
            lower and upper limit of free spectral range
        """
        if order is None:
            order = self._current_order.DoubleValue
        bwl = self._blaze_wl(order)
        wl1 = bwl - bwl / order / 2.0
        wl2 = bwl + bwl / order / 2.0

        return min(wl1, wl2), max(wl1, wl2)

    def _zos_walk_to_detector_edge(self, direction: int = -1, initial_step=0.001):
        """
        Walks in wavelength to the edge of the detector to determine boundary wavelengths of current diffraction order.
        Args:
            direction (int): Either 1 or -1. Defines direction in which wavelength are tested (1 -> larger wavelengths)
            initial_step (float): initial step size (default=1nm)
        Returns:

        """
        assert direction == 1 or direction == -1, "direction needs to be 1 or -1"
        n_surf = self._oss.LDE.NumberOfSurfaces - 1
        c_wl = self._oss.SystemData.Wavelengths.GetWavelength(1)

        backup_wl = c_wl.Wavelength

        vignetted = False  # initially not vignetted
        step = initial_step  # initial step is 1 nm
        while (not vignetted) or (step > 0.000001):
            if vignetted:
                self._oss.SystemData.Wavelengths.GetWavelength(1).Wavelength += (
                    -step + step / 5.0
                ) * direction
                step /= 5.0
            else:
                self._oss.SystemData.Wavelengths.GetWavelength(1).Wavelength += (
                    step * direction
                )
            raytrace = self._oss.Tools.OpenBatchRayTrace()
            [
                success,
                error,
                vignetted,
                xo,
                yo,
                zo,
                lo,
                mo,
                no,
                l20,
                m20,
                n20,
                opd,
                intensity,
            ] = raytrace.SingleRayNormUnpol(
                zospy.constants.Tools.RayTrace.RaysType.Real,
                -1,
                1,
                self._fields[self._current_field - 1].normalized_points[0].x,
                self._fields[self._current_field - 1].normalized_points[0].y,
                0,
                0,
                False,
            )
            raytrace.RunAndWaitForCompletion()
            raytrace.Close()
            if error or not success:
                raise ValueError(
                    f"There was an error while tracing wavelength "
                    f"{self._oss.SystemData.Wavelengths.GetWavelength(1).Wavelength} micron. "
                    f"Probably, something with the ZEMAX file is wrong."
                )
        if not vignetted == n_surf:
            self.logger.warning(
                f"WARNING: Vignetting occurred at surface {vignetted} for wavelength {c_wl.Wavelength} "
                f"and order {self._current_order.DoubleValue}"
            )
        vignetted_wavelength = c_wl.Wavelength
        c_wl.Wavelength = backup_wl
        return vignetted_wavelength

    def _check_ccd(self, ccd_index: int):
        """Checks CCD size for consistency with ZEMAX file

        Args:
            ccd_index (int): CCD index

        Returns:

        """
        zos_ccd = self._lde.GetSurfaceAt(self._lde.NumberOfSurfaces)
        xw = zos_ccd.ApertureData.CurrentTypeSettings._S_RectangularAperture.XHalfWidth
        yw = zos_ccd.ApertureData.CurrentTypeSettings._S_RectangularAperture.YHalfWidth

        assert (
            self._ccd[ccd_index].n_pix_x * self._ccd[ccd_index].pixelsize / 2000.0 <= xw
        ), (
            f"Your CCD specification of {self._ccd[ccd_index].n_pix_x} pix with {self._ccd[ccd_index].pixelsize} "
            f"micron pixel size "
            f"(={self._ccd[ccd_index].n_pix_x * self._ccd[ccd_index].pixelsize / 2000.} mm CCD half width) is "
            f"larger than the rectangular aperture in the ZEMAX file ({xw}mm). Please set the aperture in the"
            f" ZEMAX file correctly, or adjust the CCD specifications"
        )

        assert (
            self._ccd[ccd_index].n_pix_y * self._ccd[ccd_index].pixelsize / 2000.0 <= yw
        ), (
            f"Your CCD specification of {self._ccd[ccd_index].n_pix_y} pix with {self._ccd[ccd_index].pixelsize} "
            f"micron pixel size "
            f"(={self._ccd[ccd_index].n_pix_y * self._ccd[ccd_index].pixelsize / 2000.} mm CCD half width) are "
            f"is larger than "
            f"the rectangular aperture in the ZEMAX file ({yw}mm). Please set the aperture in the ZEMAX file "
            f"correctly, or adjust the CCD specifications"
        )

    def set_grating(
        self,
        surface: int | str = "Echelle",
        blaze: float = None,
        theta: float = 0,
        gamma: float = 0,
    ):
        """Sets grating specification

        Defines the grating specifications incl. all relevant optical parameter.

        Args:
            surface: ZEMAX grating surface number (or when a str is passed, the surface will be searched by comment)
            blaze: blaze angle [deg]
            theta: theta angle [deg]
            gamma: gamma angle [deg]

        Returns:

        """
        assert blaze is not None, "Blaze angle needs to be specified."
        self._zos_set_grating(surface)
        self._blaze = blaze
        self._theta = theta
        self._gamma = gamma

    def add_ccd(self, ccd_idx: int, ccd: CCD):
        if not self._ccd:
            self._ccd.update({ccd_idx: ccd})
        else:
            raise ValueError(
                "For the interactive ZEMAX model, you can't add more than one CCD object."
            )
        self._check_ccd(ccd_idx)

    def _check_field(self):
        pass

    def add_field(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        shape: str = "circular",
        name: str | None = None,
    ):
        """Add field/fiber to ZEMAX model

        Adds a field/fiber point to the current ZEMAX model. When specifying 'singlemode' as field type,
        use a small width and height for the field (e.g. 5 micron). The actual size will later be ignored,
        but to calculate the transformation matrices, it is still important, that we use a finite size.

        Args:
            x (float): x field coordinate as in ZEMAX [mm]
            y (float): y field coordinate as in ZEMAX [mm]
            width (float): width of the field [microns]
            height (float): height of the field [microns]
            shape (str): shape of the fiber, e.g. 'circular', 'octagonal', 'hexagonal', 'rectangular' or 'singlemode'.
            name (str): (Optional) name of the field/fiber

        Returns:

        """
        if name is None:
            name = f"field_{len(self._fields) + 1}"
        self._fields.append(
            Field(FieldPoint(x, y), (width / 1000.0, height / 1000.0), name, shape)
        )
        self._check_field()

    def get_fibers(self, ccd_index: int = 1) -> list[int]:
        return [i + 1 for i in range(len(self._fields))]

    def set_orders(self, ccd_index: int, field: int, orders: list[int]):
        if ccd_index not in self._orders.keys():
            self._orders[ccd_index] = {}
        self._orders[ccd_index][field] = orders

    def get_orders(self, fiber: int = 1, ccd_index: int = 1) -> list[int]:
        return self._orders[ccd_index][fiber]

    def _zos_set_current_field(self, field: int = 1):
        if self._current_field is None or self._current_field != field:
            self._fields[field - 1].push_to_zos(self._oss)
            self._current_field = field

    def get_transformation(
        self,
        wavelength: float | np.ndarray,
        order: int,
        fiber: int = 1,
        ccd_index: int = 1,
    ) -> AffineTransformation | list[AffineTransformation]:
        assert (
            ccd_index == 1
        ), "In the interactive mode, there is only one CCD index supported"
        self._zos_set_current_field(fiber)
        self._zos_set_current_order(order)

        self.logger.debug(
            f"Get transformation matrix for {wavelength=}, {order=}, {fiber=}"
        )
        if isinstance(wavelength, float):
            single_wavelength = True
            wavelength = [wavelength]
        else:
            single_wavelength = False
        at = []
        for wl in wavelength:
            self._oss.SystemData.Wavelengths.GetWavelength(1).Wavelength = wl
            ins = []
            out = []
            for norm_point in self._fields[fiber - 1].normalized_points:
                norm_x, norm_y = norm_point
                ins.append([norm_x, norm_y])

                # Pull the operand value. This will pull the value without affecting the Merit Function
                x = self._oss.MFE.GetOperandValue(
                    zospy.constants.Editors.MFE.MeritOperandType.REAX,
                    self._oss.LDE.NumberOfSurfaces,
                    1,
                    norm_x,
                    norm_y,
                    0,
                    0,
                    0,
                    0,
                )
                y = self._oss.MFE.GetOperandValue(
                    zospy.constants.Editors.MFE.MeritOperandType.REAY,
                    self._oss.LDE.NumberOfSurfaces,
                    1,
                    norm_x,
                    norm_y,
                    0,
                    0,
                    0,
                    0,
                )
                out.append([x, y])
            ins = np.array(ins)

            ins[:, 0] -= np.min(ins[:, 0])
            ins[:, 1] -= np.min(ins[:, 1])

            ins[:, 0] /= np.max(ins[:, 0])
            ins[:, 1] /= np.max(ins[:, 1])

            out = np.array(out)
            out /= self.get_ccd(1).pixelsize / 1000.0
            out += self.get_ccd(1).n_pix_x / 2.0

            at.append(
                AffineTransformation(
                    *decompose_affine_matrix(find_affine(ins, out)), wavelength=wl
                )
            )
        if single_wavelength:
            return at[0]
        else:
            ts = TransformationSet(at)
            return ts.get_affine_transformations(wavelength)

    def get_psf(
        self, wavelength: float | None, order: int, fiber: int = 1, ccd_index: int = 1
    ) -> PSF | list[PSF]:
        self.logger.debug(f"Retrieve PSF at {wavelength=},{order=}, {fiber=}")
        self._zos_set_current_field(fiber)
        self._zos_set_current_order(order)
        if isinstance(wavelength, float):
            self._oss.SystemData.Wavelengths.GetWavelength(1).Wavelength = wavelength
            psf_data = zospy.analyses.psf.huygens_psf(
                self._oss,
                pupil_sampling=self._psf_setting_sampling_pupil,
                image_sampling=self._psf_setting_sampling_image,
                image_delta=self._psf_setting_image_delta,
                wavelength=1,
                field=1,
                oncomplete="Close",
            )
            psf = PSF(
                wavelength=wavelength,
                data=psf_data.Data.values,
                sampling=self._psf_setting_image_delta,
            )
            if not psf.check(threshold=1e-3):
                self.logger.warning(
                    f"PSF at {wavelength=}, {order=} for field/fiber {fiber=} has more than 1E-3 flux "
                    f"at its border. Consider to increase sampling."
                )
            return psf
        else:
            """TODO: check wavelength range of current order, then do N psfs """
            raise NotImplementedError

    def get_wavelength_range(
        self,
        order: int | None = None,
        fiber: int | None = None,
        ccd_index: int | None = None,
    ) -> tuple[float, float]:
        self.logger.debug(f"Get wavelength range for {order=}, {fiber=}")
        if order is None:
            raise NotImplementedError("This is not yet implemented.")
        else:
            self._zos_set_current_order(order)
            self._zos_set_current_field(fiber)
            c_wl = self._oss.SystemData.Wavelengths.GetWavelength(1)
            c_wl.Wavelength = self._blaze_wl()
            max_wl = self._zos_walk_to_detector_edge(1)
            min_wl = self._zos_walk_to_detector_edge(-1)
            self.logger.info(
                f"Wavelength boundaries for order {self._current_order.DoubleValue}: "
                f"{min_wl} - {max_wl} micron"
            )
            return min_wl, max_wl

    def get_ccd(self, ccd_index: int | None = None) -> CCD | dict[int, CCD]:
        if ccd_index is None:
            return self._ccd
        else:
            return self._ccd[ccd_index]

    def get_field_shape(self, fiber: int, ccd_index: int) -> str:
        return self._fields[fiber - 1].shape

    def get_efficiency(self, fiber: int, ccd_index: int) -> SystemEfficiency:
        pass


class LocalDisturber(Spectrograph):
    """Local Disturber

    Class that adds a local disturbance to a spectrograph. The disturbance is defined by a 2D affine transformation
    matrix. The disturbance is added to the transformation matrix of the spectrograph.

    The disturbance is added locally, which means that for instance a given rotation would rotate the fiber/field
    inplace around the center of the field. This is different to a global disturbance, where the rotation would be
    around the center of the CCD.
    See also [Perturbations](https://stuermer.gitlab.io/pyechelle/models.html#perturbations) for more information.
    """

    def __init__(
        self,
        spec: Spectrograph,
        d_tx=0.0,
        d_ty=0.0,
        d_rot=0.0,
        d_shear=0.0,
        d_sx=0.0,
        d_sy=0.0,
        name: str = "LocalDisturber",
    ):
        self.spec = spec
        self.name = name
        for method in dir(Spectrograph):
            if method.startswith("get_") and method != "get_transformation":
                setattr(self, method, getattr(self.spec, method))
        self.disturber_matrix = AffineTransformation(
            d_rot, d_sx, d_sy, d_shear, d_tx, d_ty, None
        )

    def get_transformation(
        self,
        wavelength: float | np.ndarray,
        order: int,
        fiber: int = 1,
        ccd_index: int = 1,
    ) -> AffineTransformation | np.ndarray:
        if isinstance(wavelength, float):
            return (
                self.spec.get_transformation(wavelength, order, fiber, ccd_index)
                + self.disturber_matrix
            )
        else:
            return self.spec.get_transformation(
                wavelength, order, fiber, ccd_index
            ) + np.expand_dims(self.disturber_matrix.as_matrix(), axis=-1)

    def __str__(self):
        return (
            self.name + f"({str(self.spec)}): d_tx:{self.disturber_matrix.tx},"
            f"d_ty:{self.disturber_matrix.ty},"
            f"d_rot:{self.disturber_matrix.rot},"
            f"d_shear:{self.disturber_matrix.shear},"
            f"d_sx:{self.disturber_matrix.sx},"
            f"d_sy:{self.disturber_matrix.sy}"
        )


class GlobalDisturber(Spectrograph):
    """Global Disturber

    Class that adds a global disturbance to a spectrograph. The disturbance is defined by a 2D affine transformation
    matrix. The disturbance is added to the transformation matrix of the spectrograph.

    The disturbance acts on the positions of the traced fibers/fields on the CCD.
    See also [Perturbations](https://stuermer.gitlab.io/pyechelle/models.html#perturbations) for more information.
    """

    def __init__(
        self,
        spec: Spectrograph,
        tx: float = 0.0,
        ty: float = 0.0,
        rot: float = 0.0,
        shear: float = 0.0,
        sx: float = 1.0,
        sy: float = 1.0,
        reference_x: float = None,
        reference_y: float = None,
        name: str = "GlobalDisturber",
    ):
        self.spec = spec
        self.name = name
        for method in dir(Spectrograph):
            if method.startswith("get_") and method != "get_transformation":
                setattr(self, method, getattr(self.spec, method))
        self.disturber_matrix = AffineTransformation(rot, sx, sy, shear, tx, ty, None)
        self.ref_x = reference_x
        self.ref_y = reference_y

    def _get_transformation_matrix(self, dx, dy, wavelength):
        if isinstance(wavelength, float):
            return AffineTransformation(0.0, 1.0, 1.0, 0.0, dx, dy, wavelength)
        else:
            assert isinstance(wavelength, np.ndarray) or isinstance(wavelength, list)
            n_wavelength = len(wavelength)
            return np.array(
                [
                    [0.0] * n_wavelength,
                    [1.0] * n_wavelength,
                    [1.0] * n_wavelength,
                    [0.0] * n_wavelength,
                    [dx] * n_wavelength,
                    [dy] * n_wavelength,
                ]
            )

    def _get_disturbance_matrix(self, wavelength):
        if isinstance(wavelength, float):
            return AffineTransformation(
                self.disturber_matrix.rot,
                self.disturber_matrix.sx,
                self.disturber_matrix.sy,
                self.disturber_matrix.shear,
                0.0,
                0.0,
                wavelength,
            )
        else:
            assert isinstance(wavelength, np.ndarray) or isinstance(wavelength, list)
            n_wavelength = len(wavelength)
            return np.array(
                [
                    [self.disturber_matrix.rot] * n_wavelength,
                    [self.disturber_matrix.sx] * n_wavelength,
                    [self.disturber_matrix.sy] * n_wavelength,
                    [self.disturber_matrix.shear] * n_wavelength,
                    [0.0] * n_wavelength,
                    [0.0] * n_wavelength,
                ]
            )

    def get_transformation(
        self,
        wavelength: float | np.ndarray,
        order: int,
        fiber: int = 1,
        ccd_index: int = 1,
    ) -> AffineTransformation | np.ndarray:
        # by default take center of CCD as reference point
        w, h = self.spec.get_ccd(ccd_index).data.shape
        w /= 2.0
        h /= 2.0

        if self.ref_x is not None:
            w = self.ref_x
        if self.ref_y is not None:
            h = self.ref_y

        if isinstance(wavelength, float):
            tm = self.spec.get_transformation(wavelength, order, fiber, ccd_index)
            xy = tm.tx, tm.ty
            # affine transformation to shift origin to center of image
            tm_trans = self._get_transformation_matrix(
                -w / 2.0, -h / 2.0, tm.wavelength
            )
            xy = tm_trans * xy
            # affine transformation to rotate/shear/scale
            tm_trans = self._get_disturbance_matrix(wavelength)
            xy = tm_trans * xy
            # affine transformation to shift origin back
            tm_trans = AffineTransformation(
                0.0, 1.0, 1.0, 0.0, w / 2.0, h / 2.0, tm.wavelength
            )
            xy = tm_trans * xy
            tm.tx = xy[0] + self.disturber_matrix.tx
            tm.ty = xy[1] + self.disturber_matrix.ty
            return tm
        else:
            tm = self.spec.get_transformation(wavelength, order, fiber, ccd_index)
            xy = tm[4:6].T
            # affine transformation to shift origin to center of image
            tm_trans = convert_matrix(
                self._get_transformation_matrix(-w / 2.0, -h / 2.0, wavelength)
            )
            xy = np.array([apply_matrix(c, p) for c, p in zip(tm_trans.T, xy)])

            tm_trans = convert_matrix(self._get_disturbance_matrix(wavelength))
            xy = np.array([apply_matrix(c, p) for c, p in zip(tm_trans.T, xy)])

            # affine transformation to shift origin back
            tm_trans = convert_matrix(
                self._get_transformation_matrix(w / 2.0, h / 2.0, wavelength)
            )
            xy = np.array([apply_matrix(c, p) for c, p in zip(tm_trans.T, xy)])
            tm[4:6] = xy.T
            tm[4] += self.disturber_matrix.tx
            tm[5] += self.disturber_matrix.ty
            return tm


def show_fields(fields: list[Field]):
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 10))

    for i, f in enumerate(fields):
        color = plt.cm.tab10(i)
        ax.scatter(*f.center, label=f.name, c=color)

        if f.shape == "rectangular":
            rec = Rectangle(
                (f.center.x - f.field_size[0] / 2.0, f.center.y - f.field_size[1] / 2),
                f.field_size[0],
                f.field_size[1],
                fill=False,
                color=color,
            )
            ax.add_patch(rec)

        else:
            raise NotImplementedError("Field shape not implemented for plotting")
    # Set the x and y limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim((min(xlim[0], ylim[0]), max(xlim[1], ylim[1])))
    ax.set_ylim((min(xlim[0], ylim[0]), max(xlim[1], ylim[1])))

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # draw lines
    for i, f in enumerate(fields):
        color = plt.cm.tab10(i)
        # draw center lines
        ax.hlines(f.center.y, *ylim, linestyle="--", colors="k", linewidths=0.5)
        ax.vlines(f.center.x, *xlim, linestyle="--", colors="k", linewidths=0.5)

        ax.hlines(
            f.center.y + f.field_size[1] / 2 + 10,
            f.center.x - f.field_size[0] / 2,
            f.center.x + f.field_size[0] / 2,
            linestyle="--",
            colors=color,
            linewidths=0.5,
        )
        ax.vlines(
            f.center.x + f.field_size[0] / 2 + 10,
            f.center.y - f.field_size[1] / 2,
            f.center.y + f.field_size[1] / 2,
            linestyle="--",
            colors=color,
            linewidths=0.5,
        )

        # draw extent
        ax.text(
            f.center.x,
            f.center.y + f.field_size[1] / 2 + 8,
            f"{f.field_size[0]:.1f}",
            backgroundcolor="white",
            fontsize=7,
            ha="center",
            va="bottom",
            color=color,
        )
        ax.text(
            f.center.x + f.field_size[0] / 2 + 8,
            f.center.y,
            f"{f.field_size[1]:.1f}",
            rotation="vertical",
            backgroundcolor="white",
            fontsize=7,
            ha="left",
            va="center",
            color=color,
        )
        ax.text(xlim[0] + 10, f.center.y, f"{f.center.y:.1f}", va="bottom", color=color)

    plt.xlabel("X [micron]")
    plt.ylabel("Y [micron]")

    plt.legend()
    plt.show()
