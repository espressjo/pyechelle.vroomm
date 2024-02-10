from __future__ import annotations

import contextlib
from pathlib import Path

import h5py
import numpy as np

from pyechelle.CCD import CCD
from pyechelle.optics import correct_phase_jumps
from pyechelle.spectrograph import Spectrograph


class HDFBuilder(contextlib.ExitStack):
    """HDFBuilder class to generate .hdf model files

    This class is used to generate valid .hdf spectrograph model files from any valid Spectrograph class object.
    Main purpose of this class is to extract an .hdf spectrograph model from e.g. an InteractiveZEMAX model to speed up
    simulation time and also cut the ZEMAX dependency.

    Attributes:
        spec: Spectrograph model
        path: file path for .hdf output
        mode: whether to append or overwrite the .hdf file

    Examples:
        .. highlight:: python
        .. code-block:: python

            import sys
            spec = InteractiveZEMAX()
            HDFBuilder(spec, 'awesome_spectrograph.hdf', 'w')
            HDFBuilder.save_to_hdf(100, 15)

    """

    def __init__(self, spec: Spectrograph, path: Path | str, mode: str = "w"):
        super().__init__()
        self.mode = mode
        self.spec = spec
        self.path = path
        self.h5f = self.enter_context(h5py.File(path, self.mode))

    def save_to_hdf(
        self, n_transformation_per_order: int = 50, n_psfs_per_order: int = 15
    ):
        for ccd_idx, ccd in self.spec.get_ccd().items():
            self._save_spectrograph_info_to_hdf(ccd_number=ccd_idx)
            self._save_ccd_info_to_hdf(ccd, ccd_number=ccd_idx)
            for f in self.spec.get_fibers(ccd_idx):
                self._save_field_shape_to_hdf(
                    ccd_idx, f, self.spec.get_field_shape(f, ccd_idx)
                )
                for o in self.spec.get_orders(f, ccd_idx):
                    wavelength_range = self.spec.get_wavelength_range(o, f, ccd_idx)
                    wls = np.linspace(
                        *wavelength_range,
                        num=n_transformation_per_order,
                    )
                    transformations = self.spec.get_transformation(wls, o, f, ccd_idx).T
                    transformations[:, 3] = correct_phase_jumps(transformations[:, 3])
                    self._save_transformation_to_hdf(
                        wls,
                        transformations,
                        order=o,
                        fiber_number=f,
                        ccd_number=ccd_idx,
                    )

                    wls_psf = np.linspace(
                        *wavelength_range,
                        num=n_psfs_per_order,
                    )
                    psfs = [self.spec.get_psf(wl, o, f, ccd_idx) for wl in wls_psf]
                    self._save_psfs_to_hdf(wls_psf, psfs, f, o, ccd_idx)

    def _save_field_shape_to_hdf(
        self, ccd_number: int, fiber_number: int, field_shape: str
    ):
        try:
            fiber_group = self.h5f.require_group(
                f"CCD_{ccd_number}/fiber_{fiber_number}"
            )
        except TypeError:
            fiber_group = self.h5f.create_group(
                f"CCD_{ccd_number}/fiber_{fiber_number}"
            )

        fiber_group.attrs.create("field_shape", field_shape)

    def _save_ccd_info_to_hdf(self, ccd: CCD, ccd_number: int = 1):
        try:
            ccd_group = self.h5f.require_group(f"CCD_{ccd_number}")
        except TypeError:
            ccd_group = self.h5f.create_group(f"CCD_{ccd_number}")

        ccd_group.attrs.create("Nx", ccd.n_pix_x)
        ccd_group.attrs.create("Ny", ccd.n_pix_y)
        ccd_group.attrs.create("pixelsize", ccd.pixelsize)

    def _save_spectrograph_info_to_hdf(self, ccd_number: int = 1):
        try:
            spectrograph_group = self.h5f.require_group(
                f"CCD_{ccd_number}/Spectrograph"
            )
        except TypeError:
            spectrograph_group = self.h5f.create_group(f"CCD_{ccd_number}/Spectrograph")

        # TODO: handle blaze etc. better
        spectrograph_group.attrs.create("name", self.spec.name)
        try:
            spectrograph_group.attrs.create("blaze", self.spec._blaze)
        except AttributeError:
            pass

        try:
            spectrograph_group.attrs.create(
                "gpmm", self.spec._groves_per_micron * 1000.0
            )
        except AttributeError:
            pass

    def _save_psfs_to_hdf(
        self, wl_psfs, psfs, fiber_number: int, order: int, ccd_number: int
    ):
        try:
            psf_order_group = self.h5f.require_group(
                f"CCD_{ccd_number}/fiber_{fiber_number}/psf_order_{abs(order)}"
            )
        except TypeError:
            psf_order_group = self.h5f.create_group(
                f"CCD_{ccd_number}/fiber_{fiber_number}/psf_order_{abs(order)}"
            )

        for wl, psf in zip(wl_psfs, psfs):
            dataset = psf_order_group.create_dataset(
                name=f"wavelength_{wl}", data=np.array(psf.data)
            )
            dataset.attrs.create("wavelength", float(wl))
            dataset.attrs.create("order", int(abs(order)))
            dataset.attrs.create("dataSpacing", psf.sampling)

    def _save_transformation_to_hdf(
        self,
        wavelengths: np.ndarray,
        transformations: np.ndarray,
        order: int,
        fiber_number: int = 1,
        ccd_number: int = 1,
    ):
        try:
            fiber_group = self.h5f.require_group(
                f"CCD_{ccd_number}/fiber_{fiber_number}"
            )
        except TypeError:
            fiber_group = self.h5f.create_group(
                f"CCD_{ccd_number}/fiber_{fiber_number}"
            )

        columns = [
            "rotation",
            "scale_x",
            "scale_y",
            "shear",
            "translation_x",
            "translation_y",
            "wavelength",
        ]

        table = {c: [] for c in columns}

        ds_dt = np.dtype({"names": columns, "formats": [np.float32] * len(columns)})
        array = np.empty(shape=len(transformations), dtype=ds_dt)

        for wl, t in zip(wavelengths, transformations):
            table["rotation"].append(t[0])
            table["scale_x"].append(t[1])
            table["scale_y"].append(t[2])
            table["shear"].append(t[3])
            table["translation_x"].append(t[4])
            table["translation_y"].append(t[5])
            table["wavelength"].append(wl)

        for c in columns:
            array[c] = np.array([t for t in table[c]])

        fiber_group.create_dataset(
            name=f"order{abs(order)}", data=array, shape=np.shape(array)
        )
