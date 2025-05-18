import math
import random

import numba.cuda
import numpy as np

from pyechelle.CCD import CCD
from pyechelle.optics import convert_matrix
from pyechelle.randomgen import make_alias_sampling_arrays, unravel_index
from pyechelle.sources import Source
from pyechelle.spectrograph import Spectrograph
from pyechelle.telescope import Telescope
import astropy.units as u


@numba.njit(cache=True, parallel=False, nogil=True)
def raytrace(
    spectrum_wl,
    spectrum_q,
    spectrum_j,
    transformations,
    trans_wl,
    trans_wld,
    transf_deriv,
    psfs_q,
    psfs_j,
    psf_wl,
    psf_wld,
    psf_shape,
    psf_sampling,
    ccd,
    pixelsize,
    slitfun,
    nphotons,
):
    max_y, max_x = ccd.shape

    for _ in range(nphotons):
        # sample from spectrum
        k = int(math.floor(random.random() * len(spectrum_j)))
        wl = (
            spectrum_wl[k]
            if random.random() < spectrum_q[k]
            else spectrum_wl[spectrum_j[k]]
        )

        # find index for transformation
        idx_trans_float = (wl - trans_wl[0]) / trans_wld
        idx_trans = math.floor(idx_trans_float)
        r = idx_trans_float - idx_trans

        # do linear interpolation of transformation matrices
        m0, m1, m2, m3, m4, m5 = (
            transformations[:, idx_trans] + r * transf_deriv[:, idx_trans]
        )

        # random start points in slit
        x, y = slitfun(random.random(), random.random())
        # transform
        xt = m0 * x + m1 * y + m2
        yt = m3 * x + m4 * y + m5

        # apply PSF
        idx_psf = int((wl - psf_wl[0]) / psf_wld)  # find psf index
        # next 3 lines implement drawing random number via alias sampling
        k = int(math.floor(random.random() * len(psfs_j[idx_psf])))
        if not random.random() < psfs_q[idx_psf][k]:
            k = psfs_j[idx_psf][k]

        dx, dy = unravel_index(k, psf_shape)
        xt += (dx - psf_shape[1] / 2.0) * psf_sampling / pixelsize
        yt += (dy - psf_shape[0] / 2.0) * psf_sampling / pixelsize
        x_int = math.floor(xt)
        y_int = math.floor(yt)

        if (0 <= x_int < max_x) and (0 <= y_int < max_y):
            ccd[y_int, x_int] += 1


@numba.njit(cache=True, parallel=False, nogil=True)
def raytrace_singlemode(
    spectrum_wl,
    spectrum_q,
    spectrum_j,
    transformations,
    trans_wl,
    trans_wld,
    transf_deriv,
    psfs_q,
    psfs_j,
    psf_wl,
    psf_wld,
    psf_shape,
    psf_sampling,
    ccd,
    pixelsize,
    nphotons,
):
    max_y, max_x = ccd.shape

    for _ in range(nphotons):
        # sample from spectrum
        k = int(math.floor(random.random() * len(spectrum_j)))
        wl = (
            spectrum_wl[k]
            if random.random() < spectrum_q[k]
            else spectrum_wl[spectrum_j[k]]
        )

        # find index for transformation
        idx_trans_float = (wl - trans_wl[0]) / trans_wld
        idx_trans = math.floor(idx_trans_float)
        r = idx_trans_float - idx_trans

        # do linear interpolation of transformation matrices (for SM this is not a full matrix, but just the
        # translation coordinates for the chief ray
        xt = transformations[2, idx_trans] + r * transf_deriv[2, idx_trans]
        yt = transformations[5, idx_trans] + r * transf_deriv[5, idx_trans]

        # apply PSF
        idx_psf = int((wl - psf_wl[0]) / psf_wld)  # find psf index
        # next 3 lines implement drawing random number via alias sampling
        k = int(math.floor(random.random() * len(psfs_j[idx_psf])))
        if not random.random() < psfs_q[idx_psf][k]:
            k = psfs_j[idx_psf][k]

        dx, dy = unravel_index(k, psf_shape)
        # here we add random.random()-0.5, so a uniform random number in [-0.5,0.5) that is applied to the
        # 'integer' offset dx,dy this should be equivalent of drawing from a 'piecewise constant' distribution
        xt += (
            (dx - psf_shape[1] / 2.0 + random.random() - 0.5) * psf_sampling / pixelsize
        )
        yt += (
            (dy - psf_shape[0] / 2.0 + random.random() - 0.5) * psf_sampling / pixelsize
        )
        x_int = math.floor(xt)
        y_int = math.floor(yt)

        if (0 <= x_int < max_x) and (0 <= y_int < max_y):
            ccd[y_int, x_int] += 1


def raytrace_order_cpu(
    o,
    spec: Spectrograph,
    source: Source,
    slit_fun: callable,
    telescope: Telescope,
    rv: float,
    t,
    ccd: CCD,
    fiber: int,
    ccd_index: int,
    efficiency=None,
    n_cpu=1,
):
    raytrace_data = prepare_raytracing(
        o, fiber, ccd_index, efficiency, rv, source, spec, telescope, t
    )
    if raytrace_data is not None:
        (
            psf_sampler_qj,
            psf_sampling,
            psf_shape,
            psfs_wl,
            psfs_wld,
            spectrum_sampler_j,
            spectrum_sampler_q,
            total_photons,
            trans_deriv,
            trans_wl,
            trans_wld,
            transformations,
            wavelength,
        ) = raytrace_data
        if n_cpu > 1:
            ccd_new = np.zeros_like(ccd.data, dtype=np.uint32)
            if slit_fun is not None:
                raytrace(
                    wavelength,
                    spectrum_sampler_q,
                    spectrum_sampler_j,
                    transformations,
                    trans_wl,
                    trans_wld,
                    trans_deriv,
                    psf_sampler_qj[:, 0],
                    psf_sampler_qj[:, 1],
                    psfs_wl,
                    psfs_wld[0],
                    psf_shape,
                    psf_sampling,
                    ccd_new,
                    float(ccd.pixelsize),
                    slit_fun,
                    total_photons,
                )
            else:
                raytrace_singlemode(
                    wavelength,
                    spectrum_sampler_q,
                    spectrum_sampler_j,
                    transformations,
                    trans_wl,
                    trans_wld,
                    trans_deriv,
                    psf_sampler_qj[:, 0],
                    psf_sampler_qj[:, 1],
                    psfs_wl,
                    psfs_wld[0],
                    psf_shape,
                    psf_sampling,
                    ccd_new,
                    float(ccd.pixelsize),
                    total_photons,
                )
            return ccd_new, total_photons
        else:
            if slit_fun is not None:
                raytrace(
                    wavelength,
                    spectrum_sampler_q,
                    spectrum_sampler_j,
                    transformations,
                    trans_wl,
                    trans_wld,
                    trans_deriv,
                    psf_sampler_qj[:, 0],
                    psf_sampler_qj[:, 1],
                    psfs_wl,
                    psfs_wld[0],
                    psf_shape,
                    psf_sampling,
                    ccd.data,
                    float(ccd.pixelsize),
                    slit_fun,
                    total_photons,
                )
            else:
                raytrace_singlemode(
                    wavelength,
                    spectrum_sampler_q,
                    spectrum_sampler_j,
                    transformations,
                    trans_wl,
                    trans_wld,
                    trans_deriv,
                    psf_sampler_qj[:, 0],
                    psf_sampler_qj[:, 1],
                    psfs_wl,
                    psfs_wld[0],
                    psf_shape,
                    psf_sampling,
                    ccd.data,
                    float(ccd.pixelsize),
                    total_photons,
                )
            return total_photons
    else:
        return 0  # no photons in this order


def prepare_raytracing(o, fiber, ccd_index, efficiency, rv, source, spec, telescope, t):
    wavelength = np.linspace(
        *spec.get_wavelength_range(o, fiber, ccd_index), num=100000
    )
    wl_range_nm = (np.min(wavelength) * 1000, np.max(wavelength) * 1000)
    # get counts from source
    photon_counts = source.get_counts_rv(wavelength, t, rv)
    if isinstance(photon_counts, tuple):
        wavelength, photon_counts = photon_counts

    # get efficiency per order
    if efficiency is not None:
        eff = efficiency.get_efficiency_per_order(wavelength=wavelength, order=o)
        photon_counts = eff * photon_counts

    total_photons = int(np.sum(photon_counts).value)
    print(
        f"Order {o:3d}: {wl_range_nm[0]:.2f}nm - {wl_range_nm[1]:.2f}nm, {total_photons} photons"
    )

    if total_photons > 0:
        minwl, maxwl = spec.get_wavelength_range(o, fiber, ccd_index)
        trans_wl, trans_wld = np.linspace(minwl, maxwl, 10000, retstep=True)
        transformations = convert_matrix(
            np.array(spec.get_transformation(trans_wl, o, fiber, ccd_index))
        )
        # transformations = np.array(spec.transformations[f'order{o}'].get_matrices_spline(trans_wl))
        # derivatives for simple linear interpolation
        trans_deriv = np.array([np.ediff1d(t, t[-1] - t[-2]) for t in transformations])
        psf_sampler_qj = np.array(
            [
                make_alias_sampling_arrays(p.data.T.ravel())
                for p in spec.get_psf(None, o, fiber, ccd_index)
            ]
        )
        psfs_wl = np.array(
            [p.wavelength for p in spec.get_psf(None, o, fiber, ccd_index)]
        )
        psfs_wld = np.ediff1d(psfs_wl, psfs_wl[-1] - psfs_wl[-2])
        psf_shape = spec.get_psf(None, o, fiber, ccd_index)[0].data.shape
        spectrum_sampler_q, spectrum_sampler_j = make_alias_sampling_arrays(
            np.asarray(photon_counts / np.sum(photon_counts), dtype=np.float32)
        )
        psf_sampling = spec.get_psf(None, o, fiber, ccd_index)[0].sampling
        return (
            psf_sampler_qj.astype(np.float32),
            np.float32(psf_sampling),
            psf_shape,
            psfs_wl.to_value(u.micron).astype(np.float32)
            if isinstance(psfs_wl, u.Quantity)
            else psfs_wl.astype(np.float32),
            psfs_wld.to_value(u.micron).astype(np.float32)
            if isinstance(psfs_wld, u.Quantity)
            else psfs_wld.astype(np.float32),
            spectrum_sampler_j,
            spectrum_sampler_q.astype(np.float32),
            total_photons,
            trans_deriv.astype(np.float32),
            trans_wl.to_value(u.micron).astype(np.float32)
            if isinstance(trans_wl, u.Quantity)
            else trans_wl.astype(np.float32),
            trans_wld.to_value(u.micron).astype(np.float32)
            if isinstance(trans_wld, u.Quantity)
            else trans_wld.astype(np.float32),
            transformations.astype(np.float32),
            wavelength.to_value(u.micron)
            if isinstance(wavelength, u.Quantity)
            else wavelength,
        )
    else:
        return None
