import math
import pathlib

import numba
import numba.cuda
import numpy as np
from joblib import Memory
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64

from pyechelle.raytracing import prepare_raytracing
from pyechelle.sources import Source
from pyechelle.spectrograph import Spectrograph
from pyechelle.telescope import Telescope

path = pathlib.Path(__file__).parent.resolve()
cache_path = path.joinpath(".cache")
# create data directory if it doesn't exist:
pathlib.Path(cache_path).mkdir(parents=False, exist_ok=True)
memory = Memory(cache_path, verbose=0)


@memory.cache
def make_cuda_kernel(slitfun):
    @cuda.jit()
    def cuda_kernel(
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
        rng_states,
        nphotons,
    ):
        max_y, max_x = ccd.shape
        thread_id = cuda.grid(1)

        for _ in range(thread_id, nphotons, cuda.gridDim.x * cuda.blockDim.x):
            # sample from spectrum
            k = int(
                math.floor(
                    xoroshiro128p_uniform_float64(rng_states, thread_id)
                    * len(spectrum_j)
                )
            )
            wl = (
                spectrum_wl[k]
                if xoroshiro128p_uniform_float64(rng_states, thread_id) < spectrum_q[k]
                else spectrum_wl[spectrum_j[k]]
            )

            # find index for transformation
            idx_trans_float = (wl - trans_wl[0]) / trans_wld
            idx_trans = int(math.floor(idx_trans_float))
            r = idx_trans_float - idx_trans

            # do linear interpolation of transformation matrices
            # m0, m1, m2, m3, m4, m5 = transformations[:, idx_trans] + r * transf_deriv[:, idx_trans]
            m0, m1, m2, m3, m4, m5 = transformations[:, idx_trans]
            dm0, dm1, dm2, dm3, dm4, dm5 = transf_deriv[:, idx_trans]
            m0 += r * dm0
            m1 += r * dm1
            m2 += r * dm2
            m3 += r * dm3
            m4 += r * dm4
            m5 += r * dm5

            # random start points in slit
            x = xoroshiro128p_uniform_float64(rng_states, thread_id)
            y = xoroshiro128p_uniform_float64(rng_states, thread_id)
            x, y = slitfun(x, y, rng_states, thread_id)

            # transform
            xt = m0 * x + m1 * y + m2
            yt = m3 * x + m4 * y + m5

            # apply PSF
            idx_psf = int((wl - psf_wl[0]) / psf_wld)  # find psf index
            # next 3 lines implement drawing random number via alias sampling
            k = int(
                math.floor(
                    xoroshiro128p_uniform_float64(rng_states, thread_id)
                    * len(psfs_j[idx_psf])
                )
            )
            if (
                not xoroshiro128p_uniform_float64(rng_states, thread_id)
                < psfs_q[idx_psf][k]
            ):
                k = psfs_j[idx_psf][k]

            # unravel 2d index
            dy = k % psf_shape[1]
            k = k // psf_shape[1]
            dx = k % psf_shape[0]

            # dx, dy = unravel_index(k, psf_shape)
            xt += (dx - psf_shape[1] / 2.0) * psf_sampling / pixelsize
            yt += (dy - psf_shape[0] / 2.0) * psf_sampling / pixelsize
            x_int = int(math.floor(xt))
            y_int = int(math.floor(yt))

            if (0 <= x_int < max_x) and (0 <= y_int < max_y):
                numba.cuda.atomic.inc(ccd, (y_int, x_int), 4294967295)

    return cuda_kernel


@memory.cache
def make_cuda_kernel_singlemode():
    @cuda.jit()
    def cuda_kernel(
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
        rng_states,
        nphotons,
    ):
        max_y, max_x = ccd.shape
        thread_id = cuda.grid(1)

        for _ in range(thread_id, nphotons, cuda.gridDim.x * cuda.blockDim.x):
            # sample from spectrum
            k = int(
                math.floor(
                    xoroshiro128p_uniform_float64(rng_states, thread_id)
                    * len(spectrum_j)
                )
            )
            wl = (
                spectrum_wl[k]
                if xoroshiro128p_uniform_float64(rng_states, thread_id) < spectrum_q[k]
                else spectrum_wl[spectrum_j[k]]
            )

            # find index for transformation
            idx_trans_float = (wl - trans_wl[0]) / trans_wld
            idx_trans = int(math.floor(idx_trans_float))
            r = idx_trans_float - idx_trans

            # do linear interpolation of transformation matrices
            xt = transformations[2, idx_trans]
            yt = transformations[5, idx_trans]
            dm2 = transf_deriv[2, idx_trans]
            dm5 = transf_deriv[5, idx_trans]

            xt += r * dm2
            yt += r * dm5

            # apply PSF
            idx_psf = int((wl - psf_wl[0]) / psf_wld)  # find psf index
            # next 3 lines implement drawing random number via alias sampling
            k = int(
                math.floor(
                    xoroshiro128p_uniform_float64(rng_states, thread_id)
                    * len(psfs_j[idx_psf])
                )
            )
            if (
                not xoroshiro128p_uniform_float64(rng_states, thread_id)
                < psfs_q[idx_psf][k]
            ):
                k = psfs_j[idx_psf][k]

            # unravel 2d index
            dy = k % psf_shape[1]
            k = k // psf_shape[1]
            dx = k % psf_shape[0]

            # dx, dy = unravel_index(k, psf_shape)
            # random point within sampling element of PSF
            dx_psf = xoroshiro128p_uniform_float64(rng_states, thread_id)
            dy_psf = xoroshiro128p_uniform_float64(rng_states, thread_id)

            xt += (dx - psf_shape[1] / 2.0 + dx_psf - 0.5) * psf_sampling / pixelsize
            yt += (dy - psf_shape[0] / 2.0 + dy_psf - 0.5) * psf_sampling / pixelsize
            x_int = int(math.floor(xt))
            y_int = int(math.floor(yt))

            if (0 <= x_int < max_x) and (0 <= y_int < max_y):
                numba.cuda.atomic.inc(ccd, (y_int, x_int), 4294967295)

    return cuda_kernel


def raytrace_order_cuda(
    o,
    spec: Spectrograph,
    source: Source,
    telescope: Telescope,
    rv: float,
    t,
    ccd,
    ps,
    fiber: int,
    ccd_index: int,
    efficiency=None,
    seed=-1,
    cuda_kernel=None,
):
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
    ) = prepare_raytracing(
        o, fiber, ccd_index, efficiency, rv, source, spec, telescope, t
    )

    threads_per_block = 128
    blocks = 64
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=seed)

    cuda_kernel[threads_per_block, blocks](
        np.ascontiguousarray(wavelength),
        np.ascontiguousarray(spectrum_sampler_q),
        np.ascontiguousarray(spectrum_sampler_j),
        np.ascontiguousarray(transformations),
        np.ascontiguousarray(trans_wl),
        trans_wld,
        np.ascontiguousarray(trans_deriv),
        np.ascontiguousarray(psf_sampler_qj[:, 0]),
        np.ascontiguousarray(psf_sampler_qj[:, 1]),
        np.ascontiguousarray(psfs_wl),
        psfs_wld[0],
        np.ascontiguousarray(psf_shape),
        psf_sampling,
        ccd,
        float(ps),
        rng_states,
        total_photons,
    )
    return total_photons
