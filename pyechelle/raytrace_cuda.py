import numpy as np
from numba.cuda.random import create_xoroshiro128p_states

import pyechelle._kernels
from pyechelle.raytracing import prepare_raytracing
from pyechelle.sources import Source
from pyechelle.spectrograph import Spectrograph
from pyechelle.telescope import Telescope


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
):
    result = prepare_raytracing(
        o, fiber, ccd_index, efficiency, rv, source, spec, telescope, t
    )
    if result is None:
        # no photons in this order -> return
        return 0
    else:
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
        ) = result

    field_shape = spec.get_field_shape(fiber=fiber, ccd_index=ccd_index)

    # the kernel name is kernel_cuda_rectangular_Listlike_True
    kernel_name = f"kernel_cuda_{field_shape}_{'ListLike' if source.list_like else "Continuous"}_True"
    # get kernel function from _kernels.py
    cuda_kernel = getattr(pyechelle._kernels, kernel_name)

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
        np.float32(ps),
        rng_states,
        total_photons,
    )
    return total_photons
