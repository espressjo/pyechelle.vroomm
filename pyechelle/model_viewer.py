import matplotlib.pyplot as plt
import numpy as np

from pyechelle.spectrograph import ZEMAX


def plot_transformations(spec: ZEMAX):
    """

    Args:
        spec: Spectrograph model

    Returns:

    """
    fig, ax = plt.subplots(2, 3, 'all')
    fig.suptitle(f"Affine transformations of {spec.name}")
    for o in spec.order_keys:
        ax[0, 0].set_title("sx")
        ax[0, 0].plot(spec.transformations[o].sx)
        ax[0, 1].set_title("sy")
        ax[0, 1].plot(spec.transformations[o].sy)
        ax[0, 2].set_title("shear")
        ax[0, 2].plot(spec.transformations[o].shear)
        ax[1, 0].set_title("rot")
        ax[1, 0].plot(spec.transformations[o].rot)
        ax[1, 1].set_title("tx")
        ax[1, 1].plot(spec.transformations[o].tx)
        ax[1, 2].set_title("ty")
        ax[1, 2].plot(spec.transformations[o].ty)
    plt.show()


def plot_transformation_matrices(spec: ZEMAX):
    """

    Args:
        spec: Spectrograph model

    Returns:

    """
    fig, ax = plt.subplots(2, 3, 'all')
    fig.suptitle(f"Affine transformation matrices of {spec.name}")
    for o in spec.order_keys:
        ax[0, 0].set_title("m0")
        ax[0, 0].plot(spec.transformations[o].m0)
        ax[0, 1].set_title("m1")
        ax[0, 1].plot(spec.transformations[o].m1)
        ax[0, 2].set_title("m2")
        ax[0, 2].plot(spec.transformations[o].m2)
        ax[1, 0].set_title("m3")
        ax[1, 0].plot(spec.transformations[o].m3)
        ax[1, 1].set_title("m4")
        ax[1, 1].plot(spec.transformations[o].m4)
        ax[1, 2].set_title("m5")
        ax[1, 2].plot(spec.transformations[o].m5)
    plt.show()


def plot_psfs(spec: ZEMAX):
    """
    Plot PSFs as one big map
    Args:
        spec: Spectrograph model

    Returns:

    """
    plt.figure()
    n_orders = len(spec.order_keys)
    n_psfs = max([len(spec.psfs[k].psfs) for k in spec.psfs.keys()])
    shape_psfs = spec.psfs[next(spec.psfs.keys().__iter__())].psfs[0].data.shape
    img = np.empty((n_psfs * shape_psfs[0], n_orders * shape_psfs[1]))
    for oo, o in enumerate(spec.order_keys):
        for i, p in enumerate(spec.psfs[f"psf_{o[:5]}" + "_" + f"{o[5:]}"].psfs):
            if p.data.shape == shape_psfs:
                img[int(i * shape_psfs[0]):int((i + 1) * shape_psfs[0]),
                int(oo * shape_psfs[1]):int((oo + 1) * shape_psfs[1])
                ] = p.data
    plt.imshow(img, vmin=0, vmax=np.mean(img) * 10.0)
    plt.show()


if __name__ == "__main__":
    spec = ZEMAX("../models/marvel201020.hdf", 3)
    plot_transformations(spec)
    plot_transformation_matrices(spec)
    plot_psfs(spec)
