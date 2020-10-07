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
    for o in spec.orders:
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


def plot_psfs(spec: ZEMAX):
    """
    Plot PSFs as one big map
    Args:
        spec: Spectrograph model

    Returns:

    """
    plt.figure()
    n_orders = len(spec.orders)
    n_psfs = 50
    shape_psfs = spec.psfs[next(spec.psfs.keys().__iter__())].psfs[0].data.shape
    img = np.empty((n_psfs * shape_psfs[0], n_orders * shape_psfs[1]))
    for oo, o in enumerate(spec.orders):
        for i, p in enumerate(spec.psfs[f"psf_{o[:5]}" + "_" + f"{o[5:]}"].psfs):
            img[int(i * shape_psfs[0]):int((i + 1) * shape_psfs[0]),
            int(oo * shape_psfs[1]):int((oo + 1) * shape_psfs[1])
            ] = p.data
    plt.imshow(img, vmin=0, vmax=np.mean(img) * 10.0)
    plt.show()


if __name__ == "__main__":
    spec = ZEMAX("/home/stuermer/rdp_shared/marvel.hdf", 3)
    # plot_transformations(spec)
    plot_psfs(spec)
