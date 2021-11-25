import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from pyechelle.simulator import model_name_to_path
from pyechelle.spectrograph import ZEMAX


def plot_transformations(spectrograph: ZEMAX):
    """

    Args:
        spectrograph: Spectrograph model

    Returns:

    """
    fig, ax = plt.subplots(2, 3, 'all')
    fig.suptitle(f"Affine transformations of {spectrograph.name}")
    for o in spectrograph.order_keys:
        ax[0, 0].set_title("sx")
        ax[0, 0].plot(spectrograph.transformations[o].sx)
        ax[0, 1].set_title("sy")
        ax[0, 1].plot(spectrograph.transformations[o].sy)
        ax[0, 2].set_title("shear")
        ax[0, 2].plot(spectrograph.transformations[o].shear)
        ax[1, 0].set_title("rot")
        ax[1, 0].plot(spectrograph.transformations[o].rot)
        ax[1, 1].set_title("tx")
        ax[1, 1].plot(spectrograph.transformations[o].tx)
        ax[1, 2].set_title("ty")
        ax[1, 2].plot(spectrograph.transformations[o].ty)
    plt.show()


def plot_transformation_matrices(spectrograph: ZEMAX):
    """

    Args:
        spectrograph: Spectrograph model

    Returns:

    """
    fig, ax = plt.subplots(2, 3, 'all')
    fig.suptitle(f"Affine transformation matrices of {spectrograph.name}")
    for o in spectrograph.order_keys:
        ax[0, 0].set_title("m0")
        ax[0, 0].plot(spectrograph.transformations[o].m0)
        ax[0, 1].set_title("m1")
        ax[0, 1].plot(spectrograph.transformations[o].m1)
        ax[0, 2].set_title("m2")
        ax[0, 2].plot(spectrograph.transformations[o].m2)
        ax[1, 0].set_title("m3")
        ax[1, 0].plot(spectrograph.transformations[o].m3)
        ax[1, 1].set_title("m4")
        ax[1, 1].plot(spectrograph.transformations[o].m4)
        ax[1, 2].set_title("m5")
        ax[1, 2].plot(spectrograph.transformations[o].m5)
    plt.show()


def plot_psfs(spectrograph: ZEMAX):
    """
    Plot PSFs as one big map
    Args:
        spectrograph: Spectrograph model

    Returns:

    """
    plt.figure()
    n_orders = len(spectrograph.order_keys)
    n_psfs = max([len(spectrograph.psfs[k].psfs) for k in spectrograph.psfs.keys()])
    shape_psfs = spectrograph.psfs[next(spectrograph.psfs.keys().__iter__())].psfs[0].data.shape
    img = np.empty((n_psfs * shape_psfs[0], n_orders * shape_psfs[1]))

    for oo, o in enumerate(np.sort(spectrograph.orders)):
        for i, p in enumerate(spectrograph.psfs[f"psf_order_" + f"{o}"].psfs):
            if p.data.shape == shape_psfs:
                img[int(i * shape_psfs[0]):int((i + 1) * shape_psfs[0]),
                int(oo * shape_psfs[1]):int((oo + 1) * shape_psfs[1])] = p.data
    plt.imshow(img, vmin=0, vmax=np.mean(img) * 10.0)
    plt.show()


def plot_fields(spec: ZEMAX):
    plt.figure()
    with h5py.File(spec.modelpath, 'r') as h5f:
        for k in h5f.keys():
            if 'fiber_' in k:
                a = h5f[k].attrs['norm_field'].decode('utf-8').split('\n')

                for b in a:
                    if 'aF' in b:
                        print(b[2:])


def main(args):
    if not args:
        args = sys.argv[1:]

    dir_path = Path(__file__).resolve().parent.parent.joinpath("models")
    models = [x.stem for x in dir_path.glob('*.hdf')]

    parser = argparse.ArgumentParser(description='PyEchelle Simulator Model Viewer')
    parser.add_argument('-s', '--spectrograph', nargs='?', type=model_name_to_path, default=sys.stdin, required=True,
                        help=f"Filename of spectrograph model. Model file needs to be located in models/ folder. "
                             f"Options are {','.join(models)}")

    parser.add_argument('--fiber', type=int, default=1, required=False)

    args = parser.parse_args(args)

    spec = ZEMAX(args.spectrograph, args.fiber)
    # if args.plot_transformations:
    plot_transformations(spec)
    # if args.plot_transformation_matrices:
    plot_transformation_matrices(spec)
    # if args.plot_field:
    # plot_fields(spec)
    # if args.plot_psfs:
    plot_psfs(spec)


if __name__ == "__main__":
    main(sys.argv[1:])
    #
    # spec = ZEMAX("../models/marvel201020.hdf", 3)
    # plot_transformations(spec)
    # plot_transformation_matrices(spec)
    # plot_psfs(spec)
