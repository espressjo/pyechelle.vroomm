""" Optics module

PyEchelle concept is to describe the optics of an instrument by applying a wavelength dependent affine transformation
 to the input plane and applying a PSF. This module implements the two basic classes that are needed to do so.
"""
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AffineTransformation:
    r""" Affine transformation matrix

    This class represents an affine transformation matrix.

    PyEchelle uses affine transformations (which are represented by affine transformation matrices) to describe the
    mapping of a monochromatic image from the input focal plane of the spectrograph to the detector plane.

    See `wikipedia <https://en.wikipedia.org/wiki/Affine_transformation>`_ for more details about affine transformations.

    In two dimensions an affine transformation matrix can be written as:

    .. math::
        M = \begin{bmatrix}
        m0 & m1 & m2 \\
        m3 & m4 & m4 \\
        0 & 0 & 1
        \end{bmatrix}

    The last row is constant and is therefore be omitted. This is the form that is returned (as a flat array) by
    :meth:`pyechelle.AffineTransformation.as_matrix`

    There is another more intuitive representation of an affine transformation matrix in terms of the parameters:
    rotation, scaling in x- and y-direction, shear and translation in x- and y-direction.
    See :meth:`pyechelle.AffineTransformation.__post_init__` for how those representations are connected.

    Instances of this class can be sorted by wavelength.

    Attributes:
        rot (float): rotation [radians]
        sx (float): scaling factor in x-direction
        sy (float): scaling factor in y-direction
        shear (float): shearing factor
        tx (float): translation in x-direction
        ty (float): translation in y-direction
        wavelength (float): wavelength [micron] of affine transformation matrix
        m0 (float): affine transformation matrix element 0
        m1 (float): affine transformation matrix element 1
        m2 (float): affine transformation matrix element 2
        m3 (float): affine transformation matrix element 3
        m4 (float): affine transformation matrix element 4
        m5 (float): affine transformation matrix element 5

    """
    rot: float
    sx: float
    sy: float
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

    def as_matrix(self):
        """flat affine matrix

        Returns:
            flat affine transformation matrix
        """

        return self.m0, self.m1, self.m2, self.m3, self.m4, self.m5


@dataclass
class PSF:
    """Point spread function

    The point spread function describes how an optical system responds to a point source.

    Attributes:
        wavelength (float): wavelength [micron]
        data (np.ndarray): PSF data as 2D array
        sampling (float): physical size of the sampling of data [micron]

    """
    wavelength: float
    data: np.ndarray
    sampling: float

    def __le__(self, other):
        return self.wavelength <= other.wavelength

    def __lt__(self, other):
        return self.wavelength < other.wavelength

    def __str__(self):
        res = f'PSF@\n{self.wavelength:.4f}micron\n'
        letters = ['.', ':', 'o', 'x', '#', '@']
        norm = np.max(self.data)
        for d in self.data:
            for dd in d:
                i = int(math.floor(dd / norm / 0.2))
                res += letters[i]
            res += '\n'
        return res
