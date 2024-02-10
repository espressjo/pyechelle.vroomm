""" Optics module

PyEchelle concept is to describe the optics of an instrument by applying a wavelength dependent affine transformation
 to the input plane and applying a PSF. This module implements the two basic classes that are needed to do so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numba
import numpy as np
from scipy.interpolate import CubicSpline


def correct_phase_jumps(data):
    return (data + np.pi) % (2 * np.pi) - np.pi


def _center_and_normalize_points(points):
    """Center and normalize image points.

    The points are transformed in a two-step procedure that is expressed
    as a transformation matrix. The matrix of the resulting points is usually
    better conditioned than the matrix of the original points.

    Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(D).

    If the points are all identical, the returned values will contain nan.

    Parameters
    ----------
    points : (N, D) array
        The coordinates of the image points.

    Returns
    -------
    matrix : (D+1, D+1) array_like
        The transformation matrix to obtain the new points.
    new_points : (N, D) array
        The transformed image points.

    References
    ----------
    .. [1] Hartley, Richard I. "In defense of the eight-point algorithm."
           Pattern Analysis and Machine Intelligence, IEEE Transactions on 19.6
           (1997): 580-593.

    """
    n, d = points.shape
    centroid = np.mean(points, axis=0)

    centered = points - centroid
    rms = np.sqrt(np.sum(centered**2) / n)

    # if all the points are the same, the transformation matrix cannot be
    # created. We return an equivalent matrix with np.nans as sentinel values.
    # This obviates the need for try/except blocks in functions calling this
    # one, and those are only needed when actual 0 is reached, rather than some
    # small value; ie, we don't need to worry about numerical stability here,
    # only actual 0.
    if rms == 0:
        return np.full((d + 1, d + 1), np.nan), np.full_like(points, np.nan)

    norm_factor = np.sqrt(d) / rms

    part_matrix = norm_factor * np.concatenate(
        (np.eye(d), -centroid[:, np.newaxis]), axis=1
    )
    matrix = np.concatenate(
        (
            part_matrix,
            [
                [
                    0,
                ]
                * d
                + [1]
            ],
        ),
        axis=0,
    )

    points_h = np.row_stack([points.T, np.ones(n)])

    new_points_h = (matrix @ points_h).T

    new_points = new_points_h[:, :d]
    new_points /= new_points_h[:, d:]

    return matrix, new_points


def find_affine(src, dst):
    coeffs = range(6)
    src = np.asarray(src)
    dst = np.asarray(dst)
    n, d = src.shape

    src_matrix, src = _center_and_normalize_points(src)
    dst_matrix, dst = _center_and_normalize_points(dst)
    if not np.all(np.isfinite(src_matrix + dst_matrix)):
        raise ValueError("Culdn't calculate matrix")

    # params: a0, a1, a2, b0, b1, b2, c0, c1
    A = np.zeros((n * d, (d + 1) ** 2))
    # fill the A matrix with the appropriate block matrices; see docstring
    # for 2D example — this can be generalised to more blocks in the 3D and
    # higher-dimensional cases.
    for ddim in range(d):
        A[ddim * n : (ddim + 1) * n, ddim * (d + 1) : ddim * (d + 1) + d] = src
        A[ddim * n : (ddim + 1) * n, ddim * (d + 1) + d] = 1
        A[ddim * n : (ddim + 1) * n, -d - 1 : -1] = src
        A[ddim * n : (ddim + 1) * n, -1] = -1
        A[ddim * n : (ddim + 1) * n, -d - 1 :] *= -dst[:, ddim : (ddim + 1)]

    # Select relevant columns, depending on params
    A = A[:, list(coeffs) + [-1]]
    _, _, V = np.linalg.svd(A)

    # if the last element of the vector corresponding to the smallest
    # singular value is close to zero, this implies a degenerate case
    # because it is a rank-defective transform, which would map points
    # to a line rather than a plane.
    if np.isclose(V[-1, -1], 0):
        raise ValueError("Culdn't calculate matrix")

    H = np.zeros((d + 1, d + 1))
    # solution is right singular vector that corresponds to smallest
    # singular value
    H.flat[list(coeffs) + [-1]] = -V[-1, :-1] / V[-1, -1]
    H[d, d] = 1

    # De-center and de-normalize
    H = np.linalg.inv(dst_matrix) @ H @ src_matrix

    # Small errors can creep in if points are not exact, causing the last
    # element of H to deviate from unity. Correct for that here.
    H /= H[-1, -1]

    return H


def decompose_affine_matrix(m):
    scale_x, scale_y = np.sqrt(np.sum(m**2, axis=0))[:2]
    rotation = math.atan2(m[1, 0], m[0, 0])

    beta = math.atan2(-m[0, 1], m[1, 1])
    shear = beta - rotation
    tx, ty = m[0:2, 2]
    return rotation, scale_x, scale_y, shear, tx, ty


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
        wavelength (float | None): wavelength [micron] of affine transformation matrix
    """

    rot: float
    sx: float
    sy: float
    shear: float
    tx: float
    ty: float
    wavelength: float | None

    def __le__(self, other):
        return self.wavelength <= other.wavelength

    def __lt__(self, other):
        return self.wavelength < other.wavelength

    def __add__(self, other):
        wl = None
        if other.wavelength and self.wavelength:
            assert np.isclose(other.wavelength, self.wavelength)
            wl = self.wavelength
        if other.wavelength and self.wavelength is None:
            wl = other.wavelength
        if self.wavelength and other.wavelength is None:
            wl = self.wavelength
        return AffineTransformation(
            self.rot + other.rot,
            self.sx + other.sx,
            self.sy + other.sy,
            self.shear + other.shear,
            self.tx + other.tx,
            self.ty + other.ty,
            wl,
        )

    def __sub__(self, other):
        wl = None
        if other.wavelength and self.wavelength:
            assert np.isclose(other.wavelength, self.wavelength)
            wl = self.wavelength
        if other.wavelength and self.wavelength is None:
            wl = other.wavelength
        if self.wavelength and other.wavelength is None:
            wl = self.wavelength
        return AffineTransformation(
            self.rot - other.rot,
            self.sx - other.sx,
            self.sy - other.sy,
            self.shear - other.shear,
            self.tx - other.tx,
            self.ty - other.ty,
            wl,
        )

    def __iadd__(self, other):
        assert np.isclose(other.wavelength, self.wavelength)
        self.sx += other.sx
        self.sy += other.sy
        self.shear += other.shear
        self.tx += other.tx
        self.ty += other.ty
        return self

    def __isub__(self, other):
        assert np.isclose(other.wavelength, self.wavelength)
        self.sx -= other.sx
        self.sy -= other.sy
        self.shear -= other.shear
        self.tx -= other.tx
        self.ty -= other.ty
        return self

    def __eq__(self, other):
        return np.isclose(self.as_matrix(), other.as_matrix()).all()

    def __mul__(self, other):
        assert isinstance(other, tuple), (
            "You can only multiply an affine matrix with a tuple of length 2 (x,"
            "y coordinate) "
        )
        assert (
            len(other) == 2
        ), "You can only multiply an affine matrix with a tuple of length 2  (x,y coordinate)"
        x_new = (
            self.sx * math.cos(self.rot) * other[0]
            - self.sy * math.sin(self.rot + self.shear) * other[1]
            + self.tx
        )
        y_new = (
            self.sx * math.sin(self.rot) * other[0]
            + self.sy * math.cos(self.rot + self.shear) * other[1]
            + self.ty
        )
        return x_new, y_new

    def as_matrix(self) -> tuple:
        """flat affine matrix

        Returns:
            flat affine transformation matrix
        """
        return self.rot, self.sx, self.sy, self.shear, self.tx, self.ty


@dataclass
class TransformationSet:
    affine_transformations: list[AffineTransformation]
    rot: np.ndarray = field(init=False)
    sx: np.ndarray = field(init=False)
    sy: np.ndarray = field(init=False)
    shear: np.ndarray = field(init=False)
    tx: np.ndarray = field(init=False)
    ty: np.ndarray = field(init=False)
    wl: np.ndarray = field(init=False)

    _spline_affine: list = field(init=False, default=None)

    def __post_init__(self):
        self.affine_transformations.sort()

        self.rot = np.array([at.rot for at in self.affine_transformations])
        self.sx = np.array([at.sx for at in self.affine_transformations])
        self.sy = np.array([at.sy for at in self.affine_transformations])
        self.shear = np.array([at.shear for at in self.affine_transformations])
        self.tx = np.array([at.tx for at in self.affine_transformations])
        self.ty = np.array([at.ty for at in self.affine_transformations])

        # correct for possible jumps in rot and shear
        # assert max(abs(np.ediff1d(self.shear))) < 4, 'There is a jump in the shear parameter of the model file. ' \
        #                                              'Please correct the jump, by wrapping shear to e.g. to (-pi, pi) or (0, 2.*pi)'

        # self.shear = np.mod(self.shear, np.pi * 2.)

        self.wl = np.array([at.wavelength for at in self.affine_transformations])

    def get_affine_transformations(
        self, wl: float | np.ndarray
    ) -> AffineTransformation | np.ndarray:
        if self._spline_affine is None:
            self._spline_affine = [
                CubicSpline(self.wl, self.rot),
                CubicSpline(self.wl, self.sx),
                CubicSpline(self.wl, self.sy),
                CubicSpline(self.wl, self.shear),
                CubicSpline(self.wl, self.tx),
                CubicSpline(self.wl, self.ty),
            ]
        if isinstance(wl, float):
            return AffineTransformation(*[af(wl) for af in self._spline_affine], wl)
        else:
            return np.array([af(wl) for af in self._spline_affine])


def convert_matrix(input: AffineTransformation | np.ndarray) -> np.ndarray:
    if isinstance(input, AffineTransformation):
        return np.array(
            [
                input.sx * math.cos(input.rot),
                -input.sy * math.sin(input.rot + input.shear),
                input.tx,
                input.sx * math.sin(input.rot),
                input.sy * math.cos(input.rot + input.shear),
                input.ty,
            ]
        )
    else:
        assert isinstance(input, np.ndarray)
        return np.array(
            [
                input[1] * np.cos(input[0]),
                -input[2] * np.sin(input[0] + input[3]),
                input[4],
                input[1] * np.sin(input[0]),
                input[2] * np.cos(input[0] + input[3]),
                input[5],
            ]
        )


@numba.njit
def apply_matrix(matrix, coords):
    return matrix[0] * coords[0] + matrix[1] * coords[1] + matrix[2], matrix[
        3
    ] * coords[0] + matrix[4] * coords[1] + matrix[5]


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

    def __post_init__(self):
        self.data /= np.sum(self.data)

    def __le__(self, other):
        return self.wavelength <= other.wavelength

    def __lt__(self, other):
        return self.wavelength < other.wavelength

    def __str__(self):
        res = f"PSF@\n{self.wavelength:.4f}micron\n"
        letters = [".", ":", "o", "x", "#", "@"]
        norm = np.max(self.data)
        for d in self.data:
            for dd in d:
                i = int(math.floor(dd / norm / 0.2))
                res += letters[i]
            res += "\n"
        return res

    def __eq__(self, other):
        equal_wavelength = self.wavelength == other.wavelength
        if not equal_wavelength:
            print(
                f"Wavelength data is different ({self.wavelength} vs. {other.wavelength})"
            )
        equal_sampling = self.sampling == other.sampling
        if not equal_sampling:
            print(f"Data sampling is different ({self.sampling} vs. {other.sampling})")
        equal_data = np.array_equal(self.data, other.data)
        if not equal_data:
            print(f"Data is different ({self.data} vs. {other.data})")
        return equal_wavelength and equal_sampling and equal_wavelength

    def check(self, threshold=1e-3):
        """Checks PSF for consistency.

        Checks if the PSF has 'high' flux on the outer edge of its sampling area. Significant flux on the border of
        the PSF means that the sampling area of the PSF is most likely not large enough. This could lead to an
        artificial cutoff of the PSF. As a result, spectra would be 'sharper' than they are in reality.

        TODO:
            Right now, the function merely gives an indication whether the PSF is sampled correctly or not. Ideally,
            the threshold should be adapted to the pixel size of the CCD to determine the spill-over per pixel.

        Args:
            threshold (float): threshold up to which the PSF is considered OK

        Returns:
            True if check is OK False if flux is higher than threshold
        """
        return (
            sum(self.data[0])
            + sum(self.data[-1])
            + sum(self.data[:, 0])
            + sum(self.data[:, -1])
            < threshold
        )
