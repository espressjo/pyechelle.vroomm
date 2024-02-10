""" Detector module

Implementing handling of CCDs/detectors for PyEchelle. It is recommended to use a dedicated detector simulation
framework such as pyxel for including CCD effects such as cosmics, CTI or the brighter-fatter effect.

Therefore, this module is kept rather simple.
"""
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("CCD")


@dataclass
class CCD:
    """A CCD detector

    Attributes:
        data (np.ndarray): data array (uint) that will be filled by the simulator
        n_pix_x (int): number of pixels in x-direction
        n_pix_y (int): number of pixels in y-direction
        maxval (int): maximum pixel value before clipping
        pixelsize (float): physical size of an individual pixel [microns]
        identifier (str): identifier of the CCD. This will also end up in the .fits header.

    """

    n_pix_x: int = 4096
    n_pix_y: int = 4096
    maxval: int = 65536
    pixelsize: float = 9.0
    identifier: str = "detector"
    data: np.ndarray = field(init=False)

    def __post_init__(self):
        self.data = np.zeros((self.n_pix_y, self.n_pix_x), dtype=np.uint32)

    def add_readnoise(self, std: float = 3.0):
        """Adds readnoise to the detector counts

        Args:
            std: standard deviation of readnoise in counts

        Returns:
            None
        """
        self.data = self.data + np.asarray(
            np.random.normal(0.0, std, self.data.shape).round(0), dtype=np.int32
        )

    def add_bias(self, value: int = 1000):
        """Adds a bias value to the detector counts

        Args:
            value: bias value to be added. If float will get rounded to next integer

        Returns:
            None
        """
        self.data += value

    def clip(self):
        """Clips CCD data

        Clips data if any count value is larger than self.maxval

        Returns:
            None
        """
        if np.any(self.data < 0):
            logger.warning(
                "There is data <0 which will be clipped. Make sure you e.g. apply the bias before the "
                "readnoise."
            )
            self.data[self.data < 0] = 0
        if np.any(self.data > self.maxval):
            self.data[self.data > self.maxval] = self.maxval

    def __eq__(self, other):
        equal_pixel = (self.n_pix_x == other.n_pix_x) and (
            self.n_pix_y == other.n_pix_y
        )
        if not equal_pixel:
            print(
                f"The number of pixels differs ({self.n_pix_x}x{self.n_pix_y} vs. {other.n_pix_x}x{other.n_pix_y}"
            )
        equal_pixelsize = self.pixelsize == other.pixelsize
        if not equal_pixelsize:
            print(f"The pixel size differs {self.pixelsize} vs. {other.pixelsize}")
        equal_maxval = self.maxval == other.maxval
        if not equal_maxval:
            print(f"The maxval differs {self.maxval} vs. {other.maxval}")
        equal_identifier = self.identifier == other.identifier
        if not equal_identifier:
            print(f"The identifier differs {self.identifier} vs. {other.identifier}")
        equal_data = np.array_equal(self.data, other.data)
        if not equal_data:
            print(f"The data differs: {self.data} vs. {other.data}")
        return (
            equal_pixel
            and equal_pixelsize
            and equal_maxval
            and equal_identifier
            and equal_data
        )
