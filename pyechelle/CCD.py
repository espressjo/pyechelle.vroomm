""" Detector module

Implementing handling of CCDs/detectors for PyEchelle. It is recommended to use a dedicated dedector simulation framework
such as pyxel. Therefore, this module is rather simple.
"""
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger('CCD')


@dataclass
class CCD:
    """ A CCD detector

    Attributes:
        data (np.ndarray): data array (uint) that will be filled by the simulator
        name (str): name of the CCD detector. This will end up in the .fits header.
        xmin (int): minimum pixel x-coordinate
        xmax (int): maximum pixel x-coordinate
        ymin (int): minimum pixel y-coordinate
        ymax (int): maximum pixel y-coordinate
        maxval (int): maximum pixel value before clipping
        pixelsize (float): physical size of an individual pixel [microns]

    """
    data: np.ndarray = field(init=False)
    name: str = 'detector'
    xmin: int = 0
    xmax: int = 4096
    ymin: int = 0
    ymax: int = 4096
    maxval: int = 65536
    pixelsize: float = 9.

    def __post_init__(self):
        self.data = np.zeros(((self.ymax - self.ymin), (self.xmax - self.xmin)), dtype=np.uint32)

    def add_readnoise(self, std: float = 3.):
        """ Adds readnoise to the detector counts

        Args:
            std: standard deviation of readnoise in counts

        Returns:
            None
        """
        self.data = self.data + np.asarray(np.random.normal(0., std, self.data.shape).round(0), dtype=np.int32)

    def add_bias(self, value: int = 1000):
        """Adds a bias value to the detector counts

        Args:
            value: bias value to be added. If float will get rounded to next integer

        Returns:
            None
        """
        self.data += value

    def clip(self):
        """ Clips CCD data

        Clips data if any count value is larger than self.maxval

        Returns:
            None
        """
        if np.any(self.data < 0):
            logger.warning('There is data <0 which will be clipped. Make sure you e.g. apply the bias before the '
                           'readnoise.')
            self.data[self.data < 0] = 0
        if np.any(self.data > self.maxval):
            self.data[self.data > self.maxval] = self.maxval
