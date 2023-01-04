"""
This module contains footprint transformation and visualization utilities.
"""

import io
from pathlib import Path
from typing import Tuple

import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import numpy.typing as npt
from netCDF4 import Dataset  # type: ignore

from heatmap_grid import Grid

matplotlib.use("Agg")


def mercator_transform(
    data: npt.ArrayLike,
    lat_bounds: Tuple[float, float],
    origin: str = "upper",
    height_out: int = None,
):
    """
    Transforms an image computed in (longitude,latitude) coordinates into
    the a Mercator projection image.
    Parameters
    ----------
    data: numpy array or equivalent list-like object.
        Must be NxM (mono), NxMx3 (RGB) or NxMx4 (RGBA)
    lat_bounds : length 2 tuple
        Minimal and maximal value of the latitude of the image.
        Bounds must be between -85.051128779806589 and 85.051128779806589
        otherwise they will be clipped to that values.
    origin : ['upper' | 'lower'], optional, default 'upper'
        Place the [0,0] index of the array in the upper left or lower left
        corner of the axes.
    height_out : int, default None
        The expected height of the output.
        If None, the height of the input is used.
    See https://en.wikipedia.org/wiki/Web_Mercator for more details.
    """
    # From Folium
    # Source: https://github.com/python-visualization/folium/blob/master/folium/utilities.py

    def mercator(x):
        return np.arcsinh(np.tan(x * np.pi / 180.0)) * 180.0 / np.pi

    array = np.atleast_3d(data).copy()
    height, width, nblayers = array.shape

    lat_min = max(lat_bounds[0], -85.051128779806589)
    lat_max = min(lat_bounds[1], 85.051128779806589)
    if height_out is None:
        height_out = height

    # Eventually flip the image
    if origin == "upper":
        array = array[::-1, :, :]

    lats = lat_min + np.linspace(0.5 / height, 1.0 - 0.5 / height, height) * (
        lat_max - lat_min
    )
    latslats = mercator(lat_min) + np.linspace(
        0.5 / height_out, 1.0 - 0.5 / height_out, height_out
    ) * (mercator(lat_max) - mercator(lat_min))

    out = np.zeros((height_out, width, nblayers))
    for i in range(width):
        for j in range(nblayers):
            out[:, i, j] = np.interp(latslats, mercator(lats), array[:, i, j])

    # Eventually flip the image.
    if origin == "upper":
        out = out[::-1, :, :]
    return out


class Footprint:
    def __init__(self, nc: Dataset):
        """Representation of gridded 2 or 3 dimensional data."""
        #self.nc_bytes = nc_bytes

        keys = list(nc.variables.keys())
        self.x = nc.variables[keys[0]][:].filled()
        self.y = nc.variables[keys[1]][:].filled()
        self.values = nc.variables[keys[3]][:].filled()
        self.crs = nc.crs

        if np.sign(np.diff(self.y).mean()) > 0:
            self.values = np.flip(self.values, axis=1)
            self.y = np.flip(self.y)

        xres = round(np.abs(np.diff(self.x).mean()), 8)
        yres = round(np.abs(np.diff(self.y).mean()), 8)
        self.grid = Grid(
            xmin=round(self.x.min() - xres / 2, 8),
            xmax=round(self.x.max() + xres / 2, 8),
            xres=xres,
            ymin=round(self.y.min() - yres / 2, 8),
            ymax=round(self.y.max() + yres / 2, 8),
            yres=yres,
        )

    @classmethod
    def from_path(cls, path: Path):
        """Load footprint from a NetCDF file."""
        with open(path, "rb") as file_obj:
            nc_bytes = file_obj.read()
        return cls(nc_bytes)

    def create_image(
        self,
        log10: bool = False,
        mercator: bool = True,
        *args,
        **kwargs,
    ) -> bytes:
        """Create visualization of gridded data.

        Args:
            log10 (bool, optional): log10 transform values prior to plotting;
                defaults False.
            mercator (bool, optional): project the image from lat/lon to mercator,
                often to overlay on a mercator-projected basemap; defaults True.
            *args, **kwargs: passed to plt.imshow().

        Returns:
            bytes: bytes representation of png image.
        """
        image = self.values
        while len(image.shape) > 2:
            image = image.sum(axis=0)

        if mercator:
            image = mercator_transform(image, (self.grid.ymin, self.grid.ymax))

        image[image == 0] = np.nan

        if log10:
            image = np.log10(image)

        vmin = kwargs.get("vmin")
        if vmin:
            image[image < vmin] = np.nan

        plt.imshow(image, *args, **kwargs)
        plt.axis("off")

        f = io.BytesIO()
        plt.savefig(f, bbox_inches="tight", dpi=300, pad_inches=0, transparent=True)
        plt.close()

        return f.getvalue()

    def _validate_raster_attributes(self, x):
        """Ensure cell-by-cell operations are valid"""
        if self.extent != x.extent:
            raise ValueError("Extents do not match.")
        if self.resolution != x.resolution:
            raise ValueError("Resolutions do not match.")
        if not np.array_equal(self.x, x.x):
            raise ValueError("x attributes do not match.")
        if not np.array_equal(self.y, x.y):
            raise ValueError("y attributes do not match.")
        if len(self.layers) != len(x.layers):
            raise ValueError("layers lengths do not match.")
        if self.crs != x.crs:
            raise ValueError("crs attributes do not match.")

    def __add__(self, x):
        y = self.copy()
        if isinstance(x, type(self)):
            self._validate_raster_attributes(x)
            y.values = y.values + x.values
            return y
        else:
            y.values = y.values + x
            return y

    def __sub__(self, x):
        y = self.copy()
        if isinstance(x, type(self)):
            self._validate_raster_attributes(x)
            y.values = y.values - x.values
            return y
        else:
            y.values = y.values - x
            return y

    def __mul__(self, x):
        y = self.copy()
        if isinstance(x, type(self)):
            self._validate_raster_attributes(x)
            y.values = y.values * x.values
            return y
        else:
            y.values = y.values * x
            return y

    def __truediv__(self, x):
        y = self.copy()
        if isinstance(x, type(self)):
            self._validate_raster_attributes(x)
            y.values = y.values / x.values
            return y
        else:
            y.values = y.values / x
            return y

    def __rtruediv__(self, x):
        y = self.copy()
        if isinstance(x, type(self)):
            self._validate_raster_attributes(x)
            y.values = x.values / y.values
            return y
        else:
            y.values = x / y.values
            return y
