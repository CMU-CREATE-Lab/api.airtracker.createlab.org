"""Models representing various spatial coordinate systems."""
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, List, Union

from pydantic import BaseModel, confloat

if TYPE_CHECKING:
    longitude_type = float
    latitude_type = float
else:
    # Constrained types in pydantic currently raise an invalid type error from MyPy,
    # likely related to this issue:
    #   https://github.com/samuelcolvin/pydantic/issues/3080
    longitude_type = confloat(ge=-180, lt=180)
    latitude_type = confloat(gt=-90, lt=90)


def from_to_by(start: float, stop: float, step: float, digits: int = 8) -> List[float]:
    """Sequence between start and stop (inclusive) by step, rounded to digits."""
    length = round((stop - start) / step)
    return [round(start + step * x, digits) for x in range(length + 1)]


class Point(BaseModel):
    x: longitude_type
    y: latitude_type


class GridExtent(BaseModel):
    xmin: longitude_type
    xmax: longitude_type
    ymin: latitude_type
    ymax: latitude_type

    def contains(self, other: Union["GridTimeExtent", "GridExtent"]) -> bool:
        """Check if the calling instance contains another extent."""
        return (
            self.xmin <= other.xmin
            and self.xmax >= other.xmax
            and self.ymin <= other.ymin
            and self.ymax >= other.ymax
        )

    def __contains__(self, other: Union["GridTimeExtent", "GridExtent"]) -> bool:
        return self.contains(other)


class GridResolution(BaseModel):
    xres: float
    yres: float


class Grid(GridExtent, GridResolution):
    def to_points(self) -> List[Point]:
        """Returns points placed on grid vertices."""
        xs = from_to_by(self.xmin, self.xmax, self.xres)
        ys = from_to_by(self.ymin, self.ymax, self.yres)
        return [Point(x=x, y=y) for x in xs for y in ys]


class GridTimeExtent(GridExtent):
    tmin: datetime
    tmax: datetime

    def expand(self, dx: float, dy: float, dt: timedelta) -> "GridTimeExtent":
        """Returns an expanded GridTimeExtent without modifying the existing object."""
        tmin = min(self.tmin, self.tmin + dt)
        tmax = max(self.tmax, self.tmax + dt)
        return GridTimeExtent(
            xmin=self.xmin - dx,
            xmax=self.xmax + dx,
            ymin=self.ymin - dy,
            ymax=self.ymax + dy,
            tmin=tmin,
            tmax=tmax,
        )

    def contains(self, other: Union["GridTimeExtent", GridExtent]) -> bool:
        """Check if the calling instance contains another extent."""
        contains_spatial_extent = super().__contains__(other)
        if isinstance(other, GridExtent):
            return contains_spatial_extent
        else:
            contains_time_extent = self.tmin <= other.tmin and self.tmax >= other.tmax
            return contains_spatial_extent and contains_time_extent

    def __contains__(self, other: Union["GridTimeExtent", GridExtent]) -> bool:
        return self.contains(other)
