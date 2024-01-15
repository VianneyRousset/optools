"""
Definition of constant uniform mesh lines.

Constant meshes have points defined before function evaluation.

The uniform distribution of points can be driven either by its division
(number of points), by its density (number of points per unit length) or
by its resolution (space between points).
"""

from .constant import ConstantMeshLine
from optools.utils import clip


import numpy as np
from paramobject import Parameter, parameter
from math import ceil


class UniformMeshLine(ConstantMeshLine):
    """
    Abstract class of a line with a constant uniform distributed point.

    Parameters
    ----------
    start : float
        The starting point of the line.
    stop : float
        The ending point of the line.
    division : int
        The number of points.
    density : float
        The number of points per unit length.
    resolution : float
        The length between points.

    Properties
    ----------
    length : float
        The length of the line.
    points : 1D-array
        The points of the line.

    Methods
    -------
    evaluate(self, func) -> 1D-array:
        Evaluate the given function at each point.
    """

    def with_params(self, *args, **kwargs):
        params = dict(*args, **kwargs)

        if "start" not in params:
            params["start"] = self.start

        if "stop" not in params:
            params["stop"] = self.stop

        if "division" in params:
            newobj = DivisionUniformMeshLine(**params)

        elif "density" in params:
            newobj = DensityUniformMeshLine(**params)

        elif "resolution" in params:
            newobj = ResolutionUniformMeshLine(**params)

        else:
            newobj = self.__class__(self, **params)

        if self.binding is not None:
            return self.binding.transfer(newobj)

        return newobj

    @property
    def points(self):
        """The points of the line."""
        return np.linspace(self.start, self.stop, self.division)

    @parameter
    def division(self):
        """The number of points."""
        raise NotImplementedError()

    @division.wither
    def with_division(self, division=None, *, min_division=None, max_division=None):
        if division is None:
            division = self.division

        division = clip(division, min_division, max_division)

        return self.with_params(division=division)

    @parameter
    def density(self):
        """The number of points per unit length."""
        raise NotImplementedError()

    @density.wither
    def with_density(self, density=None, *, min_density=None, max_density=None):
        if density is None:
            density = self.density

        density = clip(density, min_density, max_density)

        return self.with_params(density=density)

    @parameter
    def resolution(self):
        """The space between points."""
        raise NotImplementedError()

    @resolution.wither
    def with_resolution(
        self, resolution=None, *, min_resolution=None, max_resolution=None
    ):
        if resolution is None:
            resolution = self.resolution

        resolution = clip(resolution, min_resolution, max_resolution)

        return self.with_params(resolution=resolution)


class DivisionUniformMeshLine(UniformMeshLine):
    """
    Line with a constant uniform distributed point driven by division.

    This mesh ensures that a given number of points are uniformly generated
    between `start` and `stop`.

    Parameters
    ----------
    start : float
        The starting point of the line.
    stop : float
        The ending point of the line.
    division : int
        The number of points.
    density : float
        The number of points per unit length.
    resolution : float
        The space between points.

    Properties
    ----------
    length : float
        The length of the line.
    points : 1D-array
        The points of the line.

    Methods
    -------
    evaluate(self, func) -> 1D-array:
        Evaluate the given function at each point.
    """

    # the number of points
    division = Parameter(default=1000)
    with_division = division.wither(UniformMeshLine.with_division.wither_func)
    with_density = UniformMeshLine.with_density
    with_resolution = UniformMeshLine.with_resolution

    @UniformMeshLine.density.getter
    def density(self):
        """The number of points per unit length."""
        return self.division / self.length

    @UniformMeshLine.resolution.getter
    def resolution(self):
        """The space between points."""
        return self.length / self.division


class DensityUniformMeshLine(UniformMeshLine):
    """
    Line with a constant uniform distributed point driven by density.

    This mesh ensures that enough points are uniformly generated between
    `start` and `stop` for the number of point per unit length to be at least
    the given `density`.

    Parameters
    ----------
    start : float
        The starting point of the line.
    stop : float
        The ending point of the line.
    division : int
        The number of points.
    density : float
        The minimal number of points per unit length.
    resolution : float
        The space between points.

    Properties
    ----------
    length : float
        The length of the line.
    points : 1D-array
        The points of the line.

    Methods
    -------
    evaluate(self, func) -> 1D-array:
        Evaluate the given function at each point.
    """

    # the number of points per unit length
    density = Parameter(default=1000)
    with_division = UniformMeshLine.with_division
    with_density = density.wither(UniformMeshLine.with_density.wither_func)
    with_resolution = UniformMeshLine.with_resolution

    @UniformMeshLine.division.getter
    def division(self):
        """The number of points."""
        return ceil(self.density * self.length + 1)

    @UniformMeshLine.resolution.getter
    def resolution(self):
        """The space between points."""
        return self.length / self.division


class ResolutionUniformMeshLine(UniformMeshLine):
    """
    Line with a constant uniform distributed point driven by resolution.

    This mesh ensures that enough points are uniformly generated between
    `start` and `stop` for the space between points to be equal or less
    thant the given `resolution`.

    Parameters
    ----------
    start : float
        The starting point of the line.
    stop : float
        The ending point of the line.
    division : int
        The number of points.
    density : float
        The number of points per unit length.
    resolution : float
        The maximal space between points.

    Properties
    ----------
    length : float
        The length of the line.
    points : 1D-array
        The points of the line.

    Methods
    -------
    evaluate(self, func) -> 1D-array:
        Evaluate the given function at each point.
    """

    # the space between points
    resolution = Parameter(default=1e-3)
    resolution.wither_func = UniformMeshLine.with_resolution.wither_func
    with_division = UniformMeshLine.with_division
    with_density = UniformMeshLine.with_density
    with_resolution = resolution.wither(UniformMeshLine.with_resolution.wither_func)

    @UniformMeshLine.division.getter
    def division(self):
        """The number of points."""
        return ceil(self.length / self.resolution + 1)

    @UniformMeshLine.density.getter
    def density(self):
        """The number of points per unit length."""
        return self.division / self.length
