"""
Abstract definition of mesh with constant distribution.

Constant meshes have points defined before function evaluation.
"""

from .base import MeshLine
from abc import abstractmethod


class ConstantMeshLine(MeshLine):
    """
    Abstract class of a mesh line with a constant distributed point.

    Constant meshes have points defined before function evaluation.

    Parameters
    ----------
    start : float
        The starting point of the line.
    stop : float
        The ending point of the line.

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

    @property
    @abstractmethod
    def points(self):
        """The points of the line."""

    def evaluate(self, func):
        """
        Evalute the function at each point.

        Parameters
        ----------
        func : function
            The function to evaluate. The function takes a single 1D-array
            argument as input and returns an 1D-array or the result.

        Returns
        -------
        result : 1D-array
            The result of evaluating the function at each point.
        """
        x = self.points
        return x, func(x)
