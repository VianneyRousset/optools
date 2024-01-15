"""
Definition of MeshLine an abstract mesh line.
"""

from paramobject import ParametrizedObject, Parameter
from abc import abstractmethod


class MeshLine(ParametrizedObject):
    """
    Abstract class of a line with points.

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

    Methods
    -------
    evaluate(self, func) -> 1D-array:
        Evaluate the given function at each point.
    """

    start = Parameter()
    stop = Parameter()

    @property
    def length(self):
        """The length of the line."""
        return abs(self.stop - self.start)

    @abstractmethod
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
        raise NotImplementedError()
