"""
Definition of adaptive mesh line.

Adaptive line choose points during the evolaution of the function.

It can be either driven by division (number of points) or by loss.
"""

from optools.utils import clip
from .base import MeshLine

from paramobject import Parameter
from abc import abstractmethod
import adaptive


class AdaptiveMeshLine(MeshLine):
    """
    Abstract line mesh with points adapated to the evaluated function.

    Parameters
    ----------
    start : float
        The starting point of the line.
    stop : float
        The ending point of the line.
    division : int
        The number of points.
    loss : float
        Goal loss for the adaptive algorithm.

    Properties
    ----------
    length : float
        The length of the line.

    Methods
    -------
    evaluate(self, func) -> 1D-array:
        Evaluate the given function at each point.
    """

    # number of points
    division = Parameter(default=1000)

    # goal loss for the adaptive algorithm
    loss = Parameter(default=0.01)

    @abstractmethod
    def _run_learner(self, learner):
        raise NotImplementedError()

    def evaluate(self, func):
        learner = adaptive.Learner1D(
            function=func,
            bounds=(self.start, self.stop),
        )

        self._run_learner(learner)

        return learner.to_numpy().T

    def with_params(self, *args, **kwargs):
        params = dict(*args, **kwargs)

        if "start" not in params:
            params["start"] = self.start

        if "stop" not in params:
            params["stop"] = self.stop

        if "division" in params:
            newobj = DivisionAdaptiveMeshLine(**params)

        elif "loss" in params:
            newobj = LossAdaptiveMeshLine(**params)

        else:
            newobj = self.__class__(self, **params)

        if self.binding is not None:
            return self.binding.transfer(newobj)

        return newobj

    @division.wither
    def with_division(self, division=None, *, min_division=None, max_division=None):
        if division is None:
            division = self.division

        division = clip(division, min_division, max_division)

        return self.with_params(division=division)

    @loss.wither
    def with_loss(self, loss=None, *, min_loss=None, max_loss=None):
        if loss is None:
            loss = self.loss

        loss = clip(loss, min_loss, max_loss)

        return self.with_params(loss=loss)


class DivisionAdaptiveMeshLine(AdaptiveMeshLine):
    """
    Adaptive line mesh driven by division.

    Parameters
    ----------
    start : float
        The starting point of the line.
    stop : float
        The ending point of the line.
    division : int
        The number of points.
    loss : float
        Goal loss for the adaptive algorithm.

    Properties
    ----------
    length : float
        The length of the line.

    Methods
    -------
    evaluate(self, func) -> 1D-array:
        Evaluate the given function at each point.
    """

    def _run_learner(self, learner):
        adaptive.runner.simple(
            learner=learner,
            npoints_goal=self.division,
        )


class LossAdaptiveMeshLine(AdaptiveMeshLine):
    """
    Adaptive line mesh driven by loss.

    Parameters
    ----------
    start : float
        The starting point of the line.
    stop : float
        The ending point of the line.
    division : int
        The number of points.
    loss : float
        Goal loss for the adaptive algorithm.

    Properties
    ----------
    length : float
        The length of the line.

    Methods
    -------
    evaluate(self, func) -> 1D-array:
        Evaluate the given function at each point.
    """

    def _run_learner(self, learner):
        adaptive.runner.simple(
            learner=learner,
            loss_goal=self.loss,
        )
