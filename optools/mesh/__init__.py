"""
Definition of mesh lines.

A mesh line is a line that evaluates a function at points.

>>> def peak(x, a=0.01):
...     return x + a**2 / (a**2 + x**2)
...
... mesh = UniformMeshLine(start=0, stop=10, division=100)
... x, y = mesh.evaluate(peak)

A mesh can either be constant or adaptive:
- Constant meshes have points defined before function evaluation.
- Adaptive line choose points during the evolaution of the function.
"""

__all__ = [
    "MeshLine",
    "ConstantMeshLine",
    "UniformMeshLine",
    "DivisionUniformMeshLine",
    "DensityUniformMeshLine",
    "ResolutionUniformMeshLine",
    "AdaptiveMeshLine",
    "DivisionAdaptiveMeshLine",
    "LossAdaptiveMeshLine",
]

from .base import MeshLine
from .constant import ConstantMeshLine
from .uniform import (
    DivisionUniformMeshLine,
    DensityUniformMeshLine,
    ResolutionUniformMeshLine,
)
from .adaptive import (
    AdaptiveMeshLine,
    DivisionAdaptiveMeshLine,
    LossAdaptiveMeshLine,
)
