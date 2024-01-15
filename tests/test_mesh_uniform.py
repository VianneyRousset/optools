#!/usr/bin/env python

import unittest
import numpy as np
from paramobject import ParametrizedObject, Parameter
from optools.mesh import (
    DivisionUniformMeshLine,
    DensityUniformMeshLine,
    ResolutionUniformMeshLine,
)


class TestDivisionUniformMeshLine(unittest.TestCase):
    """Test DivisionUniformMeshLine."""

    def test_evalute(self):
        def func(x):
            return x**2

        mesh = DivisionUniformMeshLine(
            start=5,
            stop=10,
            division=11,
        )

        x, y = mesh.evaluate(func)

        self.assertTrue(np.all(x == np.linspace(5, 10, 11)))
        self.assertTrue(np.all(y == np.linspace(5, 10, 11) ** 2))

    def test_division_mesh(self):
        mesh = DivisionUniformMeshLine(
            start=5,
            stop=10,
            division=10,
        )

        # check length
        self.assertEqual(mesh.length, 5)

        # check generated points
        self.assertTrue(np.all(mesh.points == np.linspace(5, 10, 10)))

        # check retrieved density and resolution
        self.assertAlmostEqual(mesh.density, 2)
        self.assertAlmostEqual(mesh.resolution, 0.5)

        # check with_start and with_stop
        self.assertIsInstance(mesh.with_start(4), DivisionUniformMeshLine)
        self.assertIsInstance(mesh.with_stop(11), DivisionUniformMeshLine)
        self.assertEqual(mesh.with_start(4).start, 4)
        self.assertEqual(mesh.with_stop(11).stop, 11)

        # check with_division
        self.assertIsInstance(mesh.with_division(1000), DivisionUniformMeshLine)
        self.assertEqual(mesh.with_division(1000).division, 1000)

        # check with_division (min/max)
        self.assertEqual(mesh.with_division(min_division=5).division, 10)
        self.assertEqual(mesh.with_division(min_division=100).division, 100)
        self.assertEqual(mesh.with_division(max_division=100).division, 10)
        self.assertEqual(mesh.with_division(max_division=5).division, 5)

        # check with_density and with_resolution
        self.assertIsInstance(mesh.with_density(100), DensityUniformMeshLine)
        self.assertIsInstance(mesh.with_resolution(1), ResolutionUniformMeshLine)
        self.assertEqual(mesh.with_density(100).density, 100)
        self.assertEqual(mesh.with_resolution(1).resolution, 1)

    def test_binded_division_mesh(self):
        # class having a mesh
        class TestObject(ParametrizedObject):
            mesh = Parameter(
                default=DivisionUniformMeshLine(
                    start=5,
                    stop=10,
                    division=11,
                ),
            )

        obj = TestObject()

        # check with_division
        self.assertIsInstance(obj.mesh.with_division(1000), TestObject)
        self.assertIsInstance(
            obj.mesh.with_division(1000).mesh, DivisionUniformMeshLine
        )
        self.assertEqual(obj.mesh.with_division(1000).mesh.division, 1000)

        # check with_density and with_resolution
        self.assertIsInstance(obj.mesh.with_density(100), TestObject)
        self.assertIsInstance(obj.mesh.with_resolution(1), TestObject)
        self.assertIsInstance(obj.mesh.with_density(100).mesh, DensityUniformMeshLine)
        self.assertIsInstance(
            obj.mesh.with_resolution(1).mesh, ResolutionUniformMeshLine
        )
        self.assertEqual(obj.mesh.with_density(100).mesh.density, 100)
        self.assertEqual(obj.mesh.with_resolution(1).mesh.resolution, 1)


class TestDensityUniformMeshLine(unittest.TestCase):
    """Test DensityUniformMeshLine."""

    def test_evalute(self):
        def func(x):
            return x**2

        mesh = DensityUniformMeshLine(
            start=5,
            stop=10,
            density=8.0,
        )

        x, y = mesh.evaluate(func)

        self.assertTrue(np.all(1 / np.max(np.diff(x)) <= 8))
        self.assertTrue(np.all(y == x**2))

    def test_density_mesh(self):
        mesh = DensityUniformMeshLine(
            start=5,
            stop=10,
            density=8.0,
        )

        # check retrieved density and resolution
        self.assertGreaterEqual(mesh.division, 5 * 8)
        self.assertLessEqual(mesh.resolution, 1 / 8)

        # check with_start and with_stop
        self.assertIsInstance(mesh.with_start(4), DensityUniformMeshLine)
        self.assertIsInstance(mesh.with_stop(11), DensityUniformMeshLine)
        self.assertEqual(mesh.with_start(4).start, 4)
        self.assertEqual(mesh.with_stop(11).stop, 11)

        # check with_density
        self.assertIsInstance(mesh.with_density(100), DensityUniformMeshLine)
        self.assertEqual(mesh.with_density(100).density, 100)

        # check with_density (min/max)
        self.assertEqual(mesh.with_density(min_density=5).density, 8)
        self.assertEqual(mesh.with_density(min_density=100).density, 100)
        self.assertEqual(mesh.with_density(max_density=100).density, 8)
        self.assertEqual(mesh.with_density(max_density=5).density, 5)

        # check with_division and with_resolution
        self.assertIsInstance(mesh.with_division(1000), DivisionUniformMeshLine)
        self.assertIsInstance(mesh.with_resolution(1), ResolutionUniformMeshLine)
        self.assertEqual(mesh.with_division(1000).division, 1000)
        self.assertEqual(mesh.with_resolution(1).resolution, 1)

    def test_binded_density_mesh(self):
        # class having a mesh
        class TestObject(ParametrizedObject):
            mesh = Parameter(
                default=DensityUniformMeshLine(
                    start=5,
                    stop=10,
                    density=11,
                ),
            )

        obj = TestObject()

        # check with_density
        self.assertIsInstance(obj.mesh.with_density(100), TestObject)
        self.assertIsInstance(obj.mesh.with_density(100).mesh, DensityUniformMeshLine)
        self.assertEqual(obj.mesh.with_density(100).mesh.density, 100)

        # check with_division and with_resolution
        self.assertIsInstance(obj.mesh.with_division(1000), TestObject)
        self.assertIsInstance(obj.mesh.with_resolution(1), TestObject)
        self.assertIsInstance(
            obj.mesh.with_division(1000).mesh, DivisionUniformMeshLine
        )
        self.assertIsInstance(
            obj.mesh.with_resolution(1).mesh, ResolutionUniformMeshLine
        )
        self.assertEqual(obj.mesh.with_division(1000).mesh.division, 1000)
        self.assertEqual(obj.mesh.with_resolution(1).mesh.resolution, 1)


class TestResolutionUniformMeshLine(unittest.TestCase):
    """Test ResolutionUniformMeshLine."""

    def test_evalute(self):
        def func(x):
            return x**2

        mesh = ResolutionUniformMeshLine(
            start=5,
            stop=10,
            resolution=0.5,
        )

        x, y = mesh.evaluate(func)

        self.assertTrue(np.all(np.max(np.diff(x)) <= 0.5))
        self.assertTrue(np.all(y == x**2))

    def test_resolution_mesh(self):
        mesh = ResolutionUniformMeshLine(
            start=5,
            stop=10,
            resolution=0.5,
        )

        # check retrieved density and resolution
        self.assertGreaterEqual(mesh.division, 5 / 0.5)
        self.assertGreaterEqual(mesh.density, 1 / 0.5)

        # check with_start and with_stop
        self.assertIsInstance(mesh.with_start(4), ResolutionUniformMeshLine)
        self.assertIsInstance(mesh.with_stop(11), ResolutionUniformMeshLine)
        self.assertEqual(mesh.with_start(4).start, 4)
        self.assertEqual(mesh.with_stop(11).stop, 11)

        # check with_resolution
        self.assertIsInstance(mesh.with_resolution(0.1), ResolutionUniformMeshLine)
        self.assertEqual(mesh.with_resolution(0.1).resolution, 0.1)

        # check with_resolution (min/max)
        self.assertEqual(mesh.with_resolution(min_resolution=0.1).resolution, 0.5)
        self.assertEqual(mesh.with_resolution(min_resolution=2).resolution, 2)
        self.assertEqual(mesh.with_resolution(max_resolution=2).resolution, 0.5)
        self.assertEqual(mesh.with_resolution(max_resolution=0.1).resolution, 0.1)

        # check with_division and with_density
        self.assertIsInstance(mesh.with_division(1000), DivisionUniformMeshLine)
        self.assertIsInstance(mesh.with_density(12), DensityUniformMeshLine)
        self.assertEqual(mesh.with_division(1000).division, 1000)
        self.assertEqual(mesh.with_density(12).density, 12)

    def test_binded_resolution_mesh(self):
        # class having a mesh
        class TestObject(ParametrizedObject):
            mesh = Parameter(
                default=ResolutionUniformMeshLine(
                    start=5,
                    stop=10,
                    resolution=0.5,
                ),
            )

        obj = TestObject()

        # check with_resolution
        self.assertIsInstance(obj.mesh.with_resolution(0.1), TestObject)
        self.assertIsInstance(
            obj.mesh.with_resolution(0.1).mesh, ResolutionUniformMeshLine
        )
        self.assertEqual(obj.mesh.with_resolution(0.1).mesh.resolution, 0.1)

        # check with_division and with_density
        self.assertIsInstance(obj.mesh.with_division(1000), TestObject)
        self.assertIsInstance(obj.mesh.with_density(1), TestObject)
        self.assertIsInstance(
            obj.mesh.with_division(1000).mesh, DivisionUniformMeshLine
        )
        self.assertIsInstance(obj.mesh.with_density(1).mesh, DensityUniformMeshLine)
        self.assertEqual(obj.mesh.with_division(1000).mesh.division, 1000)
        self.assertEqual(obj.mesh.with_density(1).mesh.density, 1)
