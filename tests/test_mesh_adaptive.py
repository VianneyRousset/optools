#!/usr/bin/env python

import unittest
import numpy as np
from paramobject import ParametrizedObject, Parameter
from optools.mesh import DivisionAdaptiveMeshLine, LossAdaptiveMeshLine


class TestDivisionAdaptiveMeshLine(unittest.TestCase):
    """Test DivisionAdaptiveMeshLine."""

    def test_evalute(self):
        mesh = DivisionAdaptiveMeshLine(
            start=-1,
            stop=1,
            division=1000,
        )

        def func(x, a=0.01):
            return x + a**2 / (a**2 + x**2)

        x, y = mesh.evaluate(func)

        self.assertTrue(np.all(y == func(x)))

        # check higher density at the peak position
        mask = (x > -0.1) & (x < 0.1)
        self.assertGreater(np.sum(mask) / 0.2, np.sum(~mask) / (2 - 0.2))

    def test_division_mesh(self):
        mesh = DivisionAdaptiveMeshLine(
            start=-1,
            stop=1,
            division=1000,
        )

        # check length
        self.assertEqual(mesh.length, 2)

        # check retrieved division
        self.assertAlmostEqual(mesh.division, 1000)

        # check with_start and with_stop
        self.assertIsInstance(mesh.with_start(4), DivisionAdaptiveMeshLine)
        self.assertIsInstance(mesh.with_stop(11), DivisionAdaptiveMeshLine)
        self.assertEqual(mesh.with_start(4).start, 4)
        self.assertEqual(mesh.with_stop(11).stop, 11)

        # check with_division
        self.assertIsInstance(mesh.with_division(100), DivisionAdaptiveMeshLine)
        self.assertEqual(mesh.with_division(100).division, 100)

        # check with_division (min/max)
        self.assertEqual(mesh.with_division(min_division=100).division, 1000)
        self.assertEqual(mesh.with_division(min_division=2000).division, 2000)
        self.assertEqual(mesh.with_division(max_division=2000).division, 1000)
        self.assertEqual(mesh.with_division(max_division=100).division, 100)

        # check with_loss
        self.assertIsInstance(mesh.with_loss(0.01), LossAdaptiveMeshLine)
        self.assertEqual(mesh.with_loss(0.01).loss, 0.01)

    def test_binded_division_mesh(self):
        # class having a mesh
        class TestObject(ParametrizedObject):
            mesh = Parameter(
                default=DivisionAdaptiveMeshLine(
                    start=-1,
                    stop=1,
                    division=1000,
                ),
            )

        obj = TestObject()

        # check with_division
        self.assertIsInstance(obj.mesh.with_division(2000), TestObject)
        self.assertIsInstance(
            obj.mesh.with_division(2000).mesh, DivisionAdaptiveMeshLine
        )
        self.assertEqual(obj.mesh.with_division(2000).mesh.division, 2000)

        # check with_loss
        self.assertIsInstance(obj.mesh.with_loss(0.01), TestObject)
        self.assertIsInstance(obj.mesh.with_loss(0.01).mesh, LossAdaptiveMeshLine)
        self.assertEqual(obj.mesh.with_loss(0.01).mesh.loss, 0.01)


class TestLossAdaptiveMeshLine(unittest.TestCase):
    """Test LossAdaptiveMeshLine."""

    def test_evalute(self):
        mesh = LossAdaptiveMeshLine(
            start=-1,
            stop=1,
            loss=0.01,
        )

        def func(x, a=0.01):
            return x + a**2 / (a**2 + x**2)

        x, y = mesh.evaluate(func)

        self.assertTrue(np.all(y == func(x)))

        # check higher density at the peak position
        mask = (x > -0.1) & (x < 0.1)
        self.assertGreater(np.sum(mask) / 0.2, np.sum(~mask) / (2 - 0.2))

    def test_loss_mesh(self):
        mesh = LossAdaptiveMeshLine(
            start=-1,
            stop=1,
            loss=0.01,
        )

        # check length
        self.assertEqual(mesh.length, 2)

        # check retrieved loss
        self.assertAlmostEqual(mesh.loss, 0.01)

        # check with_start and with_stop
        self.assertIsInstance(mesh.with_start(4), LossAdaptiveMeshLine)
        self.assertIsInstance(mesh.with_stop(11), LossAdaptiveMeshLine)
        self.assertEqual(mesh.with_start(4).start, 4)
        self.assertEqual(mesh.with_stop(11).stop, 11)

        # check with_loss
        self.assertIsInstance(mesh.with_loss(0.01), LossAdaptiveMeshLine)
        self.assertEqual(mesh.with_loss(0.01).loss, 0.01)

        # check with_loss (min/max)
        self.assertEqual(mesh.with_loss(min_loss=0.005).loss, 0.01)
        self.assertEqual(mesh.with_loss(min_loss=0.1).loss, 0.1)
        self.assertEqual(mesh.with_loss(max_loss=0.1).loss, 0.01)
        self.assertEqual(mesh.with_loss(max_loss=0.005).loss, 0.005)

        # check with_division
        self.assertIsInstance(mesh.with_division(1000), DivisionAdaptiveMeshLine)
        self.assertEqual(mesh.with_division(1000).division, 1000)

    def test_binded_division_mesh(self):
        # class having a mesh
        class TestObject(ParametrizedObject):
            mesh = Parameter(
                default=LossAdaptiveMeshLine(
                    start=-1,
                    stop=1,
                    loss=0.01,
                ),
            )

        obj = TestObject()

        # check with_loss
        self.assertIsInstance(obj.mesh.with_loss(0.01), TestObject)
        self.assertIsInstance(obj.mesh.with_loss(0.01).mesh, LossAdaptiveMeshLine)
        self.assertEqual(obj.mesh.with_loss(0.01).mesh.loss, 0.01)

        # check with_division
        self.assertIsInstance(obj.mesh.with_division(2000), TestObject)
        self.assertIsInstance(
            obj.mesh.with_division(2000).mesh, DivisionAdaptiveMeshLine
        )
        self.assertEqual(obj.mesh.with_division(2000).mesh.division, 2000)
