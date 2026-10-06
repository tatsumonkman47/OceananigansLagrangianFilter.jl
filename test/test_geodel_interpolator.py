"""Numerical checks for the optional 2D and 3D GeoDel interpolation map."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "Utils"))
from geodel_interpolator import GeoDelInterpolator2D, GeoDelInterpolator3D


class GeoDelInterpolatorTests(unittest.TestCase):
    def check_dimension(self, dimension, interpolator_type):
        rng = np.random.default_rng(2026 + dimension)
        points = rng.random((300, dimension))
        targets = np.vstack((points[:50], rng.random((80, dimension)),
                             np.full((1, dimension), -1.0)))
        values = np.column_stack((1.5 + points @ np.arange(1, dimension + 1),
                                  -2.0 + points @ np.arange(dimension, 0, -1)))

        actual = interpolator_type(points, targets, nb_threads=2)(values)
        expected = LinearNDInterpolator(points, values)(targets)
        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
        finite = np.isfinite(expected).all(axis=1)
        np.testing.assert_allclose(actual[finite], expected[finite], atol=1e-11, rtol=0)
        self.assertTrue(np.isnan(actual[-1]).all())

    def test_2d(self):
        self.check_dimension(2, GeoDelInterpolator2D)

    def test_3d(self):
        self.check_dimension(3, GeoDelInterpolator3D)


if __name__ == "__main__":
    unittest.main()
