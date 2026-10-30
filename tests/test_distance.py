"""Unit tests for VectorForge distance metrics and vector math."""

import math
import unittest
from vectorforge.distance import (
    euclidean_distance,
    cosine_distance,
    dot_product,
    normalize,
    norm_l2,
    project_to_3d,
)


class TestDistanceMetrics(unittest.TestCase):
    def test_euclidean_distance(self):
        u = [1.0, 2.0, 3.0]
        v = [4.0, 6.0, 8.0]
        # sqrt(3^2 + 4^2 + 5^2) = sqrt(9 + 16 + 25) = sqrt(50) = 7.0710678
        expected = math.sqrt(50)
        self.assertAlmostEqual(euclidean_distance(u, v), expected, places=5)

    def test_euclidean_identical(self):
        u = [0.5, -1.2, 3.4]
        self.assertAlmostEqual(euclidean_distance(u, u), 0.0, places=6)

    def test_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            euclidean_distance([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_dot_product(self):
        u = [1.0, 2.0, 3.0]
        v = [4.0, -5.0, 6.0]
        # 1*4 + 2*(-5) + 3*6 = 4 - 10 + 18 = 12
        self.assertAlmostEqual(dot_product(u, v), 12.0, places=5)

    def test_cosine_distance_identical(self):
        u = [2.0, 4.0, 6.0]
        # Parallel vectors should have cosine distance 0
        self.assertAlmostEqual(cosine_distance(u, [1.0, 2.0, 3.0]), 0.0, places=5)

    def test_cosine_distance_orthogonal(self):
        u = [1.0, 0.0, 0.0]
        v = [0.0, 1.0, 0.0]
        # Orthogonal vectors should have cosine distance 1.0
        self.assertAlmostEqual(cosine_distance(u, v), 1.0, places=5)

    def test_cosine_distance_opposite(self):
        u = [1.0, 0.0, 0.0]
        v = [-1.0, 0.0, 0.0]
        # Opposite vectors should have cosine distance 2.0
        self.assertAlmostEqual(cosine_distance(u, v), 2.0, places=5)

    def test_normalize(self):
        u = [3.0, 4.0]
        normed = normalize(u)
        self.assertAlmostEqual(norm_l2(normed), 1.0, places=6)
        self.assertAlmostEqual(normed[0], 0.6, places=5)

        self.assertAlmostEqual(normed[1], 0.8, places=5)

    def test_project_to_3d(self):
        u = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        p1 = project_to_3d(u, seed=42)
        p2 = project_to_3d(u, seed=42)
        self.assertEqual(len(p1), 3)
        self.assertEqual(p1, p2)  # Deterministic projection


if __name__ == "__main__":
    unittest.main()
