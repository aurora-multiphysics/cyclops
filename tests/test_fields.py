"""
Tests for the cyclops Fields class.

(c) Copyright UKAEA 2023.
"""
import sys
import os

# Sort the paths out to run from this file
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(os.path.sep, parent_path, "src")
sys.path.append(src_path)

from fields import ScalarField, VectorField
from regressors import CSModel, LModel
import numpy as np
import unittest


class TestFields(unittest.TestCase):
    """Tests for Fields."""

    def test_scalar_field_plane_1(self):
        """Test 2D planar scalar field."""
        grid = []
        for x in np.linspace(-10, 10, 20):
            for y in np.linspace(-10, 10, 20):
                grid.append(np.array([x, y]))
        grid = np.array(grid)
        scalars = np.sum(grid, axis=1).reshape(-1, 1)
        scalars = np.square(scalars)

        new_grid = []
        for x in np.linspace(-10, 10, 40):
            for y in np.linspace(-10, 10, 40):
                new_grid.append(np.array([x, y]))
        new_grid = np.array(new_grid)
        new_scalars = np.sum(new_grid, axis=1).reshape(-1, 1)
        new_scalars = np.square(new_scalars)

        scalar_plane = ScalarField(LModel, np.array([[-10, -10], [10, 10]]))
        scalar_plane.fit_model(grid, scalars)
        test_scalars = scalar_plane.predict_values(new_grid)

        for i, scalar in enumerate(new_scalars):
            self.assertAlmostEqual(scalar[0], test_scalars[i, 0], -1)

    def test_scalar_field_plane_2(self):
        """Test 2D planar scalar field."""
        grid = []
        for x in np.linspace(-10, 10, 20):
            for y in np.linspace(-10, 10, 20):
                grid.append(np.array([x, y]))
        grid = np.array(grid)
        scalars = np.sum(grid, axis=1).reshape(-1, 1)
        scalars = np.sin(scalars)

        new_grid = []
        for x in np.linspace(-10, 10, 40):
            for y in np.linspace(-10, 10, 40):
                new_grid.append(np.array([x, y]))
        new_grid = np.array(new_grid)
        new_scalars = np.sum(new_grid, axis=1).reshape(-1, 1)
        new_scalars = np.sin(new_scalars)

        scalar_plane = ScalarField(LModel, np.array([[-10, -10], [10, 10]]))
        scalar_plane.fit_model(grid, scalars)
        test_scalars = scalar_plane.predict_values(new_grid)

        for i, scalar in enumerate(new_scalars):
            self.assertAlmostEqual(scalar[0], test_scalars[i, 0], -1)

    def test_scalar_field_line_1(self):
        """Test 1D linear scalar field."""
        line = np.linspace(-10, 10, 20).reshape(-1, 1)
        scalars = np.sum(line, axis=1).reshape(-1, 1)
        scalars = np.square(scalars)

        new_line = np.linspace(-10, 10, 40).reshape(-1, 1)
        new_scalars = np.sum(new_line, axis=1).reshape(-1, 1)
        new_scalars = np.square(new_scalars)

        scalar_line = ScalarField(CSModel, np.array([[-10], [10]]))
        scalar_line.fit_model(line, scalars)
        test_scalars = scalar_line.predict_values(new_line)

        for i, scalar in enumerate(new_scalars):
            self.assertAlmostEqual(scalar[0], test_scalars[i, 0], -1)

    def test_scalar_field_line_2(self):
        """Test 1D linear scalar field."""
        line = np.linspace(-10, 10, 20).reshape(-1, 1)
        scalars = np.sum(line, axis=1).reshape(-1, 1)
        scalars = np.sin(scalars)

        new_line = np.linspace(-10, 10, 40).reshape(-1, 1)
        new_scalars = np.sum(new_line, axis=1).reshape(-1, 1)
        new_scalars = np.sin(new_scalars)

        scalar_line = ScalarField(CSModel, np.array([[-10], [10]]))
        scalar_line.fit_model(line, scalars)
        test_scalars = scalar_line.predict_values(new_line)

        for i, scalar in enumerate(new_scalars):
            self.assertAlmostEqual(scalar[0], test_scalars[i, 0], -1)

    def test_vector_field_plane_1(self):
        """Test 2D planar vector field."""
        grid = []
        for x in np.linspace(-10, 10, 20):
            for y in np.linspace(-10, 10, 20):
                grid.append(np.array([x, y]))
        grid = np.array(grid)

        vectors = []
        for pos in grid:
            vectors.append(np.array([pos[0], pos[1], pos[0] + pos[1]]))
        vectors = np.array(vectors)

        new_grid = []
        for x in np.linspace(-10, 10, 40):
            for y in np.linspace(-10, 10, 40):
                new_grid.append(np.array([x, y]))
        new_grid = np.array(grid)

        new_vectors = []
        for pos in new_grid:
            new_vectors.append(np.array([pos[0], pos[1], pos[0] + pos[1]]))
        new_vectors = np.array(new_vectors)

        vector_plane = VectorField(LModel, np.array([[-10, -10], [10, 10]]))
        vector_plane.fit_model(grid, vectors)
        test_vectors = vector_plane.predict_values(new_grid)

        for i, vector in enumerate(new_vectors):
            self.assertAlmostEqual(vector[0], test_vectors[i, 0], -1)
            self.assertAlmostEqual(vector[1], test_vectors[i, 1], -1)
            self.assertAlmostEqual(vector[2], test_vectors[i, 2], -1)

    def test_vector_field_plane_2(self):
        """Test 2D planar vector field."""
        grid = []
        for x in np.linspace(-10, 10, 20):
            for y in np.linspace(-10, 10, 20):
                grid.append(np.array([x, y]))
        grid = np.array(grid)

        vectors = []
        for pos in grid:
            vectors.append(np.array([pos[0], pos[1], pos[0] * pos[1]]))
        vectors = np.array(vectors)

        new_grid = []
        for x in np.linspace(-10, 10, 40):
            for y in np.linspace(-10, 10, 40):
                new_grid.append(np.array([x, y]))
        new_grid = np.array(grid)

        new_vectors = []
        for pos in new_grid:
            new_vectors.append(np.array([pos[0], pos[1], pos[0] * pos[1]]))
        new_vectors = np.array(new_vectors)

        vector_plane = VectorField(LModel, np.array([[-10, -10], [10, 10]]))
        vector_plane.fit_model(grid, vectors)
        test_vectors = vector_plane.predict_values(new_grid)

        for i, vector in enumerate(new_vectors):
            self.assertAlmostEqual(vector[0], test_vectors[i, 0], -1)
            self.assertAlmostEqual(vector[1], test_vectors[i, 1], -1)
            self.assertAlmostEqual(vector[2], test_vectors[i, 2], -1)

    def test_vector_field_l1(self):
        """Test 1D linear vector field."""
        line = np.linspace(-10, 10, 20).reshape(-1, 1)
        vectors = []
        for x in line:
            vectors.append(np.array([x[0], x[0] ** 2, x[0] + 5]))
        vectors = np.array(vectors)

        new_line = np.linspace(-10, 10, 40).reshape(-1, 1)
        new_vectors = []
        for x in new_line:
            new_vectors.append(np.array([x[0], x[0] ** 2, x[0] + 5]))
        new_vectors = np.array(new_vectors)

        vector_line = VectorField(CSModel, np.array([[-10], [10]]))
        vector_line.fit_model(line, vectors)
        test_vectors = vector_line.predict_values(new_line)

        for i, vector in enumerate(new_vectors):
            self.assertAlmostEqual(vector[0], test_vectors[i, 0], -1)
            self.assertAlmostEqual(vector[1], test_vectors[i, 1], -1)
            self.assertAlmostEqual(vector[2], test_vectors[i, 2], -1)

    def test_vector_field_l2(self):
        """Test 1D linear vector field."""
        line = np.linspace(-10, 10, 20).reshape(-1, 1)
        vectors = []
        for x in line:
            vectors.append(np.array([np.sin(x[0]), x[0] ** 2, x[0] * 5]))
        vectors = np.array(vectors)

        new_line = np.linspace(-10, 10, 40).reshape(-1, 1)
        new_vectors = []
        for x in new_line:
            new_vectors.append(np.array([np.sin(x[0]), x[0] ** 2, x[0] * 5]))
        new_vectors = np.array(new_vectors)

        vector_line = VectorField(CSModel, np.array([[-10], [10]]))
        vector_line.fit_model(line, vectors)
        test_vectors = vector_line.predict_values(new_line)

        for i, vector in enumerate(new_vectors):
            self.assertAlmostEqual(vector[0], test_vectors[i, 0], -1)
            self.assertAlmostEqual(vector[1], test_vectors[i, 1], -1)
            self.assertAlmostEqual(vector[2], test_vectors[i, 2], -1)


if __name__ == "__main__":
    unittest.main()
