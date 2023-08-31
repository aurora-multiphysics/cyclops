import sys
import os

# Sort the paths out to run from this file
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(os.path.sep,parent_path, 'src')
sys.path.append(src_path)

from regressors import RBFModel, LModel, GPModel, PModel, CSModel, CTModel
import numpy as np
import unittest
import meshio




class Test(unittest.TestCase):
    def setUp(self):
        pass
    

    def test_rbf_1D(self):
        # Test interpolation 1
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = x
        rbf_model = RBFModel(1)
        rbf_model.fit(x, y)

        new_x = np.linspace(0, 10, 20).reshape(-1, 1)
        new_y = new_x
        test_y = rbf_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], 2)
        
        # Test interpolation 2
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.square(x)
        rbf_model = RBFModel(1)
        rbf_model.fit(x, y)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = np.square(new_x)
        test_y = rbf_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], 2)
        
        
        # Test interpolation 3
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.sqrt(x)
        rbf_model = RBFModel(1)
        rbf_model.fit(x, y)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = np.sqrt(new_x)
        test_y = rbf_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], 2)

        
        # Test interpolation 4
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.sin(x)
        rbf_model = RBFModel(1)
        rbf_model.fit(x, y)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = np.sin(new_x)
        test_y = rbf_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], 2)

        
        # Test extrapolation 1
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = x
        rbf_model = RBFModel(1)
        rbf_model.fit(x, y)

        new_x = np.linspace(-2, 12, 40).reshape(-1, 1)
        new_y = new_x
        test_y = rbf_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], -2)


        # Test extrapolation 2
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.square(x)
        rbf_model = RBFModel(1)
        rbf_model.fit(x, y)

        new_x = np.linspace(-2, 12, 40).reshape(-1, 1)
        new_y = np.square(new_x)
        test_y = rbf_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], -2)

        
        # Test extrapolation 3
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.sin(x)
        rbf_model = RBFModel(1)
        rbf_model.fit(x, y)

        new_x = np.linspace(-2, 12, 40).reshape(-1, 1)
        new_y = np.sin(new_x)
        test_y = rbf_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], -2)


        # Test error catching 1
        error_1 = 'Input data should have d >= 1 dimensions.'
        with self.assertRaises(Exception) as context:
            rbf_model = RBFModel(0)
        self.assertTrue(error_1 in str(context.exception))

        with self.assertRaises(Exception) as context:
            rbf_model = RBFModel(-1)
        self.assertTrue(error_1 in str(context.exception))

        with self.assertRaises(Exception) as context:
            rbf_model = RBFModel(-10)
        self.assertTrue(error_1 in str(context.exception))

        rbf_model = RBFModel(1)
        rbf_model = RBFModel(5)
        rbf_model = RBFModel(10)

        
        # Test error catching 2
        error_2 = 'Input data should be a numpy array of shape (-1, 1).'
        with self.assertRaises(Exception) as context:
            rbf_model = RBFModel(1)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            xy = np.concatenate((x, y), axis=1)
            z = np.sum(xy, axis=1)
            rbf_model.fit(xy, z)
        self.assertTrue(error_2 in str(context.exception))

        error_2 = 'Input data should be a numpy array of shape (-1, 2).'
        with self.assertRaises(Exception) as context:
            rbf_model = RBFModel(2)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            rbf_model.fit(x, y)
        self.assertTrue(error_2 in str(context.exception))
        

        # Test error catching 3
        error_3 = 'Input data should be '
        with self.assertRaises(Exception) as context:
            rbf_model = RBFModel(1)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            xy = np.concatenate((x, y), axis=1)
            z = np.sum(xy, axis=1)
            rbf_model.fit(xy, z)
        self.assertTrue(error_2 in str(context.exception))






if __name__ == "__main__":
    unittest.main()