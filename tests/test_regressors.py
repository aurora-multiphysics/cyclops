import sys
import os

# Sort the paths out to run from this file
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(os.path.sep,parent_path, 'src')
sys.path.append(src_path)

from regressors import RBFModel, LModel, GPModel, PModel, CSModel, CTModel
import numpy as np
import unittest


# Key
# i means interpolation test
# r means extrapolation test
# e means exception handling test


class Test(unittest.TestCase):
    def test_rbf_model_i1(self):
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
        
    
    def test_rbf_model_i2(self):
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
        
    
    def test_rbf_model_i3(self):
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

    
    def test_rbf_model_i4(self):
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

    
    def test_rbf_model_r1(self):
        # Test extrapolation 1
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = x
        rbf_model = RBFModel(1)
        rbf_model.fit(x, y)

        new_x = np.linspace(-2, 12, 40).reshape(-1, 1)
        new_y = new_x
        test_y = rbf_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], 2)


    def test_rbf_model_r2(self):
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

    
    def test_rbf_model_r3(self):
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


    def test_rbf_model_e1(self):
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

    
    def test_rbf_model_e2(self):
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
        

    def test_rbf_model_e3(self):
        # Test error catching 3
        error_3 = 'Output data should be a numpy array of shape (-1, 1).'
        with self.assertRaises(Exception) as context:
            rbf_model = RBFModel(2)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            xy = np.concatenate((x, y), axis=1)
            z = xy.copy()
            rbf_model.fit(xy, z)
        self.assertTrue(error_3 in str(context.exception))


    def test_rbf_model_e4(self):
        # Test error catching 4
        error_4 = 'Input data should have a length of >= 2.'
        with self.assertRaises(Exception) as context:
            rbf_model = RBFModel(2)
            x = np.array([[1, 1]])
            y = np.array([[1]])
            rbf_model.fit(x, y)
        self.assertTrue(error_4 in str(context.exception)) 


    def test_rbf_model_e5(self):
        # Test error catching 5
        error_5 = 'Input data should be a numpy array of shape (-1, 1)'
        with self.assertRaises(Exception) as context:
            rbf_model = RBFModel(1)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            rbf_model.fit(x, y)
            rbf_model.predict([[1, 1], [2, 2]])
        self.assertTrue(error_5 in str(context.exception)) 





    def test_l_model_i1(self):
        # Test interpolation 1
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        all_xy = []
        for x_val in x:
            for y_val in y:
                all_xy.append(np.array([x_val, y_val]))
        xy = np.array(all_xy)
        z = np.sum(xy, axis=1).reshape(-1, 1)

        l_model = LModel(2)
        l_model.fit(xy, z)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = new_x.copy()
        new_xy = np.concatenate((new_x, new_y), axis=1)
        new_z = np.sum(new_xy, axis=1).reshape(-1, 1)
        new_z = new_z

        test_z = l_model.predict(new_xy)
        for i, z_val in enumerate(test_z):
            self.assertAlmostEqual(z_val[0], new_z[i, 0], 3)

    
    def test_l_model_i2(self):
        # Test interpolation 2
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        all_xy = []
        for x_val in x:
            for y_val in y:
                all_xy.append(np.array([x_val, y_val]))
        xy = np.array(all_xy)
        z = np.sum(xy, axis=1).reshape(-1, 1)
        z = np.square(z)

        l_model = LModel(2)
        l_model.fit(xy, z)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = new_x.copy()
        new_xy = np.concatenate((new_x, new_y), axis=1)
        new_z = np.sum(new_xy, axis=1).reshape(-1, 1)
        new_z = np.square(new_z)

        test_z = l_model.predict(new_xy)
        for i, z_val in enumerate(test_z):
            self.assertAlmostEqual(z_val[0], new_z[i, 0], 0)


    def test_l_model_i3(self):
        # Test interpolation 3
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        all_xy = []
        for x_val in x:
            for y_val in y:
                all_xy.append(np.array([x_val, y_val]))
        xy = np.array(all_xy)
        z = np.sum(xy, axis=1).reshape(-1, 1)
        z = np.sqrt(z)

        l_model = LModel(2)
        l_model.fit(xy, z)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = new_x.copy()
        new_xy = np.concatenate((new_x, new_y), axis=1)
        new_z = np.sum(new_xy, axis=1).reshape(-1, 1)
        new_z = np.sqrt(new_z)

        test_z = l_model.predict(new_xy)
        for i, z_val in enumerate(test_z):
            self.assertAlmostEqual(z_val[0], new_z[i, 0], 0)


    def test_l_model_i4(self):
        # Test interpolation 4
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        all_xy = []
        for x_val in x:
            for y_val in y:
                all_xy.append(np.array([x_val, y_val]))
        xy = np.array(all_xy)
        z = np.sum(xy, axis=1).reshape(-1, 1)
        z = np.sin(z)

        l_model = LModel(2)
        l_model.fit(xy, z)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = new_x.copy()
        new_xy = np.concatenate((new_x, new_y), axis=1)
        new_z = np.sum(new_xy, axis=1).reshape(-1, 1)
        new_z = np.sin(new_z)

        test_z = l_model.predict(new_xy)
        for i, z_val in enumerate(test_z):
            self.assertAlmostEqual(z_val[0], new_z[i, 0], 0)
        

    def test_l_model_r1(self):
        # Test extrapolation 1
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)
        all_xy = []
        for x_val in x:
            for y_val in y:
                all_xy.append(np.array([x_val, y_val]))
        xy = np.array(all_xy)
        z = np.sum(xy, axis=1).reshape(-1, 1)
        z = np.sin(z)

        l_model = LModel(2)
        l_model.fit(xy, z)

        new_x = np.linspace(2, 8, 40).reshape(-1, 1)
        new_y = new_x.copy()
        new_xy = np.concatenate((new_x, new_y), axis=1)
        new_z = np.sum(new_xy, axis=1).reshape(-1, 1)
        new_z = np.sin(new_z)

        test_z = l_model.predict(new_xy)
        for i, z_val in enumerate(test_z):
            self.assertEqual(z_val[0], np.mean(z))


    def test_l_model_e1(self):
        # Test error catching 1
        error_1 = 'Input data should have d >= 2 dimensions.'
        with self.assertRaises(Exception) as context:
            l_model = LModel(1)
        self.assertTrue(error_1 in str(context.exception))

        with self.assertRaises(Exception) as context:
            l_model = LModel(-2)
        self.assertTrue(error_1 in str(context.exception))

        with self.assertRaises(Exception) as context:
            l_model = LModel(0)
        self.assertTrue(error_1 in str(context.exception))

        l_model = LModel(2)
        l_model = LModel(5)
        l_model = LModel(10)

    
    def test_l_model_e2(self):
        # Test error catching 2
        error_2 = 'Input data should be a numpy array of shape (-1, 2).'
        with self.assertRaises(Exception) as context:
            l_model = LModel(2)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            l_model.fit(x, x)
        self.assertTrue(error_2 in str(context.exception))
        

    def test_l_model_e3(self):
        # Test error catching 3
        error_3 = 'Output data should be a numpy array of shape (-1, 1).'
        with self.assertRaises(Exception) as context:
            l_model = LModel(2)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            xy = np.concatenate((x, y), axis=1)
            z = xy.copy()
            l_model.fit(xy, z)
        self.assertTrue(error_3 in str(context.exception))


    def test_l_model_e4(self):
        # Test error catching 4
        error_4 = 'Input data should have a length of >= 3.'
        with self.assertRaises(Exception) as context:
            l_model = LModel(2)
            x = np.array([[1, 1]])
            y = np.array([[1]])
            l_model.fit(x, y)
        self.assertTrue(error_4 in str(context.exception)) 


    def test_l_model_e5(self):
        # Test error catching 5
        error_5 = 'Input data should be a numpy array of shape (-1, 2)'
        with self.assertRaises(Exception) as context:
            l_model = LModel(2)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            l_model.fit(x, y)
            l_model.predict([[1], [2]])
        self.assertTrue(error_5 in str(context.exception)) 





    def test_ct_model_i1(self):
        # Test interpolation 1
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        all_xy = []
        for x_val in x:
            for y_val in y:
                all_xy.append(np.array([x_val, y_val]))
        xy = np.array(all_xy)
        z = np.sum(xy, axis=1).reshape(-1, 1)

        ct_model = CTModel(2)
        ct_model.fit(xy, z)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = new_x.copy()
        new_xy = np.concatenate((new_x, new_y), axis=1)
        new_z = np.sum(new_xy, axis=1).reshape(-1, 1)
        new_z = new_z

        test_z = ct_model.predict(new_xy)
        for i, z_val in enumerate(test_z):
            self.assertAlmostEqual(z_val[0], new_z[i, 0], 3)

    
    def test_ct_model_i2(self):
        # Test interpolation 2
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        all_xy = []
        for x_val in x:
            for y_val in y:
                all_xy.append(np.array([x_val, y_val]))
        xy = np.array(all_xy)
        z = np.sum(xy, axis=1).reshape(-1, 1)
        z = np.square(z)

        ct_model = CTModel(2)
        ct_model.fit(xy, z)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = new_x.copy()
        new_xy = np.concatenate((new_x, new_y), axis=1)
        new_z = np.sum(new_xy, axis=1).reshape(-1, 1)
        new_z = np.square(new_z)

        test_z = ct_model.predict(new_xy)
        for i, z_val in enumerate(test_z):
            self.assertAlmostEqual(z_val[0], new_z[i, 0], 0)


    def test_ct_model_i3(self):
        # Test interpolation 3
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        all_xy = []
        for x_val in x:
            for y_val in y:
                all_xy.append(np.array([x_val, y_val]))
        xy = np.array(all_xy)
        z = np.sum(xy, axis=1).reshape(-1, 1)
        z = np.sqrt(z)

        ct_model = CTModel(2)
        ct_model.fit(xy, z)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = new_x.copy()
        new_xy = np.concatenate((new_x, new_y), axis=1)
        new_z = np.sum(new_xy, axis=1).reshape(-1, 1)
        new_z = np.sqrt(new_z)

        test_z = ct_model.predict(new_xy)
        for i, z_val in enumerate(test_z):
            self.assertAlmostEqual(z_val[0], new_z[i, 0], 0)


    def test_ct_model_i4(self):
        # Test interpolation 4
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        all_xy = []
        for x_val in x:
            for y_val in y:
                all_xy.append(np.array([x_val, y_val]))
        xy = np.array(all_xy)
        z = np.sum(xy, axis=1).reshape(-1, 1)
        z = np.sin(z)

        ct_model = CTModel(2)
        ct_model.fit(xy, z)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = new_x.copy()
        new_xy = np.concatenate((new_x, new_y), axis=1)
        new_z = np.sum(new_xy, axis=1).reshape(-1, 1)
        new_z = np.sin(new_z)

        test_z = ct_model.predict(new_xy)
        for i, z_val in enumerate(test_z):
            self.assertAlmostEqual(z_val[0], new_z[i, 0], 0)
        

    def test_ct_model_r1(self):
        # Test extrapolation 1
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)
        all_xy = []
        for x_val in x:
            for y_val in y:
                all_xy.append(np.array([x_val, y_val]))
        xy = np.array(all_xy)
        z = np.sum(xy, axis=1).reshape(-1, 1)
        z = np.sin(z)

        ct_model = CTModel(2)
        ct_model.fit(xy, z)

        new_x = np.linspace(2, 8, 40).reshape(-1, 1)
        new_y = new_x.copy()
        new_xy = np.concatenate((new_x, new_y), axis=1)
        new_z = np.sum(new_xy, axis=1).reshape(-1, 1)
        new_z = np.sin(new_z)

        test_z = ct_model.predict(new_xy)
        for i, z_val in enumerate(test_z):
            self.assertEqual(z_val[0], np.mean(z))


    def test_ct_model_e1(self):
        # Test error catching 1
        error_1 = 'Input data should have d = 2 dimensions.'
        with self.assertRaises(Exception) as context:
            ct_model = CTModel(1)
        self.assertTrue(error_1 in str(context.exception))

        with self.assertRaises(Exception) as context:
            ct_model = CTModel(5)
        self.assertTrue(error_1 in str(context.exception))

        with self.assertRaises(Exception) as context:
            ct_model = CTModel(8)
        self.assertTrue(error_1 in str(context.exception))

        ct_model = CTModel(2)

    
    def test_ct_model_e2(self):
        # Test error catching 2
        error_2 = 'Input data should be a numpy array of shape (-1, 2).'
        with self.assertRaises(Exception) as context:
            ct_model = CTModel(2)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            ct_model.fit(x, x)
        self.assertTrue(error_2 in str(context.exception))
        

    def test_ct_model_e3(self):
        # Test error catching 3
        error_3 = 'Output data should be a numpy array of shape (-1, 1).'
        with self.assertRaises(Exception) as context:
            ct_model = CTModel(2)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            xy = np.concatenate((x, y), axis=1)
            z = xy.copy()
            ct_model.fit(xy, z)
        self.assertTrue(error_3 in str(context.exception))


    def test_ct_model_e4(self):
        # Test error catching 4
        error_4 = 'Input data should have a length of >= 3.'
        with self.assertRaises(Exception) as context:
            ct_model = CTModel(2)
            x = np.array([[1, 1]])
            y = np.array([[1]])
            ct_model.fit(x, y)
        self.assertTrue(error_4 in str(context.exception)) 


    def test_ct_model_e5(self):
        # Test error catching 5
        error_5 = 'Input data should be a numpy array of shape (-1, 2)'
        with self.assertRaises(Exception) as context:
            ct_model = CTModel(2)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            ct_model.fit(x, y)
            ct_model.predict([[1], [2]])
        self.assertTrue(error_5 in str(context.exception)) 





    def test_cs_model_i1(self):
        # Test interpolation 1
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = x
        cs_model = CSModel(1)
        cs_model.fit(x, y)

        new_x = np.linspace(0, 10, 20).reshape(-1, 1)
        new_y = new_x
        test_y = cs_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], 2)
        
    def test_cs_model_i2(self):
        # Test interpolation 2
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.square(x)
        cs_model = CSModel(1)
        cs_model.fit(x, y)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = np.square(new_x)
        test_y = cs_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], 2)
        
        
    def test_cs_model_i3(self):
        # Test interpolation 3
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.sqrt(x)
        cs_model = CSModel(1)
        cs_model.fit(x, y)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = np.sqrt(new_x)
        test_y = cs_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], 2)

    
    def test_cs_model_i4(self):
        # Test interpolation 4
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.sin(x)
        cs_model = CSModel(1)
        cs_model.fit(x, y)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = np.sin(new_x)
        test_y = cs_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], 2)

    
    def test_cs_model_r1(self):
        # Test extrapolation 1
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = x
        cs_model = CSModel(1)
        cs_model.fit(x, y)

        new_x = np.linspace(-2, 12, 40).reshape(-1, 1)
        new_y = new_x
        test_y = cs_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], 2)


    def test_cs_model_r2(self):
        # Test extrapolation 2
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.square(x)
        cs_model = CSModel(1)
        cs_model.fit(x, y)

        new_x = np.linspace(-2, 12, 40).reshape(-1, 1)
        new_y = np.square(new_x)
        test_y = cs_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], -2)


    def test_cs_model_r3(self):
        # Test extrapolation 3
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.sin(x)
        cs_model = CSModel(1)
        cs_model.fit(x, y)

        new_x = np.linspace(-2, 12, 40).reshape(-1, 1)
        new_y = np.sin(new_x)
        test_y = cs_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], -2)


    def test_cs_model_e1(self):
        # Test error catching 1
        error_1 = 'Input data should have d = 1 dimensions.'
        with self.assertRaises(Exception) as context:
            cs_model = CSModel(0)
        self.assertTrue(error_1 in str(context.exception))

        with self.assertRaises(Exception) as context:
            cs_model = CSModel(2)
        self.assertTrue(error_1 in str(context.exception))

        with self.assertRaises(Exception) as context:
            cs_model = CSModel(-1)
        self.assertTrue(error_1 in str(context.exception))

        cs_model = CSModel(1)
        

    def test_cs_model_e2(self):
        # Test error catching 2
        error_2 = 'Input data should be a numpy array of shape (-1, 1).'
        with self.assertRaises(Exception) as context:
            cs_model = CSModel(1)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            xy = np.concatenate((x, y), axis=1)
            z = np.sum(xy, axis=1)
            cs_model.fit(xy, z)
        self.assertTrue(error_2 in str(context.exception))
        

    def test_cs_model_e3(self):
        # Test error catching 3
        error_3 = 'Output data should be a numpy array of shape (-1, 1).'
        with self.assertRaises(Exception) as context:
            cs_model = CSModel(1)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            xy = np.concatenate((x, y), axis=1)
            z = xy.copy()
            cs_model.fit(x, z)
        self.assertTrue(error_3 in str(context.exception))


    def test_cs_model_e4(self):
        # Test error catching 4
        error_4 = 'Input data should have a length of >= 2.'
        with self.assertRaises(Exception) as context:
            cs_model = CSModel(1)
            x = np.array([[1]])
            y = np.array([[1]])
            cs_model.fit(x, y)
        self.assertTrue(error_4 in str(context.exception)) 


    def test_cs_model_e5(self):
        # Test error catching 5
        error_5 = 'Input data should be a numpy array of shape (-1, 1)'
        with self.assertRaises(Exception) as context:
            cs_model = CSModel(1)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            cs_model.fit(x, y)
            cs_model.predict([[1, 1], [2, 2]])
        self.assertTrue(error_5 in str(context.exception)) 





    def test_p_model_i1(self):
        # Test interpolation 1
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = x
        p_model = PModel(1)
        p_model.fit(x, y)

        new_x = np.linspace(0, 10, 20).reshape(-1, 1)
        new_y = new_x
        test_y = p_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], 2)
        

    def test_p_model_i2(self):
        # Test interpolation 2
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.square(x)
        p_model = PModel(1)
        p_model.fit(x, y)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = np.square(new_x)
        test_y = p_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], 2)
        

    def test_p_model_i3(self):
        # Test interpolation 3
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.sqrt(x)
        p_model = PModel(1)
        p_model.fit(x, y)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = np.sqrt(new_x)
        test_y = p_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], 0)

    
    def test_p_model_i4(self):
        # Test interpolation 4
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.sin(x)
        p_model = PModel(1)
        p_model.fit(x, y)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = np.sin(new_x)
        test_y = p_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], -1)

    
    def test_p_model_r1(self):
        # Test extrapolation 1
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = x
        p_model = PModel(1)
        p_model.fit(x, y)

        new_x = np.linspace(-2, 12, 40).reshape(-1, 1)
        new_y = new_x
        test_y = p_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], 2)


    def test_p_model_r2(self):
        # Test extrapolation 2
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.square(x)
        p_model = PModel(1)
        p_model.fit(x, y)

        new_x = np.linspace(-2, 12, 40).reshape(-1, 1)
        new_y = np.square(new_x)
        test_y = p_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], -2)

    
    def test_p_model_r3(self):
        # Test extrapolation 3
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.sin(x)
        p_model = PModel(1)
        p_model.fit(x, y)

        new_x = np.linspace(-2, 12, 40).reshape(-1, 1)
        new_y = np.sin(new_x)
        test_y = p_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], -2)


    def test_p_model_e1(self):
        # Test error catching 1
        error_1 = 'Input data should have d = 1 dimensions.'
        with self.assertRaises(Exception) as context:
            p_model = PModel(0)
        self.assertTrue(error_1 in str(context.exception))

        with self.assertRaises(Exception) as context:
            p_model = PModel(2)
        self.assertTrue(error_1 in str(context.exception))

        with self.assertRaises(Exception) as context:
            p_model = PModel(-1)
        self.assertTrue(error_1 in str(context.exception))

        p_model = PModel(1)
        

    def test_p_model_e2(self):
        # Test error catching 2
        error_2 = 'Input data should be a numpy array of shape (-1, 1).'
        with self.assertRaises(Exception) as context:
            p_model = PModel(1)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            xy = np.concatenate((x, y), axis=1)
            z = np.sum(xy, axis=1)
            p_model.fit(xy, z)
        self.assertTrue(error_2 in str(context.exception))
        

    def test_p_model_e3(self):
        # Test error catching 3
        error_3 = 'Output data should be a numpy array of shape (-1, 1).'
        with self.assertRaises(Exception) as context:
            p_model = PModel(1)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            xy = np.concatenate((x, y), axis=1)
            z = xy.copy()
            p_model.fit(x, z)
        self.assertTrue(error_3 in str(context.exception))


    def test_p_model_e4(self):
        # Test error catching 4
        error_4 = 'Input data should have a length of >= 3.'
        with self.assertRaises(Exception) as context:
            p_model = PModel(1)
            x = np.array([[1]])
            y = np.array([[1]])
            p_model.fit(x, y)
        self.assertTrue(error_4 in str(context.exception)) 


    def test_p_model_e5(self):
        # Test error catching 5
        error_5 = 'Input data should be a numpy array of shape (-1, 1)'
        with self.assertRaises(Exception) as context:
            p_model = PModel(1)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            p_model.fit(x, y)
            p_model.predict([[1, 1], [2, 2]])
        self.assertTrue(error_5 in str(context.exception)) 




    def test_gp_model_i1(self):
        # Test interpolation 1
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = x
        gp_model = GPModel(1)
        gp_model.fit(x, y)

        new_x = np.linspace(0, 10, 20).reshape(-1, 1)
        new_y = new_x
        test_y = gp_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], 2)
        

    def test_gp_model_i2(self):
        # Test interpolation 2
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.square(x)
        gp_model = GPModel(1)
        gp_model.fit(x, y)

        new_x = np.linspace(1, 9, 40).reshape(-1, 1)
        new_y = np.square(new_x)
        test_y = gp_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], 2)

    
    def test_gp_model_r1(self):
        # Test extrapolation 1
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = x
        gp_model = GPModel(1)
        gp_model.fit(x, y)

        new_x = np.linspace(-2, 12, 40).reshape(-1, 1)
        new_y = new_x
        test_y = gp_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], 2)


    def test_gp_model_r2(self):
        # Test extrapolation 2
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.square(x)
        gp_model = GPModel(1)
        gp_model.fit(x, y)

        new_x = np.linspace(-2, 12, 40).reshape(-1, 1)
        new_y = np.square(new_x)
        test_y = gp_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], -2)

    
    def test_gp_model_r3(self):
        # Test extrapolation 3
        x = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.sin(x)
        gp_model = GPModel(1)
        gp_model.fit(x, y)

        new_x = np.linspace(-2, 12, 40).reshape(-1, 1)
        new_y = np.sin(new_x)
        test_y = gp_model.predict(new_x)
        for i, y_val in enumerate(new_y):
            self.assertAlmostEqual(y_val[0], test_y[i, 0], -2)


    def test_gp_model_e1(self):
        # Test error catching 1
        error_1 = 'Input data should have d >= 1 dimensions.'
        with self.assertRaises(Exception) as context:
            gp_model = GPModel(0)
        self.assertTrue(error_1 in str(context.exception))

        with self.assertRaises(Exception) as context:
            gp_model = GPModel(-1)
        self.assertTrue(error_1 in str(context.exception))

        with self.assertRaises(Exception) as context:
            gp_model = GPModel(-10)
        self.assertTrue(error_1 in str(context.exception))

        gp_model = GPModel(1)
        gp_model = GPModel(5)
        gp_model = GPModel(10)

    
    def test_gp_model_e2(self):
        # Test error catching 2
        error_2 = 'Input data should be a numpy array of shape (-1, 1).'
        with self.assertRaises(Exception) as context:
            gp_model = GPModel(1)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            xy = np.concatenate((x, y), axis=1)
            z = np.sum(xy, axis=1)
            gp_model.fit(xy, z)
        self.assertTrue(error_2 in str(context.exception))

        error_2 = 'Input data should be a numpy array of shape (-1, 2).'
        with self.assertRaises(Exception) as context:
            gp_model = GPModel(2)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            gp_model.fit(x, y)
        self.assertTrue(error_2 in str(context.exception))
        

    def test_gp_model_e3(self):
        # Test error catching 3
        error_3 = 'Output data should be a numpy array of shape (-1, 1).'
        with self.assertRaises(Exception) as context:
            gp_model = GPModel(2)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            xy = np.concatenate((x, y), axis=1)
            z = xy.copy()
            gp_model.fit(xy, z)
        self.assertTrue(error_3 in str(context.exception))


    def test_gp_model_e4(self):
        # Test error catching 4
        error_4 = 'Input data should have a length of >= 3.'
        with self.assertRaises(Exception) as context:
            gp_model = GPModel(2)
            x = np.array([[1, 1]])
            y = np.array([[1]])
            gp_model.fit(x, y)
        self.assertTrue(error_4 in str(context.exception)) 


    def test_gp_model_e5(self):
        # Test error catching 5
        error_5 = 'Input data should be a numpy array of shape (-1, 1)'
        with self.assertRaises(Exception) as context:
            gp_model = GPModel(1)
            x = np.linspace(0, 10, 20).reshape(-1, 1)
            y = x.copy()
            gp_model.fit(x, y)
            gp_model.predict([[1, 1], [2, 2]])
        self.assertTrue(error_5 in str(context.exception)) 





if __name__ == "__main__":
    unittest.main()