"""
Tests for the Cyclops BPCA functionality.

(c) Copyright UKAEA 2024.
"""
import numpy as np
import unittest
import pytest
import sys
sys.path.insert(0, '/home/cbyers/projects/working_branches/cyclops/src')
import cyclops.BPCA as target
from sklearn import decomposition as decomp

# Test data
data = np.array([2, 6, 2, 3, 3]).reshape(1, 5)
data2 = np.array([[2, 3, 5, 3, 3],
                  [2, 3, 2, 3, 3]]).reshape(2, 5)
data_temp = ((data[0] + data2[1]) * 80).T
data = data.T
data2 = data2.T
data = np.hstack((data, data2, data_temp))

# Boundary arrays
low_col = np.ones((5, 1)) * -2
low_bound = np.hstack((low_col, low_col, low_col, low_col))
high_col = low_col * -5
high_bound = np.hstack((high_col, high_col, high_col, high_col + 900))
bounds_vec = np.array((low_bound, high_bound))


class TestMathsFunctions:
    """Class to test the functions in 'MathsFunctions' """

    PM = target.MathsFunctions.pad_matrices
    PS = target.MathsFunctions.pad_to_subtract
    IP = target.MathsFunctions.inverse_or_pseudo
    CS = target.MathsFunctions.check_shape
    Std = target.MathsFunctions.standardise

    def pad_matrices(self):
        M1 = data.reshape(1, 5)
        M2 = data2.reshape(2, 5)
        M1_pad, M2_pad = self.PM(M1, M2)
        assert M1_pad.shape(1) == M2_pad.shape(0)

    def pad_to_subtract(self):
        M1 = data.reshape(1, 5)
        M2 = data2.reshape(5, 2) 
        M1_pad, M2_pad = self.PS(M1, M2)       
        assert M1_pad.shape == M2_pad.shape

    def check_shape(self):
        new_shape = self.CS(data_temp).shape
        assert new_shape == (5, 1)

    def standardise(self):
        M1 = data2.reshape(2, 5)
        M1 = np.std(M1)
        M2 = self.Std(M1)(0)
        assert M1 == M2


class TestBPCAFunctions:
    """Class to test the functions in 'BPCAFunctions' """

    CB = target.solve_BPCA.check_xi_bounds
    Solve = target.solve_BPCA.solve_for_A_B
    update = target.solve_BPCA.update_row
    LSI_in = target.solve_BPCA.LSI_to_LDP
    LDP_in = target.solve_BPCA.LDP_to_NNLS_sol

    M1 = [[-1.68425124,  0.95763605, -0.03744297],
          [3.18222495,  0.23825748,  0.01730576],
          [-0.96866249, -0.50463255,  0.2446228],
          [-0.52931122, -0.69126098, -0.2244856]] 

    M2 = [[1.04803401],
          [0.30717975],
          [1.43885348],
          [1.43885348]]

    X = [[[-3.20246998, -12.59823955,  -4.42635206,  -4.44559707],
          [-3.20246998, -12.59823955,  -4.42635206,  -4.44559707],
          [-3.20246998, -12.59823955,  -4.42635206,  -4.44559707],
          [-3.20246998, -12.59823955, -4.42635206,  -4.44559707]],
          [[10.21740421,  47.98345884,  12.50925583,   4.42635206],
           [10.21740421,  47.98345884,  12.50925583,   4.42635206],
           [10.21740421,  47.98345884,  12.50925583,  4.42635206],
           [10.21740421,  47.98345884,  12.50925583,   4.42635206]]]
    R = 0
    col_row = 0
    sol_vec_x = [[0.]]
    resid = [-7.55515230e-14,  0.0000000e+00, -1.93726732e-13, -1.0000000e+00]

    pca = decomp.PCA(n_components=1)
    pca.fit(data)
    CSs = pca.transform(data)
    PAs = pca.components_.T
    CLM = np.dot(PAs, np.sqrt(pca.explained_variance_))

    def check_bounds(self):
        CSs = self.CSs*-100
        CLM = self.CLM
        with pytest.raises(Exception) as exc_info:
            self.CB(bounds_vec, 1, CSs, CLM)
            assert str(exc_info.value) == "Boundary conditions violated \
              during'check_xi_bounds'!"


if __name__ == "__main__":
    unittest.main()
