"""
Tests for the Cyclops BPCA functionality.

(c) Copyright UKAEA 2024.
"""
import numpy as np
import unittest
import pytest
target = __import__("BPCA_testing.py")

data = np.array([2, 6, 2, 3, 3])
data2 = np.array([[2, 3, 3, 2.6, 2.7], [2, 3, 6, 4, 2]])
data_temp = ((data[0] + data2[1]) * 80).T


class TestMathsFunctions:
    """Class to test the functions in 'MathsFunctions' """

    PM = target.pad_matrices
    PS = target.pad_to_subtract
    IP = target.inverse_or_pseudo
    CS = target.check_shape
    Std = target.standardise

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


if __name__ == "__main__":
    unittest.main()
