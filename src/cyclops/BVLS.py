"""
Experimental BVLS function for cyclops.

Finds the solution to a 'least squares' problem while maintaining boundary conditions.

# Author: Ciara Byers <ciara.byers@ukaea.uk>
(c) Copyright UKAEA 2023.
"""
import numpy as np
import scipy
from scipy.optimize import least_squares


def create_E(G: np.array, h: np.array) -> np.array:
    """Takes the boundary conditions for a problem and uses these to define a
    matrix 'E' 
    Args:
        G: (np.ndarray[float]): n by m array  
        h: (np.ndarray[float]): 1 by m array
    Returns:
        E: (np.ndarray[float]): (n+1) by m array
        """
    Gt = G.T
    ht = h.T
    E = np.c_[Gt, ht]
    E = E.T

    return E


G = np.array(([0,0,0], [1,1,1]))
h = np.c_[3,3,3]

#print(G.shape, h.shape)

E_ = create_E(G, h)
#print(E_.shape[0])
f = np.zeros(E_.shape[0])

#Use scipy to find a solution to the nnls problem r = Eu - f
x_ , rnorm_ = scipy.optimize.nnls(E_,f)
print(x_, rnorm_)
