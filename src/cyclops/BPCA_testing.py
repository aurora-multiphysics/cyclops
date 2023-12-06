"""
Experimental BPCA class for cyclops.

Reduces dimensionality of input while maintaining boundary conditions.

# Author: Ciara Byers <ciara.byers@ukaea.uk>
(c) Copyright UKAEA 2023.
"""
import numpy as np
from scipy.optimize import least_squares, Bounds, nnls

def standardise(input_dims: np.array) -> np.array:
    """ Takes a dataset and standardises the variance of the different 
    dimensions. This is done to prevent bias in a particular direction due to 
    the values of the variables in it skewing higher than other directions."""

    if not isinstance(input_dims, np.ndarray):
        input_dims = np.array(input_dims)

    std_dims = np.zeros(input_dims.shape)
    i=0
    for dim_array in input_dims:
        if not isinstance(dim_array, np.ndarray):
            dim_array = np.array(dim_array)
    
        #creating mean-centered, standardised data
        mean_array = dim_array.mean()
        stdev = np.std(dim_array) * np.ones(dim_array.shape)
        standised_dim = dim_array.astype(float) - mean_array.astype(float)
        standised_dim = np.divide(standised_dim.astype(float), stdev.astype(float))
        std_dims[i] = standised_dim
        i += 1
    
    return(np.array(std_dims))


def calc_cov(input_dims: np.array) -> np.array:
    """Function applies numpy cov function to get covariance matrix for input.
    Outputs this matrix as a numpy array"""

    if not isinstance(input_dims, np.ndarray):
        input_dims = np.array(input_dims)
    cov_matrix = np.cov(input_dims)

    return(cov_matrix)


def update_a(A: np.array, B: np.array, X: np.array, I: float):
    """Function updates row I of matrix A according to BPCA algorithm"""

    #print(A)
    #Calculating the constant in row I
    X_noi = np.delete(X, I, 0)
    A_noi = np.delete(A, I, 0)
    inner = np.subtract(X_noi, np.matmul(A_noi,B.T))
    absou = np.absolute(inner)
    row_const = np.matmul(absou,absou.T)
    print(row_const)

    a_i, res = nnls(B, X[I])
    print(a_i)
    print(res)

    U = B
    W = a_i
    V = X[I]
    G = np.c_[B, -B]
    h = np.c_[X[I], -X[I]]

    #S = Qy-P_i.T*v
    #minimise s**2 + p_2.T v
    LDP_to_NNLS(G, h, U, False)


def LDP_to_NNLS(G: np.array, h: np.array, U: np.array, phi: bool):
    """Transforms a bounded LDP problem to a NNLS one"""

    E = np.hstack((G, h))
    E = E.T

    P, Q, R = np.linalg.svd(E, full_matrices=False)


a = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
std_input = standardise(a)

#Performing SVD on standardised, mean-centered data
U, s, Vt = np.linalg.svd(std_input, full_matrices=False)

#Principal components (PCs) and Component Scores (CSs) are in SVD output
PCs = Vt.T #B
CSs = np.dot(a, PCs) #A

reconst = np.dot(CSs,PCs.T) #X
#print(reconst)
update_a(CSs, PCs, std_input, 1)