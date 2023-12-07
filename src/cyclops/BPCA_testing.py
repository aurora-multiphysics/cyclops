"""
Experimental BPCA class for cyclops.

Reduces dimensionality of input while maintaining boundary conditions.

# Author: Ciara Byers <ciara.byers@ukaea.uk>
(c) Copyright UKAEA 2023.
"""
import numpy as np
from scipy.linalg import svd
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
    #print(X_noi)
    A_noi = np.delete(A, I, 0)
    #print(A_noi.shape, "A_noi")    
    #print(X_noi.shape, "X_noi")
    #print((B.T).shape, "B")    
    inner = np.subtract(X_noi, np.matmul(A_noi,B.T))
    absou = np.absolute(inner)
    row_const = np.matmul(absou,absou.T)

    G = np.c_[B, -B]
    h = np.c_[X[I], -X[I]]
    #print("G", G.shape)
    #print("h", h.shape)


    #S = Qy-P_i.T*v
    #now have LDP problem minimise s**2 + p_2.T*v s.t. GRQ^1s > h -GRQ^-1P_1'*v
    #Convert the LDP problem to an NNLS one 
    E, n, f = LDP_to_NNLS(G, h)
    #r, rnorm = nnls(E, f)


    U = B
    W = r
    V = X[I]


def LDP_to_NNLS(G: np.array, h: np.array):
    """Transforms a bounded LDP problem to a NNLS one"""

    E = np.hstack((G, h))
    print(E)
    E = E.T
    n = E.shape[0]
    f = np.zeros((n,1))

    return(E, n, f)


def check_shape(n: np.array):
    """Function to ensure any 1D arrays are in correct form for later use"""

    if len(n.shape) == 1:
        #Potential issue -> what if n is expected to be a row vector not column?
        n = np.expand_dims(n, axis=1)
    return n


# Data needs to be organised as having samples in rows, variables in columns for later calculations
data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).T
std_input = standardise(data)
bounds_vec = np.array([[0, 0, 0, 0, 0], [8, 8, 8, 8, 8]]).T
#Performing SVD on standardised, mean-centered data
U, s, Vt = np.linalg.svd(std_input, full_matrices=False)

#Principal Axes (PAs) and Component Scores (CSs) calculated from SVD output
#for ||X-AB'||**2
CSs = np.dot(U, s) # A (Component Scores)
CSs = check_shape(CSs)
print("CSs", CSs.shape)
CLM = np.dot(Vt.T, s)/(len(std_input)-1)**(1/2) # B (Component loading matrix)
CLM = check_shape(CLM)
print("CLM", CLM.shape)
PAs = Vt.T # Principal Axes
PAs = check_shape(PAs)
print("PAs", PAs.shape)

update_a(CSs, CLM, bounds_vec, 1)