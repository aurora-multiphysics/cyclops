"""
Experimental BPCA class for cyclops.

Reduces dimensionality of input while maintaining boundary conditions.

# Author: Ciara Byers <ciara.byers@ukaea.uk>
(c) Copyright UKAEA 2023.
"""
import numpy as np
from scipy.linalg import svd, diagsvd
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

    #Calculating the constant for row being updated, I
    #const is given by mod(X_noi - A_noi*B.T)^2
    X_noi = np.delete(X, I, 0)
    A_noi = np.delete(A, I, 0)
    # X_noi = check_shape(X_noi)
    # X_noi[I] = np.zeros(len(X[0]))
    # A_noi = check_shape(A_noi)
    # A_noi[I] = np.zeros(len(A[0]))
    # print(A_noi.shape, "A_noi")    
    # print(X_noi.shape, "X_noi")
    #B_noi = np.delete(B.T, I, 0)
    #print((B_noi).shape, "B_noi")
    #Pad X_noi to allow matrix subtraction

    bshape = B.shape[1]
    xshape = X_noi.shape[1]
    X_noi = np.pad(X_noi, pad_width=((0,0), (0, bshape - xshape)))
    ashape = A_noi.shape[0]    
    A_noi = np.pad(A_noi, pad_width=((0,0), (0, bshape - ashape)))
    #print(X_noi.shape)
    #print(np.matmul(A_noi,B))
    inner = np.subtract(X_noi, np.matmul(A_noi,B))
    #print('inner', inner)
    row_const = np.linalg.det(inner) # Constant to be added 
    print(row_const)

    U = B
    W = A[I]
    V = X[I]
    G = np.c_[B, -B]
    h = np.c_[X[I], -X[I]]
    #print("G", G.shape, G)
    #print("h", h.shape, h)


    #S = Qy-P_i.T*v
    #now have LDP problem minimise s**2 + p_2.T*v s.t. GRQ^1s > h -GRQ^-1P_1'*v
    #Convert the LDP problem to an NNLS one 
    E, n, f = LSI_to_LDP(G, h, U, W, V)
    r, rnorm = nnls(E, f)
    print("r", r)
    print("rnorm", rnorm)


def LSI_to_LDP(G: np.array, h: np.array, U: np.array, W: np.array, V: np.array):
    """Transforms a bounded LSI problem to a bounded LDP one with use of SVD on
    U, rewriting the problem as an LDP + constant with new bounds. See 
    'Principal component analysis with boundary constraints' by Paolo Giordani
     and Henk A. L. Kiers"""

    # Still needs to calculate the constant for the row mod(P_2' * v)**2 Should return
    # S for minimisation, the constant and the new form of the boundary conditions.

    #Decompose matrix U
    P, Q, R_t = np.linalg.svd(U, full_matrices=False)
    Q = check_shape(Q)
    print("P", P.shape)
    print("Q", Q.shape)    
    print("R_t", R_t.shape)
    hshape = h.shape[1]
    gshape = G.shape[1]
    h = np.pad(h, pad_width=((0,0), (0, gshape - hshape) ))
    #print(h)
    E = np.hstack((G.T, h.T))
    print("E", E.shape)
    E = E.T
    n = E.shape[0]
    f = np.zeros((n))

    return(E, n, f)


def check_shape(n: np.array):
    """Function to ensure any 1D arrays are in correct form for later use"""

    if len(n.shape) == 1:
        #Potential issue -> what if n is expected to be a row vector not column?
        n = np.expand_dims(n, axis=1)
    return n


# Data needs to be organised as having samples in rows, variables in columns for later calculations
data = np.array([[1, 15, 3, 4, 5], [6, 7, 8, 9, 10]]).T 
data2 = np.array([[5, 4, 3, 3, 1], [10, 20 , 8, 7, 6]]).T
data = np.hstack((data,data2))
n = 1
std_input = standardise(data)
print("OG", std_input.shape)
bounds_vec = np.array([[0, 0, 0, 0, 0], [8, 8, 8, 8, 8]]).T
#Performing SVD on standardised, mean-centered data
U, s, Vt = svd(std_input, full_matrices=True)
s = diagsvd(s, U.shape[0], Vt.shape[-1])
print("s", s.shape, s)
# print("U",U.shape, U)
# print("s",s.shape, s)
#print("Vt",Vt.shape, Vt)
#Principal Axes (PAs) and Component Scores (CSs) calculated from SVD output
#for ||X-AB'||**2
CSs = np.matmul(U, s) # A (Component Scores)
CSs = check_shape(CSs)
print("CSs", CSs.shape)
#explained_variance = (s**2)/ (len(data) - 1)
#print(explained_variance.shape)
sshape = s.shape[0]
Vtshape = Vt.shape[1]
Vt = np.pad(Vt, pad_width=((0, sshape - Vtshape ), (0,0)))
print("Vt",Vt.shape, Vt)
CLM = np.dot(Vt.T, np.dot(s, np.sqrt(len(data)-n))) # B (Component loading matrix)
CLM = check_shape(CLM)
print("CLM", CLM.shape)
PAs = Vt.T # Principal Axes
PAs = check_shape(PAs)
#print("PAs", PAs.shape)

update_a(CSs, CLM, bounds_vec, n)