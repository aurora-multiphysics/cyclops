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


def pad_matrices(matrix1, matrix2):
    """Function to allow matrix multiplication of otherwise incompatible
    matrices. The matrices MUST be entered in the order they would appear
    in the multiplication, otherwise this will pad the wrong sections!"""

    rows1, cols1 = matrix1.shape
    rows2, cols2 = matrix2.shape

    # Check if compatible, if so, return them
    if cols1 == rows2:
        return matrix1, matrix2

    # Pad matrix1 with rows of zeros along bottom if needed
    if cols1 < rows2:
        zeros_to_add = rows2 - cols1
        zeros = np.zeros((rows1, zeros_to_add))
        padded_matrix1 = np.hstack((matrix1, zeros))
        return padded_matrix1, matrix2

    # Pad matrix2 with columns of zeros on the right side if needed
    if cols1 > rows2:
        zeros_to_add = cols1 - rows2
        zeros = np.zeros((zeros_to_add, cols2))
        padded_matrix2 = np.vstack((matrix2, zeros))
        return matrix1, padded_matrix2
    

def pad_to_subtract(matrix1, matrix2):
    """Function to allow subtraction between matrices of different sizes. Pads
    rows and columns of the matrices to ensure they have the same shape then
    returns these new padded matrices."""

    rows1, cols1 = matrix1.shape
    rows2, cols2 = matrix2.shape

    # Check shapes are not already the same, return if they are
    if rows1 == rows2 and cols1 == cols2:
        return matrix1, matrix2

    # Pad matrix1 with rows to match matrix2 if it has more
    if rows1 < rows2:
        zeros_to_add = rows2 - rows1
        padded_matrix1 = np.pad(matrix1, pad_width=((0, zeros_to_add), (0, 0)))
        padded_matrix2 = matrix2
        print(padded_matrix1)
    # Pad matrix2 with rows to match matrix1 if it has more
    elif rows1 > rows2:
        zeros_to_add = rows1 - rows2
        padded_matrix1 = matrix1
        padded_matrix2 = np.pad(matrix2, pad_width=((0, zeros_to_add), (0, 0)))

    # Pad matrix1 with columns if matrix2 has more
    if cols1 < cols2:
        zeros_to_add = cols2 - cols1
        padded_matrix1 = np.pad(padded_matrix1, pad_width=((0, 0), (0, zeros_to_add)))
    # Pad matrix2 with columns if matrix1 has more
    elif cols1 > cols2:
        zeros_to_add = cols1 - cols2
        padded_matrix2 = np.pad(padded_matrix2, pad_width=((0, 0), (0, zeros_to_add)))
    
    return padded_matrix1, padded_matrix2


def update_a(A: np.array, B: np.array, X: np.array, I: float):
    """Function updates row I of matrix A according to BPCA algorithm"""

    #Calculating the constant for row being updated, I
    #const is given by mod(X_noi - A_noi*B.T)^2
    X_noi = np.delete(X, I, 0)
    A_noi = np.delete(A, I, 0)
    print(A_noi.shape, "A_noi")    
    print(X_noi.shape, "X_noi")

    bshape = B.shape[0]
    print(B.shape)
    #Pad A_noi/B to allow matrix multiplication
    A_noi, B = pad_matrices(A_noi, B)
    AB = np.matmul(A_noi,B)

    #Pad X_noi/AB to allow matrix subtraction
    X_noi, AB = pad_to_subtract(X_noi, AB)
    inner = np.subtract(X_noi, AB)
    In0 = inner.shape[0]
    In1 = inner.shape[1]
    if In0>In1:
        inner = np.pad(inner, pad_width=((0,0), (0, In0 - In1))) 
    elif In1>In0:
        inner = np.pad(inner, pad_width=((0, In1 - In0), (0,0)))

    null_cols = []
    for col in range(inner.shape[1]):
        if sum(inner[:,col]) == 0:
            null_cols.append(col)
    
    inner = np.delete(inner, null_cols, axis=1)
    print(inner.shape)

    row_const = np.linalg.norm(inner) # Constant to be added 
    print(row_const)
    #Remeber to come back and change U = B to avoid accidental linking
    U = B
    W = A[I]
    V = check_shape(X[I])
    G = np.c_[B, -B]
    h = np.c_[X[I], -X[I]]

    #S = Qy-P_i.T*v
    #now have LDP problem minimise s**2 + p_2.T*v s.t. GRQ^1s > h -GRQ^-1P_1'*v
    #Convert the LDP problem to an NNLS one 
    E, n, min_S_sq, LDP_const, f = LSI_to_LDP(G, h, U, W, V)
    #E, f = pad_to_subtract(E, f)
    f_len = f.shape[0]
    E_len = E.shape[0]
    E_pad = np.pad(E, pad_width=((0, f_len - E_len), (0, 0)))
    r, rnorm = nnls(E_pad, f)
    print("r", r)
    print("rnorm", rnorm)


def LSI_to_LDP(G: np.array, h: np.array, U: np.array, W: np.array, V: np.array):
    """Transforms a bounded LSI problem to a bounded LDP one with use of SVD on
    U, rewriting the problem as an LDP + constant with new bounds. See 
    'Principal component analysis with boundary constraints' by Paolo Giordani
     and Henk A. L. Kiers"""

    # Still needs to calculate the constant for the row mod(P_2' * v)**2 Should return
    # S for minimisation, the constant and the new form of the boundary conditions.
    # min w ||Uw -v||**2 s.t. Gw>=h

    #Decompose matrix U
    P, Q, R_t = np.linalg.svd(U, full_matrices=False) 
    P_shape = P.shape[1]
    R_t_shape = R_t.shape[0]  
    Q_shape = Q.shape
    Q0 = Q_shape[0]

    if len(Q_shape) == 1:
        #Need to check if Q can be of the form (,n) not (n,)
        Q = check_shape(Q)
        Q1 = Q.shape[1]
        Q_padded1 = np.pad(Q, pad_width=((0,0), (0, R_t_shape-Q1) )) 
        print("Q", Q.shape) 
    else:
        Q_padded1 = np.pad(Q, pad_width=((0,P_shape - Q_shape[0]), (0, R_t_shape - Q_shape[1]) )) 
        print("Q", Q.shape) 

    hshape = h.shape[1]
    gshape = G.shape[1]
    y = np.matmul(R_t, W)
    y = check_shape(y)
    yshape = y.shape
    h = np.pad(h, pad_width=((0,0), (0, gshape - hshape) ))
    #Let s = Qy-P_1.T v  s.t. our problem is now min s ||s||**2 ||P_2.T v||**2
    # s.t. GRQ^-1 s >=h - GRQ^-1 P_1.T v
    E = np.hstack((G.T, h.T))
    E = E.T
    P_1 = P[:R_t_shape+1,:] 
    P_2 = P[R_t_shape+1:,:]
    P1_shape0 = P_1.shape[0]
    P2_shape0 = P_2.shape[0]    

    V0 = V.shape[0]
    V1 = V.shape[1]
    if P1_shape0 > V0:
       V_padded1 = np.pad(V, pad_width=((0,P1_shape0 - V0), (0, 0) ))
       P_1_padded = P_1
    elif P1_shape0 < V0:
       V_padded1 = V
       P_1_padded = np.pad(P_1, pad_width=((0, V0 - P1_shape0), (0, 0)))

    Qy = np.matmul(Q_padded1, y)
    P1TV = np.matmul(P_1.T, V_padded1)
    S_LDP = np.subtract(Qy, P1TV)
    min_S_sq = np.square(np.linalg.norm(S_LDP))

    if P2_shape0 > V0:
       V_padded2 = np.pad(V, pad_width=((0,P2_shape0 - V0), (0, 0) ))
       P_2_padded = P_2
    elif P2_shape0 < V0:
       V_padded2 = V
       P_2_padded = np.pad(P_2, pad_width=( (0, V0 - P2_shape0) , (0, 0)))
  
    P2TV = np.matmul(P_2_padded.T, V_padded2)
    LDP_const = np.square(np.linalg.norm(P2TV))
    n = E.shape[0]
    f = np.zeros((n))
    f = np.append(f, [1])
    f = f.T
    print('f', f)

    #Calulate new boundaries
    
    def make_square(matrix):
        rows, cols = matrix.shape
        max_dim = max(rows, cols)

        # If already square, return the original matrix
        if rows == cols:
            return matrix

        # Pad the matrix with zeros to make it square
        if rows < cols:
            # Add rows of zeros to make the matrix square
            zeros_to_add = max_dim - rows
            zeros = np.zeros((zeros_to_add, cols))
            squared_matrix = np.vstack((matrix, zeros))
        else:
            # Add columns of zeros to make the matrix square
            zeros_to_add = max_dim - cols
            zeros = np.zeros((rows, zeros_to_add))
            squared_matrix = np.hstack((matrix, zeros))

        return squared_matrix

    #Q may not be square, must test to see, calculate pseudoinverse instead
    # if it is not
    #Q_square = make_square(Q)
    def inverse_or_pseudo(matrix):
        try:
            # Try to compute the inverse
            inv_matrix = np.linalg.inv(matrix)
            return inv_matrix
        except np.linalg.LinAlgError:
            # If an error occurs, compute the pseudo-inverse
            pseudo_inv_matrix = np.linalg.pinv(matrix)
            return pseudo_inv_matrix

    Q_minus1 = inverse_or_pseudo(Q)
    R_t, Q_minus1 = pad_matrices(R_t, Q_minus1)
    RQ_minus1 = np.matmul(R_t.T, Q_minus1)
    G, RQ_minus1 = pad_matrices(G, RQ_minus1)
    GRQ_minus1 = np.matmul(G, RQ_minus1)

    return(E, n, min_S_sq, LDP_const, f)


def check_shape(n: np.array):
    """Function to ensure any 1D arrays are in correct form for later use"""

    if len(n.shape) == 1:
        #Potential issue -> what if n is expected to be a row vector not column?
        n = np.expand_dims(n, axis=1)
    return n


def LDP_to_NNLS():
    """Function to calculate needed matrices and constants for the transformation
    of a problem between LDP form and that of NNLS."""


# Data needs to be organised as having samples in rows, variables in columns for later calculations
data = np.array([[1, 15, 3, 4, 5, 5.2, 18, 4, 0.2, 1], [6, 7, 8, 9, 10, 3, 4, 5, 5.2, 18]]).T 
data2 = np.array([[5, 4, 3, 3, 1, 12, 50 , 8.55, 7.2, 90], [10, 20 , 8, 7, 6, 50 , 8.55, 7.2, 90, 1]]).T
data = np.hstack((data,data2))
n = 1
std_input = standardise(data)
print("OG", std_input.shape)
bounds_vec = np.array([[0, 0, 0, 0, 0], [8, 8, 8, 8, 8]]).T

#Performing SVD on standardised, mean-centered data
U, s, Vt = svd(std_input, full_matrices=True)
s = diagsvd(s, U.shape[0], Vt.shape[-1])

#Principal Axes (PAs) and Component Scores (CSs) calculated from SVD output
#for ||X-AB'||**2
CSs = np.matmul(U, s) # A (Component Scores)
CSs = check_shape(CSs)

#explained_variance = (s**2)/ (len(data) - 1)
#print(explained_variance.shape)

sshape = s.shape[0]
Vtshape = Vt.shape[1]
if Vtshape < sshape:
    Vt = np.pad(Vt, pad_width=((0, sshape - Vtshape ), (0,0)))
elif sshape < Vtshape:
    s = np.pad(s, pad_width=((0, Vtshape-sshape), (0,0)))    
CLM = np.dot(Vt.T, np.dot(s, np.sqrt(len(data)-n))) # B (Component loading matrix)
CLM = check_shape(CLM)
PAs = Vt.T # Principal Axes
PAs = check_shape(PAs)

update_a(CSs, CLM, bounds_vec, n)