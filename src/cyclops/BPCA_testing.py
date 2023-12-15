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
        #print(padded_matrix1)
    # Pad matrix2 with rows to match matrix1 if it has more
    elif rows1 > rows2:
        zeros_to_add = rows1 - rows2
        padded_matrix1 = matrix1
        padded_matrix2 = np.pad(matrix2, pad_width=((0, zeros_to_add), (0, 0)))
    else:
        padded_matrix1 = matrix1
        padded_matrix2 = matrix2

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

    print("Run", I, " X shape ", X.shape)
    #Start from LSI problem
    def calc_top_const(A: np.array, B: np.array, X: np.array, I: float):
        """Function calculates the value of the constant in the initial
        minimisation problem for a row I"""
        #Calculating the constant for row being updated, row I
        #const is given by mod(X_noi - A_noi*B.T)^2
        X_noi = np.delete(X, I, 0)
        A_noi = np.delete(A, I, 0)
        #Pad A_noi and/or B to allow matrix multiplication
        A_noi, B_T = pad_matrices(A_noi, B.T)
        AB_T = np.matmul(A_noi,B_T)
        #Pad X_noi/AB to allow matrix subtraction
        X_noi, AB_T = pad_to_subtract(X_noi, AB_T)
        inner = np.subtract(X_noi, AB_T)
        In0 = inner.shape[0]
        In1 = inner.shape[1]
        if In0>In1:
            inner = np.pad(inner, pad_width=((0,0), (0, In0 - In1))) 
        elif In1>In0:
            inner = np.pad(inner, pad_width=((0, In1 - In0), (0,0)))

        inner = drop_null_cols(inner)
        row_const = np.linalg.norm(inner)
        return row_const

    #Remeber to come back and change U = B to avoid accidental linking
    #seeking optimal row A[I] in boundaries now
    U = B
    W = A[I]
    V = check_shape(X[I])
    G = np.c_[B, -B]
    h = np.c_[X[I], -X[I]]

    #As in paper, let S = Qy-P_i.T*v to get to a LDP problem as follows
    #min||s||**2 + ||p_2.T*v||**2 s.t. GRQ^1s > h -GRQ^-1P_1'*v
    E, n, min_S_sq, LDP_const, f, new_bounds = LSI_to_LDP(G, h, U, W, V)
    #E, f = pad_to_subtract(E, f)
    f_len = f.shape[0]
    E_len = E.shape[0]
    E_pad = np.pad(E, pad_width=((0, f_len - E_len), (0, 0)))
    solution_vec, resid = nnls(E_pad, f)
    #E_pad, solu_vec_pad = pad_matrices(E_pad, solution_vec)
    E_sol = np.matmul(E_pad, solution_vec)

    #E_sol = drop_null_rows(E_sol)
    #print("E_sol", E_sol.shape)
    #solution_vec = drop_null_cols(solution_vec)
    #print("solution_vec", solution_vec.shape)
    #print("f", f.shape)
    if E_sol.shape[0] >= f_len:
        zeros = np.zeros(E_sol.shape[0] - f_len)
        f_pad = np.hstack((f, zeros))
    #    print(f_pad.shape)
    r = np.subtract(E_sol, f_pad)
    #print("r", r)
    
    return r, resid


def LSI_to_LDP(G: np.array, h: np.array, U: np.array, W: np.array, V: np.array):
    """Transforms a bounded LSI problem to a bounded LDP one with use of SVD on
    U, rewriting the problem as an LDP + constant with new bounds. See 
    'Principal component analysis with boundary constraints' by Paolo Giordani
     and Henk A. L. Kiers"""

    # Still needs to calculate the constant for the row mod(P_2' * v)**2 Should return
    # S for minimisation, the constant and the new form of the boundary conditions.
    # min w ||Uw -v||**2 s.t. Gw>=h

    #Decompose matrix U
    P, Q, R_t = np.linalg.svd(U, full_matrices=True) 
    Q = diagsvd(Q, P.shape[0], R_t.shape[-1])
    print(P.shape, Q.shape, R_t.shape)
    P_shape = P.shape[1]
    R_t_shape = R_t.shape[0]  
    Q_shape = Q.shape
    Q0 = Q_shape[0]
 
    W = check_shape(W)
    R_t, W = pad_matrices(R_t, W)
    y = np.matmul(R_t, W)
    hshape = h.shape[0]
    gshape = G.shape[0]
    print("h.T,G.T", h.T.shape, G.T.shape)
    h_padded = np.pad(h, pad_width=((0, gshape - hshape), (0,0) ))
    print("h_padded.T", h_padded.T.shape)

    #Let s = Qy-P_1.T v  s.t. our problem is now min s ||s||**2 ||P_2.T v||**2
    # s.t. GR(Q^-1)*s + GR(Q^-1)*(P_1.T)*v>=h
    E = np.vstack((G.T, h_padded.T))
    print("E", E.shape)
    #Note the shapes of P_1 and P_2 are (nxm) and (nx(n-m)) where P from 
    # earlier is (nxn) and R_t is (mxm)
    P_1 = P[:, :R_t_shape] 
    P_2 = P[:, :(P.shape[0] - R_t_shape)] 
#CURRENT EDITING POINT
    P_1_padded_T, V_padded1 = pad_matrices(P_1.T, V)
    Qy = np.matmul(Q, y)
    P1TV = np.matmul(P_1_padded_T, V_padded1)
    S_LDP = np.subtract(Qy, P1TV)
    min_S_sq = np.square(np.linalg.norm(S_LDP))
    print(min_S_sq)

    P_2_padded_T, V_padded2 = pad_matrices(P_2.T, V)
    P2TV = np.matmul(P_2_padded_T, V_padded2)
    LDP_const = np.square(np.linalg.norm(P2TV))
    n = E.shape[0]
    f = np.zeros((n-1))
    f = np.append(f, [1])
    f = f.T

    #Calulate new, transformed boundaries

    #Q may not be square, check and calculate pseudoinverse instead of
    # inverse if it is not
    Q_minus1 = inverse_or_pseudo(Q)
    Q_minus1_s = Q_minus1*min_S_sq
    #print(Q_minus1_s)
    #print("R_t.shape, Q_minus1_s.shape", R_t.shape, Q_minus1_s.shape)
    R, Q_minus1_s = pad_matrices(R_t.T, Q_minus1_s)
    RQ_minus1_s = np.matmul(R, Q_minus1_s)
    print("G, RQ_minus1_s", G.shape, RQ_minus1_s.shape)
    G, RQ_minus1_s = pad_matrices(G, RQ_minus1_s)
    print("G, RQ_minus1_s", G)
    print("RQ_minus1_s", RQ_minus1_s)
    GRQ_minus1_s = np.matmul(G, RQ_minus1_s)

    P1_Tv = np.matmul(P_1_padded_T, V_padded1)
    Q_minus1, P1_Tv = pad_matrices(Q_minus1, P1_Tv)
    Q_minus1P1_Tv = np.matmul(Q_minus1, P1_Tv)
    R, Q_minus1P1_Tv = pad_matrices(R, Q_minus1P1_Tv)
    RQ_minus1P1_Tv = np.matmul(R, Q_minus1P1_Tv)
    G, RQ_minus1P1_Tv = pad_matrices(G, RQ_minus1P1_Tv)
    GRQ_minus1P1_Tv = np.matmul(G, RQ_minus1P1_Tv)
    #print("GRQ_minus1_s", GRQ_minus1_s.shape)
    #print("GRQ_minus1P1_Tv", GRQ_minus1P1_Tv.shape)
    #GRQ_minus1_s, GRQ_minus1P1_Tv = pad_to_subtract(GRQ_minus1_s, GRQ_minus1P1_Tv)
    #upper_boundaries = GRQ_minus1_s + GRQ_minus1P1_Tv
    h, GRQ_minus1P1_Tv = pad_to_subtract(h, GRQ_minus1P1_Tv)
    upper_boundaries = GRQ_minus1_s
    lower_boundaries = h - GRQ_minus1P1_Tv
    upper_boundaries, lower_boundaries = pad_to_subtract(upper_boundaries, lower_boundaries)
    print("upper_boundaries", upper_boundaries)
    print("lower_boundaries", lower_boundaries)
    #upper_boundaries, h = pad_to_subtract(upper_boundaries, h)

    new_boundaries = np.array((upper_boundaries, h))
    #print(new_boundaries)

    return(E, n, min_S_sq, LDP_const, f, new_boundaries)


def solve_for_A(A: np.array, B: np.array, X: np.array, n):
    """Function to find optimal matrix A for PCA performed on original data 
    matrix. Will calculate a new row for each in initial A value and update
      the matrix with these."""
    
    for i in range(len(A)):
        new_row, resid = update_a(A, B, X, i)
        A_shape = A.shape
        print("A shape", A.shape)
        if new_row.shape[0] >= A.shape[1]:
            #print("A shape", A_shape)
            #print("new_row", new_row.shape)
            zeros_to_add = len(new_row) - A_shape[1]
            #print(zeros_to_add)
            new_A = np.pad(A, pad_width=((0, 0), (0, zeros_to_add)))
            #A = np.hstack((A, zeros))
            print("A shape", new_A.shape)
        A = new_A    
        A[i] = new_row
    
    print(A)


def solve_for_B(A: np.array, B: np.array, X: np.array):
    """Function to find optimal matrix B for PCA performed on original data
    matrix. Will calculate a new row for each initial row in B and update 
    the matrix with these."""


def inverse_or_pseudo(matrix):
    try:
        # Try to compute the inverse
        inv_matrix = np.linalg.inv(matrix)
        return inv_matrix
    except np.linalg.LinAlgError:
        # If an error occurs, compute the pseudo-inverse
        pseudo_inv_matrix = np.linalg.pinv(matrix)
        return pseudo_inv_matrix


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


def check_shape(n: np.array):
    """Function to ensure any 1D arrays are in correct form for later use"""

    if len(n.shape) == 1:
        #Potential issue -> what if n is expected to be a row vector not column?
        n = np.expand_dims(n, axis=1)
    return n


def LDP_to_NNLS():
    """Function to calculate needed matrices and constants for the transformation
    of a problem between LDP form and that of NNLS."""


def problem_setup(data: np.array):
    """Function takes data matrix and boundary vectors then
    sets up needed variables for the BPCA algorithm, including
    performing standardisation, SVD and calculation of the
    component scores and component loading matrices"""

    std_input = standardise(data)
    #Performing SVD on standardised, mean-centered data
    U, s, Vt = svd(std_input, full_matrices=True)
    s = diagsvd(s, U.shape[0], Vt.shape[-1])

    #Principal Axes (PAs) and Component Scores (CSs) calculated from SVD output
    #for ||X-AB'||**2
    CSs = np.matmul(U, s) # A (Component Scores)
    CSs = check_shape(CSs)

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

    return s, CSs, CLM, PAs


def drop_null_cols(AMatrix: np.array):
    null_cols = []

    for col in range(AMatrix.shape[1]):
        if sum(AMatrix[:,col]) == 0:
            null_cols.append(col)
    matrix_out = np.delete(AMatrix, null_cols, axis=1)
    return matrix_out


def drop_null_rows(AMatrix: np.array):
    null_rows = []

    for row in range(AMatrix.shape[0]):
        if sum(AMatrix[:,row]) == 0:
            null_rows.append(row)
    matrix_out = np.delete(AMatrix, null_rows, axis=0)
    return matrix_out


# Data needs to be organised as having samples in rows, variables in columns for later calculations
data = np.array([[1, 15, 3, 4, 5, 5.2, 18, 4, 0.2, 1], [6, 7, 8, 9, 10, 3, 4, 5, 5.2, 18]]).T 
data2 = np.array([[5, 4, 3, 3, 1, 12, 50 , 8.55, 7.2, 90], [10, 20 , 8, 7, 6, 50 , 8.55, 7.2, 90, 1]]).T
data = np.hstack((data,data2))
print(data.shape)
#Number of dimensions to reduce by
n = 1
#Boundaries arrays
bounds_vec = np.array(([0,0,0,0,0,0,0,0,0,0], [99,99,99,99,99,99,99,99,99,99])).T

s, CSs, CLM, PAs = problem_setup(data)

solve_for_A(CSs, CLM, bounds_vec, n)