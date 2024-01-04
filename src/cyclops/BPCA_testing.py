"""
Experimental BPCA class for cyclops.

Reduces dimensionality of input while maintaining boundary conditions.

# Author: Ciara Byers <ciara.byers@ukaea.uk>
(c) Copyright UKAEA 2023.
"""
import numpy as np
import random as rnd
from scipy.linalg import svd, diagsvd, norm
from scipy.optimize import least_squares, Bounds, nnls, lsq_linear
from sklearn import decomposition as decomp

#checked
def standardise(input_dims: np.array, X: np.array) -> np.array:
    """ Takes a dataset and standardises the variance of the different 
    dimensions. This is done to prevent bias in a particular direction due to 
    the values of the variables in it skewing higher than other directions."""

    if not isinstance(input_dims, np.ndarray):
        input_dims = np.array(input_dims)

    Upper = X[1]
    Lower = X[0]
    #reshape to be compatible with data - maybe not needed for real data?
    std_dims = np.zeros(input_dims.shape)
    std_hi = np.zeros(X[0].shape)
    std_lo = np.zeros(X[1].shape)
    #Want to take mean along columns
    for i in range(4):
        dim_array = np.array(input_dims[:, i])
        hi_array = np.array(Upper[:, i])
        lo_array = np.array(Lower[:, i])
        if not isinstance(dim_array, np.ndarray):
            dim_array = np.array(dim_array)
        
        #creating mean-centered, standardised data
        mean_array = dim_array.mean()
        stdev = np.std(dim_array)
        standised_dim = np.subtract(dim_array, mean_array)
        standised_dim = np.divide(standised_dim.astype(float), stdev.astype(float))
        std_dims[:,i] = standised_dim
        #performing the same transform on boundary conditions
        stdhi_cond = np.subtract(hi_array.astype(float), mean_array.astype(float))
        stdhi_cond = np.divide(stdhi_cond.astype(float), stdev.astype(float))
        std_hi[:,i] = stdhi_cond
        stdlo_cond = np.subtract(lo_array.astype(float), mean_array.astype(float))
        stdlo_cond = np.divide(stdlo_cond.astype(float), stdev.astype(float))
        std_lo[:,i] = stdlo_cond
        i += 1

    return(np.array(std_dims),np.array((std_lo, std_hi)))


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


def update_row(A: np.array, B: np.array, X: np.array, I: float):
    """Function updates row I of matrix A according to BPCA algorithm
    (This is used to update rows in both "A" and "B" from the in-paper
    equations, but is written using the A orientated INSERTFORGOTTENWORDHERE)"""

    print("Run", I, " X shape ", X.shape) 
    X_low = X[0][I] 
    X_high = X[1][I]

    #Start from LSI problem
    #checked
    def calc_top_const(A: np.array, B: np.array, X: np.array, I: float):
        """Function calculates the value of the constant in the initial
        minimisation problem for a given row I"""
        #Calculating the constant for row being updated, row I
        #const is given by mod(X_noi - A_noi*B.T)^2
        #print(X[0])
        X_noi_L = np.delete(X[0][:], I, 0)
        X_noi_H = np.delete(X[1][:], I, 0) 
        new_row_len = X_noi_L.shape[0] + X_noi_H.shape[0]
        X_noi = np.array([X_noi_L, X_noi_H])
        X_noi = np.reshape(X_noi, (new_row_len, X.shape[2]))
        A_noi = np.delete(A, I, 0) 
        #Pad A_noi and/or B to allow matrix multiplication
        A_noi, B_T = pad_matrices(A_noi, B.T) 
        AB_T = np.matmul(A_noi,B_T)
        #Pad X_noi/AB to allow matrix subtraction
        X_noi, AB_T = pad_to_subtract(X_noi, AB_T) 
        inner = X_noi - AB_T 
        In0, In1 = inner.shape
        if In0>In1: 
            inner = np.pad(inner, pad_width=((0,0), (0, In0 - In1))) 
        elif In1>In0: 
            inner = np.pad(inner, pad_width=((0, In1 - In0), (0,0)))
        LSI_row_const= np.square(norm(inner))
        print("LSI_row_const", LSI_row_const)
        return LSI_row_const

    LSI_row_const= calc_top_const(A, B, X, I)

    #seeking optimal row A[I] in boundaries now
    W = A[I]
    #Initial guess for V, not expected to be accurate
    X_high_sample = len(range(0, int(max(X_high))))
    V = rnd.sample(range(int(max(X_high))), min(X_high_sample,len(B))) 
    V = np.array(V) 
    V = check_shape(V) 
    G = np.c_[B, -B] 
    h = np.c_[X_low, -X_high] 

    #As in paper, let S = Qy-P_i.T*v to get to a LDP problem as follows
    #min||s||**2 + ||p_2.T*v||**2 s.t. GRQ^1s > h -GRQ^-1P_1'*v
    Z, R_minus1, f_1, LDP_const, new_bounds = LSI_to_LDP(G, h, B, W, V)

    #LDP problem Iz>=L is r = Eu -f where column vec E= [I.T, L.T], f = vec
    #let I=new_bounds[1], L=new_bounds[0] and x = Z
    solution_vec, resid = LDP_to_NNLS_sol(new_bounds[1], new_bounds[0], Z)

        #solve for solution vector x = Ky = R-1(Z + f_1)
    #sol_vec_x = np.matmul()
    print("solution_vec", solution_vec)
    solution_vec += LDP_const

    r = [x + LSI_row_const for x in solution_vec]
    
    return r, resid


def LDP_to_NNLS_sol(G: np.array, h: np.array, x: np.array):
    """Function to calculate needed matrices and constants for the transformation
    of a problem between LDP form and that of NNLS."""

    ht_padded, gt_padded = pad_to_subtract(h.T, G.T)
    E = np.vstack((gt_padded, ht_padded))
    n = E.shape[0]
    f = np.zeros((n))
    f = np.append(f, [1])
    f = f.T   
    #can now use NNLS to compute an m-vector, u, to solve NNLS problem:
    #Minimize ||Eu — f|| subject to  u>=0
    E_pad = np.pad(E, pad_width=((0, 1), (0, 0)))
    u, resid = nnls(E_pad, f)
    # print("E", E)
    # print("u", u)
    u = check_shape(u)
    E_pad, u_pad = pad_matrices(E, u)
    Eu = np.matmul(E_pad, u_pad)
    f = check_shape(f)
    Eu_pad, f_pad = pad_to_subtract(Eu, f)
    r = np.subtract(Eu_pad, f_pad)
    # print("r", r)

    phi = True
    print("r[n+1]", r[-1])
    rn = r[-1]
    x_sol = -1*np.divide(r, rn)
    # while phi == True:
    #     if norm(r) == 0:
    #         phi = False
    #     r = -np.divide(r[0], r[n])
    #     for j in range(1, n):
    #         print("rj", r[j-1])
    #         x_j = -np.divide()
    #         # print("x_j", x_j)
    #         x_sol[j] = x_j
            # print(x_sol)

    return x_sol, resid


def LSI_to_LDP(G: np.array, h: np.array, E: np.array, X: np.array, F: np.array):
    """Transforms a bounded LSI problem to a bounded LDP one with use of SVD on
    E, rewriting the problem as an LDP + constant with new bounds. See 
    'Principal component analysis with boundary constraints' by Paolo Giordani
     and Henk A. L. Kiers. This does not aim to *solve* the problem, simply
     return the components of the transformed form of it."""

    #Decompose matrix E
    Q, R, K_t = np.linalg.svd(E, full_matrices=True) 
    R = diagsvd(R, Q.shape[-1], K_t.shape[0]) 
    reconst = np.matmul(R, K_t)
    reconst = np.matmul(Q, reconst)
    n_size = K_t.shape[0] #m
    X = check_shape(X) 
    K_t, X = pad_matrices(K_t, X) # because x=Ky
    y = np.matmul(K_t, X) # check
 
    #Let z = Ry-Q_1.T*f  s.t. our problem is now min z ||z||**2 ||Q_2.T v||**2
    # s.t. GK(R^-1)*z >= h - GK(R^-1)*(Q_1.T)*f
    h_padded, g_padded = pad_to_subtract(h.T, G.T) 
 
    #Note the shapes of Q_1 and Q_2 are (nxm) and (nx(n-m)) where Q from 
    # earlier is (mxm) and K_t is (nxn)
    Q_1 = Q[:, :n_size] 
    Q_2 = Q[:, n_size:]

    Ry = np.matmul(R, y)
    #f_tilde = Q_T * f
    Q_1_padded_T, F_padded = pad_matrices(Q_1.T, F) 
    f_1 = np.matmul(Q_1_padded_T, F_padded) 
    Ry_padded, f1_padded = pad_to_subtract(Ry, f_1)
    z_LDP = np.subtract(Ry_padded, f1_padded)
    Z = [Ry_padded, f1_padded] # will be passed back to be treated as LDP 

    #calc constant
    Q_2_padded_T, F_padded = pad_matrices(Q_2.T, F)
    f_2 = np.matmul(Q_2_padded_T, F_padded)
    LDP_const = np.square(np.linalg.norm(f_2)) #should this be squared?
    print("LDP_const", LDP_const)

    #Calulate new, transformed boundaries
    G, K = pad_matrices(G, K_t.T)
    GK = np.matmul(G, K)
    #R may not be square, check and calculate pseudoinverse instead of
    # inverse if not
    R_minus1 = inverse_or_pseudo(R)
    R_minus1_z = np.matmul(R_minus1, z_LDP)
    GK, R_minus1_z = pad_matrices(GK, R_minus1_z)
    upper_boundaries = np.matmul(GK, R_minus1_z)

    R_minus1, f_1 = pad_matrices(R_minus1, f_1)
    R_minus1f1 = np.matmul(R_minus1, f_1)
    GK, R_minus1f1 = pad_matrices(GK, R_minus1f1)
    GKR_minus1f1 = np.matmul(GK, R_minus1f1)
    h, GKR_minus1f1 = pad_to_subtract(h, GKR_minus1f1)
    lower_boundaries = np.subtract(h, GKR_minus1f1)
    upper_boundaries, lower_boundaries = pad_to_subtract(upper_boundaries,
                                                         lower_boundaries)

    new_boundaries = np.array((upper_boundaries, lower_boundaries))
    print("LDP_bounds", new_boundaries)

    return(Z, R_minus1, f_1, LDP_const, new_boundaries)


def check_xi_bounds(X: np.array, I: float, B: np.array, A: np.array):
    """Function to check if the new row found for matrix A obeys the
    necessary boundary conditions for acceptance, else raises an error."""

    X_low = X[0][I]
    X_high = X[1][I]
    Row_Xi = check_shape(X_low)
    Col_Xi = check_shape(X_high)
    A_i = A[I]
    A_i = check_shape(A_i)
    Ai_T, B_T = pad_matrices(A_i.T, B.T)
    AiB_T = np.matmul(Ai_T, B_T)

    low_check = X_low < AiB_T
    high_check = X_high > AiB_T
    print(low_check)
    print(high_check)
    contains_false = ((low_check == False).any() or
                      (high_check == False).any())
    print(contains_false)

    if contains_false:
        raise ValueError("Boundary conditions violated during 'check_xi_bounds'!")


def solve_for_A(A: np.array, B: np.array, X: np.array, n):
    """Function to find optimal matrix A for PCA performed on original data 
    matrix. Will calculate a new row for each in initial A value and update
      the matrix with these."""

    A = check_shape(A) 
    B = check_shape(B) 
    A, B_T = pad_matrices(A, B.T) 
    AB_T = np.matmul(A, B_T) 
    X_lo = X[0] 
    X_hi = X[1] 
    lo_check = X_lo <= AB_T 
    hi_check = X_hi >= AB_T 
    print("lo_check", (lo_check == False).any())
    print("hi_check",  (hi_check == False).any())
    contains_false = ((lo_check == False).any() or
                      (hi_check == False).any()) 
    if contains_false:
         print("Boundary conditions violated in 'solve_for_A'!")

    for i, j in zip(range(2),range(2)):
        print("I is : ", i) 
        print("J is : ", j)         
        new_row_a, resid = update_row(A, B, X, i)
        #Check boundary conditions apply: row(X_i).T =< A_i.T * B.T =< col(X_i).T
        # new_row_a = np.array(new_row_a)
        # A_shape = A.shape
        # print("Ashape", A_shape)
        # new_row_a = check_shape(new_row_a)
        # if new_row_a.shape[0] > A.shape[1]:
        #     zeros_to_add = len(new_row_a) - A_shape[1]
        #     new_A = np.pad(A, pad_width=((0, 0), (0, zeros_to_add)))
        # elif A.shape[1] > new_row_a.shape[0]:
        #     zeros_to_add = A.shape[1] - len(new_row_a)
        #     new_row_a = np.pad(new_row_a, pad_width=((0, 0), (0, zeros_to_add)))
        #     print("new_row_a.shape", new_row_a.shape)
        # if len(new_row_a.shape) > 1:
        #     A[i] = new_row_a.flatten()
        # else:
        #     A[i] = new_row_a
        # print("A", A)
        # check_xi_bounds(X, i, B, A)

    # for i in range(len(B)):

    #     new_row_b, resid = update_row(B, A, X, i)
    #     #Check boundary conditions apply: row(X_i).T =< A_i.T * B.T =< col(X_i).T
    #     check_xi_bounds(X, i, B, A)

    #     B_shape = B.shape
    #     if new_row_b.shape[0] >= B.shape[1]:
    #         zeros_to_add = len(new_row_b) - B_shape[1]
    #         new_B = np.pad(A, pad_width=((0, 0), (0, zeros_to_add)))
    #     B = new_B    
    #     B[i] = new_row_b


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


def problem_setup(data: np.array, X: np.array,n: int):
    """Function takes data matrix and boundary vectors then
    sets up needed variables for the BPCA algorithm, including
    performing standardisation, SVD and calculation of the
    component scores and component loading matrices"""

    std_input, std_conds = standardise(data, X) 
    print("std_input", std_input)
    print("std_conds", std_conds)
    #Performing SVD on standardised, mean-centered data
    U, s, Vt = svd(std_input, full_matrices=True) 
    s = diagsvd(s, U.shape[0], Vt.shape[-1])

    pca = decomp.PCA(n_components=n)
    pca.fit(std_input)
    # A (Component Scores/ Principal Components)
    pca_PAs = pca.components_.T
    component_scores = pca.transform(std_input)
    # B (Component loading matrix)
    pca_clm = np.dot(pca.components_.T, np.sqrt(pca.explained_variance_))

    return s, component_scores, pca_clm, pca_PAs, std_conds

def drop_null_cols(AMatrix: np.array):
    null_cols = []

    cov_matrix = np.dot(std_input.T, std_input) / len(std_input)
    for eigenvector in pca.components_:
        print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))


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
data = np.array([[2, 6, 2, 3, 3, 2, 3, 2, 7, 3], [3, 2, 3, 3, 3, 2, 3, 2, 1, 3]]).T 
data2 = np.array([[2, 3, 3, 2.6, 2.7, 2, 3, 5, 3, 3], [2, 3, 6, 4, 2, 2, 3, 2, 3, 3]]).T
data = np.hstack((data,data2))
#Number of dimensions to reduce by
n = 1
#Boundaries arrays
low_bound = np.ones((10,4))*-2
high_bound = low_bound*-10
bounds_vec = np.array((low_bound, high_bound))

s, CSs, CLM, PAs, bounds_vec = problem_setup(data, bounds_vec, n)
solve_for_A(CSs, CLM, bounds_vec, n)