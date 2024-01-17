"""
Experimental BPCA class for cyclops.

Reduces dimensionality of input while maintaining boundary conditions.

(c) Copyright UKAEA 2023.
"""
import numpy as np
from scipy.linalg import svd, diagsvd, norm
from scipy.optimize import lsq_linear
from sklearn import decomposition as decomp


def pad_matrices(matrix1: np.array, matrix2: np.array):
    """Function to enable matrix multiplication of otherwise incompatible
    matrices. The matrices MUST be entered in the order they will appear
    in the multiplication, otherwise this will pad the wrong dimensions!
    
    Args:
    ----
    matrix1 : ndarray with shape (n, m) containing floats. Will be treated
        as if it were the first matrix in a matrix multiplication, if dim m 
        is less than the first dim of matrix2 it will be increased to match 
        the first dim of matrix2 by the addition of columns of zeros.
    matrix2 : ndarray with shape (i, j) containing floats. Will be treated
        as if it were the second matrix in a matrix multiplication, if dim i
        is less than the second dim of matrix1 it will be increased to match 
        the second dim of matrix1 by the addition of rows of zeros.
    
    Returns:
    -------
    Returned variables dependent on relative sizes of input matrices, it will
    be a combination of the following:
    matrix1 : ndarray with shape (n, m) containing floats. This is unchanged
        from the input variable matrix1.
    matrix2 : ndarray with shape (i, j) containing floats. This is unchanged
        from the input variable matrix2.
    padded_matrix1 : ndarray with shape (n, j) containg floats.
    padded_matrix2 : ndarray with shape (i, m) containing floats."""

    rows1, cols1 = matrix1.shape
    rows2, cols2 = matrix2.shape

    # Check if compatible, if so, return as is
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
    rows and columns of the matrices to ensure they have the same shape the
    returns these new padded matrices.
    
    Args:
    ----
    matrix1 : (ndarray with shape (n, m) of floats) Will be treated as
        if it were to be added or subtracted from another matrix, if n is
        smaller than the first dim of matrix2 then it will be increased to
        match first dim of matrix2 by the addition of columns of zeros. The
        same will happen to the rows if there are less than in matrix2.
    matrix2 : (ndarray with shape (i, j) of floats) Will be treated as
        if it were to be added or subtracted from another matrix, if i
        is smaller than the first dim of matrix1 then it will be increased
        to match first dim of matrix1 by the addition of rows of zeros. The
        same will happen to the rows if there are less than in matrix1.
    
    Returns:
    -------
    Returned variables dependent on relative sizes of input matrices, will
    be a combination of the following:
    padded_matrix1 : ndarray with shape (n, m) = (i, j) containg floats.
    padded_matrix2 : ndarray with shape (i, j) = (n, m) containing floats."""

    rows1, cols1 = matrix1.shape
    rows2, cols2 = matrix2.shape

    # Check shapes are not already the same, return if they are
    if rows1 == rows2 and cols1 == cols2:
        return matrix1, matrix2

    # Pad matrix1 with rows to match matrix2 if it has more
    if rows1 < rows2:
        zeros_to_add = rows2 - rows1
        padded_matrix1 = np.pad(matrix1, pad_width=(
                                            (0, zeros_to_add), (0, 0)))
        padded_matrix2 = matrix2
    # Pad matrix2 with rows to match matrix1 if it has more
    elif rows1 > rows2:
        zeros_to_add = rows1 - rows2
        padded_matrix1 = matrix1
        padded_matrix2 = np.pad(matrix2, pad_width=(
                                            (0, zeros_to_add), (0, 0)))
    else:
        padded_matrix1 = matrix1
        padded_matrix2 = matrix2

    # Pad matrix1 with columns if matrix2 has more
    if cols1 < cols2:
        zeros_to_add = cols2 - cols1
        padded_matrix1 = np.pad(padded_matrix1, pad_width=(
                                                (0, 0), (0, zeros_to_add)))
    # Pad matrix2 with columns if matrix1 has more
    elif cols1 > cols2:
        zeros_to_add = cols1 - cols2
        padded_matrix2 = np.pad(padded_matrix2, pad_width=(
                                                (0, 0), (0, zeros_to_add)))
    
    return padded_matrix1, padded_matrix2


def inverse_or_pseudo(matrix: np.array):
    """Function attempts to invert the matrix, if this results in an error
    then it finds the pseudo-inverse instead. Both methods used for these
    calculations are from NumPy's linalg module.
    
    Args:
    ----
    matrix : (ndarray of floats) Function will attempt to invert this
        matrix or alternatively find the pseudo-inverse.
    
    Returns:
    -------
    inv_matrix : (ndarray of floats) This is the true inverse of the input
        matrix.
    pseudo_inv_matrix : (ndarray of floats) This is the pseudo-inverse of the
        input matrix."""

    try:
        # Try to compute the inverse
        inv_matrix = np.linalg.inv(matrix)
        return inv_matrix
    except np.linalg.LinAlgError:
        # If an error occurs, compute the pseudo-inverse
        pseudo_inv_matrix = np.linalg.pinv(matrix)
        return pseudo_inv_matrix


def check_shape(n: np.array):
    """Function to ensure 1D arrays have two indices in order to perform
    calculations with them.
    
    Args:
    ----
    n : (ndarray of floats) If n has dimension (m,) then it will be expanded
        to (m,1) in order to allow calculations.
    
    Returns:
    -------
    n : (ndarray of floats) If the input array already had at least 2
        dimensions this will be the same as the input array, if it had only 1
        dimension then this will be the expanded version."""

    if len(n.shape) == 1:
        n = np.expand_dims(n, axis=1)
    return n


def standardise(input_dims: np.array, X: np.array) -> np.array:
    """ Takes a dataset and standardises the variance of the different
    dimensions. This is done to prevent bias in a particular direction due to
    the values of the variables in it skewing higher than other directions.
    
    Args:
    ----
    input_dims : (ndarray of floats) Data input with original number of
        dimensions. This will be standardised by subtracting the column mean
        from each entry and dividing by the column standard deviation of that
        column.
    X : (ndarray of floats) Boundary input in original format. This will be
        standardised to match the dataset it is associated with, having the
        mean of each of the data columns subtracted from the corresponding
        columns and being divided by the column standard deviation of the
        same columns. 
    
    Returns:
    -------
    std_dims : (ndarray of floats) Standardised data created from the
        original data.
    std_bounds : (ndarray of floats) Array contains two sub-arrays, the
        first is the standardised lower bounds for the data, the second is
        the standardised upper bounds for the data.
    rev_data_std : (ndarray of floats) Array contains two sub-arrays, the
        first contains the mean values for each column in the dataset, the
        second contains the standard deviation values for each column in the
        dataset."""

    if not isinstance(input_dims, np.ndarray):
        input_dims = np.array(input_dims)

    Upper = X[1]
    Lower = X[0]
    #reshape to be compatible with data - maybe not needed for real data?
    std_dims = np.zeros(input_dims.shape)
    std_hi = np.zeros(X[0].shape)
    std_lo = np.zeros(X[1].shape)
    #variables to hold data for reversal of standardisation
    rev_data_std = []
    array_mean_val = 0
    mean_val_list = []
    stdev_val_list = []
    #Want to take mean along columns
    for i in range(4):
        dim_array = np.array(input_dims[:, i])
        hi_array = np.array(Upper[:, i])
        lo_array = np.array(Lower[:, i])
        if not isinstance(dim_array, np.ndarray):
            dim_array = np.array(dim_array)
        #Creating mean-centered, standardised data
        array_mean_val = dim_array.mean()
        stdev = np.std(dim_array)
        standised_dim = np.subtract(dim_array, array_mean_val)
        standised_dim = np.divide(
                            standised_dim.astype(float), stdev.astype(float))
        std_dims[:,i] = standised_dim
        mean_val_list.append(array_mean_val)
        stdev_val_list.append(stdev)
        #Performing the same transform on boundary conditions
        stdhi_cond = np.subtract(hi_array.astype(float),
                                 array_mean_val.astype(float))
        stdhi_cond = np.divide(stdhi_cond.astype(float),
                               stdev.astype(float))
        std_hi[:,i] = stdhi_cond
        stdlo_cond = np.subtract(lo_array.astype(float),
                                 array_mean_val.astype(float))
        stdlo_cond = np.divide(stdlo_cond.astype(float),
                               stdev.astype(float))
        std_lo[:,i] = stdlo_cond
        i += 1
    mean_val_list = np.array(mean_val_list)
    stdev_val_list = np.array(stdev_val_list)
    rev_data_std = np.vstack(((mean_val_list.T), (stdev_val_list.T)))
    std_bounds = np.array((std_lo, std_hi))

    return(np.array(std_dims), std_bounds, rev_data_std)


def problem_setup(data: np.array, X: np.array, n: int):
    """Function takes data matrix and boundary vectors then sets up needed
    variables for the BPCA algorithm, including performing standardisation,
    SVD and calculation of the component scores and component loading
    matrices.
        
    Args:
    ----
    data : (ndarray of floats) This is the standardised data matrix input to
        the function.
    X : (ndarray of floats) This is the pair of lower and upper boundary
        arrays that has been standardised based on the data matrix.
    n : (int) This is the number of dimensions to keep when performing the
        PCA.
    
    Returns:
    -------
    s : (ndarray of floats) This is the matrix of singular values found when 
        performing SVD on the data input.
    component_scores : (ndarray of floats) This is an array containing the
        component_scores found when performing PCA on the data input.
    pca_clm : (ndarray of floats) This is the component loading matrix from
        the PCA.
    pca_PAs : (ndarray of floats) This is the matrix of the 'Principal
        Components' from performing PCA on the data input.
    std_conds : (ndarray of floats) This contains the transformed
        (standardised) upper and lower bounds.
    rev_std : (ndarray of floats) Array contains two sub-arrays, the
        first contains the mean values for each column in the dataset, the
        second contains the standard deviation values for each column in the
        dataset."""

    std_input, std_conds, rev_std = standardise(data, X)
    #Performing SVD on standardised, mean-centered data
    U, s, Vt = svd(std_input, full_matrices=True)
    s = diagsvd(s, U.shape[0], Vt.shape[-1])
    #Performing PCA on same data
    pca = decomp.PCA(n_components=n)
    pca.fit(std_input)
    # A (Component Scores/ Principal Components)
    pca_PAs = pca.components_.T
    component_scores = pca.transform(std_input)
    # B (Component loading matrix)
    pca_clm = np.dot(pca_PAs, np.sqrt(pca.explained_variance_))

    return s, component_scores, pca_clm, pca_PAs, std_conds, rev_std


def solve_for_A_B(A: np.array, B: np.array, X: np.array, org_dat: np.array,
                  epsilon: float):
    """Function to perform BPCA on a dataset 'org_dat'. Will find optimal
    matrices A and B to reconstruct the original dataset. Finds a new row for
    A and B in alternating turns until all have been updated, then checks the
    change in the loss function to see if the algorithm has converged.
    CSs, CLM,
    Args:
    ----
    A : (ndarray of floats) This should be the component score matrix from a
        PCA performed on the original data matrix.
    B : (ndarray of floats) This should be the Component Loading matrix from a
        PCA performed on the original data matrix.
    X : (ndarray of floats) This should contain the data-standardised lower
        and upper bounds to be applied to the dataset.
    org_dat : (ndarray of floats) The original data matrix containing the
        standardised data to have BPCA performed on.
    epsilion : (float) This value is used to determine whether convergence
        has been reached in a given iteration. The smaller this value is the
        stricter the requirement for convergence. 
    
    Returns:
    -------
    A : (ndarray of floats) The final matrix, A, that holds the reduced
        dimension data. If right-multiplied by B this will reconstruct an
        approximation of the original dataset. Accuracy is determined by
        the value of epsilon and on the underlying algorithms' ability to
        find a solution meeting that critera.
    B : (ndarray of floats) The final matrix, B, that holds the data needed
        to reconstruct the missing dimensionality in the matrix A to transform
        it back to an approximation of the original dataset. Reconstruction
        requires that B is left-multiplied."""

    A = check_shape(A) 
    B = check_shape(B) 
    A, B_T = pad_matrices(A, B.T) 
    AB_T = np.matmul(A, B_T)
    X_lo = X[0] 
    X_hi = X[1]
    lo_check = X_lo <= AB_T
    hi_check = X_hi >= AB_T
    contains_false = ((lo_check == False).any() or
                      (hi_check == False).any()) 
    if contains_false:
         raise ValueError("Boundary conditions violated in 'solve_for_A_B'!")

    def F(A: np.array, B: np.array, X: np.array):
        """Calculates the loss function for the matrices A and B according to
        F = ||X-AB.T||**2
        
        Args:
        ----
        A : (ndarray of floats) This should be the result of the updates to
            the matrix A that was originally input at the start of the
            solve_A_B function.
        B : (ndarray of floats) This should be the result of the updates to
            the matrix B that was originally input at the start of the
            solve_A_B function.
        X : (ndarray of floats) This should contain the transformed lower
            and upper bounds to be applied to the dataset.
        
        Returns:
        -------
        f : (float) This is the newly calculated cost function value for the
            updated A and B matrices.
        """

        A, B_T = pad_matrices(A, B.T) 
        AB_T = np.matmul(A, B_T)
        X_pad, AB_T_pad = pad_to_subtract(X, AB_T)
        XminAB_T = np.subtract(X_pad, AB_T_pad)
        f = (norm(XminAB_T))**2

        return float(f)

    f = F(A, B, org_dat)
    new_f = f-1
    #Loop calculates an update row for A then B in succession
    while abs(f-new_f) > epsilon:
        print("Difference between current and previous f value is {0}".format(
            f-new_f))
        f = new_f
        for i,j in zip(range(len(A)-1), range(len(B)-1)):
            new_row_a, resid = update_row(A, B, X, i, 0)
            new_row_a = np.array(new_row_a)
            A_shape = A.shape
            new_row_a = check_shape(new_row_a)
            if new_row_a.shape[0] > A.shape[1]:
                zeros_to_add = len(new_row_a) - A_shape[1]
                A = np.pad(A, pad_width=((0, 0), (0, zeros_to_add)))
                A[i] = new_row_a.flatten()
            
            elif A.shape[1] > new_row_a.shape[0]:
                zeros_to_add = A.shape[1] - len(new_row_a)
                new_row_a = np.pad(new_row_a, pad_width=((0, 0), 
                                                         (0, zeros_to_add)))
                A[i] = new_row_a
            if len(new_row_a.shape) > 1:
                A[i] = new_row_a.flatten()
            else:
                A[i] = new_row_a
            #Check boundary conditions apply:
            #row(X_i).T =< A_i.T * B.T =< col(X_i).T    
            check_xi_bounds(X, i, A, B, row_col=0)

            new_row_b, resid = update_row(B, A, X, j, 1)
            new_row_b = np.array(new_row_b)
            B_shape = B.shape
            new_row_b = check_shape(new_row_b)
            if new_row_b.shape[0] > B.shape[1]:
                zeros_to_add = len(new_row_b) - B_shape[1]
                B = np.pad(B, pad_width=((0, 0), (0, zeros_to_add)))
                B[i] = new_row_b.flatten()
            
            elif B.shape[1] > new_row_b.shape[0]:
                zeros_to_add = B.shape[1] - len(new_row_b)
                new_row_b = np.pad(new_row_b, pad_width=((0, 0),
                                                         (0, zeros_to_add)))
                B[i] = new_row_b
            if len(new_row_b.shape) > 1:
                B[i] = new_row_b.flatten()
            else:
                B[i] = new_row_b
            #Check boundary conditions apply:
            #row(X_i).T =< B_i.T * B.T =< col(X_i).T 
            check_xi_bounds(X, j, B, A, row_col=1)
        print("old f ", f)
        new_f = F(A, B, org_dat)
        print("new_f ", new_f)

    return A, B


def update_row(M1: np.array, M2: np.array, X: np.array, I: int, col_row: int):
    """Function updates row I of matrix M1 according to the BPCA algorithm
    found in Chemometrics 2007; 21: 547–556 #DOI: 10.1002/cem . This is used
    to update rows in both "A" and "B", if updating A then M1=A and col_row=0,
    if updating B then M1=B and col_row=1. 
    
    Args:
    ----
    M1 : (ndarray of floats) the matrix to update the row of, either A or B. 
    M2 : (ndarray of floats) the 'partner matrix' i.e. either B or A, needed
        to calculate the update to the first matrix.
    
    Returns:
    -------
    sol_vec_x : (ndarray of floats) Current solution vector X for
        ||X-AB.T||**2 .
    resid : (float) Residual for the current solution vector X."""

    #Seperating the matrices for the lower and upper bounds as they are passed
    #in a single array
    X_low = X[0]
    X_high = X[1]

    #Determining if function needs to calculate a row of the original matrix A
    #or a row of the original matrix B which are col_row=0,1 respectively
    if col_row == 0:
        #Algorithm requires rows of X bounds
        X_low = X_low[I]
        X_high = X_high[I]      
    elif col_row == 1:
        #Algorithm requires columns of X bounds
        X_low = X_low[:,I]
        X_high = X_high[:,I]

    #seeking optimal row M1[I] in boundaries now
    W = M1[I]
    #Initial guess for V, not expected to be accurate
    V = np.zeros((10,1))
    G = np.c_[M2, -M2] 
    h = np.c_[X_low, -X_high]
    #As in paper, let S = Qy-P_i.T*v to get to a LDP problem as follows
    #min||s||**2 + ||p_2.T*v||**2 s.t. GRQ^1s > h -GRQ^-1P_1'*v
    Z, R_minus1, f_1, LDP_const, new_bounds = LSI_to_LDP(G, h, M2, W, V)

    #LDP problem Iz>=L is r = Eu -f where column vec E= [I.T, L.T], f = vec
    #let I=new_bounds[1], L=new_bounds[0], can use to find Z
    u_vec, r_vec, resid = LDP_to_NNLS_sol(new_bounds[1], new_bounds[0])
    
    #Algorithm from "Solving Least Squares Problems indicates that vector Z is
    #given by I.T * u_vec * norm(r_vec)**-2"
    r_norm = norm(r_vec)
    r_minus2 = r_norm**(-2)
    Ur = np.multiply(u_vec, r_minus2)
    GT_padded, Ur_padded = pad_matrices(G.T, Ur)
    z_vec = np.matmul(GT_padded, Ur_padded)

    #Can now use z_vec to find solution vector x = Ky = R-1(z_vec + f_1)
    z_vec_padded, f_1_padded = pad_matrices(z_vec, f_1)
    z_f1 = np.matmul(z_vec_padded, f_1_padded)
    R_min1_padded, z_f1_padded = pad_matrices(R_minus1, z_f1) 
    sol_vec_x = np.matmul(R_min1_padded, z_f1_padded)
    sol_vec_x += LDP_const 
    
    return sol_vec_x, resid



def LSI_to_LDP(G: np.array, h: np.array, E: np.array, w: np.array, F: np.array):
    """Transforms a bounded LSI problem to a bounded LDP one with use of SVD on
    E, rewriting the problem as an LDP + constant with new bounds. See 
    'Principal component analysis with boundary constraints' by Paolo Giordani
     and Henk A. L. Kiers. This does not aim to *solve* the problem, simply
     return the components of the transformed form of it.
     
    Args:
    ----
    G : (ndarray of floats) This should be a column array containing a matrix
        M2 in the top entry and -M2 in the bottom. This should correspond to
        one of the matrices A or B in the bounded problem |X - AB.T|**2
    h : (ndarray of floats) This should be a column array containing the upper
        bound matrix in the top entry and the lower bound matrix on the
        bounded problem |X - AB.T|**2
    E: (ndarray of floats) This should be a matrix corresponding to either A
        or B from the bounded problem |X - AB.T|**2, specifically it should be
        the one that is NOT currently being updated.
    w: (ndarray of floats) This should be a row array taken from either A or 
        B from the bounded problem |X - AB.T|**2, specifically it should be
        the row that is currently being updated.
    F: (ndarray of floats) This is an initial guess for what w should be
        updated to, it will be used to calculate an improved guess.
    
    Returns:
    -------
    Z : (list of ndarrays) This contains two ndarrays of floats which are
        needed to calculate the solution vector to the original LSI problem.
        The first is the matrix of singular values accquired through the 
        decomposition of input E, multiplied by y, which is the result of
        the right-singular vectors of the decomposition multiplied by the
        input w. The second array is the transpose of the top n rows of the
        left-singular vectors of the decomposition of E multiplied by input
        array F 
    R_minus1 : (ndarray of floats) This is the inverse (or if this does not
        exist, pseudo-inverse) of the matrix of singular values that comes
        from performing SVD on input matrix E.
    f_1 : (ndarray of floats) This is the transpose of the top n rows of the
        left-singular vectors of the decomposition of E multiplied by input
        array F. 
    LDP_const : (ndarray of floats) This is the result of subtracting the two
        components of Z from one another. It is a constant that will be added
        to the solution row to the current problem once it is found to obtain
        the final answer.
    new_boundaries : (ndarray of arrays) This contains the newly transformed
        boundaries for the LDP problem to adhere to.
    """

    #Decompose matrix E
    Q, R, K_t = np.linalg.svd(E, full_matrices=True) 
    R = diagsvd(R, Q.shape[-1], K_t.shape[0]) 
    reconst = np.matmul(R, K_t)
    reconst = np.matmul(Q, reconst)
    n_size = K_t.shape[0] #m
    w = check_shape(w)
    #let w=Ky as in "Solving Least Squares by Charles Lawson and
    #Richard Henson, chapter 23"
    K_t, w = pad_matrices(K_t, w)
    y = np.matmul(K_t, w)
 
    #Let z = Ry-Q_1.T*f  s.t. our problem is now min z ||z||**2 ||Q_2.T v||**2
    # s.t. GK(R^-1)*z >= h - GK(R^-1)*(Q_1.T)*f
    h_padded, g_padded = pad_to_subtract(h.T, G.T) 
 
    #Note the shapes of Q_1 and Q_2 are (nxm) and (nx(n-m)) where Q from 
    # earlier is (mxm) and K_t is (nxn)
    Q_1 = Q[:, :n_size] 
    Q_2 = Q[:, n_size:]

    Ry = np.matmul(R, y)
    F = check_shape(F)
    Q_1_padded_T, F_padded = pad_matrices(Q_1.T, F) 
    f_1 = np.matmul(Q_1_padded_T, F_padded) 
    Ry_padded, f1_padded = pad_to_subtract(Ry, f_1)
    z_LDP = np.subtract(Ry_padded, f1_padded)
    Z = [Ry_padded, f1_padded] # will be passed back to be treated as LDP 

    #calc constant
    Q_2_padded_T, F_padded = pad_matrices(Q_2.T, F)
    f_2 = np.matmul(Q_2_padded_T, F_padded)
    LDP_const = np.square(np.linalg.norm(f_2))

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

    new_boundaries = np.array((lower_boundaries, upper_boundaries))

    return(Z, R_minus1, f_1, LDP_const, new_boundaries)


def LDP_to_NNLS_sol(G: np.array, h: np.array):
    """Function to calculate needed matrices and constants for the
    transformation of a problem between LDP form and that of NNLS then find
    the solution vector of the NNLS problem.
    
    Args:
    ----
    G : (ndarray of floats) This should be a column array containing a matrix
        M2 in the top entry and -M2 in the bottom. This should correspond to
        one of the matrices A or B in the bounded problem |X - AB.T|**2
    h : (ndarray of floats) This should be a column array containing the upper
        bound matrix in the top entry and the lower bound matrix on the
        bounded problem |X - AB.T|**2
       
    Returns:
    -------
    u : (narray of floats) The solution vector to the NNLS problem
        min u |Eu - f|**2 .
    r : (ndarray of floats) The value of min u |Eu - f|**2, i.e. the solution
        of the LDP problem that was converted to NNLS form.
    resid : (float) The residual value for the solution vector u."""

    ht_padded, gt_padded = h.T, G.T #pad_to_subtract(h.T, G.T)
    E = np.vstack((gt_padded, ht_padded))
    n = E.shape[0]
    f = np.zeros((n-1))
    f = np.append(f, [1])
    f = f.T   
    #can now use NNLS to compute an m-vector, u, to solve NNLS problem:
    #Minimize ||Eu — f|| subject to  u>=0
    reg_nnls = lsq_linear(E, f, bounds=(0, np.inf))
    u = reg_nnls.x
    resid = reg_nnls.fun
    u = check_shape(u)
    E_pad, u_pad = pad_matrices(E, u)
    Eu = np.matmul(E_pad, u_pad)
    f = check_shape(f)
    Eu_pad, f_pad = pad_to_subtract(Eu, f)
    r = np.subtract(Eu_pad, f_pad)

    return u, r, resid


def check_xi_bounds(X: np.array, I: float, A: np.array, B: np.array, row_col: int):
    """Function to check if the new row found for matrix A obeys the
    necessary boundary conditions for acceptance, else raises an error. row_col 
    determines whether bounds are being checked using the rows or columns of X.
        
    Args:
    ----
    X : (ndarray of arrays) Contains two matrices, the upper and lower bounds
        on the update of either A or B. 
    I : (ndarray of floats) the 'partner matrix' i.e. either B or A, needed
        to calculate the update to the first matrix.
    A : (ndarray of floats) The updated matrix, either A or B depending on 
        stage of algorithm.
    B : (ndarray of floats) The 'partner matrix' either B or A, needed to 
        calculate the update to the first matrix.
    row_col : (int) Number to indicate whether it is a row in A or B being 
        updated. 0 indicates A and 1 indicates B. Important as the calculation
        differs slightly between the two.
        
    Returns:
    -------
    None"""

    X_low = X[0]
    X_high = X[1]
    if row_col == 0:
        X_low = X_low[I]
        X_high = X_high[I]
    elif row_col == 1:
        X_low = X_low[:,I]
        X_high = X_high[:,I]
    
    Row_Xi = check_shape(X_low)
    Col_Xi = check_shape(X_high)
    A_i = A[I]
    A_i = check_shape(A_i)
    Ai_T, B_T = pad_matrices(A_i.T, B.T)
    AiB_T = np.matmul(Ai_T, B_T)
    X_low = check_shape(X_low)
    X_high = check_shape(X_high)
    low_check = X_low <= AiB_T
    high_check = X_high >= AiB_T
    contains_false = ((low_check == False).any() or
                      (high_check == False).any())

    if contains_false:
        raise ValueError("Boundary conditions violated during 'check_xi_bounds'!")


# Data needs to be organised as having samples in rows, variables in columns for later calculations
# Current data is arbritary, for testing only
data = np.array([2, 6, 2, 3, 3, 2, 3, 2, 7, 3])
data2 = np.array([[2, 3, 3, 2.6, 2.7, 2, 3, 5, 3, 3], [2, 3, 6, 4, 2, 2, 3, 2, 3, 3]])
data_temp = ((data[0] + data2[1])*80).T
data_temp = check_shape(data_temp)
data = check_shape(data.T)
data2 = data2.T
data = np.hstack((data,data2,data_temp))
#Number of dimensions to reduce to
n = 1
#Boundary arrays
low_col = np.ones((10,1))*-2
low_bound = np.hstack((low_col, low_col, low_col, low_col))
high_col = low_col*-10
high_bound = np.hstack((high_col, high_col, high_col, high_col+900))
bounds_vec = np.array((low_bound, high_bound))

#Set up problem by standardising data and bounds (in accordance with Chemometrics 2007; 21: 547–556
#DOI: 10.1002/cem) and performing initial PCA on data 
s, CSs, CLM, PAs, bounds_vec, rev_stdise = problem_setup(data, bounds_vec, (4-n))

#Passing to function that carries out BPCA (as in Chemometrics 2007; 21: 547–556
#DOI: 10.1002/cem)
A, B = solve_for_A_B(CSs, CLM, bounds_vec, data, 10**(-12))
CLM = check_shape(CLM)
A_pad, CLM_pad = pad_matrices(A, CLM)
A, B_T = pad_matrices(A, B.T)
partial = np.matmul(A, B_T)
partial_artificial = np.matmul(A_pad, CLM_pad)
reconst = (partial*rev_stdise[1])+rev_stdise[0]
reconst_artificial = (partial_artificial*rev_stdise[1])+rev_stdise[0]
print("reconstructed from {0}/4 dimensions ".format(n))
print(reconst)
print("original 4D")
print(data)
print("Reconstructed using original PCA B.T ")
print(reconst_artificial)