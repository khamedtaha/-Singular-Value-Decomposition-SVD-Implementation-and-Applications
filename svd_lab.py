'''
SVD Lab Homework
Student: Mohammed taha KHAMED
Date: 19/03/2025
Course: Mathematics For Machine Learning (MML) 2
'''

import numpy as np
import numpy.typing as npt
from sympy import Matrix



def compute_at_a(A: npt.NDArray) -> npt.NDArray:
    """
    Compute A^T * A.

    Parameters:
        A (np.ndarray): An m x n matrix.

    Returns:
        np.ndarray: The n x n matrix A^T A.
    """
    return A.T @ A                      # A transpose *  A



def compute_a_at(A: npt.NDArray) -> npt.NDArray:
    """
    Compute A * A^T.

    Parameters:
        A (np.ndarray): An m x n matrix.

    Returns:
        np.ndarray: The m x m matrix A A^T.
    """
    return A @ A.T                       # A * A transpose




def eigen_decomposition(M: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Compute the eigenvalues and eigenvectors of a square matrix M.

    Parameters:
        M (np.ndarray): A square matrix.

    Returns:
        eigenvalues (np.ndarray): The eigenvalues of M.
        eigenvectors (np.ndarray): The corresponding eigenvectors of M.
    """
    
    eigenvalues, eigenvectors = np.linalg.eigh(M)    # Compute eigenvalues and eigenvectors using np.linalg.eigh

    indices = np.argsort(eigenvalues)[::-1]          # Get indices for sorting in descending order

    eigenvalues = eigenvalues[indices]               # Apply sorting to eigenvalues

    eigenvectors = eigenvectors[:, indices]          # Reorder eigenvectors

    return eigenvalues , eigenvectors                # return (eigenvalues, eigenvectors) Ex: eigen_decomposition(M)[0] = eigenvalues




def compute_singular_values(eigenvalues: npt.NDArray) -> npt.NDArray:
    """
    Compute the singular values from the eigenvalues.

    Parameters:
        eigenvalues (np.ndarray): Eigenvalues of A^T A.

    Returns:
        singular_values (np.ndarray): Singular values.
    """
    _new = np.maximum(eigenvalues , 0 )              # convert negative eigenvalues to zero

    return np.sqrt(_new)                             # Compute the square root of the positive eigenvalues


def compute_left_singular_vectors(
    A: npt.NDArray,
    singular_values: npt.NDArray,
    right_singular_vectors: npt.NDArray
) -> npt.NDArray:
    """
    Compute the left singular vectors U using u_i = (1/sigma_i) * A * v_i.

    Parameters:
        A (np.ndarray): Original m x n matrix.
        singular_values (np.ndarray) : Array of singular values.
        right_singular_vectors (np.ndarray): n x n matrix whose columns are the right singular vectors.

    Returns:
        U (np.ndarray): m x m matrix whose columns are the left singular vectors.
    """
    m, n = A.shape

    # Define a tolerance for numerical zero
    tol = np.finfo(float).eps * max(m, n) * np.max(singular_values)        
    
    # The number of singular values is at most min(m, n)
    r = min(min(m, n), np.sum(singular_values > tol))
    
    
    # Compute U_reduced for each nonzero singular value: u_i = (1/sigma_i) * A * v_i.
    U = np.zeros((m, m))
    
    
    for i in range(r):                                            # Compute the first r columns of U for nonzero singular values
        if singular_values[i] > tol:
            U[:, i] = A @ right_singular_vectors[:, i] / singular_values[i]
    
                                        
    for i in range(r):                                            # Normalize columns to ensure orthonormality
        norm = np.linalg.norm(U[:, i])
        if norm > tol:
            U[:, i] = U[:, i] / norm
    

    # If r < m, complete U to a full m x m orthogonal matrix
    if r < m:
        # Use QR decomposition to find orthonormal basis
        # First, create a random matrix
        Q = np.eye(m)
        # Make it orthogonal to existing columns
        for i in range(r):
            for j in range(m):
                Q[:, j] = Q[:, j] - np.dot(Q[:, j], U[:, i]) * U[:, i]
        
        # Use QR to get orthonormal basis for remaining columns
        remaining_Q, _ = np.linalg.qr(Q)
        
        # Check orthogonality with existing columns and set remaining columns
        for j in range(m - r):
            v = remaining_Q[:, j]
            # Orthogonalize against existing vectors
            for i in range(r):
                v = v - np.dot(v, U[:, i]) * U[:, i]
            
            # Normalize
            norm = np.linalg.norm(v)
            if norm > tol:
                U[:, r + j] = v / norm
    
    return U

def complete_U(U_reduced: npt.NDArray, A: npt.NDArray) -> npt.NDArray:
    """
    Complete U_reduced (of size m x r) to a full orthogonal matrix U (m x m)
    by solving A*A^T u = 0 using the built-in nullspace functionality in Sympy.

    Parameters:
        U_reduced (np.ndarray): m x r matrix of computed left singular vectors.
        A (np.ndarray): The original m x n matrix.

    Returns:
        U (np.ndarray): An m x m orthogonal matrix.
    """
    m, r = U_reduced.shape
    assert r < m, "Initial U already has full column rank."
    
    M = compute_a_at(A)              # Compute A * A^T
    
    # Convert M to a Sympy Matrix and compute its nullspace
    M_sympy = Matrix(M.tolist())
    null_space = M_sympy.nullspace()


    # Convert the nullspace basis vectors to a NumPy array
    null_vectors = np.zeros((m, len(null_space)))
    for i, vec in enumerate(null_space):
        null_vectors[:, i] = np.array(vec).astype(float).flatten()
    
    # Use only the first (m - r) additional vectors
    additional_vectors = null_vectors[:, :m-r]
    
    # Combine U_reduced with the additional vectors
    U = np.zeros((m, m))
    U[:, :r] = U_reduced
    U[:, r:] = additional_vectors
    
    # Optional: Ensure orthonormality using QR decomposition
    Q, R = np.linalg.qr(U)
    U = Q
    
    return U


def svd_decomposition(A: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Perform the Singular Value Decomposition (SVD) on matrix A.

    Parameters:
        A (np.ndarray): An m x n matrix.

    Returns:
        U (np.ndarray): m x m matrix of left singular vectors.
        Sigma (np.ndarray): m x n diagonal matrix with singular values.
        V (np.ndarray): n x n matrix of right singular vectors.
    """
    # Compute A^T * A.
    A_transpose_A = compute_at_a(A)

    # Compute eigen-decomposition of A^T * A
    eigenvalues, eigenvectors = eigen_decomposition(A_transpose_A)
    
    # Compute singular values from eigenvalues
    singular_values = compute_singular_values(eigenvalues)
    
    # Set V to the eigenvectors (right singular vectors)
    V = eigenvectors
    
    # Compute the left singular vectors U
    U = compute_left_singular_vectors(A, singular_values, V )
    
    m, n = A.shape

    # Construct the Sigma matrix as an m x n diagonal matrix with the singular values on the diagonal
    m, n = A.shape
    Sigma = np.zeros((m, n))
    r = min(m, n)
    for i in range(r):
        Sigma[i, i] = singular_values[i]
    
    return U, Sigma, V

def low_rank_svd(
    U: npt.NDArray,
    Sigma: npt.NDArray,
    V: npt.NDArray,
    k: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Obtain a low-rank SVD decomposition from the full SVD decomposition.

    Parameters:
        U (np.ndarray): m x m orthogonal matrix from full SVD.
        Sigma (np.ndarray): m x n diagonal matrix from full SVD.
        V (np.ndarray): n x n orthogonal matrix from full SVD.
        k (int): Desired rank for the low-rank approximation (k <= min(m, n)).

    Returns:
        U_k (np.ndarray): m x k matrix.
        Sigma_k (np.ndarray): k x k diagonal matrix.
        V_k (np.ndarray): n x k matrix.
    """
    m, n = Sigma.shape

    # Check that k is not larger than min(m, n) and raise an error if it is
    _min = min(m, n)
    if k > _min :
        raise ValueError(f"k ({k}) cannot be larger than min(m, n) ({_min})")
    
    
    # Extract the first k columns of U to form U_k
    U_k = U[:, :k]
    
    # Extract the top-left k x k submatrix of Sigma to form Sigma_k
    Sigma_k = Sigma[:k, :k]
    
    # Extract the first k columns of V to form V_k
    V_k = V[:, :k].T
    
    return U_k, Sigma_k, V_k


def main():
    examples = [
        np.array([[3, 2, 2],
                  [2, 3, -2]], dtype=float),
        np.array([[1, 2, 3],
                  [2, 4, 6]], dtype=float),
        np.array([[1, 2, 3],
                  [4, 5, 6],
                  [5, 7, 9]], dtype=float)
    ]

    # Choose an example matrix to test.
    A = examples[0]
    
    print('Original matrix A:')
    print(A)
    
    # Perform SVD decomposition.
    U, Sigma, V = svd_decomposition(A)
    
    print('\nComputed U:')
    print(U)
    print('\nComputed Sigma:')
    print(Sigma)
    print('\nComputed V:')
    print(V)
    
    # Reconstruct A from U, Sigma, and V^T.
    A_reconstructed = U @ Sigma @ V.T
    print('\nReconstructed A:')
    print(A_reconstructed)

    # Compute the element-wise difference matrix between A and A_reconstructed.
    diff_matrix =  A - A_reconstructed
    
    # Compute the reconstruction error (error) using the Frobenius norm.
    error = np.linalg.norm(diff_matrix, 'fro')
    print('\nReconstruction error (Frobenius norm):', error)
    # Check if error is close to zero (True/False).
    print('Close to zero?', np.isclose(error, 0) )

    # Extra credit: Compute the reconstruction error using Frobenius norm's trace formula.
    # Compute trace_error.
    trace_error = np.sqrt(np.trace(diff_matrix.T @ diff_matrix))
    print('\nReconstruction error (Trace formula):', trace_error)
    # TODO: Check if trace_error is close to zero (True/False).
    print('Close to zero?', np.isclose(trace_error, 0))

    # Extra credit: Compute the reconstruction error using Frobenius norm's singular values formula.
    # Hint: use your svd_decomposition function with the difference matrix.
    #  Compute sigma_error.
    _ , singular_values_diff , _ = svd_decomposition(diff_matrix)
    trace_sigma = np.sqrt(np.sum(np.square(np.diag(singular_values_diff))))
    print('\nReconstruction error (Singular values formula):', trace_sigma)
    # Check if sigma_error is close to zero (True/False).
    print('Close to zero?', np.isclose(trace_sigma, 0) )

    # To complete the extra points, do your own research and link your sources for any formulas you use
    # If you do not wish to solve the additional problems, comment their associated code lines


if __name__ == '__main__':
    main()
