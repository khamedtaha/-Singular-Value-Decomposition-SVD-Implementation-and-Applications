'''
SVD Lab Homework
Student: Mohammed taha KHAMED
Date: DD/MM/YYYY
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
    
    eigenvalues, eigenvectors = np.linalg.eigh(M)    #  Compute eigenvalues and eigenvectors using np.linalg.eigh

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
        singular_values (np.ndarray): Array of singular values.
        right_singular_vectors (np.ndarray): n x n matrix whose columns are the right singular vectors.

    Returns:
        U (np.ndarray): m x m matrix whose columns are the left singular vectors.
    """
    m, n = A.shape
    # TODO: Define a tolerance for numerical zero.
    
    # TODO: Determine r, the number of nonzero singular values (at most min(m, n)).
    
    # TODO: Compute U_reduced for each nonzero singular value: u_i = (1/sigma_i) * A * v_i.
    
    # TODO: If r < m, complete U to a full m x m orthogonal matrix.


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
    # TODO: Compute M = A * A^T.
    
    # TODO: Convert M to a Sympy Matrix and compute its nullspace using sympy.Matrix.nullspace.
    
    # TODO: Convert the nullspace basis vectors to a NumPy array.
    
    # TODO: Use only the first (m - r) additional vectors.
    
    # TODO: Combine U_reduced with the additional vectors.
    
    # TODO: (Optional) The resulting U is expected to be orthonormal;
    # if not, consider orthonormalizing it.


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
    # TODO: Compute A^T * A.
    
    # TODO: Compute eigen-decomposition of A^T * A.
    
    # TODO: Compute singular values from eigenvalues.
    
    # TODO: Set V to the eigenvectors (right singular vectors).
    
    # TODO: Compute the left singular vectors U.
    
    m, n = A.shape
    # TODO: Construct the Sigma matrix as an m x n diagonal matrix with the singular values on the diagonal.


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
    # TODO: Check that k is not larger than min(m, n) and raise an error if it is.
    
    # TODO: Extract the first k columns of U to form U_k.
    
    # TODO: Extract the top-left k x k submatrix of Sigma to form Sigma_k.
    
    # TODO: Extract the first k columns of V to form V_k.


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

    # TODO: Choose an example matrix to test.
    A = examples[0]
    
    print('Original matrix A:')
    print(A)
    
    # TODO: Perform SVD decomposition.
    U, Sigma, V = ...
    
    print('\nComputed U:')
    print(U)
    print('\nComputed Sigma:')
    print(Sigma)
    print('\nComputed V:')
    print(V)
    
    # TODO: Reconstruct A from U, Sigma, and V^T.
    A_reconstructed = ...
    print('\nReconstructed A:')
    print(A_reconstructed)

    # TODO: Compute the element-wise difference matrix between A and A_reconstructed.
    diff_matrix = ...
    
    # TODO: Compute the reconstruction error (error) using the Frobenius norm.
    ...
    print('\nReconstruction error (Frobenius norm):', error)
    # TODO: Check if error is close to zero (True/False).
    print('Close to zero?', ...)

    # Extra credit: Compute the reconstruction error using Frobenius norm's trace formula.
    # TODO: Compute trace_error.
    ...
    print('\nReconstruction error (Trace formula):', trace_error)
    # TODO: Check if trace_error is close to zero (True/False).
    print('Close to zero?', ...)

    # Extra credit: Compute the reconstruction error using Frobenius norm's singular values formula.
    # Hint: use your svd_decomposition function with the difference matrix.
    # TODO: Compute sigma_error.
    ...
    print('\nReconstruction error (Singular values formula):', trace_sigma)
    # TODO: Check if sigma_error is close to zero (True/False).
    print('Close to zero?', ...)

    # To complete the extra points, do your own research and link your sources for any formulas you use
    # If you do not wish to solve the additional problems, comment their associated code lines


if __name__ == '__main__':
    main()
