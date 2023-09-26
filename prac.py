import numpy as np
from scipy.linalg import fractional_matrix_power

def nearest_positive_definite(matrix, epsilon=1e-6):
    """
    Compute the nearest positive definite matrix to the input matrix using the Higham algorithm.
    
    Parameters:
    matrix (np.ndarray): Input matrix
    epsilon (float): Tolerance level for eigenvalue modification
    
    Returns:
    np.ndarray: Nearest positive definite matrix
    """
    # Ensure the matrix is symmetric
    symmetric_matrix = (matrix + matrix.T) / 2
    
    # Compute eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eigh(symmetric_matrix)
    
    # Adjust eigenvalues to ensure they are positive
    eig_vals[eig_vals < epsilon] = epsilon
    
    # Reconstruct the matrix using adjusted eigenvalues and eigenvectors
    positive_definite_matrix = np.dot(eig_vecs, np.dot(np.diag(eig_vals), eig_vecs.T))
    
    return positive_definite_matrix

# Example usage
original_matrix = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
nearest_pd_matrix = nearest_positive_definite(original_matrix)
print(nearest_pd_matrix)
