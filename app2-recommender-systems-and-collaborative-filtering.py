import numpy as np
import pandas as pd

# When ready, import your SVD decomposition function.
# Uncomment the line below when ready.
from svd_lab import svd_decomposition, low_rank_svd

def load_ratings_matrix(file_path):
    """
    Reads a TSV file with columns 'userId', 'movieId', and 'rating',
    and transforms it into a user-item ratings matrix.

    Parameters:
    - file_path: Path to the TSV file.

    Returns:
    - A 2D numpy array representing the user-item ratings matrix.
    """
    # Load the dataset.
    df = pd.read_csv(file_path, sep='\t', header=None,
                     names=['userId', 'movieId', 'rating', 'timestamp'])
    # Pivot the table to create a user (rows) x movie (columns) matrix.
    ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    return ratings_matrix.values

def recommender_svd(ratings, k):
    """
    Generates a low-rank approximation of the ratings matrix using SVD.

    Parameters:
    - ratings: 2D numpy array with users as rows and movies as columns.
    - k: Number of latent factors to retain.

    Returns:
    - ratings_pred: The reconstructed ratings matrix as a low-rank approximation.
    """
    # Compute the SVD of the ratings matrix using your svd_decomposition function.
    U, Sigma, V = svd_decomposition(ratings)
    
    # Truncate the SVD components to keep only the top k latent factors.
    U_k, Sigma_k, V_k = low_rank_svd(U, Sigma, V, k)
    
    # Reconstruct the ratings matrix using the truncated SVD components.
    ratings_pred =  U_k @ Sigma_k @ V_k
    return ratings_pred

def main():
    # The path to the ratings dataset.
    file_path = r'application-datasets\app2-recommender-systems-and-collaborative-filtering\u.data'
    ratings = load_ratings_matrix(file_path)

    m, n = ratings.shape
    print(f'Number of users = {m}')
    print(f'Number of movies = {n}')
    
    # Set the number of latent factors (k) to experiment with.
    k = 25
    
    # Generate the predicted ratings matrix using the SVD-based recommender.
    predicted_ratings = recommender_svd(ratings, k)
    
    print("Original Ratings Matrix:")
    print(ratings)
    print("\nPredicted Ratings Matrix (Low-Rank Approximation):")
    print(np.round(predicted_ratings, 2))

if __name__ == '__main__':
    main()
