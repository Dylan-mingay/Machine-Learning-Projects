import numpy as np
import pandas as pd
from torch import threshold  # Note: This import appears unused; likely a remnant from previous code

# Default path to the movie ratings dataset file
data_path = 'C:/Users/dylan/OneDrive/Documents/python/ML_Projects/first_ml_project/data/ratings.dat'

def load_movie_ratings_data(data_path='C:/Users/dylan/OneDrive/Documents/python/ML_Projects/first_ml_project/data/ratings.dat'):
    """
    Docstring for load_movie_ratings_data
    Load movie ratings data from a specified path.
    @param data_path: path to the ratings data file
    @return: DataFrame containing the movie ratings data
    """
    # Read the CSV file using '::' as separator (common in MovieLens dataset)
    df = pd.read_csv(data_path, sep='::', engine='python', header=None)
    # Assign column names to the DataFrame
    df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

    # Calculate the number of unique users and movies
    n_users = df['UserID'].nunique()
    n_movies = df['MovieID'].nunique()
    # Print the DataFrame and summary statistics
    print(df)
    print(f"Number of users: {n_users}")
    print(f"Number of movies: {n_movies}")

    return df, n_users, n_movies



def load_user_rating_data(df, n_users, n_movies):
    """
    Docstring for load_user_rating_data
    Load user rating data into a user-item matrix.
    @param df: DataFrame containing user ratings
    @param n_users: number of unique users
    @param n_movies: number of unique movies
    @return: user-item matrix and movie ID mapping
    """
    # Initialize a matrix to store user-item ratings, with zeros as default (no rating)
    user_item_matrix = np.zeros([n_users, n_movies], dtype=np.intc)
    # Dictionary to map original movie IDs to column indices in the matrix
    movie_id_map = {}
    # Iterate through each row in the DataFrame to populate the matrix
    for user_id, movie_id, rating in zip(df['UserID'], df['MovieID'], df['Rating']):
        # Convert user_id to 0-based indexing (assuming original IDs start from 1)
        user_id = int(user_id) - 1
        # If movie_id not seen before, assign it a new column index
        if movie_id not in movie_id_map:
            movie_id_map[movie_id] = len(movie_id_map)
        # Set the rating in the matrix at the appropriate user and movie position
        user_item_matrix[user_id, movie_id_map[movie_id]] = rating

    # Analyze the distribution of ratings in the matrix
    values, counts = np.unique(user_item_matrix, return_counts=True)
    for value, count in zip(values, counts):
        print(f'Number of rating: {value}, Count: {count}')

    return user_item_matrix, movie_id_map

def construct_dataset(data, movie_id_map, target_movie_id=2858):
    """
    Docstring for construct_dataset
    Construct feature matrix and label vector for a specific movie.
    @param data: user-item matrix
    @param movie_id_map: mapping of movie IDs to column indices
    @param target_movie_id: the movie ID for which to construct the dataset
    @return: feature matrix X and label vector Y
    """
    # Remove the column corresponding to the target movie from the data to create features
    X_raw = np.delete(data, movie_id_map[target_movie_id], axis=1)
    # Extract the ratings for the target movie as the labels
    Y_raw = data[:, movie_id_map[target_movie_id]]
    # Filter to include only users who have rated the target movie (Y_raw > 0)
    X = X_raw[Y_raw > 0]
    Y = Y_raw[Y_raw > 0]
    # Print the shapes of the resulting matrices
    print('Shape of X:', X.shape)
    print('Shape of Y:', Y.shape)
    
    return X, Y


def pos_neg_review_count(Y, threshold=3):
    """
    Docstring for pos_neg_review_count
    Count the number of positive and negative reviews based on a threshold.
    @param Y: label vector
    @param threshold: rating threshold to distinguish positive and negative reviews
    @return: number of positive and negative reviews
    """
    # Convert ratings to binary labels: 0 for negative (rating <= threshold), 1 for positive (rating > threshold)
    Y[Y <= threshold] = 0  # negative review
    Y[Y > threshold] = 1  # positive review

    # Count the number of positive and negative samples
    n_pos = (Y == 1).sum()
    n_neg = (Y == 0).sum()
    # Print the counts
    print(f'{n_pos} positive samples and {n_neg} negative samples.')
    return n_pos, n_neg, Y

# The purpose of this is to analyse the label distribution in the classification problem.
# To see if there is any class imbalance that needs to be addressed during model training.