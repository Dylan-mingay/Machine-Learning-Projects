import numpy as np
import pandas as pd
from torch import threshold

# load the movie ratings dataset

data_path = 'C:/Users/dylan/OneDrive/Documents/python/ML_Projects/first_ml_project/data/ratings.dat'
def load_movie_ratings_data(data_path='C:/Users/dylan/OneDrive/Documents/python/ML_Projects/first_ml_project/data/ratings.dat'):
    """
    Docstring for load_movie_ratings_data
    Load movie ratings data from a specified path.
    @param data_path: path to the ratings data file
    @return: DataFrame containing the movie ratings data
    """
    df = pd.read_csv(data_path, sep='::', engine='python', header=None)
    df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

    n_users = df['UserID'].nunique()
    n_movies = df['MovieID'].nunique()
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
    # initialize the user-item matrix with zeros
    user_item_matrix = np.zeros([n_users, n_movies], dtype=np.intc)
    movie_id_map = {}
    for user_id, movie_id, rating in zip(df['UserID'], df['MovieID'], df['Rating']):
        user_id = int(user_id) - 1  # assuming UserID starts from 1
        if movie_id not in movie_id_map:
            movie_id_map[movie_id] = len(movie_id_map)
        user_item_matrix[user_id, movie_id_map[movie_id]] = rating

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
    X_raw = np.delete(data, movie_id_map[target_movie_id], axis=1)
    Y_raw = data[:, movie_id_map[target_movie_id]]
    X = X_raw[Y_raw > 0]
    Y = Y_raw[Y_raw > 0]
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
    Y[Y <= threshold] = 0  # negative review
    Y[Y > threshold] = 1  # positive review

    n_pos = (Y == 1).sum()
    n_neg = (Y == 0).sum()
    print(f'{n_pos} positive samples and {n_neg} negative samples.')
    return n_pos, n_neg, Y

# The purpose of this is to analyse the label distribution in the classification problem.
# To see if there is any class imbalance that needs to be addressed during model training.