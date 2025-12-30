# Import necessary modules and custom functions
import movie_recommendation_data_analysis as mrda
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from move_recommendation_performance_analysis import analyze_performance, receiver_operating_characteristic
from movie_recommendation_cross_validation import cross_validate_model, train_best_model

# Load the movie ratings data from the default path
df, n_users, n_movies = mrda.load_movie_ratings_data()
# Convert the DataFrame into a user-item matrix and create a movie ID mapping
data, movie_id_map = mrda.load_user_rating_data(df, n_users, n_movies)
# Construct the feature matrix X and label vector Y for the target movie (ID 2858)
X, Y = mrda.construct_dataset(data, movie_id_map, target_movie_id=2858) 
# Convert ratings to binary labels (positive/negative) and count them
n_pos, n_neg, Y = mrda.pos_neg_review_count(Y, threshold=3)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Print the sizes of the training and testing sets
print(len(Y_train), len(Y_test))

# Initialize a Multinomial Naive Bayes classifier with smoothing alpha=1.0 and fit_prior=True
clf = MultinomialNB(alpha=1.0, fit_prior=True)
# Train the model on the training data
clf.fit(X_train, Y_train)

# The trained model is now stored in the variable 'clf'

# Predict probabilities for the test set
prediction_probabilities = clf.predict_proba(X_test)
# Print the first 10 probability predictions
print(prediction_probabilities[0:10])

# Predict class labels for the test set
prediction = clf.predict(X_test)
# Print the first 10 class predictions
print(prediction[:10])

# Calculate and print the accuracy of the model on the test set
accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is: {accuracy*100:.1f}%')

# Analyze performance using confusion matrix, precision, recall, F1-score
analyze_performance(Y_test, prediction)
# Plot and calculate ROC curve and AUC
receiver_operating_characteristic(Y_test, prediction_probabilities)

# Perform cross-validation to tune hyperparameters
cross_validate_model(clf, X, Y, k=5)
# Train the best model based on cross-validation results
best_clf = train_best_model(X_train, Y_train)
