import movie_recommendation_data_analysis as mrda
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df, n_users, n_movies = mrda.load_movie_ratings_data()
data, movie_id_map = mrda.load_user_rating_data(df, n_users, n_movies)
X, Y = mrda.construct_dataset(data, movie_id_map, target_movie_id=2858) 
n_pos, n_neg, Y = mrda.pos_neg_review_count(Y, threshold=3)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(len(Y_train), len(Y_test))

clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)

# our trained model is stored in clf

prediction_probabilities = clf.predict_proba(X_test)
print(prediction_probabilities[0:10])

prediction = clf.predict(X_test)
print(prediction[:10])

accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is: {accuracy*100:.1f}%')