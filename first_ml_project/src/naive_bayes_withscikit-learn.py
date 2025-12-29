from sklearn.naive_bayes import BernoulliNB
import numpy as np

X_train = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0],
    [1, 1, 0]
])

Y_train = np.array(['Y', 'N', 'Y', 'Y'])
X_test = np.array([1, 1, 0])

#alpha is the Laplace smoothing factor and fit_prior indicates whether to learn class prior probabilities or not
clf = BernoulliNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)

predicted_proba = clf.predict_proba([X_test])
print("Predicted probabilities:", predicted_proba)

predicted_label = clf.predict([X_test])
print("Predicted label:", predicted_label)