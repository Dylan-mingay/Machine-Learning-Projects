from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def cross_validate_model(clf, X, Y, k=5):
    k_fold = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)

    smoothing_factors = [1, 2, 3, 4, 5, 6]
    fit_priors = [True, False]
    auc_records = {}

    for train_indices, test_indices in k_fold.split(X, Y):
        X_train_k, X_test_k = X[train_indices], X[test_indices]
        Y_train_k, Y_test_k = Y[train_indices], Y[test_indices]

        for alpha in smoothing_factors:
            if alpha not in auc_records:
                auc_records[alpha] = {}
            for fit_prior in fit_priors:
                clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)

                clf.fit(X_train_k, Y_train_k)
                prediction_probabilities = clf.predict_proba(X_test_k)
                pos_probs = prediction_probabilities[:, 1]
                auc = roc_auc_score(Y_test_k, pos_probs)
                auc_records[alpha][fit_prior] = auc + auc_records[alpha].get(fit_prior, 0.0)


    for smoothing, smoothing_record in auc_records.items():
        for fit_prior, total_auc in smoothing_record.items():
            average_auc = total_auc / k
            print(f'Alpha: {smoothing}, Fit Prior: {fit_prior}, Average AUC: {average_auc:.4f}')
                


def train_best_model(X, Y):
    best_clf = MultinomialNB(alpha=2.0, fit_prior=False)
    best_clf.fit(X, Y)
    pos_probabilities = best_clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(Y, pos_probabilities)
    print(f'Best Model AUC on full dataset: {auc:.4f}')
    return best_clf