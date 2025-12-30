# Import necessary libraries for cross-validation and evaluation
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def cross_validate_model(clf, X, Y, k=5):
    # Initialize Stratified K-Fold cross-validator to maintain class distribution
    k_fold = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)

    # Define hyperparameter values to test: smoothing factors (alpha) and whether to fit priors
    smoothing_factors = [1, 2, 3, 4, 5, 6]
    fit_priors = [True, False]
    # Dictionary to accumulate AUC scores for each hyperparameter combination
    auc_records = {}

    # Loop through each fold of the cross-validation
    for train_indices, test_indices in k_fold.split(X, Y):
        # Split the data into training and testing sets for this fold
        X_train_k, X_test_k = X[train_indices], X[test_indices]
        Y_train_k, Y_test_k = Y[train_indices], Y[test_indices]

        # Test each combination of alpha and fit_prior
        for alpha in smoothing_factors:
            # Initialize sub-dictionary for this alpha if not present
            if alpha not in auc_records:
                auc_records[alpha] = {}
            for fit_prior in fit_priors:
                # Create a new classifier instance with current hyperparameters
                clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)

                # Train the model on the current fold's training data
                clf.fit(X_train_k, Y_train_k)
                # Predict probabilities on the test fold
                prediction_probabilities = clf.predict_proba(X_test_k)
                # Extract probabilities for the positive class
                pos_probs = prediction_probabilities[:, 1]
                # Calculate AUC for this fold and hyperparameters
                auc = roc_auc_score(Y_test_k, pos_probs)
                # Accumulate the AUC score (will be averaged later)
                auc_records[alpha][fit_prior] = auc + auc_records[alpha].get(fit_prior, 0.0)

    # After all folds, calculate and print the average AUC for each hyperparameter combination
    for smoothing, smoothing_record in auc_records.items():
        for fit_prior, total_auc in smoothing_record.items():
            average_auc = total_auc / k
            print(f'Alpha: {smoothing}, Fit Prior: {fit_prior}, Average AUC: {average_auc:.4f}')
                


def train_best_model(X, Y):
    # Train the model with the best hyperparameters found from cross-validation (alpha=2.0, fit_prior=False)
    best_clf = MultinomialNB(alpha=2.0, fit_prior=False)
    best_clf.fit(X, Y)
    # Predict probabilities on the full training set
    pos_probabilities = best_clf.predict_proba(X)[:, 1]
    # Calculate AUC on the full dataset
    auc = roc_auc_score(Y, pos_probabilities)
    print(f'Best Model AUC on full dataset: {auc:.4f}')
    return best_clf