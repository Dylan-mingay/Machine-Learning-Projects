import numpy as np

# Naive Bayes Classifier from scratch

#=======================Training & testing data========================#
X_train = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0],
    [1, 1, 0]
])

Y_train = np.array(['Y', 'N', 'Y', 'Y'])
X_test = np.array([1, 1, 0])

#=======================Naive Bayes Classifier========================#
def get_label_indices(labels):
    """
    Docstring for get_label_indices
    Group samples based on their labels and return their indices.
    @param labels: list of labels
    @return: dictionary mapping each label to a list of indices of samples with that label
    """
    from collections import defaultdict
    label_indices = defaultdict(list)

    for index, label in enumerate(labels):
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(index)
    return label_indices

def get_prior(label_indicies):
    """
    Docstring for get_prior
    Calculate the prior probabilities of each label.
    @param label_indices: dictionary mapping each label to a list of indices of samples with that label
    @return: dictionary mapping each label to its prior probability
    """
    total_samples = sum(len(indices) for indices in label_indicies.values())
    prior = {}

    for label, indices in label_indicies.items():
        prior[label] = len(indices) / total_samples
    return prior

def get_likelihood(features, label_indicies, smoothing=0.01):
    """
    Docstring for get_likelihood
    Calculate the likelihood of each feature given each label.
    @param features: 2D array of feature values
    @param label_indices: dictionary mapping each label to a list of indices of samples with that label
    @param smoothing: Laplace smoothing factor
    @return: nested dictionary mapping each label to a dictionary of feature likelihoods
    """
    likelihood = {}

    for label, indices in label_indicies.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing

        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)
    return likelihood

def get_posterior(prior, likelihood, sample):
    """
    Docstring for get_posterior
    Calculate the posterior probabilities for each label given a sample.
    @param prior: dictionary mapping each label to its prior probability
    @param likelihood: nested dictionary mapping each label to a dictionary of feature likelihoods
    @param sample: 1D array of feature values for the sample
    @return: dictionary mapping each label to its posterior probability
    """
    posteriors = []

    for x in sample:
        #posterior is proportional to prior * likelihood
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_val in enumerate(sample):
                posterior[label] *= likelihood_label[index] if bool_val else (1 - likelihood_label[index])
                #normalise so that all sums up to 1
            sum_posterior = sum(posterior.values())
            for label in posterior:
                if posterior[label] == float('inf'):
                    posterior[label] = 1.0
                else:
                    posterior[label] /= sum_posterior
            posteriors.append(posterior.copy())
        return posteriors

def main():
    # Group samples by labels
    label_indices = get_label_indices(Y_train)

    # Calculate prior probabilities
    prior = get_prior(label_indices)

    # Calculate likelihoods
    likelihood = get_likelihood(X_train, label_indices, smoothing=1)

    # Calculate posterior probabilities for the test sample
    posteriors = get_posterior(prior, likelihood, X_test)

    # Output the posterior probabilities
    for i, posterior in enumerate(posteriors):
        print(f"Posterior probabilities for sample {i+1}: {posterior}")

if __name__ == "__main__":
    main()