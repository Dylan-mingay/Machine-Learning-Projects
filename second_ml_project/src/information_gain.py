import numpy as np
import matplotlib.pyplot as plt

criterion_function = {'gini': gini_impurity, 'entropy': entropy}
# information gain calculation works out the entropy of a set of labels
def information_gain(pos_fractions, labels):
    """
    Calculate the Information Gain for a list or array of class labels.

    Parameters:
    pos_fractions (list or np.array): Array of positive fractions.

    Returns:
    float: Information Gain value.
    """
    # Calculate entropy for the given positive fractions
    entropy = - (pos_fractions * np.log2(pos_fractions) + (1 - pos_fractions) * np.log2(1 - pos_fractions))


    plt.plot(pos_fractions, entropy)
    plt.ylim(0, 1)
    plt.ylabel("Positive fraction")
    plt.xlabel("Entropy")
    plt.show()

    if len(labels) == 0:
        return 0
    
    # Count the occurrences of each label
    counts = np.unique(labels, return_counts=True)[1]
    fraction = counts / len(labels)

    # Calculate impurity based on the label distribution using entropy formula
    impurity = -np.sum(fraction * np.log2(fraction + 1e-9))

    return impurity

def weighted_impurity(groups, criterion='gini'):
    '''
    Docstring for weighted_impurity 
    Weighted impurity of a split given the groups and the criterion after a split.
    
    :param groups: Description
    :param citerion: Description
    :return: weighted impurity
    '''
    total = sum(len(group) for group in groups)
    weighted_sum = 0.0
    for group in groups:
        weighted_sum += len(group) / float(total) * criterion_function[criterion](group)
    return weighted_sum

pos_fractions = np.linspace(0.001, 0., 1000)

print(information_gain(pos_fractions, [1, 1, 0, 1, 0]))
print(information_gain(pos_fractions, []))
print(information_gain(pos_fractions, [1, 1, 1, 1, 1]))
print(information_gain(pos_fractions, [0, 0, 0, 0, 0]))
print(information_gain(pos_fractions, [1, 0, 1, 0, 1, 0]))
print(information_gain(pos_fractions, [1, 1, 1, 0, 0, 0, 1, 0, 1]))

