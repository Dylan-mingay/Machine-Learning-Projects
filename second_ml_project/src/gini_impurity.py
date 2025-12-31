import matplotlib.pyplot as plt
import numpy as np

# gini impurity calculation works out the impurity of a set of labels
def gini_impurity(pos_fractions, labels):
    """
    Calculate the Gini impurity for a list or array of class labels.

    Parameters:
    pos_fractions (list or np.array): Array of positive fractions.

    Returns:
    float: Gini impurity value.
    """
    gini = 1 - pos_fractions**2 - (1 - pos_fractions)**2

    plt.plot(pos_fractions, gini)
    plt.ylim(0, 1)
    plt.ylabel("Positive fraction")
    plt.xlabel("Gini Impurity")
    plt.show()

    if len(labels) == 0:
        return 0
    
    #count the occurances of each label
    counts = np.unique(labels, return_counts=True)[1]
    fraction = counts / len(labels)
    impurity = 1 - np.sum(fraction**2)

    return impurity

pos_fractions = np.linspace(0.00, 1.00, 1000)

print(gini_impurity(pos_fractions, [1, 1, 0, 1, 0]))
print(gini_impurity(pos_fractions, []))
print(gini_impurity(pos_fractions, [1, 1, 1, 1, 1]))
print(gini_impurity(pos_fractions, [0, 0, 0, 0, 0]))
print(gini_impurity(pos_fractions, [1, 0, 1, 0, 1, 0]))
print(gini_impurity(pos_fractions, [1, 1, 1, 0, 0, 0, 1, 0, 1]))