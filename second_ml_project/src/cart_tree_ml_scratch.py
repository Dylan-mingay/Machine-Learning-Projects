import numpy as np
from information_gain import weighted_impurity

def split_node(x, y, index, value):
    """
    Splits the dataset into two groups based on the given feature index and value.

    Parameters:
    x (np.array): Feature dataset.
    y (np.array): Labels corresponding to the dataset.
    index (int): Index of the feature to split on.
    value (float): Value of the feature to split on.

    Returns:
    tuple: Two tuples containing the left and right splits of (x, y).
    """
    x_index = x[:, index]
    # if this feature is numerical
    if x[0, index].dtype.kind in  ['i', 'f']:
        mask = x_index >= value
    #if this feature is categorical
    else:
        mask = x_index == value
    #split the dataset
    left = [x[~mask, :], y[~mask]]
    right = [x[mask, :], y[mask]]
    return left, right

def get_best_split(x, y, criterion):
    best_index, best_value, best_score, children = None, None, 1, None

    for index in range(len(x)):
        for value in np.unique(x[:, index]):
            groups = split_node(x, y, index, value)
            impurity = weighted_impurity([groups[0][1], groups[1][1]], criterion)

            if impurity < best_score:
                best_index, best_value, best_score, children = index, value, impurity, groups
    
    return {'index': best_index, 'value': best_value, 'children': children}

def get_leaf(labels):
    '''
    Get the most common label in the list of labels.

    :param labels: List or array of labels.
    :return: Most common label.
    '''
    most_common = np.bincount(labels).argmax()
    return most_common

def split(node, max_depth, min_size, depth, criterion):
    left, right = node['children']
    del(node['children'])

    # check for a no split
    if left[1].size == 0:
        node['right'] = get_leaf(right[1])
        return
    
    if right[1].size == 0:
        node['left'] = get_leaf(left[1])
        return

    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = get_leaf(left[1]), get_leaf(right[1])
        return

    # check if left child has enough samples
    if left[1].size <= min_size:
        node['left'] = get_leaf(left[1])
    else:
        # it has enough samples, we further split it
        result = get_best_split(left[0], left[1], criterion)
        result_left, result_right = result['children']

        if result_left[1].size == 0:
            node['left'] = get_leaf(result_right[1])
            return
        elif result_right[1].size == 0:
            node['left'] = get_leaf(result_left[1])
            return
        else:
            node['left'] = result
            split(node['left'], max_depth, min_size, depth + 1, criterion)

    # process right child in same way
    if right[1].size <= min_size:
        node['right'] = get_leaf(right[1])
    else:
        # it has enough samples, we further split it
        result = get_best_split(left[0], left[1], criterion)
        result_left, result_right = result['children']

        if result_left[1].size == 0:
            node['left'] = get_leaf(result_right[1])
            return
        elif result_right[1].size == 0:
            node['left'] = get_leaf(result_left[1])
            return
        else:
            node['left'] = result
            split(node['left'], max_depth, min_size, depth + 1, criterion)