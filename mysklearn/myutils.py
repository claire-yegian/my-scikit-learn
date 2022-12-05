import copy
import numpy as np

from mysklearn import myevaluation

def compute_distance(point1, point2):
    """Compute the Euclidean distance between two points of either 2 or 3 dimensions
    Args:
        point1(list of ints or floats): the first point
        point2(list of ints or floats): the second point
    Returns:
        float: the distance between the two points
    """
    try:
        if (len(point1) != 2 and len(point1) != 3) or (len(point2) != 2 and len(point2) != 3):
            raise Exception("Data must be 2 or 3 dimensional and parallel. Try again.")
        if len(point1) == 2:
            return np.sqrt((point2[0] - point1[0])**2 + (point2[1]-point1[1])**2)
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1]-point1[1])**2 + (point2[2]-point1[2])**2)
    except:
        if point1 == point2:
            return 0
        return 1

def get_unique_values(non_unique_values):
    """finds the unique values in a given list of values
    
    Args:
        non_unique_values(list): list of values
    Returns:
        list: list of all unique values
    """
    unique_values = []
    for val in non_unique_values:
        if val not in unique_values:
            unique_values.append(val)
    return unique_values

def get_frequency_of_value(target_val, all_value_instances):
    """finds the frequency of a given value in a given list of values

        Args:
            target_val(): value of which to find frequency in list
            all_value_instances(list): values from which to find frequency of given values

        Returns:
            int: number of occurrences of given value in given list of values
    """
    count = 0
    for instance in all_value_instances:
        if target_val == instance:
            count += 1
    return count

def get_frequency_of_multiple_values(target_values, all_value_instances):
    """finds the frequencies of multiple values in a given list

        Args:
            target_values(list): list of values of which to find frequencies
            all_value_instances(list): list of values from which to find frequencies of given values

        Returns:
            list of ints: list of numbers of occurences of given values in given list
    """
    frequencies = []
    for val in target_values:
        frequencies.append(get_frequency_of_value(val, all_value_instances))
    return frequencies

def compute_equal_frequency_cutoffs(instances, num_bins):
    bin_size = len(instances)//num_bins
    cutoffs = []
    for i in range(num_bins):
        cutoffs.append(instances[i * bin_size])
    cutoffs.append(instances[-1])
    return cutoffs

def discretize_with_cut_offs(y_values, cut_offs:list):
    """discretizes a set of data based on given cut off values.

        Args:
            y_values(list of ints): continuous values to be categorized
            cut_offs(list): list of intervals with which to split data set
        Returns:
            float: discretized values of y_values
    """
    y_copy = y_values.copy()
    num_values = len(y_copy)
    subrange = 1
    for cut_off_index in range(1, len(cut_offs) - 1):
        for y_index in range(num_values):
            if cut_offs[cut_off_index - 1] <= y_copy[y_index] < cut_offs[cut_off_index]:
                y_copy[y_index] = subrange
        subrange += 1
    for y_index in range(num_values):
        if cut_offs[-2] <= y_copy[y_index] <= cut_offs[-1]:
            y_copy[y_index] = subrange
    for index, value in enumerate(y_copy):
        if value == 1:
            y_copy[index] = "least popular"
        elif value == 2:
            y_copy[index] = "less popular"
        elif value == 3:
            y_copy[index] = "popular"
        else:
            y_copy[index] = "very popular"
    return y_copy

def count_occurances(X_data, att_idx, attribute, y_data, classification):
    """Count the occurances of an instance with a specific classification and attribute value
    Args:
        X_data(list of lists of obj): the list of instances we're looking through to find occurances
        att_idx(int): the index of the attribute in a given instance
        attribute(obj): the attribute value we're trying to match
        y_data(list of obj): a list of classifications matching and parallel to X_data
        classification(obj): the classification we're trying to match
    Returns:
        int: the number of occurances
    """
    try:
        attribute = int(attribute)
    except:
        pass
    count = 0
    for i in range(len(X_data)):
        if (X_data[i][att_idx] == attribute) and (y_data[i] == classification):
            count += 1
    return count

def get_attributes_dict(header, instances):
    """Creates a dictionary of all values in each attribute domain for a given header and set of
        instances. Keys of the dictionary are stored as "att1", "att2", etc, not the string attribute
        names in the header.
    Args:
        header(list of str): the string names of the attributes
        instances(list of lists): the instances in the dataset from which to extract attribute domains
    Returns:
        dict of lists: keys are the attributes, values are all values in that attribute's domain
    """
    attributes = {} #build a dictionary to track all of the occurances of each attribute
    for i in range(len(header)):
        occurances = []
        for value in instances:
            if value[i] not in occurances:
                occurances.append(value[i])
        attributes["att"+str(i+1)] = occurances
    return attributes

def split(folds, idx):
    """Split a list of folds into training and testing sets given the index of
    the testing fold
    Args:
        folds(list of lists of ints): a list of the folds
        idx(int): the index of the particular fold we will use for the test set
    Returns:
        list of ints: the training indices
        list of ints: the testing indices
    """
    test = folds[idx]
    train = []
    for i in range(len(folds)):
        if i != idx:
            for item in folds[i]:
                train.append(item)
    return train, test

def strat_cross_val_predict(k, X, y, n_splits, random_state=9, shuffle=False):
    """Build k kfold train/test splits using myevaluation's kfold_split and
    stratified_kfold_split methods
    Args:
        k(int): the number of splits we want
        X(list of lists of obj): the 2D X data we're splitting
        y(list of obj): the parallel y data we're splitting
        n_splits(int): the number of folds
        random_state(int): optional, the seed for our random number generator
        shuffle(bool): optional, False if we're not shuffling before splitting, True if we are
    Returns:
        list of lists: a list of 10 splits, each with 4 elements, X_train, X_test,
            y_train, and y_test
    """
    splits = []
    for i in range(k):
        this_split = []
        kfolds = myevaluation.stratified_kfold_split(X, y, n_splits)

        # convert the folds from indicies to values
        for fold in kfolds:
            this_split.append([[X[i] for i in fold[0]], [X[j] for j in fold[1]], \
                [y[i] for i in fold[0]], [y[j] for j in fold[1]]])
        splits.append(this_split)
    return splits

def confusion_matrix_values(predicted, actual, positive_class):
    """Finds and returns true positive, true negative, false positive, and false negative
    for a given pair of predicted and actual values
    Args:
        predicted(list of obj): the predicted values
        actual(list of obj): parallel to predicted, the actual/expected values
        positive_class(obj): the class we will be calling positive for distinguishing P and N
    Returns:
        int: number of true positives
        int: number of true negatives
        int: number of false positives
        int: number of false negatives
    """
    true_p, true_n, false_p, false_n = 0, 0, 0, 0
    for i in range(len(predicted)):
        if predicted[i] == positive_class:
            if actual[i] == positive_class:
                true_p += 1
            else:
                false_p += 1
        else:
            if actual[i] != positive_class:
                true_n += 1
            else:
                false_n += 1
    return true_p, true_n, false_p, false_n
