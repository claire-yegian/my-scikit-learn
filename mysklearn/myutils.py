import copy

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
    subrange = ["least popular", "less popular", "popular", "very popular"]
    i = 0
    for cut_off_index in range(1, len(cut_offs) - 1):
        for y_index in range(num_values):
            if cut_offs[cut_off_index - 1] <= y_copy[y_index] < cut_offs[cut_off_index]:
                y_copy[y_index] = subrange[i]
        i += 1
    for y_index in range(num_values):
        if cut_offs[-2] <= y_copy[y_index] <= cut_offs[-1]:
            y_copy[y_index] = subrange[-1]
    return y_copy
