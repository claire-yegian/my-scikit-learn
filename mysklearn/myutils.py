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