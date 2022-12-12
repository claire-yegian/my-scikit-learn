import copy
import numpy as np

#from mysklearn import myevaluation
import myevaluation
from operator import itemgetter


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
            raise Exception(
                "Data must be 2 or 3 dimensional and parallel. Try again.")
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


def discretize_with_cut_offs(y_values, cut_offs: list):
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
    attributes = {}  # build a dictionary to track all of the occurances of each attribute
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
            this_split.append([[X[i] for i in fold[0]], [X[j] for j in fold[1]],
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


def select_attribute(instances, header, attributes, attribute_domains):
    """Finds the attribute in a given set of attributes with the lowest entropy
    Args:
        instances(list of lists): the instances of the dataset
        header(list of str): all attributes in the dataset
        attributes(list of str): the attributes from which we can selecct
        attribute_domains(dict of lists): domains of all attributes in the header
    Returns:
        str: the attribute with the lowest entropy
    """
    if len(attributes) == 1:
        return attributes[0]

    all_entropies = {}
    for attribute in attributes:
        att_idx = header.index(attribute)
        weighted_entropy = 0
        for value in attribute_domains[attribute]:
            entropy = 0

            # count all occurances of the attribute with that value
            occurances_of_value = 0
            for instance in instances:
                if instance[att_idx] == value:
                    occurances_of_value += 1

            if occurances_of_value > 0:
                # count all occurances of each classification for the attribute with that value
                for classification in attribute_domains[header[-1]]:
                    count = 0
                    for instance in instances:
                        if instance[att_idx] == value and instance[-1] == classification:
                            count += 1
                    if count > 0:
                        entropy += - (count / occurances_of_value) * \
                            np.log2(count / occurances_of_value)
            # compute weighted entroppy for that attribute as we go through each value in the domain
            weighted_entropy += (occurances_of_value /
                                 len(instances)) * entropy
        # store weighted entropies in a dictonary of all attributes
        all_entropies[attribute] = weighted_entropy

    # find the smallest entropy
    min_entropy = 1
    best_att = attributes[0]
    for att, score in all_entropies.items():
        if score < min_entropy:
            min_entropy = score
            best_att = att
    return best_att  # and return that attribute


def partition_instances(instances, attribute, header, attribute_domains):
    """Groups instances by attribute domain
    Args:
        instances(list of lists): the instances of the dataset
        attribute(str): the attribute we're grouping by domain
        header(list of str): all attributes in the dataset
        attribute_domains(dict of lists): domains of all attributes in the header
    Returns:
        dict of lists of lists: key is a value in the attribute's domain, value is a list of the
            instances that have that value for that attribute
    """
    att_index = header.index(attribute)
    att_domain = attribute_domains["att" + str(att_index + 1)]
    partitions = {}
    for att_value in att_domain:
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)
    return partitions


def same_class_label(instances):
    """Finds if a list of instances are all members of the same class
    Args:
        instances(list of lists): the instances of the dataset
    Returns:
        boolean: True if all instances have the same class label, false if otherwise
    Note: assumes class label is the last element in an instance's list of attribute values
    """
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False
    # if we get here, all class labels are the same
    return True


def make_majority_leaf(instances, out_of):
    """Makes a majority leaf node out of a list of instances
    Args:
        instances(list of lists): the instances from which to make the node
    Returns:
        list of obj: a list representation of a majority leaf node, including the majority class
            vote at index 1 (alphabetical first vote if it's a tie)
    """
    # count number of instances matching each classification
    vote_count = {}
    for instance in instances:
        if instance[-1] not in vote_count:
            vote_count[instance[-1]] = 1
        else:
            vote_count[instance[-1]] += 1

    # find the classification that appears most
    tie = []
    most_votes = 0
    most_voted = None
    for voted, votes in vote_count.items():
        if votes > most_votes:
            most_votes = votes
            most_voted = voted
            tie = [voted]
        elif votes == most_votes:
            tie.append(voted)

    winning_vote = most_voted
    if len(tie) > 1:  # if it's a tie, pick the alphabetical first classification
        sorted_votes = sorted(tie)
        winning_vote = sorted_votes[0]

    return ["Leaf", winning_vote, len(instances), out_of]


def tdidt(current_instances, available_attributes, header, attribute_domains, last_split_len):
    """Recursively builds a decision tree using TDIDT (top-down induction of decision trees)
    Args:
        current_instances(list of lists): the currently available instances of the dataset
        available_attributes(list ofstr): the currently available attributes to split on
        header(list of str): all attributes in the dataset
        attribute_domains(dict of lists): domains of all attributes in the header
        last_split_len(int): the number of instances in the last split (used if we need to backtrack
            on an empty partition)
    Returns:
        lists of lists: a list representation of a decision tree
    """
    # select an attribute to split on
    split_attribute = select_attribute(
        current_instances, header, available_attributes, attribute_domains)
    available_attributes.remove(split_attribute)
    # cannot split on this attribute again in this branch of tree
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(
        current_instances, split_attribute, header, attribute_domains)

    # restructure partitions so that it's in alphabetical order
    alphabetized_atts = sorted(partitions.keys())
    alphabetized_atts = [(att_value, partitions[att_value])
                         for att_value in alphabetized_atts]
    partitions = alphabetized_atts

    # for each partition, repeat unless one of the following occurs (base case)
    for att_value, att_partition in partitions:
        value_subtree = ["Value", att_value]
        # base case 1: all class labels of the partition are the same
        if len(att_partition) > 0 and same_class_label(att_partition):
            subtree = ["Leaf", att_partition[0][-1],
                       len(att_partition), len(current_instances)]
        # base case 2: no more attributes to select (clash)
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            subtree = make_majority_leaf(att_partition, len(current_instances))
        # base case 3: no more instances to partition (empty partition)
        elif len(att_partition) == 0:
            return make_majority_leaf(current_instances, last_split_len)
        # otherwise, recurse
        else:
            subtree = tdidt(att_partition, available_attributes.copy(
            ), header, attribute_domains, len(current_instances))
        value_subtree.append(subtree)
        tree.append(value_subtree)
    return tree


def tdidt_predict(tree, instance, header):
    """Recursively predicts the class of an instance using TDIDT and a preconstructed decision tree
    Args:
        tree(list of lists): the tree we're using to predict on
        instance(list of lists): the instance for we're trying to classify
        header(list of str): all attributes in the dataset
    Returns:
        str: the class prediction
    """
    info_type = tree[0]  # Attribute, Value, or Leaf
    if info_type == "Leaf":
        return tree[1]  # base case, return label
    # if Attribute node, need to find value
    att_index = header.index(tree[1])
    for i in range(2, len(tree)):
        value_list = tree[i]
        if value_list[1] == instance[att_index]:
            # we have a match, recurse on this value's subtree
            return tdidt_predict(value_list[2], instance, header)


def tdidt_write_rules(tree, attribute_names, class_name, header):
    """Recursively builds a list of elements of rules that can later be constructed into strings
    Args:
        tree(list of lists): the tree we're writing rules for
        attribute_names(list of str): names to use for the attributes in the rules (if none given,
            "att1", "att2", etc...)
        class_name(str): name to use for class in the rules
        header(list of str): all attributes in the dataset
    Returns:
        list of lists of str: a list of elements of a rule where all but the last element are on the
            left hand side, and the last element is the classification on the right hand side
    """
    if class_name is None:
        class_name = "class"

    info_type = tree[0]  # Attribute or Leaf
    if info_type == "Leaf":
        # base case, return class label (THEN part of rule)
        return [class_name + " = " + tree[1]]

    # otherwise, add the attribute (IF part of rule) to the front of each recursively discovered rule
    rule_list = []
    for i in range(2, len(tree)):
        if attribute_names:  # if we've been given attribute names to use
            new_rule = [attribute_names[header.index(
                tree[1])] + " == " + tree[i][1]]
        else:  # otherwise, default to 'att1', 'att2', etc...
            new_rule = [tree[1] + " == " + tree[i][1]]
        rules_to_add = tdidt_write_rules(
            tree[i][2], attribute_names, class_name, header)  # recursively discovered rules
        for rule in rules_to_add:
            to_add = new_rule.copy()
            if rule[:len(class_name)] == class_name:  # if it's a class label, append it
                to_add.append(rule)
            else:
                to_add += rule  # otherwise, add it so we don't get nested brackets
            rule_list.append(to_add)
    return rule_list


def make_dotfile(tree, lines, id, last_item_id, last_item_val, connections_made):
    """Collects the lines that need to be added to a .dot file in order to make a visual representation of the
        tree using a Graphviz .pdf file.
    Args:
        tree(list of lists): the tree we're trying to visually represent
        lines(list of str): the collection of lines to be written to the .dot file
        id(int): one of many identifiers used to distinguish between nodes (see note below)
        last_item_id(str): the full string identifier of the previous node
        last_item_val(str): the value connecting the previous and current nodes
        connections_made(list of str): a record of the connections between nodes, so we don't make the same one twice
    Returns:
        list of str: a complete collection of all lines that must be added to the .dot file
    Note: the id, current_id, and len(connections_made) are used to help distinguish between nodes as the function
        builds the tree
    """
    info_type = tree[0]  # at a leaf node
    if info_type == "Leaf":
        lines.append(tree[1] + str(id) + str(len(connections_made)
                                             ) + " [label=" + tree[1] + ", shape=circle];\n")
        lines.append(last_item_id + " -- " + tree[1] + str(id) + str(len(connections_made)) + " [label=" +
                     last_item_val + "];\n")
        return lines

    for i in range(2, len(tree)):
        current_id = 0  # one distinguishing identifier
        if info_type == "Attribute":
            lines.append(tree[1] + str(id) + str(current_id) +
                         " [label=" + tree[1] + ", shape=box];\n")
            if last_item_id:  # if it's not a root node
                connection = last_item_id + " -- " + \
                    tree[1] + str(id) + str(current_id)
                if connection not in connections_made:  # make the connection if not already made
                    lines.append(last_item_id + " -- " + tree[1] + str(id) + str(current_id) +
                                 " [label=" + last_item_val + ", shape=box];\n")
                    connections_made.append(connection)
            make_dotfile(tree[i], lines, id + 1, tree[1] +
                         str(id) + str(current_id), None, connections_made)
        else:  # at a value node
            new_lines = make_dotfile(
                tree[i], lines, id + 1, last_item_id, tree[1], connections_made)
            lines.append(new_lines)
        current_id += 1
    return lines


def sort_parallel_lists(list1, list2):
    """sorts two parallel lists while keeping them parallel (sorts on the first list)

        Args:
            list1(list): first parallel list to be sorted (will determine order)
            list2(list): second parallel list to be sorted

        Returns:
            sorted_list1: first parallel list sorted
            sorted_list2: second parallel list sorted
    """
    list_combined = []
    for index, list1_value in enumerate(list1):
        list_combined.append([list1_value, list2[index]])
    sorted_list_combined = sorted(list_combined, key=itemgetter(0))
    sorted_list1 = []
    sorted_list2 = []
    for value in sorted_list_combined:
        sorted_list1.append(value[0])
        sorted_list2.append(value[1])
    return sorted_list1, sorted_list2
